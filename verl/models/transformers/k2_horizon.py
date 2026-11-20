"""Routing replay for K2-Horizon (the Hugging Face custom code) under the FSDP actor.

K2-Horizon picks experts twice in every routed layer: the value experts inside attention (MoVA, top 4 of 64) and
the MLP experts (top 8 of 100). The choice is a hard top-k over router scores, so the rounding differences
between the rollout engine and the trainer flip it for a large share of tokens, and the trainer then scores a
sampled token with a different network than the one that sampled it (measured 2026-11-20 on
K2-Horizon-MoVA-36B-A4B: mean |log-prob gap| to vLLM 0.023 with free routing, 0.011 with vLLM's MLP-expert
choices replayed; a dense model sits at about 0.009).

With replay, the expert ids the rollout engine recorded for every token position (``routed_experts`` in the
batch, ``[batch, length, layers, k]``, vLLM's ``enable_return_routed_experts``) are used in the forward pass in
place of the trainer's own top-k. The router scores, their normalisation and the gradient through them are
unchanged: only which experts are selected is taken from the record.

How it is done without copying the model's forward code: the model's own module calls ``torch.topk`` in exactly
two places, the two routers. Its ``torch`` name is replaced by a proxy whose ``topk`` consults the replay state,
and forward hooks on the two router modules say which layer and which router is running. Everything else the
proxy passes through to ``torch``. Positions with no record (padding, value 255, or a row that is not a valid
set of expert ids) keep the trainer's own choice.

Installed by the FSDP engine (workers/engine/fsdp/transformer_impl.py) on every K2-Horizon model it builds; it
does nothing until a batch carries ``routed_experts``.

The record stays attached until the next micro-batch replaces it, so the layers recomputed during the backward
pass (gradient checkpointing) are routed the same way as in the forward pass.
"""

import sys
from typing import Optional

import torch

PAD = 255  # no recorded choice at this position
# A row's columns: the MLP experts first (config.num_experts_per_tok of them), then the value experts
# (config.mova_num_experts_per_tok), when the engine records them.


class RoutingReplay:
    """Replay state of one model: the recorded choices of the current micro-batch and what is running."""

    def __init__(self):
        self.routes: Optional[torch.Tensor] = None  # [positions, layers, k], in the order of the hidden states
        self.current = None  # (layer index, first column, k) while a router module runs
        self.replayed = 0  # router rows that took the recorded choice, since the last `set`
        self.free = 0  # router rows that kept the trainer's choice while a record was attached

    def set(self, routed_experts, attention_mask=None, lengths=None, packed: bool = False) -> None:
        """Attach the micro-batch's record, ``[rows, length, layers, k]``, padded or as a jagged nested tensor
        (None detaches). ``packed`` (for a padded record): the forward
        pass runs on the unpadded tokens of all rows as one sequence (verl's remove-padding path), so the record
        is packed the same way: by ``attention_mask``, or by the rows' ``lengths`` when they are right-padded."""
        self.replayed = self.free = 0
        if routed_experts is None:
            self.routes = None
            return
        if routed_experts.is_nested:
            # verl's unpadded batch layout: one jagged row per sequence; its values are already all rows'
            # tokens in order, which is what the forward pass runs on.
            self.routes = routed_experts.values()
        elif packed:
            if attention_mask is None:
                positions = torch.arange(routed_experts.shape[1], device=routed_experts.device)
                attention_mask = positions[None] < lengths.to(routed_experts.device)[:, None]
            self.routes = routed_experts[attention_mask.bool()]
        else:
            self.routes = routed_experts.reshape(-1, *routed_experts.shape[2:])

    def enter(self, layer: int, column: int, k: int) -> None:
        self.current = (layer, column, k)

    def leave(self) -> None:
        self.current = None

    def topk(self, x, k, *args, **kwargs):
        out = torch.topk(x, k, *args, **kwargs)
        if self.routes is None or self.current is None:
            return out
        layer, column, expected_k = self.current
        if k != expected_k or column + k > self.routes.shape[-1]:
            return out  # this router's choices were not recorded (e.g. value experts with an 8-column record)
        if self.routes.shape[0] != x.shape[0]:
            raise RuntimeError(
                f"routing replay: {self.routes.shape[0]} recorded positions for {x.shape[0]} token positions in the "
                "forward pass; the record must be aligned with input_ids (and packed like them)."
            )
        forced = self.routes[:, layer, column : column + k].to(device=x.device, dtype=torch.long)
        ordered = forced.sort(-1).values
        valid = (ordered[:, -1] < x.shape[-1]) & (ordered[:, 1:] != ordered[:, :-1]).all(-1)
        indices = torch.where(valid[:, None], forced, out.indices)
        n = int(valid.sum())
        self.replayed += n
        self.free += x.shape[0] - n
        return torch.return_types.topk((torch.gather(x, -1, indices), indices))


class _TorchProxy:
    """``torch`` for the model's module: ``topk`` goes through the replay state of the model whose router is
    running (several models, e.g. actor and reference, share the module), everything else is torch's."""

    def __init__(self):
        self.active: Optional[RoutingReplay] = None

    def topk(self, *args, **kwargs):
        if self.active is None:
            return torch.topk(*args, **kwargs)
        return self.active.topk(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(torch, name)


def _unwrap(model):
    while hasattr(model, "module") and isinstance(getattr(model, "module"), torch.nn.Module):
        model = model.module
    return model


def get_routing_replay(model) -> Optional[RoutingReplay]:
    """The replay state installed on ``model`` (through any FSDP / DDP wrapper), or None."""
    return getattr(_unwrap(model), "_routing_replay", None)


def install_routing_replay(model) -> RoutingReplay:
    """Make ``model`` (a K2-Horizon causal LM, before it is wrapped by FSDP) take recorded expert choices.

    Inactive until a record is attached with :meth:`RoutingReplay.set`.
    """
    core = _unwrap(model)
    existing = getattr(core, "_routing_replay", None)
    if existing is not None:
        return existing
    config = core.config
    replay = RoutingReplay()
    module = sys.modules[type(core).__module__]
    if not isinstance(module.torch, _TorchProxy):
        module.torch = _TorchProxy()
    proxy = module.torch

    def enter(layer_index, column, k):
        proxy.active = replay
        replay.enter(layer_index, column, k)

    def leave():
        proxy.active = None
        replay.leave()

    routed = 0
    for layer_index, layer in enumerate(core.model.layers):
        routers = []
        if hasattr(layer.self_attn, "v_router"):
            routers.append((layer.self_attn, config.num_experts_per_tok, config.mova_num_experts_per_tok))
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts"):
            routers.append((layer.mlp, 0, config.num_experts_per_tok))
        for router, column, k in routers:
            router.register_forward_pre_hook(
                lambda _module, _args, layer_index=layer_index, column=column, k=k: enter(layer_index, column, k)
            )
            router.register_forward_hook(lambda _module, _args, _output: leave())
        routed += bool(routers)
    if routed == 0:
        raise RuntimeError("routing replay: the model has no routed layers")
    core._routing_replay = replay
    return replay
