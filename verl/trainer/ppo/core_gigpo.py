"""Multi-turn GiGPO advantage estimator.

GiGPO (Group-in-Group Policy Optimization, NeurIPS 2025) is critic-free and combines two
group-relative signals:

- **episode advantage** — group each trajectory's outcome by prompt group (``uid``) and
  normalize (GRPO-style), broadcast to its tokens; and
- **step advantage** — cluster *turns* across the group's trajectories by their **anchor**
  (the observation the agent acted on), and normalize each turn's discounted step-return
  within its anchor cluster — a critic-free, state-baselined action advantage.

Unlike the original (verl-agent), which explodes each trajectory into one row per env step,
this operates on AgentFly's native **one row per trajectory**: the per-turn signals arrive as
per-row lists (``step_observations``/``step_rewards``), and the step advantage is scattered
onto each turn's token span via a ``turn_ids`` derived from the action mask. The two layouts
are equivalent because the step advantage is uniform across a turn's tokens either way.
"""

from collections import defaultdict
from typing import Any, Optional, Tuple

import numpy as np
import torch


def compute_turn_ids(response_mask: torch.Tensor) -> torch.Tensor:
    """Map each token to its turn index, derived from the action/response mask.

    A turn is one contiguous run of policy (assistant) tokens. A ``0→1`` transition starts a
    new turn. Returns ``[B, L]`` int64: turn index ``0..n-1`` on that turn's tokens, ``-1`` on
    non-policy tokens.
    """
    am = (response_mask > 0).long()
    prev = torch.cat([torch.zeros_like(am[:, :1]), am[:, :-1]], dim=1)
    starts = (am == 1) & (prev == 0)  # first token of each turn
    turn_idx = torch.cumsum(starts.long(), dim=1) - 1  # running turn count - 1
    return torch.where(am == 1, turn_idx, torch.full_like(turn_idx, -1))


def _discounted_returns(rewards, gamma: float):
    """Backward discounted return-to-go for one trajectory's per-turn rewards."""
    out = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        out[t] = running
    return out


def _to_hashable(x: Any):
    """Make an anchor observation hashable for exact grouping (str/ndarray/list/dict)."""
    if isinstance(x, np.ndarray):
        return tuple(x.flatten().tolist())
    if isinstance(x, (list, tuple)):
        return tuple(_to_hashable(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _to_hashable(v)) for k, v in x.items()))
    return x


def _group_normalize(
    scores: torch.Tensor, index, norm_by_std: bool, epsilon: float
) -> torch.Tensor:
    """Subtract the per-group mean (optionally ÷ group std) — GRPO-style, over ``index``.

    Population std (``unbiased=False``) so singleton groups are std 0; their numerator is
    also 0 (score == group mean), so singletons yield a 0 advantage.
    """
    groups = defaultdict(list)
    for i in range(len(scores)):
        groups[index[i]].append(scores[i])
    mean = {k: torch.stack(v).mean() for k, v in groups.items()}
    std = {
        k: (torch.stack(v).std(unbiased=False) if len(v) > 1 else torch.zeros((), device=scores.device))
        for k, v in groups.items()
    }
    out = scores.clone()
    for i in range(len(scores)):
        out[i] = scores[i] - mean[index[i]]
        if norm_by_std:
            out[i] = out[i] / (std[index[i]] + epsilon)
    return out


def compute_gigpo_multiturn_advantage(
    token_level_rewards: torch.Tensor,   # [B, L] outcome reward on tokens
    response_mask: torch.Tensor,         # [B, L] 1 on policy tokens
    index,                               # [B] uid (prompt/env group id)
    step_observations,                   # [B] object array of per-turn anchor lists
    step_rewards,                        # [B] object array of per-turn reward lists
    gamma: float = 1.0,
    step_advantage_w: float = 1.0,
    norm_by_std: bool = True,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Joint episode + step advantage on one-row-per-trajectory batches.

    Returns ``(advantages, returns)`` both ``[B, L]`` (critic-free: returns == advantages),
    where ``advantages = episode_adv (broadcast) + step_advantage_w * step_adv (per turn span)``.
    """
    B, L = token_level_rewards.shape
    device = token_level_rewards.device
    turn_ids = compute_turn_ids(response_mask)

    # --- episode advantage: group each trajectory's outcome by uid (GRPO-style) ---
    episode_scores = (token_level_rewards * response_mask).sum(dim=-1)  # [B]
    episode_adv = _group_normalize(episode_scores, index, norm_by_std, epsilon)  # [B]
    episode_advantages = episode_adv.unsqueeze(-1) * response_mask  # [B, L]

    # --- per-turn discounted step returns; cluster turns by (uid, anchor) ---
    per_turn_returns = []
    group_returns = defaultdict(list)   # (uid, anchor) -> [returns]
    turn_keys = []                      # per row: [(uid, anchor) per turn]
    for b in range(B):
        rewards = [0.0 if r is None else float(r) for r in (step_rewards[b] or [])]
        returns = _discounted_returns(rewards, gamma)
        per_turn_returns.append(returns)
        anchors = step_observations[b] or []
        keys_b = []
        for t in range(len(returns)):
            anchor = anchors[t] if t < len(anchors) else None
            key = (index[b], _to_hashable(anchor))
            group_returns[key].append(returns[t])
            keys_b.append(key)
        turn_keys.append(keys_b)

    group_mean = {k: float(np.mean(v)) for k, v in group_returns.items()}
    group_std = {k: float(np.std(v)) for k, v in group_returns.items()}

    # Diagnostic: do anchors actually recur? If step-groups are mostly singletons,
    # the step advantage is ~0 and GiGPO degenerates to GRPO regardless of the math.
    sizes = [len(v) for v in group_returns.values()]
    if sizes:
        total_turns = sum(sizes)
        non_singleton = sum(s for s in sizes if s >= 2)
        # verl does not configure the root logger, so logger.info is dropped; print
        # (like verl's own diagnostics) so this reaches the training console.
        print(
            "[GiGPO] step-groups: %d groups / %d turns | %.1f%% turns in non-singleton "
            "groups | avg size %.2f | max %d"
            % (
                len(sizes), total_turns, 100.0 * non_singleton / total_turns,
                total_turns / len(sizes), max(sizes),
            ),
            flush=True,
        )

    # --- step advantage per turn, scattered onto that turn's token span ---
    step_advantages = torch.zeros(B, L, device=device, dtype=episode_advantages.dtype)
    for b in range(B):
        for t, key in enumerate(turn_keys[b]):
            adv = per_turn_returns[b][t] - group_mean[key]
            if norm_by_std:
                adv = adv / (group_std[key] + epsilon)
            step_advantages[b][turn_ids[b] == t] = adv
    step_advantages = step_advantages * response_mask

    # Diagnostic: how much does the step term actually move the advantage vs the
    # episode term? If the step contribution is ~0 (singleton anchor groups) or the
    # nonzero fraction is tiny, GiGPO reduces to GRPO regardless of the group stats.
    resp = response_mask.bool()
    if resp.any():
        ep_mag = episode_advantages[resp].abs().mean().item()
        st_mag = step_advantages[resp].abs().mean().item()
        nz = (step_advantages[resp].abs() > 1e-8).float().mean().item()
        n_turns_am = int((turn_ids >= 0).any(dim=0).sum())  # sanity: alignment check
        print(
            "[GiGPO] |episode_adv|=%.4f  w*|step_adv|=%.4f  step/episode=%.2f  "
            "nonzero-step-tokens=%.1f%%  (turn_ids seen up to %d)"
            % (ep_mag, step_advantage_w * st_mag,
               (step_advantage_w * st_mag) / (ep_mag + 1e-8), 100.0 * nz, n_turns_am),
            flush=True,
        )

    advantages = episode_advantages + step_advantage_w * step_advantages
    return advantages, advantages
