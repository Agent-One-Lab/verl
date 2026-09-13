# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple
import logging
import numpy as np
import torch
from omegaconf import DictConfig
import os
import sys
from ....verl.utils import torch_functional as verl_F
from ....verl.trainer.config import AlgoConfig
from ....verl.utils import as_torch_index, group_mean_std
from ....verl.utils.import_utils import deprecated
from ....verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARNING"))
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARNING"))
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_h)
    logger.propagate = False

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    CONTEXTRL = "contextrl"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"
    GDPO = "gdpo"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]



class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """

    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# @register_adv_est(AdvantageEstimator.CONTEXTRL)
# def compute_contextrl_advantage_return(
#     token_level_rewards: torch.Tensor,  # [B, L]
#     values: torch.Tensor,               # [B, L]
#     response_mask: torch.Tensor,        # [B, L]
#     index: np.ndarray,                  # [B] group id
#     segment_index: np.ndarray,          # [B] segment order within group
#     gamma: torch.Tensor = 1.0,
#     lam: torch.Tensor = 1.0,
#     prevent_penalize_context: bool = False,
#     penalize_context_max_segments: int = 0,
#     empty_segment_penalty: float = 0.0,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Hierarchical advantage estimation for summarization-based context management.

#     Two-level credit assignment:
#       - Low-level (action) advantage: GAE within each segment. The last step
#         bootstraps to V^high of the next segment, coupling the two levels.
#       - High-level (summarization) advantage: Applied ONLY to summary tokens
#         in non-final segments. Measures the segment-level TD error — how much
#         better/worse this segment (including its summary) performed relative
#         to V^high's prediction at the boundary.

#     Key design decisions:
#       1. The low-level GAE already incorporates cross-segment information via
#          boundary bootstrapping. The high-level advantage provides an ADDITIONAL
#          signal specifically for summary tokens, reflecting how the quality of
#          the summary affected downstream outcomes.
#       2. The final segment produces a final answer, not a summary, so no
#          high-level advantage is added there.
#       3. Single-segment rollouts have no summarization, so only low-level
#          GAE is computed.

#     Args:
#         token_level_rewards: [B, L] per-token rewards (sparse, mostly 0)
#         values: [B, L] critic predictions
#         response_mask: [B, L] binary mask (1 = LLM-generated response token)
#         index: [B] rollout group id (same rollout shares same index)
#         segment_index: [B] segment order within rollout (0, 1, 2, ...)
#         gamma: discount factor
#         lam: GAE lambda
#         prevent_penalize_context: if True, clamp negative high-level advantages
#             for summary tokens using a dynamic schedule based on segment index.
#         penalize_context_max_segments: segment index at which full penalty
#             is restored. Only used when prevent_penalize_context is True.
#             If set to 0, all negative high-level advantages are fully clamped.
#         empty_segment_penalty: negative reward added to the last response token
#             of non-first segments that contain only one contiguous block of
#             response tokens. This penalizes degenerate behavior where the model
#             immediately summarizes again or gives up without performing any
#             tool-calling actions. A segment with multiple contiguous blocks
#             (action -> observation -> action -> ... -> summary) is healthy.
#             A segment with only one block means no tool interaction happened.
#             Recommended value: -1.0 to -2.0 (should be small relative to
#             success reward of 10.0).

#     Returns:
#         advantages: [B, L] hierarchical advantages
#         returns: [B, L] target returns for critic training
#     """
#     with torch.no_grad():
#         B, L = token_level_rewards.shape
#         device = token_level_rewards.device
#         dtype = token_level_rewards.dtype

#         advantages = torch.zeros(B, L, device=device, dtype=dtype)
#         returns = torch.zeros(B, L, device=device, dtype=dtype)

#         # =========================================================================
#         # Step 1: Build group structure
#         # group_map: group_id -> [(segment_order, batch_index), ...] sorted
#         # =========================================================================
#         group_map = defaultdict(list)
#         for b in range(B):
#             group_map[index[b]].append((int(segment_index[b]), b))
#         for gid in group_map:
#             group_map[gid].sort(key=lambda x: x[0])

#         # =========================================================================
#         # Step 2: Precompute per-segment response positions
#         # =========================================================================
#         seg_resp_positions = {}
#         for b in range(B):
#             seg_resp_positions[b] = torch.nonzero(response_mask[b], as_tuple=False).squeeze(-1)
#             if seg_resp_positions[b].dim() == 0:
#                 seg_resp_positions[b] = seg_resp_positions[b].unsqueeze(0)

#         # =========================================================================
#         # Step 2.5: Apply empty segment penalty
#         #
#         # For non-first segments (segment_index > 0), check if the response
#         # tokens form only a single contiguous block. If so, the model did
#         # no tool calling — it either immediately summarized again or directly
#         # gave a final answer without interacting with the environment.
#         #
#         # A healthy segment has multiple blocks:
#         #   [response] [observation gap] [response] [gap] ... [response]
#         #   = multiple contiguous blocks = tool calling happened
#         #
#         # A degenerate segment has one block:
#         #   [response response response ...]
#         #   = no observation gaps = no tool calling
#         #
#         # We add the penalty to the LAST response token of such segments,
#         # so it propagates backward through GAE to all tokens in the segment.
#         # =========================================================================
#         if empty_segment_penalty != 0.0:
#             # Make a writable copy so we don't modify the caller's tensor
#             token_level_rewards = token_level_rewards.clone()

#             for b in range(B):
#                 # Skip first segments (segment_index == 0) — they always start fresh
#                 if int(segment_index[b]) == 0:
#                     continue

#                 positions = seg_resp_positions[b]
#                 if len(positions) == 0:
#                     continue

#                 n_blocks = _count_contiguous_blocks(positions)
#                 if n_blocks <= 1:
#                     # Degenerate segment: only one response block, no tool calling
#                     last_pos = positions[-1].item()
#                     token_level_rewards[b, last_pos] += empty_segment_penalty

#         # =========================================================================
#         # Step 3: Extract V^high at segment boundaries and determine which
#         #         segments are final (last in their rollout group)
#         # =========================================================================
#         def _get_boundary_value(b: int) -> torch.Tensor:
#             positions = seg_resp_positions[b]
#             if len(positions) == 0:
#                 return torch.tensor(0.0, device=device, dtype=dtype)
#             return values[b, positions[0].item()]

#         v_high_self = {}
#         v_high_next = {}
#         is_final_segment = {}

#         for gid, segs in group_map.items():
#             for k, (seg_idx, b) in enumerate(segs):
#                 v_high_self[b] = _get_boundary_value(b)
#                 if k + 1 < len(segs):
#                     v_high_next[b] = _get_boundary_value(segs[k + 1][1])
#                     is_final_segment[b] = False
#                 else:
#                     v_high_next[b] = torch.tensor(0.0, device=device, dtype=dtype)
#                     is_final_segment[b] = True

#         # =========================================================================
#         # Step 4: Low-level GAE within each segment (BATCHED)
#         #
#         # Process all B segments in parallel by iterating backward over the
#         # sequence dimension L once. This matches the standard GAE's O(L)
#         # structure instead of O(B * n_response_tokens) Python iterations.
#         #
#         # KEY COUPLING: at the last response token of each segment, the GAE
#         # bootstraps to V^high of the next segment. We handle this by setting
#         # a per-element bootstrap value that's injected at the right point.
#         #
#         # The backward loop uses the same mask-based carry-through as standard
#         # GAE: at observation tokens, nextvalues and lastgaelam carry forward
#         # unchanged. At response tokens, they update normally.
#         # =========================================================================
#         c = gamma * lam

#         # Build per-element bootstrap: V^high of next segment (0 for final segments)
#         v_bootstrap_vec = torch.zeros(B, device=device, dtype=dtype)
#         for b in range(B):
#             v_bootstrap_vec[b] = v_high_next[b]

#         # Backward loop over all positions, batched across B
#         nextvalues = v_bootstrap_vec.clone()  # [B]
#         lastgaelam = torch.zeros(B, device=device, dtype=dtype)

#         for t in reversed(range(L)):
#             mask_t = response_mask[:, t]  # [B]
#             r_t = token_level_rewards[:, t]  # [B]
#             v_t = values[:, t]  # [B]

#             delta = r_t + gamma * nextvalues - v_t
#             new_gae = delta + c * lastgaelam

#             # At response tokens: update nextvalues and lastgaelam
#             # At observation tokens: carry forward unchanged
#             nextvalues = v_t * mask_t + nextvalues * (1 - mask_t)
#             lastgaelam = new_gae * mask_t + lastgaelam * (1 - mask_t)

#             advantages[:, t] = lastgaelam
#             returns[:, t] = lastgaelam + v_t

#         # =========================================================================
#         # Step 5: High-level advantage for summary tokens (NON-FINAL segments only)
#         #
#         # The low-level GAE already captures cross-segment signal via boundary
#         # bootstrapping. The high-level advantage provides an ADDITIONAL signal
#         # specifically for summary tokens, measuring how the choice of summary
#         # affected outcomes beyond what V^high predicted.
#         #
#         # For segment k (non-final), the high-level TD error is:
#         #   delta^high_k = r_macro_k + gamma_macro_k * V^high(s_{b_{k+1}}) - V^high(s_{b_k})
#         #
#         # We use GAE across the segment chain for multi-step high-level credit,
#         # but ONLY apply the result to summary tokens in non-final segments.
#         # =========================================================================
#         for gid, segs in group_map.items():
#             n_segs = len(segs)
#             if n_segs <= 1:
#                 # Single-segment rollout: no summarization, skip high-level
#                 continue

#             # --- Compute macro quantities per segment ---
#             macro_rewards = []
#             macro_discounts = []

#             for k, (seg_idx, b) in enumerate(segs):
#                 positions = seg_resp_positions[b]
#                 reward = token_level_rewards[b]

#                 if len(positions) == 0:
#                     macro_rewards.append(torch.tensor(0.0, device=device, dtype=dtype))
#                     macro_discounts.append(torch.tensor(1.0, device=device, dtype=dtype))
#                     continue

#                 # Discounted sum of rewards within this segment
#                 n_pos = len(positions)
#                 gamma_pows = torch.pow(gamma, torch.arange(n_pos, device=device, dtype=dtype))
#                 r_macro = torch.sum(gamma_pows * reward[positions])

#                 macro_rewards.append(r_macro)
#                 macro_discounts.append(torch.pow(gamma, torch.tensor(n_pos, device=device, dtype=dtype)))

#             # --- High-level GAE backward across segments ---
#             high_advs = [None] * n_segs
#             gae_high = torch.tensor(0.0, device=device, dtype=dtype)

#             for k in range(n_segs - 1, -1, -1):
#                 v_k = v_high_self[segs[k][1]]
#                 r_k = macro_rewards[k]
#                 gamma_k = macro_discounts[k]

#                 if k + 1 < n_segs:
#                     v_k_next = v_high_self[segs[k + 1][1]]
#                 else:
#                     v_k_next = torch.tensor(0.0, device=device, dtype=dtype)

#                 delta_high = r_k + gamma_k * v_k_next - v_k
#                 gae_high = delta_high + gamma_k * lam * gae_high
#                 high_advs[k] = gae_high

#             # --- Assign high-level advantage to summary tokens ---
#             # ONLY for non-final segments (which produce summaries).
#             #
#             # If prevent_penalize_context is True, apply dynamic clamping:
#             #   - Segment 0 (first summary): full protection, clamp to 0
#             #   - Segment k: allow k/penalize_context_max_segments of the
#             #     negative signal through
#             #   - Segment >= penalize_context_max_segments: full penalty (no clamp)
#             #
#             # This way early summarizations are protected (the model is still
#             # learning to summarize), while later ones get penalized if they
#             # indicate the agent is being inefficient.
#             for k, (seg_idx, b) in enumerate(segs):
#                 if is_final_segment[b]:
#                     continue

#                 positions = seg_resp_positions[b]
#                 if len(positions) == 0:
#                     continue

#                 summary_positions = _find_last_contiguous_block(positions)
#                 adv_high = high_advs[k]

#                 if prevent_penalize_context and adv_high < 0:
#                     # k is the segment index (0 = first segment/summary)
#                     # Compute penalty ratio: 0.0 at k=0, 1.0 at k>=max_segments
#                     if penalize_context_max_segments > 0:
#                         penalty_ratio = min(k / penalize_context_max_segments, 1.0)
#                     else:
#                         # max_segments=0 means full protection for all segments
#                         penalty_ratio = 0.0

#                     # Scale the negative advantage: 0% penalty -> fully clamped,
#                     # 100% penalty -> original negative value passes through
#                     adv_high = adv_high * penalty_ratio

#                 for pos in summary_positions:
#                     p = pos.item()
#                     advantages[b, p] += adv_high

#         return advantages, returns


@register_adv_est(AdvantageEstimator.CONTEXTRL)
def compute_contextrl_advantage_return(
    token_level_rewards: torch.Tensor,  # [B, L]
    values: torch.Tensor,               # [B, L]
    response_mask: torch.Tensor,        # [B, L]
    index: np.ndarray,                  # [B] group id
    segment_index: np.ndarray,          # [B] segment order within group
    gamma: torch.Tensor = 1.0,
    lam: torch.Tensor = 1.0,
    prevent_penalize_context: bool = False,
    penalize_context_max_segments: int = 0,
    empty_segment_penalty: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype
 
    # =========================================================================
    # Input validation
    # =========================================================================
    _check_tensor("token_level_rewards [INPUT]", token_level_rewards)
    _check_tensor("values [INPUT]", values)
    _check_tensor("response_mask [INPUT]", response_mask)
 
    if isinstance(gamma, torch.Tensor):
        _check_tensor("gamma [INPUT]", gamma)
    if isinstance(lam, torch.Tensor):
        _check_tensor("lam [INPUT]", lam)
 
    logger.debug(f"ContextRL advantage: B={B}, L={L}, gamma={gamma}, lam={lam}, "
                 f"prevent_penalize={prevent_penalize_context}, "
                 f"max_segments={penalize_context_max_segments}, "
                 f"empty_penalty={empty_segment_penalty}")
 
    # Check for degenerate inputs
    resp_count = response_mask.sum().item()
    logger.debug(f"Total response tokens: {resp_count}, "
                 f"reward range: [{token_level_rewards.min().item():.4f}, {token_level_rewards.max().item():.4f}], "
                 f"value range: [{values.min().item():.4f}, {values.max().item():.4f}]")
 
    advantages = torch.zeros(B, L, device=device, dtype=dtype)
    returns = torch.zeros(B, L, device=device, dtype=dtype)
 
    # =========================================================================
    # Step 1: Build group structure
    # =========================================================================
    group_map = defaultdict(list)
    for b in range(B):
        group_map[index[b]].append((int(segment_index[b]), b))
    for gid in group_map:
        group_map[gid].sort(key=lambda x: x[0])
 
    # Log group structure
    n_groups = len(group_map)
    seg_counts = [len(segs) for segs in group_map.values()]
    logger.debug(f"Groups: {n_groups}, segments per group: "
                 f"min={min(seg_counts)}, max={max(seg_counts)}, "
                 f"multi-segment groups: {sum(1 for c in seg_counts if c > 1)}")
 
    # =========================================================================
    # Step 2: Precompute per-segment response positions
    # =========================================================================
    seg_resp_positions = {}
    for b in range(B):
        seg_resp_positions[b] = torch.nonzero(response_mask[b], as_tuple=False).squeeze(-1)
        if seg_resp_positions[b].dim() == 0:
            seg_resp_positions[b] = seg_resp_positions[b].unsqueeze(0)
 
    # Log segments with zero response tokens
    empty_segs = sum(1 for b in range(B) if len(seg_resp_positions[b]) == 0)
    if empty_segs > 0:
        logger.debug(f"WARNING: {empty_segs}/{B} segments have zero response tokens")
 
    # =========================================================================
    # Step 2.5: Apply empty segment penalty
    # =========================================================================
    if empty_segment_penalty != 0.0:
        token_level_rewards = token_level_rewards.clone()
        penalty_count = 0
 
        for b in range(B):
            if int(segment_index[b]) == 0:
                continue
 
            positions = seg_resp_positions[b]
            if len(positions) == 0:
                continue
 
            n_blocks = _count_contiguous_blocks(positions)
            if n_blocks <= 1:
                last_pos = positions[-1].item()
                token_level_rewards[b, last_pos] += empty_segment_penalty
                penalty_count += 1
 
        if penalty_count > 0:
            logger.debug(f"Empty segment penalty applied to {penalty_count} segments")
            _check_tensor("token_level_rewards [AFTER PENALTY]", token_level_rewards)
 
    # =========================================================================
    # Step 3: Extract V^high at segment boundaries
    # =========================================================================
    def _get_boundary_value(b: int) -> torch.Tensor:
        positions = seg_resp_positions[b]
        if len(positions) == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return values[b, positions[0].item()]
 
    v_high_self = {}
    v_high_next = {}
    is_final_segment = {}
 
    for gid, segs in group_map.items():
        for k, (seg_idx, b) in enumerate(segs):
            v_high_self[b] = _get_boundary_value(b)
            if k + 1 < len(segs):
                next_b = segs[k + 1][1]
                v_high_next[b] = _get_boundary_value(next_b)
                is_final_segment[b] = False
 
                # Log cross-segment boundary values
                v_self = v_high_self[b]
                v_next = v_high_next[b]
                if torch.isnan(v_self) or torch.isinf(v_self):
                    logger.debug(f"NaN/Inf in v_high_self: batch={b}, seg_idx={seg_idx}, "
                                 f"group={gid}, value={v_self.item()}")
                if torch.isnan(v_next) or torch.isinf(v_next):
                    logger.debug(f"NaN/Inf in v_high_next: batch={b} -> next_batch={next_b}, "
                                 f"seg_idx={seg_idx}, group={gid}, value={v_next.item()}")
                    # Log the source of the NaN
                    next_positions = seg_resp_positions[next_b]
                    if len(next_positions) > 0:
                        first_pos = next_positions[0].item()
                        logger.debug(f"  Source: values[{next_b}, {first_pos}] = "
                                     f"{values[next_b, first_pos].item()}")
                    else:
                        logger.debug(f"  Source: next segment has no response tokens")
            else:
                v_high_next[b] = torch.tensor(0.0, device=device, dtype=dtype)
                is_final_segment[b] = True
 
    # Aggregate check on boundary values
    v_high_self_vals = torch.tensor([v_high_self[b].item() for b in range(B)])
    v_high_next_vals = torch.tensor([v_high_next[b].item() for b in range(B)])
    logger.debug(f"V^high_self range: [{v_high_self_vals.min():.4f}, {v_high_self_vals.max():.4f}], "
                 f"NaN: {torch.isnan(v_high_self_vals).sum().item()}, "
                 f"Inf: {torch.isinf(v_high_self_vals).sum().item()}")
    logger.debug(f"V^high_next range: [{v_high_next_vals.min():.4f}, {v_high_next_vals.max():.4f}], "
                 f"NaN: {torch.isnan(v_high_next_vals).sum().item()}, "
                 f"Inf: {torch.isinf(v_high_next_vals).sum().item()}")
 
    # =========================================================================
    # Step 4: Low-level GAE (BATCHED)
    # =========================================================================
    c = gamma * lam
 
    v_bootstrap_vec = torch.zeros(B, device=device, dtype=dtype)
    for b in range(B):
        v_bootstrap_vec[b] = v_high_next[b]
 
    _check_tensor("v_bootstrap_vec [BEFORE GAE]", v_bootstrap_vec)
 
    nextvalues = v_bootstrap_vec.clone()
    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
 
    nan_detected_at_step = None
 
    for t in reversed(range(L)):
        mask_t = response_mask[:, t]
        r_t = token_level_rewards[:, t]
        v_t = values[:, t]
 
        delta = r_t + gamma * nextvalues - v_t
        new_gae = delta + c * lastgaelam
 
        nextvalues = v_t * mask_t + nextvalues * (1 - mask_t)
        lastgaelam = new_gae * mask_t + lastgaelam * (1 - mask_t)
 
        advantages[:, t] = lastgaelam
        returns[:, t] = lastgaelam + v_t
 
        # Check for NaN at each step (log only first occurrence)
        if nan_detected_at_step is None:
            if torch.isnan(lastgaelam).any() or torch.isnan(delta).any():
                nan_batch_indices = torch.nonzero(
                    torch.isnan(lastgaelam) | torch.isnan(delta)
                ).squeeze(-1).tolist()
                nan_detected_at_step = t
                logger.debug(f"NaN DETECTED in low-level GAE at step t={t}")
                logger.debug(f"  Affected batch indices: {nan_batch_indices[:10]}...")
                for b_nan in nan_batch_indices[:3]:  # Log details for first 3
                    logger.debug(f"  batch={b_nan}: r_t={r_t[b_nan].item():.4f}, "
                                 f"v_t={v_t[b_nan].item():.4f}, "
                                 f"nextvalues={nextvalues[b_nan].item():.4f}, "
                                 f"delta={delta[b_nan].item():.4f}, "
                                 f"lastgaelam={lastgaelam[b_nan].item():.4f}, "
                                 f"mask={mask_t[b_nan].item()}, "
                                 f"seg_idx={segment_index[b_nan]}, "
                                 f"group={index[b_nan]}")
 
    _check_tensor("advantages [AFTER LOW-LEVEL GAE]", advantages)
    _check_tensor("returns [AFTER LOW-LEVEL GAE]", returns)
 
    # Log advantage statistics at response positions only
    resp_advs = advantages[response_mask.bool()]
    resp_rets = returns[response_mask.bool()]
    if len(resp_advs) > 0:
        logger.debug(f"Low-level advantages (response tokens): "
                     f"mean={resp_advs.mean():.4f}, std={resp_advs.std():.4f}, "
                     f"min={resp_advs.min():.4f}, max={resp_advs.max():.4f}, "
                     f"NaN={torch.isnan(resp_advs).sum().item()}, "
                     f"Inf={torch.isinf(resp_advs).sum().item()}")
        logger.debug(f"Returns (response tokens): "
                     f"mean={resp_rets.mean():.4f}, std={resp_rets.std():.4f}, "
                     f"min={resp_rets.min():.4f}, max={resp_rets.max():.4f}, "
                     f"NaN={torch.isnan(resp_rets).sum().item()}, "
                     f"Inf={torch.isinf(resp_rets).sum().item()}")
 
    # =========================================================================
    # Step 5: High-level advantage for summary tokens
    # =========================================================================
    high_level_applied = 0
    high_level_clamped = 0
 
    for gid, segs in group_map.items():
        n_segs = len(segs)
        if n_segs <= 1:
            continue
 
        # --- Compute macro quantities ---
        macro_rewards = []
        macro_discounts = []
 
        for k, (seg_idx, b) in enumerate(segs):
            positions = seg_resp_positions[b]
            reward = token_level_rewards[b]
 
            if len(positions) == 0:
                macro_rewards.append(torch.tensor(0.0, device=device, dtype=dtype))
                macro_discounts.append(torch.tensor(1.0, device=device, dtype=dtype))
                continue
 
            n_pos = len(positions)
            gamma_pows = torch.pow(gamma, torch.arange(n_pos, device=device, dtype=dtype))
            r_macro = torch.sum(gamma_pows * reward[positions])
 
            macro_rewards.append(r_macro)
            macro_discounts.append(torch.pow(gamma, torch.tensor(n_pos, device=device, dtype=dtype)))
 
            if torch.isnan(r_macro) or torch.isinf(r_macro):
                logger.debug(f"NaN/Inf in macro_reward: group={gid}, seg={k}, "
                             f"batch={b}, r_macro={r_macro.item()}, n_pos={n_pos}")
 
        # --- High-level GAE ---
        high_advs = [None] * n_segs
        gae_high = torch.tensor(0.0, device=device, dtype=dtype)
 
        for k in range(n_segs - 1, -1, -1):
            v_k = v_high_self[segs[k][1]]
            r_k = macro_rewards[k]
            gamma_k = macro_discounts[k]
 
            if k + 1 < n_segs:
                v_k_next = v_high_self[segs[k + 1][1]]
            else:
                v_k_next = torch.tensor(0.0, device=device, dtype=dtype)
 
            delta_high = r_k + gamma_k * v_k_next - v_k
            gae_high = delta_high + gamma_k * lam * gae_high
            high_advs[k] = gae_high
 
            if torch.isnan(gae_high) or torch.isinf(gae_high):
                logger.debug(f"NaN/Inf in high-level GAE: group={gid}, seg={k}, "
                             f"batch={segs[k][1]}, delta={delta_high.item()}, "
                             f"gae={gae_high.item()}, r_k={r_k.item()}, "
                             f"gamma_k={gamma_k.item()}, v_k={v_k.item()}, "
                             f"v_k_next={v_k_next.item()}")
 
        # --- Assign to summary tokens ---
        for k, (seg_idx, b) in enumerate(segs):
            if is_final_segment[b]:
                continue
 
            positions = seg_resp_positions[b]
            if len(positions) == 0:
                continue
 
            summary_positions = _find_last_contiguous_block(positions)
            adv_high = high_advs[k]
 
            if prevent_penalize_context and adv_high < 0:
                if penalize_context_max_segments > 0:
                    penalty_ratio = min(k / penalize_context_max_segments, 1.0)
                else:
                    penalty_ratio = 0.0
                original_adv = adv_high.item()
                adv_high = adv_high * penalty_ratio
                if penalty_ratio < 1.0:
                    high_level_clamped += 1
 
            high_level_applied += len(summary_positions)
 
            for pos in summary_positions:
                p = pos.item()
                advantages[b, p] += adv_high
 
    if high_level_applied > 0:
        logger.debug(f"High-level advantage applied to {high_level_applied} summary tokens, "
                     f"clamped: {high_level_clamped}")
 
    _check_tensor("advantages [FINAL]", advantages)
    _check_tensor("returns [FINAL]", returns)
 
    # Final statistics
    final_resp_advs = advantages[response_mask.bool()]
    final_resp_rets = returns[response_mask.bool()]
    if len(final_resp_advs) > 0:
        logger.debug(f"Final advantages (response tokens): "
                     f"mean={final_resp_advs.mean():.4f}, std={final_resp_advs.std():.4f}, "
                     f"NaN={torch.isnan(final_resp_advs).sum().item()}, "
                     f"Inf={torch.isinf(final_resp_advs).sum().item()}")
        logger.debug(f"Final returns (response tokens): "
                     f"mean={final_resp_rets.mean():.4f}, std={final_resp_rets.std():.4f}, "
                     f"NaN={torch.isnan(final_resp_rets).sum().item()}, "
                     f"Inf={torch.isinf(final_resp_rets).sum().item()}")
 
    return advantages, returns


def _check_tensor(name: str, tensor: torch.Tensor):
    """Log warning if tensor contains NaN or Inf."""
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    if n_nan > 0 or n_inf > 0:
        logger.debug(f"BAD TENSOR '{name}': shape={list(tensor.shape)}, "
                     f"NaN={n_nan}, Inf={n_inf}, "
                     f"min={tensor[~torch.isnan(tensor)].min().item() if n_nan < tensor.numel() else 'all_nan'}, "
                     f"max={tensor[~torch.isnan(tensor)].max().item() if n_nan < tensor.numel() else 'all_nan'}")


def _find_last_contiguous_block(positions: torch.Tensor) -> torch.Tensor:
    """
    Find the last contiguous block of token positions.

    Given sorted positions like [2, 3, 4, 10, 11, 15, 16, 17, 18],
    returns [15, 16, 17, 18].

    This identifies summary tokens: the final LLM generation turn
    at the end of a segment, separated from earlier action turns
    by observation tokens (which have response_mask=0).
    """
    if len(positions) <= 1:
        return positions

    n = len(positions)
    block_start = n - 1

    for i in range(n - 2, -1, -1):
        if positions[i + 1] - positions[i] == 1:
            block_start = i
        else:
            break

    return positions[block_start:]


def _count_contiguous_blocks(positions: torch.Tensor) -> int:
    """
    Count the number of contiguous blocks in sorted token positions.

    Examples:
      [2, 3, 4, 10, 11, 15, 16, 17, 18] -> 3 blocks
      [1, 2, 3, 4, 5] -> 1 block (degenerate: no tool calling)
      [1, 2, 5, 6, 9] -> 3 blocks (healthy: has observation gaps)
      [] -> 0 blocks

    A healthy segment has >= 2 blocks (action-observation-action pattern).
    A degenerate segment has <= 1 block (no tool interaction happened).
    """
    if len(positions) == 0:
        return 0

    n_blocks = 1
    for i in range(len(positions) - 1):
        if positions[i + 1] - positions[i] > 1:
            n_blocks += 1

    return n_blocks

    
# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GDPO)  # or simply: @register_adv_est("gdpo")
def compute_gdpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    non_tensor_batch: Optional[dict] = None,
    batch: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GDPO: Group reward-Decoupled Normalization Policy Optimization.

    Instead of summing all reward dimensions first (like GRPO), GDPO normalizes
    each reward dimension independently within each group before aggregation.
    This prevents a dominant reward signal from drowning out weaker ones.

    Mathematical formulation:
        Step 1 – Group-wise decoupled normalization (via GRPO per dimension):
            For each reward dimension k, within each group g:
            A_k = (r_k - μ_group(r_k)) / (σ_group(r_k) + ε)

        Step 2 – Weighted aggregation:
            A_sum = Σ_k w_k · A_k

        Step 3 – Batch-level normalization (via masked_whiten):
            A_final = whiten(A_sum, response_mask)

    Args:
        token_level_rewards: (bs, response_length) – standard token-level rewards.
            Used as fallback when per-dimension rewards are not provided.
        response_mask: (bs, response_length)
        index: (bs,) – group id per sample (from ``uid``).
        epsilon: Numerical stability constant.
        norm_adv_by_std_in_grpo: Whether to normalize by std in GRPO.
        config: Algorithm configuration (optional).
        non_tensor_batch: Non-tensor batch data containing per-dimension reward scores.
        batch: Batch data containing prompts, attention_mask, etc.

    Note:
        Ref GDPO (https://arxiv.org/abs/2601.05242).

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) – same as advantages (outcome-only).
    """
    score_list = None
    reward_weights = None

    if config is not None and non_tensor_batch is not None and batch is not None:
        gdpo_reward_keys = config.get("gdpo_reward_keys", None)
        assert gdpo_reward_keys, (
            "GDPO requires 'algorithm.gdpo_reward_keys' listing the individual reward "
            "component keys returned by compute_score (e.g. ['format_reward', 'accuracy_reward'])."
        )
        device = token_level_rewards.device
        prompt_length = batch["prompts"].size(1)
        valid_response_length = batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1

        score_list = []
        for key in gdpo_reward_keys:
            assert key in non_tensor_batch, (
                f"GDPO reward key '{key}' not found in non_tensor_batch. "
                f"Available keys: {list(non_tensor_batch.keys())}. "
                f"Make sure your compute_score returns a dict containing '{key}'."
            )
            comp = non_tensor_batch[key]
            rm_score = torch.tensor(np.asarray(comp, dtype=np.float32), device=device)
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(rm_scores.size(0), device=device), valid_response_length] = rm_score
            score_list.append(rm_scores)

        gdpo_weights = config.get("gdpo_reward_weights", None)
        if gdpo_weights is not None:
            reward_weights = list(gdpo_weights)

    if score_list is None:
        score_list = [token_level_rewards]

    num_scores = len(score_list)

    if reward_weights is not None:
        weights = torch.tensor(reward_weights, dtype=torch.float32, device=token_level_rewards.device)
    else:
        weights = torch.ones(num_scores, dtype=torch.float32, device=token_level_rewards.device)

    new_advantage = None

    for i in range(num_scores):
        normalized_score, _ = compute_grpo_outcome_advantage(
            token_level_rewards=score_list[i],
            response_mask=response_mask,
            index=index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

        if new_advantage is None:
            new_advantage = weights[i] * normalized_score
        else:
            new_advantage += weights[i] * normalized_score

    advantages = verl_F.masked_whiten(new_advantage, response_mask) * response_mask

    return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # Maybe we need turn-based implementation
    # Currently, same as the original implementation
    # print(f"token_level_rewards: {token_level_rewards.shape}")
    # print(f"response_mask: {response_mask.shape}")
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # We don't reset after EOS, we use action_mask to mask the return and advantages
            # running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    # print(f"advantages after just compute: {advantages.shape}")
    # print(f"returns after just compute: {returns.shape}")
    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: True
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute returns (reward-to-go) for each timestep
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines tensor [batch_size, seq_len]
        baselines = torch.zeros_like(returns)

        # Compute per-step baseline for each prompt group
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                # Single trajectory - no baseline (advantage = return)
                continue

            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data [N, seq_len]
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]

            # Compute per-timestep baseline: B_t = Σ[G_t × W_t] / Σ[W_t]
            # where W_t = Σ_{j=1}^t ||s_j||² (cumulative path variance)
            # Shape: [seq_len]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)  # Sum over trajectories
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon

            baseline_per_step = numerator / denominator  # [seq_len]

            # Assign to all trajectories in this group
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths = mask_group.sum(dim=-1)
                sorted_lengths, _ = torch.sort(response_lengths)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages: A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE)
def compute_multi_turn_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        # Compute returns (reward-to-go) for each timestep
        token_returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Step 4: Concatenate returns and w_cumulative for each trajectory
        # This allows us to compute baseline per timestep for each trajectory
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)  # [shape: (bs * n, )]
        max_response_length = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
        all_w_values = w_cumulative.new_zeros(
            (len(response_lengths), max_response_length)
        )  # [shape: (bs * n, max_response_length)]
        all_returns = torch.zeros_like(all_w_values)
        for i in range(len(response_lengths)):
            length = int(response_lengths[i].item())
            if length == 0:
                continue
            mask = response_mask[i].bool()
            all_w_values[i, :length] = w_cumulative[i, mask]
            all_returns[i, :length] = token_returns[i, mask]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(all_returns)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=all_returns.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = all_w_values[traj_idx]  # [shape: (N, max_response_length)]
            R_group = all_returns[traj_idx]  # [shape: (N, max_response_length)]
            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths_group = response_lengths[traj_idx]
                sorted_lengths, _ = torch.sort(response_lengths_group)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths_group == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages
        all_advantages = all_returns - baselines  # [shape: (bs * n, max_response_length)]

        advantages = torch.zeros_like(token_returns)  # [shape: (bs * n, turn * response_length)]
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            advantages[i, response_mask[i].bool()] = all_advantages[i, : response_lengths[i]]

        advantages = advantages * response_mask  # [shape: (bs * n * turn, response_length)]

    return advantages, token_returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
    eps: float = 1e-8,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / (batch_num_tokens + eps) * dp_size
    elif loss_agg_mode in ["seq-mean-token-sum", "seq-mean-token-sum-norm"]:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / (global_batch_size + eps) * dp_size  # seq-mean
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                horizon = loss_mask.shape[-1]
                loss_scale_factor = horizon
            loss /= loss_scale_factor
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / (global_batch_size + eps) * dp_size  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    logger.debug(f"[compute_policy_loss] advantages: {advantages.shape}")
    logger.debug(f"[compute_policy_loss] ratio: {ratio.shape}")
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("dppo_tv")
def compute_policy_loss_dppo_tv(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-TV.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Note: the clip_ratio is different from the standard PPO, it is the TV divergence threshold for DPPO.
    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Instead of dual-clip PPO, we use truncated importance sampling (TIS) to clip the policy loss.
    # However, a large threshold is recommended to avoid performance degradation due to the truncation bias.
    # See Section 5.4 in https://arxiv.org/pdf/2602.04879 for more details.
    clip_ratio_c = config.get("clip_ratio_c", 20.0)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    # Compute valid mask for DPPO-Binary-TV
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    valid_positive_mask = (prob - old_prob) <= clip_divergence_high
    valid_negative_mask = (prob - old_prob) >= -clip_divergence_low
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("dppo_kl")
def compute_policy_loss_dppo_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-KL.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Note: the clip_ratio is different from the standard PPO, it is the KL divergence threshold for DPPO.
    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Instead of dual-clip PPO, we use truncated importance sampling (TIS) to clip the policy loss.
    # However, a large threshold is recommended to avoid performance degradation due to the truncation bias.
    # See Section 5.4 in https://arxiv.org/pdf/2602.04879 for more details.
    clip_ratio_c = config.get("clip_ratio_c", 20.0)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    # Compute valid mask for DPPO-Binary-KL
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    binary_kl = old_prob * (old_log_prob - log_prob) + (1 - old_prob) * torch.log(
        (1.0 - old_prob + 1e-8) / (1.0 - prob + 1e-8)
    )
    valid_positive_mask = (binary_kl <= clip_divergence_high) | (prob <= old_prob)
    valid_negative_mask = (binary_kl <= clip_divergence_low) | (prob >= old_prob)
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard DPPO)
    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # return metrics dict
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Compute importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CISPO: Clip the importance sampling weights
    # KEY: Apply stop gradient to the clipped ratio
    # This prevents gradients from flowing through the ratio computation and clipping
    # Gradients only flow through log_prob in the final loss term
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO objective function (to maximize): J = sg(clip(ratio)) * A * log π_θ
    # Loss function (to minimize): L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # Track clipping statistics
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in CISPO)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2


    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)

    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expected value of KL, but the expected gradient of k1 and k3
    estimator is not the expected gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level (see rollout_corr_helper for supported modes)
        rollout_rs_threshold: Threshold specification for rejection sampling
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from ....verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
            )
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics
