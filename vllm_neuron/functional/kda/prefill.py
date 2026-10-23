# SPDX-License-Identifier: Apache-2.0
"""KDA prefill recurrence through the nkilib chunked delta-rule kernel.

The kernel takes the raw projections and applies the q and k L2 norms, the
query scale, the beta sigmoid and the bounded gate
``lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`` itself. Tensors are
head-major fp32, and the recurrent state is key-major ``[H, K, V]``: the
transpose of the ``[H, V, K]`` layout that upstream vLLM stores.
"""

from typing import Optional, Tuple

import torch

from nkilib.experimental.kimi_k3.kda.kda_prefill_nki import (
    CHUNK,
    SUB_CHUNK,
    kda_prefill as _kda_prefill_kernel,
)

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel


def kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    *,
    lower_bound: float,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the KDA recurrence over one sequence and return its output and final state.

    Args:
        q, k, g: ``[H, T, K]`` raw query, key and gate projections.
        v: ``[H, T, V]`` values.
        beta: ``[H, T]`` or ``[H, T, 1]`` beta logits.
        a_log: ``[H]`` or ``[H, 1]`` log gate scale.
        dt_bias: ``[H, K]`` gate bias.
        state: ``[H, K, V]`` fp32 state carried from the previous chunk of the
            same sequence, or None for a fresh sequence. A fresh sequence
            starts from a zero-filled state, never from a reused buffer scaled
            by zero, so a stale NaN cannot leak into it.
        lower_bound: gate lower bound, in ``[-5, 0]``.
        scale: query scale; defaults to ``K ** -0.5``.

    Returns:
        ``(out, final_state)``: ``[H, T, V]`` fp32 and ``[H, K, V]`` fp32. On
        the device the kernel writes the final state into ``state`` in place.
    """
    num_heads, num_tokens, head_k = q.shape
    head_v = v.shape[-1]
    assert k.shape == g.shape == q.shape, (
        f"q {tuple(q.shape)}, k {tuple(k.shape)} and g {tuple(g.shape)} must match"
    )
    assert v.shape[:2] == (num_heads, num_tokens), (
        f"v {tuple(v.shape)} must be [{num_heads}, {num_tokens}, V]"
    )
    assert num_tokens >= CHUNK and num_tokens % CHUNK == 0, (
        f"T must be a positive multiple of {CHUNK}, got {num_tokens}"
    )
    assert head_k <= 128 and head_v <= 128, f"K {head_k} and V {head_v} must be <= 128"
    assert -5.0 <= lower_bound <= 0.0, f"lower_bound {lower_bound} must be in [-5, 0]"
    if not can_run_kernel(q):
        raise RuntimeError(
            "kda_prefill has no torch path: run it on a Neuron device, or set "
            "NKI_SIMULATOR=1 in CPU mode"
        )

    device = q.device
    fp32 = torch.float32
    if state is None:
        state = torch.zeros(num_heads, head_k, head_v, dtype=fp32, device=device)
    assert state.shape == (num_heads, head_k, head_v) and state.dtype == fp32, (
        f"state must be fp32 [{num_heads}, {head_k}, {head_v}], got "
        f"{state.dtype} {tuple(state.shape)}"
    )
    strict, inclusive, eye = _block_masks(device)
    out = torch.empty(num_heads, num_tokens, head_v, dtype=fp32, device=device)
    out, final_state = wrap_nki(_kda_prefill_kernel)[get_logical_nc_config()](
        q_ref=q.to(fp32),
        k_ref=k.to(fp32),
        v_ref=v.to(fp32),
        g_ref=g.to(fp32),
        beta_ref=beta.reshape(num_heads, num_tokens, 1).to(fp32),
        a_log_ref=a_log.reshape(num_heads, 1).to(fp32),
        dt_bias_ref=dt_bias.to(fp32),
        blockdiag_ref=strict,
        blockdiag_incl_ref=inclusive,
        eye_ref=eye,
        out=out,
        state_ref=state,
        # The seed always enters at scale 1: a zero scale would multiply a
        # possibly non-finite buffer instead of starting from a filled one.
        seed_scale_ref=torch.ones(1, 1, dtype=fp32, device=device),
        scale_ref=torch.full(
            (1, 1), head_k**-0.5 if scale is None else scale, dtype=fp32, device=device
        ),
        lower_bound_ref=torch.full((1, 1), lower_bound, dtype=fp32, device=device),
    )
    return out, final_state


def _block_masks(device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Return the kernel's strict and inclusive block-lower masks and the identity."""
    idx = torch.arange(CHUNK, device=device)
    same_block = (idx[:, None] // SUB_CHUNK) == (idx[None, :] // SUB_CHUNK)
    strict = (same_block & (idx[:, None] > idx[None, :])).to(torch.float32)
    inclusive = (same_block & (idx[:, None] >= idx[None, :])).to(torch.float32)
    return strict, inclusive, torch.eye(CHUNK, dtype=torch.float32, device=device)
