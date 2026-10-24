# SPDX-License-Identifier: Apache-2.0
"""KDA prefill recurrence through its NKI kernel.

q, k and v come as the short conv stores them, bf16 channel-major ``[H * K, T]``.
The kernel applies the q and k L2 norms, the query scale, the beta sigmoid and
the bounded gate ``lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`` itself.
The recurrent state is fp32 key-major ``[H, K, V]``: the transpose of the
``[H, V, K]`` layout that upstream vLLM stores.
"""

from typing import Optional, Tuple

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.kda.prefill_kernel import CHUNK, SUB_CHUNK, kda_prefill_kernel
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
    valid: Optional[torch.Tensor] = None,
    *,
    lower_bound: float,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the KDA recurrence over one sequence and return its output and final state.

    Args:
        q, k, v: ``[H * K, T]`` bf16 channel-major query, key and value, head by head.
        g: ``[H, T, K]`` raw gate projection.
        beta: ``[H, T]`` or ``[H, T, 1]`` beta logits.
        a_log: ``[H]`` or ``[H, 1]`` log gate scale.
        dt_bias: ``[H, K]`` gate bias.
        state: ``[H, K, K]`` fp32 state carried from the previous chunk of the
            same sequence, or None for a fresh sequence, which starts from a
            zero-filled state.
        valid: ``[T]`` bool, False for a padding token, or None when every token
            is real. A padding token adds nothing to the state and its output row
            is 0. A token with a non-finite q, k, v, g or beta adds nothing to
            the state either, but the output rows from it to the end of the
            sequence, and the final state, are NaN.
        lower_bound: gate lower bound, in ``[-5, 0]``.
        scale: query scale; defaults to ``K ** -0.5``.

    Returns:
        ``(out, final_state)``: ``[H, T, K]`` fp32 and ``[H, K, K]`` fp32. On
        the device the kernel writes the final state into ``state`` in place.
        A ``T`` that is not a multiple of 128 is padded up with padding tokens.
    """
    num_heads, num_tokens, head_dim = g.shape
    channels = num_heads * head_dim
    for name, x in (("q", q), ("k", k), ("v", v)):
        assert x.shape == (channels, num_tokens) and x.dtype == torch.bfloat16, (
            f"{name} must be bf16 [{channels}, {num_tokens}], got {x.dtype} {tuple(x.shape)}"
        )
    assert num_tokens >= 1, "the sequence has no token"
    assert head_dim <= 128, f"head dim {head_dim} must be <= 128"
    assert -5.0 <= lower_bound <= 0.0, f"lower_bound {lower_bound} must be in [-5, 0]"
    if not can_run_kernel(q):
        raise RuntimeError(
            "kda_prefill has no torch path: run it on a Neuron device, or set "
            "NKI_SIMULATOR=1 in CPU mode"
        )

    device = q.device
    fp32 = torch.float32
    if state is None:
        state = torch.zeros(num_heads, head_dim, head_dim, dtype=fp32, device=device)
    assert state.shape == (num_heads, head_dim, head_dim) and state.dtype == fp32, (
        f"state must be fp32 [{num_heads}, {head_dim}, {head_dim}], got "
        f"{state.dtype} {tuple(state.shape)}"
    )
    valid = torch.ones(num_tokens, dtype=fp32, device=device) if valid is None else valid.to(fp32)
    g = g.to(fp32)
    beta = beta.reshape(num_heads, num_tokens).to(fp32)
    extra = -num_tokens % CHUNK
    if extra:
        # Zero tokens marked padding: the kernel gives them gate 0, so they leave the state as it is.
        q, k, v = (torch.nn.functional.pad(x, (0, extra)) for x in (q, k, v))
        g = torch.nn.functional.pad(g, (0, 0, 0, extra))
        beta, valid = torch.nn.functional.pad(beta, (0, extra)), torch.nn.functional.pad(valid, (0, extra))
    padded = num_tokens + extra
    strict, inclusive, eye = _block_masks(device)
    out = torch.empty(num_heads, padded, head_dim, dtype=fp32, device=device)
    out, final_state = wrap_nki(kda_prefill_kernel)[get_logical_nc_config()](
        q_ref=q,
        k_ref=k,
        v_ref=v,
        g_ref=g,
        beta_ref=beta.reshape(num_heads, padded, 1),
        a_log_ref=a_log.reshape(num_heads, 1).to(fp32),
        dt_bias_ref=dt_bias.to(fp32),
        valid_ref=valid.reshape(padded, 1),
        strict_ref=strict,
        incl_ref=inclusive,
        eye_ref=eye,
        out=out,
        state_ref=state,
        scale_ref=torch.full(
            (1, 1), head_dim**-0.5 if scale is None else scale, dtype=fp32, device=device
        ),
        lower_bound_ref=torch.full((1, 1), lower_bound, dtype=fp32, device=device),
    )
    return out[:, :num_tokens], final_state


def _block_masks(device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Return the kernel's strict and inclusive block-lower masks and the identity."""
    idx = torch.arange(CHUNK, device=device)
    same_block = (idx[:, None] // SUB_CHUNK) == (idx[None, :] // SUB_CHUNK)
    strict = (same_block & (idx[:, None] > idx[None, :])).to(torch.float32)
    inclusive = (same_block & (idx[:, None] >= idx[None, :])).to(torch.float32)
    return strict, inclusive, torch.eye(CHUNK, dtype=torch.float32, device=device)
