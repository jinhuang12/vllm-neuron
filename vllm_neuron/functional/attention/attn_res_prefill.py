# SPDX-License-Identifier: Apache-2.0
"""AttnRes prefill mix: a softmax mixture of the residual stream with its banked snapshots.

Wraps the nkilib ``attn_res_cte`` kernel, which puts tokens on the partitions in tiles of
up to 128 and splits them across the LNC cores. It equals the reference
``_apply_attn_res`` with ``score_w = norm.weight * proj.weight``. There is no torch path:
on CPU the kernel runs in the NKI simulator (``NKI_SIMULATOR=1``).
"""

import torch
from nkilib.experimental.kimi_k3.attn_res.attn_res_cte import attn_res_cte

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.hardware_config import get_logical_nc_config

# The kernel's contract: tokens in whole tiles of 128, H in chunks of 512, 1 to 15 bank rows.
TOKEN_TILE = 128
HIDDEN_CHUNK = 512
MAX_BANK_ROWS = 15

_attn_res_cte = wrap_nki(attn_res_cte)


def attn_res_prefill(
    bank: torch.Tensor,
    stream: torch.Tensor,
    score_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Mix ``stream`` with the ``bank`` rows of each token by a softmax over their scores.

    Args:
        bank: ``[T, C, H]`` snapshots of the stream, ``C`` from 1 to 15, H contiguous.
        stream: ``[T, H]`` the current stream, the last source. Same dtype as ``bank``.
        score_w: ``[H]`` float32, the norm weight times the score projection weight.
        eps: RMSNorm epsilon of the score norm.

    Returns:
        ``[T, H]`` in the dtype of ``stream``.
    """
    assert bank.dim() == 3 and stream.dim() == 2, (
        f"bank must be [T, C, H] and stream [T, H], got {tuple(bank.shape)} "
        f"and {tuple(stream.shape)}"
    )
    tokens, rows, hidden = bank.shape
    assert tuple(stream.shape) == (tokens, hidden), (
        f"stream {tuple(stream.shape)} must be [{tokens}, {hidden}]"
    )
    assert tokens % TOKEN_TILE == 0, f"T={tokens} must be a multiple of {TOKEN_TILE}"
    assert hidden % HIDDEN_CHUNK == 0, f"H={hidden} must be a multiple of {HIDDEN_CHUNK}"
    assert 1 <= rows <= MAX_BANK_ROWS, f"bank rows C={rows} must be in [1, {MAX_BANK_ROWS}]"
    assert bank.dtype == stream.dtype, (
        f"bank {bank.dtype} and stream {stream.dtype} must share a dtype"
    )
    assert score_w.dtype == torch.float32 and tuple(score_w.shape) == (hidden,), (
        f"score_w must be float32 [{hidden}], got {score_w.dtype} {tuple(score_w.shape)}"
    )

    # wrap_nki(...)[N] selects the LNC degree (grid size). It must equal the LNC the
    # graph compiles at, or the compiler rejects the launch.
    return _attn_res_cte[get_logical_nc_config()](
        bank=bank.contiguous(),
        stream=stream.contiguous(),
        score_w=score_w.contiguous(),
        eps=eps,
    )
