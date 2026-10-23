# SPDX-License-Identifier: Apache-2.0
"""fp64 oracle test for the AttnRes prefill mix, one item per numerics regime.

The oracle is ``_apply_attn_res`` from the checkpoint's ``modeling_kimi_linear.py``, run
in fp64. Pass rule, per token row against the oracle: every element within
2^-7 of its value plus 2^-7 of the row RMS, and row cosine at least 0.9999.

On a Neuron device the wrapper runs under ``torch.compile`` on core 0 of
``NEURON_RT_VISIBLE_CORES``; the runtime needs ``NEURON_RT_ROOT_COMM_ID`` set to load the
program. With ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1`` it runs in the NKI simulator.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_neuron import envs
from vllm_neuron.functional.attention.attn_res_prefill import attn_res_prefill

HIDDEN = 7168
EPS = 1e-5
ELEMENT_TOL = 2.0**-7
ROW_RMS_TOL = 2.0**-7
MIN_ROW_COSINE = 0.9999


def _reference(bank, stream, norm_weight, proj_weight, eps):
    """fp64 transcription of the reference ``_apply_attn_res``."""
    v = torch.cat((bank, stream.unsqueeze(1)), dim=1).double()
    variance = v.pow(2).mean(-1, keepdim=True)
    k = v * torch.rsqrt(variance + eps)
    score_weight = norm_weight.double() * proj_weight.squeeze(0).double()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v).squeeze(1)


def _score_weights(gen):
    """Score norm weight [H] and projection weight [1, H] in bf16, as the checkpoint stores them."""
    norm_weight = 1.0 + 0.1 * torch.randn(HIDDEN, generator=gen)
    proj_weight = 0.035 * torch.randn(1, HIDDEN, generator=gen)
    return norm_weight.bfloat16(), proj_weight.bfloat16()


_compiled = None


def _run(bank, stream, norm_weight, proj_weight):
    """Run the wrapper on this venue and return its output on the CPU."""
    global _compiled
    score_w = norm_weight.float() * proj_weight.squeeze(0).float()
    if envs.VLLM_NEURON_CPU_MODE:
        return attn_res_prefill(bank, stream, score_w, EPS)
    if _compiled is None:
        torch.classes.neuron.Runtime().initialize()
        _compiled = torch.compile(
            attn_res_prefill, backend=envs.get_compile_backend_name(), fullgraph=True
        )
    device = torch.device("neuron", 0)
    out = _compiled(bank.to(device), stream.to(device), score_w.to(device), EPS)
    return out.cpu()


def _assert_within_bf16_rule(out, ref):
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape
    out64 = out.double()
    bound = ELEMENT_TOL * ref.abs() + ROW_RMS_TOL * ref.pow(2).mean(-1, keepdim=True).sqrt()
    over = (out64 - ref).abs() > bound
    assert not over.any(), (
        f"{int(over.sum())} elements over the bound, first at {over.nonzero()[0].tolist()}, "
        f"max error {float((out64 - ref).abs().max()):.3e}"
    )
    cosine = F.cosine_similarity(out64, ref, dim=-1)
    assert float(cosine.min()) >= MIN_ROW_COSINE, f"min row cosine {float(cosine.min()):.6f}"


def _correlated(gen):
    """Sources 0.9 u + 0.1 noise around one shared u, with one hidden channel 100x the rest."""
    tokens, rows = 128, 8
    u = torch.randn(HIDDEN, generator=gen)
    v = 0.9 * u + 0.1 * torch.randn(tokens, rows + 1, HIDDEN, generator=gen)
    v[..., 1234] *= 100.0
    v = v.bfloat16()
    return v[:, :rows].contiguous(), v[:, rows].contiguous()


def _repeated_tokens(gen):
    """One token's two sources repeated on every token row."""
    tokens, rows = 128, 1
    v = torch.randn(1, rows + 1, HIDDEN, generator=gen).expand(tokens, -1, -1).bfloat16()
    return v[:, :rows].contiguous(), v[:, rows].contiguous()


def _equal_scores(gen):
    """Every source a positive multiple of one vector, so the RMS norm gives equal scores."""
    tokens, rows = 128, 8
    scales = 2.0 ** (torch.arange(rows + 1) / 2.0)
    v = torch.randn(tokens, 1, HIDDEN, generator=gen) * scales.view(1, -1, 1)
    v = v.bfloat16()
    return v[:, :rows].contiguous(), v[:, rows].contiguous()


def _nine_sources(gen):
    """Nine sources of different scales over two 128-token tiles."""
    tokens, rows = 256, 8
    scales = 0.5 + 3.5 * torch.rand(tokens, rows + 1, 1, generator=gen)
    v = (torch.randn(tokens, rows + 1, HIDDEN, generator=gen) * scales).bfloat16()
    return v[:, :rows].contiguous(), v[:, rows].contiguous()


REGIMES = {
    "correlated": _correlated,
    "repeated_tokens": _repeated_tokens,
    "equal_scores": _equal_scores,
    "nine_sources": _nine_sources,
}


@pytest.mark.parametrize("regime", list(REGIMES))
def test_matches_fp64_reference(regime):
    gen = torch.Generator().manual_seed(20261023)
    norm_weight, proj_weight = _score_weights(gen)
    bank, stream = REGIMES[regime](gen)
    out = _run(bank, stream, norm_weight, proj_weight)
    _assert_within_bf16_rule(out, _reference(bank, stream, norm_weight, proj_weight, EPS))


def test_nan_row_stays_in_its_row():
    gen = torch.Generator().manual_seed(20261023)
    norm_weight, proj_weight = _score_weights(gen)
    tokens, rows, nan_row = 128, 8, 37
    v = torch.randn(tokens, rows + 1, HIDDEN, generator=gen).bfloat16()
    bank, stream = v[:, :rows].contiguous(), v[:, rows].contiguous()
    clean = _run(bank, stream, norm_weight, proj_weight)
    bank[nan_row] = float("nan")
    stream[nan_row] = float("nan")
    out = _run(bank, stream, norm_weight, proj_weight)

    assert torch.isnan(out[nan_row]).all()
    others = torch.arange(tokens) != nan_row
    assert torch.equal(out[others], clean[others])
    ref = _reference(bank[others], stream[others], norm_weight, proj_weight, EPS)
    _assert_within_bf16_rule(out[others], ref)
