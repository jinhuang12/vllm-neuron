# SPDX-License-Identifier: Apache-2.0
"""KDA prefill output kernel against an fp64 oracle of the checkpoint's KDA output tail.

The oracle is ``KimiDeltaAttention``'s last two steps, ``o_norm`` (``FusedRMSNormGated``
with a sigmoid gate: ``x * rsqrt(mean(x^2) + eps) * w * sigmoid(g)`` per head) and
``o_proj``, in fp64 on CPU, for one rank's 3 heads. Each regime passes when every
element is within 2^-7 of its value plus 2^-7 of its row RMS and every row's cosine is
at least 0.9999. The kernel runs on the first visible Neuron core, or on the NKI CPU
simulator under ``VLLM_NEURON_CPU_MODE=1`` with ``NKI_SIMULATOR=1``.
"""

import pytest
import torch
from torch import nn

from vllm_neuron import envs
from vllm_neuron.functional.kda.output import kda_output_prefill

HEADS, KDIM, HIDDEN, EPS = 3, 128, 7168, 1e-5
WIDTH = HEADS * KDIM
REL_BOUND = RMS_BOUND = 2.0**-7
MIN_COSINE = 0.9999
DEVICE = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"


class GatedOutput(nn.Module):
    """The KDA output tail, ``o_proj(o_norm(o, g))``, in fp64."""

    def __init__(self, o_norm_weight: torch.Tensor, o_proj_weight: torch.Tensor):
        super().__init__()
        self.o_norm_weight = nn.Parameter(o_norm_weight.double(), requires_grad=False)
        self.o_proj = nn.Linear(WIDTH, HIDDEN, bias=False, dtype=torch.float64)
        self.o_proj.weight.data.copy_(o_proj_weight.double())

    def forward(self, core: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        o = core.double().reshape(-1, HEADS, KDIM)
        g = gate.double().reshape(-1, HEADS, KDIM)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
        o = o * self.o_norm_weight * torch.sigmoid(g)
        return self.o_proj(o.reshape(-1, WIDTH))


def _weights(gen: torch.Generator):
    """A rank's ``o_norm`` (fp32) and ``o_proj`` slice (bf16) at the checkpoint's scales."""
    o_norm = 0.034 + 0.0034 * torch.randn(KDIM, generator=gen)
    o_proj = (0.023 * torch.randn(HIDDEN, WIDTH, generator=gen)).to(torch.bfloat16)
    return o_norm, o_proj


def _regime(name: str):
    """``(core, gate, o_norm, o_proj)`` for one registered regime."""
    gen = torch.Generator().manual_seed(sum(map(ord, name)))
    tokens = 8192 if name == "long_chunk" else 2048
    core = torch.randn(tokens, WIDTH, generator=gen)
    gate = torch.randn(tokens, WIDTH, generator=gen)
    if name == "correlated":
        core = 0.9 * torch.randn(1, WIDTH, generator=gen) + 0.1 * core
        core[:, 200] *= 100.0
    elif name == "repeated":
        core = core[:1].expand(tokens, WIDTH).clone()
        gate = gate[:1].expand(tokens, WIDTH).clone()
    elif name == "saturated_gate":
        gate = 12.0 * torch.sign(gate)
    return (core, gate, *_weights(gen))


def _kernel(core, gate, o_norm, o_proj) -> torch.Tensor:
    args = [t.to(DEVICE) for t in (core, gate, o_norm, o_proj)]
    if DEVICE == "cpu":
        return kda_output_prefill(*args, EPS)
    fn = torch.compile(lambda c, g, w, p: kda_output_prefill(c, g, w, p, EPS),
                       backend=envs.get_compile_backend_name(), fullgraph=True, dynamic=False)
    return fn(*args).cpu()


def _assert_class_b(out: torch.Tensor, ref: torch.Tensor) -> None:
    out = out.double()
    assert torch.isfinite(out).all()
    row_rms = ref.pow(2).mean(-1, keepdim=True).sqrt()
    excess = (out - ref).abs() - (REL_BOUND * ref.abs() + RMS_BOUND * row_rms)
    assert excess.max() <= 0, f"worst element exceeds its bound by {excess.max().item():.3g}"
    cosine = nn.functional.cosine_similarity(out, ref, dim=-1)
    assert cosine.min() >= MIN_COSINE, f"lowest row cosine {cosine.min().item():.6f}"


@pytest.mark.parametrize("name", ["correlated", "repeated", "saturated_gate", "long_chunk"])
def test_matches_oracle(name):
    core, gate, o_norm, o_proj = _regime(name)
    ref = GatedOutput(o_norm, o_proj)(core, gate)
    _assert_class_b(_kernel(core, gate, o_norm, o_proj), ref)


def test_nan_row_stays_in_its_row():
    core, gate, o_norm, o_proj = _regime("nan_row")
    clean = _kernel(core, gate, o_norm, o_proj)
    nan_row = 1000
    core[nan_row] = float("nan")
    out = _kernel(core, gate, o_norm, o_proj)
    assert torch.isnan(out[nan_row]).all()
    keep = torch.arange(core.shape[0]) != nan_row
    assert torch.equal(out[keep], clean[keep])
    _assert_class_b(out[keep], GatedOutput(o_norm, o_proj)(core[keep], gate[keep]))
