# SPDX-License-Identifier: Apache-2.0
"""The KDA prefill short conv kernel against a float64 oracle of the reference conv.

The reference layer runs q, k and v each through a ShortConvolution of 4 taps with
SiLU: a depthwise ``nn.Conv1d`` with no bias, causal, whose cache supplies the 3
input rows before the first token. The oracle is that conv as a float64
``nn.Conv1d`` module over ``[history; x]``, then SiLU. Each item is one numerics
regime at one rank's 3 x 384 channels, judged by the bf16-output class: per element
error at most 2^-7 of the value plus 2^-7 of the token row's RMS, and token-row
cosine at least 0.9999.

The kernel runs on the Neuron device, compiled as a one-call graph, when
``NEURON_RT_VISIBLE_CORES`` names a core; else eagerly on the NKI CPU simulator.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.functional.kda.depthwise_conv1d import depthwise_conv1d_silu

DEVICE = "neuron:0" if os.environ.get("NEURON_RT_VISIBLE_CORES") else "cpu"
BLOCKS = (384, 384, 384)
CHANNELS = sum(BLOCKS)
TAPS = 4
TOKENS = 2048

if DEVICE == "cpu":
    _conv = depthwise_conv1d_silu
else:
    _conv = torch.compile(
        depthwise_conv1d_silu,
        backend=get_compile_backend_name(),
        fullgraph=True,
        dynamic=False,
    )


@pytest.fixture(autouse=True)
def _venue(monkeypatch):
    if DEVICE == "cpu":
        monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
        monkeypatch.setenv("NKI_SIMULATOR", "1")


def _oracle(history, x, weight):
    """``[C, T]`` float64: SiLU of the causal depthwise conv over ``[history; x]``."""
    conv = torch.nn.Conv1d(CHANNELS, CHANNELS, TAPS, groups=CHANNELS, bias=False,
                           dtype=torch.float64)
    with torch.no_grad():
        conv.weight.copy_(weight.double().unsqueeze(1))
        rows = torch.cat((history, x)).double()
        return F.silu(conv(rows.t().unsqueeze(0)))[0]


def _run(history, x, weight):
    """The kernel's ``[C, T]`` output for ``[history; x]``, back on the CPU."""
    q, k, v = (b.contiguous().to(DEVICE) for b in torch.split(x, BLOCKS, dim=1))
    history, weight = history.contiguous().to(DEVICE), weight.contiguous().to(DEVICE)
    out = _conv(history, q, k, v, weight)
    return out.cpu()


def _assert_class_b(got, want):
    """Per element and token-row cosine bounds of the bf16-output class."""
    got, want = got.double().t(), want.t()
    rms = want.pow(2).mean(dim=1, keepdim=True).sqrt()
    excess = (got - want).abs() - 2**-7 * (want.abs() + rms)
    worst = excess.max().item()
    assert worst <= 0.0, f"{int((excess > 0).sum())} elements out, worst by {worst:.3e}"
    cosine = F.cosine_similarity(got, want, dim=1).min().item()
    assert cosine >= 0.9999, f"lowest token-row cosine {cosine:.6f}"


def _inputs(seed, rows, scale=0.3):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, CHANNELS, generator=generator)
    weight = torch.randn(CHANNELS, TAPS, generator=generator) * scale
    return x, weight


def test_r1_correlated_rows_and_one_large_channel():
    x, weight = _inputs(1, TAPS - 1 + TOKENS)
    u = x[0].clone()
    x = 0.9 * u + 0.1 * x
    x[:, 700] *= 100.0
    x = x.bfloat16()
    history, tokens = x[:TAPS - 1], x[TAPS - 1:]
    _assert_class_b(_run(history, tokens, weight), _oracle(history, tokens, weight))


def test_r2_repeated_rows():
    x, weight = _inputs(2, 1)
    x = x.expand(TAPS - 1 + TOKENS, CHANNELS).bfloat16()
    history, tokens = x[:TAPS - 1], x[TAPS - 1:]
    _assert_class_b(_run(history, tokens, weight), _oracle(history, tokens, weight))


def test_r3_silu_inputs_at_plus_and_minus_12():
    noise, _ = _inputs(3, TAPS - 1 + TOKENS)
    sign = torch.tensor([1.0, -1.0]).repeat(CHANNELS // 2)
    x = (12.0 * sign + 0.05 * noise).bfloat16()
    weight = 0.25 + 1e-3 * _inputs(4, 1)[1]
    history, tokens = x[:TAPS - 1], x[TAPS - 1:]
    _assert_class_b(_run(history, tokens, weight), _oracle(history, tokens, weight))


def test_r4_state_carried_over_16_chunks():
    chunk, chunks = 512, 16
    x, weight = _inputs(5, TAPS - 1 + chunk * chunks)
    x = x.bfloat16()
    lead = TAPS - 1
    outs = [
        _run(x[s:s + lead], x[s + lead:s + lead + chunk], weight)
        for s in range(0, chunk * chunks, chunk)
    ]
    _assert_class_b(torch.cat(outs, dim=1), _oracle(x[:lead], x[lead:], weight))


def test_r5_one_nan_row_stays_in_its_window():
    x, weight = _inputs(6, TAPS - 1 + TOKENS)
    x = x.bfloat16()
    history, tokens = x[:TAPS - 1], x[TAPS - 1:]
    clean = _run(history, tokens, weight)
    poisoned = tokens.clone()
    nan_row = TOKENS // 2 - 2
    poisoned[nan_row] = float("nan")
    got = _run(history, poisoned, weight)
    window = torch.zeros(TOKENS, dtype=torch.bool)
    window[nan_row:nan_row + TAPS] = True
    assert got[:, window].isnan().all()
    bits = torch.int16
    assert torch.equal(got[:, ~window].view(bits), clean[:, ~window].view(bits))
    _assert_class_b(clean, _oracle(history, tokens, weight))
