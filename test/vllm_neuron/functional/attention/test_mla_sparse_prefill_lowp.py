# SPDX-License-Identifier: Apache-2.0
"""The reworked sparse MLA prefill path against the 8aa22fa kernel, in the simulator.

Two precision modes serve the production geometry (one head, latent 512, the selector's
2176-column index rows, a paged bf16 window):

* ``VLLM_NEURON_MLA_SPARSE_FP32=1`` (the kill switch) keeps 8aa22fa's fp32 arithmetic.
  Its output must be *bitwise* 8aa22fa's: the structural changes (the sentinel mask on the
  head partitions, the rotating per-query buffer sets) move no arithmetic.
* The default feeds the PE 2-byte operands: MM1 on the stored bf16 query and cache (exact
  products, fp32 accumulation), MM2 on a bf16 hi/lo split of the fp32 probabilities (about
  16 significand bits). Its output is read against 8aa22fa within :data:`LOWP_REL_L2`.

Shapes: token counts {1, 64, 1024} x contexts {1024, 2051, 8192}. The 1024-token cases run
the full chunk through the simulator and take about a minute each per kernel. A one-token
call is served by the body's single-query (DMA-widened fp32) path in both modes.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

from vllm_neuron.functional.attention import mla_sparse as MS

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "test" / "hardware"))

from baselines.mla_sparse_8aa22fa import load as load_baseline  # noqa: E402


def _benchmark_module():
    """The slice benchmark, for its input builders (the same operands the device times)."""
    name = "benchmark_mla_sparse_prefill_inputs"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = ROOT / "test" / "hardware" / "benchmark_mla_sparse_prefill.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCH = _benchmark_module()
SCALE = BENCH.SCALE
PAGE = BENCH.PAGE

#: Low-precision path against 8aa22fa (fp32 outputs, relative L2 over the whole output).
#: Measured in the simulator at every case below: at most 6.2e-6 (bf16 hi/lo split of p,
#: 16 significand bits); the bound leaves 3x room.
LOWP_REL_L2 = 2e-5
#: The same, element-wise: |after - before| <= LOWP_ATOL + LOWP_RTOL * |before|.
LOWP_RTOL = 2e-4
LOWP_ATOL = 1e-5
#: Both kernels against the float64 reference, relative L2.
ORACLE_REL_L2 = 3e-5

TOKENS = (1, 64, 1024)
CONTEXTS = (1024, 2051, 8192)

_BASELINE_OUTPUTS: dict[tuple[int, int], torch.Tensor] = {}


def _reference(q, window, indices):
    """float64 reference, independent of both kernels' oracle."""
    qd = q.double()
    cd = window.double()
    rows = indices.to(torch.int64)
    keep = rows >= 0
    out = torch.empty(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float64)
    for s in range(q.shape[0]):
        gathered = cd[rows[s].clamp(min=0)]
        scores = torch.einsum("hl,kl->hk", qd[s], gathered) * SCALE
        scores = scores.masked_fill(~keep[s], float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1))
        out[s] = weights @ gathered
    return out


def _rel_l2(got, want) -> float:
    got, want = got.double(), want.double()
    return float((got - want).norm() / want.norm().clamp_min(1e-30))


def _call(module, q, bank, indices, table, written, offset):
    return module.mla_sparse_attention(q, bank, indices, SCALE, block_table_row=table,
                                       written=written, write_offset=offset, page_size=PAGE)


def _case(tokens: int, ctx: int):
    return BENCH.make_case(tokens, ctx, seed=1000 * tokens + ctx)


def _baseline(tokens: int, ctx: int, operands):
    key = (tokens, ctx)
    if key not in _BASELINE_OUTPUTS:
        q, bank, indices, table, written, offset, _ = operands
        _BASELINE_OUTPUTS[key] = _call(load_baseline(), q, bank, indices, table, written, offset)
    return _BASELINE_OUTPUTS[key]


@pytest.mark.parametrize("ctx", CONTEXTS)
@pytest.mark.parametrize("tokens", TOKENS)
def test_fp32_kill_switch_is_bitwise_8aa22fa(monkeypatch, tokens, ctx):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    monkeypatch.setenv("VLLM_NEURON_MLA_SPARSE_FP32", "1")
    operands = _case(tokens, ctx)
    q, bank, indices, table, written, offset, window = operands
    before = _baseline(tokens, ctx, operands)
    got = _call(MS, q, bank, indices, table, written, offset)
    assert got.dtype == torch.float32 and got.shape == q.shape
    torch.testing.assert_close(got, before, rtol=0.0, atol=0.0)
    assert _rel_l2(got, _reference(q, window, indices)) <= ORACLE_REL_L2


@pytest.mark.parametrize("ctx", CONTEXTS)
@pytest.mark.parametrize("tokens", TOKENS)
def test_lowp_default_within_tolerance_of_8aa22fa(monkeypatch, tokens, ctx):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    monkeypatch.delenv("VLLM_NEURON_MLA_SPARSE_FP32", raising=False)
    operands = _case(tokens, ctx)
    q, bank, indices, table, written, offset, window = operands
    before = _baseline(tokens, ctx, operands)
    got = _call(MS, q, bank, indices, table, written, offset)
    assert got.dtype == torch.float32 and got.shape == q.shape
    assert torch.isfinite(got).all()
    rel = _rel_l2(got, before)
    assert rel <= LOWP_REL_L2, rel
    torch.testing.assert_close(got, before, rtol=LOWP_RTOL, atol=LOWP_ATOL)
    want = _reference(q, window, indices)
    assert _rel_l2(got, want) <= ORACLE_REL_L2
    assert _rel_l2(before, want) <= ORACLE_REL_L2
    # The low-precision path is a different arithmetic, not a relabelled fp32 one. It is
    # a multi-query path: a one-query call keeps the DMA-widened fp32 operands (the body
    # takes ``native_kv`` only at ``seq > 1``), so there it is 8aa22fa bit for bit.
    if tokens > 1:
        assert not torch.equal(got, before)
    else:
        torch.testing.assert_close(got, before, rtol=0.0, atol=0.0)


def test_lowp_sentinel_edges_match_8aa22fa_and_zero_an_empty_row(monkeypatch):
    """Whole-sentinel rows, a sentinel first tile, repeated rows and a 128-wide ragged tile."""
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    monkeypatch.delenv("VLLM_NEURON_MLA_SPARSE_FP32", raising=False)
    gen = torch.Generator().manual_seed(17)
    seq, rows, width = 5, 2600, 2176
    q = (torch.randn(seq, 1, 512, generator=gen) * 0.5).to(torch.bfloat16)
    cache = torch.randn(rows, 512, generator=gen).to(torch.bfloat16)
    indices = torch.randint(rows, (seq, width), generator=gen, dtype=torch.int32)
    indices[0] = -1                     # attends nothing: exact zeros
    indices[1, :512] = -1               # a wholly-sentinel first tile initialises the merge
    indices[2] = -1
    indices[2, :1024] = rows - 1        # one row repeated across two tiles, nothing else
    indices[3, 2048:] = -1              # the ragged 128-wide last tile is all sentinel
    indices[4, 1::2] = -1               # alternating sentinels inside every chunk
    got = MS.mla_sparse_attention(q, cache, indices, SCALE)
    before = load_baseline().mla_sparse_attention(q, cache, indices, SCALE)
    assert torch.count_nonzero(got[0]) == 0
    assert torch.isfinite(got).all()
    assert _rel_l2(got[1:], before[1:]) <= LOWP_REL_L2
    torch.testing.assert_close(got, before, rtol=LOWP_RTOL, atol=LOWP_ATOL)
    torch.testing.assert_close(got[2, 0], cache[rows - 1].float(), rtol=1e-3, atol=1e-5)


def test_one_buffer_set_is_bitwise_two(monkeypatch):
    """The rotation is scheduling only: one set and two sets give identical bits, both modes."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    operands = _case(48, 2051)
    q, bank, indices, table, written, offset, _ = operands
    for mode in ("1", None):
        if mode is None:
            monkeypatch.delenv("VLLM_NEURON_MLA_SPARSE_FP32", raising=False)
        else:
            monkeypatch.setenv("VLLM_NEURON_MLA_SPARSE_FP32", mode)
        monkeypatch.setenv("VLLM_NEURON_MLA_SPARSE_QUERY_BUFFERS", "1")
        one = _call(MS, q, bank, indices, table, written, offset)
        monkeypatch.delenv("VLLM_NEURON_MLA_SPARSE_QUERY_BUFFERS")
        two = _call(MS, q, bank, indices, table, written, offset)
        torch.testing.assert_close(one, two, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "fp32_env,buffers_env,programs_env,lnc,expect_lowp,expect_buffers,expect_grid",
    [
        (None, None, None, "2", True, 2, 2),
        ("1", None, None, "2", False, 2, 2),
        ("0", None, None, "2", True, 2, 2),
        (None, "1", None, "2", True, 1, 2),
        (None, None, "1", "2", True, 2, 1),
        (None, None, None, None, True, 2, 1),
    ],
)
def test_seam_passes_precision_buffers_digest_and_grid(
    monkeypatch, fp32_env, buffers_env, programs_env, lnc, expect_lowp, expect_buffers, expect_grid
):
    for key, value in (("VLLM_NEURON_MLA_SPARSE_FP32", fp32_env),
                       ("VLLM_NEURON_MLA_SPARSE_QUERY_BUFFERS", buffers_env),
                       ("VLLM_NEURON_MLA_SPARSE_PROGRAMS", programs_env),
                       ("NEURON_LOGICAL_NC_CONFIG", lnc)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    calls = []

    class RecordingCall:
        def __init__(self, entry):
            self.entry, self.grid = entry, 1

        def __getitem__(self, requested):
            self.grid = requested
            return self

        def __call__(self, *args):
            calls.append((self.entry, self.grid, args))
            return torch.zeros(args[0].shape, dtype=torch.float32)

    monkeypatch.setattr(MS, "wrap_nki", RecordingCall)
    q = torch.zeros(32, 1, 512, dtype=torch.bfloat16)
    bank = torch.zeros(2048, 512, dtype=torch.bfloat16)
    selected = torch.zeros(32, 2176, dtype=torch.int32)
    MS.mla_sparse_attention(q, bank, selected, SCALE,
                            block_table_row=torch.arange(16).int().reshape(16, 1),
                            written=torch.zeros(32, 512, dtype=torch.bfloat16),
                            write_offset=torch.tensor([[2016]], dtype=torch.int32),
                            page_size=128)
    assert len(calls) == 1
    entry, grid, args = calls[0]
    assert entry is MS.mla_sparse_attention_nope_row_tiled_kernel
    assert grid == expect_grid
    # (q, c_kv, topk, scale, table, written, offset, page, BLOCK_N, STREAM_KV, PE_LOWP,
    #  QUERY_BUFFERS, source_digest)
    assert len(args) == 13
    assert args[8] == MS.MOVING_MAX and args[9] is True
    assert args[10] is expect_lowp
    assert args[11] == expect_buffers
    assert args[12] == MS.SOURCE_DIGEST


def test_source_digest_keys_on_this_file():
    import hashlib

    digest = int(hashlib.sha256(pathlib.Path(MS.__file__).read_bytes()).hexdigest()[:7], 16)
    assert MS.SOURCE_DIGEST == digest
    assert load_baseline().mla_sparse_attention_torch_oracle is not MS.mla_sparse_attention_torch_oracle
