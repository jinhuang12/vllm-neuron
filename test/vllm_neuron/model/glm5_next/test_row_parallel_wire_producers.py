# SPDX-License-Identifier: Apache-2.0
"""Each row-parallel partial's producer stores it in the dtype its reduction rounds to.

``_row_parallel_wire_dtype`` names the dtype ``_row_parallel_sum`` rounds a partial to
before it crosses the tensor-parallel wire, or None where the partial crosses no wire
(world size one) or crosses it in fp32. Three sites hand that answer to their producer:
the KDA output (``Glm5NextKDAAttention._gated_output``), the MLA output projection
(``Glm5NextMLAAttention.project_output``) and the dense MLP at the feed-forward site
(``Glm5NextModel._ffn_half``). On a prefill chunk's default bf16 wire the producer is
asked for bf16 and rounds once, in its store. Everywhere else it is asked for None and
its fp32 partial reaches the reduction untouched, as before the bf16 store existed.

The producers are recorded and the reduction is the identity: these tests read what each
site asks for. The producers' own bf16 stores are tested beside their kernels
(``test_kda_output.py``, ``test_mla_projection_lowp.py``,
``test_blockwise_fp8_mlp_bf16_store.py``).
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from test.vllm_neuron.functional.dsa import dsa_decode_case
from test.vllm_neuron.functional.glue import glue_case
from test.vllm_neuron.model.glm5_next import test_ffn_reduction as ffn
from vllm_neuron.functional.attention import mla_projections
from vllm_neuron.functional.glue import kda_output
from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.glm5_next.collective_policy import StepRows

#: The module, not the function of the same name that ``vllm_neuron.functional`` exports.
blockwise_fp8_mm = importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")

#: A served-like step rule: decode batches up to 64 rows, and a 128-row verify step
#: (64 requests x (1 + 1 draft)).
STEPS = StepRows(64, frozenset({128}))
PREFILL, DECODE, VERIFY = 1024, 4, 128
#: A stand-in TP group. Only its presence is read before the sum, and the sum is stubbed.
GROUP = object()

#: (rows, TP group, VLLM_NEURON_TP_ALLREDUCE_DTYPE, the dtype the producer is asked for)
CASES = [
    pytest.param(PREFILL, GROUP, None, torch.bfloat16, id="prefill-bf16-wire"),
    pytest.param(DECODE, GROUP, None, None, id="decode"),
    pytest.param(VERIFY, GROUP, None, None, id="verify"),
    pytest.param(PREFILL, GROUP, "fp32", None, id="prefill-fp32-wire"),
    pytest.param(PREFILL, None, None, None, id="world-1"),
]


@pytest.fixture
def wire(monkeypatch):
    """Set the TP group and the wire switch; the row-parallel sum returns its partial."""

    def set_wire(group, dtype):
        monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: group)
        monkeypatch.setattr(model_fp8, "_row_parallel_sum",
                            lambda partial, *, site, module: partial)
        if dtype is None:
            monkeypatch.delenv("VLLM_NEURON_TP_ALLREDUCE_DTYPE", raising=False)
        else:
            monkeypatch.setenv("VLLM_NEURON_TP_ALLREDUCE_DTYPE", dtype)

    return set_wire


def _recorder(requests, rows_of, width):
    """A producer stand-in: records ``out_dtype``, returns zeros in it (fp32 for None)."""

    def produce(*args, out_dtype=None, **kwargs):
        requests.append(out_dtype)
        return torch.zeros(rows_of(*args), width, dtype=out_dtype or torch.float32)

    return produce


@pytest.mark.parametrize("rows, group, dtype, expected", CASES)
def test_the_wire_dtype_is_bf16_only_on_a_bf16_wire(rows, group, dtype, expected, wire):
    wire(group, dtype)
    assert model_fp8._row_parallel_wire_dtype(rows, module=SimpleNamespace(
        step_rows=STEPS)) is expected


@pytest.mark.parametrize("rows, group, dtype, expected", CASES)
def test_the_kda_output_asks_its_producer_for_the_wire_dtype(rows, group, dtype, expected,
                                                             wire, monkeypatch):
    wire(group, dtype)
    cfg = glue_case.text_config()
    hidden = int(cfg.hidden_size)
    requests = []
    monkeypatch.setattr(kda_output, "kda_gated_projection",
                        _recorder(requests, lambda core, *_: int(core.shape[0]), hidden))
    attn = model_fp8.Glm5NextKDAAttention(cfg, glue_case.TP_WORLD)
    attn.step_rows = STEPS
    core = torch.zeros(rows, int(attn.num_kv_heads_per_rank) * int(attn.head_dim))
    out = attn._gated_output(core, torch.zeros_like(core),
                             torch.zeros(rows, hidden, dtype=torch.bfloat16),
                             phase="prefill" if rows == PREFILL else "decode")
    assert requests == [expected]
    assert out.dtype == torch.bfloat16


@pytest.fixture(scope="module")
def mla():
    return dsa_decode_case.build_attention(model_fp8, dsa_decode_case.decode_config(), seed=5)


@pytest.mark.parametrize("collect", [False, True], ids=["forward", "collector"])
@pytest.mark.parametrize("rows, group, dtype, expected", CASES)
def test_the_mla_output_projection_asks_for_the_wire_dtype(rows, group, dtype, expected,
                                                           collect, mla, wire, monkeypatch):
    """A collecting forward (the layer-stream dump) reads the fp32 partial: None always."""
    wire(group, dtype)
    hidden = int(mla.hidden_size)
    requests = []
    monkeypatch.setattr(mla_projections, "mla_projection_prepared",
                        _recorder(requests, lambda x, *_: int(x.shape[0]), hidden))
    monkeypatch.setattr(mla, "step_rows", STEPS)
    attn_out = torch.zeros(rows, mla._heads_per_rank() * int(mla.v_head_dim),
                           dtype=torch.bfloat16)
    collector = [] if collect else None
    out = mla.project_output(attn_out, collector=collector)
    assert requests == [None if collect else expected]
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (rows, hidden)


#: (the layer's step rule, the TP group, the wire switch, the dtype asked for) at the
#: harness's 128-row call: a prefill chunk past 64-row decode batches, a decode batch
#: within 128-row ones, and a verify step of 128.
FFN_CASES = [
    pytest.param(StepRows(64), GROUP, None, torch.bfloat16, id="prefill-bf16-wire"),
    pytest.param(StepRows(128), GROUP, None, None, id="decode"),
    pytest.param(STEPS, GROUP, None, None, id="verify"),
    pytest.param(StepRows(64), GROUP, "fp32", None, id="prefill-fp32-wire"),
    pytest.param(StepRows(64), None, None, None, id="world-1"),
]


@pytest.mark.parametrize("steps, group, dtype, expected", FFN_CASES)
def test_the_dense_mlp_asks_its_kernel_for_the_wire_dtype(steps, group, dtype, expected,
                                                          wire, monkeypatch):
    wire(group, dtype)
    requests = []
    monkeypatch.setattr(blockwise_fp8_mm, "blockwise_fp8_mlp",
                        _recorder(requests, lambda x, *_: int(x.shape[0]), ffn.HIDDEN_SIZE))
    text_config = ffn._text_config()
    operands = ffn._whole_operands()
    layer = ffn._layer(ffn._dense_module(text_config, operands), operands["gain"])
    layer.step_rows = steps
    out = ffn._run_half(ffn._dependencies(text_config), layer, operands["hidden"])
    assert requests == [expected]
    assert tuple(out.shape) == (ffn.TOKENS, ffn.HIDDEN_SIZE)
