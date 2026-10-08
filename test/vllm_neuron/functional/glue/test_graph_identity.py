# SPDX-License-Identifier: Apache-2.0
"""Under ``1`` a decode graph is ``0``'s graph, and a 128-row prefill is mhc_pre + mhc_post's.

The compile-cache key (``libtorch_neuronx_lite.compile.cache.create_cache_hash``) is the
traced FX graph's text plus the inputs' shapes, dtypes and strides, the tool versions and
the compiler arguments. Only the graph text depends on ``VLLM_NEURON_GLUE_FUSED``, so two
values whose traced graphs print the same compile to the same key, and the second is a
cache hit. This test traces one decoder layer's step (the model's own layer forward and
FFN site, ``glue_case.layer_step``, at one rank's TP=64 / EP=16 shapes) with dynamo under
several values and compares the graphs. It stops at the backend, so nothing is compiled.
(In CPU mode the tracer still runs each NKI kernel in the simulator once, on ones-filled
inputs, to get its output shapes; that probe can warn of a float overflow, which does
not reach the graph.)

* Decode, B in {1, 64}, both layer families: ``1`` traces ``0``'s graph, so the default
  cannot change a decode step.
* Prefill, 128 rows: ``1`` traces exactly ``mhc_pre,mhc_post``'s graph, neither ``0``'s
  nor ``all``'s.

The 1024-row prefill is not traced here: one trace at 1024 rows takes about 300 s on the
CPU. At 1024 rows only mhc_post can serve (the other kernels' row bounds), so ``1`` and
``all`` select the same kernels there. ``test_layer_prefill.py`` shows both outputs
bit-identical to ``0``'s, and ``test/hardware/benchmark_glue_block.py`` records the
device compile keys (``same_graph_as``).
"""

from __future__ import annotations

import types

import pytest
import torch

from test.hardware import benchmark_layer_glue as micro
from test.vllm_neuron.functional.glue import glue_case
from vllm_neuron.functional import glue
from vllm_neuron.model.glm5_next import model_fp8 as live


class _Traced(Exception):
    """Raised by the recording backend once dynamo hands it the graph."""


@pytest.fixture(scope="module")
def layers():
    torch.manual_seed(0)
    return {"kda": glue_case.kda_layer(live, seed=glue_case.KDA_LAYER),
            "dsa": glue_case.dsa_layer(live, seed=glue_case.DSA_LAYER)}


def _carriers(family: str, phase: str, rows: int, case) -> dict:
    if phase == "prefill":
        return glue_case.kda_prefill_carriers(case, rows)
    if family == "kda":
        return glue_case.kda_carriers(case, rows)
    return glue_case.dsa_carriers(case, rows)


def _graph(case, family: str, phase: str, rows: int, value: str, monkeypatch) -> str:
    """The FX graph dynamo traces for one layer step with the switch at ``value``."""
    quant = glue_case.quant_config(live)
    names, tensors, statics = micro._flatten(_carriers(family, phase, rows, case))
    rank = torch.tensor(0, dtype=torch.int64)
    seen = []

    def backend(gm, example_inputs):
        seen.append(str(gm.graph))
        raise _Traced

    def step(streams, rank, *flat):
        carriers = micro._unflatten(names, flat, statics)
        return glue_case.layer_step(live, case, streams, carriers, rank, quant)

    # A code object of its own per trace, so no dynamo cache entry is shared.
    tag = f"step_{family}_{phase}_{rows}_{abs(hash(value))}"
    code = step.__code__.replace(co_name=tag)
    fresh = types.FunctionType(code, step.__globals__, tag, step.__defaults__,
                               step.__closure__)
    compiled = torch.compile(fresh, backend=backend, fullgraph=True, dynamic=False)
    streams = glue_case.streams_input(case.cfg, rows)
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, value)
    torch._dynamo.reset()
    try:
        compiled(streams, rank, *tensors)
    except Exception as stop:  # dynamo wraps the backend's exception
        if not isinstance(stop, _Traced) and "_Traced" not in repr(stop):
            raise
    finally:
        torch._dynamo.reset()
    assert len(seen) == 1, f"{family} {phase} {rows} under {value!r}: {len(seen)} graphs"
    return seen[0]


@pytest.mark.parametrize(("family", "rows"), (("kda", 1), ("kda", 64), ("dsa", 1),
                                              ("dsa", 64)))
def test_decode_under_one_is_the_zero_graph(layers, family, rows, monkeypatch):
    zero = _graph(layers[family], family, "decode", rows, "0", monkeypatch)
    assert _graph(layers[family], family, "decode", rows, "1", monkeypatch) == zero
    if rows == 1:
        # Not vacuous: the trace shows a fused kernel where one is selected.
        assert _graph(layers[family], family, "decode", rows, "all", monkeypatch) != zero


def test_the_128_row_prefill_under_one_is_mhc_pre_and_mhc_post(layers, monkeypatch):
    graphs = {v: _graph(layers["kda"], "kda", "prefill", 128, v, monkeypatch)
              for v in ("0", "1", "all", "mhc_pre,mhc_post")}
    assert graphs["1"] == graphs["mhc_pre,mhc_post"]
    assert graphs["1"] != graphs["0"]
    assert graphs["1"] != graphs["all"]
