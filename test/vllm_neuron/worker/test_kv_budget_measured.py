# SPDX-License-Identifier: Apache-2.0
"""The KV budget is what the device has free, less what the compiled graphs need.

    budget = free HBM at budget time - graph need - margin

* free HBM: ``Runtime.get_vnc_memory_stats`` on the logical core, after the
  weights and prepared operands are resident and before the KV cache exists;
* graph need: read from the NEFFs the compile cache holds for this configuration
  (:mod:`vllm_neuron.vllm.worker.neff_memory`), or the 5 GiB reserve when the cache
  holds none of them yet, or ``VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB`` when set;
* margin: :data:`~vllm_neuron.vllm.worker.neuron_worker.KV_BUDGET_MARGIN_BYTES`.

``VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION`` caps the budget only when set. There is
no per-physical-core bound: the runtime accounts every tensor of a rank on the
even physical core (14.46 GiB on ND 0 NC 0 in the recorded bs=64 @ 8k serve run, more than half the
logical core's 24 GiB), so the logical core is the only capacity there is.

The served figures are the bs=64 @ 8k line as measured
server log (every rank): "Neuron HBM: 7.06 GiB used, 16.94 GiB free", parameters
1.63 GiB, resident 6.79 GiB, need 5.845 GiB, allocated 6.329 GiB (side caches
0.347 GiB).

A warm compile cache is built in a temporary directory from the recorded decode
graph of that line (``fixtures/glm53f_bs64x8k_decode_b1_ctx2048.neff``, see
``test_neff_memory.py``). One test reads the line's real 15-graph cache; it runs
only when the test-only variable ``VLLM_NEURON_TEST_COMPILE_CACHE_ROOT`` names the
recorded run's compile cache (its ``neuron/compile_cache`` directory) and is skipped when
it is unset. It is a test knob, not a serving knob, so it is not registered in
``vllm_neuron/envs.py``.

Run with ``VLLM_NEURON_CPU_MODE=1`` (``test/conftest.py`` pins it); the tests clear
it where they drive the device path.
"""

from __future__ import annotations

import logging

import pytest

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv
from test.vllm_neuron.worker.test_neff_memory import (
    _cache_entry,
    _fixture_entry,
    compile_cache_root,
    needs_compile_cache,
)
from vllm_neuron.vllm.worker import neff_memory

GIB = 1024**3
MIB = 1024**2

TIP_USED_BYTES = int(7.06 * GIB)
TIP_FREE_BYTES = kv.TOTAL_HBM_BYTES - TIP_USED_BYTES
TIP_PARAM_BYTES = kv.TIP_PARAM_BYTES
TIP_RESIDENT_BYTES = kv.TIP_RESIDENT_BYTES
#: 64 x 8192 on the served line (test_kv_budget_side_caches.py's hand total).
TIP_FOOTPRINT_BYTES = 6795902976
TIP_NEED_BYTES = 6276120576
#: The two figures the old budget took in the recorded run: 0.30 x 0.92 x 24 GiB and
#: 24 / 2 - 5 GiB.
OLD_CAP_BYTES = int(int(kv.TOTAL_HBM_BYTES * kv.GPU_MEMORY_UTILIZATION) * 0.30)
OLD_CORE_BOUND_BYTES = kv.TOTAL_HBM_BYTES // 2 - 5 * GIB

#: The bs=64 line warms 15 graphs: 1 prefill bucket x 1 segment, 7 batch x 2 ctx decode.
TIP_B64_GRAPHS = 15

KNOBS = (
    "VLLM_NEURON_CPU_MODE",
    "VLLM_NEURON_CPU_COMPILE",
    "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB",
    "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION",
)


@pytest.fixture(autouse=True)
def _device_path(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def model_specs():
    return kv.glm53f_layer_specs()


def _tip_worker(model_specs, *, seqs=64, length=8192, graph_cache_dir=None, expected_graphs=TIP_B64_GRAPHS):
    layers, text_config = model_specs
    runner = kv.fake_runner(
        layers, max_num_seqs=seqs, max_model_len=length, text_config=text_config
    )
    return kv.fake_worker(
        runner,
        param_bytes=TIP_PARAM_BYTES,
        resident_bytes=TIP_RESIDENT_BYTES,
        runtime_used_bytes=TIP_USED_BYTES,
        graph_cache_dir=graph_cache_dir,
        expected_graphs=expected_graphs,
    )


def _graph(worker, need_bytes: int, source: str = "fixture"):
    """Pin the graph need the worker reads, as a warm cache would report it."""
    from vllm_neuron.vllm.worker.neuron_worker import GraphNeed

    worker._graph_need = lambda *_args, **_kw: GraphNeed(need_bytes, source)
    return worker


def _margin() -> int:
    from vllm_neuron.vllm.worker.neuron_worker import KV_BUDGET_MARGIN_BYTES

    return KV_BUDGET_MARGIN_BYTES


def test_the_budget_is_free_memory_less_graph_need_less_margin(model_specs) -> None:
    """In the recorded run: 16.94 GiB free, a 1.31 GiB graph need, the margin: 15.4 GiB."""
    graph_bytes = int(1.31 * GIB)
    worker = _graph(_tip_worker(model_specs), graph_bytes)

    budget = worker._determine_available_memory_neuron(kv.GPU_MEMORY_UTILIZATION)

    assert budget.available_bytes == TIP_FREE_BYTES - graph_bytes - _margin()
    assert budget.free_bytes == TIP_FREE_BYTES
    assert budget.free_source == "runtime"
    assert round(budget.available_bytes / GIB, 2) == 15.38
    # Neither the 0.30 cap nor the per-physical-core bound takes part by default.
    assert budget.available_bytes > max(OLD_CAP_BYTES, OLD_CORE_BOUND_BYTES)
    assert budget.cap_bytes is None


def test_the_served_point_is_admitted_and_the_need_returned(model_specs) -> None:
    worker = _graph(_tip_worker(model_specs), int(1.31 * GIB))

    assert worker.determine_available_memory() == TIP_NEED_BYTES


def test_measured_is_the_default_spelling_of_both_knobs(model_specs, monkeypatch) -> None:
    unset = _graph(_tip_worker(model_specs), int(1.31 * GIB))
    expected = unset._determine_available_memory_neuron(kv.GPU_MEMORY_UTILIZATION)

    monkeypatch.setenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "measured")
    monkeypatch.setenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", "Measured")
    spelled = _graph(_tip_worker(model_specs), int(1.31 * GIB))

    assert spelled._determine_available_memory_neuron(kv.GPU_MEMORY_UTILIZATION) == expected


def test_the_cap_fraction_caps_only_when_set(model_specs, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "0.30")
    worker = _graph(_tip_worker(model_specs), int(1.31 * GIB))

    budget = worker._determine_available_memory_neuron(kv.GPU_MEMORY_UTILIZATION)

    assert budget.cap_bytes == OLD_CAP_BYTES
    assert budget.available_bytes == OLD_CAP_BYTES


def test_the_graph_reserve_override_replaces_the_measured_need(
    model_specs, monkeypatch, tmp_path
) -> None:
    """A warm cache is present (its one graph supplies the need); the override wins."""
    _fixture_entry(tmp_path, "decode")
    worker = _tip_worker(model_specs, graph_cache_dir=tmp_path, expected_graphs=1)
    assert "1 compiled graphs" in worker._graph_need(worker._kv_cache_need_bytes()).source

    monkeypatch.setenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", "2.5")
    budget = worker._determine_available_memory_neuron(kv.GPU_MEMORY_UTILIZATION)

    assert budget.graph.bytes == int(2.5 * GIB)
    assert "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB" in budget.graph.source
    assert budget.available_bytes == TIP_FREE_BYTES - int(2.5 * GIB) - _margin()


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", "0"),
        ("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", "-1"),
        ("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", "inf"),
        ("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", "lots"),
        ("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "0"),
        ("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "1.5"),
        ("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "most"),
    ],
)
def test_an_unusable_override_is_refused_by_name(model_specs, monkeypatch, knob, value) -> None:
    monkeypatch.setenv(knob, value)
    worker = _tip_worker(model_specs)

    with pytest.raises(RuntimeError) as refusal:
        worker._determine_available_memory_neuron(kv.GPU_MEMORY_UTILIZATION)

    assert knob in str(refusal.value)


def test_a_cold_cache_assumes_the_reserve_and_says_so(model_specs, tmp_path) -> None:
    worker = _tip_worker(model_specs, graph_cache_dir=tmp_path)

    graph = worker._graph_need(worker._kv_cache_need_bytes())

    assert graph.bytes == 5 * GIB
    assert "cold" in graph.source
    assert "0 of the 15" in graph.source


def test_a_partly_compiled_cache_assumes_the_reserve(model_specs, tmp_path) -> None:
    """Graphs of this configuration in the cache, but fewer than warmup will load."""
    worker = _tip_worker(model_specs, graph_cache_dir=tmp_path)
    pool_bytes = worker._kv_cache_largest_tensor_bytes(worker._kv_cache_need_bytes())
    for key in ("aaa", "bbb"):
        _cache_entry(tmp_path, key, kv_bytes=pool_bytes)

    graph = worker._graph_need(worker._kv_cache_need_bytes())

    assert graph.bytes == 5 * GIB
    assert "2 of the 15" in graph.source


def test_a_warm_cache_supplies_the_need_from_its_graphs(model_specs, tmp_path) -> None:
    worker = _tip_worker(model_specs, graph_cache_dir=tmp_path, expected_graphs=2)
    pool_bytes = worker._kv_cache_largest_tensor_bytes(worker._kv_cache_need_bytes())
    for key in ("aaa", "bbb"):
        _cache_entry(tmp_path, key, kv_bytes=pool_bytes)
    _cache_entry(tmp_path, "ccc", kv_bytes=pool_bytes + 1)  # another configuration
    scan = neff_memory.scan_compile_cache(tmp_path, kv_input_bytes=pool_bytes)

    graph = worker._graph_need(worker._kv_cache_need_bytes())

    expected = neff_memory.graph_memory(
        scan.graphs, page_bytes=neff_memory.scratchpad_page_bytes()
    )
    assert graph.bytes == expected.total_bytes
    assert "2 compiled graphs" in graph.source


def test_an_unreadable_cache_assumes_the_reserve(model_specs, tmp_path, monkeypatch) -> None:
    """A cache the worker cannot list costs the reserve, not the server."""

    def denied(*_args, **_kw):
        raise PermissionError(13, "Permission denied", str(tmp_path))

    monkeypatch.setattr(neff_memory, "scan_compile_cache", denied)
    worker = _tip_worker(model_specs, graph_cache_dir=tmp_path)

    graph = worker._graph_need(worker._kv_cache_need_bytes())

    assert graph.bytes == 5 * GIB
    assert "unreadable" in graph.source


def test_the_kv_tensor_that_ties_graphs_is_the_latent_pool_layer(model_specs) -> None:
    """4353 blocks x 131072 B at 64 x 8192: the (557184, 1, 512) bf16 graph input."""
    worker = _tip_worker(model_specs)

    assert worker._kv_cache_largest_tensor_bytes(TIP_NEED_BYTES) == 4353 * 131072


@needs_compile_cache
def test_the_bs64_line_cache_gives_the_measured_need(model_specs) -> None:
    """The 15 graphs of the warm bs=64 cache: about 1.3 GiB, not 5 GiB."""
    worker = _tip_worker(model_specs, graph_cache_dir=compile_cache_root())

    budget = worker._determine_available_memory_neuron(kv.GPU_MEMORY_UTILIZATION)

    assert "15 compiled graphs" in budget.graph.source
    assert 1.25 * GIB < budget.graph.bytes < 1.40 * GIB
    assert budget.available_bytes == TIP_FREE_BYTES - budget.graph.bytes - _margin()
    assert worker.determine_available_memory() == TIP_NEED_BYTES


def test_the_budget_log_prints_every_term(model_specs, caplog) -> None:
    worker = _graph(_tip_worker(model_specs), int(1.31 * GIB), source="15 compiled graphs in X")

    with caplog.at_level(logging.INFO):
        worker.determine_available_memory()

    line = next(r.getMessage() for r in caplog.records if "KV cache budget in neuron mode" in r.getMessage())
    for term in (
        "free=16.94 GiB (runtime)",
        "graph need=1.31 GiB (15 compiled graphs in X)",
        f"margin={_margin() / GIB:.2f} GiB",
        "budget=15.38 GiB",
        "need=5.845 GiB",
        "side caches 0.347 GiB",
        "allocated=6.329 GiB",
        "admitted=yes",
    ):
        assert term in line, (term, line)


def test_a_point_over_the_budget_is_refused_naming_every_term(model_specs) -> None:
    """64 x 8192 against a 12 GiB graph need: the 6.329 GiB footprint does not fit."""
    graph_bytes = 12 * GIB
    worker = _graph(_tip_worker(model_specs), graph_bytes, source="fixture source")
    budget = TIP_FREE_BYTES - graph_bytes - _margin()

    with pytest.raises(RuntimeError) as refusal:
        worker.determine_available_memory()

    message = str(refusal.value)
    for term in (
        "free 16.94 GiB",
        "graph need 12.00 GiB (fixture source)",
        f"margin {_margin() / GIB:.2f} GiB",
        f"budget {budget / GIB:.3f} GiB",
        f"{TIP_FOOTPRINT_BYTES / GIB:.3f} GiB",
        f"short by {(TIP_FOOTPRINT_BYTES - budget) / GIB:.3f} GiB",
        "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB",
    ):
        assert term in message, (term, message)


def test_the_compile_estimate_charges_the_resident_operands(model_specs, monkeypatch) -> None:
    """Without a device the free memory is HBM less parameters and prepared operands."""
    monkeypatch.setenv("VLLM_NEURON_CPU_COMPILE", "1")
    worker = _graph(_tip_worker(model_specs), int(1.31 * GIB))

    budget = worker._estimate_available_memory_neuron(kv.GPU_MEMORY_UTILIZATION)

    assert budget.free_bytes == kv.TOTAL_HBM_BYTES - TIP_RESIDENT_BYTES
    assert budget.free_source == "estimate"
    assert budget.available_bytes == budget.free_bytes - int(1.31 * GIB) - _margin()


# ---------------------------------------------------------------------------
# Derived defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "gib"), [("trn2", 24), ("trn2n", 24), ("trn3", 36), ("trn1", 16)]
)
def test_the_platform_reports_one_logical_cores_hbm(monkeypatch, target, gib) -> None:
    """vLLM reads the device memory to pick batch defaults; Neuron reported none."""
    from vllm_neuron.vllm.platform import NeuronPlatform

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", target)

    assert NeuronPlatform.get_device_total_memory() == gib * GIB


def test_without_an_override_the_instance_family_names_the_target(monkeypatch) -> None:
    from vllm_neuron.utils import hardware_config
    from vllm_neuron.vllm.platform import NeuronPlatform

    monkeypatch.delenv("NEURON_PLATFORM_TARGET_OVERRIDE", raising=False)
    monkeypatch.setattr(hardware_config, "get_instance_family", lambda: "trn2")

    assert NeuronPlatform.get_device_total_memory() == 24 * GIB


def test_an_unknown_target_is_refused_by_name(monkeypatch) -> None:
    from vllm_neuron.vllm.platform import NeuronPlatform

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trnX")

    with pytest.raises(RuntimeError) as refusal:
        NeuronPlatform.get_device_total_memory()

    assert "trnX" in str(refusal.value)


def test_upstream_still_defaults_max_num_seqs_to_256_on_24_gib(monkeypatch) -> None:
    """vLLM's batch defaults step at 70 GiB, so 24 GiB and 0 give the same 256."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext

    from vllm_neuron.vllm.platform import NeuronPlatform

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setattr("vllm.engine.arg_utils.current_platform", NeuronPlatform())

    _, max_num_seqs = EngineArgs.get_batch_defaults(world_size=64)

    assert NeuronPlatform.get_device_total_memory() == 24 * GIB
    assert max_num_seqs[UsageContext.OPENAI_API_SERVER] == 256


def test_the_largest_max_num_seqs_that_fits_is_derived_from_the_budget(model_specs) -> None:
    """At 8192 tokens and the recorded run's budget: the S whose footprint fits, and not S + 1."""
    worker = _graph(_tip_worker(model_specs, seqs=256), int(1.31 * GIB))
    budget = worker._determine_available_memory_neuron(kv.GPU_MEMORY_UTILIZATION)

    fit = worker._max_num_seqs_that_fit(budget.available_bytes)

    assert _tip_worker(model_specs, seqs=fit)._kv_cache_footprint_bytes(
        _tip_worker(model_specs, seqs=fit)._kv_cache_need_bytes()
    ) <= budget.available_bytes
    over = _tip_worker(model_specs, seqs=fit + 1)
    assert over._kv_cache_footprint_bytes(over._kv_cache_need_bytes()) > budget.available_bytes
    # The live configuration is left as it was.
    assert worker.vllm_config.scheduler_config.max_num_seqs == 256
    # 15.37 GiB at about 0.099 GiB per sequence of 8192 tokens.
    assert 150 <= fit <= 160


def _footprint_gib(model_specs, seqs: int) -> str:
    worker = _tip_worker(model_specs, seqs=seqs)
    return f"{worker._kv_cache_footprint_bytes(worker._kv_cache_need_bytes()) / GIB:.3f} GiB"


def test_vllms_default_256_at_8k_is_refused_naming_the_seqs_that_fit(model_specs, monkeypatch) -> None:
    """256 x 8192 needs about 25 GiB; the refusal names the S that fits and both footprints."""
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "model", "--max-model-len", "8192"])
    worker = _graph(_tip_worker(model_specs, seqs=256), int(1.31 * GIB))
    budget = worker._determine_available_memory_neuron(kv.GPU_MEMORY_UTILIZATION)
    fit = worker._max_num_seqs_that_fit(budget.available_bytes)

    with pytest.raises(RuntimeError) as refusal:
        worker.determine_available_memory()

    message = str(refusal.value)
    assert f"relaunch with --max-num-seqs {fit}" in message
    assert "vLLM's default" in message
    assert f"max_num_seqs=256 is vLLM's default (no --max-num-seqs given), allocated as {_footprint_gib(model_specs, 256)}" in message
    assert f"the largest max_num_seqs that fits at 8192 tokens is {fit}, allocated as {_footprint_gib(model_specs, fit)}" in message


@pytest.mark.parametrize(
    "argv", [["--max-num-seqs", "256"], ["--max-num-seqs=256"], ["--max_num_seqs", "256"]]
)
def test_an_explicit_max_num_seqs_is_refused_without_a_derived_count(model_specs, monkeypatch, argv) -> None:
    """The operator's value is checked as before: refused, with no count substituted."""
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "model", *argv])
    worker = _graph(_tip_worker(model_specs, seqs=256), int(1.31 * GIB))
    budget = worker._determine_available_memory_neuron(kv.GPU_MEMORY_UTILIZATION)
    fit = worker._max_num_seqs_that_fit(budget.available_bytes)

    with pytest.raises(RuntimeError) as refusal:
        worker.determine_available_memory()

    message = str(refusal.value)
    assert "short by" in message
    assert _footprint_gib(model_specs, 256) in message
    assert "Lower max_num_seqs or max_model_len" in message
    assert f"--max-num-seqs {fit}" not in message
    assert "vLLM's default" not in message


def test_one_sequence_that_does_not_fit_names_no_seqs_count(model_specs) -> None:
    """1 x 1M (11.70 GiB) against a 6 GiB graph need: only max_model_len can fix it."""
    worker = _graph(_tip_worker(model_specs, seqs=1, length=1024 * 1024), 6 * GIB)

    with pytest.raises(RuntimeError) as refusal:
        worker.determine_available_memory()

    assert "not even one sequence" in str(refusal.value).lower()
    assert "relaunch with --max-num-seqs" not in str(refusal.value)
