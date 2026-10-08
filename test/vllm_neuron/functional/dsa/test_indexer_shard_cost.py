# SPDX-License-Identifier: Apache-2.0
"""The arithmetic of the query-sharded selection's cost model (``indexer_shard_cost.py``).

These pin what the report's numbers rest on: the baseline is worker-3's calibrated model
unchanged, degree 1 changes nothing, the top-k fit reproduces its two measured points, the
all-gather is the calibrated all-reduce's own half, and the committed report JSON is what
the model computes today. They read worker-3's directory and never write to it.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from test.vllm_neuron import artifacts
from test.vllm_neuron.functional.dsa import indexer_shard_cost as cost_model

REPORT_JSON = str(artifacts.campaign_path("glm53f-wt5", "reports", "indexer_shard.json"))
PUBLISHED_JSON = str(artifacts.campaign_path("glm53f-wt3", "reports",
                                             "prefill_calibrated.json"))

pytestmark = pytest.mark.skipif(
    not (cost_model.calibration_available() and os.path.exists(PUBLISHED_JSON)),
    reason=(f"worker-3's calibrated prefill model is not on this host: needs "
            f"{cost_model.CALIB_DIR}, {cost_model.CAL_CONSTANTS}, "
            f"{cost_model.ENTITLEMENT_JSON}, {cost_model.DEVICE_RECORDS_DIR} and "
            f"{PUBLISHED_JSON} ({artifacts.CAMPAIGN_KNOB})"))


@pytest.fixture(scope="module")
def cost():
    return cost_model.IndexerCost()


def _published_rows() -> dict:
    """worker-3's published rows, TP = 64 colocated, bs = 1, chunk 1024, keyed by
    ``(context, prompt)``."""
    with open(PUBLISHED_JSON) as handle:
        rows = json.load(handle)["rows"]
    return {(r["context"], r["prompt_tokens"]): r["ttft_ms"] for r in rows
            if r["config_id"] == "colocated-c16-tp64-cp1" and r["line"] == "bs1"
            and r["chunk"] == cost_model.CHUNK}


def test_the_baseline_is_the_published_calibrated_table(cost):
    """"before" is worker-3's own number wherever ``prefill_calibrated.json`` has the row."""
    published = _published_rows()
    checked = 0
    for prompt in cost_model.TTFT_PROMPTS.values():
        for column in ("asbuilt", "withbranches"):
            row = cost.ttft(prompt, column)
            want = published.get((row["context"], prompt))
            if want is None:
                continue
            # The published rows round to 0.1 ms.
            assert row["before_ms"] == pytest.approx(want[column], abs=0.05), (prompt, column)
            checked += 1
    assert checked >= 6, "the 1k, 8k and 256k rows are published for both columns"


def test_the_model_dials_are_the_calibrations_own(cost):
    arch = cost.m.r["architecture"]
    assert cost_model.INDEX_KPOOL == int(arch["index_kpool"])
    assert cost_model.SELECT_K == int(arch["index_topk"]) // int(arch["index_kpool"])


@pytest.mark.parametrize("cands", [1024, 2048, 16384, 65536])
def test_degree_one_changes_nothing(cost, cands):
    one = cost.query_sharded(cands, 1)
    assert one["compute_after_ms"] == pytest.approx(one["compute_before_ms"], rel=1e-12)
    assert one["comm_added_ms"] == one["glue_added_ms"] == 0.0
    assert one["net_saved_ms"] == pytest.approx(0.0, abs=1e-9)


def test_the_topk_fit_reproduces_both_measured_points(cost):
    for cands in cost_model.FIT_CANDS:
        fit = cost.layers * cost.topk_ms_layer(cost_model.CHUNK, cands)
        assert fit == pytest.approx(cost.m.idx_value("rotational_topk", cands), rel=1e-9)
    # A physical fit: time grows with cycles, and a per-call fixed part remains.
    assert cost.beta > 0 and cost.alpha > 0


def test_the_all_gather_is_half_the_calibrated_all_reduce():
    import sys

    sys.path.insert(0, cost_model.PLANNER_DIR)
    try:
        import entitlement_formulas as ef
    finally:
        sys.path.remove(cost_model.PLANNER_DIR)
    with open(cost_model.CAL_CONSTANTS) as handle:
        cal = json.load(handle)
    hw = ef.HW(**{k: v for k, v in cal.items() if not k.startswith("_")})
    for tp in (8, 16, 64):
        for nbytes in (1 << 21, 1 << 23):
            assert 2 * cost_model.allgather_ms(tp, nbytes) == pytest.approx(
                ef.allreduce_ms(hw, tp, nbytes), rel=1e-12)


def test_small_tiles_cost_a_whole_tile_in_the_score_and_bound(cost):
    """d = 8 (128 rows) and d = 64 (16 rows) both run one row tile: equal score and bound."""
    for cands in (2048, 65536):
        d8, d64 = cost.query_sharded(cands, 8), cost.query_sharded(cands, 64)
        for bid in ("score_gemm", "causal_bound", "sentinel_order"):
            assert d8["sharded_ms"][bid] == pytest.approx(d64["sharded_ms"][bid], rel=1e-12)
        # Only the top-k and the collective tell them apart; the top-k wins for d = 64.
        assert d64["sharded_ms"]["rotational_topk"] < d8["sharded_ms"]["rotational_topk"]
        assert d64["net_saved_ms"] > d8["net_saved_ms"]


#: "The saving dominates what sharding adds": at least an order of magnitude above the
#: collective and the glue together, at every priced context.
DOMINANCE = 10


def test_the_saving_grows_with_context_and_beats_its_comm(cost):
    nets = []
    for context in cost_model.CONTEXTS.values():
        q = cost.query_sharded(context // cost_model.INDEX_KPOOL, cost_model.TP)
        assert q["compute_saved_ms"] > DOMINANCE * (q["comm_added_ms"] + q["glue_added_ms"])
        nets.append(q["net_saved_ms"])
    assert nets == sorted(nets)


def test_the_ttft_after_is_before_less_the_per_chunk_saving(cost):
    for prompt in cost_model.TTFT_PROMPTS.values():
        row = cost.ttft(prompt, "asbuilt")
        assert row["n_chunks"] == math.ceil(prompt / cost_model.CHUNK)
        assert row["after_ms"] == pytest.approx(
            row["before_ms"] - row["n_chunks"] * row["saved_per_chunk_ms"], rel=1e-12)


def test_every_device_run_passed_its_checks(cost):
    """Every repeat of every run: the rank's rows select the replicated sets, the cut is
    bit-exact, every stage took NKI; and each C = 65536 path's compile log names its error."""
    ab = cost_model.device_ab(cost)
    assert set(ab) == {*cost_model.DEVICE_RUNS, "compile_failures"}
    for run in cost_model.DEVICE_RUNS:
        assert ab[run]["repeats"] == cost_model.DEVICE_REPEATS, run
        assert ab[run]["checks_pass"], run
    assert set(ab["compile_failures"]) == set(cost_model.COMPILE_FAILURE_LOGS)


def test_every_profiled_execution_divides_into_named_ops():
    """The segments of each profiled execution add up to it, and in the selection graphs
    each one is an op the report names (no unattributed compiler segment)."""
    for run in cost_model.DEVICE_RUNS:
        with open(os.path.join(cost_model.DEVICE_RECORDS_DIR, f"{run}_r1.ops.json")) as handle:
            profiled = json.load(handle)
        for graph in ("replicated", "sharded", "precut"):
            for execution in profiled["graphs"][graph]["executions"]:
                times, _ = cost_model._segment_ops(execution)
                assert sum(times.values()) == pytest.approx(execution["execution_us"], abs=1e-6)
                assert "compiler op (other)" not in times, (run, graph, times)


def test_every_compiled_selection_graph_orders_every_overlapping_pair():
    """trn2-1's pipeline-hazard check (report section 17). In every compiled graph, on both
    cores: no unordered pair, no stepped DMA, every wait holds, and no device opened. Each
    recompiled device graph is the NEFF that ran. The only instructions that write over
    their own input, other than exact in-place ops, are the gathers of
    ``OWN_DATA_GATHERS``; each writes at most one piece from the first byte of its data, and
    only the as-built chain has the partial one. On the device no piece of those gathers
    reads what another wrote, and none writes over its indices."""
    data = cost_model.depcheck()
    assert set(data["cpu_graphs"]) == set(cost_model.DEPCHECK_CPU_GRAPHS)
    assert set(data["device_graphs"]) == set(cost_model.DEPCHECK_DEVICE_GRAPHS)
    kernels = {(kernel, int(line.rsplit(":", 1)[1]))
               for line, names in cost_model.OWN_DATA_GATHERS.items() for kernel in names}
    for name, graph in {**data["cpu_graphs"], **data["device_graphs"]}.items():
        assert graph["device_opens"] == 0, name
        assert set(graph["cores"]) == set(cost_model.DEPCHECK_CORES), name
        as_built = graph.get("mode", graph.get("graph")) == "replicated"
        for core, row in graph["cores"].items():
            found = (row["overlapping_pairs"] - row["ordered"], row["unsync"],
                     row["queue_order_only"], row["stepped_partition_dmas"],
                     row["waits_not_ok"])
            assert found == (0, 0, 0, 0, 0), (name, core, row)
            partial = [a for a in row["gather_aliases"] if not a["identical"]]
            assert row["alias_partial"] == len(partial), (name, core, row)
            assert as_built or not partial, (name, core, partial)
            for alias in row["gather_aliases"]:
                assert (alias["kernel"], alias["source_line"]) in kernels, (name, core, alias)
                assert alias["dst_at_data_base"], (name, core, alias)
                assert alias["write_bytes"] <= cost_model.GATHER_PIECE_BYTES, (name, core, alias)
    for name, graph in data["device_graphs"].items():
        assert graph["neff_bin_same"] > 0 and graph["neff_bin_diff"] == 0, name
        assert graph["kernel_hashes_same"], name
    assert set(data["gather_pieces_on_device"]) == set(cost_model.OWN_DATA_GATHERS)
    for line, graphs in data["gather_pieces_on_device"].items():
        assert graphs, line
        for name, entry in graphs.items():
            assert all(entry[key] == 0 for key in cost_model.GATHER_HAZARDS), (line, name, entry)


def _leaf_paths(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaf_paths(value, (*path, key))
    else:
        yield path


def test_every_value_in_the_report_json_has_a_label():
    data = cost_model.report()
    unlabeled = [".".join(path) for path in _leaf_paths(data)
                 if path[0] not in ("what", "labels")
                 and not any(key in cost_model.LABELS for key in path)]
    assert not unlabeled, unlabeled
    assert data["labels"] == cost_model.LABELS


def test_the_committed_report_json_is_what_the_model_computes():
    if not os.path.exists(REPORT_JSON):
        pytest.skip(f"{REPORT_JSON} is not written yet")
    fresh = json.loads(json.dumps(cost_model.report(), sort_keys=True, default=float))
    with open(REPORT_JSON) as handle:
        assert json.load(handle) == fresh
