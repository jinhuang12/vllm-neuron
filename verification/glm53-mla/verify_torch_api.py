"""Check the real public MLA seam through the Lite Torch compiler.

This uses saved standalone-kernel input and output tensors. It also changes
selection values without changing tensor shapes, and checks CPU references.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import libtorch_neuronx_lite  # Registers the Neuron device.
from libtorch_neuronx_lite.compile.backend import compile as neuron_compile

from benchmarks.glm53_mla.reference import Inputs, metrics, sparse_reference
from vllm_neuron.functional.attention import mla_sparse


def write_then_attend(query, cache, indices, rows, values, scale, q_pe, k_pe):
    """Use the model's out-of-place write before reading selected cache rows."""
    updated = cache.index_copy(0, rows, values)
    return mla_sparse.mla_sparse_attention(query, updated, indices, scale, q_pe, k_pe)


def check_eager_rejections(case):
    """Exercise real CPU-readable seam guards before any device compilation."""
    calls = []
    for name, value in (("index_below_sentinel", -2),
                        ("index_past_cache", case.cache.shape[0])):
        indices = case.indices.clone()
        indices[0, 0] = value
        calls.append((name, case.q_lift, case.cache, indices))
    calls.append(("invalid_query_rank", case.q_lift[0], case.cache, case.indices))
    calls.append(("invalid_selected_width", case.q_lift, case.cache,
                  torch.zeros((case.q_lift.shape[0], 130), dtype=torch.int32)))
    records = []
    mla_sparse.reset_mla_sparse_dispatch_counters()
    for name, query, cache, indices in calls:
        try:
            mla_sparse.mla_sparse_attention(query, cache, indices, case.scale,
                                           case.q_pe, case.k_pe)
        except mla_sparse.MlaSparseAttentionError as error:
            records.append({"case": name, "pass": True,
                            "exception": type(error).__name__, "message": str(error)})
        except Exception as error:
            records.append({"case": name, "pass": False,
                            "exception": type(error).__name__, "message": str(error)})
        else:
            records.append({"case": name, "pass": False, "message": "No rejection"})
    counters = mla_sparse.mla_sparse_dispatch_counters()
    return {"cases": records, "dispatch_counters": list(counters),
            "pass": all(record["pass"] for record in records) and counters == (0, 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Directory with inputs.pt and output.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "NEURON_RT_VISIBLE_CORES" not in os.environ:
        raise RuntimeError("Set the lead-assigned logical core")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = Path(inspect.getfile(mla_sparse))
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    (args.output / "kernel_source.py").write_bytes(source_bytes)
    inputs = torch.load(args.run / "inputs.pt", map_location="cpu", weights_only=True)
    expected_device = torch.load(args.run / "output.pt", map_location="cpu", weights_only=True)
    case = Inputs(**inputs)
    guards = check_eager_rejections(case)
    (args.output / "eager-guards.json").write_text(json.dumps(guards, indent=2) + "\n")
    if not guards["pass"]:
        raise RuntimeError("The public seam failed its eager input guards")
    compile_graphs = []

    def backend(graph, sample_inputs, **kwargs):
        compile_graphs.append(str(graph.graph))
        return neuron_compile(graph, sample_inputs, **kwargs)

    operation = torch.compile(mla_sparse.mla_sparse_attention, backend=backend,
                              fullgraph=True, dynamic=False,
                              options={"compiler_workdir": str(args.output / "compile")})
    dev_q = case.q_lift.to("neuron:0")
    dev_cache = case.cache.to("neuron:0")
    dev_qpe = None if case.q_pe is None else case.q_pe.to("neuron:0")
    dev_kpe = None if case.k_pe is None else case.k_pe.to("neuron:0")
    reports = []
    for kind in ("original", "reordered", "all_sentinel", "one_valid"):
        indices = case.indices.clone()
        if kind == "reordered":
            indices = indices.flip(1).contiguous()
        elif kind == "all_sentinel":
            indices.fill_(-1)
        elif kind == "one_valid":
            indices.fill_(-1)
            indices[:, -1] = case.cache.shape[0] - 1
        variant = Inputs(case.q_lift, case.cache, indices, case.scale, case.q_pe, case.k_pe)
        expected_cpu = sparse_reference(variant)
        got = operation(dev_q, dev_cache, indices.to("neuron:0"), case.scale,
                        dev_qpe, dev_kpe).cpu()
        record = {"case": kind, "cpu_accuracy": metrics(got, expected_cpu),
                  "dtype": str(got.dtype), "shape": list(got.shape),
                  "compile_calls": len(compile_graphs)}
        if kind == "original":
            record["standalone_bitwise_equal"] = bool(torch.equal(got, expected_device))
            record["standalone_accuracy"] = metrics(got, expected_device)
        if kind == "all_sentinel":
            record["empty_output_exact_zero"] = bool(torch.count_nonzero(got) == 0)
        reports.append(record)
        torch.save(got, args.output / f"{kind}.pt")
        (args.output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    write_graphs = []

    def write_backend(graph, sample_inputs, **kwargs):
        write_graphs.append(str(graph.graph))
        return neuron_compile(graph, sample_inputs, **kwargs)

    write_operation = torch.compile(write_then_attend, backend=write_backend,
                                    fullgraph=True, dynamic=False,
                                    options={"compiler_workdir": str(args.output / "write-compile")})
    rows = torch.tensor([0, case.cache.shape[0] - 1], dtype=torch.int64)
    selected = torch.full_like(case.indices, -1)
    selected[:, -1] = case.cache.shape[0] - 1
    if case.q_lift.shape[0] > 1:
        selected[0].fill_(-1)  # A padded query must stay zero after either write.
    stale = sparse_reference(Inputs(case.q_lift, case.cache, selected, case.scale,
                                    case.q_pe, case.k_pe))
    write_reports = []
    for offset in (8., 16.):
        values = (case.cache[rows].float() + offset).to(case.cache.dtype)
        updated = case.cache.index_copy(0, rows, values)
        expected = sparse_reference(Inputs(case.q_lift, updated, selected, case.scale,
                                           case.q_pe, case.k_pe))
        got = write_operation(dev_q, dev_cache, selected.to("neuron:0"), rows.to("neuron:0"),
                              values.to("neuron:0"), case.scale, dev_qpe, dev_kpe).cpu()
        record = {"write_offset": offset, "cpu_accuracy": metrics(got, expected),
                  "stale_cache_control_rejected": not metrics(stale, expected)["allclose"],
                  "padded_rows_zero": bool(torch.count_nonzero(got[(selected == -1).all(dim=1)]) == 0),
                  "compile_calls": len(write_graphs)}
        write_reports.append(record)
        torch.save(got, args.output / f"updated-{int(offset)}.pt")
        print(json.dumps(record), flush=True)
    report = {"boundary": "public mla_sparse_attention through torch.compile on the Neuron device",
              "compile_backend": "libtorch_neuronx_lite.compile.backend.compile",
              "execution_backend": os.environ.get("NEURON_EXECUTION_BACKEND", "lite"),
              "source_sha256": source_hash,
              "source_unchanged": hashlib.sha256(source.read_bytes()).hexdigest() == source_hash,
              "versions": {name: importlib.metadata.version(name) for name in
                           ("torch", "nki", "neuronx-cc", "libtorch-neuronx-lite")},
              "visible_cores": os.environ["NEURON_RT_VISIBLE_CORES"],
              "input_run": str(args.run),
              "eager_guards": guards,
              "compile_calls": len(compile_graphs), "cases": reports,
              "cache_write_cases": write_reports,
              "cache_write_compile_calls": len(write_graphs),
              "cache_write_in_graph": any("index_copy" in graph and "nki" in graph.lower()
                                           for graph in write_graphs),
              "nki_in_graph": any("nki" in graph.lower() for graph in compile_graphs)}
    report["pass"] = (len(compile_graphs) == 1 and report["nki_in_graph"] and report["source_unchanged"]
        and all(row["cpu_accuracy"]["allclose"] and row["dtype"] == "torch.float32"
                and row["shape"] == list(case.q_lift.shape) for row in reports)
        and reports[0]["standalone_bitwise_equal"]
        and reports[2]["empty_output_exact_zero"]
        and len(write_graphs) == 1 and report["cache_write_in_graph"]
        and all(row["cpu_accuracy"]["allclose"] and row["stale_cache_control_rejected"]
                and row["padded_rows_zero"] for row in write_reports))
    (args.output / "graphs.txt").write_text("\n\n".join(compile_graphs))
    (args.output / "write-graphs.txt").write_text("\n\n".join(write_graphs))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
