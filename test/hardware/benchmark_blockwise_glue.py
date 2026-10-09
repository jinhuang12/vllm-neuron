# SPDX-License-Identifier: Apache-2.0
"""Device time of the blockwise-fp8 SwiGLU MLP call sites, and of the glue between kernels.

Run it through the device lease (``devlease.py slice <name>``), which pins the cores
and the LNC; this script does not select cores, and refuses to start without them.

What runs. One graph per (case, variant, seed): one MLP call site at a step of ``rows``
tokens on one TP=64 rank's shard.

* ``shared``: ``Glm5NextSharedExperts.shared_expert_mm``, H=4096, I=128 (the
  2048-wide shared expert over 64 ranks is 32 columns, padded to one 128 block).
* ``dense``: ``Glm5NextDenseMLP.forward``, H=4096, I=256 (12288 over 64 ranks is 192,
  padded to two blocks).

``rows`` is a whole number of 128-token tiles (the prefill route: three kernel calls and
the glue between them), or ``1 <= rows < 128`` (the decode route: one small-M fused MLP
kernel); either way neither call site pads or slices. Weights are random fp8-e4m3 inside
+-240 with random block grids; ``x`` is random bf16.

Variants, on the same operands in one process:

* ``tree``: this tree's call site itself, the model class's method on a module built
  from the checkpoint's text config, its weights and grids bound in the compute frame
  and its ``prepare_scale_operands`` run as the load path runs it (by keyword).
* ``base`` (with ``--baseline-module``): the call site as it is at ab4f37fc
  (``model_fp8.py``: ``shared_expert_mm`` passes the load-time kernel scale operands,
  ``Glm5NextDenseMLP.forward`` passes the public grids alone), reproduced as the
  ``blockwise_fp8_mlp`` call it makes, through the module file given there loaded
  beside this tree's. At these row counts neither call site pads or slices, so the call
  is the whole site.

What is measured:

1. Device ms per execution from the runtime system trace (``nc_exec_running``, the two
   physical cores' intervals merged). ``--rounds`` rounds; each round runs every
   variant ``--iterations`` times, rotating the order per iteration (after
   ``--warmup`` calls in round 0). Per variant the per-round medians, their median and
   spread (max - min); per variant against ``base`` the paired per-round delta.
2. With ``--profile-dir``: a device profile of ``--profile-iterations`` calls per
   variant (seed 0), for the per-region attribution of the glue between the kernel
   calls (done offline from the profile and the compile's intermediates).
3. Numerics: each variant's first-call output per seed (``--seeds``), saved with its
   operands under ``--save-io``; ``bit_equal`` and max |d| of ``tree`` against ``base``.

The compile cache is used (``NEURON_LIBTORCH_CACHE_ROOT``): its key folds the kernel
modules' sources, but give it a fresh root when the compile's intermediates are kept.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

#: The worktree root, ahead of any installed copy (the lease command sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
from test.hardware import benchmark_layer_glue as micro  # noqa: E402
from test.vllm_neuron.functional.glue import glue_case  # noqa: E402
from vllm_neuron.model.glm5_next import model_fp8  # noqa: E402

# By module path: ``vllm_neuron.functional`` re-exports a function of the same name.
tree_module = importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")

DEVICE = "neuron:0"
HIDDEN = 4096
#: One TP=64 rank's intermediate width per site, each padded to whole 128 blocks.
SITE_INTERMEDIATE = {"shared": 128, "dense": 256}
#: Sites whose call passes the load-time kernel scale operands at ab4f37fc.
BASE_SITE_PREBUILT = {"shared": True, "dense": False}
#: The model's names of the three projections, in the order the call sites take them.
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
#: fp8-e4m3's largest finite magnitude on trn2 (the checkpoint load squeezes into it).
FP8_LIMIT = 240.0


def parse_case(text: str) -> dict:
    site, rows = text.split(":")
    if site not in SITE_INTERMEDIATE or int(rows) < 1 or (int(rows) >= 128 and int(rows) % 128):
        raise ValueError(f"case {text!r}: want shared|dense : rows (1-127 or a multiple of 128)")
    return {"site": site, "rows": int(rows), "tag": text}


def load_module(path: Path):
    """The ``blockwise_fp8_mm`` module file at ``path``, as its own module."""
    spec = importlib.util.spec_from_file_location("_blockwise_glue_base", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def operands(case: dict, seed: int) -> dict:
    """``x``, three fp8 weights and three public grids for one site, on the CPU."""
    gen = torch.Generator().manual_seed(1000 * seed + case["rows"]
                                        + SITE_INTERMEDIATE[case["site"]])
    inter = SITE_INTERMEDIATE[case["site"]]

    def weight(rows, cols):
        return (torch.randn((rows, cols), generator=gen) * 64).clamp(
            -FP8_LIMIT, FP8_LIMIT).to(torch.float8_e4m3fn)

    def grid(rows, cols, magnitude):
        return ((torch.rand((rows // 128, cols // 128), generator=gen) + 0.37)
                * magnitude).float()

    x = torch.randn((case["rows"], HIDDEN), generator=gen).to(torch.bfloat16)
    return {"x": x, "gate_weight": weight(HIDDEN, inter), "up_weight": weight(HIDDEN, inter),
            "down_weight": weight(inter, HIDDEN), "gate_scale": grid(HIDDEN, inter, 3e-3),
            "up_scale": grid(HIDDEN, inter, 3e-3), "down_scale": grid(inter, HIDDEN, 1e-4)}


def compile_site(site, args):
    return torch.compile(site, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": args.compiler_args})


def build_base(case: dict, module, ops: dict, text_config, args):
    """``(graph, inputs)``: the ab4f37fc call site's ``blockwise_fp8_mlp`` call through
    ``module``."""
    names = ("gate", "up", "down")
    tensors = [ops["x"], *(ops[f"{n}_weight"] for n in names),
               *(ops[f"{n}_scale"] for n in names)]
    if BASE_SITE_PREBUILT[case["site"]]:
        # Built once on the CPU, as the shared expert's load-time prep does.
        tensors += [module.to_kernel_scale_layout(ops[f"{n}_scale"],
                                                  *ops[f"{n}_weight"].shape) for n in names]

    def site(x, gate_w, up_w, down_w, gate_s, up_s, down_s, *prebuilt):
        return module.blockwise_fp8_mlp(
            x, gate_w, up_w, down_w, gate_s, up_s, down_s,
            swiglu_limit=float(text_config.swiglu_limit),
            prebuilt_scale_t=tuple(prebuilt) if prebuilt else None)

    return compile_site(site, args), tuple(t.to(DEVICE) for t in tensors)


def build_tree(case: dict, ops: dict, text_config, quant_config, args):
    """``(graph, inputs)``: this tree's call site, on a module whose scale prep ran."""
    weights = {f"{p}_weight": ops[f"{p.split('_')[0]}_weight"].to(DEVICE) for p in PROJECTIONS}
    grids = {f"{p}_scale": ops[f"{p.split('_')[0]}_scale"].to(DEVICE) for p in PROJECTIONS}
    x = ops["x"].to(DEVICE)
    if case["site"] == "shared":
        module = model_fp8.Glm5NextSharedExperts(text_config)
        module.prepare_scale_operands(**weights, **grids)

        def site(x, *operands):
            return module.shared_expert_mm(
                x, *operands, quant_config=quant_config)

        return compile_site(site, args), (x, *weights.values(), *grids.values())
    module = model_fp8.Glm5NextDenseMLP(text_config)
    for leaf, weight in weights.items():
        setattr(module, leaf, torch.nn.Parameter(weight, requires_grad=False))
        grid_name = model_fp8.Glm5NextForConditionalGeneration._sibling_scale_grid_name(leaf)
        setattr(module, grid_name, grids[leaf.replace("_weight", "_scale")])
    module.prepare_scale_operands(**weights, **grids)

    def site(x):
        return module.forward(x, quant_config=quant_config)

    return compile_site(site, args), (x,)


def summarise(rounds: list[dict], variants: list[str]) -> dict:
    medians = {v: [rd[v]["device"]["median_ms"] for rd in rounds] for v in variants}
    out = {}
    for v in variants:
        m = medians[v]
        row = {"round_medians_ms": m, "median_ms": statistics.median(m),
               "spread_ms": max(m) - min(m)}
        if v != "base" and "base" in medians:
            deltas = [a - b for a, b in zip(m, medians["base"])]
            row.update(delta_vs_base_ms_per_round=deltas,
                       delta_vs_base_median_ms=statistics.median(deltas),
                       ratio_vs_base=statistics.median(m) / statistics.median(medians["base"]))
        out[v] = row
    return out


def run_case(case: dict, base_module, args) -> dict:
    torch._dynamo.reset()
    text_config = glue_case.text_config()
    quant_config = glue_case.quant_config(model_fp8)
    result = {"case": case["tag"], **case, "intermediate": SITE_INTERMEDIATE[case["site"]],
              "base_prebuilt_scale_operands": BASE_SITE_PREBUILT[case["site"]],
              "numerics": []}
    timed = {}
    for seed in args.seeds:
        ops = operands(case, seed)
        outs = {}
        built = {}
        if base_module is not None:
            built["base"] = build_base(case, base_module, ops, text_config, args)
        built["tree"] = build_tree(case, ops, text_config, quant_config, args)
        for name, (graph, inputs) in built.items():
            t0 = time.time()
            outs[name] = graph(*inputs).to("cpu")
            if seed == args.seeds[0]:
                timed[name] = (graph, inputs)
                result.setdefault("compile_and_first_call_s", {})[name] = time.time() - t0
        row = {"seed": seed}
        if "base" in outs:
            diff = (outs["tree"].float() - outs["base"].float()).abs()
            row.update(bit_equal=bool(torch.equal(outs["tree"], outs["base"])),
                       max_abs_diff=float(diff.max()),
                       elements_differing=int((outs["tree"] != outs["base"]).sum()),
                       max_abs_base=float(outs["base"].abs().max()))
        if args.save_io is not None:
            args.save_io.mkdir(parents=True, exist_ok=True)
            path = args.save_io / f"{case['site']}_{case['rows']}_seed{seed}.pt"
            torch.save({"operands": ops, "outputs": outs}, path)
            row["io_file"] = str(path)
        result["numerics"].append(row)
        print(json.dumps({"case": case["tag"], **row}), flush=True)
    graphs = {n: g for n, (g, _) in timed.items()}
    inputs = {n: i for n, (_, i) in timed.items()}
    rounds = []
    for r in range(args.rounds):
        saved = args.warmup
        args.warmup = args.warmup if r == 0 else 1
        try:
            rounds.append(micro.time_graphs(graphs, inputs, args))
        finally:
            args.warmup = saved
    result["rounds"] = [{n: rd[n]["device"] for n in graphs} for rd in rounds]
    result["summary"] = summarise(rounds, list(graphs))
    if args.profile_dir is not None:
        result["profiles"] = {}
        for name, graph in graphs.items():
            pname = f"{case['site']}_{case['rows']}_{name}"
            directory = micro.profile_graph(pname, graph, inputs[name], args)
            result["profiles"][name] = {"profile_dir": str(directory), "name": pname}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+",
                        default=["shared:1024", "shared:2048", "dense:1024", "dense:2048"])
    parser.add_argument("--baseline-module", type=Path, default=None,
                        help="a blockwise_fp8_mm.py to run as variant 'base'")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument("--save-io", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=None,
                        help="the compilers' working directory (default: a new temp dir)")
    parser.add_argument("--compiler-args", nargs="*",
                        default=(os.environ.get("NEURON_CC_FLAGS", "").split()
                                 or micro.MODEL_COMPILER_ARGS))
    args = parser.parse_args()
    for name in ("output", "profile_dir", "save_io", "baseline_module", "workdir"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Set NEURON_RT_VISIBLE_CORES to the cores this benchmark may use; "
                         "it does not select cores itself")
    for module in (tree_module, model_fp8):
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise RuntimeError(f"{module.__name__} imported from {module.__file__}, not {ROOT}")
    cases = [parse_case(c) for c in args.cases]
    base_module = None if args.baseline_module is None else load_module(args.baseline_module)
    modules = {"tree": model_fp8, "tree_blockwise": tree_module}
    if base_module is not None:
        modules["base"] = base_module
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="blockwise-glue-bench-cwd-"))
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)  # the compilers write their intermediates into the working directory
    report = {
        "environment": {k: os.environ.get(k) for k in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_LIBTORCH_CACHE_ROOT")},
        "device": "neuron:0 (one logical core = 2 physical cores at LNC2)",
        "compiler_args": args.compiler_args, "workdir": str(workdir),
        "modules": {n: str(Path(m.__file__).resolve()) for n, m in modules.items()},
        "method": __doc__.split("What is measured:", 1)[1].split("The compile cache", 1)[0]
        .strip(),
        "rounds": args.rounds, "warmup": args.warmup, "iterations": args.iterations,
        "seeds": args.seeds, "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        report["cases"].append(run_case(case, base_module, args))
        args.output.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({case["tag"]: report["cases"][-1]["summary"]}), flush=True)


if __name__ == "__main__":
    main()
