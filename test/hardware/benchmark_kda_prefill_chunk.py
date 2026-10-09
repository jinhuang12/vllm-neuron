# SPDX-License-Identifier: Apache-2.0
"""The KDA prefill chunk kernels on Neuron: revisions of them against each other.

Times :func:`kda_intra_chunk` (stages 1 to 3), :func:`kda_inter_chunk` (stages 4
and 5) and the two chained, intra-chunk outputs into the inter-chunk call as a
prefill layer runs them, at the served prefill sizes, 512, 1024 and 2048 tokens per
call. The variants are the baseline revision, this checkout and any
``--variant NAME=REV``. Each variant runs at the chunk width ``C`` the KDA layer
resolves when it runs that variant's kernels, so its call shape is
``[NC, C, K] = [tokens / C, C, 128]``: ``[128, 8, 128]`` for a 1024-token call at
``C = 8`` and ``[64, 16, 128]`` at ``C = 16``.

A revision's variant is its ``chunked_recurrence.py`` plus every module of the
KDA package it imports, at any depth, read with ``git show``. Those modules
import one another by their package names. While they load, those names are
bound to the variant's copies, and then the live modules are restored. A module
this checkout does not have stays bound to the variant's copy for the whole run,
because a compiled graph that imported it re-checks that binding at every call.
A revision that imports, inside a function, a module this checkout also has is
refused, because its graphs would resolve this checkout's module.

One compiled graph holds ``--layers`` independent calls of one entry point (or
one chained pair), each on its own operands, because a prefill chunk calls each
kernel once per KDA layer. A graph's time is the host wall time of one launch
plus the copy of its first output to CPU; an empty graph over the same operands
that returns a tensor of the same shape (``floor``) is timed alongside, and the
per-call figure is ``(graph - floor) / layers``.

The graphs are timed interleaved: every repetition times every graph once, in the
same order, ``--iterations`` launches each after ``--warmup``. A graph's figure is
the median over repetitions of the per-repetition medians. A kernel's per-call
figure in one repetition is that repetition's ``(graph - floor) / layers``, and
its spread is ``(max - min) / median`` of those over the repetitions. The noise
floor is the largest per-call spread over every kernel and variant, i.e. how far
one variant's own per-call figure moves between repetitions in this run; the
graphs' own spreads, the floor graph's included, are reported beside it.

Numerics. For each ``--seeds`` entry, each ``--gates`` draw and each size,
one-call graphs of every variant run on operands drawn per token, so every chunk
width sees the same flat sequence. The outputs are saved as ``.pt`` files under
``--numerics-dir`` and compared with ``torch.equal`` against the baseline where
the two run at the same chunk width, and against the CPU references at the
variant's own width (:func:`kda_intra_chunk_torch_oracle`,
:func:`rebuild_i_plus_a`, :func:`kda_inter_chunk_torch_oracle`). The inter-chunk
operands are the CPU reference's intra-chunk outputs, so every variant's
inter-chunk kernel reads identical bits at one width; the chained graph reads each
variant's own intra-chunk outputs. The inter-chunk and chained outputs are also
compared, across widths, with a float64 sequential scan of the flat sequence.

The compiled-kernel cache keys a kernel on its own source text, not on the
helpers or constants it emits through, so a warm cache can serve a kernel built
from older helpers. ``NEURON_LIBTORCH_CACHE_ROOT`` must therefore name an empty
or absent directory, and ``--workdir`` must be empty or absent too; the report
records both, and the kernel artifacts the run compiled.

Run it through the device lease, which pins the cores and the LNC; this script
refuses to run without ``NEURON_RT_VISIBLE_CORES``.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

#: The worktree root, ahead of any installed copy: the lease command sets no
#: PYTHONPATH, and the venv's own ``vllm_neuron`` is another tree.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

import torch

import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
from vllm_neuron.functional.kda import chunked_recurrence as live

DEVICE = "neuron:0"

#: The package the kernel module lives in, and the kernel module's leaf name in it.
KDA_PACKAGE = "vllm_neuron.functional.kda"
KERNEL_LEAF = "chunked_recurrence"

#: Where a leaf of :data:`KDA_PACKAGE` lives in a revision.
PACKAGE_DIR = "vllm_neuron/functional/kda"

#: The revision this series starts from; ``--baseline-rev`` names another.
BASELINE_REV = "f3a833f"

#: The variant names of the baseline revision and of this checkout.
BASELINE = "baseline"
CHECKOUT = "checkout"

#: Tokens per call at the served prefill chunk sizes: 512 (the 256k-context line),
#: 1024 (the short-context line) and 2048 (the 64k-context line).
SERVED_PREFILL_TOKENS = (512, 1024, 2048)

#: The per-rank key and value widths at TP=64 (one head per rank), which
#: ``test_chunked_recurrence`` reads off the layer too. The chunk width is not a
#: constant here: :func:`layer_chunk` asks the layer, per variant.
SERVED_KDIM = 128
SERVED_VDIM = 128

#: KDA layers per prefill chunk, i.e. calls of each kernel per chunk.
KDA_LAYERS = 34

#: The checkpoint's gate lower bound: each per-token log gate lies in
#: ``(GATE_LOWER_BOUND, 0)``, as the gate clamp produces it.
GATE_LOWER_BOUND = -5.0

#: The gate draws the numerics run: ``served`` spreads each log gate over
#: ``(GATE_LOWER_BOUND, 0)`` as the timing operands do; ``bound`` puts every entry
#: at ``GATE_LOWER_BOUND``, the largest chunk-local cumulative gate the layer can
#: send at its chunk width.
GATES = ("served", "bound")

#: The timed and checked graphs: the two entry points and the two chained.
KERNELS = ("intra", "inter", "chain")

INTRA_FIELDS = ("w", "u", "kg", "a_inv", "aqk")
INTER_FIELDS = ("o", "final_state", "v_new")
FIELDS = {"intra": INTRA_FIELDS, "inter": INTER_FIELDS, "chain": INTER_FIELDS}

#: Operands per call of each graph: the entry point's, or for the chain the
#: intra-chunk operands and the entering state.
OPERANDS = {"intra": 5, "inter": 7, "chain": 6}


def _git_show(rev: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "show", f"{rev}:{path}"],
        check=True, capture_output=True, text=True,
    ).stdout


def _imported_leaves(nodes) -> set[str]:
    """The leaves of :data:`KDA_PACKAGE` that the import statements in ``nodes`` name."""
    prefix = KDA_PACKAGE + "."
    leaves = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == KDA_PACKAGE:
                leaves.update(alias.name for alias in node.names)
            elif node.module.startswith(prefix):
                leaves.add(node.module[len(prefix):].split(".")[0])
        elif isinstance(node, ast.Import):
            leaves.update(alias.name[len(prefix):].split(".")[0]
                          for alias in node.names if alias.name.startswith(prefix))
    return leaves


def package_imports(source: str) -> tuple[set[str], set[str]]:
    """The leaves of :data:`KDA_PACKAGE` that ``source`` imports at any scope, and
    those it imports inside a function, i.e. when that function runs."""
    tree = ast.parse(source)
    functions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
    in_functions = _imported_leaves(
        node for scope in ast.walk(tree) if isinstance(scope, functions)
        for node in ast.walk(scope))
    return _imported_leaves(ast.walk(tree)), in_functions


@contextlib.contextmanager
def bound(modules: dict[str, ModuleType]):
    """Bind each leaf of :data:`KDA_PACKAGE` to ``modules[leaf]``, then restore.

    A leaf is bound both in ``sys.modules`` and as the package's attribute, the
    two places ``from package import leaf`` reads.
    """
    package = sys.modules[KDA_PACKAGE]
    absent = object()
    saved = {
        leaf: (sys.modules.get(f"{KDA_PACKAGE}.{leaf}", absent),
               getattr(package, leaf, absent))
        for leaf in modules
    }
    for leaf, module in modules.items():
        sys.modules[f"{KDA_PACKAGE}.{leaf}"] = module
        setattr(package, leaf, module)
    try:
        yield
    finally:
        for leaf, (entry, attribute) in saved.items():
            name = f"{KDA_PACKAGE}.{leaf}"
            if entry is absent:
                del sys.modules[name]
            else:
                sys.modules[name] = entry
            if attribute is absent:
                delattr(package, leaf)
            else:
                setattr(package, leaf, attribute)


def _checkout_has(leaf: str) -> bool:
    """Whether this checkout's package has ``leaf``, on disk, whatever is bound."""
    package = sys.modules[KDA_PACKAGE]
    return importlib.machinery.PathFinder.find_spec(
        f"{KDA_PACKAGE}.{leaf}", package.__path__) is not None


@dataclass
class Variant:
    """One revision of the kernel module: its entry points and the sources loaded."""

    name: str
    rev: str
    module: ModuleType
    sha256: dict[str, str]
    chunk: int


def load_revision(name: str, rev: str, workdir: Path) -> Variant:
    """``rev``'s kernel module and the package modules it imports, as fresh modules.

    Each source is written into ``workdir`` under a name of its own, because the
    NKI front end reads a kernel's source from its file and the compiled-kernel
    cache keys on that file's base name. Each module is registered under its own
    name, because a compiled graph resolves the kernel's globals by module name.
    The package names are bound as the module docstring states.
    """
    sources, in_functions = {}, {}
    pending = [KERNEL_LEAF]
    while pending:
        leaf = pending.pop(0)
        sources[leaf] = _git_show(rev, f"{PACKAGE_DIR}/{leaf}.py")
        every, in_functions[leaf] = package_imports(sources[leaf])
        pending.extend(sorted(every - sources.keys() - set(pending)))
    kept = {leaf for leaf in sources if not _checkout_has(leaf)}
    unbindable = sorted(set().union(*in_functions.values()) - kept)
    if unbindable:
        raise ValueError(
            f"{rev} imports {unbindable} inside a function, and this checkout has "
            f"them too: its graphs would trace this checkout's modules")
    taken = sorted(leaf for leaf in kept if f"{KDA_PACKAGE}.{leaf}" in sys.modules)
    if taken:
        raise ValueError(f"{rev} needs {taken} bound for the run, and another variant holds them")
    modules = {}
    for leaf, source in sources.items():
        path = workdir / f"{name}_{leaf}.py"
        path.write_text(source)
        module_name = f"_kda_prefill_{name}_{leaf}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load {rev}:{PACKAGE_DIR}/{leaf}.py from {path}")
        modules[leaf] = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = modules[leaf]
    # In discovery order, so the kernel module runs before the modules that
    # import it at their top level.
    with bound(modules):
        for module in modules.values():
            module.__spec__.loader.exec_module(module)
    package = sys.modules[KDA_PACKAGE]
    for leaf in sorted(kept):
        sys.modules[f"{KDA_PACKAGE}.{leaf}"] = modules[leaf]
        setattr(package, leaf, modules[leaf])
    return Variant(
        name=name, rev=rev, module=modules[KERNEL_LEAF],
        sha256={leaf: hashlib.sha256(s.encode()).hexdigest() for leaf, s in sources.items()},
        chunk=layer_chunk(modules[KERNEL_LEAF]),
    )


def checkout_variant() -> Variant:
    path = Path(live.__file__)
    return Variant(name=CHECKOUT, rev="checkout", module=live,
                   sha256={KERNEL_LEAF: hashlib.sha256(path.read_bytes()).hexdigest()},
                   chunk=layer_chunk(live))


def layer_chunk(module: ModuleType) -> int:
    """The chunk width the KDA layer resolves when ``module`` is its kernel module.

    The layer derives the width (``_resolve_chunk_size``) from the kernel module's
    gate limit, which it imports by package name when it resolves, so it is asked
    with ``module`` bound to that name. Its gate lower bound must be the one this
    benchmark draws gates from, or the width would be for other inputs.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextKDAAttention

    layer = Glm5NextKDAAttention(Glm5NextTextConfig(), world_size=1)
    if layer.gate_lower_bound != GATE_LOWER_BOUND:
        raise ValueError(f"the layer's gate lower bound is {layer.gate_lower_bound}, "
                         f"not the {GATE_LOWER_BOUND} the operands are drawn from")
    with bound({KERNEL_LEAF: module}):
        return int(layer._resolve_chunk_size(None))


def flat_inputs(tokens: int, seed: int, gate: str = "served") -> tuple:
    """One call's flat operands, ``[T, *]``, and an entering state ``[V, K]``.

    Drawn per token, so every chunk width regroups the same sequence. ``gate`` is
    one of :data:`GATES`; the ``bound`` draw replaces the gate after drawing it, so
    the other operands are the same for both draws.
    """
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn((tokens, SERVED_KDIM), generator=gen)
    k = torch.randn((tokens, SERVED_KDIM), generator=gen)
    v = torch.randn((tokens, SERVED_VDIM), generator=gen)
    beta = torch.sigmoid(torch.randn(tokens, generator=gen))
    gk = GATE_LOWER_BOUND * torch.sigmoid(torch.randn((tokens, SERVED_KDIM), generator=gen) * 2)
    if gate == "bound":
        gk = torch.full_like(gk, GATE_LOWER_BOUND)
    state = torch.randn((SERVED_VDIM, SERVED_KDIM), generator=gen) * 0.5
    return q, k, v, beta, gk, state


def make_inputs(flat: tuple, chunk: int) -> dict:
    """One call's operands for every graph at ``chunk``, as CPU float32 tensors.

    The intra-chunk operands are the flat ones regrouped into chunks; the
    inter-chunk operands are the CPU reference's intra-chunk outputs on them, plus
    the entering state.
    """
    *per_token, state = flat
    q, k, v, beta, gk = (x.reshape(-1, chunk, *x.shape[1:]).contiguous() for x in per_token)
    ref = live.kda_intra_chunk_torch_oracle(q, k, v, beta, gk)
    return {
        "intra": (q, k, v, beta, gk),
        "inter": (ref.kg, ref.w, ref.u, gk, q, ref.aqk, state),
        "chain": (q, k, v, beta, gk, state),
        "intra_reference": ref,
    }


def sequential_float64(flat: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """The delta rule one token at a time in float64 from the entering state.

    :func:`kda_sequential_torch_oracle`'s steps at float64 and from ``state``
    rather than zero: no cumulative gate is ever exponentiated, so it is a
    reference for every chunk width alike. Returns ``(o [T, V], final_state [V, K])``.
    """
    q, k, v, beta, gk, state = (x.double() for x in flat)
    kdim = q.shape[1]
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + live.L2_NORM_EPS) * kdim**-0.5
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + live.L2_NORM_EPS)
    o = torch.empty(q.shape[0], v.shape[1], dtype=torch.float64)
    for t in range(q.shape[0]):
        state = state * torch.exp(gk[t]).unsqueeze(0)
        delta = (v[t] - state @ kn[t]) * beta[t]
        state = state + torch.outer(delta, kn[t])
        o[t] = state @ qn[t]
    return o, state


def graph_fn(kernel: str, module, layers: int):
    """``layers`` independent calls of ``kernel`` from ``module``, one graph."""
    width = OPERANDS[kernel]

    def run(*args):
        outs = []
        for layer in range(layers):
            call = args[width * layer: width * (layer + 1)]
            if kernel == "intra":
                outs.extend(module.kda_intra_chunk(*call))
            elif kernel == "inter":
                outs.extend(module.kda_inter_chunk(*call[:6], state=call[6]))
            else:
                q, k, v, beta, gk, state = call
                intra = module.kda_intra_chunk(q, k, v, beta, gk)
                outs.extend(module.kda_inter_chunk(
                    intra.kg, intra.w, intra.u, gk, q, intra.aqk, state=state))
        return tuple(outs)

    return run


def floor_graph(*args):
    # The same operands and no kernel: the launch, the first-output copy and the
    # synchronisation every timed graph contains.
    return (args[0] + 0.0,)


def compiled(fn):
    return torch.compile(
        fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
        options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")},
    )


def to_device(tensors) -> tuple:
    return tuple(t.contiguous().to(DEVICE) for t in tensors)


def time_once(model, inputs, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        model(*inputs)[0].to("cpu")
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        model(*inputs)[0].to("cpu")
        samples.append((time.perf_counter_ns() - started) / 1_000)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def summarize_reps(per_rep: list[float]) -> dict:
    median = statistics.median(per_rep)
    return {
        "median_us": median,
        "min_us": min(per_rep),
        "max_us": max(per_rep),
        "spread": (max(per_rep) - min(per_rep)) / median,
        "per_rep_median_us": per_rep,
    }


def bench_shape(tokens: int, variants: list[Variant], args) -> dict:
    """Timed graphs for one call size: every kernel of every variant, and the floor.

    Each variant's graphs take operands at its own chunk width; every width
    regroups the same flat draws.
    """
    torch._dynamo.reset()
    layers = args.layers
    flats = [flat_inputs(tokens, args.timing_seed + layer) for layer in range(layers)]
    inputs = {}
    for chunk in sorted({variant.chunk for variant in variants}):
        cases = [make_inputs(flat, chunk) for flat in flats]
        for kernel in KERNELS:
            inputs[(kernel, chunk)] = to_device([t for c in cases for t in c[kernel]])
    graphs = {
        (kernel, variant.name): (compiled(graph_fn(kernel, variant.module, layers)),
                                 inputs[(kernel, variant.chunk)])
        for kernel in KERNELS for variant in variants
    }
    graphs[("floor", "")] = (compiled(floor_graph), inputs[("intra", variants[0].chunk)])
    label = {key: "floor" if key[0] == "floor" else f"{key[0]}_{key[1]}" for key in graphs}
    result = {"tokens": tokens, "layers": layers,
              "chunk": {variant.name: variant.chunk for variant in variants},
              "first_call_s": {}}
    for key, (model, operands) in graphs.items():
        started = time.perf_counter()
        first = model(*operands)[0].to("cpu")
        result["first_call_s"][label[key]] = time.perf_counter() - started
        if not torch.isfinite(first).all():
            raise AssertionError(f"{label[key]} at tokens={tokens} returned non-finite values")

    per_rep = {key: [] for key in graphs}
    raw = {label[key]: [] for key in graphs}
    for _ in range(args.reps):
        for key, (model, operands) in graphs.items():
            got = time_once(model, operands, args.warmup, args.iterations)
            per_rep[key].append(got["median_us"])
            raw[label[key]].append(got["samples_us"])
    summaries = {key: summarize_reps(values) for key, values in per_rep.items()}
    floor = summaries[("floor", "")]
    per_call = {kernel: {} for kernel in KERNELS}
    for (kernel, name), summary in summaries.items():
        if kernel == "floor":
            continue
        per_rep_us = [(value - floor_rep) / layers for value, floor_rep in
                      zip(summary["per_rep_median_us"], floor["per_rep_median_us"])]
        per_call[kernel][name] = {
            "median_us": (summary["median_us"] - floor["median_us"]) / layers,
            "per_rep_us": per_rep_us,
            "spread": (max(per_rep_us) - min(per_rep_us)) / statistics.median(per_rep_us),
        }
    speedup = {
        kernel: {name: per_call[kernel][BASELINE]["median_us"] / figure["median_us"]
                 for name, figure in per_call[kernel].items() if name != BASELINE}
        for kernel in KERNELS
    }
    result.update({
        "graphs_us": {label[key]: summary for key, summary in summaries.items()},
        "per_call_us": per_call,
        "speedup_vs_baseline": speedup,
        "noise_floor_spread": max(figure["spread"] for kernel in KERNELS
                                  for figure in per_call[kernel].values()),
        "graph_spread": {label[key]: summary["spread"] for key, summary in summaries.items()},
        "samples_us": raw,
    })
    return result


def bit_equal_summary(by_field: dict) -> bool | None:
    """Whether every field of one output compared bit-equal to the baseline's.

    ``None`` when the variant runs at another chunk width than the baseline, so no
    field was compared bit for bit.
    """
    flags = [f["bit_equal_to_baseline"] for f in by_field.values() if "bit_equal_to_baseline" in f]
    return None if None in flags else all(flags)


def tensor_diff(got: torch.Tensor, want: torch.Tensor) -> dict:
    got, want = got.double(), want.double()
    residual = got - want
    scale = want.abs().max()
    return {
        "max_abs": float(residual.abs().max()),
        "max_abs_over_scale": float(residual.abs().max() / scale) if scale > 0 else None,
        "rel_l2": float(residual.norm() / want.norm()) if want.norm() > 0 else None,
        "mismatched_elements": int((got != want).sum()),
    }


def numerics_shape(tokens: int, variants: list[Variant], args) -> list[dict]:
    """One-call graphs of every variant on ``--seeds`` x ``--gates`` operands; outputs saved."""
    torch._dynamo.reset()
    models = {
        (kernel, variant.name): compiled(graph_fn(kernel, variant.module, 1))
        for kernel in KERNELS for variant in variants
    }
    baseline = next(variant for variant in variants if variant.name == BASELINE)
    rows = []
    for gate in args.gates:
        for seed in args.seeds:
            flat = flat_inputs(tokens, seed, gate)
            cases = {chunk: make_inputs(flat, chunk)
                     for chunk in sorted({variant.chunk for variant in variants})}
            directory = args.numerics_dir / f"t{tokens}" / gate / f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            torch.save({"flat": flat, "chunks": {c: {kernel: case[kernel] for kernel in KERNELS}
                                                 for c, case in cases.items()}},
                       directory / "inputs.pt")
            outputs = {}
            for (kernel, name), model in models.items():
                chunk = next(v.chunk for v in variants if v.name == name)
                got = model(*to_device(cases[chunk][kernel]))
                outputs[(kernel, name)] = {f: t.to("cpu") for f, t in zip(FIELDS[kernel], got)}
                torch.save(outputs[(kernel, name)], directory / f"{kernel}_{name}.pt")

            scan_o, scan_state = sequential_float64(flat)
            reference = {}
            for chunk, case in cases.items():
                inter_ref = live.kda_inter_chunk_torch_oracle(*case["inter"][:6],
                                                              state=case["inter"][6])
                reference[chunk] = {"intra": case["intra_reference"], "inter": inter_ref,
                                    "chain": inter_ref}
            row = {"tokens": tokens, "gate": gate, "seed": seed, "dir": str(directory)}
            for kernel in KERNELS:
                row[kernel] = {}
                for variant in variants:
                    got = outputs[(kernel, variant.name)]
                    same_width = variant.chunk == baseline.chunk
                    row[kernel][variant.name] = {
                        f: {
                            "bit_equal_to_baseline": (
                                bool(torch.equal(got[f], outputs[(kernel, BASELINE)][f]))
                                if same_width else None),
                            "vs_baseline": (tensor_diff(got[f], outputs[(kernel, BASELINE)][f])
                                            if same_width else None),
                            "vs_reference": tensor_diff(
                                got[f], getattr(reference[variant.chunk][kernel], f)),
                        }
                        for f in FIELDS[kernel]
                    }
                    if kernel != "intra":
                        row[kernel][variant.name]["vs_float64_scan"] = {
                            "o": tensor_diff(got["o"].reshape(scan_o.shape), scan_o),
                            "final_state": tensor_diff(got["final_state"], scan_state),
                        }
            row["intra_inverse_residual_max_abs"] = {}
            for variant in variants:
                _, k, _, beta, gk = cases[variant.chunk]["intra"]
                i_plus_a = live.rebuild_i_plus_a(k, beta, gk)
                identity = torch.eye(variant.chunk).expand_as(i_plus_a)
                a_inv = outputs[("intra", variant.name)]["a_inv"]
                row["intra_inverse_residual_max_abs"][variant.name] = float(
                    (i_plus_a @ a_inv - identity).abs().max())
            rows.append(row)
    return rows


def _named_rev(text: str) -> tuple[str, str]:
    name, sep, rev = text.partition("=")
    if not sep or not name.isidentifier() or not rev or name in (BASELINE, CHECKOUT):
        raise argparse.ArgumentTypeError(
            f"expected NAME=REV with NAME an identifier other than {BASELINE!r} and "
            f"{CHECKOUT!r}, got {text!r}")
    return name, rev


def _require_fresh(path: Path, what: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{what} {path} is not empty; name a fresh directory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-rev", default=BASELINE_REV)
    parser.add_argument("--variant", type=_named_rev, action="append", default=[],
                        metavar="NAME=REV", help="another revision to time and check")
    parser.add_argument("--tokens", type=int, nargs="+", default=list(SERVED_PREFILL_TOKENS))
    parser.add_argument("--layers", type=int, default=KDA_LAYERS)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timing-seed", type=int, default=20261008)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--gates", nargs="+", choices=GATES, default=list(GATES))
    parser.add_argument("--numerics-dir", type=Path)
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run through the device lease, which pins NEURON_RT_VISIBLE_CORES")
    if args.reps < 1 or args.iterations < 1 or args.warmup < 0 or args.layers < 1:
        raise ValueError("Use positive reps, iterations and layers, nonnegative warmup")
    names = [name for name, _ in args.variant]
    if len(set(names)) != len(names):
        raise ValueError(f"variant names repeat: {names}")
    cache_root = os.environ.get("NEURON_LIBTORCH_CACHE_ROOT")
    if not cache_root:
        raise ValueError("Set NEURON_LIBTORCH_CACHE_ROOT to an empty or absent directory")
    _require_fresh(Path(cache_root), "NEURON_LIBTORCH_CACHE_ROOT")
    args.out = args.out.resolve()
    args.workdir = (args.workdir or args.out.parent / f"{args.out.stem}_workdir").resolve()
    args.numerics_dir = (
        args.numerics_dir or args.out.parent / f"{args.out.stem}_numerics"
    ).resolve()
    _require_fresh(args.workdir, "--workdir")
    args.workdir.mkdir(parents=True, exist_ok=True)
    variants = [load_revision(BASELINE, args.baseline_rev, args.workdir), checkout_variant()]
    variants += [load_revision(name, rev, args.workdir) for name, rev in args.variant]
    # Every kernel's graph of every variant is one more compile of the same code
    # object (``graph_fn``'s ``run``); past the recompile limit dynamo would run
    # that frame eagerly, untimed as a graph, so raise the limit and fail past it.
    torch._dynamo.config.recompile_limit = len(KERNELS) * len(variants)
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    # The kernel compiler writes its artifacts into the working directory; keep
    # them out of the checkout.
    os.chdir(args.workdir)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", f"{PACKAGE_DIR}/{KERNEL_LEAF}.py"],
        check=True, capture_output=True, text=True).stdout.strip()
    report = {
        "environment": {
            key: os.environ.get(key)
            for key in ("NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG",
                        "NEURON_CC_FLAGS", "NEURON_PLATFORM_TARGET_OVERRIDE",
                        "NEURON_LIBTORCH_CACHE_ROOT")
        },
        "tree": {"head": head, "kernel_file_dirty": bool(dirty)},
        "variants": {v.name: {"rev": v.rev, "sha256": v.sha256, "chunk": v.chunk}
                     for v in variants},
        "shape": {"kdim": SERVED_KDIM, "vdim": SERVED_VDIM},
        "timing_unit": (
            "microseconds per kernel call = (graph median - floor median) / layers; "
            "graph time is host wall time of one launch plus its first-output copy"
        ),
        "args": {k: str(v) for k, v in vars(args).items()},
        "timing": [],
        "numerics": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write() -> None:
        report["compiled_kernels"] = sorted(p.name for p in args.workdir.rglob("*.colz"))
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    for tokens in args.tokens:
        result = bench_shape(tokens, variants, args)
        report["timing"].append(result)
        write()
        print(json.dumps({"tokens": tokens, "chunk": result["chunk"], "per_call_us": {
            kernel: {name: figure["median_us"] for name, figure in by_name.items()}
            for kernel, by_name in result["per_call_us"].items()},
            "noise_floor_spread": result["noise_floor_spread"]}), flush=True)
    for tokens in args.tokens:
        rows = numerics_shape(tokens, variants, args)
        report["numerics"].extend(rows)
        write()
        for row in rows:
            print(json.dumps({
                "tokens": tokens, "gate": row["gate"], "seed": row["seed"],
                "bit_equal_to_baseline": {
                    kernel: {name: bit_equal_summary(by_field) for name, by_field in row[kernel].items()}
                    for kernel in KERNELS},
                "chain_o_vs_float64_scan": {
                    name: by_field["vs_float64_scan"]["o"]["max_abs_over_scale"]
                    for name, by_field in row["chain"].items()},
            }), flush=True)


if __name__ == "__main__":
    main()
