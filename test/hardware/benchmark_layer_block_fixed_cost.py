# SPDX-License-Identifier: Apache-2.0
"""One GLM-5.3-Flash layer block at a served prefill chunk, per layer family, on device.

Run it through the device lease, which pins the cores and the LNC; this script does not
select cores, and refuses to start without ``NEURON_RT_VISIBLE_CORES``.

What runs. ``benchmark_glue_block.py``'s in-graph layer benchmark (its graphs, variants,
interleaved rounds, profiles and output agreement), at prefill, for both layer families:

* ``kda:prefill:ROWS``: checkpoint layer 4 (linear attention + MoE) on one opening
  request's ``ROWS``-row chunk, with ``glue_case.kda_prefill_carriers`` (the case
  ``benchmark_glue_block.py`` already serves).
* ``dsa:prefill:ROWS[:LINE][@START]``: checkpoint layer 3 (sparse attention + MoE) on a
  chunk of ``ROWS`` real tokens that starts at position ``START`` (default 0) of one
  request, on the served line ``LINE`` (default ``std``, :data:`SERVED_LINES`). The
  carriers are the runner's own (``NeuronModelRunner``'s carrier helpers): per-row causal
  lengths, the pooled-key slot mapping, the latent slots, the block table padded to the
  chunk's window and the request's own pooled-key and ring rows. The window is what the
  runner hands that chunk: ``min(max_model_len, segment + ROWS)`` rows in pages, with
  ``segment`` the smallest KV segment bucket that holds the request (the chunk's end).
  On ``std`` a 1024-row chunk at position 0 has a 2048-row window, which the dense-window
  kernel serves; on ``8k`` every chunk has an 8192-row window and takes the sparse path.

Shapes are one rank's TP=64 / EP=16 shard (``glue_case``). Collectives: ``one-rank``
(default) makes both reductions of the layer HLO all-reduces over a one-rank world, so the
graph holds the served graph's two collectives; their time is not the served TP=64
all-reduce's (the payload is the same f32 ``[ROWS, hidden]``, but no peer ring runs).

What is measured is ``benchmark_glue_block.py``'s list (device ms per execution over
interleaved rounds, the A/A floor, one profile per variant, each variant's output against
``off``'s), per case. A variant whose name starts with a prefix of :data:`ROUTES` (for
example ``torchslots=1``) is traced and called with that site on its other route, so one run
holds a fold's before and after under the same glue switch value; a second such variant
(``torchslots_aa=1``) is that side's A/A floor:

* ``torchslots``: the MoE combine's slot build on its torch route
  (``combine_slots.combine_slots_torch``, the construction the kernel replaced);
* ``fusedrouter``: the prefill router on 8aa22fa's fused kernel
  (``router.noaux_tc_rmsnorm_router_topk``, its RMSNorm in the kernel) instead of
  ``router_prefill`` (its RMSNorm scale in XLA).

Each case also records whether each variant's first output is bit-equal to each such
variant's (``bit_equal_to_route_variants``). The per-op split of a profile is not done
here: give the compile a known working directory (``--work-dir``) and keep the compiler's
intermediates from it while the graphs compile; the op inventory's tools (graph map, debug
map, attribution) join a profile's instructions to the HLO ops and NKI calls through them.

The compile cache is used (``NEURON_LIBTORCH_CACHE_ROOT``); its keys fold the digest of the
kernel sources each graph reaches (``vllm_neuron.compile_cache_key``), so a kernel edit
recompiles the graphs that call it. A graph served from the cache leaves no compiler
intermediates, so a run whose profiles are to be split per op needs a fresh root.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

#: The worktree root, ahead of any installed copy (the lease command sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch

from test.hardware import benchmark_glue_block as block
from test.vllm_neuron.functional.dsa import dsa_batch_case
from test.vllm_neuron.functional.glue import glue_case
from vllm_neuron import envs
from vllm_neuron.functional.moe import combine_slots, router_prefill
from vllm_neuron.model.glm5_next import model_fp8 as live

DEVICE = block.DEVICE
CASES = ("kda:prefill:1024", "dsa:prefill:1024:std", "dsa:prefill:1024:8k@7168")


@dataclass(frozen=True)
class ServedLine:
    """The prefill geometry of one serving line: its model length and KV segment buckets."""

    max_model_len: int
    kv_segments: tuple[int, ...]


#: The serving lines a DSA case is built on. ``std``: the bs=1 line of the 1k TTFT gate
#: (``max_model_len`` 4096, ``kv_segment_size_buckets`` [1024, 2048, 4096]). ``8k``: the
#: line the 8000-token request is served on (``max_model_len`` 8192, one KV segment).
SERVED_LINES = {
    "std": ServedLine(max_model_len=4096, kv_segments=(1024, 2048, 4096)),
    "8k": ServedLine(max_model_len=8192, kv_segments=(8192,)),
}


def parse_case(text: str) -> dict:
    """``family:prefill:rows[:line][@start]`` -> the case dict ``benchmark_glue_block`` takes."""
    head, _, start = text.partition("@")
    family, phase, rows, *line = head.split(":")
    if family not in block.FAMILY_KERNELS or phase != "prefill" or int(rows) < 1:
        raise ValueError(f"case {text!r}: want kda|dsa : prefill : rows")
    if len(line) > 1 or (line and line[0] not in SERVED_LINES):
        raise ValueError(f"case {text!r}: the line is one of {sorted(SERVED_LINES)}")
    if family == "kda" and (line or start):
        raise ValueError(f"case {text!r}: a kda case is one opening request; it takes no "
                         f"line or start (its carriers do not depend on the context)")
    return {"family": family, "phase": phase, "rows": int(rows), "tag": text,
            "line": line[0] if line else "std", "start": int(start or 0)}


def dsa_window_pages(line: ServedLine, end: int, rows: int, page: int) -> int:
    """Pages of the block table the runner hands a chunk ending at ``end`` (its window).

    ``min(table width, ceil((segment + rows) / page))``: the segment is the smallest KV
    segment bucket at least as long as the request, else the largest; the table is as
    wide as ``max_model_len`` in pages.
    """
    segment = next((s for s in line.kv_segments if end <= s), line.kv_segments[-1])
    return min(-(-line.max_model_len // page), -(-(segment + rows) // page))


def dsa_prefill_carriers(layer_case, case: dict, *, seed: int = 37,
                         device: str = DEVICE) -> dict:
    """The runner's prefill carriers for one request's ``case['rows']``-row DSA chunk.

    The request's pages are the first pages of a latent bank with three spare pages and its
    side caches are slot 1 of a three-slot bank, so no other row is read as its own. The
    banks hold random bf16 values, as an earlier chunk would have left them.
    """
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner as runner

    cfg = layer_case.attention_cfg
    line = SERVED_LINES[case["line"]]
    rows, start, page = case["rows"], case["start"], glue_case.PAGE
    end = start + rows
    if end > line.max_model_len:
        raise ValueError(f"case {case['tag']!r}: the chunk ends at {end}, past the line's "
                         f"max_model_len {line.max_model_len}")
    pages = dsa_window_pages(line, end, rows, page)
    used = -(-end // page)
    gen = torch.Generator().manual_seed(int(seed) + end)
    pool, head_dim = int(cfg.index_kpool), int(cfg.index_head_dim)
    pool_rows = dsa_batch_case.pool_rows(line.max_model_len, pool)
    slot = 1
    banks = {
        "latent_cache": (torch.randn((pages + 3) * page, 1, int(cfg.kv_lora_rank),
                                     generator=gen) * 0.5).to(torch.bfloat16),
        "pool_bank": (torch.randn(3, pool_rows, head_dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
        "tail_bank": (torch.randn(3, 2, pool, head_dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
    }
    banks = {k: v.to(device) for k, v in banks.items()}
    ids = list(range(used))
    table = torch.tensor([[one] for one in ids] + [[-1]] * (pages - used), dtype=torch.int32)
    request = [(rows, start)]
    return {
        "banks": tuple(banks.values()),
        "latent_cache": banks["latent_cache"],
        "pool_cache": banks["pool_bank"][slot],
        "prefill_tail": banks["tail_bank"][slot],
        "seq_lens": runner._glm5next_batch_row_seq_lens(request, device=device),
        "start_position": runner._glm5next_start_position(start, device),
        "slot_mapping": runner._glm5next_batch_pool_slot_mapping(
            request, index_kpool=pool, device=device, real_tokens=[rows]),
        "prefill_end_position": runner._glm5next_start_position(end, device),
        "block_table_row": table.to(device),
        "latent_slots": runner._glm5next_latent_slot_mapping(
            rows=[ids], starts=[start], tokens=rows, block_size=page, reals=[rows],
            device=device),
        "softmax_scale": float(int(cfg.qk_nope_head_dim) + int(cfg.qk_rope_head_dim)) ** -0.5,
        "max_seq_len": int(line.max_model_len),
        "page_size": page,
    }


def carriers_for(case: dict, layer_case, device: str = DEVICE) -> dict:
    """Prefill carriers of either family (``benchmark_glue_block.carriers_for``'s role)."""
    if case["family"] == "dsa":
        return dsa_prefill_carriers(layer_case, case, device=device)
    return glue_case.kda_prefill_carriers(layer_case, case["rows"], device=device)


class ServedRankGroup(block._OneRankGroup):
    """``benchmark_glue_block``'s one-rank group, as the last rank of the served group.

    The DSA indexer divides its prefill selection over the tensor-parallel ranks
    (``vllm_neuron/functional/dsa/indexer_shard.py``): rank ``rank_in_group`` of
    ``world_size`` selects ``ceil(rows / world_size)`` of the chunk's rows, and the
    group's ``all_gather`` puts every rank's rows back in order. This group answers as
    rank ``world_size - 1`` of ``glue_case.TP_WORLD``, so the selection runs on the rows
    one served rank selects. Its ``all_gather`` is no collective: it repeats this rank's
    rows once per rank, a device copy of the gathered size. The ids every row then
    attends with are those of the chunk's last rows, which see the most pooled keys, so
    each row selects as many real pools as the served chunk's last rows (a full top-k
    once the context holds that many pools); the values are not the served ones.

    A chunk whose selection is a no-op (the dense-window path) selects nothing, and the
    group gives the indexer only its rank at load.
    """

    world_size = glue_case.TP_WORLD
    rank_in_group = glue_case.TP_WORLD - 1

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.cat([tensor] * self.world_size, dim=dim)


#: Variant-name prefixes that put one site on its other route while the variant is traced
#: and called: prefix -> (module, predicate). The predicate answers False meanwhile.
ROUTES = {
    "torchslots": (combine_slots, "can_run_combine_slots"),
    "fusedrouter": (router_prefill, "prefill_router_enabled"),
}

#: The route prefix of the variant being traced or called, or None. The trace reads it
#: (dynamo guards on its value), so every call sets it to its variant's value first.
_ROUTE = [None]


def route_of(name: str) -> str | None:
    """The :data:`ROUTES` prefix ``name`` starts with, or None."""
    return next((prefix for prefix in ROUTES if name.startswith(prefix)), None)


def _routed_predicate(prefix: str, predicate):
    """``predicate``, False while a variant of route ``prefix`` runs."""

    def admits(*args):
        return _ROUTE[0] != prefix and predicate(*args)

    return admits


def _route_build(build_variant, routes: dict, firsts: dict):
    """``build_variant`` that traces and calls each variant on the route ``routes`` gives
    its tag (None: the tree's own), and keeps each variant's first output in ``firsts``."""

    def build(case, layers, spec, tag, args):
        route = routes.get(tag)
        _ROUTE[0] = route
        call, *rest = build_variant(case, layers, spec, tag, args)
        firsts[tag] = rest[1]

        def routed_call(*inputs):
            _ROUTE[0] = route
            return call(*inputs)

        return (routed_call, *rest)

    return build


def run_case(case: dict, collectives: str, specs: dict, args) -> dict:
    """``benchmark_glue_block.run_case`` with this module's prefill carriers.

    That function builds every variant's operands through its module's ``carriers_for``,
    which serves decode and KDA prefill only, so the DSA prefill carriers are handed to it
    for the duration of the case; a DSA case also runs with :class:`ServedRankGroup` as
    its one-rank group. A variant named for a :data:`ROUTES` prefix runs on that route.
    """
    prefix = block.case_key(case, collectives, args)
    tags = {name: f"{prefix}_{name}" for name in specs}
    routed = [name for name in specs if route_of(name)]
    firsts: dict = {}
    with contextlib.ExitStack() as patches:
        patches.enter_context(mock.patch.object(block, "carriers_for", carriers_for))
        if case["family"] == "dsa":
            patches.enter_context(mock.patch.object(block, "_OneRankGroup", ServedRankGroup))
        patches.enter_context(mock.patch.object(
            block, "build_variant",
            _route_build(block.build_variant, {tags[n]: route_of(n) for n in routed}, firsts)))
        for route, (module, predicate) in ROUTES.items():
            patches.enter_context(mock.patch.object(
                module, predicate, _routed_predicate(route, getattr(module, predicate))))
        result = block.run_case(case, collectives, specs, args)
    result["bit_equal_to_route_variants"] = {
        ref: {name: bool(torch.equal(firsts[tags[name]], firsts[tags[ref]])) for name in specs}
        for ref in routed}
    if case["family"] == "dsa":
        line = SERVED_LINES[case["line"]]
        result["window_rows"] = glue_case.PAGE * dsa_window_pages(
            line, case["start"] + case["rows"], case["rows"], glue_case.PAGE)
        result["max_model_len"] = line.max_model_len
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--variants", nargs="*", default=["aa=0", "default"],
                        help="benchmark_glue_block.py's variants; off is always added; a "
                             "name starting with a ROUTES prefix runs that site's other "
                             "route")
    parser.add_argument("--collectives", nargs="+", default=["one-rank"],
                        choices=block.COLLECTIVES)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--profile-dir", type=Path, default=None,
                        help="where the device profiles go (required unless --no-profile)")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--save-io", type=Path, default=None,
                        help="save each case's operands, weights and first outputs here")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="the compilers' working directory (default: a new temporary "
                             "directory); they write their intermediates here")
    parser.add_argument("--merge", action="store_true",
                        help="keep the cases already in --output that this run does not redo")
    parser.add_argument("--compiler-args", nargs="*",
                        default=(os.environ.get("NEURON_CC_FLAGS", "").split()
                                 or block.micro.MODEL_COMPILER_ARGS))
    args = parser.parse_args()
    # benchmark_glue_block.run_case's settings that this benchmark keeps fixed: one layer
    # per graph and one all-reduce per reduction site.
    args.stack, args.ar_chain = 1, [1, 1]
    if not args.no_profile and args.profile_dir is None:
        parser.error("--profile-dir is required unless --no-profile")
    for name in ("output", "profile_dir", "save_io", "work_dir"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.no_profile:
        args.profile_dir = None
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Set NEURON_RT_VISIBLE_CORES to the cores this benchmark may use; "
                         "it does not select cores itself")
    if not Path(live.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(f"model_fp8 imported from {live.__file__}, not {ROOT}")
    plans = [(case, block.variant_specs(case, args.variants))
             for case in (parse_case(c) for c in args.cases)]
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="layer-block-bench-cwd-"))
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    report = {
        "environment": {k: os.environ.get(k) for k in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_LIBTORCH_CACHE_ROOT")},
        "device": "neuron:0 (one logical core = 2 physical cores at LNC2)",
        "compiler_args": args.compiler_args,
        "compiler_work_dir": str(work),
        "default_spec": envs.DEFAULT_GLUE_FUSED_SPEC,
        "shapes": block._shapes(),
        "served_lines": {k: {"max_model_len": v.max_model_len,
                             "kv_segments": list(v.kv_segments)}
                         for k, v in SERVED_LINES.items()},
        "rounds": args.rounds, "warmup": args.warmup, "iterations": args.iterations,
        "profile_iterations": args.profile_iterations,
        "cases": [],
    }
    if args.merge and args.output.exists():
        redo = {block._summary_key({"case": c["tag"], "collectives": m, "stack": 1,
                                    "ar_chain": [1, 1] if m == "one-rank" else None})
                for c, _ in plans for m in args.collectives}
        report["cases"] = [c for c in json.loads(args.output.read_text()).get("cases", [])
                           if block._summary_key(c) not in redo]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    block._record_graph_keys()
    report["environment"]["cache_entries_at_start"] = block.cache_entries()
    if "one-rank" in args.collectives:
        block.init_one_rank_world()
    for case, specs in plans:
        for collectives in args.collectives:
            t0 = time.time()
            result = run_case(case, collectives, specs, args)
            result["wall_s"] = time.time() - t0
            report["cases"].append(result)
            report["summary"] = block.summarise(report["cases"])
            args.output.write_text(json.dumps(report, indent=1) + "\n")
            key = block._summary_key(result)
            print(json.dumps({key: {n: {k: r.get(k) for k in (
                "median_ms", "spread_ms", "delta_median_ms", "verdict")}
                for n, r in report["summary"][key].items()}}), flush=True)


if __name__ == "__main__":
    main()
