# SPDX-License-Identifier: Apache-2.0
"""Readers of the wave-1 hardware microbenchmark JSONs (written new for the ledger).

Each family file (``reports/{mhc,kda,dsa,moe-t,dense,host_sampler}_micro.json``) holds
"before" (the 5938748 kernel, from a ``git show`` snapshot) and "after" (the worker's
kernel) medians in microseconds at the real decode shapes. The readers turn every case
into ``KernelResult``s keyed ``(config_name, variant)``. ``config_name`` is built from the
shapes the benchmark recorded (``MATCH_KEYS``), so the graph finds a measurement only
when its own shapes match the benchmark's; a shape drift shows as "no measurement".

What each reader takes (one rank's shard, one decode call):
  - mHC: ``device_per_call`` of sinkhorn and of combine (hyper_connection), per site.
  - KDA: ``summary["B=n"]``: per layer in the 34-layer graph (the headline reading).
  - DSA: ``layer`` table: one ``Glm5NextMLAAttention.forward``; ``window_rows`` of each
    variant goes into its config name. Cases that time the same config are averaged.
  - MoE: router medians (bias chain, the worker's headline); routed-expert medians as a
    curve over the number of distinct local experts the case fixed.
  - dense: ``per_site_median_us`` (shared expert MLP, dense MLP, lm_head GEMV, RMSNorm).
  - sampler: before = the host path (``before_host_path``, off the device), after =
    the on-device greedy sampler (``after_device_sampler``).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

from test.vllm_neuron import artifacts

from .emf import KernelResult

#: The team's reports directory (read-only input), below the campaign directory.
REPORTS_DIR = artifacts.campaign_path("glm53f-wt", "reports")

MICRO_FILES = {
    "mhc": "mhc_micro.json",
    "kda": "kda_micro.json",
    "dsa": "dsa_micro.json",
    "moe": "moe-t_micro.json",
    "dense": "dense_micro.json",
    "sampler": "host_sampler_micro.json",
}

#: More mHC batches (B=8, 16, 32 and B=64, 128), same method as ``mhc_micro.json``; read if present.
MHC_BATCH_FILES = ("mhc_micro_mid.json", "mhc_micro_large.json")


def mhc_batch_files(reports_dir: Path = REPORTS_DIR) -> List[str]:
    return [f for f in MHC_BATCH_FILES if (Path(reports_dir) / f).is_file()]


#: Shape fields that identify a measured config, per kernel (KernelName values).
MATCH_KEYS: Dict[str, Tuple[str, ...]] = {
    "mhc_sinkhorn_tkg": ("B", "hidden", "streams", "iters"),
    "mhc_combine_tkg": ("B", "hidden", "streams"),
    "kda_decode_tkg": ("B", "heads_per_rank", "head_dim", "conv_taps", "conv_channels"),
    "dsa_mla_layer_tkg": ("batch", "ctx", "max_seq_len", "window_rows"),
    "rmsnorm_router_topk_tkg": ("T", "H", "E_global", "top_k"),
    "moe_experts_tkg": ("T", "H", "E_global", "top_k", "E_local", "I_local"),
    "shared_expert_tkg": ("M", "H", "I"),
    "dense_mlp_tkg": ("M", "H", "I"),
    "lm_head": ("M", "H", "vocab", "shard_rows"),
    "rmsnorm_tkg": ("M", "H"),
    "sampling": ("B", "vocab", "logits_dtype", "mode"),
}

VARIANTS = ("before", "after")


def config_name(kernel: str, record: Dict) -> str:
    """``kernel|k1=v1,k2=v2`` over the kernel's match keys (all keys if unknown)."""
    keys = MATCH_KEYS.get(kernel, tuple(record))
    missing = [k for k in keys if k not in record]
    if missing:
        raise ValueError(f"{kernel}: shape record lacks {missing}: {record}")
    return kernel + "|" + ",".join(f"{k}={record[k]}" for k in sorted(keys))


_CARRIER = re.compile(r"^\s*(\w+)\s*\[([^\]]*)\]")


def parse_carrier(text: str, batch: int) -> Dict:
    """``"bfloat16 [B, 3, 384] (SD)"`` -> ``{"dtype": "bfloat16", "shape": [batch, 3, 384]}``."""
    m = _CARRIER.match(text)
    if not m:
        raise ValueError(f"cannot parse carrier description {text!r}")
    dims = [d.strip() for d in m.group(2).split(",")]
    return {"dtype": m.group(1), "shape": [batch if d == "B" else int(d) for d in dims]}


class _Collector:
    """Gathers samples per ``(config_name, variant)``. Cases of one config are averaged,
    except cases marked ``repeat``: those are kept in ``repeats`` and not used (unless no
    other case exists)."""

    def __init__(self):
        self._rows: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    def add(self, kernel: str, variant: str, record: Dict, median: float, *, p90: Optional[float],
            iterations: Optional[int], source: str, location: str = "device", points=None, repeat: bool = False):
        name = config_name(kernel, record)
        self._rows[(name, variant)].append(dict(
            kernel=kernel, record=record, median=median, p90=p90, iterations=iterations,
            source=source, location=location, points=points, repeat=repeat,
        ))

    def results(self) -> Dict[Tuple[str, str], KernelResult]:
        out = {}
        for (name, variant), all_rows in self._rows.items():
            rows = [r for r in all_rows if not r["repeat"]] or all_rows
            first = rows[0]
            if any(r["location"] != first["location"] for r in rows):
                raise ValueError(f"{name}/{variant}: cases disagree on location")
            p90s = [r["p90"] for r in rows if r["p90"] is not None]
            its = [r["iterations"] for r in rows if r["iterations"] is not None]
            out[(name, variant)] = KernelResult(
                config_name=name,
                kernel_api=first["kernel"],
                latency_us=mean(r["median"] for r in rows),
                test_name=first["source"],
                variant=variant,
                p90_us=mean(p90s) if p90s else None,
                iterations=min(its) if its else None,
                location=first["location"],
                record=first["record"],
                points=first["points"],
                sources=tuple(r["source"] for r in rows),
                repeats=tuple((r["source"], r["median"]) for r in all_rows if r not in rows),
            )
        return out


def _stat(block: Dict, median="median_us", p90="p90_us", its="iterations"):
    return block[median], block.get(p90), block.get(its)


def _read_mhc(path: Path, c: _Collector):
    for case in json.loads(path.read_text())["cases"]:
        rec = {"B": case["B"], "hidden": case["hidden"], "streams": case["streams"], "iters": case["iters"]}
        comb = {k: rec[k] for k in ("B", "hidden", "streams")}
        for v in VARIANTS:
            for kernel, part, r in (("mhc_sinkhorn_tkg", "sinkhorn", rec), ("mhc_combine_tkg", "combine", comb)):
                med, p90, its = _stat(case[v][part]["device_per_call"])
                c.add(kernel, v, r, med, p90=p90, iterations=its, source=f"{path.name}#B{case['B']}/{part}")


def _read_kda(path: Path, c: _Collector):
    d = json.loads(path.read_text())
    s = d["shapes"]
    for key, summ in d["summary"].items():
        b = int(key.split("=")[1])
        layers = summ["layers_in_graph"]
        its = {v: case["variants"][v]["timing"]["iterations"]
               for case in d["cases"] if case["B"] == b and case["layers"] == layers for v in VARIANTS}
        rec = {"B": b, "heads_per_rank": s["heads_per_rank"], "head_dim": s["head_dim"],
               "conv_taps": s["conv_taps"], "conv_channels": s["conv_channels"],
               "conv_carrier": parse_carrier(s["conv_carrier"], b),
               "recurrent_carrier": parse_carrier(s["recurrent_carrier"], b)}
        for v in VARIANTS:
            c.add("kda_decode_tkg", v, rec, summ[v]["median_us"], p90=summ[v]["p90_us"],
                  iterations=its.get(v), source=f"{path.name}#summary/{key}/L{layers}")


#: (case, variant) of the DSA layer table that repeat another case at the same config. The
#: bypass_vs_default_window case times "before" at the 5938748 window (4096 rows); its
#: "after" (2048 rows) repeats the "bypass" case, the one dsa.md uses for its headline
#: (286.6 us) and its savings.
DSA_REPEATS = {("bypass_vs_default_window", "after")}


def _read_dsa(path: Path, c: _Collector):
    for case in json.loads(path.read_text())["cases"]:
        if case.get("table") != "layer":
            continue
        for v in VARIANTS:
            rec = {"batch": case["batch"], "ctx": case["ctx"], "max_seq_len": case["max_seq_len"],
                   "window_rows": case["window_rows"][v]}
            med, p90, its = _stat(case[v])
            c.add("dsa_mla_layer_tkg", v, rec, med, p90=p90, iterations=its,
                  source=f"{path.name}#layer/{case['case']}/B{case['batch']}",
                  repeat=(case["case"], v) in DSA_REPEATS)


def _read_moe(path: Path, c: _Collector):
    d = json.loads(path.read_text())
    s = d["shapes"]
    curves: Dict[Tuple[int, str], List[tuple]] = defaultdict(list)
    for case in d["cases"]:
        t = case["T"]
        if case["family"] == "router":
            rec = {"T": t, "H": s["H"], "E_global": s["E_global"], "top_k": s["top_k"]}
            for v in VARIANTS:
                if v in case:
                    med, p90, its = _stat(case[v])
                    c.add("rmsnorm_router_topk_tkg", v, rec, med, p90=p90, iterations=its,
                          source=f"{path.name}#router/T{t}")
        elif case["family"] == "experts":
            distinct = case["distinct_local_experts_per_layer"]
            if len(set(distinct)) != 1:
                raise ValueError(f"{path.name}: experts case {case['scenario']} varies its hit count")
            for v in VARIANTS:
                if v in case:
                    curves[(t, v)].append((distinct[0], case[v], case["scenario"]))
    for (t, v), pts in curves.items():
        pts.sort(key=lambda p: p[0])
        rec = {"T": t, "H": s["H"], "E_global": s["E_global"], "top_k": s["top_k"],
               "E_local": s["E_local"], "I_local": s["I_local"]}
        med, p90, its = _stat(pts[0][1])
        c.add("moe_experts_tkg", v, rec, med, p90=p90, iterations=its,
              source=f"{path.name}#experts/T{t}/" + "+".join(p[2] for p in pts),
              points=tuple((p[0], p[1]["median_us"]) for p in pts))


_DENSE_CASES = (
    ("shared_mlp_", "shared_expert_tkg", ("M", "H", "I")),
    ("dense_mlp_", "dense_mlp_tkg", ("M", "H", "I")),
    ("lm_head_", "lm_head", ("M", "H", "vocab", "shard_rows")),
    ("norm_", "rmsnorm_tkg", ("M", "H")),
)


def _read_dense(path: Path, c: _Collector):
    for case in json.loads(path.read_text())["cases"]:
        for prefix, kernel, keys in _DENSE_CASES:
            if case["case"].startswith(prefix):
                rec = {k: case[k] for k in keys}
                for v in VARIANTS:
                    med, p90, its = _stat(case[v], "per_site_median_us", "per_site_p90_us")
                    c.add(kernel, v, rec, med, p90=p90, iterations=its, source=f"{path.name}#{case['case']}")
                break


def _read_sampler(path: Path, c: _Collector):
    for case in json.loads(path.read_text())["cases"]:
        rec = {"B": case["B"], "vocab": case["vocab"], "logits_dtype": case["logits_dtype"], "mode": case["mode"]}
        src = f"{path.name}#B{case['B']}/{case['mode']}"
        med, p90, its = _stat(case["before_host_path"])
        c.add("sampling", "before", rec, med, p90=p90, iterations=its, source=src, location="host")
        med, p90, its = _stat(case["after_device_sampler"])
        c.add("sampling", "after", rec, med, p90=p90, iterations=its, source=src, location="device")


_READERS = {"mhc": _read_mhc, "kda": _read_kda, "dsa": _read_dsa, "moe": _read_moe,
            "dense": _read_dense, "sampler": _read_sampler}


def missing_micro_files(reports_dir: Path = REPORTS_DIR) -> List[str]:
    return [f for f in MICRO_FILES.values() if not (Path(reports_dir) / f).is_file()]


def load_micro_results(reports_dir: Path = REPORTS_DIR) -> Dict[Tuple[str, str], KernelResult]:
    """Read every family file present in ``reports_dir``; absent files are skipped
    (``missing_micro_files`` names them), an absent directory is an error."""
    reports_dir = Path(reports_dir)
    if not reports_dir.is_dir():
        raise FileNotFoundError(f"reports directory not found: {reports_dir}")
    c = _Collector()
    for family, fname in MICRO_FILES.items():
        path = reports_dir / fname
        if path.is_file():
            _READERS[family](path, c)
    for fname in mhc_batch_files(reports_dir):
        _read_mhc(reports_dir / fname, c)
    return c.results()
