# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode cells file: the 12 measured shapes, the forms that serve them, their limits.

Set ``KIMI_K3_ENTITLEMENT`` to the planner's entitlement file, ``KIMI_K3_COST_ROWS`` to the
cost rows file and ``KIMI_K3_DECODE_PLAN`` to the decode increment plan, whose served-forms
and slots tables name each cell's form. Set ``KIMI_K3_CHECKPOINT`` to the
checkpoint directory (default ``~/models/moonshotai/Kimi-K3``). Run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.bucket_utils import (
    get_default_kv_segment_size_buckets,
    validate_decode_context_length_buckets,
    validate_kv_segment_size_buckets,
)

SERVING = Path(__file__).resolve().parent
CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)
CONTEXTS = (2048, 8192, 32768, 131072)
USERS = (1, 8, 32)


def _cells():
    return json.loads((SERVING / "cells.json").read_text(encoding="utf-8"))


def _source(name):
    path = os.environ.get(name)
    assert path, f"set {name}"
    return Path(path).read_text(encoding="utf-8")


def _table(text, *headers):
    """The rows of the first markdown table whose header holds every name in ``headers``."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and all(h in cols for h in headers):
            rows = []
            for row in lines[i + 2:]:
                if not row.lstrip().startswith("|"):
                    break
                rows.append(dict(zip(cols, (c.strip() for c in row.strip().strip("|").split("|")))))
            return rows
    raise AssertionError(f"no table with columns {headers}")


def _served_forms(plan):
    """Map each cell name, such as ``8K-32``, to the (DCP size, replica slots) that serve it."""
    slots = {r["Form"]: int(r["Slots"]) for r in _table(plan, "Form", "Slots")}
    forms = {}
    for row in _table(plan, "Form", "Cells it serves"):
        for part in row["Cells it serves"].split(";"):
            context, _, users = part.strip().partition(" at ")
            for c in re.findall(r"\d+", users):
                forms[f"{context}-{c}"] = (int(row["Form"].split()[1]), slots[row["Form"]])
    return forms


def test_cells_agree_with_their_sources():
    """Item 1: 12 cells; floor from the entitlement file, floor-based limit 2.75x the floor,
    limit of record from the cost rows, form from the served-forms table: 0 cells differ."""
    doc = _cells()
    entitlement = json.loads(_source("KIMI_K3_ENTITLEMENT"))
    recost = json.loads(_source("KIMI_K3_COST_ROWS"))["recost"]
    served = _served_forms(_source("KIMI_K3_DECODE_PLAN"))
    floors = {(r["S"], r["c"]): r["step_floor_ms"] for r in entitlement["decode"]
              if r["variant"] == "mx"}
    assert sorted(floors) == [(s, c) for s in CONTEXTS for c in USERS]
    assert sorted((c["S"], c["concurrency"]) for c in doc["cells"]) == sorted(floors)
    differ = {}
    for cell in doc["cells"]:
        floor = floors[cell["S"], cell["concurrency"]]
        row = recost[f"{cell['S'] // 1024}K c{cell['concurrency']}"]
        form = doc["forms"][cell["form"]]
        checks = {
            "floor": cell["floor_ms"] == floor,
            "floor-based limit": abs(cell["floor_based_limit_ms"] - 2.75 * floor) <= 1e-9,
            "limit of record": cell["limit_of_record_ms"] == row["provisional_limit_ms"],
            "form": (form["decode_context_parallel_size"], form["expert_replica_slots"])
            == served[cell["name"]],
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            differ[cell["name"]] = bad
    assert len(doc["cells"]) == 12 and differ == {}


def _engine_config(flags):
    """The engine config ``vllm serve`` builds from the flags, after the block-size step."""
    args = make_arg_parser(FlexibleArgumentParser()).parse_args([str(CHECKPOINT), *flags])
    args.model = args.model_tag
    validate_parsed_serve_args(args)
    config = AsyncEngineArgs.from_cli_args(args).create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER)
    current_platform.update_block_size_for_backend(config)
    return config


def _form_reading(form, measurements):
    """What the form's serve flags give: the config's values, checked by the DCP check and the
    bucket validators as the worker runs them."""
    flags = [f.replace("{checkpoint}", str(CHECKPOINT)).replace("{measurements}", str(measurements))
             for f in form["server_flags"]]
    config = _engine_config(flags)
    current_platform._validate_dcp_config(config)
    neuron = config.additional_config["neuron_config"]
    max_len = config.model_config.max_model_len
    scheduler = config.scheduler_config
    if "kv_segment_size_buckets" in neuron:
        segments = validate_kv_segment_size_buckets(neuron["kv_segment_size_buckets"])
    else:
        segments = get_default_kv_segment_size_buckets(scheduler.max_num_batched_tokens, max_len)
    dcp = config.parallel_config.decode_context_parallel_size
    threshold = scheduler.long_prefill_token_threshold
    return {
        "trust_remote_code": config.model_config.trust_remote_code,
        "decode_context_parallel_size": dcp,
        "expert_replica_slots": NeuronConfig.from_dict(neuron).expert_replica_slots,
        "max_model_len": max_len,
        "decode_context_length_buckets": validate_decode_context_length_buckets(
            neuron["decode_context_length_buckets"], max_len),
        "kv_segment_size_buckets": segments,
        "threshold_divides": threshold > 0 and threshold % (dcp * config.cache_config.block_size) == 0,
    }


def test_each_form_builds_a_valid_config(tmp_path):
    """Item 2: each form's serve flags build a config in CPU mode, with no workers, that passes
    the DCP check and the bucket validator, with remote code trusted, the form's slots,
    max-model-len and buckets, segmented prefill on and the prefill threshold a multiple of
    DCP times the block size: 0 raised errors over the 3 forms."""
    doc = _cells()
    errors = {}
    for name, form in doc["forms"].items():
        want = {
            "trust_remote_code": True,
            "decode_context_parallel_size": form["decode_context_parallel_size"],
            "expert_replica_slots": form["expert_replica_slots"],
            "max_model_len": form["max_model_len"],
            "decode_context_length_buckets": form["decode_context_length_buckets"],
            "kv_segment_size_buckets": [form["kv_segment_size"]],
            "threshold_divides": True,
        }
        try:
            got = _form_reading(form, tmp_path)
            assert got == want, {k: got[k] for k in want if got[k] != want[k]}
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    assert sorted(doc["forms"]) == ["dcp16", "dcp32", "dcp4"] and errors == {}


def _runs(dst, cell, gap_ms, non_finite=None):
    """Write one cell's 6 timed result files, processes A and B, runs 1 to 3, every gap gap_ms;
    ``itls`` puts a NaN gap in A/1, ``ttfts`` an infinite first-token time in B/2."""
    n = 8 if cell["concurrency"] == 1 else cell["concurrency"]
    dst.mkdir()
    for process in "AB":
        for run in (1, 2, 3):
            itls = [[gap_ms / 1000.0] * 127 for _ in range(n)]
            ttfts = [1.0] * n
            if non_finite == "itls" and (process, run) == ("A", 1):
                itls[0][3] = float("nan")
            if non_finite == "ttfts" and (process, run) == ("B", 2):
                ttfts[n - 1] = float("inf")
            result = {"num_prompts": n, "completed": n, "failed": 0,
                      "input_lens": [cell["S"] - 128] * n, "output_lens": [128] * n,
                      "ttfts": ttfts, "itls": itls, "process": process, "run": str(run)}
            (dst / f"{process}-{run}.json").write_text(json.dumps(result))
    return dst


def test_reducer_reads_each_cell_limit(tmp_path):
    """Item 3: at each cell's limit of record, flat requests exit 0 and 1 % over exit 1; a NaN
    gap or an infinite first-token time exits 3, not adjudicable; all 12 cells."""
    got, want = {}, {}
    for cell in _cells()["cells"]:
        name, limit = cell["name"], cell["limit_of_record_ms"]
        got[name] = []
        for case, gap_ms, non_finite in (("at", limit, None), ("over", 1.01 * limit, None),
                                         ("nan", limit, "itls"), ("inf", limit, "ttfts")):
            path = _runs(tmp_path / f"{name}-{case}", cell, gap_ms, non_finite)
            r = subprocess.run([sys.executable, "-B", str(SERVING / "steady_tpot.py"), str(path),
                                "--cell", name], capture_output=True, text=True)
            record = json.loads(r.stdout)
            got[name].append((r.returncode, record["limit_of_record_ms"], record["result"]))
        want[name] = [(0, limit, "pass"), (1, limit, "fail"), (3, limit, "not adjudicable"),
                      (3, limit, "not adjudicable")]
    assert len(got) == 12 and got == want
