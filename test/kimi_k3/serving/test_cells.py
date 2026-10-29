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
import types
from pathlib import Path

import pytest
import torch
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.structured_output import StructuredOutputManager

from vllm_neuron.model.kv_cache import KeyOnlyAttentionSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.bucket_utils import (
    get_default_kv_segment_size_buckets,
    validate_decode_context_length_buckets,
    validate_kv_segment_size_buckets,
)
from vllm_neuron.vllm.core.scheduler import NeuronScheduler
from vllm_neuron.vllm.worker import neuron_model_runner
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

SERVING = Path(__file__).resolve().parent
CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)
CONTEXTS = (2048, 8192, 32768, 131072)
USERS = (1, 8, 32)
# Per served form: prefill chunk, prefill reserve (GiB), KV floor (GiB) and the decode context
# buckets the cells use. Both forms compile the batch buckets the cells use.
SERVED = {"dcp4": (1024, 5.31, 0.54, [2048, 8192]), "dcp16": (1024, 5.42, 0.54, [8192, 45056])}
SERVED_BATCHES = [1, 4, 16, 32]


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
        "threshold_on_stride": threshold > 0
        and threshold % current_platform.dcp_prefill_stride(config) == 0,
    }


def _at_chunk(form, chunk):
    """The form with its prefill chunk, long-prefill threshold and KV segment at ``chunk``."""
    flags = list(form["server_flags"])
    for name in ("--max-num-batched-tokens", "--long-prefill-token-threshold"):
        flags[flags.index(name) + 1] = str(chunk)
    return {**form, "server_flags": flags, "kv_segment_size": chunk}


@pytest.mark.parametrize("chunk", [None, 512, 1024, 2048])
def test_each_form_builds_a_valid_config(tmp_path, chunk):
    """Item 2: each form's serve flags build a config in CPU mode, with no workers, that passes
    the DCP check and the bucket validator, with remote code trusted, the form's slots,
    max-model-len and buckets, segmented prefill on and the prefill threshold a multiple of
    the platform's DCP prefill stride: 0 raised errors over the 3 forms as they stand, and
    over dcp16 and dcp32 with their prefill chunk at ``chunk``."""
    doc = _cells()
    forms = doc["forms"] if chunk is None else {
        name: _at_chunk(doc["forms"][name], chunk) for name in ("dcp16", "dcp32")}
    errors = {}
    for name, form in forms.items():
        want = {
            "trust_remote_code": True,
            "decode_context_parallel_size": form["decode_context_parallel_size"],
            "expert_replica_slots": form["expert_replica_slots"],
            "max_model_len": form["max_model_len"],
            "decode_context_length_buckets": form["decode_context_length_buckets"],
            "kv_segment_size_buckets": [form["kv_segment_size"]],
            "threshold_on_stride": True,
        }
        try:
            got = _form_reading(form, tmp_path)
            assert got == want, {k: got[k] for k in want if got[k] != want[k]}
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    assert sorted(doc["forms"]) == ["dcp16", "dcp32", "dcp4"] and errors == {}


def _kv_cache_config(config):
    """vLLM's KV cache config of 64 blocks over the form's layers: one latent group, three state."""
    cache, text = config.cache_config, config.model_config.hf_text_config
    kda = text.linear_attn_config
    shapes = MambaStateShapeCalculator.kda_state_shape(
        config.parallel_config.tensor_parallel_size, kda["num_heads"], kda["head_dim"],
        conv_kernel_size=kda["short_conv_kernel_size"])
    dtypes = MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, cache.mamba_cache_dtype)
    state = MambaSpec(block_size=cache.mamba_block_size, shapes=tuple(shapes),
                      dtypes=tuple(dtypes), page_size_padded=cache.mamba_page_size_padded,
                      mamba_cache_mode=cache.mamba_cache_mode)
    latent = KeyOnlyAttentionSpec(block_size=cache.block_size, num_kv_heads=1,
                                  head_size=text.kv_lora_rank + text.qk_rope_head_dim,
                                  dtype=torch.bfloat16)
    specs = {f"layers.{i}.self_attn": state if i + 1 in kda["kda_layers"] else latent
             for i in range(text.num_hidden_layers)}
    layouts = NeuronWorker.__new__(NeuronWorker).get_supported_kv_cache_layouts()
    resolve_kv_cache_layout(config, [layouts], list(specs.values()))
    groups = get_kv_cache_groups(config, specs)
    slots = max(len(g.layer_names) for g in groups)
    return get_kv_cache_config_from_groups(config, groups, 64 * slots * latent.page_size_bytes)


def _served_reading(form, measurements):
    """What the scheduler, the model runner and the worker select from the form's serve flags,
    in one process with no workers."""
    flags = [f.replace("{checkpoint}", str(CHECKPOINT)).replace("{measurements}", str(measurements))
             for f in form["server_flags"]]
    config = _engine_config(flags)
    current_platform._validate_dcp_config(config)
    kv_config = _kv_cache_config(config)
    config.cache_config.num_gpu_blocks = kv_config.num_blocks
    scheduler = NeuronScheduler(config, kv_config, StructuredOutputManager(config),
                                block_size=resolve_kv_cache_block_sizes(kv_config, config)[0])
    with pytest.MonkeyPatch.context() as patch:
        tp_group = types.SimpleNamespace(rank_in_group=0, world_size=32)
        patch.setattr(neuron_model_runner, "get_tp_group", lambda: tp_group)
        runner = NeuronModelRunner(config, torch.device("cpu"))
    worker = NeuronWorker.__new__(NeuronWorker)
    worker.model_runner = runner
    groups = kv_config.kv_cache_groups
    state = sum(isinstance(g.kv_cache_spec, MambaSpec) for g in groups)
    return {
        "latent and state groups": (len(groups) - state, state),
        "prefill buckets": scheduler.num_batched_tokens_buckets,
        "kv segment buckets": runner.neuron_config.kv_segment_size_buckets,
        "long-prefill threshold": config.scheduler_config.long_prefill_token_threshold,
        "prefill reserve": runner.neuron_config.prefill_reserve_gib,
        "kv floor": runner.neuron_config.min_kv_budget_gib,
        "decode targets": worker._decode_compile_targets(),
    }


def test_served_forms_select_their_chunk_reserve_and_decode_programs(tmp_path):
    """Item 4: DCP 4's and DCP 16's serve flags at their chunk C build a config in CPU mode
    with no workers. The scheduler selects prefill buckets [C], the runner the KV segment
    [C], the threshold is C, the reserve and floor are the form's, and the worker lists 12
    decode targets, the cells' batch buckets by their context buckets: 0 differ."""
    forms = _cells()["forms"]
    differ = {}
    for name, (chunk, reserve, floor, contexts) in SERVED.items():
        want = {
            "latent and state groups": (1, 3),
            "prefill buckets": [chunk],
            "kv segment buckets": [chunk],
            "long-prefill threshold": chunk,
            "prefill reserve": reserve,
            "kv floor": floor,
            "decode targets": [(b, s) for b in SERVED_BATCHES
                               for s in contexts + [forms[name]["max_model_len"]]],
        }
        got = _served_reading(forms[name], tmp_path)
        bad = {k: got[k] for k in want if got[k] != want[k]}
        if bad:
            differ[name] = bad
    assert differ == {}


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
