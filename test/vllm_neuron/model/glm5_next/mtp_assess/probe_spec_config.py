# SPDX-License-Identifier: Apache-2.0
"""Reachability probe: ``--speculative-config`` on the tiny GLM-5.3-Flash CPU runner.

For every speculative method string vLLM 0.24 accepts that could name the checkpoint's
own draft layer (``mtp`` and its deprecated aliases) plus the two the Neuron runner
mentions (``eagle3``, ``eagle``) and ``ngram``, this script walks the serve path stage
by stage and records the exact outcome of each stage:

* A. ``EngineArgs(..., speculative_config=...).create_engine_config()``
* B. ``NeuronModelRunner(config, device=cpu)`` (the runner's method check)
* C. bind the tiny root, warm up prefill and decode, as the worker does
* D. one prefill, then one decode step that schedules one draft token

A fifth row, ``mtp[bypass]``, is an EXPERIMENT ONLY: it lifts the two method checks that
stand in front of the model so the walk reaches the GLM-specific code behind them. Stage A
spells ``glm5_next`` into vLLM's ``MTPModelTypes`` for the duration of ``create_engine_config``
(``vllm/config/speculative.py`` raises ``NotImplementedError`` for a target whose
``model_type`` is not in that list); stage B spells the method ``eagle3`` while the runner's
constructor runs (``neuron_model_runner.py`` accepts only that method) and puts ``mtp`` back.
Stage C0 gives the proposer a stub ``model`` (the real one cannot be built: no GLM draft
class exists) so the runner's own warmups run with a drafter present. No production file is
edited; the swaps are on in-process objects and the first two are undone afterwards.

Output: one JSON document on stdout (and to ``--output``), one entry per method per
stage, each with ``outcome`` (``ok`` / ``raise``), the exception type and message, and
the innermost traceback frame inside ``vllm_neuron`` or ``vllm`` as ``file:line``.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 PYTHONPATH=<worktree> \\
        <venv>/bin/python -m test.vllm_neuron.model.glm5_next.mtp_assess.probe_spec_config \\
        --output /home/ubuntu/glm53f-wt2/reports/mtp-logs/probe_spec_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import traceback

import torch

#: The engine config the tiny first-request test builds, here with a speculative config.
REPO = pathlib.Path(__file__).resolve().parents[5]
FIXTURE = REPO / "test" / "vllm_neuron" / "model" / "glm5_next" / "fixtures"
REQUEST = "req-0"

METHODS: dict[str, dict] = {
    # The checkpoint's own draft layer: vLLM's generic name and the GLM alias.
    "mtp": {"method": "mtp", "num_speculative_tokens": 1},
    "glm4_moe_mtp": {"method": "glm4_moe_mtp", "num_speculative_tokens": 1},
    "deepseek_mtp": {"method": "deepseek_mtp", "num_speculative_tokens": 1},
    # The one method the runner implements, pointed at the only checkpoint we have.
    "eagle3": {"method": "eagle3", "model": str(FIXTURE), "num_speculative_tokens": 1},
    "eagle": {"method": "eagle", "model": str(FIXTURE), "num_speculative_tokens": 1},
    "ngram": {"method": "ngram", "num_speculative_tokens": 1, "prompt_lookup_max": 2},
}


def _frame(exc: BaseException) -> str:
    """``file:line`` of the innermost traceback frame inside vllm_neuron or vllm."""
    frames = traceback.extract_tb(exc.__traceback__)
    chosen = None
    for frame in frames:
        if "/vllm_neuron/" in frame.filename or "/vllm/" in frame.filename:
            chosen = frame
    if chosen is None and frames:
        chosen = frames[-1]
    if chosen is None:
        return "?"
    path = chosen.filename
    for marker in ("/vllm_neuron/", "/site-packages/"):
        if marker in path:
            path = path.split(marker, 1)[1] if marker == "/site-packages/" else "vllm_neuron/" + path.split(marker, 1)[1]
            break
    return f"{path}:{chosen.lineno} ({chosen.name})"


def _record(stage: str, fn):
    try:
        value = fn()
        return {"stage": stage, "outcome": "ok", "detail": value}, True
    except BaseException as exc:  # noqa: BLE001 -- the outcome IS the exception
        message = " ".join(str(exc).split())
        return {
            "stage": stage,
            "outcome": "raise",
            "exception": type(exc).__name__,
            "message": message[:600],
            "frame": _frame(exc),
        }, False


def _engine_config(spec: dict | None, *, fr, e2e, async_scheduling=None):
    from vllm.engine.arg_utils import EngineArgs

    neuron_config = {
        "num_batched_tokens_buckets": [fr.PREFILL_BUCKET, e2e.E2E_MAX_SEQ_LEN],
        "num_seqs_buckets": [fr.DECODE_BATCH],
    }
    return EngineArgs(
        model=str(FIXTURE),
        skip_tokenizer_init=True,
        max_model_len=e2e.E2E_MAX_SEQ_LEN,
        max_num_seqs=e2e.E2E_MAX_NUM_SEQS,
        max_num_batched_tokens=e2e.E2E_MAX_SEQ_LEN,
        block_size=fr.tiny.MLA_PAGE_SIZE,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=async_scheduling,
        speculative_config=spec,
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()


def _spec_summary(config) -> dict:
    spec = config.speculative_config
    if spec is None:
        return {"speculative_config": None}
    draft = spec.draft_model_config
    return {
        "method": spec.method,
        "num_speculative_tokens": spec.num_speculative_tokens,
        "model": spec.model,
        "draft_architecture": getattr(draft, "architecture", None) if draft else None,
        "draft_hf_model_type": getattr(getattr(draft, "hf_config", None), "model_type", None) if draft else None,
        "draft_n_predict": getattr(getattr(draft, "hf_config", None), "n_predict", None) if draft else None,
        "async_scheduling": config.scheduler_config.async_scheduling,
        "scheduler_cls": str(config.scheduler_config.scheduler_cls).rsplit(".", 1)[-1],
    }


def _decode_step_with_draft(position: int, generated: int, groups: int, draft: list[int], *, fr):
    """The tiny test's decode step, with ``draft`` scheduled as this request's draft tokens."""
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    step = fr._decode_step(position=position, generated=generated, groups=groups)
    tokens = 1 + len(draft)
    cached = CachedRequestData(
        req_ids=[REQUEST],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[position],
        num_output_tokens=[generated],
    )
    out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={REQUEST: tokens},
        total_num_scheduled_tokens=tokens,
        scheduled_spec_decode_tokens={REQUEST: list(draft)},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    out.num_scheduled_tokens_padded = {REQUEST: tokens}
    del step
    return out


def probe(name: str, spec: dict, *, bypass: bool, fr, e2e) -> list[dict]:
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    rows: list[dict] = []
    holder: dict = {}

    def stage_a():
        if not bypass:
            holder["config"] = _engine_config(spec, fr=fr, e2e=e2e)
            return _spec_summary(holder["config"])
        import typing

        import vllm.config.speculative as speculative

        real_types = speculative.MTPModelTypes
        speculative.MTPModelTypes = typing.Literal[typing.get_args(real_types) + ("glm5_next",)]
        try:
            holder["config"] = _engine_config(spec, fr=fr, e2e=e2e)
        finally:
            speculative.MTPModelTypes = real_types
        return _spec_summary(holder["config"]) | {
            "note": "EXPERIMENT ONLY: 'glm5_next' added to vllm.config.speculative.MTPModelTypes "
            "during create_engine_config, restored afterwards",
        }

    row, ok = _record("A.create_engine_config", stage_a)
    rows.append(row)
    if not ok:
        return rows
    config = holder["config"]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"mtp-probe-{name}-"))
    with fr._parallel_state(tmp, config):
        if bypass:
            real_method = config.speculative_config.method

            def stage_b():
                config.speculative_config.method = "eagle3"
                try:
                    holder["runner"] = NeuronModelRunner(config, device=torch.device("cpu"))
                finally:
                    config.speculative_config.method = real_method
                return {
                    "note": "EXPERIMENT ONLY: method spelled eagle3 during __init__, restored to "
                    f"{real_method!r} afterwards",
                    "drafter": type(holder["runner"].drafter).__name__,
                }
        else:

            def stage_b():
                holder["runner"] = NeuronModelRunner(config, device=torch.device("cpu"))
                return {"drafter": type(holder["runner"].drafter).__name__}

        row, ok = _record("B.NeuronModelRunner.__init__", stage_b)
        rows.append(row)
        if not ok:
            return rows
        runner = holder["runner"]
        if bypass:

            def stage_c0():
                import types

                drafter = runner.drafter
                drafter.model = types.SimpleNamespace(
                    get_kv_spec=lambda: types.SimpleNamespace(layers=[]),
                    bind_kv_cache=lambda kv_caches: None,
                )
                for name in ("graph_extract", "warmup", "validate_same_kv_cache_group"):
                    setattr(drafter, name, lambda *args, **kwargs: None)
                return {
                    "note": "EXPERIMENT ONLY: the proposer has no model (no GLM draft class exists; "
                    "eagle.py compile_and_load_draft_model would raise), so `drafter.model` is a stub "
                    "with no KV layers and graph_extract/warmup/validate_same_kv_cache_group are no-ops; "
                    "the runner's own warmups then run with a drafter present",
                }

            row, ok = _record("C0.stub_drafter_model", stage_c0)
            rows.append(row)
            if not ok:
                return rows

        def stage_c():
            fixture = e2e._fixture()
            root = fixture["root"]
            runner.model = root
            runner.vocab_size = fr.tiny.STACK_VOCAB_SIZE
            runner.initialize_kv_cache(fr._kv_cache_config(runner))
            runner.warmup_prefill(fr.PREFILL_BUCKET, 0)
            runner.warmup_decode(fr.DECODE_BATCH, ctx_bucket=runner.max_model_len)
            holder["prompt"] = fr._prompt()
            return {
                "spec_decode_enabled_in_warmup": bool(
                    runner.drafter is not None and runner.speculative_config is not None
                ),
                "decode_token_threshold": runner._decode_token_threshold(),
            }

        row, ok = _record("C.bind_root+warmups", stage_c)
        rows.append(row)
        if not ok:
            return rows

        def stage_d():
            prompt = holder["prompt"]
            groups = fr._groups(runner)
            _, prefill = fr._step(runner, fr._prefill_step(prompt, groups))
            first = prefill.sampled_token_ids
            draft = [int(first[0][0])]
            _, decode = fr._step(
                runner,
                _decode_step_with_draft(len(prompt), 1, groups, draft, fr=fr),
            )
            return {"prefill_ids": first, "decode_ids": decode.sampled_token_ids, "draft": draft}

        row, _ = _record("D.prefill+decode_with_one_draft_token", stage_d)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    for name in ("VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR"):
        if os.environ.get(name) != "1":
            sys.exit(f"{name}=1 must be set in the environment (the kernels read it at import)")
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr

    import vllm_neuron
    report = {
        "worktree": str(REPO),
        "vllm_neuron": vllm_neuron.__file__,
        "fixture": str(FIXTURE),
        "methods": {},
    }
    names = args.only or list(METHODS) + ["mtp[bypass]"]
    for name in names:
        bypass = name.endswith("[bypass]")
        spec = METHODS[name.removesuffix("[bypass]")]
        print(f"=== {name}: {spec}", file=sys.stderr, flush=True)
        rows = probe(name, spec, bypass=bypass, fr=fr, e2e=e2e)
        report["methods"][name] = {"speculative_config": spec, "stages": rows}
        for row in rows:
            print(f"  {row['stage']}: {row['outcome']} "
                  f"{row.get('exception', '')} {row.get('frame', '')}", file=sys.stderr, flush=True)
    text = json.dumps(report, indent=1, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
