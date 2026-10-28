# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 engine config with the DSpark drafter on the V1 model runner.

The build is the served one with no workers: TP 32 by DP 2, 4 draft tokens. Set
``KIMI_K3_CHECKPOINT`` and ``KIMI_K3_DSPARK_DRAFT`` to the checkpoint and draft
directories; the defaults are under ``~/models``. Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import json
import multiprocessing
import os
from pathlib import Path

import pytest

CHECKPOINT = Path(
    os.environ.get(
        "KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3"
    )
)
DRAFT = Path(
    os.environ.get(
        "KIMI_K3_DSPARK_DRAFT",
        Path.home() / "models" / "Inferact" / "Kimi-K3-DSpark-Block5",
    )
)

pytestmark = pytest.mark.skipif(
    not ((CHECKPOINT / "config.json").is_file() and (DRAFT / "config.json").is_file()),
    reason=f"Kimi K3 checkpoint or DSpark draft not found at {CHECKPOINT}, {DRAFT}",
)

TP, DP, NUM_DRAFT_TOKENS = 32, 2, 4
CELLS = Path(__file__).resolve().parents[1] / "serving" / "spec_cells.json"


def _build(**speculative):
    """Build the engine config with the DSpark drafter; no worker starts."""
    from vllm.engine.arg_utils import EngineArgs

    return EngineArgs(
        model=str(CHECKPOINT),
        tensor_parallel_size=TP,
        data_parallel_size=DP,
        enable_expert_parallel=True,
        # Sequence cap: the platform refuses more sequences than the KDA kernels take.
        max_num_seqs=16,
        speculative_config={
            "method": "dspark",
            "model": str(DRAFT),
            "num_speculative_tokens": NUM_DRAFT_TOKENS,
            **speculative,
        },
    ).create_engine_config()


def _wrap_depth(fn) -> int:
    """Count the wrappers above the original function."""
    depth = 0
    while hasattr(fn, "__wrapped__"):
        fn, depth = fn.__wrapped__, depth + 1
    return depth


def _build_in_fresh_process(results) -> None:
    """Build the config in this process, apply the policy again, report the wrap."""
    try:
        config = _build()
    except Exception as exc:
        results.put({"error": f"{type(exc).__name__}: {exc}"})
        return
    from vllm.config.vllm import VllmConfig

    from vllm_neuron.vllm.model_runner_v2_policy import _apply_neuron_policy_patch

    _apply_neuron_policy_patch(VllmConfig)
    features = VllmConfig._get_v1_model_runner_unsupported_features
    results.put(
        {
            "error": None,
            "method": config.speculative_config.method,
            "marked": getattr(features, "_neuron_dspark_admitted", False),
            "depth": _wrap_depth(features),
        }
    )


def test_dspark_builds_on_v1_runner():
    config = _build()
    assert config.speculative_config.method == "dspark"
    assert config.speculative_config.num_speculative_tokens == NUM_DRAFT_TOKENS
    assert config.use_v2_model_runner is False


def test_adaptive_verification_still_refused_on_v1_runner():
    with pytest.raises(ValueError) as refusal:
        _build(enable_adaptive_verification=True)
    message = str(refusal.value)
    assert "Model Runner V1 does not support: adaptive draft verification" in message
    assert "dspark speculative decoding" not in message


def test_dspark_builds_in_fresh_process_with_one_wrap():
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    child = context.Process(target=_build_in_fresh_process, args=(results,))
    child.start()
    report = results.get(timeout=600)
    child.join(timeout=60)
    assert child.exitcode == 0
    assert report == {"error": None, "method": "dspark", "marked": True, "depth": 1}


def _flag_value(flags, name):
    """The value after ``name`` in a flag list, or None."""
    return flags[flags.index(name) + 1] if name in flags else None


def _max_num_seqs(form):
    """The form's per-replica sequence cap: its flag, else the API server's default."""
    value = _flag_value(form["serve_flags"], "--max-num-seqs")
    if value is not None:
        return int(value)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext

    _, defaults = EngineArgs.get_batch_defaults(form["tensor_parallel_size"])
    return defaults[UsageContext.OPENAI_API_SERVER]


def test_registered_forms_fit_the_verification_row_limit():
    """Each form's sequence cap fits the proposer's rows and every cell's share."""
    from vllm_neuron.vllm.spec_decode.dspark import DSPARK_MAX_VERIFY_ROWS

    doc = json.loads(CELLS.read_text())
    too_many_rows, too_few_seqs = [], []
    for name, form in doc["forms"].items():
        seqs = _max_num_seqs(form)
        speculative = _flag_value(form["speculative_flags"], "--speculative-config")
        k = json.loads(speculative)["num_speculative_tokens"]
        if seqs * (k + 1) > DSPARK_MAX_VERIFY_ROWS:
            too_many_rows.append((name, seqs, k, seqs * (k + 1)))
        for cell in doc["cells"]:
            share = -(-cell["concurrency"] // form["data_parallel_size"])
            if cell["form"] == name and share > seqs:
                too_few_seqs.append((cell["name"], share, seqs))
    assert too_many_rows == []
    assert too_few_seqs == []


def test_registered_forms_fit_the_verify_kernel():
    """Each form's sequence cap is within the KDA verify kernel's sequence limit."""
    from vllm_neuron.functional.kda.fused_decode import MAX_SEQUENCES

    forms = json.loads(CELLS.read_text())["forms"]
    caps = {name: _max_num_seqs(form) for name, form in forms.items()}
    over = {name: cap for name, cap in caps.items() if cap > MAX_SEQUENCES}
    assert not over, f"sequence cap above the verify kernel's {MAX_SEQUENCES}: {over}"
