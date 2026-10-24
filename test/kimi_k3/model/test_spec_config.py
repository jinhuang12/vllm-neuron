# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 engine config with the DSpark drafter on the V1 model runner.

The build is the served one with no workers: TP 32 by DP 2, 4 draft tokens. Set
``KIMI_K3_CHECKPOINT`` and ``KIMI_K3_DSPARK_DRAFT`` to the checkpoint and draft
directories; the defaults are under ``~/models``. Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

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


def _build(**speculative):
    """Build the engine config with the DSpark drafter; no worker starts."""
    from vllm.engine.arg_utils import EngineArgs

    return EngineArgs(
        model=str(CHECKPOINT),
        tensor_parallel_size=TP,
        data_parallel_size=DP,
        enable_expert_parallel=True,
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
