# SPDX-License-Identifier: Apache-2.0
"""Decode context parallelism config check for the Kimi K3 MLA hybrid.

The platform's DCP check counts one shared latent KV head for an MLA model and reads
the query-head count from the text config. Set ``KIMI_K3_CHECKPOINT`` to the
checkpoint directory; the default is ``~/models/moonshotai/Kimi-K3``. Run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from vllm.engine.arg_utils import EngineArgs

CHECKPOINT = Path(
    os.environ.get(
        "KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3"
    )
)

TP, DP = 32, 2
CELLS = Path(__file__).resolve().parents[1] / "serving" / "cells.json"


def _build_checkpoint_config(dcp: int, max_num_seqs: int = 32):
    """The served engine config of the checkpoint at the given DCP size, no workers."""
    return EngineArgs(
        model=str(CHECKPOINT),
        tensor_parallel_size=TP,
        data_parallel_size=DP,
        decode_context_parallel_size=dcp,
        enable_expert_parallel=True,
        max_model_len=131072,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.90,
    ).create_engine_config()


@pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)
def test_checkpoint_passes_check_at_dcp_4_16_32():
    errors = {}
    for dcp in (4, 16, 32):
        try:
            config = _build_checkpoint_config(dcp)
        except Exception as error:
            errors[dcp] = f"{type(error).__name__}: {error}"
        else:
            assert config.parallel_config.decode_context_parallel_size == dcp
    assert errors == {}


def test_grouped_query_model_keeps_kv_head_rule(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 2,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "vocab_size": 1024,
                "max_position_embeddings": 4096,
                "torch_dtype": "bfloat16",
            }
        )
    )
    engine_args = EngineArgs(
        model=str(tmp_path),
        tensor_parallel_size=8,
        decode_context_parallel_size=4,
        max_model_len=4096,
        skip_tokenizer_init=True,
    )
    with pytest.raises(ValidationError, match=r"num_kv_heads=8; need tp >= 32"):
        engine_args.create_engine_config()


def _flag_value(flags, name):
    """The value after ``name`` in a flag list, or None."""
    return flags[flags.index(name) + 1] if name in flags else None


def _max_num_seqs(flags):
    """The form's per-engine sequence cap: its flag, else the API server's default."""
    value = _flag_value(flags, "--max-num-seqs")
    if value is not None:
        return int(value)
    from vllm.usage.usage_lib import UsageContext

    _, defaults = EngineArgs.get_batch_defaults(int(_flag_value(flags, "--tensor-parallel-size")))
    return defaults[UsageContext.OPENAI_API_SERVER]


def test_registered_forms_fit_the_kda_decode_kernel():
    """Each form's sequence cap fits the KDA decode kernel and every cell; the config step enforces it."""
    from vllm_neuron.vllm.platform import KDA_DECODE_MAX_SEQS

    doc = json.loads(CELLS.read_text())
    too_many_seqs, too_few_seqs = [], []
    for name, form in doc["forms"].items():
        flags = form["server_flags"]
        seqs = _max_num_seqs(flags)
        if seqs > KDA_DECODE_MAX_SEQS:
            too_many_seqs.append((name, seqs))
        replicas = int(_flag_value(flags, "--data-parallel-size"))
        for cell in doc["cells"]:
            share = -(-cell["concurrency"] // replicas)
            if cell["form"] == name and share > seqs:
                too_few_seqs.append((cell["name"], share, seqs))
    assert too_many_seqs == []
    assert too_few_seqs == []

    if not (CHECKPOINT / "config.json").is_file():
        pytest.skip(f"Kimi K3 checkpoint not found at {CHECKPOINT}")
    with pytest.raises(ValueError, match="--max-num-seqs"):
        _build_checkpoint_config(4, max_num_seqs=2 * KDA_DECODE_MAX_SEQS)
    config = _build_checkpoint_config(4, max_num_seqs=KDA_DECODE_MAX_SEQS)
    assert config.scheduler_config.max_num_seqs == KDA_DECODE_MAX_SEQS
