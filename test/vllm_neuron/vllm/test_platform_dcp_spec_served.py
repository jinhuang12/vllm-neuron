# SPDX-License-Identifier: Apache-2.0
"""The served GLM-5.3-Flash DCP-2 form, with the shadow draft and with MTP k=3: the startup proof.

Each test builds vLLM's engine config the way the server does
(``EngineArgs(...).create_engine_config()``, which runs the platform's DCP validation) from
the served form's own arguments and environment, recorded by the gate arms
``stageB-df1-ctl-dcp2`` (DCP 2, 128k) and ``stageB-df1-ctl`` (MTP k=3, async drafter):

* argv: ``--max-model-len 131072 --max-num-batched-tokens 1024 --max-num-seqs 1
  --tensor-parallel-size 64 --enable-expert-parallel --distributed-timeout-seconds 900
  --async-scheduling --decode-context-parallel-size 2 --no-enable-prefix-caching`` and the
  ``neuron_config`` below; the MTP arm adds ``--speculative-config '{"method": "mtp",
  "num_speculative_tokens": 3}'``;
* environment: ``VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA=1``,
  ``VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING=1``, ``VLLM_NEURON_TP_ALLREDUCE_DTYPE=fp32``,
  ``VLLM_NEURON_TP_ALLREDUCE_FUSE=0``; the MTP arm adds ``VLLM_NEURON_GLM5NEXT_MTP_ASYNC=1``;
* the model directory holds the published checkpoint's ``config.json``
  (``fixtures/hf-config.json``, a byte-identical copy pinned by its sha256).

Pinned:

1. the DCP-2 form without either is admitted (the served control form);
2. the DCP-2 form with ``VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT=3`` is admitted, the head is
   built for 3 iterations, and at the config's own geometry (block 128, the rank's share of
   131072 = 65536) its draft iterations take the ownership slots and clamp at 131071;
3. the MTP k=3 form is refused by name, listing each capability the tree lacks, and is
   admitted when all three are present; then a verify step of ``1 + k`` rows per request
   takes the T-row DCP operands at the attention.

No test loads weights, starts a worker or touches a device.
"""

from __future__ import annotations

import hashlib
import importlib
import shutil
from pathlib import Path

import pytest
import torch
from vllm.engine.arg_utils import EngineArgs

from vllm_neuron.utils import dcp_ownership

from test.vllm_neuron.model.glm5_next import test_mtp_under_dcp as model_side
from test.vllm_neuron.vllm import test_platform_dcp_spec as admission

pytestmark = [pytest.mark.forked]

FIXTURE = Path(__file__).resolve().parents[1] / "model" / "glm5_next" / "fixtures" / "hf-config.json"
FIXTURE_SHA256 = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
MAX_MODEL_LEN = 131072
MTP = {"method": "mtp", "num_speculative_tokens": 3}
NEURON_CONFIG = {
    "ep_degree": 16,
    "kv_segment_size_buckets": [1024, 2048, 4096, 131072],
    "num_batched_tokens_buckets": [1024],
    "decode_context_length_buckets": [2048],
    "on_device_sampling_config": {"all_greedy": True},
    "hybrid_kv_block_size": 128,
}
SERVED_ENV = {
    "VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA": "1",
    "VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING": "1",
    "VLLM_NEURON_TP_ALLREDUCE_DTYPE": "fp32",
    "VLLM_NEURON_TP_ALLREDUCE_FUSE": "0",
}


@pytest.fixture
def served_model(tmp_path: Path) -> Path:
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256, "the fixture is the published config.json"
    shutil.copyfile(FIXTURE, tmp_path / "config.json")
    return tmp_path


def _served_env(monkeypatch, *, shadow: str | None = None, mtp_async: bool = False) -> None:
    for name, value in SERVED_ENV.items():
        monkeypatch.setenv(name, value)
    for name, value in ((admission.KNOB, shadow), (admission.ASYNC_KNOB, "1" if mtp_async else None)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def _engine_config(model: Path, speculative_config: dict | None):
    return EngineArgs(
        model=str(model),
        skip_tokenizer_init=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        tensor_parallel_size=64,
        enable_expert_parallel=True,
        distributed_timeout_seconds=900,
        async_scheduling=True,
        decode_context_parallel_size=2,
        enable_prefix_caching=False,
        speculative_config=speculative_config,
        additional_config={"neuron_config": NEURON_CONFIG},
    ).create_engine_config()


def _rank_share(config) -> int:
    from vllm_neuron.vllm.worker.neuron_model_runner import indexer_context_per_rank

    return indexer_context_per_rank(config)


# --------------------------------------------------------------------------- #
# 1. the control form.


def test_the_served_dcp2_form_is_admitted(monkeypatch, served_model):
    _served_env(monkeypatch)
    config = _engine_config(served_model, None)
    assert config.parallel_config.decode_context_parallel_size == 2
    assert config.speculative_config is None
    assert config.cache_config.block_size == model_side.BLOCK


# --------------------------------------------------------------------------- #
# 2. the shadow draft.


def test_the_served_shadow_draft_form_is_admitted_and_drafts_on_ownership_slots(
    monkeypatch, served_model
):
    from vllm_neuron.model.glm5_next.mtp import shadow_draft_k

    _served_env(monkeypatch, shadow="3")
    config = _engine_config(served_model, None)
    dcp = int(config.parallel_config.decode_context_parallel_size)
    block = int(config.cache_config.block_size)
    share = _rank_share(config)
    assert (dcp, block, share) == (2, 128, 65536)
    assert shadow_draft_k() == 3, "the head is built for the knob's 3 iterations"
    entries = share // block
    row = list(range(1, entries + 1))
    table = torch.tensor(row, dtype=torch.int32).reshape(entries, 1)
    position = 100_000  # past the rank's share, where the old clamp held every draft
    for rank in range(dcp):
        head = model_side._tiny_head(dcp, rank)
        for iteration in range(1, 4):
            carrier = head._iteration_carrier(
                model_side._carrier(table, position, max_seq_len=share), iteration, 1, "ring"
            )
            assert carrier["position"].tolist() == [position + iteration]
            want = model_side._oracle_slot(
                row, position + iteration, rank, block_size=block, dcp_size=dcp
            )
            assert carrier["latent_slots"].tolist() == [want], (rank, iteration)
        edge = head._iteration_carrier(
            model_side._carrier(table, MAX_MODEL_LEN - 2, max_seq_len=share), 3, 1, "ring"
        )
        assert edge["position"].tolist() == [MAX_MODEL_LEN - 1]


# --------------------------------------------------------------------------- #
# 3. MTP k=3.


def test_the_served_mtp_form_is_refused_naming_every_missing_capability(monkeypatch, served_model):
    _served_env(monkeypatch, mtp_async=True)
    admission._speculative_tree(monkeypatch, attention=False, indexer=False, slots=False)
    with pytest.raises(ValueError) as refused:
        _engine_config(served_model, MTP)
    message = str(refused.value)
    assert admission.speculative_message(3, 2) in message
    for capability in admission.SPECULATIVE:
        assert admission._label(capability) in message


def test_the_served_mtp_form_on_this_tree_names_what_the_tree_lacks(monkeypatch, served_model):
    """No capability is set: the tree's own attributes decide. In the model member's tree
    alone the T-row attention, indexer and async slots are other members' and absent."""
    _served_env(monkeypatch, mtp_async=True)
    absent = [
        capability for capability in admission.SPECULATIVE
        if getattr(importlib.import_module(capability[0]), capability[1], None) is not True
    ]
    if not absent:
        assert _engine_config(served_model, MTP).speculative_config.num_speculative_tokens == 3
        return
    with pytest.raises(ValueError) as refused:
        _engine_config(served_model, MTP)
    for capability in admission.SPECULATIVE:
        assert (admission._label(capability) in str(refused.value)) is (capability in absent)


def test_the_served_mtp_form_is_admitted_and_its_verify_step_takes_the_t_row_path(
    monkeypatch, served_model
):
    _served_env(monkeypatch, mtp_async=True)
    admission._speculative_tree(monkeypatch, attention=True, indexer=True, slots=True)
    config = _engine_config(served_model, MTP)
    dcp = int(config.parallel_config.decode_context_parallel_size)
    block = int(config.cache_config.block_size)
    rows = 1 + int(config.speculative_config.num_speculative_tokens)
    assert (dcp, block, rows) == (2, 128, 4)
    start = [MAX_MODEL_LEN // 2 - 2]  # a verify step crossing a block boundary at 128k
    for rank in range(dcp):
        seen = model_side._record_partial_entry(monkeypatch)
        attention = model_side._attention(dcp, rank)
        q_lift, c_kv, table, position, written, topk = model_side._decode_operands(
            1, rows, start, selected=True
        )
        with pytest.raises(model_side._Stop):
            attention._dcp_decode(q_lift, c_kv, table, position, written, 0.125, block,
                                  topk_indices=topk)
        (call,) = seen
        at = torch.tensor([start[0] + t for t in range(rows)], dtype=torch.int64)
        assert call["owner"].tolist() == (
            dcp_ownership.token_owner(at, block_size=block, dcp_size=dcp) == rank
        ).tolist()
        assert call["position"].to(torch.int64).tolist() == dcp_ownership.owned_length(
            at, rank, block_size=block, dcp_size=dcp
        ).tolist()
        assert int(call["topk_indices"].shape[0]) == rows
