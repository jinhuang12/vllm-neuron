# SPDX-License-Identifier: Apache-2.0
"""The MTP proposer and the runner's ``method == "mtp"`` branch.

``vllm_neuron/vllm/spec_decode/mtp.py`` holds ``MtpProposer``: the GLM-5.3-Flash
draft served from the target graph (the root's own ``mtp`` head), so the
proposer owns no model, compiles no graph, and hands the runner the ``[B, k]`` draft
ids the root returned. Covered here: the runner builds it from a ``method: mtp``
engine config and keys its decode threshold on ``k``; the two configurations the
proposer cannot serve (CPU sampling, async scheduling) are refused by name at
construction; the other methods keep the runner's refusal text; ``load_model`` binds
the root's own head and refuses a root without one or with another ``k``; sentinel
rows become "no drafts"; the eagle-only warmup and capture hooks are no-ops.

Until the head's reader honours the speculative config, the tests that need a
built head set the shadow-draft knob to the same ``k`` the config names.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_proposer.py
"""

from __future__ import annotations

import pathlib

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs

from vllm_neuron import envs
from vllm_neuron.model.glm5_next import mtp as head_module
from vllm_neuron.vllm.spec_decode.mtp import MtpProposer
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr

pytestmark = [pytest.mark.forked]

FIXTURE = pathlib.Path(__file__).resolve().parents[2] / "vllm_neuron" / "model" / "glm5_next" / "fixtures"
#: The served draft count.
DRAFT_K = 3
KNOB = head_module.SHADOW_DRAFT_ENV
#: The async drafter's opt-in (envs.py); off = the shipped synchronous drafter.
ASYNC_KNOB = "VLLM_NEURON_GLM5NEXT_MTP_ASYNC"


def _engine_config(spec: dict | None, *, async_scheduling: bool | None = False,
                   on_device_sampling: bool | None = None,
                   max_num_seqs: int = e2e.E2E_MAX_NUM_SEQS):
    """The tiny first-request engine config, with a speculative config.

    Async scheduling is off unless a test turns it on: with a declared sampler the
    platform's default is on, and the mtp proposer refuses that (its second series).
    ``on_device_sampling`` is left unset unless the test declared a sampler on the
    class first (``fr._declaring_a_sampler``); the platform refuses the knob otherwise.
    """
    neuron_config: dict = {
        "num_batched_tokens_buckets": [fr.PREFILL_BUCKET, e2e.E2E_MAX_SEQ_LEN],
        "num_seqs_buckets": [max(fr.DECODE_BATCH, int(max_num_seqs))],
    }
    if on_device_sampling is not None:
        neuron_config["on_device_sampling_config"] = {} if on_device_sampling else None
    return EngineArgs(
        model=str(FIXTURE),
        skip_tokenizer_init=True,
        max_model_len=e2e.E2E_MAX_SEQ_LEN,
        max_num_seqs=int(max_num_seqs),
        max_num_batched_tokens=e2e.E2E_MAX_SEQ_LEN,
        block_size=fr.tiny.MLA_PAGE_SIZE,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=async_scheduling,
        speculative_config=spec,
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()


def _mtp(k: int = DRAFT_K) -> dict:
    return {"method": "mtp", "num_speculative_tokens": k}


def test_the_runner_builds_an_mtp_proposer_from_the_spec_config(tmp_path, monkeypatch):
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    config = _engine_config(_mtp(), on_device_sampling=True)
    with fr._parallel_state(tmp_path, config):
        runner = NeuronModelRunner(config, device=torch.device("cpu"))
    assert isinstance(runner.drafter, MtpProposer)
    assert runner.is_mtp_spec is True and runner.is_eagle3_spec is False
    assert runner.on_device_sampling is True
    assert runner.drafter.num_speculative_tokens == DRAFT_K
    assert runner._decode_token_threshold() == 1 + DRAFT_K
    # The head is a submodule of the root: the worker must not count its bytes twice.
    assert runner.drafter.shares_target_parameters is True


def test_mtp_without_on_device_sampling_is_refused_by_name(tmp_path):
    """The verify step's rejection sampling and the draft from the accepted row run in
    the target graph, so the runner must sample on device."""
    e2e._require_cpu_mode()
    config = _engine_config(_mtp(), on_device_sampling=False)
    with fr._parallel_state(tmp_path, config), pytest.raises(ValueError, match="on-device sampling"):
        NeuronModelRunner(config, device=torch.device("cpu"))


def test_mtp_with_async_scheduling_is_refused_by_name(tmp_path, monkeypatch):
    """Async scheduling is the second series (the accepted count arrives one step late)."""
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    config = _engine_config(_mtp(), async_scheduling=True, on_device_sampling=True)
    assert config.scheduler_config.async_scheduling is True
    with fr._parallel_state(tmp_path, config), pytest.raises(ValueError, match="async scheduling"):
        NeuronModelRunner(config, device=torch.device("cpu"))


def test_the_async_knob_defaults_off(monkeypatch):
    """Unset, the knob reads False: the shipped opt-in drafter is the synchronous one."""
    monkeypatch.delenv(ASYNC_KNOB, raising=False)
    assert envs.VLLM_NEURON_GLM5NEXT_MTP_ASYNC is False
    monkeypatch.setenv(ASYNC_KNOB, "1")
    assert envs.VLLM_NEURON_GLM5NEXT_MTP_ASYNC is True


def test_the_async_knob_admits_async_scheduling_for_one_sequence(tmp_path, monkeypatch):
    """Knob on, ``--async-scheduling``, ``max_num_seqs = 1``: the drafter runs its async steps."""
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    monkeypatch.setenv(ASYNC_KNOB, "1")
    config = _engine_config(_mtp(), async_scheduling=True, on_device_sampling=True)
    assert config.scheduler_config.async_scheduling is True
    with fr._parallel_state(tmp_path, config):
        runner = NeuronModelRunner(config, device=torch.device("cpu"))
    assert runner.is_mtp_spec is True
    assert runner.drafter.async_steps is True
    assert runner.drafter.num_speculative_tokens == DRAFT_K


def test_the_async_knob_with_synchronous_scheduling_is_refused_by_name(tmp_path, monkeypatch):
    """The knob names a path the synchronous scheduler never takes; a set knob that would
    be ignored is refused rather than silently dropped."""
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    monkeypatch.setenv(ASYNC_KNOB, "1")
    config = _engine_config(_mtp(), async_scheduling=False, on_device_sampling=True)
    with fr._parallel_state(tmp_path, config), pytest.raises(ValueError, match=ASYNC_KNOB):
        NeuronModelRunner(config, device=torch.device("cpu"))


def test_the_async_knob_with_more_than_one_sequence_is_refused_by_name(tmp_path, monkeypatch):
    """The async drafter serves bs = 1; a server that could batch two is refused by name."""
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    monkeypatch.setenv(ASYNC_KNOB, "1")
    config = _engine_config(_mtp(), async_scheduling=True, on_device_sampling=True, max_num_seqs=2)
    with fr._parallel_state(tmp_path, config), pytest.raises(ValueError, match="max_num_seqs"):
        NeuronModelRunner(config, device=torch.device("cpu"))


def test_the_synchronous_drafter_records_no_async_steps(tmp_path, monkeypatch):
    """Knob off and synchronous scheduling: the shipped path, ``async_steps`` False."""
    e2e._require_cpu_mode()
    monkeypatch.delenv(ASYNC_KNOB, raising=False)
    proposer = MtpProposer(_engine_config(_mtp()), torch.device("cpu"), True)
    assert proposer.async_steps is False


def test_other_methods_keep_the_runners_refusal_text(tmp_path, monkeypatch):
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    config = _engine_config({"method": "ngram", "num_speculative_tokens": 1, "prompt_lookup_max": 2},
                            on_device_sampling=True)
    with fr._parallel_state(tmp_path, config), pytest.raises(ValueError) as raised:
        NeuronModelRunner(config, device=torch.device("cpu"))
    assert str(raised.value) == "Unsupported speculative decoding method: ngram"


def test_load_model_binds_the_roots_own_head(monkeypatch):
    e2e._require_cpu_mode()
    monkeypatch.setenv(KNOB, str(DRAFT_K))
    root = e2e._fixture()["root"]
    assert root.mtp is not None
    assert root.draft_k == DRAFT_K, "the root records the k it built the head for"
    proposer = MtpProposer(_engine_config(_mtp()), torch.device("cpu"), True)
    proposer.load_model(root)
    assert proposer.model is root.mtp


def test_load_model_refuses_a_root_without_a_head_by_name(monkeypatch):
    e2e._require_cpu_mode()
    monkeypatch.delenv(KNOB, raising=False)
    root = e2e._fixture()["root"]
    assert root.draft_k == 0
    assert root.mtp is None
    proposer = MtpProposer(_engine_config(_mtp()), torch.device("cpu"), True)
    with pytest.raises(ValueError, match="draft head"):
        proposer.load_model(root)


def test_load_model_refuses_a_head_built_for_another_k(monkeypatch):
    e2e._require_cpu_mode()
    monkeypatch.setenv(KNOB, str(DRAFT_K + 2))
    root = e2e._fixture()["root"]
    proposer = MtpProposer(_engine_config(_mtp()), torch.device("cpu"), True)
    with pytest.raises(ValueError, match=f"k={DRAFT_K + 2}"):
        proposer.load_model(root)


def test_sentinel_rows_become_no_drafts():
    drafts = torch.tensor([[5, 6, 7], [-1, -1, -1], [8, -1, -1]], dtype=torch.int32)
    assert MtpProposer.take_drafts(drafts) == [[5, 6, 7], [], [8]]


def test_the_eagle_only_hooks_are_no_ops():
    proposer = MtpProposer(_engine_config(_mtp()), torch.device("cpu"), True)
    assert proposer.warmup(num_tokens=4, num_reqs=1, attn_metadata={}) is None
    assert proposer.graph_extract(num_tokens=4, num_reqs=1, attn_metadata={}, device=None) is None


# ── the head's reader of k honours the speculative config ──


def test_the_reader_takes_k_from_the_current_speculative_config(monkeypatch):
    """Inside the worker's config context (``load_model``) the production reader is the
    spec config; outside it, and without the knob, nothing drafts."""
    monkeypatch.delenv(KNOB, raising=False)
    config = _engine_config(_mtp(DRAFT_K))
    with set_current_vllm_config(config, check_compile=False):
        assert head_module.shadow_draft_k() == DRAFT_K
    assert head_module.shadow_draft_k() == 0


def test_the_knob_and_the_config_must_agree(monkeypatch):
    config = _engine_config(_mtp(DRAFT_K))
    monkeypatch.setenv(KNOB, str(DRAFT_K + 2))
    with set_current_vllm_config(config, check_compile=False), pytest.raises(ValueError) as raised:
        head_module.shadow_draft_k()
    assert KNOB in str(raised.value) and "num_speculative_tokens" in str(raised.value)
    monkeypatch.setenv(KNOB, str(DRAFT_K))
    with set_current_vllm_config(config, check_compile=False):
        assert head_module.shadow_draft_k() == DRAFT_K


def test_a_config_without_speculation_leaves_the_knob_to_the_shadow_path(monkeypatch):
    monkeypatch.setenv(KNOB, "5")
    with set_current_vllm_config(_engine_config(None), check_compile=False):
        assert head_module.shadow_draft_k() == 5
    monkeypatch.delenv(KNOB, raising=False)
    with set_current_vllm_config(_engine_config(None), check_compile=False):
        assert head_module.shadow_draft_k() == 0
