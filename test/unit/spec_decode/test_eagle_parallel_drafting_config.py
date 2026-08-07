# SPDX-License-Identifier: Apache-2.0
"""Increment 1 (P-EAGLE config plumbing) CPU-only unit tests.

Covers the config acceptance matrix for threading ``parallel_drafting`` and the
resolved ``ptd_token_id`` into ``EagleProposer`` construction. These tests run
without Neuron hardware: distributed init is not available on CPU, so
``get_world_group`` (called at the tail of ``EagleProposer.__init__``) is
patched to a stub. The parallel-drafting mask-token resolution runs before that
call, so the flag-without-token-id error path needs no patching.
"""
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_neuron.vllm.spec_decode import eagle as eagle_mod
from vllm_neuron.vllm.spec_decode.eagle import EagleProposer


def _make_vllm_config(num_speculative_tokens=4, hf_config=None, method="eagle3"):
    """Minimal duck-typed VllmConfig reaching only what __init__ reads."""
    draft_model_config = SimpleNamespace(hf_config=hf_config)
    speculative_config = SimpleNamespace(
        draft_model_config=draft_model_config,
        method=method,
        num_speculative_tokens=num_speculative_tokens,
    )
    return SimpleNamespace(speculative_config=speculative_config)


def _build_proposer(parallel_drafting=False, hf_config=None, num_speculative_tokens=4):
    vllm_config = _make_vllm_config(
        num_speculative_tokens=num_speculative_tokens, hf_config=hf_config
    )
    fake_world_group = SimpleNamespace(rank=0)
    with mock.patch.object(eagle_mod, "get_world_group", return_value=fake_world_group):
        return EagleProposer(
            vllm_config,
            torch.device("cpu"),
            on_device_sampling=False,
            parallel_drafting=parallel_drafting,
        )


def test_sequential_eagle3_unchanged():
    """(a) eagle3 + parallel_drafting=False: sequential path, unchanged."""
    proposer = _build_proposer(parallel_drafting=False)
    assert proposer.parallel_drafting is False
    assert proposer.ptd_token_id is None
    assert proposer.extra_slots_per_request == 1


def test_parallel_drafting_reads_ptd_token_id():
    """(b) eagle3 + parallel_drafting=True + ptd_token_id: flag + token id set."""
    hf_config = SimpleNamespace(ptd_token_id=201020)
    proposer = _build_proposer(
        parallel_drafting=True, hf_config=hf_config, num_speculative_tokens=4
    )
    assert proposer.parallel_drafting is True
    assert proposer.ptd_token_id == 201020
    # Upstream extra_slots_per_request analog: K masked slots per request.
    assert proposer.extra_slots_per_request == 4


def test_parallel_drafting_without_token_id_raises():
    """(c) parallel_drafting=True with no mask token id: ValueError, never inert."""
    hf_config = SimpleNamespace()  # neither pard_token nor ptd_token_id
    with pytest.raises(ValueError, match="parallel_drafting is enabled"):
        _build_proposer(parallel_drafting=True, hf_config=hf_config)


def test_parallel_drafting_missing_hf_config_raises():
    """parallel_drafting=True with no hf_config at all: same ValueError."""
    with pytest.raises(ValueError, match="parallel_drafting is enabled"):
        _build_proposer(parallel_drafting=True, hf_config=None)


def test_pard_token_takes_precedence_over_ptd_token_id():
    """(d) pard_token wins over ptd_token_id, matching upstream order."""
    hf_config = SimpleNamespace(pard_token=999, ptd_token_id=201020)
    proposer = _build_proposer(parallel_drafting=True, hf_config=hf_config)
    assert proposer.ptd_token_id == 999


def test_default_parallel_drafting_is_false():
    """Omitting the kwarg keeps the sequential default (behavioral safety)."""
    vllm_config = _make_vllm_config()
    fake_world_group = SimpleNamespace(rank=0)
    with mock.patch.object(eagle_mod, "get_world_group", return_value=fake_world_group):
        proposer = EagleProposer(vllm_config, torch.device("cpu"), on_device_sampling=False)
    assert proposer.parallel_drafting is False
    assert proposer.ptd_token_id is None
    assert proposer.extra_slots_per_request == 1
