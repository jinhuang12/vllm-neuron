# SPDX-License-Identifier: Apache-2.0
"""Increment 4 (P-EAGLE dispatch + bucketing) CPU-only unit tests — group A.

DESIGN 1 (confirmed by the lead): the drafter graph's INPUT WINDOW is the
target model's per-step verify window ``batch_size * (1 + num_speculative_tokens)``
for BOTH sequential eagle3 and parallel (P-EAGLE) drafting. The parallel vs.
sequential difference lives entirely inside the traced drafter forward
(``_forward_parallel`` vs. the unrolled recurrent loop), which runs an internal
single backbone pass of ``batch_size * num_speculative_tokens`` masked slots in
parallel mode. Instruction #1's literal formula (input window =
``batch_size * num_speculative_tokens``) is superseded: applying it would make
the traced graph's input window disagree with the runtime input and force a
recompile on the first real decode.

Two groups here, both CPU-only, no Neuron parallel state required:
  (1) bucket-arithmetic table: (batch, K, parallel on/off) -> expected drafter
      graph INPUT-WINDOW token count (unchanged in both modes, and byte-equal to
      today's ``batch_size * (1 + K)``) AND internal backbone token count
      (``batch_size * K`` parallel vs. ``batch_size * 1`` sequential).
  (2) synthetic-input column-keying HAZARD guard: proves the input-window count
      ``batch_size * (1 + K)`` keys ``_build_synthetic_inputs`` to the DECODE
      branch (``raw_sampled_token_ids`` has ``K + 1`` columns), while the
      superseded ``batch_size * K`` window would silently fall to the PREFILL
      branch (1 column). This is the tripwire that stops anyone reintroducing
      the ``bs * num_spec`` window later.
"""
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_neuron.vllm.spec_decode import eagle as eagle_mod
from vllm_neuron.vllm.spec_decode.eagle import EagleProposer


def _make_vllm_config(num_speculative_tokens=4, hf_config=None, method="eagle3"):
    draft_model_config = SimpleNamespace(hf_config=hf_config)
    speculative_config = SimpleNamespace(
        draft_model_config=draft_model_config,
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        model="amazon/GPT-OSS-20B-P-EAGLE",
    )
    return SimpleNamespace(speculative_config=speculative_config)


def _build_proposer(parallel_drafting=False, num_speculative_tokens=4):
    # Parallel needs a resolvable mask token id.
    hf_config = SimpleNamespace(ptd_token_id=201020) if parallel_drafting else None
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


# ---------------------------------------------------------------------------
# (1) bucket-arithmetic table
# ---------------------------------------------------------------------------

# (batch, K) -> the sequential-mode INPUT-WINDOW token counts as computed today
# by ``batch_size * (1 + num_speculative_tokens)``. These MUST NOT change: the
# sequential path is byte-identical.
_INPUT_WINDOW_GOLDEN = {
    (1, 2): 3,
    (1, 4): 5,
    (2, 2): 6,
    (3, 4): 15,
    (4, 1): 8,
    (8, 5): 48,
}


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("batch,K", sorted(_INPUT_WINDOW_GOLDEN))
def test_draft_graph_input_window_is_verify_shape_both_modes(parallel, batch, K):
    """Input window = bs*(1+K) for sequential AND parallel (DESIGN 1)."""
    p = _build_proposer(parallel_drafting=parallel, num_speculative_tokens=K)
    expected = _INPUT_WINDOW_GOLDEN[(batch, K)]
    assert p.draft_graph_input_tokens(batch) == expected
    # Explicit invariant: parallel does NOT shrink the input window.
    assert p.draft_graph_input_tokens(batch) == batch * (1 + K)


@pytest.mark.parametrize("batch,K", sorted(_INPUT_WINDOW_GOLDEN))
def test_sequential_input_window_byte_identical_to_legacy_formula(batch, K):
    """Regression guard: sequential value equals the pre-inc-4 inline formula."""
    p = _build_proposer(parallel_drafting=False, num_speculative_tokens=K)
    legacy = batch * (1 + K)  # the exact expression inc-4 replaced
    assert p.draft_graph_input_tokens(batch) == legacy


@pytest.mark.parametrize("batch,K", [(1, 2), (2, 2), (3, 4), (8, 5)])
def test_parallel_backbone_tokens(batch, K):
    """Parallel internal backbone pass = bs*K (extra_slots_per_request == K)."""
    p = _build_proposer(parallel_drafting=True, num_speculative_tokens=K)
    assert p.extra_slots_per_request == K
    assert p.parallel_backbone_tokens(batch) == batch * K


@pytest.mark.parametrize("batch,K", [(1, 2), (2, 2), (3, 4), (8, 5)])
def test_sequential_backbone_tokens(batch, K):
    """Sequential: extra_slots_per_request == 1; no fused bs*K backbone pass."""
    p = _build_proposer(parallel_drafting=False, num_speculative_tokens=K)
    assert p.extra_slots_per_request == 1
    assert p.parallel_backbone_tokens(batch) == batch  # bs * 1


def test_bucket_table_full_matrix():
    """One consolidated (batch,K,parallel)->(input_window, backbone) table."""
    rows = []
    for (batch, K) in sorted(_INPUT_WINDOW_GOLDEN):
        for parallel in (False, True):
            p = _build_proposer(
                parallel_drafting=parallel, num_speculative_tokens=K
            )
            rows.append(
                (
                    batch,
                    K,
                    parallel,
                    p.draft_graph_input_tokens(batch),
                    p.parallel_backbone_tokens(batch),
                )
            )
    for batch, K, parallel, input_window, backbone in rows:
        assert input_window == batch * (1 + K)  # DESIGN 1: same both modes
        assert backbone == (batch * K if parallel else batch)


# ---------------------------------------------------------------------------
# (2) synthetic-input column-keying HAZARD guard
# ---------------------------------------------------------------------------


def _stub_proposer_for_synthetic(K, hidden=16):
    """A bare EagleProposer instance (no __init__/gloo) exposing just what
    ``_build_synthetic_inputs`` reads: ``model.config.hidden_size``,
    ``num_speculative_tokens``, ``device``."""
    p = EagleProposer.__new__(EagleProposer)
    p.num_speculative_tokens = K
    p.device = torch.device("cpu")
    p.model = SimpleNamespace(config=SimpleNamespace(hidden_size=hidden))
    return p


@pytest.mark.parametrize("batch,K", [(1, 2), (2, 4), (3, 5)])
def test_input_window_count_keys_decode_branch(batch, K):
    """bs*(1+K) window -> decode branch: raw_sampled_token_ids has K+1 cols."""
    p = _stub_proposer_for_synthetic(K)
    num_tokens = batch * (1 + K)  # == draft_graph_input_tokens
    kwargs = EagleProposer._build_synthetic_inputs(p, num_tokens, batch)
    assert kwargs["raw_sampled_token_ids"].shape == (batch, K + 1)
    # Full-window token tensors are the verify-window length.
    assert kwargs["target_token_ids"].shape == (num_tokens,)
    assert kwargs["target_positions"].shape == (num_tokens,)


@pytest.mark.parametrize("batch,K", [(1, 2), (2, 4), (3, 5)])
def test_superseded_bs_times_K_window_falls_to_prefill_branch(batch, K):
    """HAZARD tripwire: the superseded bs*K window (K>=2) silently keys the
    PREFILL branch (1 col), which would break the decode raw_sampled contract.
    This test documents WHY DESIGN 1 keeps the input window at bs*(1+K)."""
    p = _stub_proposer_for_synthetic(K)
    wrong_num_tokens = batch * K  # instruction #1's superseded formula
    kwargs = EagleProposer._build_synthetic_inputs(p, wrong_num_tokens, batch)
    # K>=2 => bs*K != bs*(1+K) => NOT the decode branch => 1 column.
    assert kwargs["raw_sampled_token_ids"].shape == (batch, 1)
    # And it disagrees with the real decode window, proving the mismatch.
    assert wrong_num_tokens != batch * (1 + K)
