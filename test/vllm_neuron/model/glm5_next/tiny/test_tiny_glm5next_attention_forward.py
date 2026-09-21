"""Forward-pass test for the tiny GLM-5.3-Flash MLA attention with the DSA indexer.

Fixtures and the dense torch reference come from ``test_tiny_glm5next_forward``.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.model.glm5_next.test_mla_decode import paged_operands
from test.vllm_neuron.model.glm5_next.tiny.test_tiny_glm5next_forward import (
    ATOL,
    MLA_HIDDEN_SIZE,
    MLA_KV_LORA_RANK,
    MLA_PAGE_SIZE,
    MLA_SOFTMAX_SCALE,
    MLA_TOKENS,
    RTOL,
    SEED_MLA,
    ReferenceShapeError,
    VacuousControlError,
    _SEAMS,
    _assert_route_predicate,
    _declare_bound_and_sentinel,
    _impl,
    _materialise_mla_indexer,
    _mla_attention_fixture,
    _mla_contract,
    _mla_dense_reference,
    _mla_latent_cache,
    _mla_latent_norm,
    _mla_pool_cache,
    _mla_selection_operands,
    _read_seam_counters,
    _reset_seam_counters,
)

pytestmark = [pytest.mark.fast]

# The scale a plain latent-space attention would use; a wrong-scale control.
MLA_WRONG_SOFTMAX_SCALE = float(MLA_KV_LORA_RANK ** -0.5)


def _mla_outside_tolerance(label: str, moved: torch.Tensor, base: torch.Tensor) -> None:
    """``moved`` must fall outside the test's tolerance band around ``base``."""
    outside = not torch.allclose(moved.float(), base.float(), rtol=RTOL, atol=ATOL)
    if not outside:
        raise VacuousControlError(
            f"{label}: the change leaves the reference inside rtol={RTOL}, "
            f"atol={ATOL}, so the test cannot tell the two apart"
        )


def test_tiny_mla_attention_forward_matches_the_reference() -> None:
    """The MLA attention forward equals dense MLA attention on the same weights."""
    attention, raw, gains, cfg = _mla_attention_fixture()
    operands = _mla_selection_operands()
    normed = (
        torch.randn(
            MLA_TOKENS, MLA_HIDDEN_SIZE,
            generator=torch.Generator().manual_seed(SEED_MLA + 1),
            dtype=torch.float32,
        )
        * 0.5
    )

    q_latent = attention.project_query_latent(normed)
    selections = []
    for _ in range(2):
        selections.append(
            attention.indexer(
                normed,
                q_latent,
                _mla_pool_cache(),
                operands["seq_lens"],
                max_seq_len=MLA_TOKENS,
                page_size=MLA_PAGE_SIZE,
                slot_mapping=operands["slot_mapping"],
            )
        )
    topk_indices, repeat = selections
    sentinels = int((topk_indices < 0).sum())
    if not torch.equal(topk_indices, repeat):
        raise VacuousControlError(
            "two indexer calls on identical operands returned different "
            "selections, so the reference cannot stand on the first while the "
            "forward makes a third"
        )
    if sentinels == 0:
        raise VacuousControlError(
            "no selected-row column carries the -1 sentinel, so the sentinel "
            "check below would measure nothing. The expansion pads its raw "
            f"width up to a multiple of the sparse kernel's key chunk, and this "
            f"geometry emits {tuple(topk_indices.shape)}"
        )

    reference_latent = _mla_latent_norm(
        _mla_contract(normed, raw["q_a_proj"]),
        gains["q_a_layernorm_weight"],
        float(cfg.rms_norm_eps),
    )
    torch.testing.assert_close(
        q_latent.float(), reference_latent.float(), rtol=RTOL, atol=ATOL
    )

    expected = _mla_dense_reference(
        attention, raw, gains, normed, _mla_latent_cache(attention), topk_indices,
        softmax_scale=MLA_SOFTMAX_SCALE,
    )

    latent_cache = _mla_latent_cache(attention)
    _reset_seam_counters()
    before = _read_seam_counters()
    got = attention.forward(
        normed,
        latent_cache=latent_cache,
        pool_cache=_mla_pool_cache(),
        seq_lens=operands["seq_lens"],
        start_position=0,
        softmax_scale=MLA_SOFTMAX_SCALE,
        max_seq_len=MLA_TOKENS,
        slot_mapping=operands["slot_mapping"],
        **paged_operands(latent_cache, 0, int(normed.shape[0]), page=MLA_PAGE_SIZE),
    )
    after = _read_seam_counters()
    route_expected = {
        "mla_projection": 9,
        "mla_absorb": 2,
        "mla_sparse": 1,
        "dsa_kpool_hadamard": 2,
        "dsa_paged_gather": 1,
        "dsa_score_gemm": 1,
        "dsa_topk_select": 1,
        "dsa_index_expand": 1,
    }
    _declare_bound_and_sentinel(route_expected)
    _assert_route_predicate("MLA attention", route_expected, before, after)

    if tuple(got.shape) != (MLA_TOKENS, MLA_HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(MLA_TOKENS, MLA_HIDDEN_SIZE)}"
        )
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)

    written = latent_cache[:MLA_TOKENS, 0, :]
    if int((written.abs().sum(dim=-1) == 0).sum()) != 0:
        raise VacuousControlError(
            "the forward left at least one cache slot all zero, so the latents "
            "it attended to are not the ones it computed"
        )

    _mla_outside_tolerance(
        "sentinel columns filled with cache row 0",
        _mla_dense_reference(
            attention, raw, gains, normed, _mla_latent_cache(attention),
            topk_indices, softmax_scale=MLA_SOFTMAX_SCALE,
            honour_sentinels=False,
        ),
        expected,
    )

    _mla_outside_tolerance(
        f"softmax scale {MLA_WRONG_SOFTMAX_SCALE:.7f} rather than the model's "
        f"{MLA_SOFTMAX_SCALE:.7f}",
        _mla_dense_reference(
            attention, raw, gains, normed, _mla_latent_cache(attention),
            topk_indices, softmax_scale=MLA_WRONG_SOFTMAX_SCALE,
        ),
        expected,
    )

    # An unprepared module refuses before any kernel dispatches.
    model_fp8 = _impl()
    bare = model_fp8.Glm5NextMLAAttention(cfg)
    for name in raw:
        setattr(bare, f"{name}_weight", torch.nn.Parameter(raw[name]))
    for gain_name, gain in gains.items():
        setattr(bare, gain_name, torch.nn.Parameter(gain))
    _reset_seam_counters()
    unprepared_before = _read_seam_counters()
    with pytest.raises(ValueError, match="prepare_projection_weights"):
        bare.forward(
            normed,
            latent_cache=_mla_latent_cache(attention),
            pool_cache=_mla_pool_cache(),
            seq_lens=operands["seq_lens"],
            start_position=0,
            softmax_scale=MLA_SOFTMAX_SCALE,
            max_seq_len=MLA_TOKENS,
            slot_mapping=operands["slot_mapping"],
            **paged_operands(
                _mla_latent_cache(attention), 0, int(normed.shape[0]),
                page=MLA_PAGE_SIZE,
            ),
        )
    unprepared_after = _read_seam_counters()
    moved = {
        seam: unprepared_after[seam][0] - unprepared_before[seam][0]
        for seam in _SEAMS
        if unprepared_after[seam][0] != unprepared_before[seam][0]
    }
    if moved:
        raise VacuousControlError(
            f"the refusal ran after {sorted(moved.items())} dispatched, so it is "
            f"not the first thing the forward does with an unprepared weight"
        )

    half = model_fp8.Glm5NextMLAAttention(cfg)
    for name in raw:
        setattr(half, f"{name}_weight", torch.nn.Parameter(raw[name]))
    for gain_name, gain in gains.items():
        setattr(half, gain_name, torch.nn.Parameter(gain))
    half.prepare_projection_weights()
    _materialise_mla_indexer(half.indexer, torch.Generator().manual_seed(SEED_MLA))
    with pytest.raises(
        model_fp8.Glm5NextMLADecodeError, match="prepare_absorb_weights"
    ):
        half.forward(
            normed,
            latent_cache=_mla_latent_cache(attention),
            pool_cache=_mla_pool_cache(),
            seq_lens=operands["seq_lens"],
            start_position=0,
            softmax_scale=MLA_SOFTMAX_SCALE,
            max_seq_len=MLA_TOKENS,
            slot_mapping=operands["slot_mapping"],
            **paged_operands(
                _mla_latent_cache(attention), 0, int(normed.shape[0]),
                page=MLA_PAGE_SIZE,
            ),
        )
