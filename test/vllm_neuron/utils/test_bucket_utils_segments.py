# SPDX-License-Identifier: Apache-2.0
"""Several KV segment buckets: the validator, the per-request choice, the window rule.

The GLM-5.3-Flash runner reads a prefill chunk's KV through a block table of
``min(table_width, ceil((segment + query_bucket) / page))`` pages, so with one segment
the served prompt length is capped at ``segment + query_bucket`` (2048 tokens on the
standard line: 1024 + 1024 at max_model_len 4096). These tests pin the rules that remove
the cap:

1. ``kv_segment_size_buckets`` may hold several values (for a model whose kernel takes the
   query length independently of the cached length), each still a supported size, strictly
   ascending, with explicit ``num_batched_tokens_buckets``.
2. The largest segment must cover ``max_model_len``: ``prefill_window_tokens`` of the
   list is at least ``max_model_len``, else the list is refused at startup naming the
   segment to add. Without the rule a prompt longer than the window stops the engine.
3. A request takes the smallest segment at least as long as the request
   (``select_kv_segment_size``), or the largest one, whose window rule 2 proved covers it.
4. The auto-enabled list (no ``kv_segment_size_buckets`` in the config) is completed
   with the smallest supported segment that covers ``max_model_len``.
"""

from __future__ import annotations

import pytest

from vllm_neuron.utils.bucket_utils import (
    SUPPORTED_KV_SEGMENT_SIZES,
    complete_kv_segment_cover,
    covering_kv_segment_size,
    prefill_window_tokens,
    select_kv_segment_size,
    validate_kv_segment_size_buckets,
)

pytestmark = [pytest.mark.fast]

PAGE = 128
STANDARD = [1024, 2048, 4096]
BS64 = [1024, 8192]


# ── the window ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "segments, queries, block, window",
    [
        ([1024], [1024], PAGE, 2048),  # the old standard line: the 2048-token cap
        (STANDARD, [1024], PAGE, 5120),  # 4096 + 1024, in whole pages
        (BS64, [1024], PAGE, 9216),
        ([8192], [1024], PAGE, 9216),
        ([2048], [1024], PAGE, 3072),
        ([512], [512], 192, 1152),  # 1024 rounded up to whole 192-token pages: 6 pages
        ([1024], [128, 1024], PAGE, 2048),  # the largest query bucket is the one read
        ([1024], [1024], None, 2048),  # no page: the exact sum
    ],
)
def test_the_prefill_window_is_the_largest_segment_plus_the_largest_query_bucket(
    segments, queries, block, window
):
    assert prefill_window_tokens(segments, queries, block) == window


# ── the validator: lists accepted ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "segments, queries, max_model_len",
    [
        (STANDARD, [1024], 4096),  # the new standard line: 5120 >= 4096
        (BS64, [1024], 8192),  # the bs=64 line with a 1k segment for 1k prompts
        ([8192], [1024], 8192),  # the bs=64 line as served today: 9216 >= 8192
        ([2048], None, 4096),  # queries copy the segments: 2048 + 2048 covers 4096
        ([1024], [1024], 2048),  # the old standard line at a max_model_len it covers
        ([512, 1024, 2048, 4096, 8192], [1024], 8192),  # every supported size at once
        (STANDARD, [128, 1024], 4096),  # several query buckets too
    ],
)
def test_multi_valued_lists_whose_window_covers_max_model_len_are_accepted(
    segments, queries, max_model_len
):
    assert validate_kv_segment_size_buckets(
        segments,
        queries,
        allow_independent_query_buckets=True,
        block_size=PAGE,
        max_model_len=max_model_len,
    ) == segments


def test_without_max_model_len_the_coverage_rule_is_not_applied():
    """A caller that does not say the model length cannot be held to a window."""
    assert validate_kv_segment_size_buckets(
        [1024], [1024], allow_independent_query_buckets=True, block_size=PAGE
    ) == [1024]


# ── the validator: the coverage rule ────────────────────────────────────────────────


def test_the_old_standard_line_is_refused_naming_the_window_and_the_segment_to_add():
    """[1024] + [1024] serve 2048 tokens at max_model_len 4096: refused at startup, and the
    message says which segment covers (3072 needed, 4096 is the supported size)."""
    with pytest.raises(ValueError) as refused:
        validate_kv_segment_size_buckets(
            [1024],
            [1024],
            allow_independent_query_buckets=True,
            block_size=PAGE,
            max_model_len=4096,
        )
    message = str(refused.value)
    assert "2048" in message and "4096" in message, message
    assert "prefill window" in message, message
    assert "3072" in message, message
    assert "max_model_len" in message, message


@pytest.mark.parametrize(
    "segments, queries, max_model_len, needed",
    [
        ([1024, 2048], [1024], 4096, 3072),  # window 3072 < 4096; 4096 - 1024 needed
        ([1024], None, 4096, 3072),  # queries copied from the segments: window 2048
        ([1024, 2048], [1024], 8192, 7168),
        ([4096], [1024], 8192, 7168),
    ],
)
def test_a_list_whose_window_falls_short_is_refused(segments, queries, max_model_len, needed):
    """`needed` = max_model_len - the largest query bucket: the segment the message asks for."""
    with pytest.raises(ValueError, match="prefill window") as refused:
        validate_kv_segment_size_buckets(
            segments,
            queries,
            allow_independent_query_buckets=True,
            block_size=PAGE,
            max_model_len=max_model_len,
        )
    message = str(refused.value)
    assert str(max_model_len) in message, message
    assert f"at least {needed} tokens" in message, message


def test_a_window_no_supported_segment_covers_is_refused_naming_the_longest_window():
    """8192 + 1024 = 9216 is the longest window the supported segments give."""
    with pytest.raises(ValueError, match="9216") as refused:
        validate_kv_segment_size_buckets(
            [8192],
            [1024],
            allow_independent_query_buckets=True,
            block_size=PAGE,
            max_model_len=32768,
        )
    assert "32768" in str(refused.value)


def test_the_generic_kernel_has_no_window_so_no_coverage_rule():
    """The generic segmented kernel walks prior KV one segment at a time."""
    assert validate_kv_segment_size_buckets(
        [1024], [1024], block_size=PAGE, max_model_len=4096
    ) == [1024]


# ── the validator: nonsense refused ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "segments, queries, match",
    [
        (None, [1024], "non-empty list"),
        ([], [1024], "non-empty list"),
        ((1024, 2048), [1024], "non-empty list"),
        (["1024", 2048], [1024], "must be an integer"),
        ([1024, 2048.0], [1024], "must be an integer"),
        ([2048, 1024], [1024], "strictly ascending"),
        ([1024, 1024, 4096], [1024], "strictly ascending"),
        ([1000, 2048, 4096], [1024], "not a supported segment size"),
        ([1024, 2048, 16384], [1024], "not a supported segment size"),
    ],
)
def test_nonsense_lists_are_refused(segments, queries, match):
    with pytest.raises(ValueError, match=match):
        validate_kv_segment_size_buckets(
            segments,
            queries,
            allow_independent_query_buckets=True,
            block_size=PAGE,
            max_model_len=4096,
        )


def test_several_segments_need_explicit_query_buckets():
    """Left unset, the query buckets copy the segments and the runner would compile every
    (query, segment) pair and run every chunk at the largest width."""
    with pytest.raises(ValueError, match="num_batched_tokens_buckets"):
        validate_kv_segment_size_buckets(
            STANDARD, None, allow_independent_query_buckets=True, block_size=PAGE,
            max_model_len=4096,
        )


def test_the_generic_kernel_keeps_one_segment():
    """Its kernel needs seqlen_q == kv_segment_size, so it takes one segment."""
    with pytest.raises(ValueError, match="Only one segment size"):
        validate_kv_segment_size_buckets([1024, 2048], [1024, 2048])
    with pytest.raises(ValueError, match="Only one segment size"):
        validate_kv_segment_size_buckets([1024, 2048], None)


def test_the_generic_kernel_keeps_equal_query_buckets():
    with pytest.raises(ValueError, match="must match"):
        validate_kv_segment_size_buckets([1024], [128, 1024])


# ── the choice per request ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tokens, segment",
    [
        (0, 1024), (1, 1024), (700, 1024), (1024, 1024),
        (1025, 2048), (1500, 2048), (2048, 2048),
        (2049, 4096), (3000, 4096), (4095, 4096), (4096, 4096),
        # Past the largest: the largest, whose window the validator proved covers.
        (9000, 4096),
    ],
)
def test_a_request_takes_the_smallest_segment_at_least_its_length(tokens, segment):
    assert select_kv_segment_size(STANDARD, tokens) == segment


@pytest.mark.parametrize(
    "tokens, segment", [(900, 1024), (1024, 1024), (3000, 8192), (7000, 8192), (8191, 8192)]
)
def test_the_bs64_line_choice(tokens, segment):
    assert select_kv_segment_size(BS64, tokens) == segment


def test_one_segment_is_always_chosen():
    assert select_kv_segment_size([8192], 5) == 8192


# ── the covering segment and the auto-completed list ────────────────────────────────


@pytest.mark.parametrize(
    "max_model_len, max_query, segment",
    [
        (4096, 1024, 4096),  # 3072 needed; 2048 + 1024 = 3072 < 4096 -> 4096
        (3072, 1024, 2048),  # 2048 + 1024 = 3072 covers exactly
        (2048, 1024, 1024),
        (8192, 1024, 8192),
        (9216, 1024, 8192),
        (9217, 1024, None),  # nothing supported covers
        (32768, 1024, None),
        (8192, 8192, 512),  # a query bucket as long as the model: the smallest segment
    ],
)
def test_the_smallest_supported_segment_whose_window_covers(max_model_len, max_query, segment):
    assert covering_kv_segment_size(max_model_len, max_query, PAGE) == segment


def test_the_covering_segment_is_a_supported_size():
    for max_model_len in (1024, 2048, 3000, 4096, 5000, 8192, 9216):
        segment = covering_kv_segment_size(max_model_len, 1024, PAGE)
        assert segment in SUPPORTED_KV_SEGMENT_SIZES


@pytest.mark.parametrize(
    "segments, queries, max_model_len, completed",
    [
        ([1024], [1024], 4096, [1024, 4096]),  # the auto path at the standard line
        ([1024], [1024], 2048, [1024]),  # covers already: unchanged
        ([8192], [1024], 8192, [8192]),
        ([1024], [1024], 8192, [1024, 8192]),
        ([2048], [2048], 4096, [2048]),  # 2048 + 2048 covers 4096
        ([1024, 2048], [1024], 4096, [1024, 2048, 4096]),
    ],
)
def test_the_auto_list_is_completed_with_the_covering_segment(
    segments, queries, max_model_len, completed
):
    assert complete_kv_segment_cover(segments, queries, max_model_len, PAGE) == completed


def test_completion_refuses_a_model_length_no_supported_segment_covers():
    with pytest.raises(ValueError, match="9216") as refused:
        complete_kv_segment_cover([1024], [1024], 32768, PAGE)
    assert "32768" in str(refused.value)


def test_a_completed_list_passes_the_validator():
    completed = complete_kv_segment_cover([1024], [1024], 4096, PAGE)
    assert validate_kv_segment_size_buckets(
        completed, [1024], allow_independent_query_buckets=True, block_size=PAGE,
        max_model_len=4096,
    ) == completed
