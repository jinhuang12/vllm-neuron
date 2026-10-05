# SPDX-License-Identifier: Apache-2.0
"""When the DSA indexer's selection keeps every token, and the bound that decides it.

The indexer scores every complete pool of ``index_kpool`` tokens and keeps the top
``select_k = index_topk // index_kpool`` pools, then appends the open tail pool's
tokens. While a sequence has at most ``select_k`` complete pools, every complete
pool is kept, so the selected token set is the whole causal prefix: selection is
a no-op and attending the prefix directly is exact. That holds for

    seq_len // index_kpool <= select_k  <=>  seq_len <= select_k * index_kpool + index_kpool - 1

which is 2051 tokens for this checkpoint (``index_topk`` 2048, ``index_kpool`` 4).

The decision has to be a trace-time constant, because the decode graph is captured
once and replayed at every position. The bound a graph can prove is the smaller of
two shapes it already carries: the model length the runner passes, and the window
the block table names (``pages * page_size``): this step's token sits inside that
window, so no sequence served by the graph is longer than it.
"""

from __future__ import annotations


class DsaDecodeBypassError(ValueError):
    """Raised for dials or shapes the bound cannot be derived from."""


def bypass_max_context(index_topk: int, index_kpool: int) -> int:
    """The longest context at which selection keeps every token.

    ``select_k * index_kpool + index_kpool - 1`` with ``select_k = index_topk //
    index_kpool``: at that length there are exactly ``select_k`` complete pools and
    a full open tail; one token more completes pool ``select_k + 1``.
    """
    topk, pool = int(index_topk), int(index_kpool)
    if pool < 1 or topk < pool:
        raise DsaDecodeBypassError(
            f"index_kpool must be positive and index_topk at least one pool; got "
            f"index_topk={index_topk}, index_kpool={index_kpool}"
        )
    return (topk // pool) * pool + pool - 1


def selection_bound(max_seq_len: int, window_rows: int | None) -> int:
    """The longest sequence a graph can serve: ``min(max_seq_len, window_rows)``.

    ``window_rows`` is ``pages * page_size`` of the block-table operand, or None
    when the caller has no window (the bound is then ``max_seq_len``).
    """
    bound = int(max_seq_len)
    if bound <= 0:
        raise DsaDecodeBypassError(f"max_seq_len must be positive; got {max_seq_len!r}")
    if window_rows is not None:
        if int(window_rows) <= 0:
            raise DsaDecodeBypassError(f"the window must be positive; got {window_rows!r}")
        bound = min(bound, int(window_rows))
    return bound


def selection_is_a_no_op(bound: int, index_topk: int, index_kpool: int) -> bool:
    """True when every sequence no longer than ``bound`` keeps all its tokens."""
    return int(bound) <= bypass_max_context(index_topk, index_kpool)
