# SPDX-License-Identifier: Apache-2.0
"""The runner's start check of the MTP decode ceilings, before any graph compiles.

Two kernels of a drafting (method "mtp") server have a ceiling a longer context crosses:

* the ``T``-row indexer scores (``decode_trow_wide``) take at most 32768 candidates per
  rank: ``max_model_len / DCP / pool``;
* the async drafter's position correction (``functional/mtp/async_step.
  mtp_async_correct_kernel``) holds its block-table column, ``ceil(max_model_len /
  block_size)`` pages on every rank, in SBUF: ``4 * max(W + 2, 4)`` bytes per page at
  ``W = 1 + num_speculative_tokens`` rows plus at most 128, against the host SBUF rule
  ``SBUF_BYTES_PER_PARTITION`` (212464 B). At ``k = 3`` that is 8847 pages, 1132416
  tokens at block 128; the 1M form holds 92 % of the bound.

``check_mtp_decode_ceilings`` refuses a form past either, naming the form, the limiting
kernel and the fix. A server that drafts nothing, or not by "mtp", is not checked.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_neuron.functional.dsa.decode_trow import DecodeTrowError
from vllm_neuron.functional.mtp.async_step import MtpAsyncStepError
from vllm_neuron.utils.neuron_utils import SBUF_BYTES_PER_PARTITION
from vllm_neuron.vllm.worker import neuron_model_runner as NMR

BLOCK = 128
POOL = 4
ONE_M = 1 << 20
#: Covered SBUF bytes per partition of the 1M correction call at width 4 (8192 pages,
#: ``table`` [8192, 1], DCP 8), read from its CPU build's lower_sync dump (neuronx-cc
#: 2.27): the measured figure behind the formula's 92 %.
MEASURED_1M_WIDTH_4_BYTES = 196660


def _config(max_model_len, dcp, drafts, method="mtp"):
    speculative = (None if drafts is None
                   else SimpleNamespace(method=method, num_speculative_tokens=drafts))
    return SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_model_len),
                           parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
                           cache_config=SimpleNamespace(block_size=BLOCK),
                           scheduler_config=SimpleNamespace(max_num_seqs=1),
                           speculative_config=speculative)


def _check(max_model_len, dcp, drafts, *, async_drafter=True, method="mtp", pool=POOL):
    NMR.check_mtp_decode_ceilings(_config(max_model_len, dcp, drafts, method), pool,
                                  async_drafter=async_drafter)


# ------------------------------------------------------------------ the formula


@pytest.mark.parametrize("width, tiles", [(1, 4), (2, 4), (3, 5), (4, 6), (5, 7), (8, 10)])
def test_the_correction_holds_max_w_plus_2_or_4_page_tiles(width, tiles):
    pages = 8192
    assert NMR.mtp_async_correct_sbuf_bytes(pages, width) == 4 * tiles * pages + 128


@pytest.mark.parametrize("width, pages", [(1, 13271), (2, 13271), (3, 10616), (4, 8847),
                                          (5, 7583), (8, 5308)])
def test_the_page_ceiling_is_the_most_pages_under_the_host_sbuf_rule(width, pages):
    assert SBUF_BYTES_PER_PARTITION == 212464
    assert NMR.mtp_async_correct_max_pages(width) == pages
    assert NMR.mtp_async_correct_sbuf_bytes(pages, width) <= SBUF_BYTES_PER_PARTITION
    assert NMR.mtp_async_correct_sbuf_bytes(pages + 1, width) > SBUF_BYTES_PER_PARTITION


def test_at_k_3_the_ceiling_is_1132416_tokens_and_1m_holds_92_percent_of_the_bound():
    assert NMR.mtp_async_correct_max_pages(4) * BLOCK == 1132416
    modelled = NMR.mtp_async_correct_sbuf_bytes(ONE_M // BLOCK, 4)
    assert MEASURED_1M_WIDTH_4_BYTES <= modelled <= SBUF_BYTES_PER_PARTITION
    assert 0.92 <= MEASURED_1M_WIDTH_4_BYTES / SBUF_BYTES_PER_PARTITION < 0.93


# ------------------------------------------------------------------ the start check


@pytest.mark.parametrize("dcp, drafts", [(8, 3), (8, 1), (16, 3)])
def test_the_served_1m_form_is_admitted(dcp, drafts):
    _check(ONE_M, dcp, drafts)


@pytest.mark.parametrize("max_model_len", [ONE_M, 1132416])
def test_at_dcp_16_the_async_ceiling_admits_up_to_its_pages(max_model_len):
    _check(max_model_len, 16, 3)


@pytest.mark.parametrize("max_model_len, pages", [(1132417, 8848), (1132544, 8848)])
def test_at_dcp_16_the_async_ceiling_refuses_past_its_pages(max_model_len, pages):
    with pytest.raises(MtpAsyncStepError,
                       match=rf"max_model_len={max_model_len}.*{pages} pages of {BLOCK}"
                             r".*mtp_async_correct_kernel.*functional/mtp/async_step\.py"
                             r".*24 B per page.*SBUF_BYTES_PER_PARTITION=212464"
                             r".*8847 pages.*1132416 tokens"
                             r".*VLLM_NEURON_GLM5NEXT_MTP_ASYNC"):
        _check(max_model_len, 16, 3)


def test_without_the_async_drafter_the_correction_is_not_checked():
    _check(1132544, 16, 3, async_drafter=False)


def test_at_k_4_the_1m_form_is_refused_by_the_async_ceiling():
    with pytest.raises(MtpAsyncStepError, match=r"8192 pages.*width 5.*7583 pages"):
        _check(ONE_M, 8, 4)


@pytest.mark.parametrize("max_model_len, dcp, per_rank", [(ONE_M + 1, 8, 131200),
                                                          (ONE_M, 4, 262144)])
def test_the_scores_ceiling_refuses_first_naming_the_form(max_model_len, dcp, per_rank):
    with pytest.raises(DecodeTrowError,
                       match=rf"decode_context_parallel_size={dcp}.*{per_rank} of "
                             rf"max_model_len={max_model_len}"
                             r".*MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES=32768"
                             r".*decode_trow_wide\.py"):
        _check(max_model_len, dcp, 3)


@pytest.mark.parametrize("drafts, method", [(None, "mtp"), (0, "mtp"), (3, "eagle3")])
def test_a_server_that_drafts_nothing_by_mtp_is_not_checked(drafts, method):
    _check(ONE_M * 4, 4, drafts, method=method)


def test_without_an_indexer_only_the_async_ceiling_is_checked():
    _check(ONE_M, 4, 3, pool=None)
    with pytest.raises(MtpAsyncStepError):
        _check(ONE_M * 2, 4, 3, pool=None)
