# SPDX-License-Identifier: Apache-2.0
"""``Glm5NextDSAIndexer.forward_requests``: one DSA/MLA layer's decode step for ``B``
requests at once, against the one-request layer run once per request.

The reference is 75090b9's ``Glm5NextMLAAttention.forward`` (the git-show snapshot in
``test/hardware/baselines/dsa_75090b9``, same weights) -- the one-request indexer chain
and ``mla_sparse_attention`` -- on each request's own views of the same banks
(``dsa_batch_case.per_request_layer``). The live tree's own one-request decode is the
batch-of-one case of the batched step, so it cannot be its own reference. The batched layer is the projections once on
``B`` rows, ``forward_requests`` and ``mla_decode_attention``
(``dsa_batch_case.batched_layer``). Lengths are unequal and straddle the selection
bound, pool completions and page edges; ``max_seq_len`` is 8192, so 2048 candidate
pools per request.

What is compared, and how closely:

* (a) Selection: each request's selected token set, exactly. The two paths score in
  different orders and agree to fp32 rounding (``test_decode_batch.py``); no seed here
  has two pools that close at the cut. The written banks -- every ring and every
  stored pool -- agree bit for bit.
* (b) Output: the attention output (before absorb-out) to ``ATTENDED_ATOL``, the
  tolerance between the two attention kernels, and the layer output to one bf16 ulp of
  its largest entry: the attention is cast to bf16 before absorb-out, and two fp32
  values that agree to 1e-7 can still round to neighbouring bf16 values.
* (c) Isolation: poisoning every bank row a request may not read -- other requests'
  slots and pages, nobody's spare slots and pages, its own pools from this step on --
  changes nothing about that request, bit for bit.

The per-request reference takes the batched run's indexer projections
(``per_request_layer(projected=...)``): the projection kernels round one row and ``B``
rows differently in the last fp32 bit, which is not what this file tests. One test runs
the reference on its own projections too.
"""

from __future__ import annotations

import functools

import pytest
import torch

import vllm_neuron.model.glm5_next.model_fp8 as model_fp8
from test.hardware.baselines.dsa_75090b9 import load as load_75090b9
from test.vllm_neuron.functional.dsa import dsa_batch_case as C
from test.vllm_neuron.functional.dsa.dsa_decode_case import (
    PAGE,
    build_attention,
    decode_config,
)
from vllm_neuron.functional.attention import mla_decode as MD
from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa.index_expand import index_expand_dispatch_counters
from vllm_neuron.functional.dsa.score_gemm import score_gemm_dispatch_counters
from vllm_neuron.functional.dsa.topk_select import topk_select_dispatch_counters
from vllm_neuron.utils.neuron_utils import can_run_kernel

MAX_SEQ_LEN = 8192
LENGTHS = {
    1: [8000],
    4: [2100, 3000, 4096, 8000],
    8: [2100, 3000, 4096, 8000, 2052, 5003, 7999, 6144],
}
ATTENDED_ATOL = 2e-5
BF16_ULP = 2.0 ** -8
POISON = 3.0e4


@functools.lru_cache(maxsize=1)
def _module():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    return build_attention(model_fp8, decode_config(hidden_size=512, q_lora_rank=256))


@functools.lru_cache(maxsize=1)
def _reference():
    """75090b9's layer with the same weights: the one-request decode."""
    return build_attention(load_75090b9().model_fp8,
                           decode_config(hidden_size=512, q_lora_rank=256))


def _cfg():
    return _module().indexer


def _per_request(ops, taps, projected=None):
    """The reference layer once per request; asserts it scored once per request."""
    before = score_gemm_dispatch_counters()
    want = C.per_request_layer(_reference(), ops, taps=taps, projected=projected)
    after = score_gemm_dispatch_counters()
    selects = int(ops["max_seq_len"]) // int(_cfg().index_kpool) > _cfg().select_k()
    assert after[0] - before[0] == (int(ops["hidden"].shape[0]) if selects else 0)
    assert after[1] == before[1]
    return want


def _ops(lengths, seed=7):
    return C.batch_operands(decode_config(hidden_size=512, q_lora_rank=256), lengths,
                            max_seq_len=MAX_SEQ_LEN, seed=seed)


def _counted_batched(ops):
    """The batched layer, and what it dispatched: the counters after it alone."""
    DB.reset_decode_batch_dispatch_counters()
    MD.reset_mla_decode_dispatch_counters()
    taps: dict = {}
    out = C.batched_layer(_module(), ops, taps=taps)
    taps["decode_batch"] = DB.decode_batch_dispatch_counters()
    taps["routes"] = DB.decode_batch_route_counts()
    taps["mla_decode"] = MD.mla_decode_route_counts()
    return out, taps


def _assert_batched_route(taps):
    """The ring step, the scores and the selection ran as one NKI launch each,
    attention as one."""
    assert taps["decode_batch"] == (3, 0)
    assert taps["routes"][:2] == (1, 1) and taps["routes"][3] == 1
    assert taps["mla_decode"][:2] == (0, 1)


def _assert_pool_banks_agree(got, want):
    """Every stored pool bit for bit; the trash row is nobody's value.

    A step that closes no pool writes a meaningless row to its slot's trash row on both
    paths, and the two compute it from differently ordered rings (the one-request step
    rotates its ring first), so that row alone is left out.
    """
    assert torch.equal(got[:, :-1], want[:, :-1])


@functools.lru_cache(maxsize=None)
def _paired(batch: int):
    """One batched run and its per-request reference on the same operands."""
    ops = _ops(LENGTHS[batch])
    mine, ref = C.cloned(ops), C.cloned(ops)
    topk_before = topk_select_dispatch_counters()
    expand_before = index_expand_dispatch_counters()
    out, taps = _counted_batched(mine)
    taps["topk"] = tuple(a - b for a, b in zip(topk_select_dispatch_counters(), topk_before))
    taps["expand"] = tuple(a - b for a, b in zip(index_expand_dispatch_counters(),
                                                 expand_before))
    want_taps: dict = {}
    want = _per_request(ref, want_taps, projected=taps["projected"])
    return ops, mine, ref, out, taps, want, want_taps


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_batched_selection_equals_the_per_request_selection(batch):
    ops, mine, ref, _out, taps, _want, want_taps = _paired(batch)
    _assert_batched_route(taps)
    # One selection kernel (``_assert_batched_route``) for top-k, sentinel, order, expand.
    assert taps["topk"] == (0, 0) and taps["expand"] == (0, 0)
    got = taps["indices"]
    assert got.shape[0] == batch
    for b in range(batch):
        want_row = want_taps["indices"][b][0]
        assert got[b].shape == want_row.shape
        assert torch.equal(got[b].sort().values, want_row.sort().values), (
            f"request {b} (length {LENGTHS[batch][b]}) selected a different token set")
    # The banks both paths wrote: rings, stored pools, latent rows.
    assert torch.equal(mine["tail_bank"], ref["tail_bank"])
    _assert_pool_banks_agree(mine["pool_bank"], ref["pool_bank"])
    assert torch.equal(mine["latent_cache"], ref["latent_cache"])


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_batched_layer_output_equals_the_per_request_output(batch):
    _ops_, _mine, _ref, out, taps, want, want_taps = _paired(batch)
    _assert_batched_route(taps)
    attended = taps["attended"]
    for b in range(batch):
        torch.testing.assert_close(attended[b:b + 1], want_taps["attended"][b],
                                   rtol=0.0, atol=ATTENDED_ATOL)
    assert out.shape == want.shape and out.dtype == want.dtype
    ulp = BF16_ULP * float(want.float().abs().max())
    assert float((out.float() - want.float()).abs().max()) <= ulp


@pytest.mark.parametrize("batch", [4, 8])
def test_the_reference_on_its_own_projections_selects_the_same_tokens(batch):
    """Without sharing the projections the selections still agree for these lengths."""
    ops = _ops(LENGTHS[batch], seed=8)
    mine, ref = C.cloned(ops), C.cloned(ops)
    out, taps = _counted_batched(mine)
    _assert_batched_route(taps)
    want_taps: dict = {}
    want = _per_request(ref, want_taps)
    for b in range(batch):
        assert torch.equal(taps["indices"][b].sort().values,
                           want_taps["indices"][b][0].sort().values)
    ulp = BF16_ULP * float(want.float().abs().max())
    assert float((out.float() - want.float()).abs().max()) <= 2 * ulp


def _poison_what_nobody_may_read(ops):
    """Spare slots and pages, and each request's own pools from this step on."""
    bad = C.cloned(ops)
    owned = {int(s) for s in bad["state_slots"]}
    for slot in range(int(bad["pool_bank"].shape[0])):
        if slot not in owned:
            bad["pool_bank"][slot] = POISON
            bad["tail_bank"][slot] = POISON
    pool = int(_cfg().index_kpool)
    for b, slot in enumerate(bad["state_slots"].tolist()):
        stored = int(bad["position"][b]) // pool
        bad["pool_bank"][slot, stored:] = POISON
    held = set(bad["block_table"][bad["block_table"] >= 0].tolist())
    latent = bad["latent_cache"].reshape(-1, PAGE, bad["latent_cache"].shape[-1])
    for page in range(int(latent.shape[0])):
        if page not in held:
            latent[page] = POISON
    return bad


def _poison_everyone_but(ops, keep: int):
    """Every other request's slot, ring and pages."""
    bad = C.cloned(ops)
    latent = bad["latent_cache"].reshape(-1, PAGE, bad["latent_cache"].shape[-1])
    for b in range(int(bad["state_slots"].shape[0])):
        if b == keep:
            continue
        slot = int(bad["state_slots"][b])
        bad["pool_bank"][slot] = POISON
        bad["tail_bank"][slot] = POISON
        for page in bad["block_table"][b].tolist():
            if page >= 0:
                latent[page] = POISON
    return bad


def test_no_request_reads_a_row_that_is_not_its_own():
    """(c) Poisoned banks: nothing a request may not read changes its answer."""
    ops = _ops(LENGTHS[8], seed=9)
    clean_out, clean = _counted_batched(C.cloned(ops))
    _assert_batched_route(clean)
    out, taps = _counted_batched(_poison_what_nobody_may_read(ops))
    _assert_batched_route(taps)
    assert torch.equal(taps["indices"], clean["indices"])
    assert torch.equal(taps["attended"], clean["attended"])
    assert torch.equal(out, clean_out)
    for keep in (0, 3, 7):
        out, taps = _counted_batched(_poison_everyone_but(ops, keep))
        _assert_batched_route(taps)
        assert torch.equal(taps["indices"][keep], clean["indices"][keep])
        assert torch.equal(taps["attended"][keep], clean["attended"][keep])
        assert torch.equal(out[keep], clean_out[keep])
        # The poison reached the others, so the check above is not vacuous.
        others = [b for b in range(8) if b != keep]
        assert not torch.equal(out[others], clean_out[others])


def test_forward_requests_writes_only_its_own_slots_and_rows():
    ops = _ops(LENGTHS[4], seed=10)
    mine = C.cloned(ops)
    _counted_batched(mine)
    pool = int(_cfg().index_kpool)
    rows = int(ops["pool_bank"].shape[1])
    owned = ops["state_slots"].tolist()
    for slot in range(int(ops["pool_bank"].shape[0])):
        if slot not in owned:
            assert torch.equal(mine["pool_bank"][slot], ops["pool_bank"][slot])
            assert torch.equal(mine["tail_bank"][slot], ops["tail_bank"][slot])
    for b, slot in enumerate(owned):
        pos = int(ops["position"][b])
        row = pos // pool if pos % pool == pool - 1 else rows - 1
        changed = (mine["pool_bank"][slot] != ops["pool_bank"][slot]).any(-1)
        assert changed.nonzero().flatten().tolist() == [row]


def test_the_bypass_regime_writes_the_same_banks_and_attends_the_same_prefix():
    """Below the selection bound forward_requests only writes; the layer attends densely."""
    lengths = [100, 1500, 2048, 777]
    ops = C.batch_operands(decode_config(hidden_size=512, q_lora_rank=256), lengths,
                           max_seq_len=2048, seed=11)
    mine, ref = C.cloned(ops), C.cloned(ops)
    out, taps = _counted_batched(mine)
    assert taps["indices"] is None
    assert taps["decode_batch"] == (1, 0) and taps["routes"][:2] == (1, 0)
    assert taps["mla_decode"][0] == 1
    want_taps: dict = {}
    want = _per_request(ref, want_taps, projected=taps["projected"])
    assert torch.equal(mine["tail_bank"], ref["tail_bank"])
    _assert_pool_banks_agree(mine["pool_bank"], ref["pool_bank"])
    ulp = BF16_ULP * float(want.float().abs().max())
    assert float((out.float() - want.float()).abs().max()) <= ulp


def test_forward_requests_refuses_mismatched_banks():
    module = _module()
    ops = _ops(LENGTHS[4])
    hidden = ops["hidden"]
    q_latent = module.project_query_latent(hidden)
    indexer = module.indexer

    def call(**over):
        args = dict(pool_bank=ops["pool_bank"], tail_bank=ops["tail_bank"],
                    slots=ops["state_slots"], seq_lens=ops["seq_lens"],
                    position=ops["position"])
        args.update(over)
        return indexer.forward_requests(hidden, q_latent, args["pool_bank"],
                                        args["tail_bank"], args["slots"],
                                        args["seq_lens"], args["position"],
                                        max_seq_len=MAX_SEQ_LEN)

    with pytest.raises(model_fp8.Glm5NextDSAIndexerError, match="tail_bank"):
        call(tail_bank=ops["tail_bank"][:-1])
    with pytest.raises(model_fp8.Glm5NextDSAIndexerError, match="slots"):
        call(slots=ops["state_slots"][:2])
    with pytest.raises(model_fp8.Glm5NextDSAIndexerError, match="trash"):
        call(pool_bank=ops["pool_bank"][:, :MAX_SEQ_LEN // 4])


def _views(ops):
    """The runner's carrier form: one view per request of each bank, in request order."""
    slots = ops["state_slots"].tolist()
    return (tuple(ops["pool_bank"][s] for s in slots),
            tuple(ops["tail_bank"][s] for s in slots))


@pytest.mark.parametrize("batch", [1, 4])
def test_the_carrier_form_selects_and_writes_as_the_bank_form(batch):
    """Views in, writes into the views: the same indices and the same banks as slots in."""
    module = _module()
    indexer = module.indexer
    ops = _ops(LENGTHS[batch], seed=12)
    q_latent = module.project_query_latent(ops["hidden"])
    projected = indexer.project_stage(ops["hidden"], q_latent)
    by_bank, by_view = C.cloned(ops), C.cloned(ops)
    DB.reset_decode_batch_dispatch_counters()
    want = indexer.forward_requests(
        ops["hidden"], q_latent, by_bank["pool_bank"], by_bank["tail_bank"],
        by_bank["state_slots"], ops["seq_lens"], ops["position"],
        max_seq_len=MAX_SEQ_LEN, projected=projected)
    pools, tails = _views(by_view)
    got = indexer.forward_requests(
        ops["hidden"], q_latent, pools, tails, None, ops["seq_lens"], ops["position"],
        max_seq_len=MAX_SEQ_LEN, projected=projected)
    # Two forward_requests calls: a ring step, scores and a selection each.
    assert DB.decode_batch_dispatch_counters() == (6, 0)
    assert torch.equal(got, want)
    assert torch.equal(by_view["tail_bank"], by_bank["tail_bank"])
    assert torch.equal(by_view["pool_bank"], by_bank["pool_bank"])
    # Something was written, so the equality is not two untouched copies.
    assert not torch.equal(by_view["tail_bank"], ops["tail_bank"])


def test_forward_requests_refuses_a_position_that_is_not_the_last_token():
    module = _module()
    ops = _ops(LENGTHS[4])
    q_latent = module.project_query_latent(ops["hidden"])
    with pytest.raises(model_fp8.Glm5NextDSAIndexerError, match="seq_lens - 1"):
        module.indexer.forward_requests(
            ops["hidden"], q_latent, ops["pool_bank"], ops["tail_bank"],
            ops["state_slots"], ops["seq_lens"], ops["position"] - 1,
            max_seq_len=MAX_SEQ_LEN)
    with pytest.raises(model_fp8.Glm5NextDSAIndexerError, match="distinct"):
        module.indexer.forward_requests(
            ops["hidden"], q_latent, ops["pool_bank"], ops["tail_bank"],
            ops["state_slots"][[0, 0, 1, 2]], ops["seq_lens"], ops["position"],
            max_seq_len=MAX_SEQ_LEN)
