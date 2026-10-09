# SPDX-License-Identifier: Apache-2.0
"""The blocks one request holds in a KV cache group on one rank, at every decode context
parallel size.

``kv_group_blocks.group_blocks_per_request`` is the per-group figure the need-sized KV
pool and the runner's block-table rows are priced with: the sequence's pages plus the
group's draft blocks. At ``dcp = 1`` it must be, integer for integer, the figure both
callers compute today, ``cdiv(max_model_len, block_size) + draft_blocks_per_request(spec)``.
Under decode context parallelism (``dcp > 1``) an attention group keeps ``1 / dcp`` of
the tokens on each rank, vLLM's own per-rank rule (``FullAttentionSpec.
max_memory_usage_bytes``); a recurrent (``MambaSpec``) group keeps one state slot per
sequence on every rank and is not divided.

Expectations are derived from vLLM's own grouping and per-spec figures on the served
model's layer specs (``test_kv_budget_glm53f``) and on synthetic specs of every class
the pricing serves; no served figure is an expectation.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_kv_group_blocks.py
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1 import kv_cache_interface as kvi

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv
from test.vllm_neuron.worker.test_kv_budget_drafter_seam import (
    K,
    _groups,
    _recurrent_layers_at_two_geometries,
    _worker,
)
from test.vllm_neuron.worker.test_mtp_kv_budget import _mtp
from vllm_neuron.vllm.worker import kv_group_blocks as KB

#: Every sequence length the KV pool and block-table tests serve (512, 4096, 8192), the
#: block-boundary neighbours of the 128-token block, 384 (whole blocks per rank at
#: ``dcp = 3``), and the long-context lines.
CONTEXTS = (1, 127, 128, 129, 255, 256, 384, 512, 1000, 4096, 4097, 8192, 65536, 131072,
            1048576)
#: The decode context parallel sizes the hybrid KV cache is resolved at, and a non-power.
DCPS = (2, 3, 4, 8)


def _attention(block_size: int, cls=kvi.FullAttentionSpec, **extra):
    return cls(block_size=block_size, num_kv_heads=1, head_size=576, dtype=torch.bfloat16,
               **extra)


def _mamba(block_size: int, drafts: int):
    return kvi.MambaSpec(block_size=block_size, shapes=((4, 128, 128),),
                         dtypes=(torch.bfloat16,), num_speculative_blocks=drafts)


def _uniform(*members):
    return kvi.UniformTypeKVCacheSpecs(
        block_size=members[0].block_size,
        kv_cache_specs={f"layers.{i}": spec for i, spec in enumerate(members)})


#: (name, spec) for every class the pricing serves, at several block sizes.
SYNTHETIC = [
    *((f"full_attention_b{b}", _attention(b)) for b in (16, 64, 128, 4096)),
    *((f"mla_b{b}", _attention(b, kvi.MLAAttentionSpec)) for b in (16, 128)),
    ("sliding_window_b128", _attention(128, kvi.SlidingWindowSpec, sliding_window=1024)),
    *((f"mamba_b{b}_k{k}", _mamba(b, k)) for b in (128, 4096) for k in (0, 1, K)),
    ("uniform_attention_b128",
     _uniform(_attention(128), _attention(128, kvi.MLAAttentionSpec))),
    ("uniform_mamba_b4096_k3", _uniform(_mamba(4096, K), _mamba(4096, K))),
]


def _fixture_groups():
    """The served model's KV cache groups as vLLM groups them: plain and drafting, with
    and without the drafter's attention layer, and the uniform-type recurrent layout."""
    out = []
    for drafter, k in ((False, 0), (True, 0), (True, K)):
        worker = _worker(drafter=drafter, k=k, seqs=1, length=4096)
        out += [(f"drafter{int(drafter)}_k{k}_group{i}", group.kv_cache_spec)
                for i, group in enumerate(_groups(worker))]
    layers, text_config = _recurrent_layers_at_two_geometries()
    runner = kv.fake_runner(layers, max_num_seqs=1, max_model_len=4096,
                            text_config=text_config)
    _mtp(runner, K)
    out += [(f"uniform_k{K}_group{i}", group.kv_cache_spec)
            for i, group in enumerate(_groups(kv.fake_worker(runner)))]
    return out


def _pre_change_figure(spec, max_model_len: int, where: str) -> int:
    """The per-group figure both callers price at ``dcp = 1``: the sequence's pages plus
    the group's draft blocks."""
    return cdiv(max_model_len, spec.block_size) + KB.draft_blocks_per_request(spec, where)


def _vllm_attention_pages(spec, max_model_len: int, dcp: int) -> int:
    """vLLM's own per-rank pages of a full-attention spec at ``dcp``."""
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp,
                                        prefill_context_parallel_size=1))
    return spec.max_memory_usage_bytes(vllm_config) // spec.page_size_bytes


def test_dcp_one_is_the_figure_the_callers_price_today() -> None:
    """At ``dcp = 1`` (default and explicit) every spec class, block size and context
    gives the integer ``cdiv(max_model_len, block_size) + draft blocks``."""
    specs = SYNTHETIC + _fixture_groups()
    assert {type(spec) for _, spec in specs} >= {
        kvi.FullAttentionSpec, kvi.MLAAttentionSpec, kvi.SlidingWindowSpec, kvi.MambaSpec,
        kvi.UniformTypeKVCacheSpecs}
    for name, spec in specs:
        for length in CONTEXTS:
            want = _pre_change_figure(spec, length, name)
            got = KB.group_blocks_per_request(spec, length, name)
            assert got == want, (name, length)
            got = KB.group_blocks_per_request(spec, length, name, dcp=1)
            assert got == want, (name, length)


@pytest.mark.parametrize("dcp", DCPS)
def test_an_attention_group_keeps_its_share_of_the_tokens_on_each_rank(dcp: int) -> None:
    """An attention group's pages per rank are ``cdiv(max_model_len, block_size * dcp)``,
    vLLM's own per-rank figure; it holds no draft blocks."""
    for name, spec in SYNTHETIC:
        if not isinstance(spec, kvi.AttentionSpec):
            continue
        for length in CONTEXTS:
            got = KB.group_blocks_per_request(spec, length, name, dcp=dcp)
            assert got == cdiv(length, spec.block_size * dcp), (name, length)
            if isinstance(spec, kvi.FullAttentionSpec):
                assert got == _vllm_attention_pages(spec, length, dcp), (name, length)


@pytest.mark.parametrize("dcp", DCPS)
def test_a_recurrent_group_is_not_divided_over_the_ranks(dcp: int) -> None:
    """A ``MambaSpec`` group keeps one state slot per sequence on every rank: its figure
    at any ``dcp`` is its ``dcp = 1`` figure, draft blocks included."""
    for name, spec in SYNTHETIC:
        if not isinstance(spec, kvi.MambaSpec):
            continue
        for length in CONTEXTS:
            assert (KB.group_blocks_per_request(spec, length, name, dcp=dcp)
                    == _pre_change_figure(spec, length, name)), (name, length)


@pytest.mark.parametrize("dcp", DCPS)
def test_a_uniform_type_group_is_divided_as_its_layers_are(dcp: int) -> None:
    by_name = dict(SYNTHETIC)
    attention = by_name["uniform_attention_b128"]
    recurrent = by_name["uniform_mamba_b4096_k3"]
    for length in CONTEXTS:
        assert KB.group_blocks_per_request(attention, length, "attention", dcp=dcp) == cdiv(
            length, 128 * dcp)
        assert KB.group_blocks_per_request(recurrent, length, "recurrent", dcp=dcp) == cdiv(
            length, 4096) + K


@pytest.mark.parametrize("dcp", DCPS)
def test_the_served_drafting_layout_per_rank(dcp: int) -> None:
    """The served drafting line (``--mamba-block-size`` = ``max_model_len``, ``k`` drafts):
    the attention group holds its per-rank pages, every recurrent group ``1 + k`` blocks,
    the same on every rank as at ``dcp = 1``."""
    length = 4096
    worker = _worker(drafter=True, k=K, seqs=1, length=length)
    groups = _groups(worker)
    assert any(isinstance(g.kv_cache_spec, kvi.MambaSpec) for g in groups)
    for index, group in enumerate(groups):
        spec = group.kv_cache_spec
        got = KB.group_blocks_per_request(spec, length, f"group {index}", dcp=dcp)
        if isinstance(spec, kvi.MambaSpec):
            assert spec.block_size == length
            assert got == 1 + K
        else:
            assert got == _vllm_attention_pages(spec, length, dcp)


@pytest.mark.parametrize("dcp", [0, -1, 1.0, True, None], ids=["zero", "negative", "float",
                                                              "bool", "none"])
def test_a_dcp_that_is_not_a_positive_int_is_refused_by_name(dcp) -> None:
    with pytest.raises(ValueError, match="dcp"):
        KB.group_blocks_per_request(_attention(128), 4096, "KV cache group 0", dcp=dcp)


def test_a_spec_class_with_no_pricing_is_refused_by_name_at_every_dcp() -> None:
    class ForeignSpec(kvi.KVCacheSpec):
        @property
        def page_size_bytes(self) -> int:
            return 131072

    for dcp in (1, *DCPS):
        with pytest.raises(ValueError, match="ForeignSpec"):
            KB.group_blocks_per_request(ForeignSpec(block_size=128), 4096,
                                        "KV cache group 0", dcp=dcp)


def test_a_group_of_attention_and_recurrent_layers_is_refused_past_dcp_one() -> None:
    """Layers that vLLM does not merge into one group: at ``dcp > 1`` one kind divides over
    the ranks and the other does not, so the group has no one figure. At ``dcp = 1`` both
    kinds price the same pages and the figure stays the one the callers price today."""
    mixed = _uniform(_attention(128), _mamba(128, 0))
    where = "KV cache group 0"
    assert (KB.group_blocks_per_request(mixed, 4096, where)
            == _pre_change_figure(mixed, 4096, where))
    for dcp in DCPS:
        with pytest.raises(ValueError, match="UniformTypeKVCacheSpecs"):
            KB.group_blocks_per_request(mixed, 4096, where, dcp=dcp)
