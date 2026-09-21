# SPDX-License-Identifier: Apache-2.0
"""Tests for the sentinel ordering: the kernel against the torch statement of the same partition.

The reference is the ordering as the model spelled it before the kernel existed: two prefix sums,
a ``where`` and an out-of-place ``scatter``. The partition is a permutation of integers, so the
contract is equality and not a tolerance -- any differing element is a defect.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.cumsum import cumsum
from vllm_neuron.functional.dsa import index_expand as expand_mod
from vllm_neuron.functional.dsa import sentinel_order as seam_mod
from vllm_neuron.functional.dsa.index_expand import dsa_index_expand
from vllm_neuron.functional.dsa.sentinel_order import (
    SBUF_BYTES_PER_PARTITION,
    SEARCH_MAX_FREE,
    can_run_dsa_sentinel_order,
    reset_sentinel_order_dispatch_counters,
    sentinel_order_dispatch_counters,
    sentinel_order_kernel_identity,
    sentinel_order_sbuf_bytes,
)
from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextDSAIndexer


def _ordering(pool_ids: torch.Tensor) -> torch.Tensor:
    """The ordering under test, taken off the indexer so the wiring is measured too."""
    return Glm5NextDSAIndexer._canonical_sentinel_order(pool_ids)


def _reference_order(pool_ids: torch.Tensor) -> torch.Tensor:
    """The ordering as the model spelled it before the kernel, kept verbatim."""
    real = (pool_ids >= 0).to(torch.int32)
    sentinel = 1 - real
    reals = cumsum(real, dim=-1)
    sentinels = cumsum(sentinel, dim=-1)
    destination = torch.where(
        real.bool(), reals - real, reals[:, -1:] + sentinels - sentinel
    ).to(torch.int64)
    return torch.zeros_like(pool_ids).scatter(1, destination, pool_ids)


def _patterns(rows: int, k: int):
    """``(name, pool_ids)`` per sentinel layout worth separating."""
    generator = torch.Generator().manual_seed(rows * 1000 + k)
    ids = torch.randint(0, 1 << 20, (rows, k), generator=generator, dtype=torch.int32)
    mixed = ids.clone()
    mixed[torch.rand(rows, k, generator=generator) < 0.4] = -1
    alternating = ids.clone()
    alternating[:, ::2] = -1
    single_real = torch.full((rows, k), -1, dtype=torch.int32)
    single_real[:, k // 2] = ids[:, k // 2]
    return (
        ("mixed", mixed),
        ("all_sentinel", torch.full((rows, k), -1, dtype=torch.int32)),
        ("no_sentinel", ids.clone()),
        ("single_real", single_real),
        ("alternating", alternating),
    )


def _assert_equal_at(rows: int, k: int) -> None:
    """Every layout at one shape equals the reference order."""
    for name, pool_ids in _patterns(rows, k):
        got = _ordering(pool_ids)
        expected = _reference_order(pool_ids)
        differing = int(torch.ne(got, expected).sum().item())
        assert got.dtype == torch.int32
        assert tuple(got.shape) == (rows, k)
        assert differing == 0, f"[{name}] {differing} of {got.numel()} elements differ"
        assert torch.equal(got, expected)


def _capture(monkeypatch, module):
    """Record every operand tuple a module hands to ``wrap_nki``'s kernel call."""
    seen = []
    real_wrap = module.wrap_nki

    def wrap(kernel):
        run = real_wrap(kernel)

        def call(*operands):
            seen.append(operands)
            return run(*operands)

        return call

    monkeypatch.setattr(module, "wrap_nki", wrap)
    return seen


def test_matches_the_reference_order_at_1_by_512():
    """Decode's one row."""
    _assert_equal_at(1, 512)


def test_matches_the_reference_order_at_127_by_512():
    """One partial partition tile."""
    _assert_equal_at(127, 512)


def test_matches_the_reference_order_at_129_by_512():
    """A full tile and a one-row tail."""
    _assert_equal_at(129, 512)


def test_matches_the_reference_order_at_2048_by_512():
    """The prefill shape: sixteen tiles."""
    _assert_equal_at(2048, 512)


def test_matches_the_reference_order_at_8_by_16():
    """A small width, two search chunks."""
    _assert_equal_at(8, 16)


def test_matches_the_reference_order_at_4_by_1():
    """The narrowest width the gate admits, padded on chip to eight."""
    _assert_equal_at(4, 1)


def test_a_width_that_is_not_a_multiple_of_eight_is_padded_on_chip():
    """Two and twelve columns are padded on chip to eight and sixteen; the order is unchanged."""
    reset_sentinel_order_dispatch_counters()
    calls = 0
    for k in (2, 12):
        for name, pool_ids in _patterns(4, k):
            got = _ordering(pool_ids)
            expected = _reference_order(pool_ids)
            differing = int(torch.ne(got, expected).sum().item())
            calls += 1
            assert tuple(got.shape) == (4, k)
            assert differing == 0, f"[{name}] width {k}: {differing} elements differ"
            assert torch.equal(got, expected)
    assert sentinel_order_dispatch_counters() == (calls, 0)


def test_an_int64_input_takes_the_torch_path_and_still_orders_correctly():
    """int64 ids are not the kernel's, so the torch path serves them and must still be right."""
    reset_sentinel_order_dispatch_counters()
    pool_ids = dict(_patterns(4, 16))["mixed"].to(torch.int64)
    got = _ordering(pool_ids)
    assert sentinel_order_dispatch_counters() == (0, 1)
    assert got.dtype == torch.int64
    assert torch.equal(got, _reference_order(pool_ids))


def test_the_kernel_receives_pool_ids_as_stored(monkeypatch):
    """At the kernel boundary the one operand is ``pool_ids`` itself, uncopied and untransposed."""
    seen = _capture(monkeypatch, seam_mod)
    pool_ids = dict(_patterns(129, 512))["mixed"]
    _ordering(pool_ids)
    assert len(seen) == 1
    operand = seen[0][0]
    assert tuple(operand.shape) == (129, 512)
    assert operand.dtype == torch.int32
    assert operand.data_ptr() == pool_ids.data_ptr()


def test_index_expand_receives_the_ordered_ids_as_stored(monkeypatch):
    """The ordering's output reaches the expand kernel as it was stored, one seam later."""
    mixed = dict(_patterns(129, 16))["mixed"]
    pool_ids = torch.where(mixed >= 0, mixed % 16, mixed)
    ordered = _ordering(pool_ids)
    seen = _capture(monkeypatch, expand_mod)
    seq_lens = torch.full((129,), 4 * 16, dtype=torch.int32)
    dsa_index_expand(ordered, seq_lens, 4)
    assert len(seen) == 1
    operand = seen[0][0]
    assert tuple(operand.shape) == (129, 16)
    assert operand.dtype == torch.int32
    assert operand.data_ptr() == ordered.data_ptr()


def test_kernel_identity_after_dispatch_names_the_nki_kernel():
    """After a dispatch the identity names the seam's own kernel, not the decorator."""
    reset_sentinel_order_dispatch_counters()
    _ordering(dict(_patterns(8, 16))["mixed"])
    assert sentinel_order_kernel_identity() == (
        "vllm_neuron.functional.dsa.sentinel_order",
        "_sentinel_order_nki",
    )


def test_the_gate_ceiling_is_the_widest_row_whose_tiles_fit_one_partition():
    """The gate's ceiling is the widest row whose footprint fits, and 2048 sits under it."""
    ceiling = max(
        k for k in range(1, 16385) if sentinel_order_sbuf_bytes(k) <= SBUF_BYTES_PER_PARTITION
    )
    assert SEARCH_MAX_FREE == ceiling
    assert sentinel_order_sbuf_bytes(ceiling + 1) > SBUF_BYTES_PER_PARTITION
    assert 2048 <= SEARCH_MAX_FREE
    assert can_run_dsa_sentinel_order(torch.zeros((1, 2048), dtype=torch.int32))
    assert not can_run_dsa_sentinel_order(
        torch.zeros((1, SEARCH_MAX_FREE + 1), dtype=torch.int32)
    )
