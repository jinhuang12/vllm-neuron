# SPDX-License-Identifier: Apache-2.0
"""The sentinel ordering reaches its NKI kernel as stored and equals the counted torch order.

The reference is the ordering as the model spelled it before the kernel: two prefix sums
through the fork's own cumsum, a where and an out-of-place scatter. Equality is the contract:
the partition is a permutation of integers, so any differing element is a defect. Run under
``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``; nothing here reads or sets an environment variable.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import textwrap

import torch

from vllm_neuron.functional.cumsum import cumsum
from vllm_neuron.functional.dsa import index_expand as expand_mod
from vllm_neuron.functional.dsa.index_expand import dsa_index_expand
from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextDSAIndexer

_SEAM = "vllm_neuron.functional.dsa.sentinel_order"
_SHAPES = ((1, 512), (127, 512), (129, 512), (2048, 512), (8, 16))
_HOST_FORMS = (r"\.scatter\(", r"torch\.where\(", r"cumsum\(")
_NO_HOST_FORMS = (0,) * len(_HOST_FORMS)


def _emit(tag: str, **values: object) -> None:
    """Print one machine-readable reading line for the transcript's reader."""
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"SOT|{tag}|{body}", flush=True)


def _ordering(pool_ids: torch.Tensor) -> torch.Tensor:
    """The ordering under test, taken off the indexer so the wiring is measured too."""
    return Glm5NextDSAIndexer._canonical_sentinel_order(pool_ids)


def _frozen_order(pool_ids: torch.Tensor) -> torch.Tensor:
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
    """Every layout at one shape equals the frozen order; differing elements are the reading."""
    for name, pool_ids in _patterns(rows, k):
        got = _ordering(pool_ids)
        want = _frozen_order(pool_ids)
        differing = int(torch.ne(got, want).sum().item())
        _emit("EQUAL", rows=rows, k=k, pattern=name, equal=torch.equal(got, want),
              differing=differing, dtype=got.dtype, shape=tuple(got.shape))
        assert got.dtype == torch.int32
        assert tuple(got.shape) == (rows, k)
        assert differing == 0
        assert torch.equal(got, want)


def _method_code() -> str:
    """The method's own statements, docstring removed, so a word in prose is not a call."""
    source = textwrap.dedent(inspect.getsource(Glm5NextDSAIndexer._canonical_sentinel_order))
    function = ast.parse(source).body[0]
    statements = [
        node for node in function.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    ]
    return "\n".join(ast.unparse(node) for node in statements)


def _host_form_counts(text: str) -> tuple[int, ...]:
    """How many times each host form the compiler laid out occurs in a text."""
    return tuple(len(re.findall(form, text)) for form in _HOST_FORMS)


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


def test_equal_to_the_frozen_order_at_1_by_512():
    """Decode's one row."""
    _assert_equal_at(1, 512)


def test_equal_to_the_frozen_order_at_127_by_512():
    """One partial partition tile."""
    _assert_equal_at(127, 512)


def test_equal_to_the_frozen_order_at_129_by_512():
    """A full tile and a one-row tail."""
    _assert_equal_at(129, 512)


def test_equal_to_the_frozen_order_at_2048_by_512():
    """The prefill shape: sixteen tiles."""
    _assert_equal_at(2048, 512)


def test_equal_to_the_frozen_order_at_8_by_16():
    """A small width, two search chunks."""
    _assert_equal_at(8, 16)


def test_no_host_scatter_on_the_nki_route():
    """The method's code holds no scatter, where or cumsum: nothing for the compiler to lay out."""
    code = _method_code()
    counts = _host_form_counts(code)
    seam_calls = len(re.findall(r"dsa_sentinel_order\(", code))
    _emit("NO_HOST_FORMS", forms=len(_HOST_FORMS), counts=counts, seam_calls=seam_calls)
    assert counts == _NO_HOST_FORMS
    assert seam_calls == 1


def test_control_the_host_form_reader_fires_on_a_planted_form():
    """The same reader counts one of each form on a text that carries them."""
    planted = "\n".join((
        "out = torch.zeros_like(x).scatter(1, d, x)",
        "d = torch.where(m, a, b)",
        "r = cumsum(m, dim=-1)",
    ))
    counts = _host_form_counts(planted)
    _emit("CONTROL_HOST_FORM_READER_FIRES", counts=counts)
    assert counts == (1,) * len(_HOST_FORMS)


def test_the_kernel_receives_pool_ids_as_stored(monkeypatch):
    """At the kernel boundary the one operand is ``pool_ids`` itself, uncopied and untransposed."""
    seam = importlib.import_module(_SEAM)
    seen = _capture(monkeypatch, seam)
    pool_ids = dict(_patterns(129, 512))["mixed"]
    _ordering(pool_ids)
    assert len(seen) == 1
    operand = seen[0][0]
    _emit("AS_STORED", shape=tuple(operand.shape), dtype=operand.dtype,
          same_storage=operand.data_ptr() == pool_ids.data_ptr(), operands=len(seen[0]))
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
    _emit("EXPAND_AS_STORED", shape=tuple(operand.shape), dtype=operand.dtype,
          same_storage=operand.data_ptr() == ordered.data_ptr())
    assert tuple(operand.shape) == (129, 16)
    assert operand.dtype == torch.int32
    assert operand.data_ptr() == ordered.data_ptr()


def test_the_five_shapes_take_nki_with_zero_fallback():
    """One reset window over the five shapes: five NKI dispatches, no torch fallback."""
    seam = importlib.import_module(_SEAM)
    seam.reset_sentinel_order_dispatch_counters()
    for rows, k in _SHAPES:
        _ordering(dict(_patterns(rows, k))["mixed"])
    nki_n, fallback_n = seam.sentinel_order_dispatch_counters()
    _emit("DISPATCH", nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert (nki_n, fallback_n) == (len(_SHAPES), 0)


def test_a_width_the_gate_refuses_takes_the_oracle_and_equals_the_frozen_order():
    """Twelve columns is not a multiple of the search width: the oracle serves it, counted."""
    seam = importlib.import_module(_SEAM)
    seam.reset_sentinel_order_dispatch_counters()
    pool_ids = dict(_patterns(4, 12))["mixed"]
    got = _ordering(pool_ids)
    nki_n, fallback_n = seam.sentinel_order_dispatch_counters()
    _emit("FALLBACK", k=12, nki_dispatch=nki_n, torch_fallback=fallback_n,
          equal=torch.equal(got, _frozen_order(pool_ids)))
    assert (nki_n, fallback_n) == (0, 1)
    assert torch.equal(got, _frozen_order(pool_ids))


def test_kernel_identity_after_dispatch_names_the_nki_kernel():
    """After a dispatch the identity names the seam's own kernel, not the decorator."""
    seam = importlib.import_module(_SEAM)
    seam.reset_sentinel_order_dispatch_counters()
    _ordering(dict(_patterns(8, 16))["mixed"])
    identity = seam.sentinel_order_kernel_identity()
    _emit("IDENTITY_AFTER", identity=identity)
    assert identity == (_SEAM, "_sentinel_order_nki")
