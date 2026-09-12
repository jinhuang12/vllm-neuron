# SPDX-License-Identifier: Apache-2.0
"""The sentinel ordering, spelled without a sort.

The graph compiler has no ``sort`` for this target, so the ordering is counted rather
than compared: each id's destination comes from exclusive prefix sums of one mask. The
items here hold the counted form against the argsort form it replaces, over the row and
width shapes the two refused graphs carried, and read the method's body for the call
that can no longer appear in it.
"""

import ast
import inspect
import pathlib
import textwrap

import pytest
import torch

pytestmark = [pytest.mark.fast, pytest.mark.forked]

SENT = "SORTFREE"

#: The method under test, located by name so a moved line changes nothing here.
METHOD = "_canonical_sentinel_order"

#: The method calls this target lowers none of, read as METHOD calls so the free
#: ``cumsum`` this package ships in place of ``torch.cumsum`` is not caught with them.
FORBIDDEN_METHODS = ("sort", "argsort", "msort", "topk", "cumsum")

#: The row counts the two refused graphs carried, plus a small one a reader can check by
#: hand. 2048 is prefill's token count and 1 is decode's.
ROWS = (1, 7, 2048)

#: The widths: the refused graphs' 512, and a small one whose rows fit in a message.
WIDTHS = (8, 512)


def say(*fields):
    """Print one machine-readable row, prefixed so a launcher can anchor on it."""
    print(f"{SENT}|" + "|".join(str(field) for field in fields))


def argsort_reference(pool_ids):
    """The ordering as the pre-repair tree spelled it, kept here as the oracle."""
    k = int(pool_ids.shape[1])
    position = torch.arange(k, device=pool_ids.device, dtype=torch.int64)
    key = (pool_ids < 0).to(torch.int64) * k + position
    return pool_ids.gather(1, key.argsort(dim=1))


def patterns(rows, k):
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


@pytest.fixture(scope="module")
def ordering():
    """The ordering under test, taken off the indexer rather than re-implemented."""
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextDSAIndexer

    return getattr(Glm5NextDSAIndexer, METHOD)


def test_a_the_counted_order_equals_the_argsort_order(ordering) -> None:
    """Every shape and every sentinel layout gives the argsort form's own answer."""
    checked = 0
    for rows in ROWS:
        for k in WIDTHS:
            for name, pool_ids in patterns(rows, k):
                got = ordering(pool_ids)
                want = argsort_reference(pool_ids)
                say("case", f"rows={rows}", f"k={k}", name,
                    f"equal={bool(torch.equal(got, want))}",
                    f"dtype={got.dtype}", f"shape={tuple(got.shape)}")
                assert torch.equal(got, want), (
                    f"rows={rows} k={k} {name}: the counted order differs from the "
                    f"argsort order it replaces"
                )
                checked += 1
    say("cases_checked", checked)


def test_b_the_result_keeps_the_inputs_dtype_and_shape(ordering) -> None:
    """The ordering permutes and does not cast: the sentinel stage hands int32.

    The brief asked for int64 preserved. The call site casts to int32
    (``pool_ids = indices.to(torch.int32)``), so both widths are read here and each is
    wanted back unchanged -- the property is preservation, whichever the caller hands.
    """
    for dtype in (torch.int32, torch.int64):
        pool_ids = torch.tensor([[5, -1, 7, -1]], dtype=dtype)
        got = ordering(pool_ids)
        say("dtype_case", f"in={dtype}", f"out={got.dtype}", f"row={got.tolist()}")
        assert got.dtype == dtype, f"input {dtype} came back as {got.dtype}"
        assert tuple(got.shape) == tuple(pool_ids.shape)


def test_c_the_method_body_calls_no_unsupported_operation(ordering) -> None:
    """Both refused operations are absent, and the fork's own cumsum is present.

    Read as a census over the method's own source rather than over the file, so an
    argsort elsewhere in the model does not answer for this one and a line moving
    changes nothing. The census counts METHOD calls, because that is what tells
    ``x.cumsum(...)`` -- which this target has no lowering for -- apart from the
    ``cumsum(x)`` this package ships to replace it.
    """
    source = textwrap.dedent(inspect.getsource(ordering))
    path = pathlib.Path(inspect.getsourcefile(ordering))
    methods, names = [], []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            methods.append(node.func.attr)
        elif isinstance(node.func, ast.Name):
            names.append(node.func.id)
    refused = [m for m in methods if m in FORBIDDEN_METHODS]
    say("body_census", f"file={path.name}", f"method={METHOD}",
        f"forbidden_calls={refused}", f"free_cumsum_calls={names.count('cumsum')}")
    assert not refused, (
        f"{METHOD} still calls {refused} as a method; this target lowers none of them"
    )
    assert names.count("cumsum") == 2, (
        f"{METHOD} makes {names.count('cumsum')} free cumsum calls, not the two the "
        f"partition needs -- the counts must go through the fork's own cumsum"
    )
