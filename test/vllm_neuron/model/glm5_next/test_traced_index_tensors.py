"""Guards on the tensors the traced path builds, and on the two index rewrites.

Graph extraction traces this model under fake tensors. A tensor built from python DATA is a
real tensor even there, so the tracer refuses an operator that is handed one beside fake
ones. The items here read the model file for that construction, reproduce the refusal and
its absence on the two forms, and check that the two index rewrites carry the same values as
the row lists they replaced.
"""

import ast
import pathlib

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from vllm_neuron.model.glm5_next.factory import partition_experts

pytestmark = [pytest.mark.fast, pytest.mark.forked]

SENT = "FAKEIDX"

#: Both build from python data, so both are real even under a fake trace. ``as_tensor`` is
#: NOT here: it passes a tensor argument straight through, and the two call sites that can
#: reach it with an int are out of this file's scope by decision.
DATA_BUILDERS = ("tensor", "new_tensor")

#: The published routed-expert count and the expert-parallel degree the serving run uses.
EXPERTS, EP_DEGREE = 288, 16

#: Variable, single, one empty request, all but one empty, and uniform.
RAGGED_CASES = ((3, 1, 2), (4,), (3, 0, 2), (0, 0, 5), (2, 2))

REFUSAL = "convert all Tensors to FakeTensors"

#: The device the worker traces on (``neuron_model_runner.py:1367``, ``:1458``),
#: and the device the refusal was recorded on. It is load-bearing: on the CPU a
#: data-built tensor is converted to a fake one inside the mode and ``where``
#: takes it, so a CPU fixture would read no refusal on any tree.
TRACED_DEVICE = "meta"


def say(*fields):
    """Print one machine-readable row, prefixed so a launcher can anchor on it."""
    print(f"{SENT}|" + "|".join(str(field) for field in fields))


def data_built_calls(path):
    """``(line, spelling)`` per python-data tensor build outside ``__init__`` in ``path``."""
    found = []

    def walk(node, where):
        for child in ast.iter_child_nodes(node):
            func = getattr(child, "func", None)
            if isinstance(func, ast.Attribute) and func.attr in DATA_BUILDERS:
                torch_call = getattr(func.value, "id", None) == "torch"
                if (torch_call or func.attr == "new_tensor") and where != "__init__":
                    found.append((child.lineno, f"{'torch.' if torch_call else ''}{func.attr}"))
            inner = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            walk(child, child.name if inner else where)

    walk(ast.parse(path.read_text()), "<module>")
    return found


def where_refuses(mask, index, other):
    """``(refused, message)`` for one ``torch.where`` under whatever mode is open."""
    try:
        torch.where(mask, index, other)
    except Exception as error:  # the tracer's refusal is the reading, not a failure here
        return True, str(error)
    return False, ""


def packed_row_index(lengths, max_len, device):
    """The row index the pack uses now: one ``arange`` and a slice per request."""
    grid = torch.arange(max_len, dtype=torch.int64, device=device)
    return torch.cat([grid[:n] + b * max_len for b, n in enumerate(lengths)])


def packed_row_list(lengths, max_len, device):
    """The row list it replaced, quoted as the oracle it has to agree with."""
    return torch.tensor(
        [b * max_len + r for b, n in enumerate(lengths) for r in range(n)],
        dtype=torch.int64,
        device=device,
    )


def test_a_the_model_file_builds_no_tensor_from_python_data_outside_init() -> None:
    """The model file this process imported carries no such build outside ``__init__``."""
    from vllm_neuron.model.glm5_next import model_fp8

    path = pathlib.Path(model_fp8.__file__)
    found = data_built_calls(path)
    named = ",".join(f"{line}:{spelling}" for line, spelling in found)
    say("census", f"file={path}", f"data_built={len(found)}", f"lines={named or 'none'}")
    assert not found, (
        f"a tensor built from python data on a traced path is real under graph "
        f"extraction, which refuses it beside fake ones; read {named} in {path}"
    )


def test_b_a_data_built_tensor_is_refused_where_a_factory_built_one_is_not() -> None:
    """The refusal is reproduced on the form that was replaced, and absent on its successor."""
    with FakeTensorMode():
        mask = torch.zeros(2048, dtype=torch.bool, device=TRACED_DEVICE)
        index = torch.zeros(2048, dtype=torch.int64, device=TRACED_DEVICE)
        data_built = torch.tensor(7, dtype=torch.int64, device=TRACED_DEVICE)
        factory_built = index.new_full((), 7, dtype=torch.int64)
        ranged = torch.arange(3, dtype=torch.int64, device=TRACED_DEVICE)

        say("kinds",
            f"data_built_is_fake={isinstance(data_built, FakeTensor)}",
            f"factory_built_is_fake={isinstance(factory_built, FakeTensor)}",
            f"arange_is_fake={isinstance(ranged, FakeTensor)}")

        old_refused, old_message = where_refuses(mask, index, data_built)
        new_refused, new_message = where_refuses(mask, index, factory_built)

    say("refusal",
        f"data_built_refused={old_refused}",
        f"carries_the_text={REFUSAL in old_message}",
        f"factory_built_refused={new_refused}",
        f"message={new_message or 'none'}")
    assert old_refused and REFUSAL in old_message, (
        f"the data-built form was expected to be refused with {REFUSAL!r}; "
        f"read refused={old_refused} message={old_message!r}"
    )
    assert not new_refused, f"the factory-built form was refused: {new_message!r}"


def test_c_every_expert_rank_owns_a_contiguous_ascending_run() -> None:
    """The premise of the expert index rewrite: each rank's indices are one run."""
    partition = partition_experts(EXPERTS, EP_DEGREE)
    covered, ragged = [], []
    for rank in range(EP_DEGREE):
        owned = partition.local_expert_indices(rank)
        if owned != tuple(range(owned[0], owned[0] + len(owned))):
            ragged.append(rank)
        covered.extend(owned)
    say("partition",
        f"experts={EXPERTS}", f"degree={EP_DEGREE}", f"per_rank={len(covered) // EP_DEGREE}",
        f"ragged_ranks={ragged or 'none'}", f"covers_every_expert_once={sorted(covered) == list(range(EXPERTS))}")
    assert not ragged, (
        f"ranks {ragged} own indices that are not one ascending run, so an arange over "
        f"the first and the count would not be the indices they own"
    )
    assert sorted(covered) == list(range(EXPERTS))


def test_d_the_packed_row_index_matches_the_row_list_it_replaced() -> None:
    """Every boundary case builds the same rows, in the same order, as the list did."""
    device = torch.device("cpu")
    disagreed = []
    for lengths in RAGGED_CASES:
        for max_len in (max(lengths), max(lengths) + 3):
            built = packed_row_index(lengths, max_len, device)
            listed = packed_row_list(lengths, max_len, device)
            same = built.dtype == listed.dtype and built.tolist() == listed.tolist()
            say("rows", f"lengths={lengths}", f"max_len={max_len}",
                f"rows={built.tolist()}", f"agrees={same}")
            if not same:
                disagreed.append((lengths, max_len))
    assert not disagreed, f"the index and the list it replaced disagree at {disagreed}"
