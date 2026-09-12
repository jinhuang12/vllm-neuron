# SPDX-License-Identifier: Apache-2.0
"""The rank the decode step's write keeps, on both sides of the assignment.

The decode leg writes one token's head output into a rank-2 float32 buffer through a
slice. Eager torch broadcasts a rank-1 value into that slice; the graph compiler lowers
the write as a slice write-back and refuses a source whose rank differs from the
target's. The items here read the two shapes the assignment brings together, and
read the model file for the form the write is spelled in.
"""

import ast
import pathlib

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

pytestmark = [pytest.mark.fast, pytest.mark.forked]

SENT = "CORERANK"

#: The device the worker traces on, which is where the refusal was recorded.
TRACED_DEVICE = "meta"

#: One head's extent and two heads of it, so the first head's slice is ``0:KDIM`` --
#: the extent the compiler named when it refused this write.
KDIM, HEADS = 128, 2

#: The decode leg's token count. One token is what makes the target's first extent one.
TOKENS = 1

#: The buffer the write lands in, as the model allocates it.
BUFFER_DTYPE = torch.float32


def say(*fields):
    """Print one machine-readable row, prefixed so a launcher can anchor on it."""
    print(f"{SENT}|" + "|".join(str(field) for field in fields))


def buffer_writes(buffer_name):
    """``(line, first_dim_is_sliced, source_reshape_args)`` per write into that buffer.

    Found by walking the model file for assignments whose target subscripts the named
    buffer with a tuple, so each reading follows the name rather than an address that
    moves whenever a line is added above it.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    path = pathlib.Path(model_fp8.__file__)
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0] if node.targets else None
        if not isinstance(target, ast.Subscript):
            continue
        if getattr(target.value, "id", None) != buffer_name:
            continue
        if not isinstance(target.slice, ast.Tuple):
            continue
        first = target.slice.elts[0]
        source = node.value
        args = None
        if isinstance(source, ast.Call):
            if getattr(source.func, "attr", "") == "reshape":
                args = len(source.args)
        found.append((node.lineno, isinstance(first, ast.Slice), args))
    return path, found


def assert_every_write_keeps_its_rank(buffer_name, extents):
    """Each write into the buffer slices dim 0 and shapes its source to ``extents``."""
    path, found = buffer_writes(buffer_name)
    say("census", f"file={path.name}", f"buffer={buffer_name}", f"writes={len(found)}",
        f"extents_wanted={extents}", f"rows={found}")
    assert found, f"no write into {buffer_name} was found in {path}"
    for line, first_is_sliced, reshape_args in found:
        assert first_is_sliced, (
            f"the write at line {line} indexes {buffer_name}'s first dimension with a "
            f"single position, which drops a dimension the target needs and leaves the "
            f"source's rank to match by accident"
        )
        assert reshape_args == extents, (
            f"the write at line {line} shapes its source with {reshape_args} extents; "
            f"the target slice has {extents}"
        )


def test_a_the_written_value_and_the_target_slice_agree_in_rank() -> None:
    """The flattened source disagrees with the target slice; the kept one agrees."""
    span = slice(0, KDIM)
    with FakeTensorMode():
        core = torch.empty(
            TOKENS, HEADS * KDIM, dtype=BUFFER_DTYPE, device=TRACED_DEVICE
        )
        step_output = torch.empty(1, KDIM, dtype=BUFFER_DTYPE, device=TRACED_DEVICE)
        target = core[0:1, span]
        flattened = step_output.reshape(-1)
        kept = step_output.reshape(1, KDIM)
        # WHAT THE MODE DOES WITH THE REFUSED FORM, recorded rather than asserted. The
        # compiler's check is on the two shapes, and this row is why the item reads
        # those shapes instead of waiting for the mode to raise.
        try:
            core[0:1, span] = flattened
            mode_refused = False
        except Exception as error:  # the reading is whether it raises at all
            mode_refused = repr(error)[:80]

        flattened_agrees = tuple(flattened.shape) == tuple(target.shape)
        kept_agrees = tuple(kept.shape) == tuple(target.shape)

    say("shapes", f"target={tuple(target.shape)}",
        f"flattened={tuple(flattened.shape)}", f"kept={tuple(kept.shape)}",
        f"flattened_agrees={flattened_agrees}",
        f"kept_agrees={kept_agrees}", f"mode_refused={mode_refused}")

    assert not flattened_agrees, (
        f"a flattened source {tuple(flattened.shape)} must disagree with the target "
        f"slice {tuple(target.shape)}, which is the disagreement the graph compiler "
        f"refuses"
    )
    assert kept_agrees, (
        f"the rank-2 source {tuple(kept.shape)} must agree with the target slice "
        f"{tuple(target.shape)} in every extent"
    )


def test_b_every_write_into_the_head_buffer_keeps_both_sides_rank_two() -> None:
    """The head buffer is rank 2, so every write into it carries two extents."""
    assert_every_write_keeps_its_rank("core", 2)


def test_c_every_write_into_the_indexer_ring_keeps_both_sides_rank_three() -> None:
    """The ring is rank 3, and its two seeding writes are the same class as item b's.

    They are on the leg whose position is a python int, which the tensor route does
    not take, so no run has refused them yet. The rank is asserted anyway: a later
    graph reaching this leg would refuse it for the reason item a reads.
    """
    assert_every_write_keeps_its_rank("tail", 3)
