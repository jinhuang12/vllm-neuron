# SPDX-License-Identifier: Apache-2.0
"""The NKI front end accepts every shipped sparse-attention entry point, paged and unpaged.

The front end parses a kernel's own source before it places one tile, and it refuses a call
form outside its subset. The simulator never parses that source -- it runs the body as plain
Python -- so no simulator test can see such a refusal. This file compiles the entries
instead, with fake operands of the served shapes, inside a child process that pins the
platform target in its environment. The target makes the toolchain read what it builds for
from the environment, so the compile opens no device node, and the child counts its own open
device nodes to show it.

The signatures are read before the compile, and the reason is load-bearing: ``wrap_nki``
binds a call by the parameters the entry declares, in their declared order, and it DROPS
extra positional operands without a word. Against an entry that does not declare the paged
operands, a paged call compiles the unpaged kernel and reports no refusal at all.

The paged path serves no RoPE half. ``k_pe`` carries one row per KV row, exactly as the
latent cache does, so a paged latent window would need a paged RoPE window beside it. This
checkpoint's RoPE width is 0 and the limb is elided at trace time, so the seam refuses that
combination by name rather than growing a second staging path nothing exercises.
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib
import subprocess
import sys

import nki
import nki.language as nl

from vllm_neuron.functional.attention import mla_sparse as MS

ROW = "sparse_frontend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
# The child writes no bytecode: it is the first thing to import the compiler in an
# environment, and a cache file left in a shared installation is a change to it that these
# tests do not intend.
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1"}

# The served widths: this checkpoint's latent rank and head count, the selected-row count the
# decode path passes, the KV block, and the model length in blocks. The untiled and
# latent-tiled bodies are reached by the widths their own gate names, because the seam picks
# the body from the widths alone.
LATENT, HEADS, TOPK_ROWS, PAGE, PAGES = 512, 64, 2048, 128, 32
TOPK_NARROW, LATENT_RAGGED, ROPE = 512, 640, 64

# The three no-RoPE entries the paged window is assembled in, and the operands they have to
# declare for a paged call to reach the front end at all.
NOPE_ENTRIES = ("mla_sparse_attention_nope_row_tiled_kernel", "mla_sparse_attention_nope_kernel",
                "mla_sparse_attention_nope_tiled_kernel")
PAGED_PARAMETERS = ("block_table_hbm", "written_hbm", "write_offset_hbm", "page_size")

# The parameter the paged operands follow. Reading them from here rather than from the end of
# the signature leaves room for the compile-time tile options the row-tiled entries declare
# after them.
SCALE_PARAMETER = "softmax_scale"

# The two entries that declare compile-time tile options, and the option values production
# runs at. No caller passes either option -- the seam calls the entry with operands alone --
# so these defaults are what every served call compiles.
ROW_TILED_ENTRIES = ("mla_sparse_attention_nope_row_tiled_kernel",
                     "mla_sparse_attention_rope_row_tiled_kernel")
SERVED_TILE_DEFAULTS = {"BLOCK_N": 512, "STREAM_KV": True}

# The served latent cache is bf16 and holds more than one window, and the 16-row DMA
# transposes specialise on the operand dtype -- 16 rows of 2 bytes fill the line 8 rows of 4
# bytes do. So the served dtype is compiled first, f32 beside it, and the bank larger than
# the window read through it.
SERVED_DTYPE = "bfloat16"
BANK_WINDOWS = 2


def _write_row(*fields: object) -> None:
    """Print one row on the child's stdout, which is how the child reports to the parent."""
    print(ROW + "|" + "|".join(str(field) for field in fields), flush=True)


def _flat(text: object, cap: int = 400) -> str:
    """One line of at most `cap` characters, whatever the text carried."""
    return " ".join(str(text).split())[:cap]


@nki.jit
def body_that_reads_an_undefined_name(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale):
    """A kernel body that reads a name nothing in this module defines.

    It is compiled in the same child as the entries. A child whose ``wrap_nki`` took the
    framework hop instead of the compiler front end would report every entry as accepted, so
    every child qualifies itself by refusing this body.
    """
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    held = nl.ndarray((1, latent), dtype=nl.float32, buffer=nl.sbuf)
    nl.store(out_hbm[0, 0:1], value=held * no_such_name_anywhere)  # noqa: F821
    return out_hbm


def _open_device_nodes() -> int:
    """How many Neuron device nodes this process holds open."""
    held = [os.path.realpath(f"/proc/self/fd/{h}") for h in os.listdir("/proc/self/fd")]
    return len([one for one in held if "/dev/neuron" in one])


def _compile_each_entry() -> None:
    """Compile every entry point in this process and print one row for each."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_neuron.functional.attention import mla_sparse as live

    i32 = torch.int32
    served, plain = getattr(torch, SERVED_DTYPE), torch.float32
    window = PAGES * PAGE
    bank_rows = BANK_WINDOWS * window

    def fake(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    def paged(entry, latent, topk, tokens, dtype):
        """One no-RoPE entry with the bank, the table, this step's rows and the offset.

        The table is a column, ``[pages, 1]``, the shape every other index operand in this
        tree takes. The bank holds more rows than the window the table reads through it, as
        the served one does. Every entry is called through ``wrap_nki``, which is how the
        seam calls it: a raw call takes the framework hop instead of the front end.

        ``tokens = 0`` is the step that writes nothing, and it hands the overlay operands as
        None rather than as zero-row tensors: the front end builds a tile per tensor operand
        and refuses a zero-extent shape by name, so an absent overlay travels as an absent
        operand.
        """
        rows = fake((tokens, latent), dtype) if tokens else None
        at = fake((1, 1), i32) if tokens else None
        return lambda: wrap_nki(entry)(
            fake((1, HEADS, latent), dtype), fake((bank_rows, latent), dtype), fake((1, topk), i32),
            0.1, fake((PAGES, 1), i32), rows, at, PAGE)

    def paged_staged(entry, latent, topk, dtype):
        """The row-tiled entry on a paged window with the staged load path, not the streamed one.

        Ten positional operands: the four the entry always took, the four paged ones, then
        the two tile parameters this base declares after them. Only the load path differs
        from the arm above, so the pair reads that both paths accept a staged window.
        """
        return lambda: wrap_nki(entry)(
            fake((1, HEADS, latent), dtype), fake((bank_rows, latent), dtype), fake((1, topk), i32),
            0.1, fake((PAGES, 1), i32), fake((1, latent), dtype), fake((1, 1), i32), PAGE,
            live.MOVING_MAX, False)

    def unpaged_rope(entry, latent, topk, dtype):
        """One RoPE entry exactly as it stands today, on a window rather than a bank."""
        return lambda: wrap_nki(entry)(
            fake((1, HEADS, latent), dtype), fake((1, HEADS, ROPE), dtype),
            fake((window, latent), dtype), fake((window, ROPE), dtype), fake((1, topk), i32), 0.1)

    shapes = (
        ("row_tiled", live.mla_sparse_attention_nope_row_tiled_kernel,
         live.mla_sparse_attention_rope_row_tiled_kernel, LATENT, TOPK_ROWS),
        ("untiled", live.mla_sparse_attention_nope_kernel,
         live.mla_sparse_attention_rope_kernel, LATENT, TOPK_NARROW),
        ("latent_tiled", live.mla_sparse_attention_nope_tiled_kernel,
         live.mla_sparse_attention_rope_tiled_kernel, LATENT_RAGGED, TOPK_NARROW),
    )
    built = []
    for shape, nope, rope, latent, topk in shapes:
        # Three paged operand forms per entry: the served dtype with a decode step's one row,
        # the same in f32, and the served dtype with no overlay at all, the form a step that
        # writes no new latent hands the kernel.
        built.append((f"nope_{shape}_paged_{SERVED_DTYPE}_one_row", paged(nope, latent, topk, 1, served)))
        built.append((f"nope_{shape}_paged_float32_one_row", paged(nope, latent, topk, 1, plain)))
        built.append((f"nope_{shape}_paged_{SERVED_DTYPE}_no_overlay", paged(nope, latent, topk, 0, served)))
        if shape == "row_tiled":
            built.append((f"nope_{shape}_paged_{SERVED_DTYPE}_staged_path",
                          paged_staged(nope, latent, topk, served)))
        built.append((f"rope_{shape}_unpaged_{SERVED_DTYPE}", unpaged_rope(rope, latent, topk, served)))
        built.append((f"rope_{shape}_unpaged_float32", unpaged_rope(rope, latent, topk, plain)))
    built.append(("undefined_name_body",
                  lambda: wrap_nki(body_that_reads_an_undefined_name)(
                      fake((1, HEADS, LATENT), plain), fake((window, LATENT), plain),
                      fake((1, TOPK_ROWS), i32), 0.1)))
    _write_row("tree", f"module={live.__file__}", f"torch={torch.__version__}",
               "cpu_compile=" + os.environ.get("VLLM_NEURON_CPU_COMPILE", "unset"),
               "target=" + os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "unset"),
               "simulator=" + os.environ.get("NKI_SIMULATOR", "unset"))
    for name, call in tuple(built):
        message = ""
        try:
            with FakeTensorMode():
                call()
        except BaseException as refusal:  # a refusal is a result here, not a test error
            message = _flat(refusal, 4000)  # room for a refusal that names every site
        _write_row(f"entry={name}", f"refused={bool(message)}",
                   f"neuron_fds={_open_device_nodes()}", f"diagnostic={message or 'none'}")


def _rows_from_a_child() -> list[str]:
    """Run the compiles in a child of this tree and return every row it printed."""
    environment = {name: value for name, value in os.environ.items() if name not in _DROP}
    environment.update(_PIN, PYTHONPATH=str(_ROOT))
    done = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "compile-each-entry"],
        cwd=_ROOT, env=environment, capture_output=True, text=True, timeout=1800, check=False)
    printed = [line for line in done.stdout.splitlines() if line.startswith(ROW + "|")]
    child = "|".join((
        ROW, "child", f"rc={done.returncode}", f"rows={len(printed)}",
        "stderr_tail=" + _flat(done.stderr.splitlines()[-1] if done.stderr.strip() else "none")))
    return [*printed, child]


def _entry_rows(printed: list[str]) -> list[dict[str, str]]:
    """One dictionary per entry row, keyed by the field names the child printed."""
    rows = []
    for line in printed:
        if not line.startswith(ROW + "|entry="):
            continue
        # The diagnostic is the last field and it is the compiler's own text, which may hold
        # a pipe of its own, so it is cut off whole before the fields in front of it split.
        head, _, diagnostic = line.partition("|diagnostic=")
        fields = dict(one.split("=", 1) for one in head.split("|")[1:])
        rows.append({**fields, "diagnostic": diagnostic})
    return rows


def _read_the_child() -> list[dict[str, str]]:
    """Run one compile child and return its entry rows, after the checks that qualify them."""
    printed = _rows_from_a_child()
    child = [line for line in printed if line.startswith(ROW + "|child|")]
    assert child and "|rc=0|" in child[0], (
        f"the compile child did not come back clean, so its rows cannot be trusted: {child}")
    tree = [line for line in printed if line.startswith(ROW + "|tree|")]
    assert tree and f"|module={_ROOT}/" in tree[0], (
        f"the child compiled another tree than {_ROOT}: {tree}")
    rows = _entry_rows(printed)
    assert {row["neuron_fds"] for row in rows} == {"0"}, (
        f"a compile opened a device node, so it was not a device-free compile: {rows}")
    # The undefined-name body compiles in this child too, and a child that accepted it never
    # parsed a body, which would make every accepted row below meaningless.
    unparsed = [row for row in rows if row["entry"].startswith("undefined_name")]
    assert unparsed and unparsed[0]["refused"] == "True", (
        f"this child accepted a body that reads an undefined name, so it read no body at "
        f"all: {unparsed}")
    return rows


def test_the_front_end_accepts_the_paged_no_rope_entries() -> None:
    """The three no-RoPE entries declare the paged operands, and then compile with them.

    The declaration is read first because an entry that does not declare the paged operands
    is handed them and never sees them, so the compile below would read a clean unpaged
    kernel and call it a paged pass.
    """
    declared = {name: tuple(inspect.signature(getattr(MS, name)).parameters) for name in NOPE_ENTRIES}
    # The positions are the reading and not only the names: `wrap_nki` binds a call by the
    # declared order, so the four paged operands in another order would mis-bind the table
    # onto the offset. They are read where they sit rather than at the tail, because an entry
    # may declare compile-time tile options after them.
    wrong = {}
    for name, parameters in declared.items():
        at = parameters.index(SCALE_PARAMETER) + 1
        if tuple(parameters[at:at + len(PAGED_PARAMETERS)]) != PAGED_PARAMETERS:
            wrong[name] = parameters
    assert wrong == {}, (
        f"an entry does not declare the paged operands right after {SCALE_PARAMETER} and in "
        f"order {PAGED_PARAMETERS}, so a paged call either drops them silently or binds them "
        f"to the wrong parameter: {wrong}")
    paged = [row for row in _read_the_child() if row["entry"].startswith("nope_")]
    assert len(paged) == 10, (
        f"the child did not compile the three no-RoPE entries in all three paged operand "
        f"forms, plus the row-tiled entry on the staged load path: "
        f"{[row['entry'] for row in paged]}")
    refused = [f"{row['entry']}: {row['diagnostic']}" for row in paged if row["refused"] == "True"]
    assert refused == [], "the front end refused a paged entry: " + " ~ ".join(refused)


def test_the_front_end_accepts_the_rope_entries_unchanged() -> None:
    """The three RoPE entries compile in both dtypes, untouched by the paged operands."""
    rope = [row for row in _read_the_child() if row["entry"].startswith("rope_")]
    assert len(rope) == 6, (
        f"the child did not compile the three RoPE entries in both dtypes: "
        f"{[row['entry'] for row in rope]}")
    refused = [f"{row['entry']}: {row['diagnostic']}" for row in rope if row["refused"] == "True"]
    assert refused == [], "the front end refused a RoPE entry this change does not touch: " + \
        " ~ ".join(refused)


def test_the_seam_refuses_a_paged_call_that_carries_a_rope_half() -> None:
    """A paged window and a RoPE half together are refused by name, not served silently.

    ``k_pe`` holds one row per KV row, so a paged latent window would need a paged RoPE
    window beside it, and none is staged at this checkpoint's RoPE width of 0.
    """
    import torch

    seq, window = 1, PAGES * PAGE
    message = ""
    try:
        MS.mla_sparse_attention(
            torch.zeros(seq, HEADS, LATENT), torch.zeros(window, LATENT),
            torch.zeros(seq, TOPK_NARROW, dtype=torch.int32), 0.1,
            q_pe=torch.zeros(seq, HEADS, ROPE), k_pe=torch.zeros(window, ROPE),
            block_table_row=torch.zeros(PAGES, 1, dtype=torch.int32),
            written=torch.zeros(1, LATENT), write_offset=torch.zeros(1, 1, dtype=torch.int32),
            page_size=PAGE)
    except MS.MlaSparseAttentionError as refusal:
        message = _flat(refusal)
    assert "rope" in message.lower() or "RoPE" in message, (
        f"a paged call carrying a RoPE half was not refused by a message that names the RoPE "
        f"half: {message or 'it was not refused at all'}")


def test_the_module_holds_no_call_form_the_front_end_refuses() -> None:
    """No call site in the module expands a mapping or a sequence into a kernel call.

    The front end refuses ``**mapping`` and ``*sequence`` in a call it parses, and a
    simulator run cannot see it, so the module's own source is walked instead.
    """
    source = pathlib.Path(MS.__file__).read_text()
    # Call nodes rather than the text: a pattern anchored on the first argument misses
    # `f(a, *rest)` and `f(dst=x, **kw)`, and a definition's `*shape` is not a call node.
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        starred = sum(isinstance(one, ast.Starred) for one in node.args)
        mapped = sum(one.arg is None for one in node.keywords)
        if starred or mapped:
            found.append(f"line {node.lineno}: {starred} positional, {mapped} mapping")
    assert found == [], f"the module expands into a call at {len(found)} sites: {found}"


def test_the_front_end_refuses_a_body_that_reads_an_undefined_name() -> None:
    """A kernel body reading an undefined name is refused, and the refusal names the name."""
    unparsed = [row for row in _read_the_child() if row["entry"].startswith("undefined_name")]
    assert len(unparsed) == 1, f"the child did not compile the undefined-name body: {unparsed}"
    assert unparsed[0]["refused"] == "True", (
        "the front end accepted a body that reads an undefined name, so it never parsed a "
        "body and no accepted row in this file means anything")
    assert "unbound variable" in unparsed[0]["diagnostic"], (
        f"the body was refused for another reason than the unresolved name: "
        f"{unparsed[0]['diagnostic']}")


def test_the_row_tiled_entries_default_to_a_512_key_tile_and_streamed_rows() -> None:
    """The row-tiled entries keep the tile option defaults production compiles with.

    The seam calls the entry with operands alone and passes neither option, so a moved
    default would change the body every served call compiles without changing one call site.
    """
    served = {}
    for name in ROW_TILED_ENTRIES:
        declared = inspect.signature(getattr(MS, name)).parameters
        served[name] = {option: declared[option].default for option in SERVED_TILE_DEFAULTS}
    moved = {name: values for name, values in served.items() if values != SERVED_TILE_DEFAULTS}
    assert moved == {}, (
        f"a row-tiled entry no longer defaults to the tile options production serves "
        f"{SERVED_TILE_DEFAULTS}, so served calls compile another body: {moved}")


if __name__ == "__main__":
    _compile_each_entry()
