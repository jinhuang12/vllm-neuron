# SPDX-License-Identifier: Apache-2.0
"""The NKI front end accepts every shipped sparse-attention entry point, paged and unpaged.

The front end parses a kernel's own source before it places one tile, and it refuses a call form
outside its subset. The simulator never parses that source -- it runs the body as plain Python -- so no
simulator item can see such a refusal. This file compiles the entries instead, with fake operands of
the served shapes, inside a child process that pins the platform target in its environment. The target
makes the toolchain read what it builds for from the environment, so the compile opens no device node,
and the child counts its own open device nodes to prove it.

FIVE tests, one per conjunct, and NO `parametrize`.

  1. the three no-RoPE entries DECLARE the paged operands, last and in order, and then compile with a
     block table, this step's rows and a write offset -- in the served dtype and in f32, with a bank
     larger than the window, and with a one-row and a zero-row overlay;
  2. the three RoPE entries still compile as they stand, in both dtypes, which is also the control that
     a refusal in test 1 belongs to the paged operands and not to this venue;
  3. a paged call carrying a RoPE half is refused BY NAME at the seam;
  4. no call site in the module expands a mapping or a sequence into a kernel call;
  5. a body that reads an undefined name IS refused, which is what makes the accepted rows of tests 1
     and 2 evidence: a child that lost this venue accepts everything and reads nothing.

THE DECLARATION IS READ BEFORE THE COMPILE, and the reason is a reading rather than a precaution:
`wrap_nki` binds a call by the parameters the entry declares, in their declared order, and it DROPS
extra positional operands without a word. Against entries that do not declare the paged operands, a
paged call therefore compiles the unpaged kernel and reports no refusal at all.

WHY THE PAGED PATH SERVES NO RoPE HALF. `k_pe` carries one row per KV row, exactly as the latent cache
does, so a paged latent window would need a paged RoPE window beside it. This checkpoint's RoPE width
is 0 and the limb is elided at trace time, so no registered measurement reaches a paged RoPE call. The
seam refuses that combination by name rather than growing a second staging path nothing exercises;
test 3 reads the refusal.
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib
import re
import subprocess
import sys

import nki
import nki.language as nl

from vllm_neuron.functional.attention import mla_sparse as MS

ROW = "sparse_frontend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
# The child writes no bytecode: it is the first thing to import the compiler in an environment, and a
# cache file left in a shared installation is a change to it that this item does not intend.
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1"}

#: The served widths: this checkpoint's latent rank and head count, the selected-row count the decode
#: path passes, the KV block, and the model length in blocks. The untiled and latent-tiled bodies are
#: reached by the widths their own gate names, because the seam picks the body from the widths alone.
LATENT, HEADS, TOPK_ROWS, PAGE, PAGES = 512, 64, 2048, 128, 32
TOPK_NARROW, LATENT_RAGGED, ROPE = 512, 640, 64

#: The three no-RoPE entries the paged window is assembled in, and the operands they have to declare
#: for a paged call to reach the front end at all. THE DECLARATION IS READ BEFORE THE COMPILE, and the
#: reason is a reading and not a precaution: `wrap_nki` binds a call by the parameters the entry
#: declares and DROPS extra positional operands without a word. Against the entries as they stand, a
#: paged call therefore compiles the unpaged kernel and reports no refusal, so the compile alone
#: cannot tell a served paged entry from an unserved one.
NOPE_ENTRIES = ("mla_sparse_attention_nope_row_tiled_kernel", "mla_sparse_attention_nope_kernel",
                "mla_sparse_attention_nope_tiled_kernel")
PAGED_PARAMETERS = ("block_table_hbm", "written_hbm", "write_offset_hbm", "page_size")

#: The served latent cache is bf16 and holds more than one window, and the 16-row DMA transposes
#: specialise on the operand dtype -- 16 rows of 2 bytes fill the line 8 rows of 4 bytes do. So the
#: served dtype is compiled first, f32 beside it, and the bank larger than the window read through it.
SERVED_DTYPE = "bfloat16"
BANK_WINDOWS = 2


def _emit(*fields: object) -> None:
    """Print one reading, one line, so a reader can anchor it by key."""
    print(ROW + "|" + "|".join(str(field) for field in fields), flush=True)


def _flat(text: object, cap: int = 400) -> str:
    """One line of at most `cap` characters, whatever the text carried."""
    return " ".join(str(text).split())[:cap]


@nki.jit
def venue_control_unbound_name(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale):
    """The venue control: a body that reads a name nothing in this module defines.

    It is compiled in the same child as the entries and item 5 reads its refusal. Without it, a child
    whose `wrap_nki` took the framework hop instead of the compiler front end would report every entry
    as accepted and the compile items would pass on nothing -- which is how the paged item was hollow.
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
    import time

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

        The table is a COLUMN, `[pages, 1]`: the shape the probe read the page number out of, and the
        shape every other index operand in this tree takes. The bank holds more rows than the window
        the table reads through it, as the served one does. Every entry is called through `wrap_nki`,
        which is how the seam calls it: a raw call takes the framework hop instead of the compiler
        front end this item is here to read.

        `tokens = 0` IS THE STEP THAT WRITES NOTHING, and it hands the overlay operands as None rather
        than as zero-row tensors: the front end builds a tile per tensor operand and refuses a
        zero-extent shape by name, so the absent overlay travels as an absent operand.
        """
        rows = fake((tokens, latent), dtype) if tokens else None
        at = fake((1, 1), i32) if tokens else None
        return lambda: wrap_nki(entry)(
            fake((1, HEADS, latent), dtype), fake((bank_rows, latent), dtype), fake((1, topk), i32),
            0.1, fake((PAGES, 1), i32), rows, at, PAGE)

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
        # THREE PAGED OPERAND FORMS PER ENTRY: the served dtype with a decode step's one row, the same
        # in f32, and the served dtype with NO overlay at all, the form a step that writes no new
        # latent hands the kernel.
        built.append((f"nope_{shape}_paged_{SERVED_DTYPE}_one_row", paged(nope, latent, topk, 1, served)))
        built.append((f"nope_{shape}_paged_float32_one_row", paged(nope, latent, topk, 1, plain)))
        built.append((f"nope_{shape}_paged_{SERVED_DTYPE}_no_overlay", paged(nope, latent, topk, 0, served)))
        built.append((f"rope_{shape}_unpaged_{SERVED_DTYPE}", unpaged_rope(rope, latent, topk, served)))
        built.append((f"rope_{shape}_unpaged_float32", unpaged_rope(rope, latent, topk, plain)))
    built.append(("venue_control_unbound_name",
                  lambda: wrap_nki(venue_control_unbound_name)(
                      fake((1, HEADS, LATENT), plain), fake((window, LATENT), plain),
                      fake((1, TOPK_ROWS), i32), 0.1)))
    cases = tuple(built)
    _emit("venue", f"module={live.__file__}", f"torch={torch.__version__}",
          "cpu_compile=" + os.environ.get("VLLM_NEURON_CPU_COMPILE", "unset"),
          "target=" + os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "unset"),
          "simulator=" + os.environ.get("NKI_SIMULATOR", "unset"))
    _emit("geometry", f"latent={LATENT}", f"heads={HEADS}", f"topk={TOPK_ROWS}", f"page={PAGE}",
          f"pages={PAGES}", f"window={window}", f"ragged_latent={LATENT_RAGGED}", f"rope={ROPE}")
    for name, call in cases:
        started = time.time()
        message = ""
        try:
            with FakeTensorMode():
                call()
        except BaseException as refusal:  # the front end's refusal is the reading
            message = _flat(refusal, 4000)  # room for a refusal that names every site
        multiplicity = re.search(r"\[x(\d+)\]", message)
        _emit(f"entry={name}", f"refused={bool(message)}",
              f"x={multiplicity.group(1) if multiplicity else 0}",
              f"seconds={time.time() - started:.1f}", f"neuron_fds={_open_device_nodes()}",
              f"keyword_expansion={'keyword expansion is not supported' in message}",
              f"diagnostic={message or 'none'}")


def _rows_from_a_child() -> list[str]:
    """Run the compiles in a child of this tree and echo every row it printed."""
    environment = {name: value for name, value in os.environ.items() if name not in _DROP}
    environment.update(_PIN, PYTHONPATH=str(_ROOT))
    done = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "compile-each-entry"],
        cwd=_ROOT, env=environment, capture_output=True, text=True, timeout=1800, check=False)
    printed = [line for line in done.stdout.splitlines() if line.startswith(ROW + "|")]
    for line in printed:
        print(line, flush=True)
    child = "|".join((
        ROW, "child", f"rc={done.returncode}", f"rows={len(printed)}",
        "stderr_tail=" + _flat(done.stderr.splitlines()[-1] if done.stderr.strip() else "none")))
    print(child, flush=True)
    return [*printed, child]


def _entry_rows(printed: list[str]) -> list[dict[str, str]]:
    """One dictionary per entry row, keyed by the field names the child printed."""
    rows = []
    for line in printed:
        if not line.startswith(ROW + "|entry="):
            continue
        # THE DIAGNOSTIC IS THE LAST FIELD AND IT IS THE COMPILER'S OWN TEXT, which may hold a pipe of
        # its own. So it is cut off whole before the fields in front of it are split.
        head, _, diagnostic = line.partition("|diagnostic=")
        fields = dict(one.split("=", 1) for one in head.split("|")[1:])
        rows.append({**fields, "diagnostic": diagnostic})
    return rows


def _read_the_child() -> list[dict[str, str]]:
    """Run one compile child and return its entry rows, after the three readings that qualify them."""
    printed = _rows_from_a_child()
    child = [line for line in printed if line.startswith(ROW + "|child|")]
    assert child and "|rc=0|" in child[0], (
        f"the compile child did not come back clean, so its rows are not a reading: {child}")
    venue = [line for line in printed if line.startswith(ROW + "|venue|")]
    assert venue and f"|module={_ROOT}/" in venue[0], (
        f"the child compiled another tree than {_ROOT}: {venue}")
    rows = _entry_rows(printed)
    assert {row["neuron_fds"] for row in rows} == {"0"}, (
        f"a compile opened a device node, so it was not a device-free compile: {rows}")
    return rows


def test_the_front_end_accepts_the_paged_no_rope_entries() -> None:
    """The three no-RoPE entries DECLARE the paged operands, and then compile with them.

    TWO CONJUNCTS IN THIS ORDER. The declaration is read first because an entry that does not declare
    the paged operands is handed them and never sees them, so the compile below would read a clean
    unpaged kernel and call it a paged pass.
    """
    declared = {name: tuple(inspect.signature(getattr(MS, name)).parameters) for name in NOPE_ENTRIES}
    for name, parameters in declared.items():
        _emit(f"declared={name}", f"parameters={'.'.join(parameters)}")
    # THE POSITIONS ARE THE READING, NOT ONLY THE NAMES: `wrap_nki` binds a call by the declared
    # ORDER, so the four paged operands in another order would mis-bind the table onto the offset.
    wrong = {name: parameters for name, parameters in declared.items()
             if tuple(parameters[-len(PAGED_PARAMETERS):]) != PAGED_PARAMETERS}
    assert wrong == {}, (
        f"an entry does not declare the paged operands last and in order {PAGED_PARAMETERS}, so a "
        f"paged call either drops them silently or binds them to the wrong parameter: {wrong}")
    paged = [row for row in _read_the_child() if row["entry"].startswith("nope_")]
    assert len(paged) == 9, (
        f"the child did not compile the three no-RoPE entries in all three paged operand forms: "
        f"{[row['entry'] for row in paged]}")
    assert len([row for row in paged if SERVED_DTYPE in row["entry"]]) == 6, (
        f"the served dtype was not compiled: {[row['entry'] for row in paged]}")
    refused = [f"{row['entry']} x{row['x']}: {row['diagnostic']}"
               for row in paged if row["refused"] == "True"]
    assert refused == [], "the front end refused a paged entry: " + " ~ ".join(refused)


def test_the_front_end_still_accepts_the_rope_entries_as_they_stand() -> None:
    """The three RoPE entries compile unchanged, which is the control for the item above."""
    rope = [row for row in _read_the_child() if row["entry"].startswith("rope_")]
    assert len(rope) == 6, (
        f"the child did not compile the three RoPE entries in both dtypes: "
        f"{[row['entry'] for row in rope]}")
    refused = [f"{row['entry']} x{row['x']}: {row['diagnostic']}"
               for row in rope if row["refused"] == "True"]
    assert refused == [], "the front end refused a RoPE entry this change does not touch: " + \
        " ~ ".join(refused)


def test_the_seam_refuses_a_paged_call_that_carries_a_rope_half() -> None:
    """A paged window and a RoPE half together are refused by name, not served silently.

    `k_pe` holds one row per KV row, so a paged latent window needs a paged RoPE window beside it, and
    no registered measurement reaches one at this checkpoint's RoPE width of 0.
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
    _emit("paged_rope_refusal", message or "none")
    assert "rope" in message.lower() or "RoPE" in message, (
        f"a paged call carrying a RoPE half was not refused by a message that names the RoPE half: "
        f"{message or 'it was not refused at all'}")


def test_no_call_site_in_the_module_expands_into_a_kernel_call() -> None:
    """The census the compile is paired with: zero mapping or sequence expansions at call sites.

    The front end refuses `**mapping` and `*sequence` in a call it parses, and a simulator run cannot
    see it, so the census is read over the module's own source text rather than inferred.
    """
    source = pathlib.Path(MS.__file__).read_text()
    # THE CENSUS WALKS CALL NODES, not the text: a pattern anchored on the first argument misses
    # `f(a, *rest)` and `f(dst=x, **kw)`, and a definition's `*shape` is not a call node at all.
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        starred = sum(isinstance(one, ast.Starred) for one in node.args)
        mapped = sum(one.arg is None for one in node.keywords)
        if starred or mapped:
            found.append(f"line {node.lineno}: {starred} positional, {mapped} mapping")
    _emit("expansion_census", f"sites={len(found)}", f"module={MS.__file__}")
    assert found == [], f"the module expands into a call at {len(found)} sites: {found}"


def test_the_compile_venue_refuses_a_body_that_reads_an_undefined_name() -> None:
    """The venue control, and what makes every other row of this file evidence.

    A child that lost the venue -- the environment, or the library behind `wrap_nki` -- takes the
    framework hop instead of the compiler front end and reports every entry as accepted. Then the two
    compile items above pass on nothing. This item compiles one body that reads a name nothing defines
    and reads its refusal, in the same child, in the same venue, under the same operands.
    """
    control = [row for row in _read_the_child() if row["entry"].startswith("venue_control")]
    assert len(control) == 1, f"the child did not compile the venue control: {control}"
    assert control[0]["refused"] == "True", (
        "the venue accepted a body that reads an undefined name, so it never parsed a body and no "
        "accepted row in this file is evidence of anything")
    assert "unbound variable" in control[0]["diagnostic"], (
        f"the venue refused the control for another reason than the unresolved name, so it is not the "
        f"control this item claims: {control[0]['diagnostic']}")


if __name__ == "__main__":
    _compile_each_entry()
