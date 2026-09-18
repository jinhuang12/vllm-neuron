# SPDX-License-Identifier: Apache-2.0
"""The NKI front end accepts every shipped sparse-attention entry point, paged and unpaged.

The front end parses a kernel's own source before it places one tile, and it refuses a call form
outside its subset. The simulator never parses that source -- it runs the body as plain Python -- so no
simulator item can see such a refusal. This file compiles the entries instead, with fake operands of
the served shapes, inside a child process that pins the platform target in its environment. The target
makes the toolchain read what it builds for from the environment, so the compile opens no device node,
and the child counts its own open device nodes to prove it.

FOUR tests, one per conjunct, and NO `parametrize`.

  1. the three no-RoPE entries compile with a block table, this step's rows and a write offset;
  2. the three RoPE entries still compile as they stand, which is also the control that a refusal in
     test 1 belongs to the paged operands and not to this venue;
  3. a paged call carrying a RoPE half is refused BY NAME at the seam;
  4. no call site in the module expands a mapping or a sequence into a kernel call.

WHY THE PAGED PATH SERVES NO RoPE HALF. `k_pe` carries one row per KV row, exactly as the latent cache
does, so a paged latent window would need a paged RoPE window beside it. This checkpoint's RoPE width
is 0 and the limb is elided at trace time, so no registered measurement reaches a paged RoPE call. The
seam refuses that combination by name rather than growing a second staging path nothing exercises;
test 3 reads the refusal.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

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


def _emit(*fields: object) -> None:
    """Print one reading, one line, so a reader can anchor it by key."""
    print(ROW + "|" + "|".join(str(field) for field in fields), flush=True)


def _flat(text: object, cap: int = 400) -> str:
    """One line of at most `cap` characters, whatever the text carried."""
    return " ".join(str(text).split())[:cap]


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

    f32, i32 = torch.float32, torch.int32
    window = PAGES * PAGE

    def fake(shape, dtype=f32):
        return torch.empty(shape, dtype=dtype, device="meta")

    def paged(entry, latent, topk, tokens):
        """One no-RoPE entry with the bank, the table, this step's rows and the offset.

        The table is a COLUMN, `[pages, 1]`: that is the shape the probe read the page number out of,
        and the shape every other index operand in this tree takes. Every entry is called through
        `wrap_nki`, which is how the seam calls it: a raw call takes the framework hop instead of
        the compiler front end this item is here to read.
        """
        return lambda: wrap_nki(entry)(
            fake((1, HEADS, latent)), fake((PAGES * PAGE, latent)), fake((1, topk), i32), 0.1,
            fake((PAGES, 1), i32), fake((tokens, latent)), fake((1, 1), i32), PAGE)

    def unpaged_rope(entry, latent, topk):
        """One RoPE entry exactly as it stands today, on a window rather than a bank."""
        return lambda: wrap_nki(entry)(
            fake((1, HEADS, latent)), fake((1, HEADS, ROPE)), fake((window, latent)),
            fake((window, ROPE)), fake((1, topk), i32), 0.1)

    cases = (
        ("nope_row_tiled_paged", paged(live.mla_sparse_attention_nope_row_tiled_kernel,
                                       LATENT, TOPK_ROWS, 1)),
        ("nope_untiled_paged", paged(live.mla_sparse_attention_nope_kernel,
                                     LATENT, TOPK_NARROW, 1)),
        ("nope_latent_tiled_paged", paged(live.mla_sparse_attention_nope_tiled_kernel,
                                          LATENT_RAGGED, TOPK_NARROW, 1)),
        ("rope_row_tiled_unpaged", unpaged_rope(live.mla_sparse_attention_rope_row_tiled_kernel,
                                                LATENT, TOPK_ROWS)),
        ("rope_untiled_unpaged", unpaged_rope(live.mla_sparse_attention_rope_kernel,
                                              LATENT, TOPK_NARROW)),
        ("rope_latent_tiled_unpaged", unpaged_rope(live.mla_sparse_attention_rope_tiled_kernel,
                                                   LATENT_RAGGED, TOPK_NARROW)),
    )
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
    """The three no-RoPE entries compile with a block table, this step's rows and a write offset."""
    paged = [row for row in _read_the_child() if row["entry"].startswith("nope_")]
    assert len(paged) == 3, f"the child did not compile the three no-RoPE entries: {paged}"
    refused = [f"{row['entry']} x{row['x']}: {row['diagnostic']}"
               for row in paged if row["refused"] == "True"]
    assert refused == [], "the front end refused a paged entry: " + " ~ ".join(refused)


def test_the_front_end_still_accepts_the_rope_entries_as_they_stand() -> None:
    """The three RoPE entries compile unchanged, which is the control for the item above."""
    rope = [row for row in _read_the_child() if row["entry"].startswith("rope_")]
    assert len(rope) == 3, f"the child did not compile the three RoPE entries: {rope}"
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
    """The census ruling 29 pairs with the compile: zero mapping or sequence expansions at call sites.

    The front end refuses `**mapping` and `*sequence` in a call it parses, and a simulator run cannot
    see it, so the census is read over the module's own source text rather than inferred.
    """
    source = pathlib.Path(MS.__file__).read_text()
    # The lookbehind is the whole reading: this module DEFINES four helpers as `def name(*shape)`,
    # and a census that counted a definition as a call site would read 5 and refuse a clean module.
    found = re.findall(r"(?<!def )\b[\w.]+\(\s*(?:\*\*|\*)[A-Za-z_]", source)
    _emit("expansion_census", f"sites={len(found)}", f"module={MS.__file__}")
    assert found == [], f"the module expands into a call at {len(found)} sites: {found}"


if __name__ == "__main__":
    _compile_each_entry()
