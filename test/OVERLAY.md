# `test/` is an OVERLAY, not an upstream tree

Upstream ships **no** test suite for this plugin, and this pin ships **no**
`test/` path at all. Everything under `test/` is fork-owned and is re-applied as
an overlay after every rebase onto the upstream release branch.

## Invariants (each one is what makes the re-apply mechanical)

1. **The overlay lives entirely under `test/`.** No file outside `test/`
   belongs to it, with the single exception of the
   `[tool.pytest.ini_options] markers` block in `pyproject.toml`.
2. **Nothing under `test/` is imported by `vllm_neuron/`.** The dependency
   runs one way only, so deleting `test/` can never break the shipped package
   and a rebase can drop the overlay without touching plugin code.
3. **The tree matches the configuration the pin already ships.**
   `pyproject.toml` declares `testpaths = ["test/unit", "test/vllm_neuron"]`;
   both directories exist here. `test/unit/test_layout.py` is the self-test
   that fails if a later change moves the tree out from under that config.
4. **`test/conftest.py` sets no environment variable.** The CPU-mode
   acceptance environment is pinned in the invocation, because
   `FP8_CLAMP_MAX` resolves at import time.
5. **Every vendored file is registered below** with its origin PR, its
   upstream path, **two** digests — its sha256 **at vendoring time** and its
   sha256 **as it sits on disk now** — its **adoption disposition**, and a
   **non-empty un-quarantine disposition**. `test/unit/test_layout.py` re-checks
   the on-disk column against the bytes on disk on every run, so a rebase can
   prove whether the upstream original moved and an edit to a vendored body
   cannot pass silently.

## Vendored-file register

All four rows below were vendored from **one** upstream pull request at **one**
commit, recorded here once so every row's provenance is unambiguous:

- **Origin PR:** [`vllm-project/vllm#53906`](https://github.com/vllm-project/vllm/pull/53906)
  — **open and unmerged** at vendoring time.
- **Head commit fetched:** `878631b6079d2cf9fb80830ef9cb41b43aded098`
  (head repo `ZJY0516/vllm`, branch `glm-release`).
- **Why the commit matters:** the PR is unmerged and still moving, so a branch
  name would not identify what was copied.
- **The quarantine marker is applied collection-wide**, by
  `test/vllm_neuron/upstream/conftest.py`, to every item collected from the
  vendored directory — never by an edit inside a vendored body. It is therefore
  a property of the directory rather than a per-row value, and it is proved
  mechanically by `test/unit/test_quarantine_marker.py` rather than asserted
  here.

### Column shape (fixed — the layout self-test reads it by position)

Each row carries the vendored path as its subject, then **six** columns:
origin PR · origin path · origin sha256-at-vendoring · on-disk sha256 ·
adoption disposition ∈ {`VERBATIM`, `ADAPTED`} · un-quarantine disposition ∈
{an increment id, `PERMANENT`}. **The two digest columns are separate on
purpose:** an `ADAPTED` file's on-disk bytes necessarily differ from its
origin's, so collapsing them into one column would make either drift detection
or tamper detection impossible. **No row may carry an empty un-quarantine
disposition.**

| Vendored path (in this tree) | Origin PR | Upstream path at vendoring | Origin sha256 at vendoring | On-disk sha256 | Adoption disposition | Un-quarantine disposition |
|---|---|---|---|---|---|---|
| `test/vllm_neuron/upstream/test_kpool_tail_slot_mapping.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/v1/attention/test_kpool_tail_slot_mapping.py` | `8a56bffb0d69a44353667ed6df79ce454bb1b16913e4663814b98da9d8fdcbdf` | `8a56bffb0d69a44353667ed6df79ce454bb1b16913e4663814b98da9d8fdcbdf` | `VERBATIM` | `PERMANENT` |
| `test/vllm_neuron/upstream/test_flashinfer_mla_sparse_sm90.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/v1/attention/test_flashinfer_mla_sparse_sm90.py` | `48c74334cb4035e38d1367cb2df9dd64f32d9f7286ff7a591fb22ef2c5e4b714` | `48c74334cb4035e38d1367cb2df9dd64f32d9f7286ff7a591fb22ef2c5e4b714` | `VERBATIM` | `PERMANENT` |
| `test/vllm_neuron/upstream/test_sparse_indexer_decode_seq_lens.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/v1/attention/test_sparse_indexer_decode_seq_lens.py` | `b501b93a304e62ae2208192e05303854868d946a8fff20be72caca2015242fbd` | `b501b93a304e62ae2208192e05303854868d946a8fff20be72caca2015242fbd` | `VERBATIM` | `PERMANENT` |
| `test/vllm_neuron/upstream/test_glm5next.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/transformers_utils/processors/test_glm5next.py` | `cfe0af2278ca47dd48ba15c06ae6974b87c2c82587ea050d6388d587d1fa2c9a` | `cfe0af2278ca47dd48ba15c06ae6974b87c2c82587ea050d6388d587d1fa2c9a` | `VERBATIM` | `PERMANENT` |

All four rows are `VERBATIM`, so each file's two digests are equal by
construction. They are recorded as two columns anyway, and the self-test checks
the **on-disk** one against the bytes on disk: the day a row turns `ADAPTED`,
the shape does not have to change and the drift record is not lost.

### Why all four dispositions are `PERMANENT` — the determination, per file

The convention is that a vendored file is un-quarantined by the increment that
lands its target. **For these four files no such increment exists, and that is a
determination rather than an omission:** every target they import is a
**vLLM-core** symbol that PR #53906 adds **to vLLM itself** — a new backend
class under `vllm/v1/attention/backends/`, a new processor under
`vllm/transformers_utils/processors/`, a new model package under `vllm/models/`.
**This fork does not author vLLM-core symbols.** The port's one patch module
wraps an existing vLLM function without introducing a symbol; every other
increment authors inside `vllm_neuron/`. So no increment can un-quarantine any of
these four at any milestone, and the honest register entry is a permanent one.

The earlier revision of this file left the column **empty** and said so
explicitly, because at that time no answer existed. This is the answer, and the
native coverage that does exercise each behaviour on Neuron is named beside it —
so "permanently quarantined" is never read as "this behaviour is untested here".

| Vendored file (L) | Target it needs | Landed by any increment in this port? | Disposition | Native coverage that DOES exercise this behaviour on Neuron |
|---|---|---|---|---|
| `test_kpool_tail_slot_mapping.py` (353) | `KpoolTailBackend` / `compute_kpool_tail_slot_mapping` in `vllm.v1.attention.backends.mla.indexer` | **No** — vLLM-core backend class | **PERMANENT** | `-047`, `-049` — the plugin's own kpool tail-slot path |
| `test_flashinfer_mla_sparse_sm90.py` (280) | `vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90` | **No** — and it is a **CUDA SM90** backend, out of this port's scope **by construction**, not by omission | **PERMANENT** | `-040`, `-041` — the Neuron sparse-MLA path |
| `test_sparse_indexer_decode_seq_lens.py` (184) | `vllm.models.glm5next` | **No** — vLLM-core model package; this port authors `vllm_neuron/model/glm5_next/`, a **different** package | **PERMANENT** | `-043`, `-047`, `-049`, `-051` |
| `test_glm5next.py` (389) | `vllm.transformers_utils.processors.glm5next` | **No** — vLLM-core processor | **PERMANENT** | `-056` |

**Then why vendor them at all, and why is that not vacuous?** Two reasons, both
of which survive permanent quarantine, and **neither of which is a coverage
claim** — these four files contribute **zero executing assertions** to this fork:

1. **Provenance and rebase-drift detection.** The rows above carry each file's
   origin PR, path and sha256 against a **moving** source — the PR is open and
   unmerged, and one file's length already changed between costing and
   vendoring. The rows are the instrument that detects the drift; the test
   bodies are the thing being fingerprinted.
2. **An executable upstream specification.** These files are the PR author's own
   statement of the intended behaviour, at a known revision, in the repository
   where the porters work — the reference the native increments named above are
   written **against**.

**Bodies are byte-identical to upstream and stay that way.** Editing a body —
including adding an import guard, a `pytest.importorskip`, or a module-level
skip to make collection green — is precisely the drift the digest columns exist
to expose, and it would turn the row into a fiction. A file that needs
re-targeting gets it from the increment that lands its target, together with a
refreshed digest row and an adoption disposition of `ADAPTED`.

## Marker vocabulary

`pyproject.toml` registers exactly three markers, and this file is where their
meaning lives:

- **`fast`** — cheap, device-free, safe to run on every change.
- **`forked`** — must run in its own process (state that does not reset
  in-process, e.g. import-time environment resolution).
- **`quarantined`** — **the marker gates SELECTION, not IMPORT.** A
  `pytest.mark` is evaluated *after* the module has been imported, so the marker
  keeps a module's tests from **running**; nothing except the target's existence
  keeps a module from **failing to import**. A module whose top-level `import`
  fails therefore produces a **collection error**, not a skip — which is exactly
  what all four rows above do at this pin, by design. Both halves of that
  sentence are load-bearing, and each is measured: the selection half on a
  module-level-clean population in `test/unit/test_quarantine_marker.py`, the
  import half on the vendored tree itself.
