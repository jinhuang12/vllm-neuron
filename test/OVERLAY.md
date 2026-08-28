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
   upstream path, and its sha256 **at vendoring time**, so a later rebase can
   prove whether the upstream original moved.

## Vendored-file register

All four rows below were vendored from **one** upstream pull request at **one**
commit, recorded here once so every row's provenance is unambiguous:

- **Origin PR:** [`vllm-project/vllm#53906`](https://github.com/vllm-project/vllm/pull/53906)
  — **open and unmerged** at vendoring time.
- **Head commit fetched:** `878631b6079d2cf9fb80830ef9cb41b43aded098`
  (head repo `ZJY0516/vllm`, branch `glm-release`).
- **Why the commit matters:** the PR is unmerged and still moving, so a branch
  name would not identify what was copied. Each row's sha256 is the sha256 of
  the file **as vendored into this tree**, taken from that commit, and
  `test/unit/test_layout.py` re-checks all four against the files on disk on
  every run.

| Vendored path (in this tree) | Origin PR | Upstream path at vendoring | sha256 at vendoring | Quarantine marker | Un-quarantined by |
|---|---|---|---|---|---|
| `test/vllm_neuron/upstream/test_kpool_tail_slot_mapping.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/v1/attention/test_kpool_tail_slot_mapping.py` | `8a56bffb0d69a44353667ed6df79ce454bb1b16913e4663814b98da9d8fdcbdf` | `quarantined` (applied collection-wide by `test/vllm_neuron/upstream/conftest.py`) | **not named in the increment plan at revision 4** — recorded, not invented |
| `test/vllm_neuron/upstream/test_flashinfer_mla_sparse_sm90.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/v1/attention/test_flashinfer_mla_sparse_sm90.py` | `48c74334cb4035e38d1367cb2df9dd64f32d9f7286ff7a591fb22ef2c5e4b714` | `quarantined` (applied collection-wide by `test/vllm_neuron/upstream/conftest.py`) | **not named in the increment plan at revision 4** — recorded, not invented |
| `test/vllm_neuron/upstream/test_sparse_indexer_decode_seq_lens.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/v1/attention/test_sparse_indexer_decode_seq_lens.py` | `b501b93a304e62ae2208192e05303854868d946a8fff20be72caca2015242fbd` | `quarantined` (applied collection-wide by `test/vllm_neuron/upstream/conftest.py`) | **not named in the increment plan at revision 4** — recorded, not invented |
| `test/vllm_neuron/upstream/test_glm5next.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/transformers_utils/processors/test_glm5next.py` | `cfe0af2278ca47dd48ba15c06ae6974b87c2c82587ea050d6388d587d1fa2c9a` | `quarantined` (applied collection-wide by `test/vllm_neuron/upstream/conftest.py`) | **not named in the increment plan at revision 4** — recorded, not invented |

**On the empty "Un-quarantined by" column — an absence declared, not an
oversight.** The convention is that each vendored file is un-quarantined by the
increment that lands its target. At revision 4 the plan names exactly one
un-quarantining increment, and it names it for a **different** file
(`test_kpool_decode_update_batched.py`, 549 L, which is deliberately **not**
vendored here). No increment in the plan claims any of the four files above, so
this column is left as a recorded gap rather than filled with a guess: a wrong
owner here would silently re-target the wrong file at un-quarantine time.

**Bodies are byte-identical to upstream and stay that way.** Re-targeting a
vendored file's imports is the work of the increment that lands its target and
removes its quarantine, together with a refreshed sha256 row. Editing a body at
any other time is precisely the drift these sha256 values exist to expose.

## Marker vocabulary

`pyproject.toml` registers exactly three markers, and this file is where their
meaning lives:

- **`fast`** — cheap, device-free, safe to run on every change.
- **`forked`** — must run in its own process (state that does not reset
  in-process, e.g. import-time environment resolution).
- **`quarantined`** — collected but skipped: the test is vendored ahead of the
  code it exercises, and the increment that lands that code removes the marker.
