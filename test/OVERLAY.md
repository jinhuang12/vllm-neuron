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

| Vendored path (in this tree) | Origin PR | Upstream path at vendoring | sha256 at vendoring | Quarantine marker | Un-quarantined by |
|---|---|---|---|---|---|

**No vendored files yet — this is an explicit none-declaration, not an
omission.** The tree at this point is fork-authored scaffolding only. The first
vendored files arrive with the upstream CPU-only test adoption; each lands with
one row above and a quarantine marker, because a vendored test whose target does
not exist at this pin would otherwise pass vacuously.

## Marker vocabulary

`pyproject.toml` registers exactly three markers, and this file is where their
meaning lives:

- **`fast`** — cheap, device-free, safe to run on every change.
- **`forked`** — must run in its own process (state that does not reset
  in-process, e.g. import-time environment resolution).
- **`quarantined`** — collected but skipped: the test is vendored ahead of the
  code it exercises, and the increment that lands that code removes the marker.
