# Kernel Source Digest in the Compile Cache Key

<!-- meta: description: Why and how a digest of the NKI kernel sources is folded into the compile cache keys -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-10-06 -->

## Problem

The [compilation cache](compilation_cache.md) keys a compiled graph on the FX
graph text, the input signatures, the tool versions and the compiler flags
(`libtorch_neuronx_lite/compile/cache.py:create_cache_hash`). The per-kernel
NKI result cache keys a compiled kernel on that kernel function's own source
(`libtorch_neuronx_lite/nki/nki_cache.py:create_nki_cache_key`).

Neither key sees an edit to:

- a helper function the kernel calls,
- a module constant or a tile table,
- a module the kernel file imports.

The FX graph still names the same kernel with the same arguments, so a warm
`NEURON_LIBTORCH_CACHE_ROOT` serves the **old** compiled kernel. The symptom is
silent: the server starts fast and runs stale code.

## Fix

`vllm_neuron/compile_cache_key.py` computes one digest of the kernel sources
and folds it into both library keys.

``` text
[1. import vllm_neuron] --> [2. sha256 over kernel sources] --> [3. wrap both key fns]
                                                                      |
   compile / capture --> cache.create_cache_hash(gm, inputs) ---------+--> fold --> 32-hex key
   compile_nki       --> nki_cache.create_nki_cache_key(fn, args) ----+--> fold --> 32-hex key
```

Step 2 reads, in sorted order, every `*.py` under `vllm_neuron/functional/`
plus the modules kernel files import from elsewhere in the package
(`KERNEL_SOURCE_FILES`: `parallel/neuron_parallel_state.py`,
`utils/bucket_utils.py`, `utils/dtype_utils.py`, `utils/neuron_utils.py`).
The relative path and the content of each file feed the hash, so a rename or
a one-byte edit changes the digest, and two checkouts of one commit at
different paths share it. 94 files, about 7 ms, once per process.

Step 3 replaces the two module attributes with wrappers that return
`sha256(library_key + "|kernel_digest:" + digest)[:32]`. The key stays 32 hex
characters, so the cache directory layout
(`<root>/neuron/compile_cache/<key>/`) is unchanged; a kernel edit lands in a
**new** directory. The wrappers return `None` where the library returns `None`
(an NKI call with an unhashable argument stays uncacheable).

The library looks both functions up through their module at every call
(`cache.create_cache_hash(...)` in `compile` and `setup_workdir_common`;
`from .nki_cache import create_nki_cache_key` inside `compile_nki`), so the
attribute replacement is the whole hook.
`test/vllm_neuron/test_compile_cache_key.py` checks that this stays true for
the installed library version.

### Why the key, not a sub-directory of the cache root

The library offers no key-component hook. The two mechanisms it does offer:

| Mechanism | Covers local cache | Covers `NEURON_LIBTORCH_REMOTE_CACHE` | Covers `compiler_workdir` option | Dir names change |
| --- | --- | --- | --- | --- |
| Digest-named sub-root (`NEURON_LIBTORCH_CACHE_ROOT/<digest>`) | yes | no | no | no |
| Digest folded into the key (chosen) | yes | yes | yes | yes |

A remote cache shared across nodes is exactly where a stale kernel does the
most damage, so the key fold was chosen.

## Startup log

``` text
INFO - compile_cache_key.py - VLLM_NEURON_KERNEL_DIGEST=2df626e3...a3b718a files=94 digest_ms=6.6
```

Print the digest without starting a server:

``` bash
python -m vllm_neuron.compile_cache_key          # digest, file count, time
python -m vllm_neuron.compile_cache_key --files  # also the file list
```

## Knob

| Variable | Default | Effect |
| --- | --- | --- |
| `VLLM_NEURON_DISABLE_KERNEL_DIGEST_KEY` | `0` | `1` keeps the library's own keys. A warm cache may then serve a stale kernel. The log line still prints the digest. |

## What it costs

- One recompile after any edit under `functional/` or to one of the four
  allowlisted modules, including comment-only edits and edits to a kernel
  the current model never traces. Over-inclusion costs one recompile;
  omission serves a stale kernel, so the allowlist errs toward inclusion.
- No change to a cache root that was filled with the same tree: the digest is
  deterministic across processes and hosts.
- Cache roots filled **before** this change miss once (the key format
  changed); the old directories are not deleted.

## Relation to `SOURCE_DIGEST` in kernel files

Four kernels (`functional/dsa/decode_batch.py`, `functional/attention/mla_decode.py`)
already pass a digest of their own file into the kernel as a trace-time
integer. That still works and is now redundant; the package digest covers
those files too.

## Manual device check

1. Start the server with an empty `NEURON_LIBTORCH_CACHE_ROOT=/tmp/root-A`;
   note `VLLM_NEURON_KERNEL_DIGEST=` in the log and
   `ls /tmp/root-A/neuron/compile_cache/`.
2. Edit one comment line in any file under `vllm_neuron/functional/`.
3. Restart with the same root: the digest line differs, every graph reports
   a cache miss, and new directory names appear next to the old ones.
4. Restart once more without an edit: every graph is a local cache hit.
