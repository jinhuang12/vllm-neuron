# Kernel Source Digest in the Compile Cache Key

<!-- meta: description: How a per-graph digest of the NKI kernel sources is folded into the compile cache keys -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-10-08 -->

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

`vllm_neuron/compile_cache_key.py` folds into each library key the digest of
**only the kernel source files that key's kernels can reach**.

``` text
[1. import vllm_neuron] --> [2. snapshot kernel sources] --> [3. wrap both key fns]

 graph key:  create_cache_hash(gm, ...)
   [kernels gm calls] --> [module.qualname] --> [import + patch closure] --> [file set] --> sha256 --> fold
 kernel key: create_nki_cache_key(func, ...)
   [func] ----------------> [module.qualname] --> [import + patch closure] --> [file set] --> sha256 --> fold
 nkilib kernel: [module.qualname] --> [package files importing it] --> [their closures]
 unmapped reference --------------------------------------------------> package digest -----> fold
```

1. Step 2 reads every `*.py` under `vllm_neuron/functional/` and the modules
   kernel files import from elsewhere in the package (`KERNEL_SOURCE_FILES`:
   `parallel/neuron_parallel_state.py`, `utils/bucket_utils.py`,
   `utils/dtype_utils.py`, `utils/neuron_utils.py`). The process keeps these
   bytes, so every key it computes reads the same snapshot. 104 files, about
   7 ms, once per process.
2. A graph's kernels come from its FX nodes:
   - each NKI kernel node (`torch.ops.higher_order.nki_kernel_wrapper`) gives
     its kernel function through the library's kernel registry
     (`kernel_idx`); the function gives `<module>.<qualname>`, the spelling of
     the HLO `func_name`. A package kernel maps to its defining module. A kernel
     of another package (`nkilib`) maps to the package files that import it
     (see [Kernels of other packages](#kernels-of-other-packages));
   - each call target, called submodule class, fetched attribute or argument
     constant that a `vllm_neuron` module defines gives that module (the
     served graphs build rotational top-k configs in the graph);
   - each custom op outside the ATen, prims, higher-order and functional
     collective namespaces gives its kernel modules through
     `CUSTOM_OP_KERNEL_MODULES`.
3. The import closure starts at each module file and follows its `import` and
   `from ... import` statements, read with `ast` (no import runs). It follows
   statements at any depth (also inside functions), except under
   `if TYPE_CHECKING:`, and stops at files outside the snapshot. Files that
   patch a reached module join the closure (see [Patches](#patches)).
4. The digest is sha256 over the sorted file set: path, length, then bytes,
   the same stream as the whole-package digest. The key is
   `sha256(library_key + "|kernel_digest:" + digest)[:32]`, so the cache
   directory layout (`<root>/neuron/compile_cache/<key>/`) does not change.

The result: an edit to kernel X recompiles only the graphs that call X (or a
kernel that imports X). An edit to a file no kernel reaches, for example torch
code around a kernel call, recompiles no graph for kernel reasons; its effect
on the graph is in the FX text, which the library already keys on.

### Closure rules

| Statement in a reached file | The closure follows |
| --- | --- |
| `import vllm_neuron.a.b` | all of `a/b.py` |
| `from vllm_neuron.a import b`, `b` a submodule | all of `a/b.py` (and the statement in `a/__init__.py` that binds `b`, if any) |
| `from vllm_neuron.a import f`, `f` only re-exported by `a/__init__.py` | `a/__init__.py` and the statement that binds `f` (not the other re-exports) |
| `from vllm_neuron.a import f`, `f` defined in `a/__init__.py` | all of `a/__init__.py` |
| `from vllm_neuron.a import *` | all of `a/__init__.py` |
| relative imports | the same, from the file's own package |
| `import importlib` or `__import__` in a reached file | nothing: the key folds the package digest |
| an import under `if TYPE_CHECKING:` | nothing: it never runs |
| an import of another package | not that package's source (it is versioned with its package), but every snapshot file that patches that package |
| an import of `vllm_neuron.envs` or `vllm_neuron.accuracy.tensor_capture` | nothing: these files hold no kernel code (a kernel's trace-time branch on a knob is in the FX graph text; tensor capture is host-side debug code), so the snapshot excludes them |

A package `__init__.py` is reached only when a reached file imports from it.
Python also runs every parent package `__init__.py` when it imports a
module; those runs are not followed.

### Patches

A snapshot file *patches* a module when, at any depth, it rebinds or deletes
an attribute or an item of an object it imported from that module
(`mod.ATTR = v`, `mod.TABLE[k] = v`, `del mod.ATTR`, `setattr(mod, ...)`, also
through an alias `cfg = mod.cfg; cfg.X = v`), or calls a mutating method
(`append`, `extend`, `insert`, `pop`, `popitem`, `clear`, `remove`, `discard`,
`add`, `update`, `setdefault`, `sort`, `reverse`, `__setitem__` and its family)
on an imported object or a module attribute (`mod.TABLE.update(...)`). The
patch changes what the patched module's code does wherever it runs, so:

| Patched module | The patcher's closure joins |
| --- | --- |
| a snapshot module M | every closure that reaches M's file |
| a module of another package P | every closure with a run-time import from P, and so every key of a P kernel |

`functional/mlp.py:18-35` patches `nkilib.core.mlp`'s `is_mlp_tkg` at import.
Every kernel file imports `nkilib` (for example `kernel_assert`), so
`functional/mlp.py` joins every kernel key: an edit there recompiles every
graph. `parallel/neuron_parallel_state.py` patches `vllm.distributed` and joins
the closures that import `vllm` at run time.
`test/vllm_neuron/test_compile_cache_key_pergraph.py` pins this list, so a new
patch fails a test until it is reviewed here. Not detected: a mutation inside
a function the file calls (`mod.set_mode(1)`), a mutating-method name called
directly on a module bound by `import` (`nl.add` is a kernel op), and `exec` /
`eval`.

### Kernels of other packages

A kernel of another package, for example nkilib's `find_nonzero_indices`, is
named `nkilib.core.subkernels.find_nonzero_indices.find_nonzero_indices`. Its
source is versioned with nkilib, but package files decide what it compiles to:
they wrap it, choose its arguments and may patch nkilib. Its key folds the
closures of the snapshot files with a run-time import of:

- the kernel itself (`from nkilib.core.mlp.mlp import mlp`),
- a module or package that holds it (`from nkilib.core.mlp import mlp`,
  `import nkilib`),
- something from inside it, or a parent package's re-export of it
  (`from nkilib import attention_cte`).

The dotted name tells a kernel from a namesake: `cumsum`, `mlp`, `moe_tkg` and
`rotational_topk` exist in both trees. A kernel of another package that no
snapshot file imports (for example a test stand-in kernel) falls back.

### Fallback

A reference that does not map to files makes the key fold the whole-package
digest (the digest of every snapshot file). A fallback costs a recompile. It
never serves a stale kernel. These references fall back:

- a custom op that is not in `CUSTOM_OP_KERNEL_MODULES`,
- a kernel module of the package that is outside the snapshot (for example a
  kernel under `vllm_neuron/model/`) or that has no source file,
- a kernel of another package that no snapshot file imports,
- a dynamic import in the closure,
- an NKI node whose `kernel_idx` is not in the process's kernel registry,
- an error during the reference scan (logged at WARNING).

Each reason is logged once per process at INFO:

``` text
INFO - compile_cache_key.py - kernel digest fallback: custom op ns::op has no entry in CUSTOM_OP_KERNEL_MODULES (vllm_neuron.compile_cache_key); keys that reach it fold the whole-package digest
```

### The custom op table

`CUSTOM_OP_KERNEL_MODULES` maps `"namespace::op"` to the modules that define
the op's kernels. A `torch.library` op is opaque in the FX graph, so the table
names its kernels. The package registers no custom op today, so the table is
empty. `test/vllm_neuron/test_compile_cache_key_pergraph.py` scans the package
(including `model/glm5_next/model_fp8.py`) for op registrations and fails when
a registered op has no entry.

### Why the key, not a sub-directory of the cache root

The library offers no key-component hook. The two mechanisms it does offer:

| Mechanism | Covers local cache | Covers `NEURON_LIBTORCH_REMOTE_CACHE` | Covers `compiler_workdir` option | Dir names change |
| --- | --- | --- | --- | --- |
| Digest-named sub-root (`NEURON_LIBTORCH_CACHE_ROOT/<digest>`) | yes | no | no | no |
| Digest folded into the key (chosen) | yes | yes | yes | yes |

The library looks both functions up through their module at every call
(`cache.create_cache_hash(...)` in `compile` and `setup_workdir_common`;
`from .nki_cache import create_nki_cache_key` inside `compile_nki`), so the
attribute replacement is the whole hook.
`test/vllm_neuron/test_compile_cache_key.py` checks that this stays true for
the installed library version.

## Startup log

``` text
INFO - compile_cache_key.py - VLLM_NEURON_KERNEL_DIGEST=f3e81b81...94ee29 files=104 digest_ms=7.1 keys=per-graph
INFO - compile_cache_key.py - kernel digest: graph key <32-hex key> folds files=25 modules=vllm_neuron.functional.attention.mla_absorb,... resolve_ms=<ms> (graphs per-graph=1 fallback=0)
```

The first line comes once per process. The second line comes once for each
new graph, at the rate of the library's own `Compilation cache key:` line. The
first graph of a process also parses the files it reaches (about 215 ms for a
served decode graph); later graphs take about 50 ms.

Print the digests without starting a server:

``` bash
python -m vllm_neuron.compile_cache_key                  # package digest, file count, time
python -m vllm_neuron.compile_cache_key --files          # also the file list
python -m vllm_neuron.compile_cache_key --graph <entry>  # the files one cached graph folds
```

`--graph` takes a cache entry directory (`<root>/neuron/compile_cache/<key>`),
its `fxgraph.txt` or its `graph.hlo`. The persisted FX text holds only the
process-local `kernel_idx` of each kernel, so the kernel names come from the
`func_name` of each NKI custom call in `graph.hlo`. FX text without a
`graph.hlo` beside it falls back.

## Knob

| Variable | Default | Effect |
| --- | --- | --- |
| `VLLM_NEURON_DISABLE_KERNEL_DIGEST_KEY` | `0` | `1` keeps the library's own keys. A warm cache may then serve a stale kernel. The log line still prints the package digest. |

`install_kernel_digest_key(digest)` still folds one digest into every key (the
earlier whole-package scheme). Startup uses `install_per_graph_digest_key`.

## What it costs

- One recompile of each graph that reaches an edited file, including
  comment-only edits. On the stack-1b served set (20 graphs), 64 of the 104
  files reach no graph, and a decode-only kernel edit recompiles the decode
  graphs only. `functional/mlp.py` (an nkilib patch) reaches every graph.
- No change to a cache root that was filled with the same tree: the digests
  are deterministic across processes and hosts.
- Cache roots filled **before** this change miss once (the key changed); the
  old directories are not deleted.

## Relation to `SOURCE_DIGEST` in kernel files

Four kernels (`functional/dsa/decode_batch.py`, `functional/attention/mla_decode.py`)
already pass a digest of their own file into the kernel as a trace-time
integer. That still works and is now redundant; the per-graph digest covers
those files too.

## Manual device check

1. Start the server with an empty `NEURON_LIBTORCH_CACHE_ROOT=/tmp/root-A`.
   Record each `kernel digest: graph key` line and
   `ls /tmp/root-A/neuron/compile_cache/`.
2. Edit one comment line in a decode-only kernel file, for example
   `vllm_neuron/functional/dsa/decode_batch.py`.
3. Run `python -m vllm_neuron.compile_cache_key --graph <entry>` for each
   entry. Write down the entries whose file list holds the edited file.
4. Restart with the same root. Only those entries report a cache miss; every
   other graph is a local cache hit.
5. Restart once more without an edit: every graph is a local cache hit.
