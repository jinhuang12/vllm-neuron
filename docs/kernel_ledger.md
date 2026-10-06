# GLM-5.3-Flash decode kernel ledger

The kernel ledger compares the team's kernel benchmarks with the measured decode step.
It gives the time of each kernel, the sum of all kernels and collectives, and the
residual. The residual is the part of the step that no kernel benchmark explains:
compiler glue, waits between kernels, and launch skew.

The ledger uses no device time. It reads the benchmark JSON files and the gate JSON
files in `/home/ubuntu/glm53f-wt/reports/`.

## 1. Commands

Run each command from the worktree root.

```bash
PY=/home/ubuntu/glm53f-campaign/venv/bin/python
PYTHONPATH=$PWD $PY -m test.kernel_ledger --tip 5938748 --bs 1 --ctx 1024   # baseline, reconciliation
PYTHONPATH=$PWD $PY -m test.kernel_ledger --tip current --bs 1 --ctx 1024   # all wave-1 kernels, prediction
PYTHONPATH=$PWD $PY -m test.kernel_ledger --tip 594d425                     # any commit: the kernels in its tree
PYTHONPATH=$PWD $PY -m test.kernel_ledger --tip current --bs 64 --ctx 8192  # roofline at bs=64
PYTHONPATH=$PWD $PY -m test.kernel_ledger --emit-shapes                     # write reports/ledger_shapes.json
PYTHONPATH=$PWD $PY -m pytest test/kernel_ledger -q -p no:cacheprovider     # tests
```

Options:

| option | default | effect |
|---|---|---|
| `--tip` | `current` | `current`, `5938748`, or a commit of this repository |
| `--bs`, `--ctx` | 1, 1024 | the decode point; `max_model_len` = max(4096, ctx) |
| `--emit-shapes [PATH]` | `reports/ledger_shapes.json` | write the per-kernel decode shapes |
| `--json PATH` | none | also write the ledger as JSON |
| `--reports-dir DIR` | `/home/ubuntu/glm53f-wt/reports` | read the JSON files from DIR |

## 2. Data flow

```
config.json --> [1. arch] --> [2. decode graph] --> [3. shapes per node] --> ledger_shapes.json
                                     |                      |
*_micro.json --> [4. readers] --> [5. match by shape and kernel set] --> [6. rollup per bucket]
gate_*.json  --> [7. tip: kernel set + gate step] ------------------------------> [8. residual]
```

1. `models/glm53f/arch.py` reads the checkpoint `config.json`.
2. `models/glm53f/decode.py` builds one decode step at TP=64, EP=16 as a layer
   template. Each node has `layer_count`: the number of times one step runs it.
3. Each node gives its shapes in the words of its benchmark (`shape_record`).
4. `readers/micro.py` reads the six `*_micro.json` files. `readers/gate.py` reads the
   `gate_*.json` files.
5. A node gets a measured time only when its shape record is the same as the shape
   that the benchmark recorded. If a shape changes, the node shows "missing".
6. `models/glm53f/ledger.py` adds the times per bucket.
7. `models/glm53f/tips.py` finds the kernels in the tree of a tip and the gate run of
   that tree.
8. Residual = gate device step - (sum of measured kernels + collectives).

## 3. Node kinds

| kind | meaning | time in the sum |
|---|---|---|
| measured | a benchmark exists at this shape | the benchmark median x count |
| missing | the node has a benchmark kernel, but not at this shape | none (roofline only) |
| fused | a DSA member; the DSA layer benchmark times it inside `mla_sparse` | none |
| unmeasured | a compiler op with no benchmark (fn GEMV, KDA projections, gates, casts) | none; it is in the residual |
| collective | 17.5 us + bytes / 100 GB/s per collective | the model value x count |
| inactive | not in the graph of this kernel set (the logit gather before wt/dense) | none |

The roofline of a node is max(FLOPs / 79 TFLOP/s, HBM bytes / 716 GB/s, 2 us). These
are the values for one trn2 logical core at LNC=2.

## 4. Kernel sets and tips

A kernel family is the work of one wave-1 branch: `mhc`, `kda`, `dsa`, `moe-t`,
`dense`, `host`. In the benchmark files, "before" is the 5938748 kernel and "after" is
the kernel of the branch.

- `5938748`: all families use "before".
- `current`: all families use "after". Exception: the NKI RMSNorm. `dense.md` measures
  it but does not wire it, so the norms use "before".
- A commit: a family uses "after" when a gate run of its branch gated a candidate that
  is an ancestor of the commit. The verdict does not change this. Ancestry shows that
  the code of the candidate is in the tree.

The gate run of a tip is the newest gate run that measured that tree. If there is no
such run, it is the newest MERGE run of a candidate in the tree.

## 5. Special terms

- Routed experts: the expected number of distinct experts per step is
  `288 x (1 - (1 - 8/288)^T)` for T tokens. One EP rank holds 18 of 288 experts. The
  ledger interpolates the benchmark cost curve at `18 x (1 - (1 - 8/288)^T)`: 0.50 at
  T=1, 15.03 at T=64. This is the cost at the expected hit count. At T=1 the expected
  cost over the hit distribution is about 5% lower for "before" (`moe-t.md`).
- DSA: the indexer has a bypass regime. At ctx <= 2051 the selection keeps all tokens,
  so the indexer does no work. The decode window is `max_model_len` rows at 5938748.
  It is the decode context bucket with wt/dsa (2048 rows at ctx 1024).
- lm_head: 5938748 uses a replicated GEMV. wt/dense shards it by vocabulary (2420
  rows per rank) and adds an all-gather of the logits.
- Sampler: 5938748 samples on the host. This time is outside the device step, so the
  ledger shows it as host time. wt/host samples on the device.

## 6. Reconciliation (5938748 only)

At `--tip 5938748 --bs 1 --ctx 1024`, the ledger compares each bucket with two
references from the wave-1 breakdown (`models/glm53f/references.py`):

- as-built: the DECODE_BREAKDOWN.md master-table rows for the same work. These are
  engine-active times. The waits are a different bucket in that report.
- in-model wall: the wall time of the same work in the model (call spans, kernel wall),
  if the breakdown gives one.

A benchmark median is a wall time of a kernel in its own graph. Thus the in-model wall
is the better reference. A test makes sure that each cited text is in its source file.

## 7. Limits

1. A benchmark times a kernel in its own graph. In the model, the kernel waits for
   other cores, DMA queues and ranks. Thus the measured sum is not the device step.
2. The roofline uses the TP=64 design graph (vocabulary-parallel lm_head) for all
   kernel sets.
3. At bs=64 ctx 8192, only the router, the experts and the sampler have benchmarks.
   The other units show roofline only. No gate run serves this point, so the ledger
   gives no residual there.
4. The collective model is the fastest traced all-reduce. The late-rank waits are in
   the residual.

## 8. Files

| path | origin |
|---|---|
| `engine/node.py`, `engine/graph.py` | ported from `model_config_gen/engine` (Apache header kept) |
| `engine/collectives.py`, `readers/emf.py` | ported interface from `kernel_to_model_benchmarking` |
| `engine/decode_nodes.py` | new: GLM decode node types |
| `models/glm53f/*.py` | new: architecture, graph, shapes, ledger, references, tips |
| `readers/micro.py`, `readers/gate.py` | new: benchmark and gate readers |
| `cli.py`, `__main__.py` | new: the command line |
| `tests/` | new: engine, nodes, graph, readers, shapes, ledger, references, tips, CLI |

To add a benchmark family:

1. Add its kernel to `KernelName` (`engine/node.py`) and to `FAMILY_OF` (`configs.py`).
2. Add its match keys to `MATCH_KEYS` and a reader to `_READERS` (`readers/micro.py`).
3. Give the node a `shape_record()` with the same keys.
4. Add a test in `tests/test_shapes.py` that compares the emitted shape with the
   benchmark JSON.
