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
*_micro.json --> [4. readers] --> [5. match by shape and kernel set] --> [6. raw rollup per bucket]
gate_*.json  --> [7. tip + gate run] --> [8. gate profile per bucket] --> [9. k, calibrated] --> [10. residual]
```

1. `models/glm53f/arch.py` reads the checkpoint `config.json`.
2. `models/glm53f/decode.py` builds one decode step at TP=64, EP=16 as a layer
   template. Each node has `layer_count`: the number of times one step runs it.
3. Each node gives its shapes in the words of its benchmark (`shape_record`).
4. `readers/micro.py` reads the six `*_micro.json` files and the two mHC batch files
   (`mhc_micro_mid.json`, `mhc_micro_large.json`). `readers/gate.py` reads the
   `gate_*.json` files, with the profile buckets (`device_step_ms.buckets_ms`).
5. A node gets a measured time only when its shape record is the same as the shape
   that the benchmark recorded. If a shape changes, the node shows "missing".
6. `models/glm53f/ledger.py` adds the raw benchmark times per bucket.
7. `models/glm53f/tips.py` finds the kernels in the tree of a tip and the gate run of
   that tree.
8. `models/glm53f/profile.py` maps the gate profile onto the ledger buckets.
9. `models/glm53f/calibration.py` calculates k per bucket and the calibrated column.
10. Residual = gate device step - calibrated sum. The ledger also shows the residual
    against the raw sum.

## 3. Node kinds

| kind | meaning | time in the raw sum |
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
- A commit with no gate record: a family uses "after" when a gate run of its branch
  has the verdict MERGE and its gated candidate (`gate_sha`) is an ancestor of the
  commit.
- A commit with a gate record (a gate run measured that tree), for all verdicts (MERGE,
  REJECT, BLOCKED): the ledger uses the kernels of that tree. A family uses "after" when
  its gated candidate is the commit or an ancestor of the commit, for all verdicts. The
  latest gate run is frequently such a tree (`gate_mhc.json`, BLOCKED, tree f083375).
- Team-lead ruling (round 2): the MERGE-only rule applies to `current` and to commits
  with no gate record. A commit with a gate record uses the kernels of the tree that the
  gate measured. A BLOCKED or REJECT candidate never counts for a commit with no gate
  record.

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

At `--tip 5938748 --bs 1 --ctx 1024`, the ledger compares the raw sum of each bucket
with one same-scope in-model reference (`models/glm53f/references.py`). The reference
is a wall time (call span, kernel wall) when the breakdown gives one. The result is
PASS when the difference is 15% or less.

| bucket | reference | expected |
|---|---|---|
| mHC | mHC total 16.61 ms | PASS |
| KDA | as-built KDA bound 12.3 ms (`attention.md`); two other readings printed | PASS |
| DSA/MLA | DSA layer span 1254 us x 11 | PASS |
| lm_head | lm_head GEMV 1.826 + 0.335 ms (seam 5) | PASS |
| MoE | router call 42.3 us x 42 + MoE glue 3.982 + expert kernel wall 3.17 ms | FAIL |
| dense | blockwise_fp8_mm 1.705 + norms 0.082 + dense glue 0.129 ms (engine-active) | FAIL |
| collectives | AR transfer 0.83 + late-rank wait 2.07 ms | FAIL |

KDA (team-lead ruling, round 2): the verdict uses the as-built KDA bound of the
breakdown, 12.3 ms (`attention.md`). The ledger also prints the two other readings with
their delta and verdict. No reference is removed from the output:

| KDA reference | ms | verdict |
|---|---|---|
| as-built KDA bound (the verdict) | 12.30 | PASS |
| layer span without glue: conv span 348 us x 34 + KDA rest | 12.47 | PASS |
| layer span with the KDA-layer glue | 13.45 | FAIL |

A "scope-dependent verdict" line shows this range. The KDA-layer glue (0.975 ms, ACT and
DVE) is in-model time that the benchmark does not model. The ledger lists it as an
"un-modeled" term, and the residual lines state that the residual holds it.

The expected FAIL results have a scope cause. The ledger prints the cause. A benchmark
median is the wall time of a call in its own graph: it includes the DMA waits of the
call and its standalone glue. The DECODE_BREAKDOWN master table gives engine-active
time. A test makes sure that each cited text is in its source file.

## 7. Calibrated column (PREDICTOR, not a test)

```
k_bucket          = in-model ms at 5938748 / raw ms at 5938748
calibrated_bucket = k_bucket x raw ms of the kernel set
```

The in-model value is the bucket in the profile of the 5938748 gate run
(`gate_baseline.json`). If the profile does not name the bucket, the ledger uses the
breakdown reference (lm_head). For a bucket with no in-model value (sampler,
embed/tail), k is 1.

The sum line of the ledger uses the calibrated column. The ledger also prints the raw
sum, so the scope gap is visible.

k moves the scope gap of the 5938748 kernel to its wave-1 replacement. This is an
assumption. For each gated tree, the ledger prints the measured profile of the gate
run next to the calibrated value. Use this table to check the assumption for each
merge.

## 8. Gate profile mapping

`gate/attribute_decode.py` divides the rank-0 device step into engine-active ms per
kernel source file, plus wait, DMA-issue, idle and unnamed compiler-op buckets.
`models/glm53f/profile.py` maps a source file to a ledger bucket by its path prefix
(`mhc/`, `kda/`, `dsa/`, `attention/mla_`, `moe/`, ...). The router RMSNorm files
(`nkilib/core/subkernels/{rmsnorm_tkg,norm_tkg_utils}.py`) go to MoE. The lm_head GEMV
is an unnamed compiler op, so the profile has no lm_head value. If a new kernel source
has no rule, the ledger shows it as UNMAPPED and keeps it in the residual. Add a rule
for it in `KERNEL_RULES`.

## 9. Limits

1. A benchmark times a kernel in its own graph. In the model, the kernel waits for
   other cores, DMA queues and ranks. Thus the raw sum is not the device step.
2. The calibrated column is a predictor. A wave-1 kernel can have a different scope
   gap than the 5938748 kernel.
3. The roofline uses the TP=64 design graph (vocabulary-parallel lm_head) for all
   kernel sets.
4. At bs=64 ctx 8192, only the mHC kernels, the router, the experts and the sampler
   have benchmarks (7 of 15 units). The other units show roofline only. No gate run serves this point, so the ledger
   gives no residual and no calibration there.
5. The collective model is the fastest traced all-reduce. The late-rank waits are in
   the residual of the raw sum.
6. A row that matches more than one benchmark case uses the mean of the cases. Its
   source starts with "mean of N cases". Exception: a case that only repeats another
   case is not used. The row note shows it as "repeat not used". The DSA
   `bypass_vs_default_window` case times "before" at the 4096-row window; its "after"
   repeats the `bypass` case (the dsa.md headline, 286.6 us). See `DSA_REPEATS` in
   `readers/micro.py`.

## 10. Files

| path | origin |
|---|---|
| `engine/node.py`, `engine/graph.py` | ported from `model_config_gen/engine` (Apache header kept) |
| `engine/collectives.py`, `readers/emf.py` | ported interface from `kernel_to_model_benchmarking` |
| `engine/decode_nodes.py` | new: GLM decode node types |
| `models/glm53f/*.py` | new: architecture, graph, shapes, ledger, references, tips, profile, calibration |
| `readers/micro.py`, `readers/gate.py` | new: benchmark and gate readers |
| `cli.py`, `__main__.py` | new: the command line |
| `tests/` | new: one test file per module |

To add a benchmark family:

1. Add its kernel to `KernelName` (`engine/node.py`) and to `FAMILY_OF` (`configs.py`).
2. Add its match keys to `MATCH_KEYS` and a reader to `_READERS` (`readers/micro.py`).
3. Give the node a `shape_record()` with the same keys.
4. Add a test in `tests/test_shapes.py` that compares the emitted shape with the
   benchmark JSON.
5. If the gate profile shows the new kernel source as UNMAPPED, add a rule to
   `KERNEL_RULES` (`models/glm53f/profile.py`).
