# moe-t: MoE decode path of GLM-5.3-Flash (worker-5)

Branch `wt/moe-t`, worktree `/home/ubuntu/glm53f-wt/moe-t`, base `5938748`.

The decode MoE path (router + one rank's routed experts) now costs 0.27x to 0.34x of
5938748 per layer at T=1, on hardware. Each new path is one kernel launch per layer.

## 1. Commits and diff

```
a4a5dd0 Chain the benchmark layers through a data dependency and import this tree
39028c1 Compile the decode microbenchmark with the served model's neuronx-cc arguments
951b07f Read the visited expert id into a uint32 tile for the dynamic gate read
f5edb56 Add a decode microbenchmark for the router and one layer's experts
023595d Route decode batches of 64 tokens or fewer through the decode router and experts
1859fa5 Count the decode kernels in their seams' families and clamp the rank operand
da61801 Give compact_decode_kernel a token axis of 1 to 64 rows per block
b95ca97 Add a token-axis routed-expert decode kernel with scales folded per PSUM bank
5da7326 Add a one-launch decode router: RMSNorm, router GEMM and noaux_tc top-8
764d67b Snapshot the 5938748 MoE decode kernels as the benchmark and test baseline
02cf393 Enable the NKI simulator in the MoE router and expert-path tests
```

`git diff --stat 5938748..HEAD`: 24 files changed, 6052 insertions(+), 9 deletions(-).
All 24 files are in the write set:

| Area | Files |
|---|---|
| New kernels | `vllm_neuron/functional/moe/router_decode.py` (279), `expert_decode.py` (376) |
| Changed kernels | `fused_fp8.py` (+67/-1: new entry `fused_fp8_decode_experts`), `moe_fused_fp8_decode.py` (+12/-8: token axis) |
| Model call sites | `vllm_neuron/model/glm5_next/model_fp8.py` (+59, two insertions, see item 5) |
| Tests | `test/vllm_neuron/functional/moe/{conftest,decode_fixtures,test_router_decode,test_expert_decode,test_compact_decode_token_axis}.py`, `test/vllm_neuron/model/glm5_next/{test_router,test_moe_path}.py` |
| Benchmark | `test/hardware/benchmark_moe_decode.py`, `test/hardware/baselines/moe_5938748/**` (5938748 copies + `pipeline.py`) |

`git status --short` is empty. The test runs write `__pycache__` directories, and this
repo does not ignore them. I removed only these untracked bytecode directories.

## 2. Simulate results

Command (acceptance 1), on HEAD `a4a5dd0`:

```
cd /home/ubuntu/glm53f-wt/moe-t && PYTHONPATH=$PWD /home/ubuntu/glm53f-campaign/venv/bin/python -m pytest \
  test/vllm_neuron/functional/moe test/vllm_neuron/model/glm5_next/test_router.py \
  test/vllm_neuron/model/glm5_next/test_moe_path.py -q -p no:cacheprovider
```

Result: **217 passed**, exit 0 (459 s). 62 of the 217 tests are new.

Every new kernel test asserts two things: the seam counter shows one NKI dispatch and no
fallback, and the simulator ran the new kernel by name (`SimulatorCounter.kernels`).

| Test file | New tests | What it proves | Shapes | Tolerance |
|---|---|---|---|---|
| `test_router_decode.py` | 9 | Decode router vs the 5938748 `route_tokens` snapshot (pad to 256, nkilib RMSNorm + router + noaux_tc) | T in {1, 4} (random and checkpoint layer-3.. router weights), T=64, H=4096, E=288, top-8 | Index sets: exactly equal on every row whose 8th/9th corrected-score gap is >= 1e-4. No such tie row occurs at T in {1, 4}; at most 1 at T=64. Weights: rtol 1e-4, atol 1e-6. Logits: rtol/atol 2e-3. |
| `test_expert_decode.py` | 22 | Expert decode kernel vs the 5938748 packed branch of `block_quant_expert_mm` (local gather, mapping, compact kernel at T=1 under LNC2, general kernel at T=4, fp32 token-gather combine) | 18 local experts, H=4096, I=512 per rank, router width 288, rank 5; T in {1, 4}. Hits T=1: {3, 11}; T=4: 4 distinct experts over 6 (token, expert) pairs, one token with no local expert | fp32 output: `abs(new-old) <= 1e-6*max(abs(old)) + 1e-5*abs(old)` (measured max 1.1e-7 of max(abs(old))). bf16 output: rtol 2^-7. Idle tokens: exactly 0 on both sides. |
| | | 1 vs 2 programs, rank as tensor or int, rank clamp, zero hits (exact 0), T=64 rows independent, 10 small-shape cases vs the torch oracle, refusal without a device, admission envelope | H=256, I=256, 4 experts for the small cases | Oracle: rtol 1e-3, atol 1e-4 of max |
| `test_compact_decode_token_axis.py` | 25 | `compact_decode_kernel` with q in {1, 2, 4, 64}, bit-equal to the general kernel at BLOCK_M=q (idle, full, padding holes; 1 and 2 programs); q=65 is refused | 8 experts, H=I=128 | Bit equality |
| `test_router.py` | 3 | The `route_tokens` call site takes the decode router at T in {1, 4} (counter + kernel name; matches the tree's prefill router) and keeps the prefill router at T=65 | Checkpoint router inputs, H=4096, world 16, EP 16 | Same as the router file |
| `test_moe_path.py` | 3 | The `block_quant_expert_mm` call site takes one decode launch at T in {1, 4} and matches the 5938748 snapshot in bf16; a collector or T=65 keeps the mapping route | GLM text config, world 64, EP 16 (18 local of 288) | bf16 rtol 2^-7 |

Regression runs on this branch (all exit 0):

- `test/vllm_neuron/model/glm5_next` + `test/vllm_neuron/functional` minus `functional/moe`, with
  `NKI_SIMULATOR=1`: 1275 passed, 1 skipped. This includes `test_meta_forward.py` and
  `tiny/`. It ran before `951b07f` (a one-dtype change in the expert kernel). On HEAD,
  acceptance 1 covers that kernel (217 passed). On HEAD `a4a5dd0`, `tiny/`,
  `test_meta_forward.py`, `test_dispatch_counters_off_the_trace.py`, `test_experts.py`
  and `test_expert_parallel_shards.py` with `NKI_SIMULATOR=1` gave **135 passed, rc=0**
  (`/tmp/moe_t_scratch/tiny_head.log`). An earlier attempt hit a 50-minute timeout on a
  loaded host and has no result.
- Review: da-2 returned PASS in round 1. On cores 40-43 it reran acceptance 1 (217
  passed) and the T=1 benchmark; its ratios were 0.257 / 0.265 / 0.341.
- Known defect, deferred: the JSON field `baseline_module` records
  `moe-t-cache/benchmark_workdir/test/hardware/baselines/moe_5938748`, because the
  script resolves the path a second time after it changes directory. The run loaded
  the in-tree snapshot (the first resolve happens before the change of directory).
- Hardware probe (cores 12-15, LNC2): the expert kernel at T in {1, 4}, with bf16 and fp8
  stationaries, vs the torch oracle: max error 2.3e-5 of max output; the idle token is
  exactly 0. The router at T in {1, 4}: index sets equal; affinities within 9e-8.

## 3. Microbenchmark

Command (acceptance 2) wrote `/home/ubuntu/glm53f-wt/reports/moe-t_micro.json`, exit 0.

Method: each variant is compiled as a 1-layer graph and as an 8-layer graph. Every
layer has its own inputs and its own weight copies, and layer l reads the output of
layer l-1. Per-layer cost = `(t_8 - median(t_1)) / 7`. I report the median and p90 of
this value over 50 timed calls (10 warm-up calls). Cores: one trn2 chip, logical cores
12-15, LNC2. Only logical core 0 runs (physical cores 0 and 1). Compiler arguments are
the served model's (`-O1`, BIR verifier off; see item 5).

Microseconds per layer, one rank's shard:

| Kernel | T | Scenario | Before median / p90 | After median / p90 | Speedup | Cores before -> after |
|---|---|---|---|---|---|---|
| Router | 1 | - | 78.9 / 84.2 | 22.9 / 47.9 | 3.45x | 2 -> 1 |
| Router | 4 | - | 76.9 / 81.7 | 23.7 / 40.9 | 3.25x | 2 -> 1 |
| Router | 64 | - | - | 33.7 / 38.8 | - | 1 |
| Experts | 1 | 0 local hits | 131.9 / 134.7 | 35.4 / 41.7 | 3.72x | 2 -> 2 |
| Experts | 1 | 1 hit | 229.1 / 237.7 | 58.9 / 63.3 | 3.89x | 2 -> 2 |
| Experts | 1 | 2 hits | 228.4 / 272.8 | 82.5 / 90.0 | 2.77x | 2 -> 2 |
| Experts | 4 | 2 distinct (4 pairs) | 332.8 / 341.2 | 88.4 / 96.9 | 3.77x | 2 -> 2 |
| Experts | 4 | 4 distinct (6 pairs) | 412.7 / 415.4 | 140.5 / 173.7 | 2.94x | 2 -> 2 |
| Experts | 64 | 15 distinct (~37 pairs) | - | 874.7 / 900.1 | - | 2 |

Router + experts per layer (acceptance criterion: after <= 0.6 x before at T=1):

| T | Scenario | Before median | After median | After / before |
|---|---|---|---|---|
| 1 | 0 hits | 210.7 | 58.3 | **0.277** |
| 1 | 1 hit | 307.9 | 81.7 | **0.265** |
| 1 | 2 hits | 307.3 | 105.3 | **0.343** |
| 4 | 2 distinct | 409.7 | 112.0 | 0.273 |
| 4 | 4 distinct | 489.5 | 164.1 | 0.335 |

Expert cost per hit at T=1 (`expert_cost_per_hit_T1` in the JSON):

- Before: 131.9 us fixed per layer, plus 97.2 us for the first hit. The second hit is
  free, because the two blocks run on the two cores.
- After: 35.4 us fixed per layer, plus 23.5 us for each hit. That is near the PE
  weight-load floor: 192 128x128 stationaries per core x ~0.105 us = 20 us.

Notes on the measurement:

1. **Router chain.** The router's dependency goes through its correction bias
   (`b_l + 2^-8 * sum(aff_{l-1})`, a constant shift, so the selection stays the same).
   The JSON also has `*_hidden_chain`, where the dependency goes through the activation.
   There the 5938748 router at T=4 costs 468 us/layer. The profile shows ~2.4 ms of
   compiler-inserted `STREAM_TRANSPOSE` over 8 layers, around the 5938748 pad to 256
   rows. This does not occur at T=1, nor with the bias chain. I use the bias chain so
   that a compiler artifact does not inflate the T=4 gain. With the hidden chain, the
   T=1 ratios are the same (0.265 to 0.334).
2. **Unchained layers.** An earlier version summed independent layers. The compiler
   overlapped them, which gave an impossible 1.2 us per router layer. That version was
   withdrawn.
3. **Run-to-run variation.** The T=1 rows repeat within ~3 us across three runs
   (`/tmp/moe_t_scratch/bench_T*.json`). The router at T=64 measured 58.5 us in one
   earlier run and 33.7 us in this one. Treat the T=64 numbers as ±50%.
4. **fp8 stationaries (`weight_fp8=True`).** The products are the same, and fp8 is
   3-11% faster at T<=4 (for example 4 distinct at T=4: 129.7 vs 146.1 us). It is 2%
   slower at T=64. It is not the default (deferred).
5. **Where the after cost goes** (device profile of the 8-layer graphs, T=1):
   - Router: the router GEMM is ~15 us/layer on the PE. It is 32 matmuls of 288
     moving columns.
   - Experts at 0 hits, ~30 us/layer: dynamic-loop entry and exit barriers (~7 us), the
     epilogue's 16 PE transposes and their casts (~6 us), the prologue's dependent DMA
     chain (rank -> affinity -> nonzero), and the one cross-core `sendrecv`.

## 4. Bounded e2e

Bound = (before - after) router + experts per layer x 42 MoE layers:

| Case | Gain per layer (us) | ms/step |
|---|---|---|
| 0 hits on this rank | 152.5 | 6.40 |
| 1 hit | 226.2 | 9.50 |
| 2 hits | 201.9 | 8.48 |
| Rank average at T=1: P(0, 1, 2+) = 0.593, 0.325, 0.082 (hypergeometric, 8 of 288 with 18 local) | 180.5 | **7.6** |
| Slowest of 16 ranks (gates the FFN all-reduce): P(max = 1) = 0.134, P(max >= 2) = 0.866 (I count 3+ as 2) | 205.2 | **8.6** |
| T=4, 2 / 4 distinct | 297.6 / 325.4 | 12.5 / 13.7 |

This is an upper bound. The "before" graph includes the 5938748 XLA glue (mapping, pad,
token-gather combine). In the served model, the scheduler can overlap part of that glue
with other work. In the gate baseline profile, the 5938748 expert kernel alone takes
2.54 ms/step on rank 0. The e2e gate run is the real measurement.

## 5. Merge risks

**`model_fp8.py`, exactly two insertions:**

1. Lines **1299-1317** (`Glm5NextRoutedExperts.route_tokens`). The insertion comes after
   the `eps` resolution and before the 5938748 seam import. When `T <= 64` and the
   router admits the call, it returns `router_decode.noaux_tc_router_decode(...)`, which
   gives the same three outputs. Any other call runs the unchanged 5938748 code below
   it.
2. Lines **1561-1600** (`block_quant_expert_mm`, packed branch). The insertion comes
   after the global-width check and before `if routed != num_experts:`. That is
   *before* the local gather, because the new kernel reads the global `[T, 288]`
   affinities and the rank operand itself. For that reason it is not at ~1650-1684,
   where the 5938748 compact/mapping call sits. The route is taken only when all of
   these are true:
   - packed weights;
   - no collector;
   - experts are not TP-sharded (`E_local % tp != 0`; GLM: 18 % 4 = 2);
   - a valid rank;
   - `can_run_expert_decode` (bf16, 1 <= T <= 64, geometry).

   Otherwise the 5938748 mapping route runs, unchanged.

No line of the dense MLP / shared-expert path (~2274-2327, ~2879-2906 at 5938748) is
touched. Those regions move down by 59 lines, but git merges them by context.

**Other risks:**

- `fused_fp8.py`: one new entry point (`fused_fp8_decode_experts`). The dispatch of
  `fused_fp8_experts` is unchanged.
- The decode kernels count into the existing seam families: `noaux_tc` for the router,
  `fused_fp8` for the experts. A test that expects exact counts at decode shapes sees
  the new route. All tests in the tree pass.
- `compact_decode_kernel` now accepts q in 1..64. It is bit-identical to the general
  kernel at BLOCK_M=q. The mapping route still calls it only at q=1.
- **Compiler verifier.** With the BIR verifier on, the 5938748 `compact_decode_kernel`
  does not compile with this toolchain (2.27.5334): a `TensorCopyDynamicSrc` gets an
  int32 offset tile where the verifier needs uint32. The served model turns the
  verifier off (`neuron_model_runner.py:1488`), so production is not affected. Any
  harness that compiles with default flags fails on the old kernel. The new kernel uses
  a uint32 tile and compiles with the verifier on or off.

**Both LNC2 cores:**

- **Experts: yes.** Two programs. Core c computes intermediate blocks
  [c x 256, (c+1) x 256) of every visited expert. That is half the weight bytes and
  half the 128x128 weight loads per core. One `sendrecv` per layer exchanges the fp32
  partial sums; each core then stores half of H.
- **Router: no, one program.** The work per call is a 2.36 MB weight read and 32
  matmuls (~15 us on the PE, from the profile). A two-core split would halve K, but it
  needs a `sendrecv` of the `[T, 288]` partial logits and a core barrier. In the expert
  kernel, the barrier and the exchange alone cost several us. The expected net gain is
  <= ~5 us/layer (~0.2 ms/step), so I deferred it until it is measured.

**Deferred (not in this packet's goal):**

1. Router two-core split (see above).
2. fp8 stationaries as the default (3-11% faster on experts at T<=4).
3. Expert epilogue: store `[T, H]` without the 16 PE transposes (~6 us/layer fixed).
4. T=64 is vector-bound (~58 us per expert): the per-bank scale ops scale with T.
