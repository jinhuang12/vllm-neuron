# SPDX-License-Identifier: Apache-2.0
"""
vLLM Neuron Environment Variables Configuration

This module provides centralized environment variable management for vLLM Neuron

All environment variables are:
- Lazily evaluated when accessed
- Type-safe with proper validation
- Prefixed with VLLM_NEURON_ for namespace isolation
"""

import functools
import logging
import os
import subprocess
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from vllm_neuron import _artifact_paths

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # Core System Variables
    VLLM_NEURON_CPU_MODE: bool = False
    VLLM_NEURON_CPU_COMPILE: bool = False
    VLLM_NEURON_LOG_LEVEL: str = "INFO"
    VLLM_NEURON_DEBUG_MODE: bool = False
    VLLM_NEURON_BARRIER_TIMEOUT: int = 3600
    VLLM_NEURON_DISABLE_PARALLEL_TRACE: bool = False
    # Renderer thread-pool size for the EPD Router's HF preprocessing offload.
    # vLLM defaults this to 1; >1 parallelizes the GIL-releasing native image
    # decode/transform. Only safe because the Router disables the mm processor
    # cache (vLLM forbids >1 workers with the cache enabled).
    VLLM_NEURON_EPD_RENDERER_WORKERS: int = 4
    # TODO: Remove VLLM_NEURON_SWITCH_CC and derive topology from instance type.
    VLLM_NEURON_SWITCH_CC: bool = False
    VLLM_NEURON_MIN_KV_BUDGET_GIB: float = 1.0
    # Both KV budget knobs default to None, spelled "measured": the budget is the
    # measured free device memory less the compiled graphs' need.
    VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION: Optional[float] = None
    VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB: Optional[float] = None
    VLLM_NEURON_WORKER_TERMINATION_TIMEOUT: int = 5
    VLLM_NEURON_MLP_FORCE_TKG: bool = False
    VLLM_NEURON_DISABLE_NKI_KERNELS: bool = False
    VLLM_NEURON_SKIP_PREFILL_WARMUP: bool = False
    VLLM_NEURON_SKIP_DECODE_WARMUP: bool = False
    VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP: bool = False
    VLLM_NEURON_SKIP_ENCODER_WARMUP: bool = False
    # Ranks that load their compiled graph at the same time. A load holds far more
    # host memory than the execution after it, so the group goes through the
    # loader in waves of this many ranks. A value at or above the world size, or
    # an unset signal directory, loads every rank at once.
    VLLM_NEURON_NEFF_LOAD_WAVE_SIZE: int = 8
    # Directory the ranks signal each other through while they load in waves. Each
    # rank writes one flag file after its own load and never reads a later wave's.
    VLLM_NEURON_NEFF_LOAD_SIGNAL_DIR: str = ""
    # Seconds a wave waits for the wave before it. On expiry it logs and
    # proceeds, because a load that hangs the group is worse than one that
    # overlaps another.
    VLLM_NEURON_NEFF_LOAD_WAIT_TIMEOUT: int = 3600
    # Force the STATIC FP8 (non-MX) attention path on TRN3 even when STATIC_MX
    # kernels are available. Used by FP8 model factories as an escape hatch.
    VLLM_NEURON_FORCE_STATIC_FP8: bool = False
    # Snapshot Capture Variables
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE: bool = False
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL: Optional[str] = None
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN: Optional[str] = None
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_REQUEST: Optional[str] = None
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES: str = "4"
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS: Optional[str] = None
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_FORMAT: str = "pt"
    VLLM_NEURON_EFA_INSTANCE_FAMILY: str = ""
    # GLM-5.3-Flash host path. Both default off, which is the as-built serve.
    # Let GLM-5.3-Flash sample its full-vocabulary logits on device, which also lets
    # async scheduling stay on.
    VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING: bool = False
    # GLM-5.3-Flash shadow draft (MTP stage A): run the layer-45 draft head k times
    # per decode step beside the trunk and return the k draft ids; 0 = off. The
    # sampled tokens never read the draft. Read through ``mtp.shadow_draft_k()``.
    # Diagnostic (shadow) path: with ``--speculative-config '{"method": "mtp", ...}'``
    # the speculative config is the production reader of k, and a set knob must
    # agree with it.
    VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT: int = 0
    # Build the GLM-5.3-Flash step's attention metadata on the host only: no per-step
    # block-table / slot-mapping uploads that its graph never reads.
    VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA: bool = False
    # Where the GLM-5.3-Flash shadow draft (MTP stage A) writes its per-step scoring
    # records (JSONL, rank 0). Empty = no log, no scoring.
    VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG: str = ""
    # Worker GC policy after warmup (vllm_neuron/vllm/worker/gc_policy.py):
    # "off" (CPython's GC), "freeze_rare_gen2" (freeze + gen-2 threshold 100000), or
    # "freeze".
    VLLM_NEURON_GC_POLICY: str = "off"
    # GLM-5.3-Flash row-parallel all-reduce. Both defaults are the as-built path.
    # The dtype a row-parallel partial crosses the wire in: "fp32" or "bf16".
    VLLM_NEURON_TP_ALLREDUCE_DTYPE: str = "fp32"
    # Keep every all-reduce one collective instead of the compiler's 8 MiB tiles.
    VLLM_NEURON_TP_ALLREDUCE_FUSE: bool = False
    # GLM-5.3-Flash DSA prefill: a chunk whose top-k selection provably keeps every
    # token attends its latent window densely (mla_dense_window.py) instead of
    # selecting and gathering. On by default; 0 restores the sparse path.
    VLLM_NEURON_MLA_DENSE_WINDOW: bool = True
    # Divide the GLM-5.3-Flash DSA prefill selection's query rows over the
    # tensor-parallel ranks. On by default; 0 restores the replicated selection.
    VLLM_NEURON_DSA_INDEXER_SHARD: bool = True
    # GLM-5.3-Flash fused glue kernels (``vllm_neuron/functional/glue``): which
    # kernel serves which call, and how the KDA kernels load their weights.
    VLLM_NEURON_GLUE_FUSED: str = "1"
    VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE: bool = True


def maybe_convert_bool(value: str | None) -> bool | None:
    """
    Safely convert string to boolean using numeric conversion.

    Args:
        value: String value to convert ("0", "1") or None

    Returns:
        True if value is "1", False if value is "0", None if value is None

    Raises:
        ValueError: If value cannot be converted to int

    Examples:
        >>> maybe_convert_bool("1")
        True
        >>> maybe_convert_bool("0")
        False
        >>> maybe_convert_bool(None)
        None
        >>> maybe_convert_bool("invalid")  # Raises ValueError
    """
    if value is None:
        return None
    return bool(int(value))


def maybe_convert_int(value: str | None) -> int | None:
    """
    Safely convert string to integer.

    Args:
        value: String value to convert or None

    Returns:
        Integer value if conversion successful, None if value is None

    Raises:
        ValueError: If value cannot be converted to int

    Examples:
        >>> maybe_convert_int("600")
        600
        >>> maybe_convert_int("0")
        0
        >>> maybe_convert_int(None)
        None
        >>> maybe_convert_int("invalid")  # Raises ValueError
    """
    if value is None:
        return None
    return int(value)


def maybe_convert_float(value: str | None) -> float | None:
    """Safely convert string to float."""
    if value is None:
        return None
    return float(value)


#: The value that leaves a KV budget knob at its default: the measured budget.
MEASURED = "measured"


def maybe_measured_float(value: str | None) -> float | None:
    """Return None for an unset, empty or ``"measured"`` value, else the float.

    Raises:
        ValueError: the value is neither ``"measured"`` nor a number.

    Examples:
        >>> maybe_measured_float(None), maybe_measured_float("Measured")
        (None, None)
        >>> maybe_measured_float("2.5")
        2.5
    """
    if value is None or value.strip().lower() in ("", MEASURED):
        return None
    return float(value)


#: Device memory assumed for the compiled graphs, in GiB per logical NeuronCore,
#: when it cannot be read from the compile cache.
#:
#: The KV cache budget takes the graphs' need from the NEFFs the compile cache
#: holds for the served configuration
#: (:mod:`vllm_neuron.vllm.worker.neff_memory`). On a cold cache, where warmup
#: has yet to compile some of those graphs, and for graphs the cache cannot tie
#: to the configuration (a drafter's, a vision encoder's), this figure stands in.
#: It is the September measurement rounded up: a GLM-5.3-Flash prefill graph at
#: bucket 1024 held 4.567 GiB per rank (3.875 GiB of shared scratchpad plus 705
#: MiB of graph). ``VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB`` overrides both the
#: measured need and this figure.
DEFAULT_DEVICE_GRAPH_RESERVE_GIB = 5.0

#: What ``VLLM_NEURON_GLUE_FUSED=1``, and an unset switch, selects: each fused glue
#: kernel at the prefill row buckets where it beat its torch route, and nothing else.
#:
#: The buckets are the ones where the in-graph device A/B
#: (``test/hardware/benchmark_glue_block.py``) measured a win. It runs one
#: GLM-5.3-Flash KDA + MoE layer per graph at one TP=64 rank's shapes on trn2, and
#: compares each value with ``0``; ``reports/glue.md`` (round 2, the in-graph A/B
#: section) and ``reports/glue-c.md`` have the tables. Per layer:
#:
#: * ``mhc_pre:prefill@128`` and ``mhc_pre:prefill@1024``: the fused mHC pre-mix and
#:   collapse at both mHC sites, with the feed-forward RMSNorm at the feed-forward
#:   site, 274.9 us faster on a 2.54 ms 128-row layer and 6.92 ms faster on a 16.0 ms
#:   1024-row layer. At 1024 rows the torch route's collapse at the feed-forward site,
#:   whose rows the MoE router reads, compiles to a per-token loop.
#: * ``mhc_post:prefill@128`` and ``mhc_post:prefill@1024``: the bf16 mHC combine,
#:   45.2 us faster at 128 rows and 275.0 us faster on a 16.0 ms 1024-row layer, but
#:   197.1 us slower on a 9.07 ms 512-row layer. So the default names the measured
#:   buckets, not a range, and a row count that was not measured keeps the torch route.
#:
#: Two loads of one graph have measured up to 11 us apart, so a gain of 11 us or less
#: is not a win. kda_projections was 7.1 us faster at 128 rows, and 0.3 us slower when
#: the layer's two reductions were chains of 4 and 8 all-reduces: inside that bound, so
#: it is not in the default. kda_output was 24.1 us slower at 128 rows. No kernel is
#: selected at decode: on the served TP=64 line, ``all`` made the bs=1 decode step
#: 1.75 ms longer, while the single-rank benchmark (no tensor-parallel collectives)
#: measured it shorter. So every decode graph under ``1`` is the graph ``0`` traces.
#:
#: Measure a bucket before adding it, on a device lease, with
#: ``python test/hardware/benchmark_glue_block.py --output <json> --cases
#: kda:prefill:<rows> --variants off aa=0 <kernel> default``.
DEFAULT_GLUE_FUSED_SPEC = (
    "mhc_pre:prefill@128,mhc_pre:prefill@1024,mhc_post:prefill@128,mhc_post:prefill@1024")


environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Core System Variables ==================
    # Enable CPU fallback mode instead of using Neuron accelerators
    "VLLM_NEURON_CPU_MODE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_CPU_MODE")) or False
    ),
    # Enable CPU graph capture and compilation
    "VLLM_NEURON_CPU_COMPILE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_CPU_COMPILE")) or False
    ),
    # Logging level for vLLM Neuron components
    "VLLM_NEURON_LOG_LEVEL": lambda: os.getenv("VLLM_NEURON_LOG_LEVEL", "INFO").upper(),
    # Enable debug mode for verbose output and additional diagnostics
    "VLLM_NEURON_DEBUG_MODE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DEBUG_MODE")) or False
    ),
    # tp_barrier timeout in seconds (raise for long serial warmups)
    "VLLM_NEURON_BARRIER_TIMEOUT": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_BARRIER_TIMEOUT")) or 3600
    ),
    # Renderer thread-pool size for the EPD Router's HF preprocessing offload.
    "VLLM_NEURON_EPD_RENDERER_WORKERS": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_EPD_RENDERER_WORKERS")) or 4
    ),
    # When True, disable the parallel-trace fork pool entirely and run
    # graph extraction sequentially in the parent process.
    "VLLM_NEURON_DISABLE_PARALLEL_TRACE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DISABLE_PARALLEL_TRACE")) or False
    ),
    # Minimum KV budget (GiB) guardrail
    "VLLM_NEURON_MIN_KV_BUDGET_GIB": lambda: (
        maybe_convert_float(os.getenv("VLLM_NEURON_MIN_KV_BUDGET_GIB"))
        if os.getenv("VLLM_NEURON_MIN_KV_BUDGET_GIB") is not None
        else 1.0
    ),
    # Optional cap on the KV cache budget, as a fraction in (0, 1] of the
    # GMU-scaled total HBM. Unset or "measured" (the default): no cap, the budget
    # is the measured free HBM less the compiled graphs' need.
    "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION": lambda: maybe_measured_float(
        os.getenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION")
    ),
    # Optional override, in GiB per logical NeuronCore, of the device memory the
    # compiled graphs need. Unset or "measured" (the default): read from the
    # compile cache's NEFFs, or DEFAULT_DEVICE_GRAPH_RESERVE_GIB on a cold cache.
    "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB": lambda: maybe_measured_float(
        os.getenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB")
    ),
    # Local cache directory for model checkpoints
    "VLLM_NEURON_CHECKPOINT_CACHE": lambda: os.getenv(
        "NXDI_CHECKPOINT_CACHE", "/tmp/vllm_neuron-checkpoints"
    ),
    # The served GLM-5.3-Flash checkpoint directory (the one that holds
    # model.safetensors.index.json) that tests and benchmarks under test/ read
    # real weights from, through test/vllm_neuron/artifacts.py only. Unset or
    # empty: _artifact_paths.CHECKPOINT_DEFAULT, the serving host's campaign
    # checkpoint /home/ubuntu/glm53f-campaign/lane-serve/models/GLM-5.3-Flash-04c4e9e9.
    # A test that needs the checkpoint skips, naming the resolved path, when the
    # index is absent; it never substitutes random weights. The server does not
    # read this knob.
    "VLLM_NEURON_GLM5NEXT_CHECKPOINT_DIR": _artifact_paths.checkpoint_dir,
    # The directory the GLM-5.3-Flash performance campaign keeps its worktrees and
    # records under (glm53f-wt*/reports, glm53f-wt3/calib, the decode breakdown,
    # device profiles). Report scaffolding, perf harnesses and hardware benchmarks
    # under test/ resolve their default inputs and outputs below it, through
    # test/vllm_neuron/artifacts.py only. Unset or empty:
    # _artifact_paths.CAMPAIGN_DEFAULT, /home/ubuntu, the campaign hosts'
    # directory. A test whose record is absent below it skips or fails naming the
    # resolved path. The server does not read this knob.
    "VLLM_NEURON_GLM5NEXT_CAMPAIGN_DIR": _artifact_paths.campaign_dir,
    # Golden cache directory (disk tier)
    "VLLM_NEURON_GOLDEN_CACHE_DIR": lambda: os.path.expandvars(
        os.getenv("VLLM_NEURON_GOLDEN_CACHE_DIR", "/tmp/vllm_neuron-goldens-$USER")
    ),
    # Golden cache S3 URI (secondary tier, empty = disabled)
    "VLLM_NEURON_S3_GOLDENS_URI": lambda: os.getenv("VLLM_NEURON_S3_GOLDENS_URI", ""),
    # TODO: Remove VLLM_NEURON_SWITCH_CC and derive topology from instance type.
    # When True, uses contiguous groups for 8x8 topology instead of custom TRN2 mesh.
    "VLLM_NEURON_SWITCH_CC": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SWITCH_CC")) or False
    ),
    # Timeout in seconds for worker termination (SIGTERM→SIGKILL).
    # Default is 5s (upstream vLLM uses 4s for workers, 5s for API servers).
    # Increase when system profiling is enabled (NEURON_RT_INSPECT_ENABLE=1),
    # as the Neuron runtime needs time to flush profiling data after SIGTERM.
    "VLLM_NEURON_WORKER_TERMINATION_TIMEOUT": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_WORKER_TERMINATION_TIMEOUT")) or 5
    ),
    # Force TKG mode for the NKI MLP kernel (bypass CTE). Workaround for
    # nkilib CTE nc_transpose FP8 dtype mismatch on gen3+ (trn3).
    "VLLM_NEURON_MLP_FORCE_TKG": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_MLP_FORCE_TKG")) or False
    ),
    # Disable NKI kernels — forces can_run_kernel() to return False
    "VLLM_NEURON_DISABLE_NKI_KERNELS": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DISABLE_NKI_KERNELS")) or False
    ),
    # Keep libtorch_neuronx_lite's own compile cache keys: do not fold the NKI
    # kernel-source digest (vllm_neuron/compile_cache_key.py) into the graph and
    # kernel cache keys. A warm cache may then serve a stale kernel.
    "VLLM_NEURON_DISABLE_KERNEL_DIGEST_KEY": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DISABLE_KERNEL_DIGEST_KEY")) or False
    ),
    # Skip prefill warmup/compilation without requiring kv-transfer-config.
    # Useful for decode-only profiling workflows.
    "VLLM_NEURON_SKIP_PREFILL_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_PREFILL_WARMUP")) or False
    ),
    # Skip both prefill and decode (language-model) warmup without requiring a
    # transfer config. Used by EPD vision-only (VE) pools, which have no
    # language model (mirrors the prefill/decode levers).
    "VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP")) or False
    ),
    # Skip vision-encoder warmup without requiring a transfer config. Used by
    # EPD language-only (PD) pools, which have no vision encoder.
    "VLLM_NEURON_SKIP_ENCODER_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_ENCODER_WARMUP")) or False
    ),
    # Ranks that load their compiled graph at the same time.
    # A value at or above the world size loads every rank at once, and zero turns the
    # staging off, so the default applies only when the variable is unset.
    "VLLM_NEURON_NEFF_LOAD_WAVE_SIZE": lambda: (
        8
        if os.getenv("VLLM_NEURON_NEFF_LOAD_WAVE_SIZE") is None
        else maybe_convert_int(os.getenv("VLLM_NEURON_NEFF_LOAD_WAVE_SIZE"))
    ),
    # Directory the ranks signal each other through while they load in waves.
    # Empty means no staging: every rank loads as soon as it reaches the loader.
    "VLLM_NEURON_NEFF_LOAD_SIGNAL_DIR": lambda: os.getenv(
        "VLLM_NEURON_NEFF_LOAD_SIGNAL_DIR", ""
    ),
    # Seconds a wave waits for the wave before it, defaulting to the barrier timeout.
    # Zero waits for nothing, so the fall-through applies only when the variable is unset.
    "VLLM_NEURON_NEFF_LOAD_WAIT_TIMEOUT": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_NEFF_LOAD_WAIT_TIMEOUT"))
        if os.getenv("VLLM_NEURON_NEFF_LOAD_WAIT_TIMEOUT") is not None
        else (maybe_convert_int(os.getenv("VLLM_NEURON_BARRIER_TIMEOUT")) or 3600)
    ),
    # Skip decode warmup/compilation without requiring kv-transfer-config.
    # Useful for prefill-only profiling workflows.
    "VLLM_NEURON_SKIP_DECODE_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_DECODE_WARMUP")) or False
    ),
    # Force the STATIC FP8 (non-MX) attention path on TRN3 even when STATIC_MX
    # kernels are available (FP8 model factory escape hatch).
    "VLLM_NEURON_FORCE_STATIC_FP8": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_FORCE_STATIC_FP8")) or False
    ),
    # ================== Snapshot Capture Variables ==================
    # Master switch for NRT-boundary input snapshot capture; off by default.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE"))
        or False
    ),
    # Comma-separated 0-based call indices to capture (-1 = all); unset selects
    # the first post-warmup call.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL", None
    ),
    # Comma-separated generated-token indices to capture (decode only); the
    # forward whose inputs *produce* the token is captured. Unset disables it.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN", None
    ),
    # Comma-separated request_ids to capture; fires on any forward whose batch
    # contains the request. Matched against the caller's base id, ignoring the
    # unique suffix vLLM appends. Unset disables it.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_REQUEST": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_REQUEST", None
    ),
    # Hard cap on captured calls per process, so a broad rule cannot fill disk
    # or stall the engine. Defaults to 4; a non-positive or malformed override
    # falls back to that default.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES", "4"
    ),
    # Comma-separated tp_rank set to capture; unset captures all ranks (each
    # worker writes its own global-rank directory). Filtered on tp_rank, so
    # with DP>1 a value selects that tp position in every DP group.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS", None
    ),
    # On-disk artifact format the serialize op writes for each tensor{i}:
    # "pt" (pickled, any dtype) or "npy" (raw element bytes, value-preserving;
    # bf16/fp8 stored as |V2/|V1). "npy" pulls in a numpy read dependency on
    # replay. Defaults "pt". Validated against the write_tensors op's accepted
    # formats, so an unsupported value fails at startup.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_FORMAT": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_FORMAT", "pt"
    ),
    # Explicit instance-family override. When set, get_instance_family()
    # returns this verbatim instead of resolving from the sysfs product_name
    # (which defaults any trn3* to trn3pds and any trn2* to trn2).
    "VLLM_NEURON_EFA_INSTANCE_FAMILY": lambda: os.getenv(
        "VLLM_NEURON_EFA_INSTANCE_FAMILY", ""
    ),
    # ================== GLM-5.3-Flash Host Path ==================
    # Accept an on-device sampling config for GLM-5.3-Flash: its root hands the
    # full-vocabulary logits to ``functional/full_vocab_sampling.py`` and returns
    # token ids, so async scheduling is no longer turned off. Off by default.
    "VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING"))
        or False
    ),
    # The GLM-5.3-Flash shadow draft's iteration count k (MTP stage A); 0 = off.
    # The one definition of the knob: everything else calls ``mtp.shadow_draft_k()``,
    # which bounds the value, rather than reading the environment itself.
    "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT")) or 0
    ),
    # Skip the per-step device copies of block tables, slot mappings and cached
    # lengths for GLM-5.3-Flash, whose graph reads only host geometry. Off by default.
    "VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA"))
        or False
    ),
    # JSONL path for the GLM-5.3-Flash shadow-draft scoring records (MTP stage A);
    # empty = no log. Read with VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT (the draft count).
    "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG": lambda: (
        os.getenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", "") or ""
    ),
    # GC policy each worker applies once after warmup. "off" (the default) keeps
    # CPython's default GC: the freezing policies made every bs=1 decode step slower
    # on TP=64 (gc_policy.py). "freeze_rare_gen2" freezes the heap and raises only
    # the gen-2 threshold, so bs=64 decode has no full-pass stalls, at that bs=1
    # cost; "freeze" is the freeze alone. Any other value is refused when the policy
    # is applied.
    "VLLM_NEURON_GC_POLICY": lambda: os.getenv("VLLM_NEURON_GC_POLICY", "off"),
    # ================== GLM-5.3-Flash Row-Parallel All-Reduce ==================
    # The wire dtype of the tensor-parallel all-reduce at GLM-5.3-Flash's
    # row-parallel sites (``model/glm5_next/collective_policy.py``). "fp32" (the
    # default) reduces each rank's fp32 partial as computed; "bf16" rounds it to
    # bfloat16 first, which halves the bytes every rank moves. Case and spaces are
    # ignored; any other value is refused where it is read.
    "VLLM_NEURON_TP_ALLREDUCE_DTYPE": lambda: (
        os.getenv("VLLM_NEURON_TP_ALLREDUCE_DTYPE", "fp32").strip().lower()
    ),
    # "1" asks neuronx-cc to keep each all-reduce one collective. By default its
    # SimpleAllReduceTiling pass splits an all-reduce larger than 8 MiB into up to
    # four. Read by ``collective_policy.fuse_compiler_args``. Off by default.
    "VLLM_NEURON_TP_ALLREDUCE_FUSE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_TP_ALLREDUCE_FUSE")) or False
    ),
    # ================== GLM-5.3-Flash DSA Prefill ==================
    # Attend a prefill chunk densely over its latent window when the bound its graph
    # can prove, min(max_model_len, window rows), is within the DSA identity bound
    # (index_topk + index_kpool - 1 tokens): the top-k then keeps every token, so the
    # indexer's query side, scoring, top-k and the sparse gathers are skipped. Read at
    # trace time by ``model/glm5_next/dsa_dense_window.py``; a change needs a new
    # compile. On by default; set 0 to restore the sparse path.
    "VLLM_NEURON_MLA_DENSE_WINDOW": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_MLA_DENSE_WINDOW")) is not False
    ),
    # ================== GLM-5.3-Flash DSA Indexer ==================
    # Shard the DSA prefill selection (score GEMM, causal bound, top-k, sentinel
    # order) by query rows over the TP ranks, with one all-gather of the pool ids
    # per DSA layer (``functional/dsa/indexer_shard.py``). Read at trace time, so
    # a compiled graph keeps the value it was traced with. On by default; 0 runs
    # the replicated selection on every rank.
    "VLLM_NEURON_DSA_INDEXER_SHARD": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DSA_INDEXER_SHARD", "1"))
    ),
    # ================== GLM-5.3-Flash Fused Glue Kernels ==================
    # Which fused glue kernel (``vllm_neuron/functional/glue``) serves which call.
    # ``0``: none, every site takes its torch route. ``1`` (the default):
    # DEFAULT_GLUE_FUSED_SPEC. ``all``: every kernel at every phase and row count.
    # Otherwise a comma list of rules ``kernel[:phase][@rows]``: ``kernel`` is one
    # of mhc_pre, kda_projections, kda_output, mhc_post; ``phase`` is prefill,
    # decode or all (the default); ``rows`` is N, N-M, N- or -M (inclusive, N >= 1).
    # A call is fused when any rule selects it, and when the kernel's own shape
    # rules admit it. Read when a graph is traced; a malformed value raises
    # ValueError there. Example: ``mhc_post:prefill,mhc_pre:decode@2-64``.
    "VLLM_NEURON_GLUE_FUSED": lambda: os.getenv("VLLM_NEURON_GLUE_FUSED", "1").strip(),
    # How the KDA glue kernels load their weights' transposes: ``1`` (the default)
    # by DMA transpose, ``0`` by a plain DMA and tensor-engine transposes.
    "VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE": lambda: bool(
        maybe_convert_bool(os.getenv("VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE", "1"))
    ),
}


def __getattr__(name: str) -> Any:
    """
    Gets environment variables lazily.

    Falls through to libtorch_neuronx_lite.envs for NEURON_LIBTORCH_* names
    so callers can access libtorch env vars via this module.

    Args:
        name: Name of the environment variable to retrieve

    Returns:
        Value of the environment variable (type depends on variable)

    Raises:
        AttributeError: If environment variable name is not defined

    Examples:
        >>> import vllm_neuron.envs as envs
        >>> cpu_mode = envs.VLLM_NEURON_CPU_MODE  # Returns bool
        >>> log_level = envs.VLLM_NEURON_LOG_LEVEL  # Returns str
        >>> envs.VLLM_NEURON_NONEXISTENT  # Raises AttributeError
    """
    if name in environment_variables:
        return environment_variables[name]()
    if name.startswith("NEURON_LIBTORCH_"):
        import libtorch_neuronx_lite.envs as libtorch_envs

        return getattr(libtorch_envs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    Return list of available environment variables.

    Returns:
        List of all defined environment variable names

    Examples:
        >>> import vllm_neuron.envs as envs
        >>> variables = dir(envs)
        >>> 'VLLM_NEURON_CPU_MODE' in variables
        True
    """
    return list(environment_variables.keys())


def is_set(name: str) -> bool:
    """
    Check if an environment variable is explicitly set.

    Args:
        name: Name of the environment variable to check

    Returns:
        True if environment variable is explicitly set, False otherwise

    Raises:
        AttributeError: If environment variable name is not defined

    Examples:
        >>> import os
        >>> import vllm_neuron.envs as envs
        >>> os.environ['VLLM_NEURON_CPU_MODE'] = '1'
        >>> envs.is_set('VLLM_NEURON_CPU_MODE')
        True
        >>> envs.is_set('VLLM_NEURON_LOG_LEVEL')  # Not set, uses default
        False
        >>> envs.is_set('VLLM_NEURON_NONEXISTENT')  # Raises AttributeError
    """
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_native_backend() -> bool:
    """Return True when VLLM_NEURON_BACKEND=neuron_native."""
    return os.getenv("VLLM_NEURON_BACKEND", "").lower() == "neuron_native"


def get_compile_backend_name() -> str:
    """Return the torch.compile backend name.

    The native route registers lite's backend on demand and returns its name.
    That name is intentionally not "neuron": "neuron" stays reserved for the
    torch_neuronx package so a separate install keeps upstream semantics.
    The XLA route uses "neuron_libtorch", registered at lite import.
    """
    if is_native_backend():
        from libtorch_neuronx_lite.compile.native_backend import register

        # register() is idempotent: repeated calls return the same name
        # without re-registering, so calling it from this getter is safe.
        return register()
    return "neuron_libtorch"


def get_dist_backend() -> str:
    """Return the distributed backend for the selected execution route.

    The native route needs Gloo for CPU control-plane collectives and Neuron
    for PrivateUse1 tensor collectives, so it returns a composite backend that
    lets PyTorch dispatch each collective by its tensor device.
    """
    return "cpu:gloo,neuron:neuron" if is_native_backend() else "gloo"


def get_neuron_compile_cache_dir() -> str:
    """Return the local compile cache directory derived from VLLM_CACHE_ROOT.

    If ``VLLM_CACHE_ROOT`` is explicitly set by the user, returns
    ``$VLLM_CACHE_ROOT/neuron/compile_cache`` unconditionally — the user's
    explicit choice is always respected.

    If ``VLLM_CACHE_ROOT`` is not set (default ``~/.cache/vllm``), probes
    the home directory filesystem type.  On multi-node clusters,the home directory
    can be NFS-mounted, which is incompatible with FileLock semantics.  In that case
    the function falls back to ``/tmp/vllm_neuron_wdir_$USER/neuron/compile_cache`` and logs a warning.

    Returns:
        str: Absolute path to the local Neuron compile cache directory.

    Examples:
        >>> import os
        >>> os.environ["VLLM_CACHE_ROOT"] = "/tmp/my_cache"
        >>> get_neuron_compile_cache_dir()
        '/tmp/my_cache/neuron/compile_cache'
    """
    # Cache by VLLM_CACHE_ROOT so the FS probe + NFS-fallback warning only
    # runs once per cache root per process.
    return _resolve_neuron_compile_cache_dir(os.environ.get("VLLM_CACHE_ROOT"))


@functools.lru_cache
def _resolve_neuron_compile_cache_dir(cache_root_override: Optional[str]) -> str:
    """Cached resolver for ``get_neuron_compile_cache_dir``.

    Keyed on the caller-supplied ``VLLM_CACHE_ROOT`` value so different
    overrides each get their own cache slot.
    """
    if cache_root_override is not None:
        # User made an explicit choice — respect it unconditionally.
        return os.path.join(cache_root_override, "neuron/compile_cache")

    # Probe the home directory to detect remote filesystem mounts (NFS or Lustre).
    # We probe ~ (guaranteed to exist) rather than ~/.cache/vllm (may not exist yet).
    if _path_is_remote_filesystem(os.path.expanduser("~")):
        fallback = os.path.join(
            os.path.expandvars("/tmp/vllm_neuron_wdir_$USER"), "neuron/compile_cache"
        )
        logger.warning(
            "Default compile cache path ~/.cache/vllm/neuron/compile_cache is on a remote filesystem (NFS or Lustre). "
            "Falling back to %s.",
            fallback,
        )
        return fallback

    return os.path.join(os.path.expanduser("~/.cache/vllm"), "neuron/compile_cache")


def _path_is_remote_filesystem(path: str) -> bool:
    """Return True if *path* is on a remote filesystem (NFS or Lustre).

    Args:
        path: An existing filesystem path to check.

    Returns:
        bool: True if the filesystem is NFS (any variant) or Lustre
        (including FSx for Lustre), False otherwise.
    """
    result = subprocess.check_output(
        ["stat", "-f", "-c", "%T", path],
        stderr=subprocess.STDOUT,
        text=True,
    )
    fs_type = result.strip().lower()
    return fs_type.startswith("nfs") or fs_type == "lustre"


def get_neuron_snapshot_dir() -> str:
    """Return the snapshot bundle root, derived like the compile cache dir.

    Snapshots reference the HLO/NEFF that live under the compile cache, so they
    resolve off the same ``VLLM_CACHE_ROOT`` base and inherit the same NFS/Lustre
    fallback. The two trees sit side by side (``neuron/compile_cache`` and
    ``neuron/snapshots``) under one root.

    Examples:
        >>> import os
        >>> os.environ["VLLM_CACHE_ROOT"] = "/tmp/my_cache"
        >>> get_neuron_snapshot_dir()
        '/tmp/my_cache/neuron/snapshots'
    """
    return _resolve_neuron_snapshot_dir(os.environ.get("VLLM_CACHE_ROOT"))


@functools.lru_cache
def _resolve_neuron_snapshot_dir(cache_root_override: Optional[str]) -> str:
    """Cached resolver for ``get_neuron_snapshot_dir``.

    Mirrors ``_resolve_neuron_compile_cache_dir``: an explicit ``VLLM_CACHE_ROOT``
    is honored unconditionally, otherwise the default home location is used with
    a ``/tmp`` fallback when home is on a remote filesystem.
    """
    if cache_root_override is not None:
        return os.path.join(cache_root_override, "neuron/snapshots")

    if _path_is_remote_filesystem(os.path.expanduser("~")):
        fallback = os.path.join(
            os.path.expandvars("/tmp/vllm_neuron_wdir_$USER"), "neuron/snapshots"
        )
        logger.warning(
            "Default snapshot path ~/.cache/vllm/neuron/snapshots is on a remote filesystem (NFS or Lustre). "
            "Falling back to %s.",
            fallback,
        )
        return fallback

    return os.path.join(os.path.expanduser("~/.cache/vllm"), "neuron/snapshots")
