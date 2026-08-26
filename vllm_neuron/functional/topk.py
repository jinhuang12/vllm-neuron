# SPDX-License-Identifier: Apache-2.0
"""
Distributed topk for tensor-parallel inference.

This module provides a distributed topk operation that works across
sharded tensors in a tensor-parallel setting.
"""

import logging

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import all_gather_tensor

import nki.language as nl
from nkilib.core.topk.rotational_topk import rotational_topk
from vllm_neuron.nki.nki_dtype import torch_to_nki_dtype
from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

logger = logging.getLogger(__name__)

# LNC grid degree passed to the rotational top-k config factories. The feasibility
# gate (``_rotational_topk_config_compiles``) and the real-run config builder
# (``_get_rotational_topk_config``) MUST use the same value so the gate evaluates
# the exact config the real run will build; keep it as one source of truth here.
# (The kernel forces n_prgs to 1 internally when BxS == 1.)
_ROTATIONAL_TOPK_NUM_PROGRAMS = 2


def topk(
    tensor: Tensor,
    k: int,
    dim: int,
    gather_dim: int,
    process_group: ProcessGroup | None = None,
    rank: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Performs distributed topk on sharded tensors using 2-step algorithm.

    Uses the rotational NKI top-k kernel automatically wherever it can run
    (see ``_can_use_nki_topk``), falling back to ``torch.topk`` otherwise.

    Args:
        tensor: Input tensor to perform topk on.
        k: Number of top elements to select.
        dim: Dimension along which to find topk.
        gather_dim: Dimension the tensor is sharded on.
        process_group: Process group for distributed operations.
        rank: Rank of this process within the group, as a scalar tensor
            (dtype int32). Required when ``process_group`` is set and
            ``gather_dim == dim``. Passing rank as a tensor avoids baking
            it as a constant during compilation, which would cause
            redundant per-rank compilations.

    Returns:
        Tuple of (values, indices) with global topk across all shards.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from vllm_neuron.functional.topk import topk
        >>>
        >>> # Tensor-parallel inference with TP=2, vocab_size=256
        >>> # Each rank has logits of shape (batch=1, vocab_per_rank=128)
        >>> logits = torch.randn(1, 128)  # Local shard on this rank
        >>> pg = dist.new_group([0, 1])
        >>> rank_t = torch.tensor(dist.get_rank(pg), dtype=torch.int32)
        >>>
        >>> # Compute global top-10 across both ranks
        >>> values, indices = topk(logits, k=10, dim=1, gather_dim=1,
        ...                        process_group=pg, rank=rank_t)
        >>> # Returns shapes (1, 10) with global values and indices
    """
    # Fast path for single device (no process group or tp_degree=1)
    if process_group is None:
        tp_degree = 1
    else:
        tp_degree = torch.distributed.get_world_size(group=process_group)

    if tp_degree == 1:
        if _can_use_nki_topk(tensor, k, dim):
            # Use NKI kernel on Neuron
            return _topk_nki(tensor, k, dim)
        else:
            # CPU fallback
            return _topk_torch(tensor, k, dim)

    # Convert to float32 for kernel accuracy
    ## TODO: remove after migrating topk to NKIv2 frontend
    tensor = tensor.to(torch.float32)

    # Step 1: Local topk on each rank
    local_k = min(k, tensor.shape[gather_dim]) if gather_dim == dim else k
    if _can_use_nki_topk(tensor, local_k, dim):
        local_value, local_index = _topk_nki(tensor, local_k, dim)
    else:
        local_value, local_index = _topk_torch(tensor, local_k, dim)

    # Apply sharding offset locally before gather (when gather_dim == dim)
    if gather_dim == dim:
        if rank is None:
            raise ValueError(
                "rank must be provided as a tensor when process_group is set "
                "and gather_dim == dim, to avoid baking rank as a compile-time "
                "constant. Pass torch.tensor(dist.get_rank(group), dtype=torch.int32)."
            )
        offset = rank * tensor.shape[gather_dim]
        local_index = local_index + offset

    # Step 2: Gather from all ranks
    global_values = all_gather_tensor(local_value, gather_dim, group=process_group)
    global_indices = all_gather_tensor(local_index, gather_dim, group=process_group)

    # Step 3: Global topk on gathered results
    if _can_use_nki_topk(global_values, k, dim):
        values, global_max_local_index = _topk_nki(global_values, k, dim)
    else:
        values, global_max_local_index = _topk_torch(global_values, k, dim)

    # Convert indices to match global_indices dtype for gather
    global_max_local_index = global_max_local_index.to(dtype=global_indices.dtype)
    final_indices = torch.gather(global_indices, dim, global_max_local_index)

    return values, final_indices


def batch_sharded_topk(
    tensor: Tensor,
    k: int,
    dim: int,
    dp_degree: int,
    all2all_group: ProcessGroup,
    gather_group: ProcessGroup,
) -> tuple[Tensor, Tensor]:
    """Distributed topk with batch sharding for data-parallel sampling.

    Uses all-to-all to redistribute batch→vocab, performs local topk on larger
    vocab slices, then gathers results for final global topk.

    Flow:
        1. All-to-all: [batch, vocab/TP] → [batch/DP, vocab/TP × DP]
        2. Local topk on each rank's larger vocab slice
        3. Index correction: local indices → global vocab indices
        4. Gather topk results across gather_group
        5. Final topk on gathered results

    Args:
        tensor: Input tensor to perform topk on. Shape: [batch, vocab_per_rank].
        k: Number of top elements to select.
        dim: Dimension along which to find topk (typically -1 for vocab).
        dp_degree: Data-parallel degree (batch sharding factor).
        all2all_group: Process group for batch↔vocab redistribution.
        gather_group: Process group for gathering topk results.

    Returns:
        Tuple of (values, indices) with global topk across all shards.
        Shapes: [batch/DP, k] for both values and indices.

    Example:
        >>> # TP=8, DP=2, batch=8, vocab_per_rank=16K
        >>> logits = torch.randn(8, 16384)
        >>> values, indices = batch_sharded_topk(
        ...     logits, k=10, dim=-1, process_group=tp_group,
        ...     dp_degree=2, all2all_group=..., gather_group=..., all2all_mesh=...
        ... )
        >>> values.shape, indices.shape  # ([4, 10], [4, 10])
    """
    from vllm_neuron.functional import all_to_all

    # Step 1: All-to-all redistribution (batch → vocab)
    tensor = all_to_all(
        tensor,
        split_dim=0,
        concat_dim=dim,
        group=all2all_group,
    )

    bs = tensor.shape[0]  # batch/DP
    sharded_size = tensor.shape[dim] // dp_degree

    # Convert to float32 for kernel accuracy
    ## TODO: remove after migrating topk to nki frontend
    tensor = tensor.to(torch.float32)

    # Step 2: Local topk on larger vocab slice
    if _can_use_nki_topk(tensor, k, dim):
        local_value, local_index = _topk_nki(tensor, k, dim)
    else:
        local_value, local_index = _topk_torch(tensor, k, dim)

    # Step 3: Index correction (local → global vocab indices)
    # After all-to-all, this rank's vocab is concatenated from the ranks in its
    # all2all group. The shard order matches the rank order in the group, so we
    # just need this rank's all2all group members to map shard ID → global rank.
    rank_map = torch.tensor(
        dist.get_process_group_ranks(all2all_group),
        dtype=torch.int32,
        device=tensor.device,
    )

    # Decompose local index: which shard + offset within shard
    quotient = local_index // sharded_size  # which shard (0, 1, 2, ...)
    remainder = local_index % sharded_size  # position within shard
    flat_quotient = quotient.flatten()
    shard_rank_id_flat = rank_map[flat_quotient]
    shard_rank_id = shard_rank_id_flat.view(
        quotient.size(0), quotient.size(1)
    )  # Dynamic shape

    # Convert to global vocab index
    global_index = shard_rank_id * sharded_size + remainder

    # Step 4: Gather topk results across gather_group
    topk_results = torch.cat([local_value, global_index.float()], dim=0)
    gathered_results = all_gather_tensor(topk_results, dim, group=gather_group)
    global_values, global_indices = torch.split(gathered_results, bs, dim=0)
    global_indices = global_indices.to(torch.int32)

    # Step 5: Final topk on gathered results
    if _can_use_nki_topk(global_values, k, dim):
        values, final_local_idx = _topk_nki(global_values, k, dim)
    else:
        values, final_local_idx = _topk_torch(global_values, k, dim)
    final_indices = torch.gather(global_indices, dim, final_local_idx.to(torch.int32))

    return values, final_indices


def _can_use_nki_topk(tensor: Tensor, k: int, dim: int) -> bool:
    """True if the rotational NKI top-k kernel can handle this (tensor, k, dim).

    The kernel (nkilib.core.topk.rotational_topk) reduces over the LAST
    dimension. Following the standard vLLM-Neuron kernel pattern (cf.
    functional.argmax._can_use_nki_max), the kernel is used automatically
    whenever it can run; everything else falls back to ``torch.topk``.

    The cheap checks below are NECESSARY conditions (runtime availability,
    rank/dim/dtype, and ``0 < k < vocab_size``). They are NOT sufficient: the
    kernel also has a shape-dependent feasibility envelope -- it splits the vocab
    into ``n_stages`` and asserts the concatenated SBUF free dimension
    ``chunk + n_stages * local_k <= 16384``. That bound is non-monotonic in k
    (small k pins ``n_stages`` at the HW minimum, so the per-stage chunk stays
    large and overflows), so it cannot be captured by static vocab/k caps. It is
    decided authoritatively by ``_rotational_topk_config_compiles`` below, which
    dry-runs the kernel's OWN config factories -- the only drift-proof way to
    admit exactly the set the kernel can compile. (Static caps were both unsafe
    and over-conservative: k=2/vocab=32768 passed them but the kernel asserts at
    concatenated_free=16400 > 16384, while the real gpt-oss vocab=201088 was
    needlessly rejected by a 151936 "tested envelope" cap it actually handles.)
    """
    if not can_run_kernel(tensor):
        return False
    if tensor.ndim < 2 or dim not in (-1, tensor.ndim - 1):
        return False
    # Kernel is tested only for bf16/fp32 (its pad sentinel _get_dtype_min has no
    # fp16 branch); sampling logits are bf16, distributed paths upcast to fp32.
    if tensor.dtype not in (torch.bfloat16, torch.float32):
        return False
    vocab_size = tensor.shape[-1]
    # Strict k < vocab_size: the rotational kernel rejects k == vocab_size for
    # sorted output ("sorted=True is not supported when k == vocab_size"), which
    # arises e.g. for small-vocab models. torch.topk handles k == vocab_size, so
    # leave that (degenerate sort-everything) case to the fallback. NOTE this
    # pre-filter is load-bearing: the dry-run below builds the config (which does
    # NOT check k==vocab_size -- that assert lives in the kernel body) and would
    # otherwise admit k==vocab_size at small vocab. Do not relax to ``k <= vocab``.
    if not (0 < k < vocab_size):
        return False
    n_rows = 1
    for d in tensor.shape[:-1]:
        n_rows *= d
    nki_dtype = getattr(nl, torch_to_nki_dtype(tensor.dtype))
    return _rotational_topk_config_compiles(n_rows, vocab_size, k, nki_dtype)


# Authoritative compile-time feasibility gate for the rotational NKI top-k
# kernel: run the kernel's OWN config factories and report whether they build
# without raising. This admits EXACTLY the (n_rows, vocab_size, k) set the kernel
# can compile and cannot drift when nkilib changes HW_PARAMS or its staging cost
# model. We stop at create_rotational_topk_config -- prepare_rotational_constants
# (the numpy/temp-file materialization, done by _topk_nki for the real run) is
# not needed here because every validity assert (kernel_assert) fires inside the
# two factories.
#
# assume_constant_result is REQUIRED, for the same reason _get_rotational_topk_config
# uses it: create_topk_config emits logger.info when BxS == 1, which Dynamo cannot
# trace under fullgraph=True. Folding the returned bool to a constant runs this
# helper eagerly (off-graph) exactly once per distinct (n_rows, vocab_size, k,
# dtype) and keeps it out of the compiled graph. The try/except MUST live inside
# the helper so the AssertionError is consumed eagerly and Dynamo only ever sees a
# clean bool constant. num_programs comes from _ROTATIONAL_TOPK_NUM_PROGRAMS so the
# gate evaluates the same config the real run (_get_rotational_topk_config) builds.
#
# CAVEAT: the kernel signals infeasibility via ``kernel_assert`` (a bare ``assert``).
# Under ``python -O`` / ``PYTHONOPTIMIZE`` asserts are stripped, so the factories
# would NOT raise on an out-of-envelope shape and this gate would wrongly return
# True (the kernel then hard-fails at real compile with no torch fallback). vLLM
# Neuron is not run under -O, but this is the one mode where the gate is unsafe.
@torch._dynamo.assume_constant_result
def _rotational_topk_config_compiles(
    n_rows: int, vocab_size: int, k: int, nki_dtype
) -> bool:
    from nkilib.core.topk.rotational_topk import (
        create_rotational_topk_config,
        create_topk_config,
    )

    try:
        topk_config = create_topk_config(
            inp_shape=(n_rows, vocab_size),
            inp_dtype=nki_dtype,
            k=k,
            num_programs=_ROTATIONAL_TOPK_NUM_PROGRAMS,
        )
        create_rotational_topk_config(
            inp_shape=(n_rows, vocab_size), topk_config=topk_config
        )
        return True
    except AssertionError:
        # The kernel signals an out-of-envelope shape via kernel_assert -> a plain
        # AssertionError. That is the ONLY expected "cannot run" signal, so treat it
        # as "fall back to torch.topk". Any OTHER exception (e.g. an ImportError or
        # a TypeError from an nkilib API/signature change) is NOT a feasibility
        # answer -- swallowing it would silently disable the kernel for every shape
        # with no test catching the regression, so let it propagate loudly instead.
        return False


# Build the rotational top-k config (the kernel's compile-time companion object)
# once at trace time and fold it in as a constant. assume_constant_result is
# required because this config-production logic is host-side Python that Dynamo
# cannot (and should not) trace into the graph: it constructs numpy rotation/
# permutation matrices and materializes them as shared constants backed by temp
# files, and emits progress logs. None of that is a graph operation -- it is
# setup that produces a frozen, shape-determined config. Folding it to a constant
# (same technique as nki_hop.register_kernel_to_torch) runs it exactly once per
# distinct (shape, dtype, k) and keeps it out of the compiled graph; without it,
# fullgraph=True tracing fails on the untraceable host code (the print/logger
# calls are merely the first thing Dynamo trips over).
@torch._dynamo.assume_constant_result
def _get_rotational_topk_config(n_rows: int, vocab_size: int, k: int, nki_dtype):
    from nkilib.core.topk.rotational_topk import (
        create_rotational_topk_config,
        create_topk_config,
        prepare_rotational_constants,
    )

    topk_config = create_topk_config(
        inp_shape=(n_rows, vocab_size),
        inp_dtype=nki_dtype,
        k=k,
        num_programs=_ROTATIONAL_TOPK_NUM_PROGRAMS,  # kernel forces 1 when BxS == 1
    )
    config = create_rotational_topk_config(
        inp_shape=(n_rows, vocab_size), topk_config=topk_config
    )
    return prepare_rotational_constants(config)


def _topk_nki(tensor: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    """Top-k via the rotational NKI kernel; reduces over the last dim.

    Returns sorted-descending (values, int64 indices) to match torch.topk
    (the kernel emits uint32 indices).
    """
    if dim < 0:
        dim += tensor.ndim
    assert dim == tensor.ndim - 1, "rotational NKI top-k reduces over the last dim"

    vocab_size = tensor.shape[-1]
    leading_shape = tensor.shape[:-1]
    inp2d = tensor.reshape(-1, vocab_size)
    n_rows = inp2d.shape[0]

    # --- ep18/LD-71 single-row dispatch gate (F-223, F-227) ----------------
    # At BxS == 1 the vendor config factory silently downgrades to a
    # single-program plan (``rotational_topk_utils.py:209-212``) while the
    # kernel body asserts the two-program shape, and at the deployed compiler
    # pin the LNC-2 lowering then silently ERASES the kernel on core 1
    # ("skip lowering", TranslateNKIASTToBIR), leaving producer-less
    # consumers whose never-written indices feed ``oob_is_err`` DMA and
    # present as an all-core NRT_TIMEOUT (F-223, ep18 ITER-63..67).
    # Dispatch the single-row case to the numerically matched torch
    # composition instead of launching ``topk_kernel[config.n_prgs]``.
    # DEPLOYED-PIN WORKAROUND, NOT A FIX (F-227): the vendor defect class is
    # unchanged, and mitigating it claims at most servability — parity
    # against the recorded standard stays the arbiter of health (F-224).
    # Both single-row sampling stages ([1, 2020] and [1, 16384]) take this
    # branch; the multi-row (router) path below is untouched.
    if n_rows == 1:
        return _topk_torch(tensor, k, dim)
    # --- end ep18/LD-71 gate ------------------------------------------------

    nki_dtype = getattr(nl, torch_to_nki_dtype(inp2d.dtype))
    config = _get_rotational_topk_config(n_rows, vocab_size, k, nki_dtype)

    topk_kernel = wrap_nki(rotational_topk)
    values, indices = topk_kernel[config.n_prgs](inp2d, config)

    out_shape = (*leading_shape, k)
    values = values.reshape(out_shape)
    indices = indices.reshape(out_shape).to(torch.int64)
    return values, indices


def _topk_torch(tensor: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    """
    Compute topk using torch.topk.

    Non-NKI fallback for inputs the rotational top-k kernel cannot handle (see
    ``_can_use_nki_topk``). This is the path taken on CPU, when the kernel runtime
    is disabled, and on Neuron when the input is out of the kernel's envelope --
    fp16, k == vocab_size, or a (vocab, k) whose staging exceeds the kernel's
    concatenated free-dimension HW limit.

    Args:
        tensor: Input tensor.
        k: Number of top elements to select.
        dim: Dimension along which to find topk.

    Returns:
        Tuple of (values, indices) with top-k elements.
    """
    return torch.topk(tensor, k, dim=dim)
