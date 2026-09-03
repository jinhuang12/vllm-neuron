# SPDX-License-Identifier: Apache-2.0
"""In-place paged KV-cache row write (ladder decision LD-79, plan §20.2 Rule 4′).

``kv_cache_write(cache, slot_ids, rows, *, valid_mask=None)`` writes one row per
token into a paged cache ``[num_blocks, num_kv_heads, block_size, width]``,
skipping padded/masked slots, with semantics BITWISE identical to the model-side
incumbent ``_masked_scatter_rows`` composition it replaces (deepseek_v4
``attention.py:771`` at ``9c461c7f``; that helper survives ONLY as this op's
torch fallback body — plan §20.2).

RULE 4′ IS THE ONE FACT THAT DOMINATES THIS FILE: no cache-page-sized scatter
may exist ANYWHERE in the traced production graph. Serve-38 measured the
deployed compiler materializing every declared-aliased page scatter as an
88,080,384 B Internal tensor plus input-side copies (assessment §18, F-245), so
a torch composition of this write CANNOT be the traced path — it would re-emit
exactly the scatter class LD-79 removes. The dispatch below therefore has NO
silent-fallback leg on device:

* gate TRUE  -> the NKI kernel (in-place row write, cache threaded in/out by the
  FX aliasing machinery — the ``attention_decode.py`` ``update_cache=True``
  contract LD-75 already uses for decode writes).
* gate FALSE on a CPU tensor -> the torch fallback (CPU equivalence and the
  simulator A/B's fallback leg ONLY — never traced for production).
* gate FALSE on a NON-CPU tensor -> ``RuntimeError``, loudly (the
  ``moe_cte.py:288-296`` STATIC_MX precedent). A ``RuntimeError`` naming
  ``VLLM_NEURON_DISABLE_NKI_KERNELS`` here is this design, not a new defect.

THE TRIAD (author_kernel_triads, prereg TRIADS79-Z0-PREREGISTRATION.md sealed
before any edit): numerics declaration ``numerics/kv_cache_write.declaration.json``
(BIT-EXACT, rtol=atol=0, committed FIRST at C1) + the ``@nki.jit`` kernel below +
this dispatch + the D8 envelope caps in the gate. The kernel is PURE DATA
MOVEMENT — no in-kernel cast exists; the single write cast ``rows.to(cache.dtype)``
happens torch-side in the dispatch wrapper, the SAME ``.to()`` the fallback
performs, so written payload bytes are the incumbent's by construction (Z0 §D2).
Zero-padding to the cache width happens in-kernel by memset-0 staging (fp8
zero bytes == ``torch.zeros`` bytes). The cache is passed WHOLE and UNRESHAPED
so the write alias threads to the root parameter (``operand_output_aliases``
frontend-derived from the kernel RETURNING its mutated input); the kernel
flattens internally via zero-copy ``.reshape`` (Z0 §D3, the LD-75 boundary
discipline). The op RETURNS the written cache (Z0 §D1 contract completion:
the c13 harness grades RETURNED tensors; every call site discards the return;
the traced-path in-place threading is the HOP aliasing, independent of the
python return). R-20c fold-back note (plan §20.5): if the FX aliasing pass
demands a consumer for a standalone in-place write, the recorded fold-back is
moving the write into the LD-76/LD-75 kernels per class — a plan-level branch,
never improvised at this surface.

Slot-id contract: ids stay int32-safe; the C2/ESFH001 clamp FORM below is
consumed as-is, never reverted (E-1 is settled on hardware; ITER-22). fp8 caches
are indexed DIRECTLY — do not reintroduce a dtype view (F-5, probe ep9-P3:
``Tensor.view(<dtype>)`` does not lower through ``convert_fx_to_hlo``;
``index_select`` / ``where`` / ``index_put_`` all lower on ``float8_e4m3fn``).

``_pad_columns`` is a local copy of the model-side helper rather than an import:
``functional/`` must not depend on ``model/`` (the ``block_fp8_linear.py``
``_pow2_ceil_scale`` local-copy precedent, its :475-483 rationale). The two must
stay identical; the G-B write-rung leg grades that equality bitwise.
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode
from nkilib.core.utils.kernel_assert import kernel_assert

import torch
from torch import Tensor

from vllm_neuron import envs
from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

# ---------------------------------------------------------------------------
# Module constants (local copies; see module docstring for why not imported)
# ---------------------------------------------------------------------------

#: Sentinel for padded tokens and — on compressed layers — tokens whose
#: compression group has not closed yet. Those slots keep what they hold.
_PAD_SLOT_ID: int = -1

#: Upper clamp bound: min-only s64 clamp synthesizes iinfo(int64).max at
#: FX->HLO; production warmup feeds slot ids int64 — NCC_ESFH001 fix, ITER-22
#: (a C2 site; FORM preserved verbatim, never reverted).
_INT32_MAX: int = 2**31 - 1

#: Partition tile: at most 128 id rows per staging tile (SBUF partition max;
#: the LD-75 write-block convention).
_PMAX: int = 128

#: Widest validated cache class (compressor state, [nb, 1, 32, 2688]) — the
#: D8 envelope's width cap. Wider caches decline the gate (CPU fallback or a
#: loud device raise; over-envelope-width declaration case).
_MAX_WIDTH: int = 2688

#: Id-tile bound: covers every production token-batch shape with headroom
#: (Z0 §D8).
_MAX_T: int = 65536


# ---------------------------------------------------------------------------
# NKI kernel (triad LD-79; prereg Z0 §D3/§D4 — deployed pins NKI 0.5.0)
# ---------------------------------------------------------------------------


@nki.jit
def _kv_cache_write_nki_kernel(cache, wr_ids, rows):
    """In-place paged cache row write — pure data movement, Rule 4′.

    Args (kernel view):
        cache: 4-D paged cache, WHOLE/UNRESHAPED (write-aliased root
            parameter), flattened here via zero-copy ``.reshape``.
        wr_ids: ``[T, 1]`` uint32 flat slot ids; uint32-max = skip (the
            ``oob_mode.skip`` sentinel — R2 ``_update_block_cache_vectorized``
            idiom, the same one LD-75's write block uses).
        rows: ``[T, n]`` payload ALREADY in cache dtype (the wrapper's single
            torch-side cast, Z0 §D2 — no in-kernel cast exists), ``n <= width``.

    Per ≤128-row tile: load the id column, stage ``[tb, width]`` in SBUF
    (memset-0 ONLY when ``n < width`` — full-width rows are wholly overwritten
    by the copy; Amendment 11 perf), copy the payload columns, then ONE
    indirect scatter DMA into the flat cache with oob-skip. Returns the
    mutated ``cache`` as the sole (aliased) output — ``compile_nki`` derives
    ``operand_output_aliases`` from exactly this return, and the HOP
    functionalize pass threads the mutation to the caller
    (``nki_hop.py:529-534``).

    Parser-legality rails carried from LD-75 P4 r2/r3: no tuple-unpacking
    loop targets, no comprehensions, no zip/enumerate, no starred expansion
    (tuple-unpack ASSIGNMENT is legal).
    """
    nb, kvh, bs, width = cache.shape
    flat = cache.reshape((nb * kvh * bs, width))
    T = wr_ids.shape[0]
    n = rows.shape[1]
    kernel_assert(n <= width, "write row wider than cache width")
    for t0 in range(0, T, _PMAX):
        tb = min(_PMAX, T - t0)
        # id column [tb, 1] uint32 (the LD-75 _load_ids form; [T, 1] ids make
        # the flat offset of row t0 exactly t0)
        wid = nl.ndarray((tb, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=wid, src=wr_ids.ap(pattern=[[1, tb], [1, 1]], offset=t0)
        )
        # SBUF staging tile [tb, width] in cache dtype
        wst = nl.ndarray((tb, width), dtype=flat.dtype, buffer=nl.sbuf)
        if n < width:
            nisa.memset(wst, 0.0)  # zero pad == _pad_columns torch.zeros bytes
        nisa.dma_copy(dst=wst[0:tb, 0:n], src=rows[t0 : t0 + tb, 0:n])
        # one indirect scatter DMA per tile; uint32-max rows skip
        nisa.dma_copy(
            dst=flat.ap(
                pattern=[[width, tb], [1, width]],
                offset=0,
                vector_offset=wid,
                indirect_dim=0,
            ),
            src=wst[0:tb, 0:width],
            oob_mode=oob_mode.skip,
        )
    return cache


# ---------------------------------------------------------------------------
# NKI kernel dispatch wrapper (Z0 §D3 boundary: small index/mask tensors only)
# ---------------------------------------------------------------------------


def _kv_cache_write_nki(
    cache: Tensor, slot_ids: Tensor, rows: Tensor, valid_mask: Tensor | None
) -> Tensor:
    """Torch-side wrapper: the incumbent write cast, the ESFH001-safe skip-id
    fold, the HOP call — nothing else. The wrapper never reads, converts, or
    scatters into a cache-shaped tensor (Z0 §D3)."""
    # The incumbent's single write cast (bitwise obligation, Z0 §D2).
    src = rows.to(cache.dtype).contiguous()

    # s64 −1 → uint32-max 0xFFFFFFFF at ``.to(torch.uint32)`` — the ESFH001-safe
    # LD-75 form (mla_decode_tkg.py:353); NO clamp is introduced on this path.
    ids64 = slot_ids.to(torch.int64)
    valid = slot_ids > _PAD_SLOT_ID
    if valid_mask is not None:
        valid = valid & valid_mask
    skip = torch.full((1,), -1, device=slot_ids.device, dtype=torch.int64)
    wr_ids = (
        torch.where(valid, ids64, skip)
        .to(torch.uint32)
        .reshape(-1, 1)
        .contiguous()
    )

    wrapped = wrap_nki(_kv_cache_write_nki_kernel)
    res = wrapped[2](cache=cache, wr_ids=wr_ids, rows=src)
    out = res[0] if isinstance(res, (tuple, list)) else res

    # Eager-simulator copy-back (Z0 §D-copyback): on the CPU-sim dispatch the
    # HOP runs with ``operand_output_aliases={}`` and returns a NEW tensor
    # (nki_hop.py:202-211) — no write-through — so the in-place contract is
    # restored here. Env-gated exactly like the HOP's own CPU-sim dispatch;
    # UNREACHABLE on any traced path (the census venue refuses CPU_MODE; the
    # production trace sets neither env), so no copy node can enter a
    # production graph.
    if envs.VLLM_NEURON_CPU_MODE and os.environ.get("NKI_SIMULATOR") == "1":
        if out is not cache:
            cache.copy_(out)

    return cache


# ---------------------------------------------------------------------------
# PyTorch fallback implementation (CPU equivalence / simulator A/B ONLY)
# ---------------------------------------------------------------------------


def _pad_columns(rows: Tensor, width: int) -> Tensor:
    """Right-pad ``[T, n]`` to ``[T, width]`` with zeros."""
    extra = width - rows.shape[-1]
    if extra == 0:
        return rows
    if extra < 0:
        raise ValueError(f"{rows.shape[-1]} columns do not fit width {width}.")
    return torch.cat(
        (
            rows,
            torch.zeros(rows.shape[0], extra, dtype=rows.dtype, device=rows.device),
        ),
        dim=-1,
    )


def _torch_kv_cache_write(
    cache: Tensor,
    slot_ids: Tensor,
    rows: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> None:
    """The incumbent ``_masked_scatter_rows`` composition, casts bitwise.

    Scatter whole slots into a paged ``cache``, skipping padded slots.
    ``slot_ids`` carries ``-1`` for padded tokens and — on compressed layers —
    for every token whose compression group has not closed yet. Those slots
    must keep whatever they already hold.

    The skip is a *masked scatter*, never a boolean-mask index: padded
    destinations are redirected to slot 0 by ``clamp`` and the value written
    there is that slot's own current content, so the redirect is a no-op
    write. Destination and mask are plain arithmetic, so all shapes stay
    static.

    ``rows`` is written as ELEMENT columns from column 0 — the convention the
    reader side uses. fp8 is indexed DIRECTLY here — no dtype view (F-5).

    ``valid_mask`` (LD-79 keyword-only option): when given, a ``[T]`` bool
    tensor ANDed onto the slot-id pad guard — a masked-out row keeps the
    slot's existing content through the same no-op-write redirect. The
    ``slot_ids > _PAD_SLOT_ID`` guard is TOTAL (it always applies), because
    the clamp redirect is only a no-op under it. ``valid_mask=None`` — every
    call site the LD-79 rewire touched — is bit-for-bit the incumbent.
    """
    num_blocks, num_kv_heads, block_size, width = cache.shape
    src = _pad_columns(rows.to(cache.dtype), width)

    flat = cache.view(num_blocks * num_kv_heads * block_size, width)

    valid = slot_ids > _PAD_SLOT_ID
    if valid_mask is not None:
        valid = valid & valid_mask
    valid = valid.unsqueeze(-1)
    # min-only s64 clamp synthesizes iinfo(int64).max at FX→HLO; production warmup feeds slot_mapping int64 (neuron_model_runner.py:4272) — NCC_ESFH001 fix, ITER-22.
    dest = torch.clamp(slot_ids, min=0, max=_INT32_MAX).to(torch.long)
    existing = torch.index_select(flat, 0, dest)
    flat.index_put_((dest,), torch.where(valid, src, existing))


# ---------------------------------------------------------------------------
# Kernel eligibility check
# ---------------------------------------------------------------------------


def _can_use_kv_cache_write(
    cache: Tensor,
    slot_ids: Tensor,
    rows: Tensor,
    valid_mask: Tensor | None,
) -> bool:
    """Cheap, total, monotone envelope. False does NOT mean silent fallback
    here (see the dispatch): on a non-CPU tensor a False answer raises.

    Surface checks (family-20) + the LD-79 kernel-envelope caps (Z0 §D8) over
    the validated production cache classes — shape/dtype only, never tensor
    values, so the gate is trace-safe under FakeTensor:

    * ``cache.dtype is float8_e4m3fn`` — every LD-79 cache class is fp8_e4m3
      (call-site dtype census, prereg session: rows are fp8 codes / fp32
      pre-cast / cache-form fp8; caches all fp8).
    * ``1 <= width <= 2688`` — the widest validated class (compressor state).
    * ``T <= 65536`` — id-tile bound, covers every production token batch.
    * ``nb * kvh * bs <= 2**31 - 1`` — int32-safe flat slot ids (the C1/C2/C3
      clamp discipline; uint32 write ids address the flat cache).
    * ``rows.dtype in {fp8_e4m3fn, bfloat16, float32}`` — the cast set the
      simulator A/B grades (declaration + supplementary probe sweep).
    """
    if not can_run_kernel(cache):
        return False
    if cache.dim() != 4:
        return False
    if slot_ids.dim() != 1 or slot_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return False
    if rows.dim() != 2 or rows.shape[0] != slot_ids.shape[0]:
        return False
    if rows.shape[1] > cache.shape[-1]:
        return False
    if valid_mask is not None and (
        valid_mask.dtype != torch.bool
        or tuple(valid_mask.shape) != tuple(slot_ids.shape)
    ):
        return False
    # ---- LD-79 kernel-envelope caps (Z0 §D8) ----
    if cache.dtype is not torch.float8_e4m3fn:
        return False
    if not 1 <= cache.shape[-1] <= _MAX_WIDTH:
        return False
    if slot_ids.shape[0] > _MAX_T:
        return False
    if cache.shape[0] * cache.shape[1] * cache.shape[2] > _INT32_MAX:
        return False
    if rows.dtype not in (torch.float8_e4m3fn, torch.bfloat16, torch.float32):
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def kv_cache_write(
    cache: Tensor,
    slot_ids: Tensor,
    rows: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Write ``rows`` into ``cache`` slots ``slot_ids`` in place, mask-skipped.

    Args:
        cache: ``[num_blocks, num_kv_heads, block_size, width]`` paged cache
            (fp8_e4m3 in this port's production config — the kernel envelope;
            on CPU any dtype the write cast reaches falls back — the cast is
            the cache's own, ``rows.to(cache.dtype)``, exactly the
            incumbent's).
        slot_ids: ``[T]`` int32/int64 flat slot ids; ``-1`` = skip (pad, or an
            unclosed compression group). int32-safe per the C1/C2/C3 forms.
        rows: ``[T, n]`` row payload, ``n <= width``; written as ELEMENT
            columns from column 0, zero-padded to ``width``.
        valid_mask: optional ``[T]`` bool, ANDed onto the pad guard.

    Returns:
        The written ``cache`` (Z0 §D1: the harness grades RETURNED tensors;
        call sites discard the return). The write is in place: on the kernel
        path the cache is an in/out operand threaded by the FX aliasing pass
        (with the eager-simulator copy-back restoring the same contract off
        the traced path); on the CPU fallback path it is a plain
        ``index_put_``.
    """
    _validate_inputs(cache, slot_ids, rows, valid_mask)

    if _can_use_kv_cache_write(cache, slot_ids, rows, valid_mask):
        return _kv_cache_write_nki(cache, slot_ids, rows, valid_mask)

    if cache.device.type != "cpu":
        # Rule 4′: the torch fallback on a traced/device path would re-emit
        # the cache-page-sized scatter class LD-79 exists to remove (G-A′
        # gates page_sized_scatter == 0). No silent fallback — the STATIC_MX
        # MoE precedent (moe_cte.py:288-296).
        raise RuntimeError(
            "NF.kv_cache_write requires the NKI kernel on a non-CPU device "
            "(the torch fallback is CPU-equivalence/simulator-only and must "
            "never trace for production — LD-79/Rule 4′, G-A′ "
            "page_sized_scatter==0). Got device "
            f"{cache.device}; VLLM_NEURON_DISABLE_NKI_KERNELS="
            f"{int(envs.VLLM_NEURON_DISABLE_NKI_KERNELS)}."
        )

    _torch_kv_cache_write(cache, slot_ids, rows, valid_mask=valid_mask)
    return cache


def _validate_inputs(
    cache: Tensor,
    slot_ids: Tensor,
    rows: Tensor,
    valid_mask: Tensor | None,
) -> None:
    """Refuse what NO path can compute (block_fp8_linear.py precedent: the
    gate answers "can the KERNEL run this"; these answer "is this call
    coherent at all" — a false answer here is a caller bug)."""
    assert cache.dim() == 4, (
        f"kv_cache_write expects a 4-D paged cache, got {tuple(cache.shape)}"
    )
    assert slot_ids.dim() == 1, (
        f"kv_cache_write expects 1-D slot_ids, got {tuple(slot_ids.shape)}"
    )
    assert rows.dim() == 2, (
        f"kv_cache_write expects 2-D rows, got {tuple(rows.shape)}"
    )
    assert rows.shape[0] == slot_ids.shape[0], (
        f"row/slot count mismatch: rows {tuple(rows.shape)} vs slot_ids "
        f"{tuple(slot_ids.shape)}"
    )
    assert rows.shape[-1] <= cache.shape[-1], (
        f"{rows.shape[-1]} columns do not fit cache width {cache.shape[-1]}"
    )
    if valid_mask is not None:
        assert valid_mask.dtype == torch.bool and tuple(
            valid_mask.shape
        ) == tuple(slot_ids.shape), (
            f"valid_mask must be bool of shape {tuple(slot_ids.shape)}, got "
            f"{valid_mask.dtype} {tuple(valid_mask.shape)}"
        )
