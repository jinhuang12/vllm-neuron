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

KERNEL STATUS: the LD-79 NKI kernel body is authored by ``author_kernel_triads``
(triad: kernel + numerics declaration + simulator A/B + envelope caps). Until
that leg lands, the kernel branch raises ``NotImplementedError`` naming the
pending triad — nothing reaches a device trace before the triad exists
(family-20 hand-off, ``kernels_pending``). R-20c fold-back note (plan §20.5):
if the FX aliasing pass demands a consumer for a standalone in-place write, the
recorded fold-back is moving the write into the LD-76/LD-75 kernels per class —
a plan-level branch, never improvised at this surface.

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

import torch
from torch import Tensor

from vllm_neuron import envs
from vllm_neuron.nki.nki_hop import can_run_kernel

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


# ---------------------------------------------------------------------------
# NKI kernel dispatch (PENDING — author_kernel_triads lands the LD-79 triad)
# ---------------------------------------------------------------------------


def _kv_cache_write_nki(
    cache: Tensor, slot_ids: Tensor, rows: Tensor, valid_mask: Tensor | None
) -> None:
    """Kernel leg placeholder. The LD-79 triad (author_kernel_triads) replaces
    this with the ``wrap_nki``-registered in-place row-write kernel, cache as
    an in/out operand threaded by the FX aliasing pass. Raising here is the
    designed pending state: loud, never a silent scatter."""
    raise NotImplementedError(
        "NF.kv_cache_write: the LD-79 NKI kernel body is pending "
        "(author_kernel_triads leg). This branch must not be reachable "
        "before that triad lands."
    )


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

    The kernel-specific validated caps over the ~6 page-sized cache variants
    (LD-79 grep line: shape-parameterized) are the triad leg's to add; this
    surface checks only what every leg of the op requires to be coherent.
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
) -> None:
    """Write ``rows`` into ``cache`` slots ``slot_ids`` in place, mask-skipped.

    Args:
        cache: ``[num_blocks, num_kv_heads, block_size, width]`` paged cache
            (fp8_e4m3 in this port's production config; any dtype the write
            cast reaches is accepted — the cast is the cache's own,
            ``rows.to(cache.dtype)``, exactly the incumbent's).
        slot_ids: ``[T]`` int32/int64 flat slot ids; ``-1`` = skip (pad, or an
            unclosed compression group). int32-safe per the C1/C2/C3 forms.
        rows: ``[T, n]`` row payload, ``n <= width``; written as ELEMENT
            columns from column 0, zero-padded to ``width``.
        valid_mask: optional ``[T]`` bool, ANDed onto the pad guard.

    Returns:
        None. The write is in place: on the kernel path the cache is an
        in/out operand threaded by the FX aliasing pass; on the CPU fallback
        path it is a plain ``index_put_``.
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

    return _torch_kv_cache_write(cache, slot_ids, rows, valid_mask=valid_mask)


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
