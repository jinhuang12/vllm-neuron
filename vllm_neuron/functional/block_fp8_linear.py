# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 linear functional API (ladder decision LD-11).

``out[M, N] = dequant(quant(x)) @ dequant(weight_fp8).T`` for the DSv4-Flash
dense/attention slice: FP8 e4m3 weights quantized on a 128x128 block grid with
one fp32 (ue8m0) dequant multiplier per block, activations quantized DYNAMICALLY
per row per 128-element K group, fp32 accumulation, ``out_dtype`` cast last.

The whole triad lives in this file, in the order the specimen
``rmsnorm_quant.py`` uses: NKI kernel, torch fallback, eligibility gate, public
API. The plugin ships no other block-scale linear -- ``QuantizationType`` has no
BLOCK member and every other quantized op takes per-tensor scales -- and the
installed ``nkilib`` wheel has no 128x128 block-scale GEMM that runs on this
venue (the one implementation, ``experimental.qkv.qkv_cte_mla``, lowers to
``nisa.nc_matmul_mx``, a NeuronCore-v4 instruction). Hence a new kernel rather
than a wrapper.

THE ONE FACT THAT DOMINATES THIS FILE: THE 1-BYTE DTYPE IS A CARRIER

``weight_fp8`` is declared ``torch.float8_e4m3fn`` because that is the only
1-byte float torch has, but the BYTES inside it are LEGACY ``nl.float8_e4m3``
(bias 7, amax **240**, exponent field 15 reserved for inf/NaN) -- re-encoded at
load by LD-24, which halves every value and doubles the paired block scale. The
carrier is what makes ``torch_to_nki_dtype`` yield ``nl.float8_e4m3`` on trn2
(``nki/nki_dtype.py:50-56``), which is the only 1-byte float the TensorE
accepts on this generation. The same doctrine is written down twice already in
this family (``deepseek_v4/moe.py:202-211``,
``deepseek_v4/quantization.py:78-81``) and in the loader
(``deepseek_v4/weight_loaders.py:712-726``).

Consequences this op RELIES on, none of which it re-implements:

1. **No encoding fix-up here, and none is needed.** Every byte reaching this op
   has exponent field <= 14, so it is finite in the legacy grid. Authoring this
   kernel against the checkpoint's raw OCP-448 bytes was measured to produce
   inf/NaN weights for ~82.6% of 128x128 blocks; that is the iteration-1
   ``plan_defect`` and its remedy is LD-24, at the loader, not here.
2. **Weight dequantization introduces no rounding.** The block scale is an exact
   power of two (ue8m0, doubled at load -- doubling a power of two stays one).
3. **The activation ceiling is 240, not 448.** It is
   ``FP8_CLAMP_MAX``, the fork's own platform-resolved constant (240.0 on trn2,
   448.0 on trn3), never a fresh literal and never the reference
   implementation's 448: with a power-of-two scale against 448 a group's top
   code lands in (224, 448], so a field-15 byte -- inf/NaN on this venue --
   appears as soon as that maximum reaches 256, which is the COMMON case, not a
   tail case. Against 240 the top code lands in [120, 240] and no emitted byte
   can carry field 15 (assessment section 2.7, R-22; in-fork precedent
   ``rmsnorm_quant.py:32``, ``attention/mla_qkv.py:211-307``).

Numerics declaration, committed BEFORE any measurement:
``numerics/block_fp8_linear.declaration.json``.
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

import torch
from torch import Tensor

from vllm_neuron.nki.nki_dtype import torch_to_nki_dtype
from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Partition-dimension size, and this op's block edge. Matches
#: ``nl.tile_size.pmax``, kept as a host-side copy because
#: ``nl.tile_size.pmax`` has no correct value until trace time (the same
#: two-constant pattern as ``moe/topk_reduce.py:18`` and
#: ``weight_loaders_mxfp4.py:27``). The contract fixes ``block_size`` at
#: (128, 128) and ``act_group_size`` at 128, so one constant serves the
#: partition dim, the block edge and the activation group.
_PMAX = 128

#: The reference implementation's own activation amax floor
#: (``dsv4_ref/kernel.py:47-49, 79``). Skipping it does not make the result
#: cleaner -- it makes it differ from the trained numerics.
_AMAX_FLOOR = 1e-4

#: The frozen contract values this op accepts. A call that differs takes the
#: fallback rather than being silently reinterpreted.
_BLOCK_SIZE = (_PMAX, _PMAX)
_ACT_GROUP_SIZE = _PMAX

#: Envelope caps, in the ``MAX_VALIDATED_*`` spirit of
#: ``attention/o_proj.py:13-15``: the shapes this triad's declared cases and
#: the port's own production shapes cover, not a hardware limit.
#: Production shapes served by this op at TP=64 (assessment section 3.1):
#: fused_wqa_wkv [1536, 4096], wq_b, wo_a/wo_b, shared-expert gate_up/down,
#: indexer wq_b; M runs from 32 decode tokens to the 65536-token prefill
#: bucket.
_MAX_VALIDATED_M = 1 << 17
_MAX_VALIDATED_N = 1 << 14
_MAX_VALIDATED_K = 1 << 14

#: SBUF budget for the ONE resident buffer whose size is not bounded by a tile:
#: the transposed weight, ``[128, ceil(K/128) * N]`` 1-byte elements, i.e.
#: ``N * K / 128`` bytes per partition. Held resident so each weight tile is
#: transposed exactly once instead of once per M tile. 128 KiB of the 192 KiB
#: per-partition SBUF, leaving room for the activation tile (``K`` 2-byte
#: elements), the transposed quantized activations (``K/128 * M_tile`` bytes)
#: and the accumulators. The bound is MONOTONIC in N and K, so a static check
#: is sufficient here and the dry-run-config-factory pattern
#: (``topk.py:298-336``) is not needed.
_MAX_WEIGHT_SBUF_BYTES_PER_PARTITION = 128 * 1024


# ---------------------------------------------------------------------------
# NKI entry-point (runs on NeuronCore)
# ---------------------------------------------------------------------------


@nki.jit
def _block_fp8_linear_nki(
    x,
    weight_fp8,
    weight_scale,
    fp8_max=240.0,
    amax_floor=_AMAX_FLOOR,
    out_dtype=nl.bfloat16,
):
    """Blockwise FP8 GEMM: ``[M, K] x [N, K].T -> [M, N]``.

    Args:
        x: ``[M, K]`` bf16 activations, in HBM.
        weight_fp8: ``[N, K]`` 1-byte carrier holding LEGACY ``nl.float8_e4m3``
            values (see the module docstring), in HBM. NOT storage-transposed:
            the checkpoint's ``[out, in]`` orientation is what this op wants.
        weight_scale: ``[ceil(N/128), ceil(K/128)]`` fp32 dequant multipliers,
            one per 128x128 weight block, each an exact power of two.
        fp8_max: the venue's fp8 ceiling, passed in as
            ``FP8_CLAMP_MAX`` (240.0 on trn2). Used for BOTH the ue8m0 scale
            divisor and the clamp; the two must be the same number, or roughly
            half of every group's top values saturate.
        amax_floor: per-group amax floor.
        out_dtype: NKI dtype string for the returned tensor.

    Structure, and why it is this one:

    * ``nisa.nc_matmul`` contracts along the PARTITION axis of both operands
      (``dst = stationary.T @ moving``), so both operands need K on partitions.
      ``x`` is ``[M, K]`` and ``weight_fp8`` is ``[N, K]`` -- both K-minor --
      so exactly one transpose per operand is unavoidable. Both are done with
      ``nisa.dma_transpose``, measured exact on 1-byte data in this venue's CPU
      simulator; the TensorE transpose path is NOT used because on
      NeuronCore-v3 its FP8 mode writes 16-bit PSUM elements.
    * The moving free dimension may be up to 512 on this generation, but the
      N tile is deliberately ONE 128-wide weight block, because then the whole
      per-partial scale ``a_scale[m, kb] * w_scale[nb, kb]`` is a single
      per-partition scalar and needs one ``tensor_scalar``. Wider N tiles would
      mix several ``w_scale`` values along the free axis. Recorded as a perf
      trade, not a correctness one (R-8).
    * Each 128-contraction partial lands in fp32 PSUM, is evicted to SBUF,
      scaled by that partial's ``a_scale * w_scale``, and accumulated in fp32
      SBUF. PSUM accumulation across K blocks is NOT usable: every K block
      carries a different scale.
    * The weight is transposed once per (n-block, k-block) and held resident;
      the activation quantization runs exactly once per element. Neither is
      recomputed in an inner loop.
    * LNC-2 is launched WITHOUT sharding, following ``argsort_unstable.py:44``:
      both programs compute the same values and write the same bytes. Recorded
      as a known perf limitation for the perf report, never a correctness one.
    """
    M, K = x.shape
    N, K_w = weight_fp8.shape
    NB, KB = weight_scale.shape
    P = nl.tile_size.pmax

    kernel_assert(K == K_w, f"contraction mismatch: x {x.shape}, w {weight_fp8.shape}")
    kernel_assert(K % P == 0, f"K must be a multiple of {P}, got {K}")
    kernel_assert(KB == K // P, f"weight_scale K extent {KB} != K/{P} = {K // P}")
    kernel_assert(
        NB == (N + P - 1) // P,
        f"weight_scale N extent {NB} != ceil({N}/{P}) = {(N + P - 1) // P}",
    )

    out_hbm = nl.ndarray((M, N), dtype=out_dtype, buffer=nl.shared_hbm, name="out")

    # -- weights: one transposed [K_block, N_block] fp8 tile per block, once --
    # Free-axis layout is k-block major: tile (kb, nb) occupies
    # ``wT[:, kb * N + n0 : kb * N + n0 + n_sz]``.
    wT = nl.ndarray((P, KB * N), dtype=weight_fp8.dtype, buffer=nl.sbuf)
    for nb in range(NB):
        n0 = nb * P
        n_sz = min(P, N - n0)
        for kb in range(KB):
            k0 = kb * P
            nisa.dma_transpose(
                dst=wT[:, kb * N + n0 : kb * N + n0 + n_sz],
                src=weight_fp8[n0 : n0 + n_sz, k0 : k0 + P],
            )

    ones = nl.full((1, P), 1.0, dtype=nl.float32, buffer=nl.sbuf)

    for mt in range((M + P - 1) // P):
        m0 = mt * P
        m_sz = min(P, M - m0)

        x_sb = nl.ndarray((m_sz, K), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(x_sb, x[m0 : m0 + m_sz, :])

        # -- per row, per 128-element K group: amax -> ue8m0 scale ------------
        # ``nl.abs_max`` reduces to the element of largest MAGNITUDE but keeps
        # its SIGN (measured), so the magnitude is taken explicitly with
        # ``max(v, -v)`` rather than assumed.
        signed = nl.ndarray((m_sz, KB), dtype=nl.float32, buffer=nl.sbuf)
        for kb in range(KB):
            nisa.tensor_reduce(
                dst=signed[:, kb : kb + 1],
                op=nl.abs_max,
                data=x_sb[:, kb * P : (kb + 1) * P],
                axis=(1,),
            )
        negated = nl.ndarray((m_sz, KB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negated, data=signed, op0=nl.multiply, operand0=-1.0)
        amax = nl.ndarray((m_sz, KB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=amax, data1=signed, data2=negated, op=nl.maximum)

        # ``s = 2 ** ceil(log2(max(amax, floor) / fp8_max))`` in the IEEE-754
        # bit form of ``dsv4_ref/kernel.py:22-37``, NOT ``exp2(ceil(log2 v))``:
        # at an exact power of two fp32 ``log2`` can land either side of the
        # integer and flip the ceiling by a whole exponent. Since
        # ``ceil_log2 + 127 == exponent_field + (mantissa != 0)``, the rebias
        # cancels and no +-127 term is needed.
        v = nl.ndarray((m_sz, KB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=v,
            data=amax,
            op0=nl.maximum,
            operand0=amax_floor,
            op1=nl.multiply,
            operand1=1.0 / fp8_max,
        )
        v_bits = v.view(nl.int32)
        exponent = nl.ndarray((m_sz, KB), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=exponent,
            data=v_bits,
            op0=nl.right_shift,
            operand0=23,
            op1=nl.bitwise_and,
            operand1=0xFF,
        )
        mantissa = nl.ndarray((m_sz, KB), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=mantissa, data=v_bits, op0=nl.bitwise_and, operand0=0x7FFFFF
        )
        inexact = nl.ndarray((m_sz, KB), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=inexact, data=mantissa, op0=nl.greater, operand0=0)
        ceil_field = nl.ndarray((m_sz, KB), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ceil_field, data1=exponent, data2=inexact, op=nl.add)
        scale_bits = nl.ndarray((m_sz, KB), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=scale_bits, data=ceil_field, op0=nl.left_shift, operand0=23
        )
        a_scale = nl.ndarray((m_sz, KB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=a_scale, src=scale_bits.view(nl.float32))

        # -- quantize to fp8 and transpose, once per (M tile, K block) --------
        xqT = nl.ndarray((P, KB * m_sz), dtype=weight_fp8.dtype, buffer=nl.sbuf)
        for kb in range(KB):
            scaled = nl.ndarray((m_sz, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=scaled,
                data=x_sb[:, kb * P : (kb + 1) * P],
                op0=nl.divide,
                operand0=a_scale[:, kb : kb + 1],
            )
            clamped = nl.ndarray((m_sz, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=clamped,
                data=scaled,
                op0=nl.maximum,
                operand0=-fp8_max,
                op1=nl.minimum,
                operand1=fp8_max,
            )
            codes = nl.ndarray((m_sz, P), dtype=weight_fp8.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=codes, src=clamped)
            nisa.dma_transpose(dst=xqT[:, kb * m_sz : (kb + 1) * m_sz], src=codes)

        for nb in range(NB):
            n0 = nb * P
            n_sz = min(P, N - n0)

            # This n-block's KB weight scales, broadcast across the M
            # partitions (a matmul against a ones row is the partition-axis
            # broadcast; ``tensor_scalar`` broadcasts along the free axis, which
            # is the wrong axis here), then folded with the per-row activation
            # scales into ONE per-partition scalar per K block.
            ws_row = nl.ndarray((1, KB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(ws_row, weight_scale[nb : nb + 1, :])
            ws_psum = nl.ndarray((m_sz, KB), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ws_psum, stationary=ones[:, :m_sz], moving=ws_row)
            ws_bcast = nl.ndarray((m_sz, KB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ws_bcast, src=ws_psum)
            combined = nl.ndarray((m_sz, KB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=combined, data1=a_scale, data2=ws_bcast, op=nl.multiply
            )

            acc = nl.zeros((m_sz, n_sz), dtype=nl.float32, buffer=nl.sbuf)
            for kb in range(KB):
                partial = nl.ndarray((m_sz, n_sz), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=partial,
                    stationary=xqT[:, kb * m_sz : (kb + 1) * m_sz],
                    moving=wT[:, kb * N + n0 : kb * N + n0 + n_sz],
                )
                evicted = nl.ndarray((m_sz, n_sz), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=evicted, src=partial)
                nisa.tensor_scalar(
                    dst=evicted,
                    data=evicted,
                    op0=nl.multiply,
                    operand0=combined[:, kb : kb + 1],
                )
                nisa.tensor_tensor(dst=acc, data1=acc, data2=evicted, op=nl.add)

            out_tile = nl.ndarray((m_sz, n_sz), dtype=out_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_tile, src=acc)
            nisa.dma_copy(out_hbm[m0 : m0 + m_sz, n0 : n0 + n_sz], out_tile)

    return out_hbm


# ---------------------------------------------------------------------------
# PyTorch fallback implementation
# ---------------------------------------------------------------------------


def _pow2_ceil_scale(amax: Tensor, fp8_max: float) -> Tensor:
    """``s = 2 ** ceil(log2(amax / fp8_max))`` via the reference's bit trick.

    A local copy of ``attention/mla_qkv.py:211-243``'s ``_round_scale_pow2``
    rather than an import: that symbol is private to another functional module,
    and this file must not depend on the import order of the attention
    subpackage. The two must stay identical -- they implement the same clause of
    the same checkpoint recipe (``dsv4_ref/kernel.py:22-37``).

    ``(mantissa != 0)`` is used as ARITHMETIC, never as an index, so the graph
    stays static-shape.
    """
    v = (amax.to(torch.float32) * (1.0 / fp8_max)).contiguous()
    bits = v.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    ceil_log2 = exponent - 127 + (mantissa != 0).to(torch.int32)
    return ((ceil_log2 + 127) << 23).view(torch.float32)


def _torch_block_fp8_linear(
    x: Tensor,
    weight_fp8: Tensor,
    weight_scale: Tensor,
    *,
    block_size: tuple[int, int] = _BLOCK_SIZE,
    act_group_size: int = _ACT_GROUP_SIZE,
    accum_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Traceable torch fallback -- the numerical reference the kernel matches.

    Authored FIRST, and it is what the compiler traces on device whenever the
    gate is false, so it stays inside the traceable static-shape subset:
    Python-level loops over statically known counts, no ``.item()``, no
    ``.tolist()``, no ``nonzero()``, no boolean-mask indexing, no
    data-dependent shape. Unlike ``attention/o_proj.py:86-97``'s fallback this
    one may NOT refuse quantization: six of the seven call sites reach this op
    precisely because ``weight_scale is not None``.

    Written in the DEQUANTIZE-THEN-MATMUL form. The declared plain-torch
    reference is written in the equivalent SCALED-PARTIAL form, so the c13
    leg-1 comparison grades two independent spellings of the contract instead
    of one spelling against a paraphrase of itself. They agree exactly in real
    arithmetic and differ only in fp32 rounding -- of the summation always, and
    of the scale multiply only when a scale is not a power of two. On the
    production load path it always is (ue8m0, doubled by LD-24), so that second
    term is exactly zero there; the harness's synthetic ``randint`` scales
    include non-powers of two, which is what the declared 2e-05 fp32 clause
    bounds.
    """
    block_n, block_k = int(block_size[0]), int(block_size[1])
    group = int(act_group_size)
    m, k = x.shape
    n = weight_fp8.shape[0]
    n_blocks = (n + block_n - 1) // block_n
    k_blocks = k // block_k

    # -- activations: dynamic per-row, per-group fp8 quant/dequant -----------
    grouped = x.to(torch.float32).reshape(m, k // group, group)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=_AMAX_FLOOR)
    a_scale = _pow2_ceil_scale(amax, FP8_CLAMP_MAX)
    a_dequant = (
        (grouped / a_scale)
        .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        * a_scale
    ).reshape(m, k)

    # -- weights: block scale expanded to elements. Exact: every scale is a
    # -- power of two, so this multiply introduces no rounding.
    w_scale = (
        weight_scale.to(torch.float32)
        .repeat_interleave(block_n, dim=0)
        .repeat_interleave(block_k, dim=1)[:n, :k]
    )
    w_dequant = weight_fp8.to(torch.float32) * w_scale

    # -- fp32 accumulation, one 128-contraction partial at a time, mirroring
    # -- the kernel's blocking. Static loop count.
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    for kb in range(k_blocks):
        k0 = kb * block_k
        k1 = k0 + block_k
        out = out + a_dequant[:, k0:k1] @ w_dequant[:, k0:k1].t()

    del n_blocks  # shape check only; the loop is over K blocks
    out = out.to(accum_dtype)
    if bias is not None:
        out = out + bias.to(accum_dtype)
    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# Kernel eligibility check
# ---------------------------------------------------------------------------


def _can_use_block_fp8_linear(
    x: Tensor,
    weight_fp8: Tensor,
    weight_scale: Tensor,
    block_size: tuple[int, int],
    act_group_size: int,
    accum_dtype: torch.dtype,
    out_dtype: torch.dtype,
    bias: Optional[Tensor],
) -> bool:
    """Cheap, total, monotonic envelope check. False means take the fallback.

    Every bound below is monotonic in a shape or exact in a dtype, so a static
    check is sufficient and the dry-run-config-factory pattern
    (``topk.py:298-336``) is deliberately not used -- that pattern exists for
    non-monotonic envelopes and carries a ``python -O`` caveat this op does not
    need to inherit.
    """
    if not can_run_kernel(x):
        return False

    # Contract knobs. A caller passing something else is not wrong -- it just
    # does not get this kernel.
    if tuple(int(v) for v in block_size) != _BLOCK_SIZE:
        return False
    if int(act_group_size) != _ACT_GROUP_SIZE:
        return False
    if accum_dtype != torch.float32:
        return False
    if bias is not None:
        return False

    # Dtypes. bf16 activations only: the kernel's fp8 quantization path is
    # authored for the 2-byte activation the family passes everywhere.
    if x.dtype != torch.bfloat16:
        return False
    if weight_fp8.dtype != torch.float8_e4m3fn:
        return False
    if weight_scale.dtype != torch.float32:
        return False
    if out_dtype not in (torch.bfloat16, torch.float32):
        return False

    # Ranks and shapes.
    if x.dim() != 2 or weight_fp8.dim() != 2 or weight_scale.dim() != 2:
        return False
    m, k = x.shape
    n, k_w = weight_fp8.shape
    if k != k_w or m < 1 or n < 1:
        return False
    # K_local % 128 == 0 is a hard invariant of this port's sharding, not a
    # convenience: it makes every contraction block exactly one partition
    # load and every activation group exactly one K block.
    if k % _PMAX != 0:
        return False
    if tuple(weight_scale.shape) != ((n + _PMAX - 1) // _PMAX, k // _PMAX):
        return False

    # Validated envelope and the one resident SBUF buffer.
    if m > _MAX_VALIDATED_M or n > _MAX_VALIDATED_N or k > _MAX_VALIDATED_K:
        return False
    if (n * (k // _PMAX)) > _MAX_WEIGHT_SBUF_BYTES_PER_PARTITION:
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def block_fp8_linear(
    x: Tensor,
    weight_fp8: Tensor,
    weight_scale: Tensor,
    *,
    block_size: tuple[int, int] = _BLOCK_SIZE,
    act_group_size: int = _ACT_GROUP_SIZE,
    accum_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Block-FP8 linear: ``[M, K] @ [N_local, K_local].T -> [M, N_local]``.

    Args:
        x: ``[M, K]`` activations, bf16 on the kernel path.
        weight_fp8: ``[N_local, K_local]`` 1-byte carrier
            (``torch.float8_e4m3fn`` dtype, LEGACY ``nl.float8_e4m3`` value
            semantics, amax 240, re-encoded at load by LD-24). NOT
            storage-transposed.
        weight_scale: ``[ceil(N_local/128), ceil(K_local/128)]`` fp32 dequant
            multipliers, already DOUBLED at load by LD-24 and still exact
            powers of two.
        block_size: weight block grid. Fixed at (128, 128) by the checkpoint.
        act_group_size: activation quantization group along K. Fixed at 128.
        accum_dtype: accumulation dtype. fp32.
        out_dtype: returned dtype.
        bias: optional additive bias, applied after accumulation.

    Returns:
        ``[M, N_local]`` tensor of ``out_dtype``.
    """
    _validate_inputs(x, weight_fp8, weight_scale, block_size, act_group_size)

    if not _can_use_block_fp8_linear(
        x,
        weight_fp8,
        weight_scale,
        block_size,
        act_group_size,
        accum_dtype,
        out_dtype,
        bias,
    ):
        return _torch_block_fp8_linear(
            x,
            weight_fp8,
            weight_scale,
            block_size=block_size,
            act_group_size=act_group_size,
            accum_dtype=accum_dtype,
            out_dtype=out_dtype,
            bias=bias,
        )

    # LNC-2 grid, matching every other functional op in this package.
    wrapped = wrap_nki(_block_fp8_linear_nki)
    return wrapped[2](
        x=x,
        weight_fp8=weight_fp8,
        weight_scale=weight_scale,
        fp8_max=FP8_CLAMP_MAX,
        amax_floor=_AMAX_FLOOR,
        out_dtype=torch_to_nki_dtype(out_dtype),
    )


def _validate_inputs(
    x: Tensor,
    weight_fp8: Tensor,
    weight_scale: Tensor,
    block_size: tuple[int, int],
    act_group_size: int,
) -> None:
    """Refuse what NO path can compute, so a caller error is not a wrong number.

    Kept separate from the gate: the gate answers "can the KERNEL run this",
    and its false answer is a legal fallback. These answer "is this call
    coherent at all", and their false answer is a bug in the caller.
    """
    assert x.dim() == 2, f"block_fp8_linear expects 2-D x, got {tuple(x.shape)}"
    assert weight_fp8.dim() == 2, (
        f"block_fp8_linear expects 2-D weight, got {tuple(weight_fp8.shape)}"
    )
    assert weight_scale.dim() == 2, (
        f"block_fp8_linear expects a 2-D block-scale grid, got "
        f"{tuple(weight_scale.shape)}"
    )
    assert x.shape[-1] == weight_fp8.shape[-1], (
        f"contraction mismatch: x is {tuple(x.shape)}, weight is "
        f"{tuple(weight_fp8.shape)}"
    )
    block_n, block_k = (int(v) for v in block_size)
    group = int(act_group_size)
    assert group == block_k, (
        f"act_group_size ({group}) must equal the weight block's K extent "
        f"({block_k}): a 128x128 output-block partial has ONE activation scale "
        f"per row only when they agree"
    )
    k = x.shape[-1]
    assert k % group == 0, (
        f"K={k} must be a whole number of {group}-element activation groups "
        f"(K_local % 128 == 0 is a hard invariant of this port's sharding)"
    )
    n = weight_fp8.shape[0]
    expected = ((n + block_n - 1) // block_n, (k + block_k - 1) // block_k)
    assert tuple(weight_scale.shape) == expected, (
        f"weight_scale must be [ceil(N/{block_n}), ceil(K/{block_k})] = "
        f"{expected}, got {tuple(weight_scale.shape)}. A mis-sharded scale grid "
        f"is not a crash on device -- it is wrong numbers after a "
        f"multi-thousand-second compile."
    )
