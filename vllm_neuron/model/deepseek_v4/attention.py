# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4 attention: MLA block, per-layer KV compressor, DSA indexer
=====================================================================

<-- MODEL-SPECIFIC: this whole file is DeepSeek-V4-Flash specific.

Primary source of truth for every formula is DeepSeek's own reference
implementation shipped with the pinned checkpoint, cited as
``dsv4_ref/model.py:<line>`` / ``dsv4_ref/kernel.py:<line>``. Where the
derived port specs disagreed with it the reference won on math; the port's
interface contract still fixes class names, parameter attribute names, the
``[out, in]`` weight orientation, the ``NF.*`` positional signatures and the
KV layer-name convention.

Three classes, in dependency order:

``DeepseekV4KVCompressor``
    Pools ``compress_ratio`` consecutive raw token latents into ONE cache
    slot with a learned softmax gate (``dsv4_ref/model.py:285-383``). Built
    on every layer whose ``compress_ratio > 1``; the DSA indexer nests its
    own narrower, Hadamard-rotated copy.

``DeepseekV4Indexer``
    The DSA "lightning indexer" (``dsv4_ref/model.py:386-439``). Scores
    every compressed slot with a cheap 64-head / 128-dim query and returns
    the top ``index_topk`` slot indices. Built only where
    ``config.has_indexer(layer_idx)``.

``DeepseekV4Attention``
    The MLA block (``dsv4_ref/model.py:442-548``), and the single source of
    truth for this layer's KV declarations (:meth:`kv_layer_specs`).

Everything here stays inside the traceable static-shape subset: no
``.item()``, no ``.tolist()``, no ``nonzero()``, no boolean-mask indexing,
no data-dependent shapes, no Python ``if`` on a tensor value. Python ``if``
on *config* (layer class, indexer presence) and on *static shapes* IS used
deliberately: it decides the graph once, at trace time.

>>> PARALLELISM: at TP=64 each core owns exactly ONE of the 64 query heads.
The hidden stream is replicated (never TP-sharded), so only the head axis
splits. The grouped o-projection reduces over an 8-core subgroup
(``oproj_group``) and then over the whole TP group, which is why the group
handle is a constructor argument rather than something this module looks
up. <<<
"""

import math

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import LayerSpec

from .config import (
    LAYER_CLASS_DENSE_C128,
    LAYER_CLASS_SPARSE_C4,
    LAYER_CLASS_SWA_ONLY,
    DeepseekV4Config,
)
from .weight_loaders import attach_attention_loaders

__all__ = [
    "DeepseekV4Attention",
    "DeepseekV4Indexer",
    "DeepseekV4KVCompressor",
]

# ---------------------------------------------------------------------------
# Numeric constants, all taken from the reference implementation
# ---------------------------------------------------------------------------

#: Cache storage dtype this layout is designed around. The runner allocates
#: from ``--kv-cache-dtype``, NOT from ``LayerSpec.dtype``
#: (``pricing-and-design.md`` §2), so this constant documents intent and is
#: also what the write side casts to.
_FP8_DTYPE: torch.dtype = torch.float8_e4m3fn

#: FP8 E4M3 absolute maximum (``dsv4_ref/kernel.py:47-48``).
_FP8_MAX: float = 448.0

#: FP4 E2M1 absolute maximum (``dsv4_ref/kernel.py:134``).
_FP4_MAX: float = 6.0

#: The latent NoPE dims are quantized in groups of 64
#: (``dsv4_ref/model.py:512``, ``:378``: ``act_quant(kv[..., :-rd], 64,
#: ...)``), giving ``448 / 64 = 7`` scales per slot.
_KV_QUANT_GROUP: int = 64
_KV_NUM_SCALES: int = 7

#: The indexer's q and compressed k use FP4 group-32 quantization
#: (``dsv4_ref/model.py:18`` ``fp4_block_size = 32``, used at ``:376`` and
#: ``:422``), so its transport scales are group-32: ``128 / 32 = 4``.
_FP4_QUANT_GROUP: int = 32
_INDEX_NUM_SCALES: int = 4

#: The 8 representable FP4 E2M1 magnitudes and the nearest-value decision
#: boundaries between them (``dsv4_ref/convert.py`` FP4 table).
_FP4_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_BOUNDS: tuple[float, ...] = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)

#: ``slot_mapping`` padding sentinel produced by the runner. Padded slots and
#: (on compressed layers) not-yet-closed groups must not be written.
_PAD_SLOT_ID: int = -1

#: ``-1`` = "no such slot", the reference's sparse-attention skip sentinel
#: (``dsv4_ref/model.py:436``, ``dsv4_ref/kernel.py:323-327``).
_PAD_INDEX: int = -1

#: Compressed-latent pair width: the 448 NoPE columns split evenly across
#: ``k_cache``/``v_cache``, which the runner allocates with identical shape.
_LATENT_PAIR_HEAD_SIZE: int = 224

#: ``self_attn.rope`` pair width: ``k_cache`` columns ``[0:64]`` are the 64
#: RoPE columns, ``v_cache`` columns ``[0:7]`` are the group-64 dequant
#: scales. 128 (rather than 64) is kept from contract §4 so the recorded
#: 704 B compressed slot does not move.
_ROPE_PAIR_HEAD_SIZE: int = 128

#: ``self_attn.swa`` pair width — RESOLVED at 512, correcting contract §4's
#: provisional 320.
#:
#: ``dsv4_ref/model.py:479-480`` allocates ONE buffer per layer,
#: ``kv_cache[max_batch, window_size + max_seq_len // compress_ratio,
#: head_dim]`` with ``head_dim = 512``; ``:497`` carves the compressed pool
#: out of its tail and ``:533``/``:538`` pass that single tensor as the only
#: KV argument. So a window slot and a compressed slot hold exactly the same
#: thing — the full 512-wide latent, 448 NoPE + 64 RoPE — and MLA has no
#: separate V (absorbed form: K and V are that one tensor).
#:
#: The port's ops read a cache as ELEMENT columns in the cache dtype
#: (``mla_sparse_attention._flat_rows`` / ``_gather_latent``), with the
#: pieces of one logical latent concatenated in column order and the
#: group-64 dequant scales living in a SEPARATE tensor
#: (``compressed_scale_cache`` / ``swa_scale_cache``). The SWA leg has
#: exactly one layer name, i.e. exactly two tensors, so the only split that
#: keeps the faithful group-64 numerics is: ``k_cache`` = the whole 512-wide
#: latent (one piece, ``swa_widths=(512,)``), ``v_cache`` = the scale
#: companion (7 of 512 columns used). Hence 512.
_SWA_PAIR_HEAD_SIZE: int = 512

#: ``self_attn.indexer`` pair width: ``k_cache`` holds the 128 index-K
#: columns, ``v_cache`` columns ``[0:4]`` hold that slot's group-32 scales.
_INDEXER_PAIR_HEAD_SIZE: int = 128


# ---------------------------------------------------------------------------
# Traceable primitives
#
# Free functions rather than methods because the compressor, the indexer and
# the MLA block all need identical math, and a shared implementation is the
# only way the three stay bit-identical to one another.
# ---------------------------------------------------------------------------


def _num_blocks(extent: int, block: int = 128) -> int:
    """``ceil(extent / block)`` — the block-FP8 scale-grid extent."""
    return (extent + block - 1) // block


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32 with a single cast at the end (``dsv4_ref/model.py:197-202``).

    The fp32 promotion is load-bearing, not defensive: the reference computes
    the variance, the rsqrt and the weight multiply all in fp32.
    """
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    return weight.to(torch.float32) * x32 * torch.rsqrt(variance + eps)


def _yarn_inv_freq(
    rope_dim: int,
    base: float,
    original_seq_len: int,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    device: torch.device,
) -> torch.Tensor:
    """Return ``[rope_dim // 2]`` inverse frequencies (``dsv4_ref/model.py:205-235``).

    <-- MODEL-SPECIFIC: DeepSeek YaRN, NTK-by-parts. Transcribed from the
    reference including ``if low == high: high += 0.001``
    (``dsv4_ref/model.py:220-221``) — a ``max(high - low, 1)`` guard would
    change the ramp whenever the correction range collapses. YaRN is active
    only when ``original_seq_len > 0`` (``:227``), and the reference passes
    ``0`` for layers WITHOUT KV compression (``:484-485``), which disables it
    on the SWA-only layers.

    Only the compressor needs this locally (it rotates at a DIFFERENT
    position than the layer's shared table); the layer's own cos/sin come
    from the backbone's ``DeepseekV4RotaryEmbedding``. Computed per call
    rather than buffered because the runner forces every buffer to meta
    (``neuron_model_runner.py:1231-1232``); it is 32 elements.
    """
    freqs = 1.0 / (
        base
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    if original_seq_len <= 0 or factor <= 1.0:
        return freqs

    def correction_dim(num_rotations: float) -> float:
        return (
            rope_dim
            * math.log(original_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), rope_dim - 1)
    high_f = float(high) + (0.001 if high == low else 0.0)

    ramp = torch.clamp(
        (torch.arange(rope_dim // 2, dtype=torch.float32, device=device) - float(low))
        / (high_f - float(low)),
        0.0,
        1.0,
    )
    smooth = 1.0 - ramp
    return freqs / factor * (1.0 - smooth) + freqs * smooth


def _cos_sin(
    positions: torch.Tensor,
    rope_dim: int,
    base: float,
    original_seq_len: int,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token GPT-J cos/sin tables of shape ``[T, rope_dim // 2]``.

    Same shape and convention as the backbone's
    ``DeepseekV4RotaryEmbedding.forward``, so the two are interchangeable.
    """
    inv_freq = _yarn_inv_freq(
        rope_dim, base, original_seq_len, factor, beta_fast, beta_slow, positions.device
    )
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    return angles.cos(), angles.sin()


def _gptj_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    """GPT-J (interleaved) RoPE on the LAST ``rope_dim`` columns of ``x``.

    <-- MODEL-SPECIFIC: ``dsv4_ref/model.py:238-250`` pairs dims via
    ``view_as_complex(x.unflatten(-1, (-1, 2)))``, i.e. pairs are
    ``(2i, 2i+1)`` inside the rope tail, NOT ``(i, i + rope_dim/2)``. Getting
    this wrong is silent: the shapes match either way and only the numerics
    move.

    ``inverse=True`` is the conjugate rotation (``:242-243``), which maps the
    attention output back out of RoPE space before the o-projection
    (``:539``).

    Args:
        x: ``[T, ..., D]`` with ``D >= rope_dim``; leading dim is tokens.
        cos: ``[T, rope_dim // 2]``.
        sin: ``[T, rope_dim // 2]``.
        rope_dim: Width of the rotated tail.
        inverse: Apply the conjugate rotation.

    Returns:
        A tensor shaped like ``x``, fp32.
    """
    nope = x.shape[-1] - rope_dim
    x32 = x.to(torch.float32)
    head = x32[..., :nope]
    tail = x32[..., nope:]

    pairs = tail.reshape(*tail.shape[:-1], rope_dim // 2, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]

    # Broadcast [T, rope_dim//2] up to even's rank with one singleton per
    # intermediate axis — no data-dependent shape.
    extra_dims = even.dim() - 2
    cos_b = cos.to(torch.float32).reshape(cos.shape[0], *([1] * extra_dims), -1)
    sin_b = sin.to(torch.float32).reshape(sin.shape[0], *([1] * extra_dims), -1)

    if inverse:
        new_even = even * cos_b + odd * sin_b
        new_odd = odd * cos_b - even * sin_b
    else:
        new_even = even * cos_b - odd * sin_b
        new_odd = odd * cos_b + even * sin_b

    return torch.cat(
        (head, torch.stack((new_even, new_odd), dim=-1).flatten(-2)), dim=-1
    )


def _quant_fp8_ue8m0(
    x: torch.Tensor, group: int = _KV_QUANT_GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    """UE8M0 block-FP8 quantization of ``[T, D]`` in groups of ``group``.

    ``dsv4_ref/kernel.py:77-98`` with ``scale_fmt="ue8m0"``: per-group absmax
    clamped at ``1e-4``, scale ``absmax / 448`` rounded UP to a power of two,
    values divided by the scale and clamped to ``[-448, 448]``.

    Returns:
        ``(codes [T, D] fp8, dequant_scales [T, D // group] fp32)`` with
        ``x ~= codes * scales``, so the read side is a plain multiply — the
        convention ``mla_sparse_attention.dequant_group_scales`` expects.
    """
    tokens, width = x.shape
    groups = width // group
    xb = x.to(torch.float32).view(tokens, groups, group)
    absmax = xb.abs().amax(dim=-1).clamp_min(1e-4)
    exponents = torch.ceil(torch.log2(absmax / _FP8_MAX))
    codes = torch.clamp(
        xb * torch.exp2(-exponents).unsqueeze(-1), -_FP8_MAX, _FP8_MAX
    ).to(_FP8_DTYPE)
    return codes.view(tokens, width), torch.exp2(exponents)


def _dequant_fp8(codes: torch.Tensor, scales: torch.Tensor, group: int) -> torch.Tensor:
    """Undo :func:`_quant_fp8_ue8m0`, returning fp32.

    The reference's ``act_quant(..., inplace=True)`` leaves the DEQUANTIZED
    value in place and feeds that to both the cache write and the current
    step's attention (``dsv4_ref/model.py:512`` runs before ``:533``), so the
    round trip is part of the numerics rather than a storage detail.
    """
    tokens, width = codes.shape
    groups = scales.shape[-1]
    return (
        codes.to(torch.float32).view(tokens, groups, group) * scales.unsqueeze(-1)
    ).view(tokens, width)


def _hadamard(size: int, device: torch.device) -> torch.Tensor:
    """Normalized Sylvester Hadamard matrix ``[size, size]``, fp32.

    ``dsv4_ref/model.py:253-257`` applies ``hadamard_transform(x, scale=D **
    -0.5)`` before the indexer's FP4 quantization, so a 3-bit grid loses less
    of each value. The Sylvester matrix is symmetric, so ``x @ H`` and
    ``H @ x`` agree and the transform's orientation cannot be got wrong.
    ``size`` is a power of two (128 here); built per call because a buffer
    would be forced to meta.
    """
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < size:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h / math.sqrt(size)


def _quant_fp4_simulate(x: torch.Tensor, group: int = _FP4_QUANT_GROUP) -> torch.Tensor:
    """FP4 E2M1 quantize/dequantize simulation in fp32.

    ``dsv4_ref/kernel.py:128-150`` plus ``dsv4_ref/model.py:376``/``:422``:
    per-group absmax, power-of-two scale against ``fp4_max = 6.0``, round to
    the nearest of the 8 representable E2M1 magnitudes, scale back. The model
    was QAT-trained against this grid, so the port has to land on it before
    transporting the value through the (finer) FP8 cache. Snapping uses
    ``bucketize`` against static boundaries — no data-dependent shape, no
    boolean indexing.
    """
    tokens, width = x.shape
    groups = width // group
    xb = x.to(torch.float32).view(tokens, groups, group)
    absmax = xb.abs().amax(dim=-1).clamp_min(1e-8)
    scales = torch.exp2(torch.ceil(torch.log2(absmax / _FP4_MAX))).unsqueeze(-1)

    scaled = torch.clamp(xb / scales, -_FP4_MAX, _FP4_MAX)
    bounds = torch.tensor(_FP4_BOUNDS, dtype=torch.float32, device=x.device)
    levels = torch.tensor(_FP4_LEVELS, dtype=torch.float32, device=x.device)
    snapped = torch.index_select(
        levels, 0, torch.bucketize(scaled.abs(), bounds).reshape(-1)
    ).view_as(scaled)
    return (torch.sign(scaled) * snapped * scales).view(tokens, width)


def _pad_columns(rows: torch.Tensor, width: int) -> torch.Tensor:
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


def _masked_scatter_rows(
    cache: torch.Tensor, slot_mapping: torch.Tensor, rows: torch.Tensor
) -> None:
    """Scatter whole slots into a paged ``cache``, skipping padded slots.

    ``slot_mapping`` carries ``-1`` for padded tokens and — on compressed
    layers — for every token whose compression group has not closed yet.
    Those slots must keep whatever they already hold.

    The skip is a *masked scatter*, never a boolean-mask index: padded
    destinations are redirected to slot 0 by ``clamp`` and the value written
    there is that slot's own current content, so the redirect is a no-op
    write. Destination and mask are plain arithmetic, so all shapes stay
    static.

    ``rows`` is written as ELEMENT columns from column 0 — the convention the
    reader side uses (``mla_sparse_attention._gather_latent`` takes
    ``part[..., :width]`` after casting the cache). The transfer goes through
    an int8 view because fp8 tensors cannot be fancy-indexed
    (``attention_decode.py:610-620``); at one byte per element that view is
    column-identical, so it is a pure dtype relabel.
    """
    num_blocks, num_kv_heads, block_size, width = cache.shape
    src = _pad_columns(rows.to(cache.dtype), width)

    flat = cache.view(num_blocks * num_kv_heads * block_size, width)
    if cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        flat = flat.view(torch.int8)
        src = src.view(torch.int8)

    valid = (slot_mapping > _PAD_SLOT_ID).unsqueeze(-1)
    dest = torch.clamp(slot_mapping, min=0).to(torch.long)
    existing = torch.index_select(flat, 0, dest)
    flat.index_put_((dest,), torch.where(valid, src, existing))


def _paged_slot_ids(
    block_table: torch.Tensor, span: int, block_size: int
) -> torch.Tensor:
    """Flat slot ids for the first ``span`` slots of every sequence.

    ``row = block_table[seq, local // block_size] * block_size + local %
    block_size`` — the port-wide paged addressing of ``dataflow-shapes.md``
    §D, which is also how the functional ops translate an index. Returns
    ``[B, span]`` int64.
    """
    device = block_table.device
    local = torch.arange(span, device=device, dtype=torch.int64)
    blocks = torch.index_select(block_table.to(torch.int64), 1, local // block_size)
    return blocks * block_size + (local % block_size)


def _gather_scale_columns(
    scale_cache: torch.Tensor, slot_ids: torch.Tensor, num_scales: int, group: int
) -> torch.Tensor:
    """Read per-group dequant scales and expand them to per-column factors.

    ``[B, S]`` slot ids -> ``[B, S, num_scales * group]`` fp32, so the caller
    can multiply them straight onto gathered keys — the ``key_scale``
    convention of ``NF.sparse_indexer_topk``, which does no group expansion
    of its own.
    """
    num_blocks, _, block_size, width = scale_cache.shape
    flat = scale_cache.to(torch.float32).view(num_blocks * block_size, width)
    rows = torch.clamp(slot_ids, 0, flat.shape[0] - 1).reshape(-1)
    grouped = torch.index_select(flat, 0, rows)[:, :num_scales]
    expanded = grouped.unsqueeze(-1).expand(-1, num_scales, group)
    return expanded.reshape(slot_ids.shape[0], slot_ids.shape[1], num_scales * group)


# ---------------------------------------------------------------------------
# Per-layer KV compressor
# ---------------------------------------------------------------------------


class DeepseekV4KVCompressor(nn.Module):
    """Pool ``compress_ratio`` raw token latents into one cache slot.

    <-- MODEL-SPECIFIC: DeepSeek-V4's "4x / 128x compression"
    (``dsv4_ref/model.py:285-383``). Per token it computes a value row
    ``kv = wkv(x)`` and a gate row ``score = wgate(x) + ape[position %
    compress_ratio]`` (``:329-330``, ``:344``, ``:351``). Once every
    ``compress_ratio`` tokens it softmaxes the gate rows over the window's
    TOKEN axis (``:348``), takes the gate-weighted sum of the value rows,
    RMSNorms the pooled vector (``:368``) and RoPEs it at the window's BASE
    position (``:370``/``:372``).

    Two window shapes, both driven by ``coff = 1 + (ratio == 4)``
    (``:296-298``):

    * ``ratio == 4`` (``coff == 2``): the window is 8 tokens and overlaps the
      previous group by 4. The projections are twice as wide because each
      token carries two roles — columns ``[0:head_dim]`` when it acts as the
      OLDER half of a window, ``[head_dim:2*head_dim]`` when it acts as the
      CURRENT half. ``overlap_transform`` (``:313-320``) is exactly that
      placement, and it fills the missing older half of the first group with
      ``0`` for values and ``-inf`` for scores (``:346-347``).
    * ``ratio == 128`` (``coff == 1``): the window is exactly 128 tokens with
      no overlap and only columns ``[0:head_dim]`` exist.

    The parameters are unquantized: the reference declares both linears
    ``dtype=torch.float32`` with no scale (``:303-304``), so quantizing them
    would be a numerics change the incumbent does not have. The checkpoint
    splits them as ``compressor.wkv`` and ``compressor.wgate``, which is why
    this module declares two weights.

    ``rotate=True`` (the indexer's copy, ``:404``) swaps the output
    quantization from group-64 FP8 to a Hadamard rotation plus group-32 FP4
    (``:374-376``).

    >>> PARALLELISM: fully replicated. The reference builds both linears as
    plain ``Linear``; the pooled latent IS the cache content, which every
    core needs whole. <<<

    RECORDED GAP — decode-time window state. The reference keeps the raw
    per-token ``(kv, score)`` rows in two persistent buffers so a window can
    span forward passes (``:309-310``)::

        kv_state:    [max_batch, coff * ratio, coff * head_dim] fp32, zeros
        score_state: [max_batch, coff * ratio, coff * head_dim] fp32, -inf

    with the decode update at ``:350-365`` and the half-window roll at
    ``:359-360``. Those buffers have no row in the recorded KV layout, and
    this module cannot create them: the runner forces every buffer to meta
    (``neuron_model_runner.py:1231-1232``), so a real-valued buffer built in
    ``__init__`` would arrive with no data. This implementation therefore
    pools over the windows visible in the CURRENT forward — exact for a full
    prefill and for any window wholly inside the current chunk.
    ``prev_state`` is accepted as a keyword-only hook so the persistent state
    can be wired in later without changing this class's public signature.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        *,
        head_dim: int | None = None,
        rope_dim: int | None = None,
        rotate: bool = False,
    ) -> None:
        """Build the compressor for ``layer_idx``.

        Args:
            config: The model config.
            layer_idx: Owning transformer block index; selects
                ``compress_ratio`` and the RoPE base.
            head_dim: Pooled-latent width. Defaults to ``config.head_dim``
                (512, the MLA latent); the DSA indexer nests a copy with
                ``config.index_head_dim`` (128). Keyword-only with a default
                so the contract's positional ``(config, layer_idx)`` stays
                exact.
            rope_dim: Width of the rotated tail; defaults to
                ``config.qk_rope_head_dim`` (64).
            rotate: Use the indexer's Hadamard + FP4 output quantization.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.eps = config.rms_norm_eps

        self.head_dim = config.head_dim if head_dim is None else head_dim
        self.rope_dim = config.qk_rope_head_dim if rope_dim is None else rope_dim
        self.hidden_size = config.hidden_size
        self.compress_ratio = config.compress_ratio(layer_idx)
        self.rotate = rotate

        # <-- MODEL-SPECIFIC: overlap mode keys off ratio 4 exactly
        # (dsv4_ref/model.py:296), not off "ratio > 1".
        self.overlap = self.compress_ratio == 4
        self.coff = 1 + int(self.overlap)
        self.window = self.coff * self.compress_ratio
        self.proj_width = self.coff * self.head_dim

        (
            self.rope_theta,
            self.rope_original_seq_len,
            self.rope_factor,
            self.rope_beta_fast,
            self.rope_beta_slow,
        ) = _rope_params(config, layer_idx)

        # >>> PARALLELISM: replicated — no shard dim on any of the four. <<<
        self.wkv_weight = nn.Parameter(
            torch.empty(self.proj_width, self.hidden_size, dtype=self.dtype),
            requires_grad=False,
        )
        self.wgate_weight = nn.Parameter(
            torch.empty(self.proj_width, self.hidden_size, dtype=self.dtype),
            requires_grad=False,
        )
        # ``ape`` is a learned per-phase additive bias on the GATE row only,
        # indexed by ``position % compress_ratio``
        # (dsv4_ref/model.py:300, :344).
        self.ape = nn.Parameter(
            torch.empty(self.compress_ratio, self.proj_width, dtype=torch.float32),
            requires_grad=False,
        )
        self.norm_weight = nn.Parameter(
            torch.empty(self.head_dim, dtype=self.dtype), requires_grad=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        prev_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compress the window ENDING at each token.

        Row ``t`` of the result is the pooled latent of the window whose last
        token is ``t``. Only rows where ``(position + 1) % compress_ratio ==
        0`` are real compressed slots (``dsv4_ref/model.py:350``); the caller
        drops the rest by handing ``-1`` in the coarse slot mapping, so the
        per-token formulation costs nothing and needs no data-dependent
        gather.

        Args:
            hidden_states: ``[T, hidden_size]`` — the layer's normed input.
            positions: ``[T]`` int64 absolute positions.
            prev_state: Reserved for the persistent raw-row buffers; see the
                class docstring. Ignored today.

        Returns:
            ``[T, head_dim]`` fp32: RMSNormed, RoPEd at the window's base
            position, and through the output quantize/dequantize round trip
            the reference applies in place.
        """
        del prev_state  # see class docstring

        tokens = hidden_states.shape[0]
        ratio = self.compress_ratio
        window = self.window
        head_dim = self.head_dim

        # ── Per-token raw rows, fp32 (dsv4_ref/model.py:328-330) ────────
        hidden32 = hidden_states.to(torch.float32)
        kv_rows = hidden32 @ self.wkv_weight.to(torch.float32).t()
        score_rows = hidden32 @ self.wgate_weight.to(torch.float32).t()

        phase = torch.remainder(positions.to(torch.long), ratio)
        score_rows = score_rows + torch.index_select(
            self.ape.to(torch.float32), 0, phase
        )

        # ── Window gather: rows [t-window+1 .. t] for every t ───────────
        # Built purely from arange arithmetic, so the shape is static and the
        # sequence head is handled by a mask rather than a short gather.
        token_idx = torch.arange(tokens, device=hidden_states.device).unsqueeze(1)
        lag = torch.arange(window - 1, -1, -1, device=hidden_states.device).unsqueeze(0)
        gather_idx = token_idx - lag  # [T, window]
        in_range = gather_idx >= 0
        gather_flat = torch.clamp(gather_idx, min=0).reshape(-1)

        kv_win = torch.index_select(kv_rows, 0, gather_flat).view(
            tokens, window, self.proj_width
        )
        score_win = torch.index_select(score_rows, 0, gather_flat).view(
            tokens, window, self.proj_width
        )

        # ── Role-dependent column selection (dsv4_ref/model.py:313-320) ─
        # C4: window slots [0:ratio] are the OLDER group and read columns
        # [0:head_dim]; slots [ratio:2*ratio] are the CURRENT group and read
        # columns [head_dim:2*head_dim].
        if self.coff == 2:
            kv_sel = torch.cat(
                (kv_win[:, :ratio, :head_dim], kv_win[:, ratio:, head_dim:]), dim=1
            )
            score_sel = torch.cat(
                (score_win[:, :ratio, :head_dim], score_win[:, ratio:, head_dim:]),
                dim=1,
            )
        else:
            kv_sel = kv_win[:, :, :head_dim]
            score_sel = score_win[:, :, :head_dim]

        # ── Softmax over the token axis, then the gated sum ─────────────
        # Out-of-range slots take -inf on the score (weight 0), matching the
        # reference's ``overlap_transform(score, float("-inf"))``.
        score_sel = torch.where(
            in_range.unsqueeze(-1),
            score_sel,
            torch.full_like(score_sel, float("-inf")),
        )
        pooled = (kv_sel * torch.softmax(score_sel, dim=1)).sum(dim=1)

        # ── RMSNorm, then RoPE at the window's BASE position ────────────
        # dsv4_ref/model.py:368 casts the fp32 pooled value to the compute
        # dtype BEFORE the norm, so the bf16 round trip is part of the
        # numerics.
        normed = _rms_norm(pooled.to(self.dtype), self.norm_weight, self.eps)
        base_positions = (positions.to(torch.long) // ratio) * ratio
        cos, sin = _cos_sin(
            base_positions,
            self.rope_dim,
            self.rope_theta,
            self.rope_original_seq_len,
            self.rope_factor,
            self.rope_beta_fast,
            self.rope_beta_slow,
        )
        roped = _gptj_rope(normed, cos, sin, self.rope_dim)

        # ── Output quantize/dequantize simulation ───────────────────────
        # The reference quantizes in place and hands the DEQUANTIZED value to
        # both the cache write and the current step's scoring, so the round
        # trip belongs here rather than at the cache boundary.
        nope = self.head_dim - self.rope_dim
        if self.rotate:
            # Indexer copy: Hadamard-rotate the whole head, then FP4 group-32
            # (dsv4_ref/model.py:374-376). The rotation covers the RoPE
            # columns too — it is applied to ``kv``, not ``kv[..., :-rd]``.
            return _quant_fp4_simulate(roped @ _hadamard(self.head_dim, roped.device))
        codes, scales = _quant_fp8_ue8m0(roped[..., :nope])
        return torch.cat(
            (_dequant_fp8(codes, scales, _KV_QUANT_GROUP), roped[..., nope:]), dim=-1
        )


# ---------------------------------------------------------------------------
# DSA lightning indexer
# ---------------------------------------------------------------------------


class DeepseekV4Indexer(nn.Module):
    """DeepSeek Sparse Attention "lightning indexer" (``dsv4_ref/model.py:386-439``).

    <-- MODEL-SPECIFIC: a cheap side attention that picks WHICH compressed
    slots the real attention may look at. It projects the shared q latent to
    ``index_n_heads`` (64) heads of ``index_head_dim`` (128), RoPEs them,
    Hadamard-rotates and FP4-quantizes them (``:417-422``), scores them
    against its own compressed K pool and keeps the ``index_topk`` (512) best
    slots per query. ``NF.sparse_indexer_topk`` implements the scoring
    (``:424-436``): ReLU on the per-head logits, weighted sum over heads with
    ``weights_proj(x) * index_head_dim**-0.5 * n_heads**-0.5``, NO softmax,
    causal cap ``slot < (pos + 1) // ratio``, ``-1`` in every unusable output
    slot.

    ``weights_proj`` is unquantized (``:400``, bf16); ``wq_b`` is block-FP8
    like the MLA projections.

    >>> PARALLELISM: declared REPLICATED, per the port's interface contract
    §2, so ``weight_loaders`` binds the full ``[8192, 1024]`` and
    ``[64, 4096]`` tensors. The reference instead builds both as
    ``ColumnParallelLinear`` over the 64 index heads and all-reduces the
    per-head scores (``:399-400``, ``:428-429``). The two are mathematically
    identical — replicated computes all 64 heads on every core instead of one
    head plus an all-reduce — so this is a cost choice, not a numerics one,
    and it is the one place the port knowingly spends 64x the indexer FLOPs.
    <<<
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        super().__init__()
        if not config.has_indexer(layer_idx):
            raise ValueError(
                f"DeepseekV4Indexer built for layer {layer_idx}, which is class "
                f"{config.layer_class(layer_idx)!r}. The indexer only exists on "
                "compress_ratio==4 layers; building it elsewhere would allocate "
                "an index cache nothing reads."
            )
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype

        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.hidden_size = config.hidden_size
        self.topk = config.index_topk
        self.compress_ratio = config.compress_ratio(layer_idx)

        # >>> PARALLELISM: replicated (see class docstring). <<<
        self.wq_b_weight = nn.Parameter(
            torch.empty(
                self.n_heads * self.head_dim, self.q_lora_rank, dtype=_FP8_DTYPE
            ),
            requires_grad=False,
        )
        self.wq_b_scale = nn.Parameter(
            torch.empty(
                _num_blocks(self.n_heads * self.head_dim),
                _num_blocks(self.q_lora_rank),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.weights_proj_weight = nn.Parameter(
            torch.empty(self.n_heads, self.hidden_size, dtype=self.dtype),
            requires_grad=False,
        )

        # The indexer's own narrower compressor: head_dim 128, same gate
        # math, its own ape/norm, Hadamard + FP4 output
        # (dsv4_ref/model.py:404).
        self.compressor = DeepseekV4KVCompressor(
            config,
            layer_idx,
            head_dim=self.head_dim,
            rope_dim=self.rope_dim,
            rotate=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_latent: torch.Tensor,
        positions: torch.Tensor,
        index_k_cache: torch.Tensor,
        index_v_cache: torch.Tensor,
        index_slot_mapping: torch.Tensor,
        *,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        pool_span: int = 0,
        block_table: torch.Tensor | None = None,
        block_size: int = 0,
    ) -> torch.Tensor:
        """Select the top-``index_topk`` compressed slots per query token.

        Args:
            hidden_states: ``[T, hidden_size]`` — feeds ``weights_proj`` and
                the nested compressor.
            q_latent: ``[T, q_lora_rank]`` — the ALREADY-normed shared q
                latent. The reference reuses the MLA layer's ``qr`` rather
                than projecting a second one (``dsv4_ref/model.py:502``,
                ``:517``).
            positions: ``[T]`` int64 absolute positions.
            index_k_cache: This layer's index-K cache (128 columns).
            index_v_cache: Its group-32 scale companion (4 columns used).
            index_slot_mapping: Coarse (compressed-cadence) slot mapping,
                ``-1`` wherever no slot closes.
            rope_cos: ``[T, rope_dim // 2]`` from the backbone's table.
            rope_sin: Matching sine table.
            pool_span: Static number of candidate slots to score out of the
                CACHE. ``0`` selects the in-forward pool, which is what a
                full prefill needs; a positive span selects the paged read,
                which is what decode needs.
            block_table: ``[B, max_blocks_per_seq]`` for the paged read.
            block_size: Slots per block, for the paged read.

        Returns:
            ``[T, index_topk]`` int32 POOL-LOCAL compressed-slot indices,
            ``-1`` where no slot qualifies.
        """
        ratio = self.compress_ratio

        # ── This step's fresh index-K rows, and the cache write ─────────
        # The compressor already applied the Hadamard rotation and the FP4
        # round trip; FP8 group-32 is only the transport format into the
        # cache, and the group size matches the FP4 group so no group's
        # dynamic range gets merged.
        fresh = self.compressor(hidden_states, positions)  # [T, 128] fp32
        codes, scales = _quant_fp8_ue8m0(fresh, _FP4_QUANT_GROUP)
        _masked_scatter_rows(index_k_cache, index_slot_mapping, codes)
        _masked_scatter_rows(index_v_cache, index_slot_mapping, scales)

        # ── Candidate pool ──────────────────────────────────────────────
        index_k: torch.Tensor | None = None
        key_slot_ids: torch.Tensor | None = None
        key_scale: torch.Tensor | None = None
        if pool_span > 0 and block_table is not None:
            # Paged read: pool-local slot j of every sequence, translated
            # through the block table. Slots past the causal frontier are
            # dropped by the op's own cap, so no extra validity map is
            # needed.
            key_slot_ids = _paged_slot_ids(block_table, pool_span, block_size)
            key_scale = _gather_scale_columns(
                index_v_cache, key_slot_ids, _INDEX_NUM_SCALES, _FP4_QUANT_GROUP
            )
        else:
            # In-forward pool: the closing token of group j is token
            # ``j * ratio + ratio - 1``, so a strided slice IS the pool in
            # pool-local order. Static shape, no gather.
            index_k = fresh[ratio - 1 :: ratio]

        return NF.sparse_indexer_topk(
            q_latent,
            self.wq_b_weight,
            None,
            self.weights_proj_weight,
            index_k_cache,
            self.topk,
            wq_b_scale=self.wq_b_scale,
            hidden_states=hidden_states,
            index_k=index_k,
            key_slot_ids=key_slot_ids,
            key_scale=key_scale,
            positions=positions,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            n_index_heads=self.n_heads,
            index_head_dim=self.head_dim,
            rope_head_dim=self.rope_dim,
            compress_ratio=ratio,
            head_activation="relu",
            index_offset=0,
            pad_index=_PAD_INDEX,
        )


# ---------------------------------------------------------------------------
# RoPE configuration
# ---------------------------------------------------------------------------


def _rope_params(
    config: DeepseekV4Config, layer_idx: int
) -> tuple[float, int, float, float, float]:
    """Return ``(theta, original_seq_len, factor, beta_fast, beta_slow)``.

    ``dsv4_ref/model.py:481-487``: compressed layers use
    ``compress_rope_theta`` WITH YaRN; layers without compression use the
    base ``rope_theta`` and pass ``original_seq_len = 0``, which turns YaRN
    OFF. ``config.rope_theta_for_layer`` already picks the theta; the
    original-length gate is applied here.
    """
    theta = config.rope_theta_for_layer(layer_idx)
    if not config.has_compressed_cache(layer_idx):
        return (theta, 0, 1.0, 32.0, 1.0)
    return (
        theta,
        int(config.rope_original_seq_len),
        float(config.rope_factor),
        float(config.rope_beta_fast),
        float(config.rope_beta_slow),
    )


# ---------------------------------------------------------------------------
# MLA attention block
# ---------------------------------------------------------------------------


class DeepseekV4Attention(nn.Module):
    """Multi-head Latent Attention with sliding-window + sparse KV legs.

    <-- MODEL-SPECIFIC: three layer classes share this one module, chosen
    once at construction from ``config.layer_class(layer_idx)``
    (``dsv4_ref/model.py:472-477``):

    ``swa_only`` (ratio 0/1; layers 0, 1, the tail layers and the draft block)
        No compressed pool and no indexer. The index list is the sliding
        window alone (``:513``).
    ``sparse_c4`` (ratio 4)
        Compressed pool + DSA indexer. The index list is the window's indices
        CONCATENATED with the indexer's top-512 (``:520``) and ONE softmax
        covers both (``dsv4_ref/kernel.py:294-351``), which is why this port
        issues one ``NF.mla_sparse_attention`` rather than two attentions it
        would have to merge by LSE.
    ``dense_c128`` (ratio 128)
        Compressed pool, no indexer: the candidate list over the pool is
        generated causally instead of learned (``:519``
        ``get_compress_topk_idxs``), which is what "dense" means here.

    All three then go through the same grouped o-projection.

    >>> PARALLELISM: one query head per core at TP=64. ``wq_b`` is
    column-sharded over heads. ``wo_a`` is the group stage: the reference
    computes ``einsum("bsgd,grd->bsgr", o, wo_a)`` where ``d`` is exactly the
    concatenation of the group's 8 heads x 512 (``:542-546``), so splitting
    ``d`` one head per core and summing the 8 partials over ``oproj_group``
    is algebraically the same reduction — no rescaling is involved. ``wo_b``
    is row-parallel over the whole TP group, and because each core takes a
    DISTINCT ``group_rank``-indexed slice of its group's ``o_lora_rank``
    vector, the final 64-way reduction is a plain sum with no ``1/8``
    prescale. That reduction is this module's, because
    ``NF.mla_grouped_oproj`` returns a per-core partial. <<<
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        *,
        oproj_group,
        oproj_group_rank: int,
        oproj_group_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.eps = config.rms_norm_eps

        # ── Config-derived shape (decides the graph, once) ──────────────
        self.layer_class = config.layer_class(layer_idx)
        self.compress_ratio = config.compress_ratio(layer_idx)
        self.has_compressed_cache = config.has_compressed_cache(layer_idx)
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.qk_nope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.o_groups = config.o_groups
        self.sliding_window = config.sliding_window
        self.index_topk = config.index_topk

        (
            self.rope_theta,
            self.rope_original_seq_len,
            self.rope_factor,
            self.rope_beta_fast,
            self.rope_beta_slow,
        ) = _rope_params(config, layer_idx)

        # <-- MODEL-SPECIFIC: the MLA softmax scale is over the FULL latent
        # width, nope + rope (dsv4_ref/model.py:470). The reference applies
        # no YaRN mscale multiplier, so there is no extra factor.
        self.scale = float(self.head_dim) ** -0.5

        # >>> PARALLELISM: head sharding + the o-projection subgroup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.num_attention_heads = config.num_attention_heads
        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"tensor_parallel_size={self.world_size} does not divide "
                f"num_attention_heads={self.num_attention_heads}."
            )
        self.num_local_heads = self.num_attention_heads // self.world_size

        self.oproj_group = oproj_group
        self.oproj_group_rank = oproj_group_rank
        self.oproj_group_size = oproj_group_size

        # ── Parameters ──────────────────────────────────────────────────
        # Every weight is [out, in] — the checkpoint's own orientation and
        # the one the block-FP8 GEMM wants. Do NOT flag these as storage
        # transposed (llama3 does, because ITS parameters are [in, out]).
        q_lora = self.q_lora_rank
        latent = self.head_dim
        heads = self.num_local_heads

        # >>> PARALLELISM: replicated. The reference builds wq_a and wkv as
        # two plain (non-parallel) Linears (dsv4_ref/model.py:463, :466); the
        # port fuses them into one [q_lora_rank + head_dim, hidden] stack so
        # the replicated GEMM happens once. <<<
        fused_out = q_lora + latent
        self.fused_wqa_wkv_weight = nn.Parameter(
            torch.empty(fused_out, self.hidden_size, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.fused_wqa_wkv_scale = nn.Parameter(
            torch.empty(
                _num_blocks(fused_out),
                _num_blocks(self.hidden_size),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        # >>> PARALLELISM: column-sharded over query heads
        # (dsv4_ref/model.py:465). <<<
        wq_b_out = heads * latent
        self.wq_b_weight = nn.Parameter(
            torch.empty(wq_b_out, q_lora, dtype=_FP8_DTYPE), requires_grad=False
        )
        self.wq_b_scale = nn.Parameter(
            torch.empty(
                _num_blocks(wq_b_out), _num_blocks(q_lora), dtype=torch.float32
            ),
            requires_grad=False,
        )

        # >>> PARALLELISM: one o_group per ``oproj_group_size`` cores; this
        # core owns its head's slice of that group's K extent. <<<
        wo_a_k_local = heads * latent
        self.wo_a_weight = nn.Parameter(
            torch.empty(self.o_lora_rank, wo_a_k_local, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.wo_a_scale = nn.Parameter(
            torch.empty(
                _num_blocks(self.o_lora_rank),
                _num_blocks(wo_a_k_local),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        # >>> PARALLELISM: row-parallel over the full TP group. <<<
        wo_b_k_local = (self.o_groups * self.o_lora_rank) // self.world_size
        self.wo_b_weight = nn.Parameter(
            torch.empty(self.hidden_size, wo_b_k_local, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.wo_b_scale = nn.Parameter(
            torch.empty(
                _num_blocks(self.hidden_size),
                _num_blocks(wo_b_k_local),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        # Norms on the two latents, replicated (dsv4_ref/model.py:464, :467).
        # NOTE: there is no third norm weight. The per-head RMSNorm applied
        # to q after wq_b is WEIGHTLESS (``:504``: ``q *= rsqrt(q.square()
        # .mean(-1) + eps)``), so it needs eps only, and NF.mla_qkv applies
        # it under ``apply_q_head_norm``.
        self.q_norm_weight = nn.Parameter(
            torch.empty(q_lora, dtype=self.dtype), requires_grad=False
        )
        self.kv_norm_weight = nn.Parameter(
            torch.empty(latent, dtype=self.dtype), requires_grad=False
        )

        # <-- MODEL-SPECIFIC: per-query-head attention sink. Its
        # un-normalized weight ``exp(sink)`` enters the softmax DENOMINATOR
        # only (``dsv4_ref/kernel.py:345-348``), so a head whose whole index
        # list is -1 yields 0 rather than NaN.
        # >>> PARALLELISM: exactly the local heads — the reference declares
        # ``torch.empty(n_local_heads)`` fp32 (``dsv4_ref/model.py:462``);
        # the pad to 16 heads is a kernel-tiling artifact, not a parameter.
        # <<<
        self.attn_sink = nn.Parameter(
            torch.empty(heads, dtype=torch.float32), requires_grad=False
        )

        # ── Optional submodules (config decides, once) ───────────────────
        self.compressor: DeepseekV4KVCompressor | None = None
        if self.has_compressed_cache:
            self.compressor = DeepseekV4KVCompressor(config, layer_idx)

        self.indexer: DeepseekV4Indexer | None = None
        if config.has_indexer(layer_idx):
            self.indexer = DeepseekV4Indexer(config, layer_idx)

        # ── KV caches, bound externally by bind_kv_cache() ──────────────
        self.latent_k_cache: torch.Tensor | None = None
        self.latent_v_cache: torch.Tensor | None = None
        self.rope_cache: torch.Tensor | None = None
        self.scale_cache: torch.Tensor | None = None
        self.swa_k_cache: torch.Tensor | None = None
        self.swa_v_cache: torch.Tensor | None = None
        self.index_k_cache: torch.Tensor | None = None
        self.index_v_cache: torch.Tensor | None = None

        # ── Weight loaders ──────────────────────────────────────────────
        # Declared here, implemented in weight_loaders.py: this module owns
        # names and shapes, that one owns checkpoint-key transformation.
        # ``shared_*`` are the shared-expert subgroup coordinates of the
        # shared loader entry point; attention holds no shared-expert weight,
        # so it passes the identity subgroup.
        attach_attention_loaders(
            self,
            config,
            tp_size=self.world_size,
            tp_rank=self.rank,
            group_rank=oproj_group_rank,
            group_size=oproj_group_size,
            shared_tp_size=1,
            shared_tp_rank=0,
        )

    # ------------------------------------------------------------------
    # KV cache declaration — the ONE place names and widths are written
    # ------------------------------------------------------------------
    def kv_layer_specs(self, layer_idx: int) -> list[LayerSpec]:
        """Return every KV cache pair this layer needs, in declaration order.

        ``get_kv_spec()`` is the concatenation of this over all layers and
        ``expected_kv_layer_names()`` derives from the same call, so a name
        or a width only ever appears here. The runner allocates each entry as
        a PAIR ``[k_cache, v_cache]`` of identical shape ``[num_blocks,
        num_kv_heads, block_size, head_size]`` and identical dtype (from
        ``--kv-cache-dtype``, not from ``LayerSpec.dtype``) — which is why
        the reference's single 512-wide row has to be re-expressed as
        same-shape pairs here.

        | name                             | heads | head_size | window | when       |
        |----------------------------------|-------|-----------|--------|------------|
        | ``layers.{i}.self_attn``         | 1     | 224       | None   | compressed |
        | ``layers.{i}.self_attn.rope``    | 1     | 128       | None   | compressed |
        | ``layers.{i}.self_attn.swa``     | 1     | 512       | 128    | every layer|
        | ``layers.{i}.self_attn.indexer`` | 1     | 128       | None   | C4 layer   |

        Column layout, written by this module and read by the ``NF`` ops
        through their ``*_widths`` / ``*_scale_cache`` arguments:

        * ``self_attn``: ``k_cache[0:224]`` = latent NoPE ``[0:224]``,
          ``v_cache[0:224]`` = NoPE ``[224:448]``.
        * ``self_attn.rope``: ``k_cache[0:64]`` = the 64 RoPE columns,
          ``v_cache[0:7]`` = the 7 group-64 UE8M0 dequant scales.
          ``compressed_widths=(224, 224, 64)``.
        * ``self_attn.swa``: ``k_cache[0:512]`` = the WHOLE latent (448 NoPE
          + 64 RoPE), ``v_cache[0:7]`` = its 7 scales. ``swa_widths=(512,)``.
        * ``self_attn.indexer``: ``k_cache[0:128]`` = the index-K columns,
          ``v_cache[0:4]`` = its 4 group-32 scales.

        Per-slot bytes at ``fp8`` (1 B per column): compressed leg
        ``2*224 + 2*128 = 704`` B with 519 used; SWA leg ``2*512 = 1024`` B
        with 519 used. The compressed number is unchanged from
        ``pricing-and-design.md`` §3; the SWA number is the delta, and it is
        the price of holding the reference's full 512-wide window row plus
        its group-64 scales inside ONE layer name (i.e. two same-shape
        tensors). The SWA leg is a ``SlidingWindowSpec``, so it is
        ``window``-bounded (128 slots per sequence per layer = 128 KiB)
        rather than context-bounded, while the compressed pool at ratio 4 is
        ``max_seq_len / 4`` slots — the leg that actually sets capacity.

        Args:
            layer_idx: The layer to declare for. Passed explicitly rather
                than read off ``self`` so the backbone can build the whole
                spec from one uniform call.
        """
        prefix = f"layers.{layer_idx}.self_attn"
        specs: list[LayerSpec] = []

        if self.config.has_compressed_cache(layer_idx):
            specs.append(
                LayerSpec(
                    name=prefix,
                    num_kv_heads=1,
                    head_size=_LATENT_PAIR_HEAD_SIZE,
                    dtype=_FP8_DTYPE,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
            # Keeping the scales at group-64 granularity (rather than one
            # per-tensor k_scale/v_scale) is what preserves the reference's
            # KV numerics (dsv4_ref/model.py:512 quantizes with block 64).
            specs.append(
                LayerSpec(
                    name=f"{prefix}.rope",
                    num_kv_heads=1,
                    head_size=_ROPE_PAIR_HEAD_SIZE,
                    dtype=_FP8_DTYPE,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )

        # Every layer has the sliding-window leg, including the SWA-only
        # layers and the draft block. ``sliding_window_size`` is what routes
        # this pair into its own KV-cache group in the runner
        # (pricing-and-design.md §2).
        specs.append(
            LayerSpec(
                name=f"{prefix}.swa",
                num_kv_heads=1,
                head_size=_SWA_PAIR_HEAD_SIZE,
                dtype=_FP8_DTYPE,
                sliding_window_size=self.config.sliding_window,
                chunk_size=None,
            )
        )

        if self.config.has_indexer(layer_idx):
            specs.append(
                LayerSpec(
                    name=f"{prefix}.indexer",
                    num_kv_heads=1,
                    head_size=_INDEXER_PAIR_HEAD_SIZE,
                    dtype=_FP8_DTYPE,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return specs

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        """Attach the runner-allocated cache pairs to this layer.

        Args:
            kv_caches: Name -> ``[k_cache, v_cache]``, keyed by exactly the
                names :meth:`kv_layer_specs` declared.

        Raises:
            KeyError: when a declared name is missing. Failing here beats
                discovering an unbound cache as a ``NoneType`` mid-compile.
        """
        prefix = f"layers.{self.layer_idx}.self_attn"

        def _pair(name: str) -> tuple[torch.Tensor, torch.Tensor]:
            if name not in kv_caches:
                raise KeyError(
                    f"KV cache for layer {name!r} not initialized; "
                    "kv_layer_specs() declared it."
                )
            pair = kv_caches[name]
            return pair[0], pair[1]

        if self.has_compressed_cache:
            self.latent_k_cache, self.latent_v_cache = _pair(prefix)
            self.rope_cache, self.scale_cache = _pair(f"{prefix}.rope")

        self.swa_k_cache, self.swa_v_cache = _pair(f"{prefix}.swa")

        if self.indexer is not None:
            self.index_k_cache, self.index_v_cache = _pair(f"{prefix}.indexer")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict[str, dict],
        *,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one MLA block.

        Args:
            hidden_states: ``[T, hidden_size]`` — the hc-reduced stream,
                already RMSNormed by the decoder layer.
            positions: ``[T]`` absolute positions.
            attn_metadata: ``dict[str, dict]``, keyed by the KV layer names
                this module declares.
            rope_cos: ``[T, rope_head_dim // 2]``, built ONCE per forward by
                the backbone with this layer class's theta. Keyword-only with
                a default so the contract's positional call form stays valid;
                when absent it is rebuilt locally, which costs two
                transcendentals per token.
            rope_sin: Matching sine table.

        Returns:
            ``[T, hidden_size]``.
        """
        prefix = f"layers.{self.layer_idx}.self_attn"
        swa_md = attn_metadata[f"{prefix}.swa"]

        # Prefill/decode dispatch. Read off THIS layer's ``.swa`` entry, not
        # off ``layers.0.self_attn``: layer 0 is SWA-only, so the bare
        # ``self_attn`` name is not declared there at all, while ``.swa``
        # exists on every layer.
        is_decode = swa_md["max_query_len"] <= swa_md["decode_token_threshold"]

        positions_l = positions.to(torch.long)
        if rope_cos is None or rope_sin is None:
            rope_cos, rope_sin = _cos_sin(
                positions_l,
                self.rope_head_dim,
                self.rope_theta,
                self.rope_original_seq_len,
                self.rope_factor,
                self.rope_beta_fast,
                self.rope_beta_slow,
            )

        # ── Q/KV latents (fused wq_a+wkv, both norms, wq_b, RoPE) ───────
        # Positional order is the plan-fixed one; the block scales and the
        # QAT knobs are keyword-only extensions. ``latent_kv`` comes back
        # with the group-64 fp8 round trip already applied to its NoPE
        # columns, so re-quantizing it below for the cache write is
        # idempotent rather than a second lossy pass.
        q_nope, q_rope, latent_kv = NF.mla_qkv(
            hidden_states,
            self.fused_wqa_wkv_weight,
            self.wq_b_weight,
            self.q_norm_weight,
            self.kv_norm_weight,
            rope_cos,
            rope_sin,
            positions,
            wqa_wkv_scale=self.fused_wqa_wkv_scale,
            wqb_scale=self.wq_b_scale,
            eps=self.eps,
            qk_rope_head_dim=self.rope_head_dim,
            apply_q_head_norm=True,
            kv_nope_fp8_qat=True,
            kv_qat_group_size=_KV_QUANT_GROUP,
        )
        # [T, heads_local, head_dim]: NoPE columns first, RoPE tail last —
        # the ordering both the caches and the inverse RoPE assume.
        query = torch.cat((q_nope, q_rope), dim=-1)

        # ── Sliding-window cache write (every token, raw cadence) ───────
        # k_cache takes the whole 512-wide latent, v_cache its 7 group-64
        # scales. Re-deriving the codes from the round-tripped value is exact
        # because the round trip left it on the quantization grid.
        nope_codes, nope_scales = _quant_fp8_ue8m0(latent_kv[..., : self.nope_head_dim])
        _masked_scatter_rows(
            self.swa_k_cache,
            swa_md["slot_mapping"],
            torch.cat(
                (nope_codes.to(torch.float32), latent_kv[..., self.nope_head_dim :]),
                dim=-1,
            ),
        )
        _masked_scatter_rows(self.swa_v_cache, swa_md["slot_mapping"], nope_scales)

        # ── Compressed cache write (coarse cadence) ─────────────────────
        if self.compressor is not None:
            self._write_compressed_cache(
                self.compressor(hidden_states, positions_l),
                self._coarse_slots(attn_metadata[prefix]["slot_mapping"], positions_l),
                self._coarse_slots(
                    attn_metadata[f"{prefix}.rope"]["slot_mapping"], positions_l
                ),
            )

        # ── Which compressed slots may this layer see? ──────────────────
        topk_indices = None
        if self.indexer is not None:
            index_md = attn_metadata[f"{prefix}.indexer"]
            span = (
                index_md["max_blocks_per_seq"] * index_md["block_size"]
                if is_decode
                else 0
            )
            # ``qr``: the indexer consumes the SAME normed q latent the MLA
            # path does (dsv4_ref/model.py:502, :517). NF.mla_qkv's fixed
            # return is (q_nope, q_rope, latent_kv) with no slot for it, so
            # it is recomputed here from the replicated wq_a half of the
            # fused stack rather than changing that return arity.
            topk_indices = self.indexer(
                hidden_states,
                self._q_latent(hidden_states),
                positions_l,
                self.index_k_cache,
                self.index_v_cache,
                self._coarse_slots(index_md["slot_mapping"], positions_l),
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                pool_span=span,
                block_table=index_md["block_table_tensor"],
                block_size=index_md["block_size"],
            )

        # ── Attention ───────────────────────────────────────────────────
        if is_decode:
            attn_out = self._decode_attention(
                query, positions_l, attn_metadata, prefix, topk_indices
            )
        elif self.layer_class == LAYER_CLASS_SWA_ONLY:
            # No compressed pool on this layer, so the sliding window is the
            # whole attention (dsv4_ref/model.py:513 with no concat). MLA has
            # no separate V: the same latent row is K and V, and the value
            # fed in is the QAT round-tripped one (``:512`` runs before
            # ``:533``).
            keys = latent_kv.unsqueeze(1)
            attn_out = NF.swa_attention(
                query,
                keys,
                keys,
                self.sliding_window,
                self.attn_sink,
                self.scale,
                positions=positions_l,
                kv_positions=positions_l,
                causal=True,
            )
        else:
            assert self.layer_class in (LAYER_CLASS_SPARSE_C4, LAYER_CLASS_DENSE_C128)
            latent_md = attn_metadata[prefix]
            if topk_indices is None:
                # C128: no indexer — the candidate list is the whole causal
                # compressed pool (dsv4_ref/model.py:519). The op applies the
                # causal cap itself from ``positions`` + ``compress_ratio``,
                # so a plain arange over the static pool span IS the faithful
                # "dense" list.
                span = latent_md["max_blocks_per_seq"] * latent_md["block_size"]
                topk_indices = (
                    torch.arange(span, device=hidden_states.device, dtype=torch.int32)
                    .unsqueeze(0)
                    .expand(hidden_states.shape[0], span)
                )
            # ONE call covers both KV legs under one shared softmax, matching
            # the reference's single concatenated index list (``:520``).
            attn_out = NF.mla_sparse_attention(
                query,
                self.latent_k_cache,
                self.swa_k_cache,
                topk_indices,
                self.attn_sink,
                self.scale,
                self.sliding_window,
                positions=positions_l,
                compressed_v_cache=self.latent_v_cache,
                compressed_rope_cache=self.rope_cache,
                compressed_scale_cache=self.scale_cache,
                compressed_widths=(
                    _LATENT_PAIR_HEAD_SIZE,
                    _LATENT_PAIR_HEAD_SIZE,
                    self.rope_head_dim,
                ),
                compressed_block_table=latent_md["block_table_tensor"],
                topk_index_offset=0,
                compress_ratio=self.compress_ratio,
                swa_scale_cache=self.swa_v_cache,
                swa_widths=(self.head_dim,),
                swa_block_table=swa_md["block_table_tensor"],
                nope_dim=self.nope_head_dim,
                rope_dim=self.rope_head_dim,
                quant_group_size=_KV_QUANT_GROUP,
            )

        # ── Inverse RoPE, then the grouped o-projection ─────────────────
        # <-- MODEL-SPECIFIC: the attention output still carries the RoPE
        # rotation on its last 64 columns and ``wo_a`` is trained against the
        # un-rotated ("content") space, so the reference de-rotates first
        # (dsv4_ref/model.py:539). NF.mla_grouped_oproj takes no positions,
        # so it cannot undo the rotation itself; the inverse is applied here,
        # once, immediately before the projection.
        attn_out = _gptj_rope(
            attn_out.reshape(
                hidden_states.shape[0], self.num_local_heads, self.head_dim
            ),
            rope_cos,
            rope_sin,
            self.rope_head_dim,
            inverse=True,
        )

        # >>> PARALLELISM: ``group_pg`` drives the intra-o_group sum that
        # reconstructs the reference's einsum over the group's 8 heads; the
        # op returns this core's row-parallel PARTIAL, so the final TP sum is
        # this module's. It accumulates in fp32 before the cast back, as
        # dsv4_ref/model.py:182-186 does. <<<
        partial = NF.mla_grouped_oproj(
            attn_out,
            self.wo_a_weight,
            self.wo_b_weight,
            self.o_groups,
            self.oproj_group,
            wo_a_scale=self.wo_a_scale,
            wo_b_scale=self.wo_b_scale,
            group_rank=self.oproj_group_rank,
            out_dtype=torch.float32,
        )
        if self.world_size > 1:
            partial = self.tp_group.all_reduce(partial)
        return partial.to(self.dtype)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _decode_attention(
        self,
        query: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict[str, dict],
        prefix: str,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Single-token-per-sequence attention over the paged caches.

        ``NF.mla_decode_attention`` wants ``q`` as ``[B, H, 512]``, so the
        flat token axis is folded by the batch size the block table reports.
        A step that generates more than one token per sequence (speculative
        decoding / MTP) has ``max_query_len > decode_token_threshold`` and so
        has already been dispatched to the prefill op instead.

        SWA-only layers have no compressed pool. Rather than a second op they
        pass their window cache in BOTH cache slots with an all-``-1``
        candidate list, which the op masks out entirely — the same sentinel
        the reference uses for absent slots
        (``dsv4_ref/kernel.py:323-327``).
        """
        swa_md = attn_metadata[f"{prefix}.swa"]
        batch = swa_md["block_table_tensor"].shape[0]
        q = query.reshape(batch, self.num_local_heads, self.head_dim)
        pos = positions.reshape(batch)

        # <-- The SWA leg's block table is TRIMMED at decode. The runner
        # replaces a SlidingWindowSpec group's ``block_table_tensor`` with a
        # window-relevant gather and publishes ``swa_kv_pos_offset`` =
        # start_block * block_size (``neuron_model_runner.py:3966-3985``);
        # feeding absolute positions against that trimmed table reads the
        # wrong blocks for every sequence past the trimmed span. The offset
        # applies to the SWA leg ONLY — the compressed leg's table is a
        # FullAttentionSpec one and its causal cap needs absolute positions —
        # which is why it is a separate op argument rather than a shift of
        # ``positions``. Absent (prefill, or a short-sequence decode) it is
        # None and nothing shifts.
        swa_pos_offset = swa_md.get("swa_kv_pos_offset")

        if self.has_compressed_cache:
            latent_md = attn_metadata[prefix]
            return NF.mla_decode_attention(
                q,
                self.latent_k_cache,
                self.swa_k_cache,
                self.scale,
                self.attn_sink,
                positions=pos,
                window=self.sliding_window,
                compress_ratio=self.compress_ratio,
                topk_indices=topk_indices,
                topk_index_offset=0,
                max_compressed_slots=latent_md["max_blocks_per_seq"]
                * latent_md["block_size"],
                latent_v_cache=self.latent_v_cache,
                latent_rope_cache=self.rope_cache,
                latent_scale_cache=self.scale_cache,
                latent_widths=(
                    _LATENT_PAIR_HEAD_SIZE,
                    _LATENT_PAIR_HEAD_SIZE,
                    self.rope_head_dim,
                ),
                latent_block_table=latent_md["block_table_tensor"],
                swa_scale_cache=self.swa_v_cache,
                swa_widths=(self.head_dim,),
                swa_block_table=swa_md["block_table_tensor"],
                swa_pos_offset=swa_pos_offset,
                nope_dim=self.nope_head_dim,
                rope_dim=self.rope_head_dim,
                quant_group_size=_KV_QUANT_GROUP,
            )

        absent = torch.full(
            (batch, 1), _PAD_INDEX, dtype=torch.int32, device=query.device
        )
        return NF.mla_decode_attention(
            q,
            self.swa_k_cache,
            self.swa_k_cache,
            self.scale,
            self.attn_sink,
            positions=pos,
            window=self.sliding_window,
            compress_ratio=0,
            topk_indices=absent,
            max_compressed_slots=1,
            latent_scale_cache=self.swa_v_cache,
            latent_widths=(self.head_dim,),
            latent_block_table=swa_md["block_table_tensor"],
            swa_scale_cache=self.swa_v_cache,
            swa_widths=(self.head_dim,),
            swa_block_table=swa_md["block_table_tensor"],
            swa_pos_offset=swa_pos_offset,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            quant_group_size=_KV_QUANT_GROUP,
        )

    def _q_latent(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Recompute the normed q latent ``qr`` for the indexer.

        The fused ``wq_a``/``wkv`` stack is replicated, so its first
        ``q_lora_rank`` rows ARE the reference's ``wq_a``
        (``dsv4_ref/model.py:463``). Slicing rows of a parameter is a static
        shape, and the scale grid slices with it at 128-row granularity
        because ``q_lora_rank`` (1024) is a whole number of blocks.
        """
        rows = self.q_lora_rank
        latent = NF.block_fp8_linear(
            hidden_states,
            self.fused_wqa_wkv_weight[:rows],
            self.fused_wqa_wkv_scale[: _num_blocks(rows)],
            block_size=(128, 128),
            act_group_size=128,
            accum_dtype=torch.float32,
            out_dtype=torch.bfloat16,
            bias=None,
        )
        return _rms_norm(latent, self.q_norm_weight, self.eps).to(self.dtype)

    def _coarse_slots(
        self, slot_mapping: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Map raw per-token slots to compressed-slot destinations.

        A compressed cache holds one slot per ``compress_ratio`` tokens, so
        two things change relative to the raw mapping:

        1. Only the token that CLOSES a group writes
           (``dsv4_ref/model.py:350``); every other token is forced to
           ``PAD_SLOT_ID`` and skipped by the masked scatter.
        2. The destination is the raw slot divided by ``compress_ratio``,
           i.e. the page geometry re-read at the compressed cadence.

        NOT SETTLED BY THE REFERENCE: it indexes its compressed region
        directly by ``start_pos // ratio`` (``:380-382``) because it has no
        paging at all. In the port the runner owns the block table and has no
        notion of ``compress_ratio``, so it cannot emit a coarse mapping
        itself. This division is the localized assumption — change it here
        and every compressed write follows.
        """
        ratio = self.compress_ratio
        fires = torch.remainder(positions + 1, ratio) == 0
        coarse = slot_mapping // ratio
        return torch.where(
            fires & (slot_mapping > _PAD_SLOT_ID),
            coarse,
            torch.full_like(coarse, _PAD_SLOT_ID),
        )

    def _write_compressed_cache(
        self,
        compressed: torch.Tensor,
        latent_slots: torch.Tensor,
        rope_slots: torch.Tensor,
    ) -> None:
        """Write one compressed latent into the ``self_attn``/``.rope`` pairs.

        Column layout is the one :meth:`kv_layer_specs` documents: NoPE
        ``[0:224]`` and ``[224:448]`` into the ``self_attn`` pair, the 64
        RoPE columns into ``.rope``'s ``k_cache`` and the 7 group-64 scales
        into its ``v_cache``.
        """
        nope = self.nope_head_dim
        half = _LATENT_PAIR_HEAD_SIZE

        codes, scales = _quant_fp8_ue8m0(compressed[..., :nope])
        _masked_scatter_rows(self.latent_k_cache, latent_slots, codes[:, :half])
        _masked_scatter_rows(self.latent_v_cache, latent_slots, codes[:, half:])
        _masked_scatter_rows(self.rope_cache, rope_slots, compressed[..., nope:])
        _masked_scatter_rows(self.scale_cache, rope_slots, scales)
