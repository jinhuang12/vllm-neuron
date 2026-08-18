# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4 weight loaders
==========================

Checkpoint -> parameter transforms for ``deepseek-ai/DeepSeek-V4-Flash-0731``
(pinned revision ``7872f01b1d1fe23eabc4c98b48bffcef5a386062``, 72317 keys over
48 shards).

Why this module exists as its own file, and why it is this long: the
checkpoint's key convention, its two coexisting quantized weight formats and
its per-layer optional tensors are all *facts about the checkpoint*, not
facts about the math. Keeping them here means ``attention.py`` / ``moe.py`` /
``model.py`` never branch on checkpoint shape, and a checkpoint revision bump
is a one-file diff.

Primary evidence used (this outranks any derived spec):

* ``model.safetensors.index.json`` of the pinned revision -- the authoritative
  key inventory (which layers have which optional tensors).
* DeepSeek's own reference implementation shipped in the checkpoint repo:
  ``convert.py`` (the key-renaming convention, the FP4 value table, the
  nibble order) and ``model.py`` (every parameter's declared shape) and
  ``kernel.py`` (how the block scale is applied, hence its orientation).

Five facts that a "port it like llama3" reflex gets wrong, each enforced
below:

1. **Keys are NOT HF-standard.** There is no ``model.`` prefix and no
   ``self_attn``. DeepSeek's ``convert.py`` (lines 89-96) is the convention:
   strip ``model.``, ``self_attn`` -> ``attn``, ``mlp`` -> ``ffn``,
   ``weight_scale_inv`` -> ``scale``, ``e_score_correction_bias`` -> ``bias``.
   So the real keys are ``layers.N.attn.wq_a.weight``,
   ``layers.N.ffn.experts.E.w1.weight``, ``embed.weight``, ``head.weight``,
   ``norm.weight``, ``layers.N.attn_norm.weight``, ``layers.N.ffn_norm.weight``.
2. **Block scales live at ``.scale``, not ``.weight_scale_inv``**, and despite
   the ``_inv`` in the pre-rename name they are *multipliers*: the reference
   dequantizes with ``weight * scale`` (``convert.py:126``) and the GEMM
   multiplies the accumulator by them (``kernel.py:242-249``). They are stored
   ``float8_e8m0fnu`` and are converted here to an fp32 power of two.
3. **Weights are stored ``[out, in]``** (``model.py:145`` ``Linear`` declares
   ``empty(out_features, in_features)``), which is already the orientation
   ``NF.block_fp8_linear(x, weight_fp8[N_local, K_local], ...)`` wants. Do NOT
   pass ``is_storage_transposed=True`` for them. llama3 does pass it, because
   ITS parameters are ``[in, out]``; ours are not. Copying that flag over
   would transpose every projection and produce silent garbage.
4. **Routed experts are MXFP4-packed and are upcast to MXFP8 at load**, because
   Trainium2 has no FP4 datapath. The upcast happens once, on the host, not per
   forward.
5. **Optional tensors are genuinely optional.** ``ffn.gate.bias`` is absent on
   layers 0, 1, 2 and ``ffn.gate.tid2eid`` exists ONLY there; the KV compressor
   is absent on layers 0, 1; the DSA indexer exists only on ratio-4 layers; and
   the checkpoint carries three MTP blocks while the config declares one.
   Neither absence is an error, so nothing here asks the checkpoint for a key
   the index says is missing.

Public entry points (fixed by the family interface contract; everything else
in this module is private):

* :func:`attach_attention_loaders`
* :func:`attach_moe_loaders`
* :func:`attach_hash_context_loaders`
* :func:`build_checkpoint_mappings`
* :func:`load_block_scale_buffers`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch import nn

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    expert_parallel_grouped_loader,
    set_weight_loader,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    from .config import DeepseekV4Config

logger = logging.getLogger(__name__)

__all__ = [
    "attach_attention_loaders",
    "attach_moe_loaders",
    "attach_hash_context_loaders",
    "build_checkpoint_mappings",
    "load_block_scale_buffers",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Block extent of the 128x128 FP8 weight-scale grid. The checkpoint's
#: ``quantization_config.weight_block_size`` is ``[128, 128]``; the reference
#: ``Linear`` derives the scale shape as
#: ``(ceil(out/128), ceil(in/128))`` (dsv4_ref/model.py:146-148).
_BLOCK = 128

#: MX group size for the routed experts: one E8M0 scale per 32 FP4 elements
#: along K (dsv4_ref/model.py:139-143, and ``fp4_block_size = 32`` in
#: dsv4_ref/convert.py:26).
_MX_GROUP = 32

_FP8_DTYPE = torch.float8_e4m3fn

#: MX tile geometry for the Neuron MoE kernels. These four numbers are not
#: derivable from the model config -- they are the kernel's own tiling
#: constants, read off ``gpt_oss/weight_loaders_mxfp4.py:27-29``
#: (``PMAX = 128``, ``Q_WIDTH = 4``, ``Q_HEIGHT = 8``) and the 512-element K
#: tile the MoE entry points document (``functional/moe/moe_tkg.py:62,70-72``,
#: ``functional/moe/moe_cte.py:117-171``: ``gate_up`` is
#: ``[E, 128, 2, ceil(H/512), I]`` and ``down`` is
#: ``[E, I_p, ceil(I/512), H]``).
_MX_PMAX = 128
_MX_Q_WIDTH = 4
_MX_Q_HEIGHT = 8
_MX_TILE_K = 512

#: FP8 elements per machine word on the expert path: the MoE wrappers hand the
#: kernel a ``uint32`` view and reinterpret it as ``float8_e4m3fn_x4``
#: (``functional/moe/moe_tkg_wrapper.py``). The tiling therefore has to treat
#: four consecutive contraction-axis bytes as one atomic element.
_MX_BYTES_PER_WORD = 4

#: FP4 (E2M1) code -> value table, verbatim from DeepSeek's own
#: ``convert.py:11-14``. Note index 8 (the "negative zero" code) maps to
#: ``+0.0`` there, NOT ``-0.0``; we reproduce that exactly rather than
#: re-deriving the table from the OCP spec, so the upcast is bit-identical to
#: the reference conversion.
_FP4_TABLE: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

#: The same table as raw ``float8_e4m3fn`` byte patterns. Every FP4 magnitude
#: (0.5 .. 6.0) is exactly representable in E4M3, so the upcast is lossless --
#: which is why the reference calls its own variant
#: ``cast_e2m1fn_to_e4m3fn`` "losslessly" (convert.py:17-19).
#:
#: We keep the *bytes* (not the fp8 values) because several torch CPU builds
#: lack float8 kernels for ``index_select`` / ``stack``; gathering and stacking
#: in ``uint8`` and reinterpreting once at the end is bit-equivalent and always
#: available. Same reasoning as
#: ``llama3/weight_pack_mx_fp8.py:121-124`` ("route the gather through the byte
#: view, which is bit-equivalent").
_EXPECTED_FP4_FP8_BYTES: tuple[int, ...] = (
    0x00,
    0x30,
    0x38,
    0x3C,
    0x40,
    0x44,
    0x48,
    0x4C,
    0x00,
    0xB0,
    0xB8,
    0xBC,
    0xC0,
    0xC4,
    0xC8,
    0xCC,
)


def _build_fp4_to_fp8_bytes() -> torch.Tensor:
    """Derive the FP4-code -> FP8-byte table and cross-check the constant.

    Derived from :data:`_FP4_TABLE` through torch's own fp8 conversion, then
    asserted against :data:`_EXPECTED_FP4_FP8_BYTES`. Two independent
    derivations of the same 16 bytes is cheap insurance: a single transposed
    nibble in a hand-written table is invisible in review and shows up only as
    degraded output quality after a multi-thousand-second compile.
    """
    table = (
        torch.tensor(_FP4_TABLE, dtype=torch.float32).to(_FP8_DTYPE).view(torch.uint8)
    )
    expected = torch.tensor(_EXPECTED_FP4_FP8_BYTES, dtype=torch.uint8)
    if not torch.equal(table, expected):
        raise RuntimeError(
            "FP4->FP8 byte table mismatch: torch produced "
            f"{[hex(b) for b in table.tolist()]} but this module documents "
            f"{[hex(b) for b in expected.tolist()]}. One of the two is wrong; "
            "do not load experts until it is resolved."
        )
    return table


_FP4_TO_FP8_BYTES = _build_fp4_to_fp8_bytes()

#: Attribute name under which each attach function records
#: ``(attribute, checkpoint_keys, loader)`` triples on the module it decorates,
#: for :func:`load_block_scale_buffers` to replay. Recorded on the module
#: itself -- not in a module-level registry -- so nothing here is global state
#: and two model instances in one process cannot collide.
_SCALE_SOURCES_ATTR = "_dsv4_loader_sources"


# ---------------------------------------------------------------------------
# Sharding description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Shard:
    """The 2-D region of a ``[out, in]`` checkpoint weight this core owns.

    One rectangle covers every sharding case in the family contract:

    * replicated  -> the whole tensor,
    * column (N)  -> a row band, full K,
    * row (K)     -> a column band, full N,
    * ``wo_a``    -> a row band (its o-group) AND a column band (its head-slice
      of that group's K), which is why a single ``shard_dim`` is not enough and
      ``sharding_weight_loader`` cannot express it.

    The indices are resolved at attach time from the caller's explicit rank
    arguments, so every transform below is a closure over plain ints and
    ignores the ``rank`` the load pipeline passes. That is deliberate: contract
    parallelism shards these weights by ranks that are NOT the global TP rank
    (the o-proj group rank, the 16-way shared-expert subgroup rank, the EP
    rank). Baking the resolved index in is the explicit form of
    :func:`vllm_neuron.utils.weight_loader.with_rank_override`, and it keeps
    each transform a pure function of its inputs.
    """

    row_start: int
    row_size: int
    col_start: int
    col_size: int

    @classmethod
    def replicated(cls, out_dim: int, in_dim: int) -> "_Shard":
        return cls(0, out_dim, 0, in_dim)

    @classmethod
    def rows(cls, start: int, size: int, in_dim: int) -> "_Shard":
        return cls(start, size, 0, in_dim)

    @classmethod
    def cols(cls, out_dim: int, start: int, size: int) -> "_Shard":
        return cls(0, out_dim, start, size)

    @property
    def local_shape(self) -> tuple[int, int]:
        return (self.row_size, self.col_size)

    def key(self) -> tuple[slice, slice]:
        return (
            slice(self.row_start, self.row_start + self.row_size),
            slice(self.col_start, self.col_start + self.col_size),
        )


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------


def _shape_of(slice_obj: Any) -> tuple[int, ...]:
    """Shape of a ``PySafeSlice`` / ``SliceView`` / tensor, as a tuple."""
    get_shape = getattr(slice_obj, "get_shape", None)
    if get_shape is not None:
        return tuple(get_shape())
    return tuple(slice_obj.shape)


def _require_shape(
    param_name: str, ckpt_key: str, slice_obj: Any, expected: tuple[int, ...]
) -> None:
    """Fail loudly, naming parameter AND checkpoint key, on a shape mismatch.

    A wrong shape here is not recoverable downstream: it either raises far away
    from the cause (inside ``load_state_dict``) or, worse, silently loads a
    plausible-looking tensor. Both cost a full recompile to discover.
    """
    actual = _shape_of(slice_obj)
    if actual != tuple(expected):
        raise ValueError(
            f"Checkpoint shape mismatch for parameter {param_name!r} loading "
            f"from key {ckpt_key!r}: expected {tuple(expected)}, checkpoint has "
            f"{actual}."
        )


def _require_slice_count(param_name: str, slices: Sequence[Any], expected: int) -> None:
    if len(slices) != expected:
        raise ValueError(
            f"Loader for {param_name!r} expects {expected} checkpoint slice(s), "
            f"got {len(slices)}. Check this parameter's entry in "
            "build_checkpoint_mappings()."
        )


def _as_bytes(tensor: torch.Tensor, param_name: str, ckpt_key: str) -> torch.Tensor:
    """Reinterpret a 1-byte-per-element tensor as ``uint8`` (never a copy).

    Used for both the ``float8_e8m0fnu`` scales and the ``int8`` MXFP4 weights.
    ``view`` (a bitcast) is mandatory here: ``copy_`` / ``to`` on an
    ``float8_e8m0fnu`` source would *convert* the value instead of moving the
    exponent byte, which is the hazard upstream vLLM records at
    ``deepseek_v4.py:1476-1484``.
    """
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.element_size() != 1:
        raise ValueError(
            f"Cannot byte-view parameter {param_name!r} from key {ckpt_key!r}: "
            f"dtype {tensor.dtype} is {tensor.element_size()} bytes per element, "
            "expected 1."
        )
    return tensor.view(torch.uint8)


def _e8m0_to_fp32(
    scale: torch.Tensor, param_name: str, ckpt_key: str
) -> torch.Tensor:
    """Convert a stored ``float8_e8m0fnu`` scale to its fp32 multiplier.

    E8M0 stores *only* a biased exponent, so its value is ``2**(byte - 127)``.
    An fp32 with that same exponent field and a zero mantissa has exactly the
    same value, so moving the byte into the fp32 exponent field reproduces the
    multiplier bit-exactly and with no floating-point work::

        (s.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)

    This is the same identity the reference kernel uses in the other direction
    (``kernel.py:30-33``: ``fast_pow2(x) = reinterpret_float32((x + 127) << 23)``,
    i.e. the unbiased exponent plus the 127 bias, shifted into place), and it is
    byte-for-byte what upstream vLLM does
    (``deepseek_v4.py:518-519``, ``_ue8m0_uint8_to_float``).

    Verified against ``torch``'s own ``float8_e8m0fnu`` -> fp32 conversion for
    all 256 byte values: identical on codes 1..254, and different only on the
    two edge codes -- code 0 (torch: ``2**-127``; trick: ``0.0``) and code 255
    (torch: ``NaN``; trick: ``+inf``).

    Code 0 is ALLOWED and deliberately yields ``0.0``, because that is what the
    reference's own ``fast_pow2`` produces for it (``kernel.py:30-33``:
    ``(0) << 23`` is ``0.0``), and matching the reference's arithmetic exactly
    matters more here than matching torch's textbook decoding of a value that
    only ever labels an all-zero weight block.

    Code 255 is REJECTED: E8M0's all-ones code is NaN, and a NaN scale would
    silently poison every activation that touches the block instead of failing
    at load. That is precisely the failure this loader is written to avoid.

    The shift is done in ``int32`` (not ``uint32``) because some torch CPU
    builds lack a ``uint32`` left-shift kernel -- the same constraint noted in
    ``llama3/weight_pack_mx_fp8.py:59-72``. Values reach at most
    ``254 << 23``, well inside int32.
    """
    raw = _as_bytes(scale, param_name, ckpt_key)
    if bool((raw == 255).any()):
        raise ValueError(
            f"Block scale for {param_name!r} (key {ckpt_key!r}) contains the "
            "E8M0 code 255, which is NaN. Refusing to load: a NaN weight scale "
            "produces NaN activations with no other symptom."
        )
    return (raw.to(torch.int32) << 23).view(torch.float32)


def _grid_extent(size: int) -> int:
    """``ceil(size / 128)`` -- the scale-grid extent of a weight axis."""
    return (size + _BLOCK - 1) // _BLOCK


def _grid_shard(
    shard: _Shard, param_name: str, ckpt_key: str
) -> tuple[slice, slice]:
    """Map a weight shard onto the block-scale grid.

    A weight sharded along an axis takes the matching slice of the scale grid,
    divided by 128; a replicated weight takes the whole grid.

    Two admissible cases per axis, and one refusal:

    * **Block-aligned** (``start`` and ``size`` both multiples of 128): the
      exact matching grid slice. Every main-stack weight is this case.
    * **Contained sub-block** (the shard is finer than 128 but lies entirely
      inside ONE scale block): the containing block's single row/column, taken
      whole. The scale is then SHARED between the cores that split that block,
      not divided -- and sharing is exactly right, because
      ``NF.block_fp8_linear`` dequantizes every element of a block by that one
      scalar, so each core's dequantized rows are bit-identical to the
      corresponding rows of the full-tensor dequant. The DSpark stage-0
      ``main_proj`` needs this: LD-18 records it column-parallel at
      ``N_local = hidden_size / tp_size`` = 64 at TP=64, which is half a scale
      block, and cores ``2k``/``2k+1`` legitimately share scale row ``k``.
    * **Straddling** (finer than 128 AND crossing a block boundary): refused.
      Here the local grid genuinely cannot be expressed -- the core would need
      a partial of each of two blocks -- so we refuse rather than guess.

    The old form of this helper refused BOTH sub-block cases on the grounds
    that a sub-block shard "no longer matches what ``NF.block_fp8_linear``
    dequantizes". That reasoning holds only for the straddling case; a
    contained shard matches exactly. The guard is narrowed here, not removed.
    """

    def axis(start: int, size: int, label: str) -> slice:
        if start % _BLOCK == 0 and size % _BLOCK == 0:
            return slice(start // _BLOCK, (start + size) // _BLOCK)
        first = start // _BLOCK
        last = (start + size - 1) // _BLOCK
        if first != last:
            raise ValueError(
                f"Block-scale misalignment for parameter {param_name!r} "
                f"(checkpoint key {ckpt_key!r}): the {label} shard "
                f"[{start}, {start + size}) is finer than the {_BLOCK}-wide "
                f"scale block AND straddles blocks {first}..{last}, so this "
                "core would need a partial of each. A sub-block shard is only "
                "admissible when it lies entirely inside one block. Re-check "
                "this weight's row in the family contract's sharding table "
                "(and note the K_local % 128 == 0 invariant in "
                "DeepseekV4Config.block_fp8_linear_plan)."
            )
        return slice(first, first + 1)

    return (
        axis(shard.row_start, shard.row_size, "row (N)"),
        axis(shard.col_start, shard.col_size, "column (K)"),
    )


# ---------------------------------------------------------------------------
# Loader factories: unquantized (bf16 / fp32 / int64) tensors
# ---------------------------------------------------------------------------


def _replicated_loader(
    param_name: str, ckpt_key: str, expected_shape: tuple[int, ...]
) -> SafetensorsWeightLoader:
    """Load a tensor whole, validating its checkpoint shape.

    Used for the norms, the KV-compressor tensors, the indexer's
    ``weights_proj`` / ``wq_b``, the router gate, ``tid2eid`` and the
    hash-context parameters -- every tensor the contract marks replicated. The
    validating identity transform (rather than no loader at all) exists purely
    so a checkpoint-revision shape change is reported here, with the key name,
    instead of surfacing as a ``load_state_dict`` size error later.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank  # replicated: every core loads the same bytes
        _require_slice_count(param_name, slices, 1)
        _require_shape(param_name, ckpt_key, slices[0], expected_shape)
        return slices[0][:]

    return SafetensorsWeightLoader(transform=transform)


def _dim0_slice_loader(
    param_name: str,
    ckpt_key: str,
    expected_shape: tuple[int, ...],
    start: int,
    size: int,
) -> SafetensorsWeightLoader:
    """Load ``[start:start+size]`` along dim 0 of an unquantized tensor.

    Only user today is ``attn_sink`` ``[64]``, sliced to this core's query
    head(s): the reference declares it per *local* head
    (``dsv4_ref/model.py:462``, ``empty(self.n_local_heads)``), so a replicated
    load would apply another core's sink bias to this core's head.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank
        _require_slice_count(param_name, slices, 1)
        _require_shape(param_name, ckpt_key, slices[0], expected_shape)
        return slices[0][start : start + size]

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------
# Loader factories: block-128x128 FP8 weights and their scale grids
# ---------------------------------------------------------------------------


def _block_fp8_weight_loader(
    param_name: str,
    ckpt_key: str,
    full_shape: tuple[int, int],
    shard: _Shard,
) -> SafetensorsWeightLoader:
    """Load this core's rectangle of a ``[out, in]`` block-FP8 weight.

    No transpose: the storage orientation already matches
    ``NF.block_fp8_linear(x, weight_fp8[N_local, K_local], ...)``. See the
    module docstring, point 3.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank  # shard resolved at attach time; see _Shard's docstring
        _require_slice_count(param_name, slices, 1)
        _require_shape(param_name, ckpt_key, slices[0], full_shape)
        return slices[0][shard.key()]

    return SafetensorsWeightLoader(transform=transform)


def _block_fp8_scale_loader(
    param_name: str,
    ckpt_key: str,
    full_shape: tuple[int, int],
    shard: _Shard,
) -> SafetensorsWeightLoader:
    """Load this core's fp32 ``[N_local/128, K_local/128]`` block-scale grid.

    Emits fp32 (not the stored ``float8_e8m0fnu``) because that is exactly what
    ``NF.block_fp8_linear``'s ``weight_scale`` argument takes.
    """
    grid_shape = (_grid_extent(full_shape[0]), _grid_extent(full_shape[1]))
    grid_key = _grid_shard(shard, param_name, ckpt_key)
    # Derived FROM the resolved slices rather than recomputed from the shard
    # sizes: a contained sub-block shard yields one grid row/column that the
    # sharing cores take whole, so ``size // 128`` would say 0 (see
    # _grid_shard). Checked explicitly because a mis-sharded scale grid is not
    # a crash -- it is a wrong-numbers result found only after a
    # multi-thousand-second compile. At shared_expert_tp = 16 this pins w1/w3
    # to [1, 32] and w2 to [32, 1]; DSpark's main_proj at TP=64 to [1, 96].
    local_grid = (
        grid_key[0].stop - grid_key[0].start,
        grid_key[1].stop - grid_key[1].start,
    )

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank
        _require_slice_count(param_name, slices, 1)
        _require_shape(param_name, ckpt_key, slices[0], grid_shape)
        grid = _e8m0_to_fp32(slices[0][grid_key], param_name, ckpt_key)
        if tuple(grid.shape) != local_grid:
            raise ValueError(
                f"Local block-scale grid for {param_name!r} (key {ckpt_key!r}) "
                f"came out {tuple(grid.shape)}, expected {local_grid} from shard "
                f"rows [{shard.row_start}, {shard.row_start + shard.row_size}) x "
                f"cols [{shard.col_start}, {shard.col_start + shard.col_size})."
            )
        return grid

    return SafetensorsWeightLoader(transform=transform)


def _fused_block_fp8_weight_loader(
    param_name: str,
    ckpt_keys: Sequence[str],
    full_shapes: Sequence[tuple[int, int]],
    shards: Sequence[_Shard],
) -> SafetensorsWeightLoader:
    """Stack several block-FP8 weights along dim 0 (the output dim).

    ``fused_wqa_wkv`` is the only user: the checkpoint has no fused key, it has
    ``attn.wq_a`` ``[1024, 4096]`` and ``attn.wkv`` ``[512, 4096]``
    (``dsv4_ref/model.py:463-466`` declares them as two separate ``Linear``s),
    and this port fuses them into one ``[1536, 4096]`` parameter so the q and
    kv down-projections share a single GEMM.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank
        _require_slice_count(param_name, slices, len(ckpt_keys))
        parts = []
        for slice_obj, key, shape, shard in zip(
            slices, ckpt_keys, full_shapes, shards
        ):
            _require_shape(param_name, key, slice_obj, shape)
            parts.append(slice_obj[shard.key()])
        return torch.cat(parts, dim=0)

    return SafetensorsWeightLoader(transform=transform)


def _fused_block_fp8_scale_loader(
    param_name: str,
    ckpt_keys: Sequence[str],
    full_shapes: Sequence[tuple[int, int]],
    shards: Sequence[_Shard],
) -> SafetensorsWeightLoader:
    """Stack the fused weights' scale grids along dim 0.

    The grids concatenate exactly like the weights, because the fusion is along
    the N axis and each source's N is a whole number of 128-blocks:
    ``[8, 32]`` and ``[4, 32]`` become ``[12, 32]``.
    """
    grid_shapes = [(_grid_extent(n), _grid_extent(k)) for n, k in full_shapes]
    grid_keys = [
        _grid_shard(shard, param_name, key) for shard, key in zip(shards, ckpt_keys)
    ]

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank
        _require_slice_count(param_name, slices, len(ckpt_keys))
        parts = []
        for slice_obj, key, grid_shape, grid_key in zip(
            slices, ckpt_keys, grid_shapes, grid_keys
        ):
            _require_shape(param_name, key, slice_obj, grid_shape)
            parts.append(_e8m0_to_fp32(slice_obj[grid_key], param_name, key))
        return torch.cat(parts, dim=0)

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------
# Loader factories: routed experts (MXFP4 in the checkpoint -> MXFP8 here)
# ---------------------------------------------------------------------------


def _unpack_mxfp4_to_fp8_bytes(
    packed: torch.Tensor, param_name: str, ckpt_key: str, logical_k: int
) -> torch.Tensor:
    """Unpack ``[N, K/2]`` int8 MXFP4 into ``[N, K]`` fp8 *bytes*.

    Nibble order and value table are taken verbatim from DeepSeek's
    ``convert.py:30-33``::

        x = x.view(torch.uint8)
        low  = x & 0x0F
        high = (x >> 4) & 0x0F
        x = torch.stack([FP4_TABLE[low], FP4_TABLE[high]], dim=-1)...

    i.e. the LOW nibble is the even (first) element along K and the HIGH nibble
    the odd (second) one, and FP4 is packed along the K (last) axis -- also
    stated by the reference GEMM: "B is stored as [N, K//2] in
    float4_e2m1fn_x2 ... packed along the K (last) dimension"
    (``dsv4_ref/kernel.py:450-451``).

    The result is returned as ``uint8`` bytes, not as fp8, so the caller can
    stack experts without needing a float8 CPU kernel.
    """
    if packed.shape[1] * 2 != logical_k:
        raise ValueError(
            f"MXFP4 unpack for {param_name!r} (key {ckpt_key!r}): stored "
            f"{tuple(packed.shape)} implies logical K={packed.shape[1] * 2}, "
            f"expected {logical_k}."
        )
    raw = _as_bytes(packed, param_name, ckpt_key)
    low = _FP4_TO_FP8_BYTES[(raw & 0x0F).long()]
    high = _FP4_TO_FP8_BYTES[(raw >> 4).long()]
    # Interleave back to [low0, high0, low1, high1, ...] along K.
    return torch.stack([low, high], dim=-1).reshape(raw.shape[0], logical_k)


# ---------------------------------------------------------------------------
# MX tiling for the routed experts
#
# WHY THIS EXISTS AT ALL: ``DeepseekV4RoutedExperts`` reinterprets these
# parameters for the MoE kernels with *pure* ``view()`` calls -- no runtime
# repacking (``moe.py:274-348``). A ``view`` cannot reorder elements, so the
# loader must write the bytes ALREADY TILED. Get this wrong and nothing
# raises: the views still succeed and the kernel silently reads garbage.
#
# WHERE THE TRANSFORM COMES FROM: it is gpt_oss's, element for element --
# ``gpt_oss/weight_loaders_mxfp4.py`` ``_tile_gate_up_blocks`` (590-656),
# ``_tile_gate_up_scale`` (659-711), ``_tile_down_blocks`` (751-782) and
# ``_tile_down_scale`` (785-811), with the tile extents from
# ``_get_h_tiling_shard_i`` (814-832) and ``_get_i_tiling_shard_i`` (834-...).
# Two differences, both mechanical:
#
#   1. gpt_oss stores its expert weights ``[in, out]`` (its tiler's input is
#      ``[E, H/4, 2I]``) while this family stores ``[out, in]`` per contract
#      §1, so each transform starts by transposing to the reference's
#      orientation.
#   2. gpt_oss's element is a *word* that already packs 4 quantized values
#      along the contraction axis. Here the stored element is one FP8 byte, so
#      the 4-byte word is materialized as an explicit innermost axis that the
#      permutation carries along untouched. That keeps the packing implicit and
#      avoids a ``uint32`` view of a non-contiguous tensor.
#
# Verified at this family's dimensions: a gate/up parameter tiles to bytes
# that ``view`` as ``[E, 128, 1, H/512, I]`` uint32 (``[4, 128, 1, 8, 2048]``,
# concatenated to ``[4, 128, 2, 8, 2048]`` by ``moe.py``), its scale to
# ``[4, 16, 1, 8, 2048]`` uint8, and ``w2`` to ``[4, 128, 4, 4096]`` uint32
# with scale ``[4, 16, 4, 4096]`` uint8.
# ---------------------------------------------------------------------------


def _mx_h_tiling(h_size: int) -> tuple[int, int]:
    """``(num_H_tiles, q_blocks_per_H_tile)`` for a contraction axis of ``h_size``.

    Verbatim from ``gpt_oss/weight_loaders_mxfp4.py:829-830``.
    """
    return (
        h_size // (_MX_PMAX * _MX_Q_WIDTH),
        _MX_TILE_K // (_MX_Q_WIDTH * _MX_Q_HEIGHT),
    )


def _mx_i_tiling(i_size: int) -> tuple[int, int]:
    """``(num_I_tiles, q_blocks_per_I_tile)`` for a free axis of ``i_size``.

    Verbatim from ``gpt_oss/weight_loaders_mxfp4.py:834-880``: once the free
    axis exceeds one 512-element tile the tile count grows and each tile holds
    ``512/32`` quantization blocks; below that there is a single tile.
    """
    if i_size > _MX_TILE_K:
        return (i_size + _MX_TILE_K - 1) // _MX_TILE_K, _MX_TILE_K // _MX_GROUP
    return 1, i_size // _MX_GROUP


def _require_mx_divisible(
    param_name: str, ckpt_key: str, axis: str, size: int, factor: int
) -> None:
    """Fail loudly when an axis does not tile exactly.

    gpt_oss pads to the tile geometry; this family's dimensions
    (``H = 4096``, ``I = 2048``) divide it exactly, so padding is deliberately
    not implemented -- and an unpadded remainder would produce a
    silently-misaligned buffer rather than an error, which is exactly the
    failure mode this port cannot afford.
    """
    if size % factor != 0:
        raise ValueError(
            f"MX tiling for {param_name!r} (key {ckpt_key!r}): {axis} extent "
            f"{size} is not a multiple of {factor}. The tile geometry "
            f"(PMAX={_MX_PMAX}, Q_WIDTH={_MX_Q_WIDTH}, "
            f"Q_HEIGHT={_MX_Q_HEIGHT}, K tile={_MX_TILE_K}) requires exact "
            "division; padding is not implemented for this family."
        )


def _tile_mx_gate_up_weight(
    byte_w: torch.Tensor, param_name: str, ckpt_key: str
) -> torch.Tensor:
    """Tile stacked gate/up expert bytes ``[E, I, H]`` in place of a ``view``.

    ``H`` is the contraction axis and is the packed one (4 bytes per word).
    """
    e_size, i_size, h_size = byte_w.shape
    _require_mx_divisible(param_name, ckpt_key, "H", h_size, _MX_PMAX * _MX_Q_WIDTH)
    _require_mx_divisible(param_name, ckpt_key, "I", i_size, _MX_TILE_K)
    num_h_tiles, qb_h = _mx_h_tiling(h_size)
    num_i_tiles, qb_i = _mx_i_tiling(i_size)

    # -> gpt_oss's orientation, with the 4-byte word as an innermost axis:
    # [E, H/4, I, 4].
    words = (
        byte_w.permute(0, 2, 1)
        .reshape(e_size, h_size // _MX_BYTES_PER_WORD, _MX_BYTES_PER_WORD, i_size)
        .permute(0, 1, 3, 2)
    )
    # [E, nHt, qbH, QH_h, gate_up=1, nIt, qbI, QH_i, QW_i, bytes]
    words = words.reshape(
        e_size,
        num_h_tiles,
        qb_h,
        _MX_Q_HEIGHT,
        1,
        num_i_tiles,
        qb_i,
        _MX_Q_HEIGHT,
        _MX_Q_WIDTH,
        _MX_BYTES_PER_WORD,
    )
    # gpt_oss's permutation (639-650), plus the trailing byte axis. The gate/up
    # axis is 1 wide here because the contract keeps w1 and w3 as separate
    # parameters; ``moe.py`` concatenates them on that axis.
    words = words.permute(0, 2, 3, 4, 1, 5, 8, 6, 7, 9)
    return words.reshape(e_size, i_size, h_size)


def _tile_mx_gate_up_scale(
    byte_s: torch.Tensor, param_name: str, ckpt_key: str
) -> torch.Tensor:
    """Tile stacked gate/up expert scale bytes ``[E, I, H/32]``."""
    e_size, i_size, groups = byte_s.shape
    h_size = groups * _MX_GROUP
    _require_mx_divisible(param_name, ckpt_key, "H", h_size, _MX_PMAX * _MX_Q_WIDTH)
    _require_mx_divisible(param_name, ckpt_key, "I", i_size, _MX_TILE_K)
    num_h_tiles, qb_h = _mx_h_tiling(h_size)
    num_i_tiles, qb_i = _mx_i_tiling(i_size)

    # -> [E, H/32, I], then gpt_oss's reshape/permute (673-710). No byte axis:
    # a scale IS one byte per 32 contraction elements.
    scale = byte_s.permute(0, 2, 1).reshape(
        e_size,
        num_h_tiles,
        qb_h,
        1,
        num_i_tiles,
        qb_i,
        _MX_Q_HEIGHT,
        _MX_Q_WIDTH,
    )
    scale = scale.permute(0, 2, 3, 1, 4, 7, 5, 6)
    return scale.reshape(e_size, i_size, groups)


def _tile_mx_down_weight(
    byte_w: torch.Tensor, param_name: str, ckpt_key: str
) -> torch.Tensor:
    """Tile stacked down-projection expert bytes ``[E, H, I]``.

    Here ``I`` is the contraction (and packed) axis and ``H`` the free axis,
    which the kernel additionally shuffles in blocks of 4
    (``_tile_down_blocks``, gpt_oss:764-780).
    """
    e_size, h_size, i_size = byte_w.shape
    _require_mx_divisible(param_name, ckpt_key, "I", i_size, _MX_TILE_K)
    _require_mx_divisible(param_name, ckpt_key, "H", h_size, _MX_BYTES_PER_WORD)
    num_i_tiles, qb_i = _mx_i_tiling(i_size)

    # -> [E, I/4, H, 4]
    words = (
        byte_w.permute(0, 2, 1)
        .reshape(e_size, i_size // _MX_BYTES_PER_WORD, _MX_BYTES_PER_WORD, h_size)
        .permute(0, 1, 3, 2)
    )
    words = words.reshape(
        e_size,
        num_i_tiles,
        qb_i,
        _MX_Q_HEIGHT,
        h_size // _MX_BYTES_PER_WORD,
        _MX_BYTES_PER_WORD,
        _MX_BYTES_PER_WORD,
    )
    # gpt_oss:777 permute(0, 2, 3, 1, 5, 4, 6), with our byte axis in the slot
    # its ``q_packed`` (size 1) occupied.
    words = words.permute(0, 2, 3, 1, 5, 4, 6)
    return words.reshape(e_size, h_size, i_size)


def _tile_mx_down_scale(
    byte_s: torch.Tensor, param_name: str, ckpt_key: str
) -> torch.Tensor:
    """Tile stacked down-projection scale bytes ``[E, H, I/32]``."""
    e_size, h_size, groups = byte_s.shape
    i_size = groups * _MX_GROUP
    _require_mx_divisible(param_name, ckpt_key, "I", i_size, _MX_TILE_K)
    _require_mx_divisible(param_name, ckpt_key, "H", h_size, _MX_BYTES_PER_WORD)
    num_i_tiles, qb_i = _mx_i_tiling(i_size)

    # -> [E, I/32, H], then gpt_oss:797-809.
    scale = byte_s.permute(0, 2, 1).reshape(
        e_size,
        num_i_tiles,
        qb_i,
        h_size // _MX_BYTES_PER_WORD,
        _MX_BYTES_PER_WORD,
    )
    scale = scale.permute(0, 2, 1, 4, 3)
    return scale.reshape(e_size, h_size, groups)


#: Which tiling pair a routed-expert leaf uses. ``w1``/``w3`` contract over
#: ``H``; ``w2`` contracts over ``I``.
_MX_TILERS = {
    "gate_up": (_tile_mx_gate_up_weight, _tile_mx_gate_up_scale),
    "down": (_tile_mx_down_weight, _tile_mx_down_scale),
}


def _mxfp4_expert_weight_loader(
    param_name: str,
    local_keys: Sequence[str],
    out_dim: int,
    logical_k: int,
    kind: str,
) -> SafetensorsWeightLoader:
    """Upcast this core's local experts from MXFP4 to MXFP8 ``[E_local, N, K]``.

    Why upcast at load instead of at runtime: Trainium2 has no FP4 datapath, so
    the expert GEMM must see ``float8_e4m3fn`` elements. Every FP4 magnitude is
    exactly representable in E4M3, so this costs accuracy nothing (the
    reference calls the same direction "lossless",
    ``dsv4_ref/convert.py:17-19``) and it costs 2x expert bytes, which the port
    plan already accounts for.

    The group-32 E8M0 scales are NOT folded in here -- they are carried through
    unchanged by :func:`_mx_expert_scale_loader`, so the served format stays
    MXFP8 group-32 (what the Neuron MX expert kernels take). This is where this
    port deliberately differs from the reference's own
    ``--expert-dtype fp8`` conversion, which instead re-blocks the scales into
    a 128x128 grid for its own fp8 GEMM (``dsv4_ref/convert.py:35-52``).

    Only the ``ep_degree``-local experts are ever materialized: the EP wrapper
    restricts ``slices`` to this core's contiguous expert range before this
    transform runs (4 of 256 at EP=64).

    The result is written ALREADY TILED for the MX MoE kernels -- see the
    tiling section above for why a ``view``-only consumer forces that.
    """
    expected = (out_dim, logical_k // 2)
    tile_weight, _ = _MX_TILERS[kind]

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank  # experts are selected by EP rank, baked in at attach time
        _require_slice_count(param_name, slices, len(local_keys))
        per_expert = []
        for slice_obj, key in zip(slices, local_keys):
            _require_shape(param_name, key, slice_obj, expected)
            per_expert.append(
                _unpack_mxfp4_to_fp8_bytes(slice_obj[:], param_name, key, logical_k)
            )
        # Stack as bytes, tile, then reinterpret once: torch CPU builds without
        # float8 kernels can still run this whole path.
        stacked = torch.stack(per_expert, dim=0)
        tiled = tile_weight(stacked, param_name, local_keys[0])
        return tiled.contiguous().view(_FP8_DTYPE)

    return SafetensorsWeightLoader(transform=transform)


def _mx_expert_scale_loader(
    param_name: str,
    local_keys: Sequence[str],
    out_dim: int,
    logical_k: int,
    kind: str,
) -> SafetensorsWeightLoader:
    """Stack this core's local experts' group-32 E8M0 scales, values unchanged.

    Shape per expert is ``[N, K/32]`` (``dsv4_ref/model.py:139-143``: "Scale is
    [out, in//32] in float8_e8m0fnu (1 scale per 32 fp4 elements along K)").
    They stay raw ``uint8`` exponent bytes -- the MX kernels consume E8M0
    directly, and converting them to an fp32 multiplier here (which is what the
    *block*-FP8 path does) would both quadruple their size and destroy the
    format the kernel expects. ``moe.py`` declares ``experts.*_scale`` as
    ``uint8`` and ``shared_expert.*_scale`` as fp32 for exactly this reason.

    Only the element ORDER changes: the scales are tiled with the same
    permutation as the weights they scale, so every scale stays paired with its
    32 elements.
    """
    expected = (out_dim, logical_k // _MX_GROUP)
    _, tile_scale = _MX_TILERS[kind]

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank
        _require_slice_count(param_name, slices, len(local_keys))
        per_expert = []
        for slice_obj, key in zip(slices, local_keys):
            _require_shape(param_name, key, slice_obj, expected)
            per_expert.append(_as_bytes(slice_obj[:], param_name, key))
        stacked = torch.stack(per_expert, dim=0)
        return tile_scale(stacked, param_name, local_keys[0]).contiguous()

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------
# Binding helper
# ---------------------------------------------------------------------------


def _bind(
    module: nn.Module,
    attr: str,
    ckpt_keys: Sequence[str],
    loader: SafetensorsWeightLoader,
    *,
    required: bool = True,
) -> bool:
    """Attach ``loader`` to ``module.<attr>`` and record its checkpoint keys.

    Two-part on purpose:

    * ``set_weight_loader`` on the tensor is what the pipelined
      ``named_parameters()`` load path reads.
    * the recorded ``(attr, keys, loader)`` triple is what
      :func:`load_block_scale_buffers` replays for anything registered as a
      *buffer* instead of a parameter -- that path cannot go through
      ``set_weight_loader``, because a buffer declared ``None`` has no tensor to
      hang the loader on yet (same problem llama3 solves with its
      ``_FP8_*_SCALE_SOURCES`` tables).

    Returns whether the attribute existed. ``required=False`` is for the
    genuinely optional tensors (compressor, indexer, ``gate_bias``,
    ``tid2eid``).

    ``attr`` may be dotted (``"attn_norm.weight"``): the loader is then hung on
    the leaf tensor and recorded against its *owning* submodule, so
    :func:`load_block_scale_buffers`'s ``named_modules()`` walk still yields the
    correct qualified name. Needed because the decoder layer owns its norms as
    ``DeepseekV4RMSNorm`` submodules, not as flat ``*_norm_weight`` tensors.
    """
    if "." in attr:
        owner_path, _, leaf = attr.rpartition(".")
        owner: nn.Module | None = module
        for part in owner_path.split("."):
            owner = getattr(owner, part, None)
            if owner is None:
                break
        if not isinstance(owner, nn.Module):
            if required:
                raise AttributeError(
                    f"{type(module).__name__} has no submodule {owner_path!r} to "
                    f"bind {attr!r} on; the family interface contract fixes this "
                    "name and the weight loaders bind to it."
                )
            return False
        return _bind(owner, leaf, ckpt_keys, loader, required=required)

    if not hasattr(module, attr):
        if required:
            raise AttributeError(
                f"{type(module).__name__} has no attribute {attr!r}; the family "
                "interface contract fixes this name and the weight loaders bind "
                "to it. Either the module was renamed or the loader call is on "
                "the wrong module."
            )
        return False

    current = getattr(module, attr)
    if isinstance(current, torch.Tensor):
        set_weight_loader(current, loader)

    sources = list(getattr(module, _SCALE_SOURCES_ATTR, ()))
    sources.append((attr, tuple(ckpt_keys), loader))
    setattr(module, _SCALE_SOURCES_ATTR, tuple(sources))
    return True


# ---------------------------------------------------------------------------
# Checkpoint key helpers
# ---------------------------------------------------------------------------


def _layer_key_prefix(layer_idx: int) -> str:
    """Checkpoint prefix for one main-stack transformer block.

    This is the MAIN-STACK spelling only. The DSpark draft stages live under
    ``mtp.{s}`` and are reached by passing ``key_prefix="mtp.{s}"`` to the
    attach functions, not by index arithmetic here: the drafter's stage index
    and the ``layer_idx`` it hands the MoE (``num_hidden_layers + stage``, so
    ``is_hash_moe_layer`` is False and the ``gate.bias`` branch is taken, which
    is what the ``mtp.*`` key census shows) are deliberately different numbers,
    and folding them together here would make one of the two wrong.
    """
    return f"layers.{layer_idx}"


def _resolve_layer_idx(module: nn.Module, kind: str) -> int:
    """Read ``module.layer_idx``, which every family module carries.

    Fixed by the contract's constructors --
    ``DeepseekV4Attention(config, layer_idx, ...)`` and
    ``DeepseekV4MoE(config, layer_idx, ...)`` -- and needed by the attach
    functions whose signature has no ``layer_idx`` of its own, because the
    checkpoint key for every tensor is layer-qualified.
    """
    layer_idx = getattr(module, "layer_idx", None)
    if not isinstance(layer_idx, int):
        raise AttributeError(
            f"{kind} loader needs an integer {type(module).__name__}.layer_idx to "
            "build layer-qualified checkpoint keys; found "
            f"{layer_idx!r}. Store the constructor's layer_idx on the module."
        )
    return layer_idx


# ---------------------------------------------------------------------------
# Public: attention
# ---------------------------------------------------------------------------


def attach_attention_loaders(
    module: nn.Module,
    config: "DeepseekV4Config",
    *,
    tp_size: int,
    tp_rank: int,
    group_rank: int,
    group_size: int,
    shared_tp_size: int,
    shared_tp_rank: int,
    key_prefix: str | None = None,
) -> None:
    """Attach every checkpoint loader a :class:`DeepseekV4Attention` needs.

    >>> PARALLELISM (contract sharding table, verified against the reference's
    own parallel Linears) <<<

    ===================== ============================== =================
    parameter             sharding                       local at TP=64
    ===================== ============================== =================
    ``fused_wqa_wkv``     replicated                     ``[1536, 4096]``
    ``wq_b``              column over query heads        ``[512, 1024]``
    ``wo_a``              one o-group per ``tp/o_groups`` ``[1024, 512]``
                          cores, each owning its
                          head-slice of that group's K
    ``wo_b``              row over K                     ``[4096, 128]``
    ``attn_sink``         head slice                     ``[1]``
    ``q_norm``/``kv_norm`` replicated                    as checkpoint
    compressor (all)      replicated                     as checkpoint
    indexer ``wq_b``      replicated                     ``[8192, 1024]``
    indexer ``weights_proj`` replicated                  ``[64, 4096]``
    ===================== ============================== =================

    ``wo_a`` is the only two-axis shard. The reference makes the layout
    explicit: it declares ``wo_a`` as
    ``ColumnParallelLinear(n_heads * head_dim // n_groups, n_groups *
    o_lora_rank)`` and then consumes it as
    ``wo_a.weight.view(n_local_groups, o_lora_rank, -1)``
    (``dsv4_ref/model.py:468`` and ``:543``). So dim 0 indexes
    ``(o_group, o_lora_rank)`` and dim 1 indexes
    ``(head_within_group, kv_lora_rank)``. With one query head per core at
    TP=64, core ``r`` owns head ``r``, hence o-group ``r // heads_per_group``
    and K-slice ``r % heads_per_group``.

    Args:
        module: The :class:`DeepseekV4Attention` instance.
        config: The family config (dims and per-layer structure).
        tp_size / tp_rank: Global attention tensor-parallel degree and rank.
        group_rank / group_size: This core's rank inside, and the size of, the
            o-projection process group (the group that all-reduces the grouped
            o-proj). ``group_size`` must be ``tp_size // config.o_groups``.
        shared_tp_size / shared_tp_rank: Accepted for signature symmetry with
            :func:`attach_moe_loaders` (the contract fixes one attach
            signature shape for both); attention has no shared-expert weights,
            so they are only validated.
    """
    if shared_tp_size <= 0 or not (0 <= shared_tp_rank < shared_tp_size):
        raise ValueError(
            f"attach_attention_loaders: invalid shared_tp_rank={shared_tp_rank} "
            f"for shared_tp_size={shared_tp_size}."
        )

    layer_idx = _resolve_layer_idx(module, "attention")
    # ``key_prefix`` overrides the main-stack spelling so the DSpark stages can
    # bind the same parameter set under ``mtp.{s}`` (the checkpoint's own
    # namespace) while still reporting a layer_idx to the sharding math.
    prefix = _layer_key_prefix(layer_idx) if key_prefix is None else key_prefix

    hidden = config.hidden_size
    q_lora = config.q_lora_rank
    kv_lora = config.kv_lora_rank
    o_lora = config.o_lora_rank
    o_groups = config.o_groups
    num_heads = config.num_attention_heads

    if num_heads % tp_size != 0:
        raise ValueError(
            f"attach_attention_loaders: num_attention_heads={num_heads} must be "
            f"divisible by tp_size={tp_size} (one contiguous head band per core)."
        )
    heads_per_rank = num_heads // tp_size
    heads_per_group = num_heads // o_groups
    if heads_per_group % heads_per_rank != 0:
        raise ValueError(
            f"attach_attention_loaders: each core's {heads_per_rank} head(s) must "
            f"sit inside one o-group of {heads_per_group} heads; "
            f"tp_size={tp_size} with o_groups={o_groups} splits a core across "
            "groups, which the grouped o-projection cannot express."
        )
    expected_group_size = tp_size // o_groups
    if group_size != expected_group_size:
        raise ValueError(
            f"attach_attention_loaders: group_size={group_size} but "
            f"tp_size // o_groups = {expected_group_size}. The o-projection "
            "group must hold exactly the cores that share one o-group."
        )
    if group_rank != tp_rank % group_size:
        # The head->group->K mapping above is derived from the *global* head
        # index, because wq_b is a plain contiguous column shard by tp_rank. If
        # the o-proj process group was built from a non-contiguous rank mesh
        # (e.g. functional/process_groups.py's TRN2_8x8_MESH rows), its
        # rank_in_group no longer matches the head ordering, and wq_b's head
        # assignment would have to be re-derived from the same ordering.
        # Refuse rather than shard one of the two the wrong way.
        raise ValueError(
            f"attach_attention_loaders: group_rank={group_rank} != "
            f"tp_rank % group_size = {tp_rank % group_size}. The o-projection "
            "group's rank order must match the contiguous query-head order that "
            "wq_b is sharded by. If the group comes from a non-contiguous mesh, "
            "wq_b's head slice must be re-derived from that same ordering."
        )

    global_head = tp_rank * heads_per_rank
    o_group_index = global_head // heads_per_group
    head_in_group = global_head % heads_per_group

    # --- fused q/kv down-projection (two checkpoint tensors, one parameter) --
    wqa_shape = (q_lora, hidden)
    wkv_shape = (kv_lora, hidden)
    fused_keys = (f"{prefix}.attn.wq_a.weight", f"{prefix}.attn.wkv.weight")
    fused_scale_keys = (f"{prefix}.attn.wq_a.scale", f"{prefix}.attn.wkv.scale")
    fused_shards = (
        _Shard.replicated(*wqa_shape),
        _Shard.replicated(*wkv_shape),
    )
    _bind(
        module,
        "fused_wqa_wkv_weight",
        fused_keys,
        _fused_block_fp8_weight_loader(
            "fused_wqa_wkv_weight", fused_keys, (wqa_shape, wkv_shape), fused_shards
        ),
    )
    _bind(
        module,
        "fused_wqa_wkv_scale",
        fused_scale_keys,
        _fused_block_fp8_scale_loader(
            "fused_wqa_wkv_scale",
            fused_scale_keys,
            (wqa_shape, wkv_shape),
            fused_shards,
        ),
    )

    # --- q up-projection: column-parallel over query heads -------------------
    wq_b_shape = (num_heads * kv_lora, q_lora)
    wq_b_shard = _Shard.rows(
        global_head * kv_lora, heads_per_rank * kv_lora, q_lora
    )
    _bind(
        module,
        "wq_b_weight",
        (f"{prefix}.attn.wq_b.weight",),
        _block_fp8_weight_loader(
            "wq_b_weight", f"{prefix}.attn.wq_b.weight", wq_b_shape, wq_b_shard
        ),
    )
    _bind(
        module,
        "wq_b_scale",
        (f"{prefix}.attn.wq_b.scale",),
        _block_fp8_scale_loader(
            "wq_b_scale", f"{prefix}.attn.wq_b.scale", wq_b_shape, wq_b_shard
        ),
    )

    # --- grouped o-projection stage A: row band (o-group) x column band (head)
    wo_a_shape = (o_groups * o_lora, heads_per_group * kv_lora)
    wo_a_shard = _Shard(
        row_start=o_group_index * o_lora,
        row_size=o_lora,
        col_start=head_in_group * kv_lora,
        col_size=heads_per_rank * kv_lora,
    )
    _bind(
        module,
        "wo_a_weight",
        (f"{prefix}.attn.wo_a.weight",),
        _block_fp8_weight_loader(
            "wo_a_weight", f"{prefix}.attn.wo_a.weight", wo_a_shape, wo_a_shard
        ),
    )
    _bind(
        module,
        "wo_a_scale",
        (f"{prefix}.attn.wo_a.scale",),
        _block_fp8_scale_loader(
            "wo_a_scale", f"{prefix}.attn.wo_a.scale", wo_a_shape, wo_a_shard
        ),
    )

    # --- grouped o-projection stage B: row-parallel over K -------------------
    wo_b_shape = (hidden, o_groups * o_lora)
    if wo_b_shape[1] % tp_size != 0:
        raise ValueError(
            f"attach_attention_loaders: wo_b K={wo_b_shape[1]} is not divisible "
            f"by tp_size={tp_size}."
        )
    wo_b_k_local = wo_b_shape[1] // tp_size
    wo_b_shard = _Shard.cols(hidden, tp_rank * wo_b_k_local, wo_b_k_local)
    _bind(
        module,
        "wo_b_weight",
        (f"{prefix}.attn.wo_b.weight",),
        _block_fp8_weight_loader(
            "wo_b_weight", f"{prefix}.attn.wo_b.weight", wo_b_shape, wo_b_shard
        ),
    )
    _bind(
        module,
        "wo_b_scale",
        (f"{prefix}.attn.wo_b.scale",),
        _block_fp8_scale_loader(
            "wo_b_scale", f"{prefix}.attn.wo_b.scale", wo_b_shape, wo_b_shard
        ),
    )

    # --- latent norms (replicated) and the per-head attention sink ----------
    _bind(
        module,
        "q_norm_weight",
        (f"{prefix}.attn.q_norm.weight",),
        _replicated_loader(
            "q_norm_weight", f"{prefix}.attn.q_norm.weight", (q_lora,)
        ),
    )
    _bind(
        module,
        "kv_norm_weight",
        (f"{prefix}.attn.kv_norm.weight",),
        _replicated_loader(
            "kv_norm_weight", f"{prefix}.attn.kv_norm.weight", (kv_lora,)
        ),
    )
    _bind(
        module,
        "attn_sink",
        (f"{prefix}.attn.attn_sink",),
        _dim0_slice_loader(
            "attn_sink",
            f"{prefix}.attn.attn_sink",
            (num_heads,),
            global_head,
            heads_per_rank,
        ),
    )

    # --- KV compressor: unquantized, replicated, ABSENT on layers 0 and 1 ---
    # The checkpoint index confirms the absence: layers 0 and 1 carry no
    # attn.compressor.* keys at all (they are the SWA-only layers,
    # compress_ratios[0:2] == [0, 0]). Asking for those keys would make the
    # loader raise on a checkpoint that is perfectly well-formed.
    if config.has_compressed_cache(layer_idx):
        ratio = config.compress_ratio(layer_idx)
        _attach_compressor_loaders(
            getattr(module, "compressor", None),
            key_prefix=f"{prefix}.attn.compressor",
            name_prefix="compressor",
            hidden=hidden,
            head_dim=kv_lora,
            compress_ratio=ratio,
            norm_dim=kv_lora,
        )

    # --- DSA lightning indexer: ratio-4 layers only -------------------------
    if config.has_indexer(layer_idx):
        indexer = getattr(module, "indexer", None)
        if indexer is None:
            raise AttributeError(
                f"layer {layer_idx} is a ratio-4 layer and its checkpoint carries "
                f"{prefix}.attn.indexer.*, but the attention module has no "
                "'indexer' submodule."
            )
        idx_heads = config.index_n_heads
        idx_dim = config.index_head_dim
        idx_wq_b_shape = (idx_heads * idx_dim, q_lora)
        # >>> PARALLELISM: replicated -- upstream builds this as a
        # ReplicatedLinear, so every core computes the full indexer logits. <<<
        idx_wq_b_shard = _Shard.replicated(*idx_wq_b_shape)
        _bind(
            indexer,
            "wq_b_weight",
            (f"{prefix}.attn.indexer.wq_b.weight",),
            _block_fp8_weight_loader(
                "indexer.wq_b_weight",
                f"{prefix}.attn.indexer.wq_b.weight",
                idx_wq_b_shape,
                idx_wq_b_shard,
            ),
        )
        _bind(
            indexer,
            "wq_b_scale",
            (f"{prefix}.attn.indexer.wq_b.scale",),
            _block_fp8_scale_loader(
                "indexer.wq_b_scale",
                f"{prefix}.attn.indexer.wq_b.scale",
                idx_wq_b_shape,
                idx_wq_b_shard,
            ),
        )
        _bind(
            indexer,
            "weights_proj_weight",
            (f"{prefix}.attn.indexer.weights_proj.weight",),
            _replicated_loader(
                "indexer.weights_proj_weight",
                f"{prefix}.attn.indexer.weights_proj.weight",
                (idx_heads, hidden),
            ),
        )
        _attach_compressor_loaders(
            getattr(indexer, "compressor", None),
            key_prefix=f"{prefix}.attn.indexer.compressor",
            name_prefix="indexer.compressor",
            hidden=hidden,
            head_dim=idx_dim,
            compress_ratio=4,
            norm_dim=idx_dim,
        )


def _attach_compressor_loaders(
    compressor: nn.Module | None,
    *,
    key_prefix: str,
    name_prefix: str,
    hidden: int,
    head_dim: int,
    compress_ratio: int,
    norm_dim: int,
) -> None:
    """Attach the four replicated KV-compressor loaders.

    Shapes follow the reference ``Compressor`` (``dsv4_ref/model.py:289-305``):
    ``coff = 1 + (compress_ratio == 4)`` (the ratio-4 compressor keeps an extra
    overlapping window, hence double width), so

    * ``ape``   -> ``[compress_ratio, coff * head_dim]`` fp32,
    * ``wkv`` / ``wgate`` -> ``[coff * head_dim, hidden]`` bf16, unquantized
      (the reference builds them with ``dtype=float32``, i.e. the no-scale
      branch of its ``Linear``),
    * ``norm``  -> ``[head_dim]``.

    All replicated: the compressor runs identically on every core.
    """
    if compressor is None:
        raise AttributeError(
            f"checkpoint carries {key_prefix}.* but the module has no "
            f"'{name_prefix}' submodule."
        )
    coff = 2 if compress_ratio == 4 else 1
    proj_shape = (coff * head_dim, hidden)
    for attr, suffix, shape in (
        ("wkv_weight", "wkv.weight", proj_shape),
        ("wgate_weight", "wgate.weight", proj_shape),
        ("ape", "ape", (compress_ratio, coff * head_dim)),
        ("norm_weight", "norm.weight", (norm_dim,)),
    ):
        key = f"{key_prefix}.{suffix}"
        _bind(
            compressor,
            attr,
            (key,),
            _replicated_loader(f"{name_prefix}.{attr}", key, shape),
        )


# ---------------------------------------------------------------------------
# Public: MoE
# ---------------------------------------------------------------------------


def attach_moe_loaders(
    module: nn.Module,
    config: "DeepseekV4Config",
    *,
    layer_idx: int,
    tp_size: int,
    tp_rank: int,
    ep_degree: int,
    ep_rank: int,
    shared_tp_size: int,
    shared_tp_rank: int,
    key_prefix: str | None = None,
) -> None:
    """Attach every checkpoint loader a :class:`DeepseekV4MoE` needs.

    >>> PARALLELISM <<<

    * **Router** (``gate_weight``, ``gate_bias``, ``tid2eid``): replicated.
      Every core must score every expert before dispatch.
    * **Routed experts**: expert-parallel, ``n_routed_experts // ep_degree``
      local experts (4 of 256 at EP=64). Weights are upcast MXFP4 -> MXFP8; the
      group-32 E8M0 scales pass through unchanged.
    * **Shared expert**: sharded over a ``shared_tp_size``-way subgroup and
      replicated across the remaining subgroups (16-way with 4-fold replication
      at TP=64). A 64-way split would give ``K_local = 32`` on ``w2``, which
      breaks the 128-element activation group the block-FP8 path quantizes over
      -- see the note on ``DeepseekV4Config.shared_expert_tp``.

    ``gate_bias`` is bound only on the 40 non-hash layers and ``tid2eid`` only
    on the 3 hash layers (0, 1, 2). The checkpoint index is unambiguous: those
    three layers carry ``ffn.gate.tid2eid`` and no ``ffn.gate.bias``, the other
    40 carry ``ffn.gate.bias`` and no ``tid2eid``, and the reference declares
    exactly that either/or (``dsv4_ref/model.py:562-567``). Neither absence is
    an error.

    Args:
        module: The :class:`DeepseekV4MoE` instance.
        config: The family config.
        layer_idx: Index of the owning decoder layer. Required, not read off the
            module, because every checkpoint key here is layer-qualified and
            because which router tensor exists (``tid2eid`` vs ``bias``) depends
            on it.
        tp_size / tp_rank: Global tensor-parallel degree and rank. Only used for
            validation here: no MoE tensor is sharded on the global TP axis
            (the routed experts use the EP axis, the shared expert its own
            subgroup).
        ep_degree / ep_rank: Expert-parallel degree and this core's EP rank.
        shared_tp_size / shared_tp_rank: The shared expert's subgroup size and
            this core's rank inside it.
    """
    if tp_size <= 0 or not (0 <= tp_rank < tp_size):
        raise ValueError(
            f"attach_moe_loaders: invalid tp_rank={tp_rank} for tp_size={tp_size}."
        )

    if not isinstance(layer_idx, int):
        raise TypeError(
            f"attach_moe_loaders: layer_idx must be an int, got {layer_idx!r}."
        )
    # See ``_layer_key_prefix``: the DSpark stages pass ``mtp.{s}`` while still
    # handing the hash/no-hash decision the out-of-range ``layer_idx``.
    prefix = _layer_key_prefix(layer_idx) if key_prefix is None else key_prefix

    hidden = config.hidden_size
    inter = config.moe_intermediate_size
    n_experts = config.n_routed_experts

    # --- router -------------------------------------------------------------
    _bind(
        module,
        "gate_weight",
        (f"{prefix}.ffn.gate.weight",),
        _replicated_loader(
            "gate_weight", f"{prefix}.ffn.gate.weight", (n_experts, hidden)
        ),
    )

    is_hash = config.is_hash_moe_layer(layer_idx)
    if is_hash:
        _bind(
            module,
            "tid2eid",
            (f"{prefix}.ffn.gate.tid2eid",),
            _replicated_loader(
                "tid2eid",
                f"{prefix}.ffn.gate.tid2eid",
                (config.vocab_size, config.num_experts_per_tok),
            ),
            required=False,
        )
    else:
        _bind(
            module,
            "gate_bias",
            (f"{prefix}.ffn.gate.bias",),
            _replicated_loader(
                "gate_bias", f"{prefix}.ffn.gate.bias", (n_experts,)
            ),
            required=False,
        )

    # --- routed experts (expert-parallel, MXFP4 -> MXFP8) -------------------
    if ep_degree <= 0 or not (0 <= ep_rank < ep_degree):
        raise ValueError(
            f"attach_moe_loaders: invalid ep_rank={ep_rank} for "
            f"ep_degree={ep_degree}."
        )
    if n_experts % ep_degree != 0:
        raise ValueError(
            f"attach_moe_loaders: n_routed_experts={n_experts} is not divisible "
            f"by ep_degree={ep_degree}."
        )
    num_local = n_experts // ep_degree
    local_indices = list(range(ep_rank * num_local, (ep_rank + 1) * num_local))

    experts = getattr(module, "experts", None)
    if experts is None:
        raise AttributeError(
            "DeepseekV4MoE has no 'experts' submodule; the contract fixes "
            "experts.w1_weight / w1_scale / w3_* / w2_* as the routed-expert "
            "parameter names."
        )

    # ``w1``/``w3`` are the gate/up projections (hidden -> intermediate) and
    # ``w2`` the down projection (intermediate -> hidden), matching the
    # reference Expert (dsv4_ref/model.py:596-598).
    # ``kind`` selects the MX tiling: ``w1``/``w3`` contract over ``H`` and tile
    # as gate/up, ``w2`` contracts over ``I`` and tiles as down.
    for leaf, out_dim, logical_k, kind in (
        ("w1", inter, hidden, "gate_up"),
        ("w3", inter, hidden, "gate_up"),
        ("w2", hidden, inter, "down"),
    ):
        all_weight_keys = [
            f"{prefix}.ffn.experts.{e}.{leaf}.weight" for e in range(n_experts)
        ]
        all_scale_keys = [
            f"{prefix}.ffn.experts.{e}.{leaf}.scale" for e in range(n_experts)
        ]
        local_weight_keys = [all_weight_keys[e] for e in local_indices]
        local_scale_keys = [all_scale_keys[e] for e in local_indices]

        # The mapping enumerates all 256 per-expert keys (build_checkpoint_mappings
        # is a pure function of the config and cannot know this core's EP rank),
        # so the EP wrapper trims ``slices`` to the local contiguous range before
        # the upcast runs -- non-local experts are never read from disk.
        _bind(
            experts,
            f"{leaf}_weight",
            all_weight_keys,
            expert_parallel_grouped_loader(
                local_indices,
                _mxfp4_expert_weight_loader(
                    f"experts.{leaf}_weight",
                    local_weight_keys,
                    out_dim,
                    logical_k,
                    kind,
                ),
                n_experts,
            ),
        )
        _bind(
            experts,
            f"{leaf}_scale",
            all_scale_keys,
            expert_parallel_grouped_loader(
                local_indices,
                _mx_expert_scale_loader(
                    f"experts.{leaf}_scale",
                    local_scale_keys,
                    out_dim,
                    logical_k,
                    kind,
                ),
                n_experts,
            ),
        )

    # --- shared expert (block-FP8 on a shared_tp_size-way subgroup) ---------
    shared = getattr(module, "shared_expert", None)
    if shared is None:
        raise AttributeError(
            "DeepseekV4MoE has no 'shared_expert' submodule; the contract fixes "
            "shared_expert.w1_weight / w1_scale / w3_* / w2_* as its parameter "
            "names."
        )
    if shared_tp_size <= 0 or not (0 <= shared_tp_rank < shared_tp_size):
        raise ValueError(
            f"attach_moe_loaders: invalid shared_tp_rank={shared_tp_rank} for "
            f"shared_tp_size={shared_tp_size}."
        )
    if inter % shared_tp_size != 0:
        raise ValueError(
            f"attach_moe_loaders: moe_intermediate_size={inter} is not divisible "
            f"by shared_tp_size={shared_tp_size}."
        )
    inter_local = inter // shared_tp_size

    shared_specs = (
        # (attr leaf, checkpoint leaf, full [out, in], shard)
        (
            "w1",
            "w1",
            (inter, hidden),
            _Shard.rows(shared_tp_rank * inter_local, inter_local, hidden),
        ),
        (
            "w3",
            "w3",
            (inter, hidden),
            _Shard.rows(shared_tp_rank * inter_local, inter_local, hidden),
        ),
        (
            "w2",
            "w2",
            (hidden, inter),
            _Shard.cols(hidden, shared_tp_rank * inter_local, inter_local),
        ),
    )
    for attr_leaf, ckpt_leaf, full_shape, shard in shared_specs:
        w_key = f"{prefix}.ffn.shared_experts.{ckpt_leaf}.weight"
        s_key = f"{prefix}.ffn.shared_experts.{ckpt_leaf}.scale"
        _bind(
            shared,
            f"{attr_leaf}_weight",
            (w_key,),
            _block_fp8_weight_loader(
                f"shared_expert.{attr_leaf}_weight", w_key, full_shape, shard
            ),
        )
        _bind(
            shared,
            f"{attr_leaf}_scale",
            (s_key,),
            _block_fp8_scale_loader(
                f"shared_expert.{attr_leaf}_scale", s_key, full_shape, shard
            ),
        )


# ---------------------------------------------------------------------------
# Public: DSpark draft stages (the stage-conditional tensors only)
# ---------------------------------------------------------------------------


def attach_dspark_stage_loaders(
    module: nn.Module,
    config: "DeepseekV4Config",
    *,
    stage_idx: int,
    tp_size: int,
    tp_rank: int,
) -> None:
    """Attach loaders for the tensors only SOME DSpark stages own.

    The shared per-stage set -- fused ``wq_a``+``wkv``, ``wq_b``, ``wo_a``,
    ``wo_b``, the norms, the hc mixes, the 256 routed experts and the shared
    expert -- is bound by :func:`attach_attention_loaders`,
    :func:`attach_hash_context_loaders` and :func:`attach_moe_loaders` with
    ``key_prefix="mtp.{s}"``, unchanged. Only these are extra:

    * stage 0: ``main_proj`` ``[hidden_size, 12288]`` block-FP8 + its scale
      grid (``dsv4_ref/model.py:832``).
    * last stage: ``confidence_head.proj`` ``[1, hidden_size + markov_rank]``
      (``:810``).

    Two tensors that deliberately get NO loader here: ``main_norm.weight`` and
    the last stage's ``norm.weight``. Both are replicated, unquantized and
    shape-identical to their checkpoint tensors, so the pipelined load's
    identity default is already correct -- the same treatment the main stack's
    final ``norm.weight`` gets. The Markov head's two vocab-sharded tensors and
    the drafter's ``embed_tokens``/``lm_head`` copies get theirs from
    ``sharding_weight_loader`` at construction, because they shard by the
    embedding / lm-head group's rank rather than the attention TP rank.

    >>> PARALLELISM: ``main_proj`` is COLUMN-PARALLEL in N and REPLICATED in K,
    as LD-18 records: ``N_local = hidden_size / tp_size`` = 64 at TP=64.

    K replication is forced rather than chosen -- 12288 / 64 = 192 and
    192 % 128 != 0, so a row shard would break the block-FP8 alignment
    invariant this family maintains everywhere.

    The N shard is half a 128-row scale block, so cores ``2k`` and ``2k+1``
    SHARE scale row ``k``. That is the contained-sub-block case
    :func:`_grid_shard` admits, and sharing is exact: every element of a block
    dequantizes by that one scalar, so each core's rows are bit-identical to
    the matching rows of a full-tensor dequant. The alternative -- replicating
    both axes -- would cost 48 MiB/core of fp8 weight AND make all 64 cores
    redundantly evaluate a ``[T, 12288] x [12288, 4096]`` GEMM that is the
    drafter's single largest, which on a long prefill is the dominant term. The
    shard's price is one all-gather of the ``[T, hidden_size]`` result, which
    :meth:`DeepseekV4DSparkStage.project_main_hidden` owns, because the
    downstream ``NF.mla_qkv`` needs the hidden stream replicated. <<<

    Args:
        module: The :class:`DeepseekV4DSparkStage` instance.
        config: The family config.
        stage_idx: Which stage this is; decides which tensors exist.
        tp_size / tp_rank: Attention tensor-parallel degree and rank.
            ``main_proj``'s N shard is taken by them; nothing else here is
            sharded.
    """
    if tp_size <= 0 or not (0 <= tp_rank < tp_size):
        raise ValueError(
            f"attach_dspark_stage_loaders: invalid tp_rank={tp_rank} for "
            f"tp_size={tp_size}."
        )
    if not (0 <= stage_idx < config.num_dspark_stages):
        raise ValueError(
            f"attach_dspark_stage_loaders: stage_idx={stage_idx} outside the "
            f"{config.num_dspark_stages} stages the weights record."
        )

    prefix = f"mtp.{stage_idx}"

    if stage_idx == 0:
        main_shape = (config.hidden_size, config.dspark_main_hidden_size)
        if main_shape[1] % _BLOCK != 0:
            raise ValueError(
                "attach_dspark_stage_loaders: main_proj K "
                f"({main_shape[1]}) must be a multiple of {_BLOCK} for the "
                "block-FP8 scale grid."
            )
        if main_shape[0] % tp_size != 0:
            raise ValueError(
                "attach_dspark_stage_loaders: main_proj N "
                f"({main_shape[0]}) must be divisible by tp_size={tp_size} for "
                "the column shard LD-18 records."
            )
        n_local = main_shape[0] // tp_size
        main_shard = _Shard.rows(tp_rank * n_local, n_local, main_shape[1])
        weight_key = f"{prefix}.main_proj.weight"
        scale_key = f"{prefix}.main_proj.scale"
        _bind(
            module,
            "main_proj_weight",
            (weight_key,),
            _block_fp8_weight_loader(
                "main_proj_weight", weight_key, main_shape, main_shard
            ),
        )
        _bind(
            module,
            "main_proj_scale",
            (scale_key,),
            _block_fp8_scale_loader(
                "main_proj_scale", scale_key, main_shape, main_shard
            ),
        )

    if stage_idx == config.num_dspark_stages - 1:
        # fp32 destination on purpose: the reference promotes this projection to
        # fp32 because that is what the confidence score needs
        # (``dsv4_ref/model.py:810``). The checkpoint ships it bf16, so the cast
        # happens in the model's ``_cast_to_model_dtype`` -- widening, so it is
        # lossless.
        conf_key = f"{prefix}.confidence_head.proj.weight"
        conf_shape = (1, config.hidden_size + config.dspark_markov_rank)
        _bind(
            module,
            "confidence_head.proj_weight",
            (conf_key,),
            _replicated_loader("confidence_head.proj_weight", conf_key, conf_shape),
        )


# ---------------------------------------------------------------------------
# Public: hash-context (mhc) residual stream
# ---------------------------------------------------------------------------


def attach_hash_context_loaders(
    module: nn.Module,
    config: "DeepseekV4Config",
    *,
    layer_idx: int | None = None,
    scope: str | None = None,
    key_prefix: str | None = None,
) -> None:
    """Attach loaders for the hash-context parameters and the block norms.

    All of these are fp32 (or bf16, for the norms) and replicated: the mhc
    machinery mixes the ``hc_mult``-wide residual stream identically on every
    core, so there is nothing to shard. The loaders exist to validate shapes
    against the config-derived dims, which are non-obvious:

    * ``hc_{attn,ffn}_base``  -> ``[(2 + hc_mult) * hc_mult]`` = ``[24]``
    * ``hc_{attn,ffn}_fn``    -> ``[(2 + hc_mult) * hc_mult, hc_mult * hidden]``
      = ``[24, 16384]``
    * ``hc_{attn,ffn}_scale`` -> ``[3]``
    * ``hc_head_base``        -> ``[hc_mult]`` = ``[4]``
    * ``hc_head_fn``          -> ``[hc_mult, hc_mult * hidden]`` = ``[4, 16384]``
    * ``hc_head_scale``       -> ``[1]``

    Those widths are the reference's (``dsv4_ref/model.py:669-678`` for the
    per-block set, ``:908-910`` for the head set; ``mix_hc = (2 + hc) * hc`` at
    ``kernel.py:374``): ``pre`` and ``post`` take ``hc_mult`` mixes each and the
    Sinkhorn combination matrix takes ``hc_mult**2``, hence ``2*hc + hc**2``,
    and the three ``hc_*_scale`` entries scale those three groups.

    Args:
        module: The module that owns the parameters -- a
            :class:`DeepseekV4HashContext` / decoder layer for the per-block
            set, or the backbone for the head set.
        config: The family config.
        layer_idx: Layer index for the per-block set. Defaults to
            ``module.layer_idx`` when present; unused for the head set, whose
            keys are top-level.
        scope: ``"layer"``, ``"head"`` or ``None``. ``None`` (default)
            auto-detects from which attributes the module actually declares, so
            a module that owns both sets gets both.

    Called positionally with exactly two arguments from ``model.py``; both
    keyword arguments exist only so the head set can be attached explicitly.
    """
    if scope not in (None, "layer", "head"):
        raise ValueError(
            f"attach_hash_context_loaders: scope must be 'layer', 'head' or None, "
            f"got {scope!r}."
        )

    hc_mult = config.hc_mult
    hidden = config.hidden_size
    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * hidden

    do_layer = scope in (None, "layer")
    do_head = scope in (None, "head")

    if do_layer:
        if layer_idx is None:
            layer_idx = getattr(module, "layer_idx", None)
        block_specs: list[tuple[str, str, tuple[int, ...]]] = []
        for which in ("attn", "ffn"):
            block_specs.extend(
                [
                    (f"hc_{which}_base", f"hc_{which}_base", (mix_hc,)),
                    (f"hc_{which}_fn", f"hc_{which}_fn", (mix_hc, hc_dim)),
                    (f"hc_{which}_scale", f"hc_{which}_scale", (3,)),
                ]
            )
        # The block norms live next to the hc parameters because the mhc reduce /
        # expand brackets them. The decoder layer owns them as RMSNorm
        # SUBMODULES (``attn_norm.weight``), so these are dotted binds; the flat
        # ``*_norm_weight`` spelling is also attempted for a module that declares
        # the tensors directly, and a spelling that does not exist is skipped.
        norm_specs = [
            ("attn_norm.weight", "attn_norm.weight", (hidden,)),
            ("ffn_norm.weight", "ffn_norm.weight", (hidden,)),
            ("attn_norm_weight", "attn_norm.weight", (hidden,)),
            ("ffn_norm_weight", "ffn_norm.weight", (hidden,)),
        ]
        if any(hasattr(module, attr) for attr, _, _ in block_specs + norm_specs):
            if not isinstance(layer_idx, int):
                raise ValueError(
                    "attach_hash_context_loaders: the per-block hash-context keys "
                    "are layer-qualified, so layer_idx (or module.layer_idx) is "
                    f"required; got {layer_idx!r}."
                )
            prefix = (
                _layer_key_prefix(layer_idx) if key_prefix is None else key_prefix
            )
            for attr, suffix, shape in block_specs + norm_specs:
                key = f"{prefix}.{suffix}"
                _bind(
                    module,
                    attr,
                    (key,),
                    _replicated_loader(attr, key, shape),
                    required=False,
                )

    if do_head:
        # Top-level keys on the main stack (no prefix). On the DSpark side the
        # SAME three tensors belong to the LAST draft stage
        # (``mtp.2.hc_head_*``, ``dsv4_ref/model.py:834-841``), which is
        # reached by passing ``key_prefix="mtp.2"``.
        head_prefix = "" if key_prefix is None else f"{key_prefix}."
        for attr, key, shape in (
            ("hc_head_base", f"{head_prefix}hc_head_base", (hc_mult,)),
            ("hc_head_fn", f"{head_prefix}hc_head_fn", (hc_mult, hc_dim)),
            ("hc_head_scale", f"{head_prefix}hc_head_scale", (1,)),
        ):
            _bind(
                module,
                attr,
                (key,),
                _replicated_loader(attr, key, shape),
                required=False,
            )


# ---------------------------------------------------------------------------
# Public: checkpoint key mapping
# ---------------------------------------------------------------------------


def build_checkpoint_mappings(
    config: "DeepseekV4Config",
    num_layers: int,
    *,
    mtp: bool = False,
    prefix: str = "model",
) -> dict[str, str | list[str]]:
    """Map each model parameter's qualified name to its checkpoint key(s).

    Every parameter needs an explicit entry, even the 1:1 ones: the load
    pipeline falls back to using the parameter's own dotted name as the
    checkpoint key, and this checkpoint has no ``model.`` prefix, so the
    fallback never matches. Parameters that come from several checkpoint
    tensors (the fused q/kv stack, the per-expert stacks) map to a list, in the
    order their loader's transform expects.

    Entries are emitted only for tensors the pinned checkpoint actually has, as
    the index confirms: ``ffn.gate.bias`` on the 40 non-hash layers,
    ``ffn.gate.tid2eid`` on layers 0-2, the KV compressor from layer 2 on, the
    DSA indexer on the ratio-4 layers. Scale entries are emitted too; they are
    inert for scales that the modules register as buffers (the pipelined load
    iterates ``named_parameters()`` only, so a mapped name with no matching
    parameter is simply never looked up) and are consumed by
    :func:`load_block_scale_buffers` instead.

    Args:
        config: The family config.
        num_layers: Number of main-stack decoder layers. Ignored when
            ``mtp=True`` (the stage count then comes from
            ``config.num_dspark_stages``, which is ruled by the weights --
            see :data:`~.config.CONTRADICTED_CHECKPOINT_FIELDS`).
        mtp: Emit the DSPARK DRAFTER's mapping instead of the main stack's.
            The drafter is a separate top-level module with its own parameter
            tree (``stages.{s}.*``, plus its own ``embed_tokens``/``lm_head``),
            so its mapping is disjoint from the main stack's rather than an
            addition to it; call it once per module. Each stage reuses
            :func:`add_block` verbatim against ``key_prefix = "mtp.{s}"`` and
            ``layer_idx = config.num_hidden_layers + s`` -- which is what makes
            every stage emit ``ffn.gate.bias`` and no ``ffn.gate.tid2eid``
            (``is_hash_moe_layer`` is False past layer 2) and no compressor or
            indexer keys (``compress_ratios[43:46] == [0, 0, 0]``), exactly as
            the ``mtp.*`` key census reports.
        prefix: Dotted attribute path of the backbone inside the top-level
            module whose ``named_parameters()`` will be matched (llama3's
            equivalent is the literal ``"model"``). Keyword-only with a default
            so the recorded call form stays valid. The drafter passes ``""``:
            its stages hang off the top-level module directly.

    Returns:
        ``{parameter name: checkpoint key | [checkpoint keys]}``.
    """
    mappings: dict[str, str | list[str]] = {}
    p = prefix.rstrip(".")

    def put(param: str, keys: str | list[str]) -> None:
        mappings[f"{p}.{param}" if p else param] = keys

    n_experts = config.n_routed_experts

    def add_block(param_layer: str, key_prefix: str, *, layer_idx: int) -> None:
        """Emit one main-stack transformer block's entries."""
        has_compressor = config.has_compressed_cache(layer_idx)
        has_indexer = config.has_indexer(layer_idx)
        is_hash = config.is_hash_moe_layer(layer_idx)

        # norms + hash-context mixes
        for suffix in (
            "attn_norm.weight",
            "ffn_norm.weight",
            "hc_attn_base",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
        ):
            attr = suffix if suffix.endswith(".weight") else suffix
            put(f"{param_layer}.{attr}", f"{key_prefix}.{suffix}")
        # ``*_norm.weight`` may instead be flat ``*_norm_weight`` attributes;
        # emit both spellings. An entry with no matching parameter is inert.
        for flat, suffix in (
            ("attn_norm_weight", "attn_norm.weight"),
            ("ffn_norm_weight", "ffn_norm.weight"),
        ):
            put(f"{param_layer}.{flat}", f"{key_prefix}.{suffix}")

        # attention
        attn = f"{param_layer}.self_attn"
        put(
            f"{attn}.fused_wqa_wkv_weight",
            [f"{key_prefix}.attn.wq_a.weight", f"{key_prefix}.attn.wkv.weight"],
        )
        put(
            f"{attn}.fused_wqa_wkv_scale",
            [f"{key_prefix}.attn.wq_a.scale", f"{key_prefix}.attn.wkv.scale"],
        )
        for leaf in ("wq_b", "wo_a", "wo_b"):
            put(f"{attn}.{leaf}_weight", f"{key_prefix}.attn.{leaf}.weight")
            put(f"{attn}.{leaf}_scale", f"{key_prefix}.attn.{leaf}.scale")
        put(f"{attn}.q_norm_weight", f"{key_prefix}.attn.q_norm.weight")
        put(f"{attn}.kv_norm_weight", f"{key_prefix}.attn.kv_norm.weight")
        put(f"{attn}.attn_sink", f"{key_prefix}.attn.attn_sink")

        if has_compressor:
            for attr, suffix in (
                ("wkv_weight", "wkv.weight"),
                ("wgate_weight", "wgate.weight"),
                ("ape", "ape"),
                ("norm_weight", "norm.weight"),
            ):
                put(
                    f"{attn}.compressor.{attr}",
                    f"{key_prefix}.attn.compressor.{suffix}",
                )

        if has_indexer:
            idx = f"{attn}.indexer"
            ikey = f"{key_prefix}.attn.indexer"
            put(f"{idx}.wq_b_weight", f"{ikey}.wq_b.weight")
            put(f"{idx}.wq_b_scale", f"{ikey}.wq_b.scale")
            put(f"{idx}.weights_proj_weight", f"{ikey}.weights_proj.weight")
            for attr, suffix in (
                ("wkv_weight", "wkv.weight"),
                ("wgate_weight", "wgate.weight"),
                ("ape", "ape"),
                ("norm_weight", "norm.weight"),
            ):
                put(f"{idx}.compressor.{attr}", f"{ikey}.compressor.{suffix}")

        # MoE
        moe = f"{param_layer}.mlp"
        put(f"{moe}.gate_weight", f"{key_prefix}.ffn.gate.weight")
        if is_hash:
            put(f"{moe}.tid2eid", f"{key_prefix}.ffn.gate.tid2eid")
        else:
            put(f"{moe}.gate_bias", f"{key_prefix}.ffn.gate.bias")

        for leaf in ("w1", "w3", "w2"):
            put(
                f"{moe}.experts.{leaf}_weight",
                [
                    f"{key_prefix}.ffn.experts.{e}.{leaf}.weight"
                    for e in range(n_experts)
                ],
            )
            put(
                f"{moe}.experts.{leaf}_scale",
                [
                    f"{key_prefix}.ffn.experts.{e}.{leaf}.scale"
                    for e in range(n_experts)
                ],
            )
            put(
                f"{moe}.shared_expert.{leaf}_weight",
                f"{key_prefix}.ffn.shared_experts.{leaf}.weight",
            )
            put(
                f"{moe}.shared_expert.{leaf}_scale",
                f"{key_prefix}.ffn.shared_experts.{leaf}.scale",
            )

    if mtp:
        # ── DSpark drafter ───────────────────────────────────────────────
        # Every stage's shared parameter set comes from ``add_block`` above,
        # unchanged: same fused wq_a+wkv pair, same wq_b/wo_a/wo_b, same 256
        # routed experts + shared expert + gate.bias, same hc mixes and block
        # norms. Only the three stage-specific groups below are extra.
        last = config.num_dspark_stages - 1
        for stage in range(config.num_dspark_stages):
            add_block(
                f"stages.{stage}",
                f"mtp.{stage}",
                layer_idx=config.num_hidden_layers + stage,
            )

        # Stage 0: the target-hidden down-projection (``dsv4_ref/model.py:832``).
        put("stages.0.main_proj_weight", "mtp.0.main_proj.weight")
        put("stages.0.main_proj_scale", "mtp.0.main_proj.scale")
        put("stages.0.main_norm.weight", "mtp.0.main_norm.weight")
        put("stages.0.main_norm_weight", "mtp.0.main_norm.weight")

        # Last stage: final norm, its OWN hc_head set (top-level on the main
        # stack, ``mtp.2``-qualified here -- ``dsv4_ref/model.py:834-841``),
        # the Markov head and the confidence head.
        put(f"stages.{last}.norm.weight", f"mtp.{last}.norm.weight")
        put(f"stages.{last}.norm_weight", f"mtp.{last}.norm.weight")
        for attr in ("hc_head_base", "hc_head_fn", "hc_head_scale"):
            put(f"stages.{last}.{attr}", f"mtp.{last}.{attr}")
        put(
            f"stages.{last}.markov_head.w1.weight",
            f"mtp.{last}.markov_head.markov_w1.weight",
        )
        put(
            f"stages.{last}.markov_head.w2.weight",
            f"mtp.{last}.markov_head.markov_w2.weight",
        )
        put(
            f"stages.{last}.confidence_head.proj_weight",
            f"mtp.{last}.confidence_head.proj.weight",
        )

        # The drafter is a separately compiled module, so it cannot share the
        # target's parameter tensors the way the reference does
        # (``dsv4_ref/model.py:903-904`` assigns the same objects). It loads
        # its own copy of the SAME two checkpoint tensors.
        put("embed_tokens.weight", "embed.weight")
        put("lm_head.weight", "head.weight")
        return mappings

    for layer_idx in range(num_layers):
        add_block(
            f"layers.{layer_idx}",
            _layer_key_prefix(layer_idx),
            layer_idx=layer_idx,
        )
    put("norm.weight", "norm.weight")
    put("hc_head_base", "hc_head_base")
    put("hc_head_fn", "hc_head_fn")
    put("hc_head_scale", "hc_head_scale")

    # Embedding and LM head. Untied (``tie_word_embeddings=false``), so ``head``
    # is its own checkpoint tensor.
    put("embed_tokens.weight", "embed.weight")
    mappings["lm_head.weight"] = "head.weight"

    return mappings


# ---------------------------------------------------------------------------
# Public: post-hoc scale-buffer load
# ---------------------------------------------------------------------------


def load_block_scale_buffers(
    module_tree: nn.Module,
    checkpoint: "SafetensorsCheckpoint",
    rank: int,
    device: torch.device,
) -> None:
    """Load every loader-bound tensor that is a *buffer*, not a parameter.

    Why this exists: the pipelined checkpoint reader walks
    ``named_parameters()`` only, so anything a module registers with
    ``register_buffer(..., persistent=False)`` -- the natural home for a
    dequant scale that must not be touched by ``load_state_dict`` -- is skipped
    entirely, and would silently stay at its ``torch.empty`` contents. llama3
    solves the same problem with its ``_FP8_*_SCALE_SOURCES`` tables; here the
    sources are recorded by the ``attach_*`` functions themselves, so the
    checkpoint-key convention stays in one file.

    Parameters are deliberately skipped: they are already covered by
    :func:`build_checkpoint_mappings` plus the pipelined load, and loading them
    twice would waste the disk read.

    Args:
        module_tree: Any module; its whole subtree is walked.
        checkpoint: An open :class:`SafetensorsCheckpoint`.
        rank: Rank passed to each loader. The shard is already baked into every
            transform in this module, so this only matters for loaders that
            read it.
        device: Device the loaded tensors are moved to.
    """
    available = checkpoint.get_tensor_names()
    loaded = 0

    for module_name, module in module_tree.named_modules():
        for attr, ckpt_keys, loader in getattr(module, _SCALE_SOURCES_ATTR, ()):
            qualified = f"{module_name}.{attr}" if module_name else attr
            current = getattr(module, attr, None)
            if isinstance(current, nn.Parameter):
                continue  # owned by the pipelined named_parameters() load

            missing = [k for k in ckpt_keys if k not in available]
            if missing:
                raise KeyError(
                    f"Missing checkpoint key(s) for buffer {qualified!r}: "
                    f"{missing}. This buffer is bound to those keys by the "
                    "deepseek_v4 weight loaders, so either the checkpoint "
                    "revision changed or the wrong attach function ran."
                )

            slices = [checkpoint._get_slice(k) for k in ckpt_keys]
            tensor = loader.load(slices, rank).to(device)
            setattr(module, attr, tensor)
            loaded += 1

    logger.info(
        "deepseek_v4: loaded %d non-parameter (buffer) tensors from the checkpoint.",
        loaded,
    )
