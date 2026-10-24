# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 vision tower, projector and encoder hook.

Each encoder DP group runs one group block per call: a bucket of patch rows in
2 x 2 window order, one image followed by zero rows, or zero rows only. A layer
is a compiler residual add, the library QKV call with its RMSNorm and 2D RoPE,
the library attention with per-row key bounds, o_proj, and the library MLP call
with the attention output added to the residual. After the final norm the 2 x 2
merge is a view, and each TP rank projects its share of the merged rows. The
hand-off writes the merged rows into the encoder cache: one real block through
one all-reduce over the encoder DP group and one copy, more real blocks through
the DP all-gather and one index write.

A lone image at the one-block bucket splits instead into one slice of whole
windows per encoder DP group. Each layer gathers k and v of every slice over
the DP group, so each query reads the whole image's keys in their unsplit
order, and the hand-off gathers the projected slices into the image's rows.

The tower takes the processor's uint8 patches. The checkpoint's normalize,
``u / 127.5 - 1`` on each value, folds into the patch embed: one fp32 matmul on
the raw values, scaled by ``1 / 127.5``, plus a bias of minus the patch weight's
row sum, all before the one bf16 round.

Each rank cuts its own group block or slice on the host, so the tower takes only
those rows and no device select. The RoPE tables come from the worker's table cache;
the tower scales them by the square root of the attention scale, so the
attention runs at scale 1.0. Outside CPU mode the build refuses a QKV width the
QKV kernel cannot run, and it refuses a vision config whose two layer counts differ.
"""

import functools
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from nkilib.core.utils.common_types import (
    ActFnType,
    NormType,
    OProjAttentionLayout,
    QKVOutputLayout,
)
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import model_parallel_is_initialized

from vllm_neuron import envs
from vllm_neuron.functional import flash_attention, mlp, o_proj, qkv_proj
from vllm_neuron.functional import rmsnorm_residual_prefill
from vllm_neuron.functional.attention.qkv import _can_use_qkv_kernel
from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig
from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_encoder_dp_group,
    get_neuron_encoder_tp_group,
    shard_select,
)
from vllm_neuron.utils.dynamo_graph import allow_in_graph_cacheable

from .config import KimiK3Config, KimiK3VisionConfig, _as_dict
from .model import KimiK3ForConditionalGeneration
from .vision_inputs import (
    HEAD_DIM,
    Slice,
    TableCache,
    block_slice,
    cut,
    grid_tables,
    key_bounds,
    order_patches,
    pos_embed_weights,
    window_order_index,
)
from .weight_loaders_vision import encoder_tp

#: ``nn.RMSNorm(eps=None)`` on the tower's activations, which it normalizes in fp32.
TOWER_EPS = float(torch.finfo(torch.float32).eps)
#: The attention reads its key bounds in tiles of 128 query rows.
ATTN_SEQ_ALIGN = 128
#: Pixel dtype of the registered K3 processor: the raw values, before any normalize.
PIXEL_DTYPE = torch.uint8
#: The checkpoint's normalize scale on a uint8 value, which the patch embed applies.
PIXEL_SCALE = 1 / 127.5
#: Each output row of a bilinear resize reads at most two input rows per axis.
CORNERS = 4
#: Grids whose corner and bound inputs a worker keeps on its device: 43 MB at most at the
#: 33,536-row bucket.
GRID_INPUTS_CACHED = 32
#: The vision config names the tower depth twice.
LAYER_COUNT_NAMES = ("vt_num_hidden_layers", "num_hidden_layers")


@allow_in_graph_cacheable
def gelu(x: torch.Tensor) -> torch.Tensor:
    """Erf GELU in fp32, traceable by torch.compile on Neuron."""
    xf = x.float()
    return (0.5 * xf * (1.0 + torch.erf(xf / 1.4142135623730951))).to(x.dtype)


class _Weight(nn.Module):
    """A checkpoint module that holds one ``weight``; the loader assigns it."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class _Gelu(nn.Module):
    """The projector's erf GELU, a module so the projector keeps its checkpoint indices."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``gelu(x)``."""
        return gelu(x)


def _encoder_groups():
    """Return the encoder TP group (None at size 1) and DP group (None at size 1)."""
    if not model_parallel_is_initialized():
        return None, None
    tp = get_neuron_encoder_tp_group()
    return (tp if tp.world_size > 1 else None), get_neuron_encoder_dp_group()


def _dp_position() -> tuple[int, int]:
    """Return this rank's index and the group count of the encoder DP group."""
    _, dp = _encoder_groups()
    return (dp.rank_in_group, dp.world_size) if dp is not None else (0, 1)


def _all_reduce(x: torch.Tensor, group) -> torch.Tensor:
    """Sum ``x`` over ``group``; no group means one rank."""
    return x if group is None else group.all_reduce(x)


def _check_qkv_width(hidden: int, width: int) -> None:
    """Raise outside CPU mode when the QKV wrapper would take its torch branch at ``width``."""
    if envs.VLLM_NEURON_CPU_MODE:
        return
    # Meta tensors stand in for the device's: the check reads shapes and the device type.
    ok = _can_use_qkv_kernel(
        hidden=torch.empty((1, 1, hidden), dtype=torch.bfloat16, device="meta"),
        qkv_weights=torch.empty((hidden, width), dtype=torch.bfloat16, device="meta"),
        mlp_prev=None,
        attn_prev=None,
        d_head=HEAD_DIM,
        output_layout=QKVOutputLayout.BSD,
        norm_type=NormType.RMS_NORM,
        norm_b=None,
        fused_rope=True,
    )
    if not ok:
        raise ValueError(
            f"the tower's QKV width {width} takes the wrapper's torch branch; "
            "pick an encoder TP size whose width the QKV kernel runs"
        )


def _check_layer_counts(hf_config) -> None:
    """Raise when the vision config's two layer-count names differ."""
    vision = _as_dict(hf_config).get("vision_config")
    if vision is None:
        return
    vision = _as_dict(vision)
    counts = {name: vision[name] for name in LAYER_COUNT_NAMES if name in vision}
    if len(set(counts.values())) > 1:
        raise ValueError(f"the vision config's layer counts differ: {counts}")


def corner_weights(grids, bucket: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear pos-embed corners of each row: table index [4, bucket] int32, weight fp32.

    Row ``r`` of the block, in window order, reads table rows ``index[:, r]`` of the
    flattened 64 x 64 table with weights ``weight[:, r]``. Pad rows have weight 0.
    """
    index = torch.zeros((CORNERS, bucket), dtype=torch.int32)
    weight = torch.zeros((CORNERS, bucket), dtype=torch.float32)
    start = 0
    for h, w in grids:
        wh, ww = pos_embed_weights(h, w)
        side = ww.shape[1]
        hv, hi = wh.topk(2, dim=1)
        wv, wi = ww.topk(2, dim=1)
        raster = window_order_index(h, w)
        rows, cols = raster // w, raster % w
        end = start + h * w
        for k, (a, b) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            index[k, start:end] = (hi[rows, a] * side + wi[cols, b]).int()
            weight[k, start:end] = hv[rows, a] * wv[cols, b]
        start = end
    return index, weight


@functools.lru_cache(maxsize=GRID_INPUTS_CACHED)
def grid_inputs(
    grids: tuple, bucket: int, device: torch.device, part: Slice | None = None
) -> tuple[torch.Tensor, ...]:
    """A block's per-grid inputs on ``device``: corner index and weight, key bounds.

    With ``part``, each input holds that slice's rows; its pad rows have weight 0 and
    an empty key range.
    """
    inputs = (*corner_weights(grids, bucket), *key_bounds(grids, bucket))
    if part is not None:
        # Corners are [4, bucket]; bounds are [bucket, 1].
        inputs = tuple(cut(t, part, dim) for t, dim in zip(inputs, (1, 1, 0, 0)))
    return tuple(t.to(device) for t in inputs)


class KimiK3PatchEmbed(nn.Module):
    """Patch matmul with the folded normalize, plus the bilinear-resized position table."""

    def __init__(self, config: KimiK3VisionConfig, dtype: torch.dtype):
        super().__init__()
        patch_dim = 3 * config.patch_size**2
        self.proj = _Weight((patch_dim, config.hidden_size), dtype)
        self.pos_emb = _Weight(
            (config.init_pos_emb_height, config.init_pos_emb_width, config.hidden_size),
            dtype,
        )
        self.bias = nn.Parameter(
            torch.empty(config.hidden_size, dtype=torch.float32), requires_grad=False
        )

    def forward(self, pixels, pos_index, pos_weight) -> torch.Tensor:
        """Return the embedded rows [bucket, hidden] of uint8 patches."""
        weight = self.proj.weight
        acc = self.bias.dtype
        # The matmul output stays fp32: the bias cancels most of a flat patch's sum.
        x = pixels.to(acc) @ weight.to(acc)
        table = self.pos_emb.weight.reshape(-1, weight.shape[1])
        corners = F.embedding(pos_index.long(), table).to(acc)
        pos = (corners * pos_weight.to(acc).unsqueeze(-1)).sum(0)
        return (x * PIXEL_SCALE + self.bias + pos).to(weight.dtype)


class KimiK3VisionBlock(nn.Module):
    """One tower layer at this rank's heads and MLP features."""

    def __init__(self, config: KimiK3VisionConfig, tp_size: int, dtype: torch.dtype):
        super().__init__()
        self.heads = config.num_attention_heads // tp_size
        width = self.heads * config.head_dim
        hidden = config.hidden_size
        inter = config.intermediate_size // tp_size
        _check_qkv_width(hidden, 3 * width)
        self.wqkv = _Weight((hidden, 3 * width), dtype)
        self.wo = _Weight((width, hidden), dtype)
        self.mlp = nn.Module()
        self.mlp.fc0 = _Weight((hidden, inter), dtype)
        self.mlp.fc1 = _Weight((inter, hidden), dtype)
        self.norm0 = _Weight((hidden,), dtype)
        self.norm1 = _Weight((hidden,), dtype)

    def forward(self, x, cos, sin, bound_min, bound_max, tp_group, kv_group=None, slice_rows=0):
        """Return the layer's MLP output and its stored residual sum.

        With ``kv_group``, ``x`` is one slice of an image: the first ``slice_rows``
        rows of k and v from every slice are gathered over ``kv_group`` as the keys.
        """
        rows, hidden = x.shape
        qkv = qkv_proj(
            x.unsqueeze(0),
            self.wqkv.weight,
            norm_weights=self.norm0.weight.reshape(1, -1),
            eps=TOWER_EPS,
            norm_type=NormType.RMS_NORM,
            d_head=HEAD_DIM,
            output_layout=QKVOutputLayout.BSD,
            cos_cache=cos,
            sin_cache=sin,
            num_q_heads=self.heads,
            num_kv_heads=self.heads,
        )
        qkv = qkv.view(rows, 3, self.heads, HEAD_DIM)
        if kv_group is None:
            q, k, v = qkv.permute(1, 2, 0, 3).contiguous()
        else:
            kv = kv_group.all_gather(qkv[:slice_rows, 1:].contiguous(), dim=0)
            q = qkv[:, 0].transpose(0, 1).contiguous()
            k, v = kv.permute(1, 2, 0, 3).contiguous()
        attn = flash_attention(
            q, k, v, scale=1.0, causal_mask=False, tp_q=True, tp_k=True,
            bound_min=bound_min, bound_max=bound_max,
        )
        attn = attn.transpose(0, 1).reshape(1, rows, self.heads, HEAD_DIM)
        out = o_proj(attn, self.wo.weight, attention_layout=OProjAttentionLayout.BSNd)
        out = _all_reduce(out.reshape(rows, hidden), tp_group)
        fc0 = self.mlp.fc0.weight
        mix, total = mlp(
            out, fc0, fc0, self.mlp.fc1.weight, eps=TOWER_EPS,
            ln_w=self.norm1.weight.reshape(1, -1), skip_gate=True,
            act_fn=ActFnType.GELU_Tanh_Approx, norm_type=NormType.RMS_NORM,
            fused_add_tensor=x, store_fused_add_result=True,
        )
        return _all_reduce(mix, tp_group), total


class KimiK3VisionEncoder(nn.Module):
    """The tower's layers and final norm, in the checkpoint's module names."""

    def __init__(self, config: KimiK3VisionConfig, tp_size: int, dtype: torch.dtype):
        super().__init__()
        self.blocks = nn.ModuleList(
            KimiK3VisionBlock(config, tp_size, dtype)
            for _ in range(config.num_hidden_layers)
        )
        self.final_layernorm = _Weight((config.hidden_size,), dtype)

    def forward(
        self, x, cos, sin, bound_min, bound_max, tp_group, kv_group=None, slice_rows=0
    ) -> torch.Tensor:
        """Return the final-normed rows; a slice keeps only its first ``slice_rows``."""
        mix = total = None
        for block in self.blocks:
            if mix is not None:
                x = total + mix
            mix, total = block(x, cos, sin, bound_min, bound_max, tp_group, kv_group, slice_rows)
        if slice_rows:
            mix, total = mix[:slice_rows], total[:slice_rows]
        normed, _ = rmsnorm_residual_prefill(
            mix, self.final_layernorm.weight, residual=total, eps=TOWER_EPS
        )
        return normed


class KimiK3VisionTower(nn.Module):
    """Patch embed and encoder, under the checkpoint's ``vision_tower`` names."""

    def __init__(self, config: KimiK3VisionConfig, tp_size: int, dtype: torch.dtype):
        super().__init__()
        self.patch_embed = KimiK3PatchEmbed(config, dtype)
        self.encoder = KimiK3VisionEncoder(config, tp_size, dtype)

    def forward(
        self, pixels, pos_index, pos_weight, cos, sin, bound_min, bound_max, tp_group,
        kv_group=None, slice_rows=0,
    ):
        """Return the final-normed rows [bucket, hidden], or a slice's [slice_rows, hidden]."""
        x = self.patch_embed(pixels, pos_index, pos_weight)
        heads = self.encoder.blocks[0].heads
        # q and k leave the QKV call scaled by sqrt(1 / sqrt(d)) each, so the
        # attention takes scale 1.0.
        root_scale = HEAD_DIM**-0.25
        cos = (cos * root_scale).to(x.dtype).unsqueeze(0)
        sin = (sin * root_scale).to(x.dtype).unsqueeze(0)
        bound_min = bound_min.unsqueeze(0).repeat(heads, 1, 1)
        bound_max = bound_max.unsqueeze(0).repeat(heads, 1, 1)
        return self.encoder(x, cos, sin, bound_min, bound_max, tp_group, kv_group, slice_rows)


class KimiK3Projector(nn.Module):
    """Linear, erf GELU, linear and RMSNorm on the merged rows, under ``mm_projector``."""

    def __init__(self, config: KimiK3VisionConfig, dtype: torch.dtype):
        super().__init__()
        merged = config.hidden_size * config.merge_factor
        self.proj = nn.ModuleList(
            [
                _Weight((merged, merged), dtype),
                _Gelu(),
                _Weight((merged, config.text_hidden_size), dtype),
            ]
        )
        self.post_norm = _Weight((config.text_hidden_size,), dtype)
        self.eps = config.projector_ln_eps

    def forward(self, merged: torch.Tensor, tp_group, norm_gathered: bool = False) -> torch.Tensor:
        """Project this rank's share of the merged rows, then gather the shares.

        ``norm_gathered`` norms after the gather, for a slice whose shares are under
        the norm kernel's 128 rows.
        """
        if tp_group is not None:
            merged = shard_select(merged, 0, tp_group)
        first, act, second = self.proj
        x = act(merged @ first.weight) @ second.weight
        if tp_group is not None and norm_gathered:
            x = tp_group.all_gather(x, dim=0)
        x, _ = rmsnorm_residual_prefill(x, self.post_norm.weight, eps=self.eps)
        if tp_group is not None and not norm_gathered:
            x = tp_group.all_gather(x, dim=0)
        return x


class KimiK3VisionModel(nn.Module):
    """The compiled encoder graph: one group block in, its merged rows written to the cache."""

    def __init__(self, config: KimiK3VisionConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        _, tp_size = encoder_tp()
        self.tp_group, self.dp_group = _encoder_groups()
        self.vision_tower = KimiK3VisionTower(config, tp_size, dtype)
        self.mm_projector = KimiK3Projector(config, dtype)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pos_index: torch.Tensor,
        pos_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
        encoder_cache_buffer: torch.Tensor,
        write_block_ids: torch.Tensor,
        gather: bool = False,
        slice_rows: int = 0,
    ) -> torch.Tensor:
        """Encode the rank's group block or slice and write the merged rows into the cache.

        Args:
            pixel_values: ``[bucket, 3 * p * p]`` uint8 patches in window order, or a slice's.
            pos_index, pos_weight: ``[4, bucket]`` bilinear corners of each row.
            cos, sin: ``[bucket, head_dim]`` fp32 RoPE tables in split-half layout.
            bound_min, bound_max: ``[bucket, 1]`` int32 key range of each row.
            encoder_cache_buffer: ``[blocks, rows, text_hidden]``, written in place.
            write_block_ids: cache block of each written block, scratch for the rest.
            gather: True when more than one group block is real.
            slice_rows: the slice's rows before its pad, when one image is split over
                the encoder DP groups; 0 for a group block.

        Returns:
            ``encoder_cache_buffer``.
        """
        rows = pixel_values.shape[0]
        if rows % ATTN_SEQ_ALIGN:
            raise ValueError(
                f"the tower's {rows} query rows are not a multiple of {ATTN_SEQ_ALIGN}"
            )
        if slice_rows and self.dp_group is None:
            raise ValueError("a split image needs the encoder DP group")
        merge = self.config.merge_factor
        x = self.vision_tower(
            pixel_values, pos_index, pos_weight, cos, sin, bound_min, bound_max,
            self.tp_group, self.dp_group if slice_rows else None, slice_rows,
        )
        rows = slice_rows or rows
        merged = x.reshape(rows // merge, merge * x.shape[1])
        out = self.mm_projector(merged, self.tp_group, norm_gathered=bool(slice_rows))
        # Pad rows go out as zeros, so a padding group adds nothing to the hand-off.
        real = (bound_max[:rows] > bound_min[:rows]).reshape(-1, merge)[:, :1]
        out = torch.where(real, out, torch.zeros_like(out))
        return self._hand_off(
            out, encoder_cache_buffer, write_block_ids, gather, bool(slice_rows)
        )

    def _hand_off(self, out, buffer, write_block_ids, gather, joined=False) -> torch.Tensor:
        """Write the merged rows: the group block's, every group's if ``gather``.

        ``joined`` first gathers every group's slice, in group order, into the image's rows.
        """
        if joined:
            out = self.dp_group.all_gather(out, dim=0)
        block_rows = buffer.shape[1]
        per_block = math.ceil(out.shape[0] / block_rows)
        out = F.pad(out, (0, 0, 0, per_block * block_rows - out.shape[0]))
        blocks = out.reshape(per_block, block_rows, out.shape[1])
        if self.dp_group is not None and not joined:
            if gather:
                blocks = self.dp_group.all_gather(blocks, dim=0)
            else:
                blocks = self.dp_group.all_reduce(blocks)
        buffer.index_put_((write_block_ids[: blocks.shape[0]],), blocks)
        return buffer


class _EncoderWarmup(nn.Module):
    """The tower as warmup calls it: once per hand-off form the host can pick at the bucket."""

    def __init__(self, tower: nn.Module, forms, synthetic):
        super().__init__()
        self.tower = tower
        self.forms = forms
        self.synthetic = synthetic

    def forward(self, **inputs) -> torch.Tensor:
        """Run the tower at each hand-off form of the inputs' bucket, a split at its slice's rows."""
        pixels = inputs["pixel_values"]
        for gather, slice_rows in self.forms(pixels.shape[0]):
            call = dict(inputs)
            if slice_rows:
                call.update(self.synthetic(_padded(slice_rows), pixels.device))
            out = self.tower(**call, gather=gather, slice_rows=slice_rows)
        return out


def _padded(rows: int) -> int:
    """``rows`` rounded up to the attention's query tile."""
    return -(-rows // ATTN_SEQ_ALIGN) * ATTN_SEQ_ALIGN


class KimiK3VisionForConditionalGeneration(KimiK3ForConditionalGeneration):
    """Kimi K3 with the vision tower as ``visual`` and the encoder hook and warmup."""

    #: Split a lone image at the one-block bucket over the encoder DP groups.
    split_lone_image = True

    def __init__(self, config: KimiK3Config, encoder_neuron_config: EncoderNeuronConfig):
        super().__init__(config)
        if config.vision_config is None:
            raise ValueError("an encoder config was given, but the checkpoint has no vision tower")
        self.encoder_neuron_config = encoder_neuron_config
        self.visual = KimiK3VisionModel(config.vision_config, config.text_config.torch_dtype)
        self._tables: TableCache | None = None

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> "KimiK3VisionForConditionalGeneration":
        """Build the model from the HF config and both NeuronConfigs."""
        _check_layer_counts(hf_config)
        return cls(KimiK3Config.from_configs(hf_config, text_neuron_config), encoder_neuron_config)

    def get_encoder(self) -> nn.Module:
        """Return the vision tower as warmup calls it, at every hand-off form of each bucket."""
        return _EncoderWarmup(
            self.visual,
            self._forms,
            lambda rows, device: self.build_encoder_synthetic_inputs(
                rows, self.encoder_neuron_config, device
            ),
        )

    def _packs(self, bucket: int) -> bool:
        """Report if a call at ``bucket`` holds one image per encoder DP group.

        Above the block size a group block fills more than one cache block, and the
        worker's write ids hold one group's blocks only, so each image takes its own call.
        """
        config = self.encoder_neuron_config
        return config.dp_size > 1 and bucket <= config.encoder_attention_block_size

    def _slice_rows(self, bucket: int) -> int:
        """Rows of each group's slice when a lone image at ``bucket`` splits, else 0.

        Only the one-block bucket splits, so the joined slices fill the image's cache
        blocks, and only into slices of whole windows that each TP rank's share of
        merged rows divides evenly.
        """
        config = self.encoder_neuron_config
        groups = config.dp_size
        whole = groups * self.config.vision_config.merge_factor * config.tp_size
        if (
            not self.split_lone_image
            or groups == 1
            or bucket != config.encoder_attention_block_size
            or bucket % whole
        ):
            return 0
        return bucket // groups

    def _forms(self, bucket: int) -> tuple[tuple[bool, int], ...]:
        """The hand-off forms a call at ``bucket`` can take, as (``gather``, ``slice_rows``)."""
        slice_rows = self._slice_rows(bucket)
        if slice_rows:
            # A lone image splits, so the one-block form is not reachable here.
            return ((False, slice_rows), (True, 0))
        return ((False, 0), (True, 0)) if self._packs(bucket) else ((False, 0),)

    def build_encoder_synthetic_inputs(
        self, bucket: int, encoder_neuron_config: EncoderNeuronConfig, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Return the tower's per-bucket inputs as zeros, at the serving shapes and dtypes."""
        patch_dim = 3 * self.config.vision_config.patch_size**2
        rows = (bucket, 1)
        return {
            "pixel_values": torch.zeros(bucket, patch_dim, dtype=PIXEL_DTYPE, device=device),
            "pos_index": torch.zeros(CORNERS, bucket, dtype=torch.int32, device=device),
            "pos_weight": torch.zeros(CORNERS, bucket, dtype=torch.float32, device=device),
            "cos": torch.zeros(bucket, HEAD_DIM, dtype=torch.float32, device=device),
            "sin": torch.zeros(bucket, HEAD_DIM, dtype=torch.float32, device=device),
            "bound_min": torch.zeros(rows, dtype=torch.int32, device=device),
            "bound_max": torch.zeros(rows, dtype=torch.int32, device=device),
        }

    def embed_multimodal(
        self,
        pixel_values: torch.Tensor | None = None,
        grid_thws: torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Encode the images and write their merged rows into the on-device encoder cache.

        Args:
            pixel_values: ``[patches, 3, p, p]`` uint8, every image's patches in raster order.
            grid_thws: ``[images, 3]`` (1, height, width) patch grid of each image.
            encoder_cache: the runner's ``EncoderCacheBlocks``.
            mm_hashes: one identifier per image, for the cache allocation.
        """
        if pixel_values is None or grid_thws is None:
            raise ValueError("embed_multimodal needs pixel_values and grid_thws")
        if pixel_values.dtype != PIXEL_DTYPE:
            raise ValueError(
                f"the patch embed folds the normalize and takes {PIXEL_DTYPE} pixels, "
                f"not {pixel_values.dtype}"
            )
        grids = []
        for t, h, w in grid_thws.tolist():
            if t != 1:
                raise ValueError(f"a grid of {t} frames is not an image")
            grids.append((h, w))
        sizes = [h * w for h, w in grids]
        patches = pixel_values.reshape(sum(sizes), -1).split(sizes)
        merge = self.config.vision_config.merge_factor
        block_ids = [
            encoder_cache.allocate(
                mm_hash,
                encoder_cache.dense_tokens_per_block(size // merge, encoder_cache.block_size),
            )
            for mm_hash, size in zip(mm_hashes, sizes)
        ]
        for bucket, members in self._calls(sizes):
            self._encode(
                bucket,
                [(patches[i], grids[i], block_ids[i]) for i in members],
                encoder_cache,
            )

    def _calls(self, sizes: list[int]) -> list[tuple[int, list[int]]]:
        """Group the images into tower calls: one image per group block, at the smallest bucket."""
        config = self.encoder_neuron_config
        buckets = sorted(config.num_encoder_input_buckets)
        by_bucket = defaultdict(list)
        for i, size in enumerate(sizes):
            bucket = next((b for b in buckets if b >= size), None)
            if bucket is None:
                raise ValueError(f"an image of {size} patches exceeds the largest bucket {buckets[-1]}")
            by_bucket[bucket].append(i)
        calls = []
        for bucket, members in sorted(by_bucket.items()):
            width = config.dp_size if self._packs(bucket) else 1
            calls += [(bucket, members[k : k + width]) for k in range(0, len(members), width)]
        return calls

    def _encode(self, bucket: int, images: list, cache) -> None:
        """Run one tower call: this rank's group block or slice, then the hand-off."""
        config = self.encoder_neuron_config
        rank, _ = _dp_position()
        device = cache.buffer.device
        if self._tables is None:
            self._tables = TableCache(device)
        slice_rows = self._slice_rows(bucket) if len(images) == 1 else 0
        if slice_rows:
            pixels, grid, _ = images[0]
            part = block_slice(bucket, config.dp_size, rank, ATTN_SEQ_ALIGN)
            inputs = self._block_inputs(bucket, pixels, (grid,), device, part)
        elif rank < len(images):
            pixels, grid, _ = images[rank]
            inputs = self._block_inputs(bucket, pixels, (grid,), device)
        else:
            inputs = self._block_inputs(bucket, images[0][0][:0], (), device)
        per_block = math.ceil(bucket / config.encoder_attention_block_size)
        groups = config.dp_size if len(images) > 1 else 1
        ids = []
        for g in range(groups):
            own = images[g][2] if g < len(images) else []
            ids += (own + [cache.scratch_block_id] * per_block)[:per_block]
        count = math.ceil(per_block / config.dp_size) * config.dp_size
        ids += [cache.scratch_block_id] * (count - len(ids))
        self.visual(
            **inputs,
            encoder_cache_buffer=cache.buffer,
            write_block_ids=torch.tensor(ids, dtype=torch.int64, device=device),
            gather=len(images) > 1,
            slice_rows=slice_rows,
        )

    def _block_inputs(
        self, bucket: int, pixels: torch.Tensor, grids: tuple, device, part: Slice | None = None
    ) -> dict:
        """The tower inputs of one group block, or of its slice ``part``, on ``device``.

        ``grids`` is empty for padding. Only the pixels cross from the host on each
        call; the per-grid inputs are cached.
        """
        if grids:
            pixels = order_patches(pixels, grids, bucket, part).to(device)
            cos, sin, _ = grid_tables(self._tables, grids, bucket, part)
        else:
            pixels = torch.zeros((bucket, pixels.shape[1]), dtype=pixels.dtype, device=device)
            cos = sin = torch.zeros((bucket, HEAD_DIM), dtype=torch.float32, device=device)
        pos_index, pos_weight, bound_min, bound_max = grid_inputs(grids, bucket, device, part)
        return {
            "pixel_values": pixels,
            "pos_index": pos_index,
            "pos_weight": pos_weight,
            "cos": cos,
            "sin": sin,
            "bound_min": bound_min,
            "bound_max": bound_max,
        }
