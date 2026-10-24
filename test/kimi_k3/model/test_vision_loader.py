# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 vision tower and projector loading against the checkpoint on disk.

The tower module stands in as a meta-device module whose parameters carry the
checkpoint keys under ``visual.``, at the per-rank shapes of a 4-way encoder
tensor-parallel split. Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory;
the default is ``~/models/moonshotai/Kimi-K3``.
"""

import json
import os
from collections import Counter
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from vllm_neuron.model.kimi_k3 import KimiK3Config, weight_loaders_vision
from vllm_neuron.model.kimi_k3.weight_loaders_mxfp4 import checkpoint_mappings, load_weights

CHECKPOINT = Path(
    os.environ.get(
        "KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3"
    )
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

TP = 4
PARENT = "visual."
GRID = 74  # a 1024 x 1024 image pads to 74 x 74 patches
PATCH_KEY = "vision_tower.patch_embed.proj.weight"

# The checkpoint's vision config: 12 heads of 128, width 1024, MLP width 4096,
# 27 layers, 14 x 14 patches, a 64 x 64 position table, 2 x 2 merge to 4096,
# text width 7168.
HEADS, HEAD_DIM, HIDDEN, INTER, LAYERS = 12, 128, 1024, 4096, 27
P, POS, MERGED, TEXT = 14, 64, 4096, 7168


def _weight_map() -> dict[str, str]:
    with open(CHECKPOINT / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _vision_keys() -> list[str]:
    """Checkpoint keys outside the language model."""
    return [k for k in _weight_map() if not k.startswith("language_model.")]


def _read(names: list[str]) -> dict[str, torch.Tensor]:
    """Read checkpoint tensors by key, as stored."""
    weight_map = _weight_map()
    out = {}
    for name in names:
        with safe_open(CHECKPOINT / weight_map[name], framework="pt") as f:
            out[name] = f.get_tensor(name)
    return out


def _shapes(tp: int) -> dict[str, tuple[int, ...]]:
    """Per-rank parameter shapes of the tower and projector, matrices as [in, out]."""
    width = HEADS // tp * HEAD_DIM
    shapes = {
        PATCH_KEY: (3 * P * P, HIDDEN),
        "vision_tower.patch_embed.pos_emb.weight": (POS, POS, HIDDEN),
        "vision_tower.encoder.final_layernorm.weight": (HIDDEN,),
        "mm_projector.proj.0.weight": (MERGED, MERGED),
        "mm_projector.proj.2.weight": (MERGED, TEXT),
        "mm_projector.post_norm.weight": (TEXT,),
    }
    for i in range(LAYERS):
        block = f"vision_tower.encoder.blocks.{i}"
        shapes[f"{block}.wqkv.weight"] = (HIDDEN, 3 * width)
        shapes[f"{block}.wo.weight"] = (width, HIDDEN)
        shapes[f"{block}.mlp.fc0.weight"] = (HIDDEN, INTER // tp)
        shapes[f"{block}.mlp.fc1.weight"] = (INTER // tp, HIDDEN)
        shapes[f"{block}.norm0.weight"] = (HIDDEN,)
        shapes[f"{block}.norm1.weight"] = (HIDDEN,)
    return shapes


def _stand_in(tp: int) -> nn.Module:
    """A meta-device module whose bf16 parameters carry the checkpoint keys under ``visual.``."""
    root = nn.Module()
    for name, shape in _shapes(tp).items():
        *path, leaf = (PARENT + name).split(".")
        module = root
        for part in path:
            if not hasattr(module, part):
                module.add_module(part, nn.Module())
            module = getattr(module, part)
        param = torch.empty(shape, dtype=torch.bfloat16, device="meta")
        module.register_parameter(leaf, nn.Parameter(param, requires_grad=False))
    return root


@pytest.fixture(scope="module")
def ranks():
    """The stand-in loaded through the text loader at each encoder rank, with its mappings."""
    from vllm.transformers_utils.config import get_config

    text_config = KimiK3Config.from_configs(
        get_config(str(CHECKPOINT), trust_remote_code=False)
    ).text_config
    out = []
    with pytest.MonkeyPatch.context() as mp:
        for rank in range(TP):
            mp.setattr(weight_loaders_vision, "encoder_tp", lambda r=rank: (r, TP))
            model = _stand_in(TP)
            mappings, _ = checkpoint_mappings(model, text_config)
            load_weights(model, text_config, CHECKPOINT, torch.device("cpu"), pipelined=False)
            out.append((dict(model.named_parameters()), mappings))
    return out


def test_every_vision_tensor_maps_once_and_fills_a_parameter(ranks):
    keys = _vision_keys()
    counts = []
    for params, mappings in ranks:
        refs = Counter()
        for value in mappings.values():
            refs.update([value] if isinstance(value, str) else value)
        counts.append((
            len(keys),
            sum(refs[k] == 0 for k in keys),
            sum(refs[k] > 1 for k in keys),
            sum(p.is_meta for p in params.values()),
        ))
    print("keys, unmapped, mapped twice, empty, per rank:", counts)
    assert counts == [(168, 0, 0, 0)] * TP


def _undo_half_split(rows: torch.Tensor) -> torch.Tensor:
    """Return stored-order rows from rows whose heads hold rotary pairs as split halves."""
    order = torch.cat([torch.arange(0, HEAD_DIM, 2), torch.arange(1, HEAD_DIM, 2)])
    inverse = torch.argsort(order)
    return rows.reshape(-1, HEAD_DIM, rows.shape[-1])[:, inverse].reshape(rows.shape)


def _rebuilt(key: str, shards: list[torch.Tensor]) -> list[torch.Tensor]:
    """The stored tensor rebuilt from the rank tensors: one join, or one per rank if whole."""
    if key.endswith(".wqkv.weight"):
        parts = [s.T.chunk(3) for s in shards]
        q, k, v = (torch.cat([p[j] for p in parts]) for j in range(3))
        return [torch.cat([_undo_half_split(q), _undo_half_split(k), v])]
    if key.endswith((".wo.weight", ".mlp.fc1.weight")):
        return [torch.cat([s.T for s in shards], dim=1)]
    if key.endswith(".mlp.fc0.weight"):
        return [torch.cat([s.T for s in shards], dim=0)]
    if key == PATCH_KEY:
        return [s.T.reshape(-1, 3, P, P) for s in shards]
    if key.startswith("mm_projector.proj."):
        return [s.T for s in shards]
    return list(shards)


def test_rank_shards_rejoin_to_the_stored_tensors(ranks):
    stored = _read(_vision_keys())
    joined, whole, differing = 0, 0, 0
    for key, tensor in stored.items():
        rebuilt = _rebuilt(key, [params[PARENT + key] for params, _ in ranks])
        joined += len(rebuilt) == 1
        whole += len(rebuilt) == TP
        for t in rebuilt:
            assert t.shape == tensor.shape, key
            differing += int((t.view(torch.int16) != tensor.view(torch.int16)).sum())
    print("joined, whole, differing elements:", (joined, whole, differing))
    assert (joined, whole, differing) == (4 * LAYERS, len(stored) - 4 * LAYERS, 0)


def _reference_freqs_cis(height: int, width: int) -> torch.Tensor:
    """The checkpoint's 2D rotary table for one image, [h * w, d / 2] complex64."""
    rows, cols = torch.meshgrid(
        torch.arange(height).float(), torch.arange(width).float(), indexing="ij"
    )
    x_pos, y_pos = cols.reshape(-1), rows.reshape(-1)
    dim_range = torch.arange(0, HEAD_DIM, 4)[: HEAD_DIM // 4].float()
    freqs = 1.0 / (10000 ** (dim_range / HEAD_DIM))
    x_freqs, y_freqs = torch.outer(x_pos, freqs), torch.outer(y_pos, freqs)
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    return torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1).reshape(
        height * width, -1
    )


def _reference_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """The checkpoint's rotary embedding on interleaved pairs, x [N, heads, d]."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs_cis.unsqueeze(-2)).flatten(-2)


def _split_half_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """RoPE(X) = [X1, X2] cos + [-X2, X1] sin, each frequency in both halves."""
    cos = torch.cat([freqs_cis.real, freqs_cis.real], dim=-1).unsqueeze(-2)
    sin = torch.cat([freqs_cis.imag, freqs_cis.imag], dim=-1).unsqueeze(-2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def _relative_error(ours: torch.Tensor, ref: torch.Tensor) -> float:
    """Largest absolute difference over the reference's largest magnitude."""
    return ((ours - ref).abs().max() / ref.abs().max()).item()


def test_split_half_scores_equal_the_interleaved_reference(ranks):
    n, per_rank = GRID * GRID, HEADS // TP
    x = torch.randn(n, HIDDEN, generator=torch.Generator().manual_seed(0))
    freqs_cis = _reference_freqs_cis(GRID, GRID)
    heads, worst = 0, 0.0
    for layer in (0, LAYERS - 1):
        key = f"vision_tower.encoder.blocks.{layer}.wqkv.weight"
        ref_qkv = (x @ _read([key])[key].float().T).reshape(n, 3, HEADS, HEAD_DIM)
        ref_q = _reference_rope(ref_qkv[:, 0], freqs_cis)
        ref_k = _reference_rope(ref_qkv[:, 1], freqs_cis)
        for rank, (params, _) in enumerate(ranks):
            qkv = (x @ params[PARENT + key].float()).reshape(n, 3, per_rank, HEAD_DIM)
            q = _split_half_rope(qkv[:, 0], freqs_cis)
            k = _split_half_rope(qkv[:, 1], freqs_cis)
            for j in range(per_rank):
                head = rank * per_rank + j
                ref = ref_q[:, head] @ ref_k[:, head].T
                worst = max(worst, _relative_error(q[:, j] @ k[:, j].T, ref))
                heads += 1
    print(f"heads {heads}, worst relative error {worst:.3e}")
    assert heads == 2 * HEADS
    assert worst <= 1e-6, worst


def test_patch_matmul_equals_the_reference_conv(ranks):
    n = GRID * GRID
    pixels = torch.rand(n, 3, P, P, generator=torch.Generator().manual_seed(1)) * 2 - 1
    ref = F.conv2d(pixels, _read([PATCH_KEY])[PATCH_KEY].float(), stride=P).reshape(n, -1)
    ours = pixels.reshape(n, -1) @ ranks[0][0][PARENT + PATCH_KEY].float()
    error = _relative_error(ours, ref)
    print(f"patches {n}, relative error {error:.3e}")
    assert error <= 1e-6
