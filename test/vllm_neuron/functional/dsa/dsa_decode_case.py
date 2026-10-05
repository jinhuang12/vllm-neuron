# SPDX-License-Identifier: Apache-2.0
"""One DSA/MLA layer and one decode step's operands, shared by the bypass tests and the
hardware benchmark.

The geometry is this checkpoint's per-rank decode shape at TP=64 -- one MLA head, latent
rank 512, head widths 256, the indexer's 32 heads of 128, ``index_topk`` 2048 and
``index_kpool`` 4 -- with the hidden and query-latent widths narrowed when a caller asks,
because only those two dials scale the simulator's projection cost and neither reaches
the bypass bound. The four MLA projections are fp8-e4m3 bytes with a 128 x 128 scale grid
and ``kv_b_proj`` and the indexer's four are bf16, as the published checkpoint stores
them.

The builder takes the model module as an argument so the live tree and the 5938748
snapshot build the same layer from the same draws.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import FP8_SCALE_SUFFIX

PAGE = 128
BLOCK = 128
#: fp8 bytes stay inside +-224, the trn2 range the checkpoint load squeezes them into.
FP8_LIMIT = 224.0


def decode_config(*, hidden_size: int = 4096, q_lora_rank: int = 1536) -> Glm5NextTextConfig:
    """The checkpoint config at one MLA head (TP=64's per-rank count)."""
    base = Glm5NextTextConfig()
    return replace(
        base,
        hidden_size=int(hidden_size),
        q_lora_rank=int(q_lora_rank),
        num_attention_heads=1,
    )


def _fp8_weight(out_features: int, in_features: int, gen: torch.Generator):
    """fp8 bytes and an ``[out/128, in/128]`` scale grid whose product has unit-ish rows."""
    raw = (torch.randn(out_features, in_features, generator=gen) * 48.0).clamp(
        -FP8_LIMIT, FP8_LIMIT
    )
    grid = (
        torch.rand(out_features // BLOCK, in_features // BLOCK, generator=gen) * 0.5 + 0.75
    ) * (in_features ** -0.5) / 48.0
    return raw.to(torch.float8_e4m3fn), grid.to(torch.float32)


def build_attention(model_module, cfg: Glm5NextTextConfig, *, seed: int = 4242,
                    device: torch.device | str | None = None):
    """A prepared ``Glm5NextMLAAttention`` (with its indexer) from ``model_module``.

    With ``device`` the parameters move there before the prepares run, so every prepared
    operand is device-resident, as the runner's are.
    """
    gen = torch.Generator().manual_seed(int(seed))
    module = model_module.Glm5NextMLAAttention(cfg)
    fp8_sites = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj")
    for name, in_features, out_features in module.projection_widths():
        if name in fp8_sites:
            weight, grid = _fp8_weight(out_features, in_features, gen)
            setattr(module, f"{name}_weight", torch.nn.Parameter(weight, requires_grad=False))
            setattr(module, f"{name}_{FP8_SCALE_SUFFIX}",
                    torch.nn.Parameter(grid, requires_grad=False))
        else:
            weight = (torch.randn(out_features, in_features, generator=gen)
                      * in_features ** -0.5).to(torch.bfloat16)
            setattr(module, f"{name}_weight", torch.nn.Parameter(weight, requires_grad=False))
    for name, width in (("q_a_layernorm_weight", int(cfg.q_lora_rank)),
                        ("kv_a_layernorm_weight", int(cfg.kv_lora_rank))):
        gain = 1.0 + torch.randn(width, generator=gen) * 0.05
        setattr(module, name, torch.nn.Parameter(gain, requires_grad=False))
    indexer = module.indexer
    for name, in_features, out_features in indexer.projection_widths():
        weight = (torch.randn(out_features, in_features, generator=gen)
                  * in_features ** -0.5).to(torch.bfloat16)
        setattr(indexer, indexer.PROJECTION_PARAMETERS[name],
                torch.nn.Parameter(weight, requires_grad=False))
    head_dim, pool = int(indexer.index_head_dim), int(indexer.index_kpool)
    indexer.k_norm_weight = torch.nn.Parameter(
        1.0 + torch.randn(head_dim, generator=gen) * 0.05, requires_grad=False)
    indexer.k_norm_bias = torch.nn.Parameter(
        torch.randn(head_dim, generator=gen) * 0.02, requires_grad=False)
    indexer.index_kpool_compress_ape = torch.nn.Parameter(
        (torch.randn(pool, head_dim, generator=gen) * 0.1).to(torch.bfloat16),
        requires_grad=False)
    if device is not None:
        module.to(device)
    assert module.prepare_projection_weights() == 5
    assert module.prepare_absorb_weights() == 2
    assert indexer.prepare_projection_weights() == 4
    return module


def decode_operands(cfg: Glm5NextTextConfig, context: int, *, window_pages: int,
                    max_seq_len: int, seed: int = 99, bank_pages: int | None = None):
    """Keyword operands for ``Glm5NextMLAAttention.forward`` at one decode step.

    ``context`` counts this step's token: it sits at window row ``context - 1``. The
    request holds ``window_pages`` pages of a bank of ``bank_pages`` (shuffled, so the
    window is not one ascending run), and the pooled-key store holds every pool the
    largest candidate axis can address plus the trash row.
    """
    gen = torch.Generator().manual_seed(int(seed) + int(context))
    pool = int(cfg.index_kpool)
    head_dim = int(cfg.index_head_dim)
    latent = int(cfg.kv_lora_rank)
    pages = int(window_pages)
    if context > pages * PAGE:
        raise ValueError(f"context {context} does not fit {pages} page(s) of {PAGE}")
    bank_pages = int(bank_pages or pages + 3)
    table = torch.randperm(bank_pages, generator=gen)[:pages].to(torch.int32)
    used = -(-int(context) // PAGE)
    table[used:] = -1
    cache = (torch.randn(bank_pages * PAGE, 1, latent, generator=gen) * 0.5).to(torch.bfloat16)
    candidates_max = int(max_seq_len) // pool
    pool_rows = -(-(candidates_max + 1) // PAGE) * PAGE
    pool_cache = (torch.randn(pool_rows, head_dim, generator=gen) * 0.5).to(torch.bfloat16)
    position = int(context) - 1
    page_of = int(table[position // PAGE])
    slot = page_of * PAGE + position % PAGE
    return {
        "normed_hidden_states": (torch.randn(1, int(cfg.hidden_size), generator=gen)
                                 * 0.5).to(torch.bfloat16),
        "latent_cache": cache,
        "pool_cache": pool_cache,
        "seq_lens": torch.tensor([int(context)], dtype=torch.int32),
        "start_position": torch.tensor(position, dtype=torch.int64),
        "softmax_scale": float(int(cfg.qk_nope_head_dim) + int(cfg.qk_rope_head_dim)) ** -0.5,
        "max_seq_len": int(max_seq_len),
        "page_size": PAGE,
        "block_table_row": table.reshape(pages, 1),
        "latent_slots": torch.tensor([slot], dtype=torch.int64),
        "tail": (torch.randn(2, pool, head_dim, generator=gen) * 0.5).to(torch.bfloat16),
        "position": torch.tensor(position, dtype=torch.int64),
    }


def cloned(operands: dict) -> dict:
    """The same operands with every tensor copied, so a forward's in-place writes stay local."""
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in operands.items()}
