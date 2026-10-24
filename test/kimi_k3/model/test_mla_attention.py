# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA attention call site on a bound latent cache, against the kernel's fp64 oracle.

A one-layer MLA model at one rank's widths takes the oracle's norm gains and ``q_b`` and
``kv_b`` slices as checkpoint weights, lays them out at load, and binds a paged cache whose
null block 0 and unowned rows are NaN. A decode step of two sequences and a padding row,
and a prefill chunk of one sequence after its cached prefix with padding tokens after it,
run through the call site on the kernels in the NKI simulator, which write the new latent
rows into the bound page buffer in place; prefill must take the full padded block table.
The oracle and its pass rule are the kernel test's. Set ``KIMI_K3_CHECKPOINT`` to the
checkpoint directory (default ``~/models/moonshotai/Kimi-K3``); run with
``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import copy
import importlib.util
import inspect
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm_neuron.functional.attention.mla_paged import prepare_mla_weights
from vllm_neuron.model.kimi_k3 import model as m

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

_spec = importlib.util.spec_from_file_location(
    "mla_paged_oracle", Path(__file__).parents[1] / "kernels" / "test_mla_paged.py"
)
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)

BLOCK, BLOCKS, WIDTH = 256, 8, 8
DECODE = {"prefixes": (300, 37), "blocks": ([4, 1], [6])}
PREFILL = {"prefix": 200, "chunk": 130, "padding": 6, "blocks": [2, 7]}


def _mla_layer(gen: torch.Generator, weights) -> m.KimiK3MLA:
    """A one-layer MLA model's attention with random weights and the oracle's MLA factors."""
    from vllm.transformers_utils.config import get_config

    config = copy.deepcopy(get_config(str(CHECKPOINT), trust_remote_code=False).to_dict())
    text = config["text_config"]
    text.update(num_hidden_layers=1, num_attention_heads=3, moe_intermediate_size=96,
                routed_expert_hidden_size=112, num_experts=64, vocab_size=5120,
                first_k_dense_replace=0)
    text["linear_attn_config"].update(num_heads=3, kda_layers=[], full_attn_layers=[1])
    with torch.device("meta"):
        model = m.KimiK3ForConditionalGeneration.from_configs(config)
    q_norm, kv_norm, q_b, kv_b = weights
    given = {"q_a_layernorm": q_norm, "kv_a_layernorm": kv_norm, "q_b_proj": q_b, "kv_b_proj": kv_b}
    for name, p in list(model.named_parameters()):
        path, _, leaf = name.rpartition(".")
        value = given.get(path.rpartition(".")[2])
        if value is None:
            value = torch.zeros(p.shape) if not p.dtype.is_floating_point else (
                0.02 * torch.randn(p.shape, generator=gen))
        param = nn.Parameter(value.to(p.dtype), requires_grad=False)
        setattr(model.get_submodule(path), leaf, param)
    model.prepare_after_load()
    return model


def _bind(model, prefixes, tables) -> torch.Tensor:
    """Bind a NaN cache holding each prefix's bf16 latent rows on its blocks; returns the cache."""
    cache = torch.full((BLOCKS, 1, BLOCK, oracle.LATENT), float("nan"), dtype=torch.bfloat16)
    for rows, blocks in zip(prefixes, tables):
        for j, block in enumerate(blocks):
            part = rows[j * BLOCK:(j + 1) * BLOCK]
            cache[block, 0, :part.shape[0]] = part.bfloat16()
    model.bind_kv_cache({"layers.0.self_attn": [cache]})
    return cache[:, 0]


def _table(blocks) -> torch.Tensor:
    """A block table row padded with -1 to the table width."""
    return torch.tensor(blocks + [-1] * (WIDTH - len(blocks)), dtype=torch.int32)


def _slot(blocks, position: int) -> int:
    return blocks[position // BLOCK] * BLOCK + position % BLOCK


@pytest.fixture(scope="module")
def seen():
    """The block table and prefix length the prefill kernel first receives."""
    args, kernel = [], m.mla_paged_prefill
    signature = inspect.signature(kernel)

    def call(*a):
        bound = signature.bind(*a).arguments
        args.append((bound["block_table"], bound["prefix_len"]))
        return kernel(*a)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m, "mla_paged_prefill", call)
        yield args


@pytest.fixture(scope="module")
def calls(seen) -> dict:
    """One decode step and one prefill chunk through the call site, each on its own cache."""
    gen = torch.Generator().manual_seed(20261024)
    weights = oracle._weights(gen)
    out = {"weights": weights}

    model = _mla_layer(gen, weights)
    mla = model.model.layers[0].self_attn
    out["layer"] = mla
    rows = [torch.randn(n + 1, oracle.QKV, generator=gen) for n in DECODE["prefixes"]]
    cache = _bind(model, [oracle._latent(r[:-1], weights[1]) for r in rows], DECODE["blocks"])
    slots = [_slot(b, n) for b, n in zip(DECODE["blocks"], DECODE["prefixes"])]
    meta = {
        "block_table_tensor": torch.stack([_table(b) for b in DECODE["blocks"]] + [_table([])]),
        "positions": torch.tensor([*DECODE["prefixes"], 0]),
        "slot_mapping": torch.tensor([*slots, -1]),
        "max_query_len": 1,
        "decode_token_threshold": 1,
    }
    new = torch.cat([torch.stack([r[-1] for r in rows]), torch.randn(1, oracle.QKV, generator=gen)])
    gate = torch.randn(3, oracle.HEADS * oracle.V_DIM, generator=gen).bfloat16()
    before = cache.clone()
    attn = mla._attend(new.float(), gate, meta)
    out["decode"] = {"rows": rows, "gate": gate, "attn": attn, "slots": slots,
                     "before": before, "after": cache.clone()}

    model = _mla_layer(gen, weights)
    p = PREFILL
    total = p["prefix"] + p["chunk"]
    seq = torch.randn(total, oracle.QKV, generator=gen)
    cache = _bind(model, [oracle._latent(seq[:p["prefix"]], weights[1])], [p["blocks"]])
    slots = [_slot(p["blocks"], i) for i in range(p["prefix"], total)]
    meta = {
        "block_table_tensor": _table(p["blocks"]).unsqueeze(0),
        "cached_seq_len": torch.tensor([[p["prefix"]]], dtype=torch.int32),
        "slot_mapping": torch.tensor(slots + [-1] * p["padding"]),
        "max_query_len": p["chunk"] + p["padding"],
        "decode_token_threshold": 1,
    }
    chunk = torch.cat([seq[p["prefix"]:], torch.zeros(p["padding"], oracle.QKV)])
    gate = torch.randn(chunk.shape[0], oracle.HEADS * oracle.V_DIM, generator=gen).bfloat16()
    before = cache.clone()
    attn = model.model.layers[0].self_attn._attend(chunk.float(), gate, meta)
    out["prefill"] = {"rows": seq, "gate": gate, "attn": attn, "slots": slots,
                      "before": before, "after": cache.clone()}
    return out


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def test_layouts_replace_q_b_and_kv_b(calls):
    """The kernels' layouts of q_b and kv_b replace them at load; only the value rows widen."""
    mla, (_, _, q_b, kv_b) = calls["layer"], calls["weights"]
    assert mla.q_b_proj.weight is None and mla.kv_b_proj.weight is None
    for got, expected in zip((mla.w_qb_t, mla.w_uk, mla.w_uv_t), prepare_mla_weights(q_b, kv_b)):
        assert got.dtype == expected.dtype and torch.equal(got, expected)
    added = sum(t.nbytes for t in (mla.w_qb_t, mla.w_uk, mla.w_uv_t)) - q_b.nbytes - kv_b.nbytes
    assert added == kv_b.nbytes // 2


def test_decode_step_matches_the_oracle(calls):
    """Each sequence's decode output, in the output projection's layout, meets the oracle's rule."""
    step = calls["decode"]
    attn = step["attn"]
    assert tuple(attn.shape) == (oracle.V_DIM, 3, oracle.HEADS, 1) and attn.dtype == torch.bfloat16
    out = attn[..., 0].permute(1, 2, 0).reshape(3, oracle.HEADS * oracle.V_DIM)
    for s, rows in enumerate(step["rows"]):
        ref, _ = oracle._reference(rows, step["gate"][s:s + 1], calls["weights"], 1)
        oracle._assert_within_bf16_rule(out[s:s + 1], ref)


def test_prefill_chunk_matches_the_oracle(calls):
    """The chunk's real rows of the ``[T, H * 128]`` prefill output meet the oracle's rule."""
    step, chunk = calls["prefill"], PREFILL["chunk"]
    attn = step["attn"]
    assert tuple(attn.shape) == (chunk + PREFILL["padding"], oracle.HEADS * oracle.V_DIM)
    ref, _ = oracle._reference(step["rows"], step["gate"][:chunk], calls["weights"], chunk)
    oracle._assert_within_bf16_rule(attn[:chunk], ref)


@pytest.mark.parametrize("kind", ["decode", "prefill"])
def test_latent_rows_land_on_their_slots_only(calls, kind):
    """New latent rows meet the oracle at their slots; padding writes only the null block."""
    step, weights = calls[kind], calls["weights"]
    after, before = step["after"].view(-1, oracle.LATENT), step["before"].view(-1, oracle.LATENT)
    if kind == "decode":
        refs = [oracle._reference(r, step["gate"][s:s + 1], weights, 1)[1]
                for s, r in enumerate(step["rows"])]
        ref = torch.cat(refs)
    else:
        ref = oracle._reference(step["rows"], step["gate"][:PREFILL["chunk"]], weights,
                                PREFILL["chunk"])[1]
    slots = torch.tensor(step["slots"])
    oracle._assert_within_bf16_rule(after[slots], ref)
    kept = torch.ones(after.shape[0], dtype=torch.bool)
    kept[slots] = False
    kept[:BLOCK] = False  # the null block takes the padding writes
    assert torch.equal(_bits(after[kept]), _bits(before[kept]))


def test_prefill_takes_the_full_padded_table(calls, seen):
    """The prefill kernel takes every entry of the table row, -1 padding included."""
    table, prefix = seen[0]
    assert torch.equal(table, _table(PREFILL["blocks"]))
    assert torch.equal(prefix, torch.tensor([PREFILL["prefix"]], dtype=torch.int32))


def test_outside_the_kernels_limits_raises(calls):
    """A prefill chunk of two sequences and a decode step of two tokens per sequence raise."""
    mla = calls["layer"]
    gate = torch.zeros(4, oracle.HEADS * oracle.V_DIM, dtype=torch.bfloat16)
    latent = torch.zeros(4, oracle.QKV)
    table = torch.stack([_table([1]), _table([2])])
    two = {"block_table_tensor": table, "cached_seq_len": torch.zeros(1, 1, dtype=torch.int32),
           "slot_mapping": torch.arange(4), "max_query_len": 2, "decode_token_threshold": 1}
    with pytest.raises(ValueError, match="takes one sequence, got 2"):
        mla._attend(latent, gate, two)
    spec = {**two, "positions": torch.zeros(4, dtype=torch.long), "decode_token_threshold": 2}
    with pytest.raises(ValueError, match=r"block_table must be \[T, pages\]"):
        mla._attend(latent, gate, spec)
