# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's weight layout for the device-free compile, which loads no weights.

The served DCP 4 form is built on meta on 8 gloo ranks, TP 4 by DP 2 with EP 8, from the
checkpoint's config with every width the served TP 32 by EP 64 form splits cut by 8, so each
rank holds the served form's per-rank shapes. After ``load_weights_lite`` every tensor the
kernels read must exist with the shape and dtype the config gives. A 4-layer model at one
rank, loaded from a random checkpoint after a lite load, must hold the same kernel tensors,
byte for byte, as one loaded alone. Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory
(its config only); run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import os
import types
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from vllm_neuron.model.kimi_k3 import model as m
from vllm_neuron.model.kimi_k3 import weight_loaders_mxfp4
from vllm_neuron.model.neuron_config import NeuronConfig

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

TP, DP, SCALE, SLOTS = 4, 2, 8, 3
SPLIT = ("num_attention_heads", "num_key_value_heads", "num_experts", "intermediate_size",
         "routed_expert_hidden_size", "vocab_size")  # the widths the served form splits
BF16, F32 = torch.bfloat16, torch.float32

# The tensors each kind of module hands its kernels, by attribute path.
READS = {
    "layer": ("attention_score_w", "mlp_score_w", "input_layernorm.weight",
              "post_attention_layernorm.weight"),
    "kda": ("input_weight", "conv_weight", "forget_weight", "output_weight", "q_conv1d.weight",
            "k_conv1d.weight", "v_conv1d.weight", "A_log", "dt_bias", "f_b_proj.weight",
            "o_norm.weight", "o_proj.weight"),
    "mla": ("input_weight", "gate_weight", "latent_weight", "output_weight", "w_qb_t", "w_uk",
            "w_uv_t", "q_a_layernorm.weight", "kv_a_layernorm.weight"),
    "mlp": ("gate_up_weight", "down_weight"),
    "moe": ("down_router_weight", "latent_norm_weight", "latent_up_weight", "shared_down_weight",
            "gate.e_score_correction_bias",
            *(f"experts.{part}" for part in weight_loaders_mxfp4.EXPERT_PARTS),
            "shared_experts.gate_up_weight", "shared_experts.down_weight"),
    "model": ("embed_tokens.weight", "norm.weight", "output_score_w"),
}


def _hf_config() -> dict:
    """The checkpoint's HF config as vLLM builds it, as a dict."""
    from vllm.transformers_utils.config import get_config

    return get_config(str(CHECKPOINT), trust_remote_code=False).to_dict()


def _served_config() -> dict:
    """The config whose TP 4 by EP 8 ranks hold the served TP 32 by EP 64 per-rank shapes."""
    config = _hf_config()
    text = config["text_config"]
    for key in SPLIT:
        text[key] //= SCALE
    text["linear_attn_config"]["num_heads"] //= SCALE
    return config


def _small_config() -> dict:
    """test_dp.py's 4-layer config at one rank: dense then MoE, three KDA layers then MLA."""
    config = _hf_config()
    text = config["text_config"]
    text.update(num_hidden_layers=4, hidden_size=7168, num_attention_heads=12, num_experts=64,
                intermediate_size=4224, moe_intermediate_size=3072, routed_expert_hidden_size=512,
                vocab_size=10240, first_k_dense_replace=1)
    text["linear_attn_config"].update(num_heads=12, kda_layers=[1, 2, 3], full_attn_layers=[4])
    return config


def _get(module, path: str):
    """The attribute at a dotted path, or None where any step is missing."""
    for part in path.split("."):
        module = getattr(module, part, None)
    return module


def _kernel_tensors(model) -> dict[str, tuple[str, str, torch.Tensor | None]]:
    """Every tensor the forward hands a kernel, by tree name: its kind, attribute and tensor."""
    out = {f"model.{a}": ("model", a, _get(model.model, a)) for a in READS["model"]}
    for i in range(2):
        half = None if model.head_weights is None else model.head_weights[i]
        out[f"head_weights.{i}"] = ("head", "half", half)
    for i, layer in enumerate(model.model.layers):
        ffn = "moe" if hasattr(layer, "block_sparse_moe") else "mlp"
        sites = [("", "layer", layer), ("self_attn.", "kda" if layer.is_kda else "mla",
                                        layer.self_attn),
                 ("block_sparse_moe." if ffn == "moe" else "mlp.", ffn, layer.feed_forward)]
        for prefix, kind, module in sites:
            for a in READS[kind]:
                out[f"model.layers.{i}.{prefix}{a}"] = (kind, a, _get(module, a))
    return out


def _expected(text: dict, programs: int) -> dict[str, dict[str, tuple]]:
    """Each kind's kernel tensors at one rank of the TP 4 by EP 8 world: shape and dtype."""
    hidden, vocab = text["hidden_size"], text["vocab_size"] // TP
    lin = text["linear_attn_config"]
    kda_heads, dim, taps = lin["num_heads"] // TP, lin["head_dim"], lin["short_conv_kernel_size"]
    width = kda_heads * dim
    heads = text["num_attention_heads"] // TP
    q_rank, kv_rank = text["q_lora_rank"], text["kv_lora_rank"]
    nope, rope, v_dim = text["qk_nope_head_dim"], text["qk_rope_head_dim"], text["v_head_dim"]
    dense, latent, experts = text["intermediate_size"] // TP, text["routed_expert_hidden_size"], \
        text["num_experts"]
    shared = text["moe_intermediate_size"] * text["num_shared_experts"] // TP
    bank_config = types.SimpleNamespace(routed_expert_hidden_size=latent,
                                        moe_intermediate_size=text["moe_intermediate_size"])
    entries = experts // (TP * DP) + SLOTS
    banks = {f"experts.{part}": ((entries, *shape), dtype)
             for part, (shape, dtype) in m.expert_bank_shapes(bank_config, programs).items()}
    return {
        "model": {"embed_tokens.weight": ((vocab, hidden), BF16), "norm.weight": ((hidden,), BF16),
                  "output_score_w": ((hidden,), F32)},
        "head": {"half": ((hidden, vocab // 2), BF16)},
        "layer": {"attention_score_w": ((hidden,), F32), "mlp_score_w": ((hidden,), F32),
                  "input_layernorm.weight": ((hidden,), BF16),
                  "post_attention_layernorm.weight": ((hidden,), BF16)},
        "kda": {"input_weight": ((hidden, 4 * width + dim + kda_heads), BF16),
                "conv_weight": ((3 * width, taps), F32), "forget_weight": ((dim, width), BF16),
                "output_weight": ((width, hidden), BF16),
                **{f"{x}_conv1d.weight": ((width, 1, taps), F32) for x in "qkv"},
                "A_log": ((kda_heads,), F32), "dt_bias": ((width,), F32),
                "f_b_proj.weight": ((width, dim), BF16), "o_norm.weight": ((dim,), F32),
                "o_proj.weight": ((hidden, width), BF16)},
        "mla": {"input_weight": ((hidden, q_rank + kv_rank + rope), BF16),
                "gate_weight": ((hidden, heads * v_dim), BF16),
                "latent_weight": ((hidden // TP, q_rank + kv_rank + rope), F32),
                "output_weight": ((heads * v_dim, hidden), BF16),
                "w_qb_t": ((q_rank, heads * (nope + rope)), BF16),
                "w_uk": ((heads, nope, kv_rank), BF16), "w_uv_t": ((heads, kv_rank, v_dim), F32),
                "q_a_layernorm.weight": ((q_rank,), BF16),
                "kv_a_layernorm.weight": ((kv_rank,), BF16)},
        "mlp": {"gate_up_weight": ((hidden, 2 * dense), BF16),
                "down_weight": ((dense, hidden), BF16)},
        "moe": {"down_router_weight": ((hidden // TP, latent + experts), F32),
                "latent_norm_weight": ((latent // TP, 1), F32),
                "latent_up_weight": ((latent // TP, hidden), BF16),
                "shared_down_weight": ((shared, hidden), BF16),
                "gate.e_score_correction_bias": ((experts,), F32), **banks,
                "shared_experts.gate_up_weight": ((hidden, 2 * shared), BF16),
                "shared_experts.down_weight": ((shared, hidden), BF16)},
    }


def _lite_rank(rank: int, root: Path) -> None:
    """One rank of the served form after the lite load: each kernel tensor's shape and dtype."""
    from vllm_neuron.parallel.neuron_parallel_state import initialize_neuron_parallel_state

    replica = rank // TP
    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv", rank=rank,
                            world_size=TP * DP)
    initialize_neuron_parallel_state(ep_degree=TP * DP, dp_size=DP, dcp_size=TP, local_rank=rank,
                                     tp_global_ranks=[replica * TP + i for i in range(TP)])
    with torch.device("meta"):
        model = m.KimiK3ForConditionalGeneration.from_configs(
            _served_config(), NeuronConfig(expert_replica_slots=SLOTS))
    model.load_weights_lite(str(CHECKPOINT), torch.device("cpu"), None)
    reads = {name: (kind, a, None if t is None else (tuple(t.shape), t.dtype))
             for name, (kind, a, t) in _kernel_tensors(model).items()}
    torch.save({"reads": reads, "programs": model.num_programs}, root / f"rank{rank}.pt")
    dist.destroy_process_group()


def test_lite_load_lays_out_every_kernel_tensor_at_the_served_rank_shapes(tmp_path):
    """On every rank of TP 4 by EP 8, each tensor a kernel reads has its served shape and dtype."""
    mp.spawn(_lite_rank, args=(tmp_path,), nprocs=TP * DP, join=True)
    text = _served_config()["text_config"]
    names, wrong = set(), {}
    for rank in range(TP * DP):
        run = torch.load(tmp_path / f"rank{rank}.pt")
        expected = _expected(text, run["programs"])
        names.add(frozenset(run["reads"]))
        for name, (kind, a, got) in run["reads"].items():
            if got != expected[kind][a]:
                wrong[(rank, name)] = got
    layers = sum(name.endswith(".attention_score_w") for name in next(iter(names)))
    assert len(names) == 1 and layers == text["num_hidden_layers"], \
        f"{len(names)} sets of kernel tensor names over the ranks, {layers} layers"
    assert not wrong, f"missing, or not the served shape and dtype, by rank and name: {wrong}"


def _write_checkpoint(model, root: Path) -> None:
    """Random stored tensors for every checkpoint key the model maps, in test_dp.py's form."""
    mappings, _ = weight_loaders_mxfp4.checkpoint_mappings(model, model.text_config)
    params, text = dict(model.named_parameters()), model.text_config
    width, latent = text.moe_intermediate_size, text.routed_expert_hidden_size
    stored_shape = {"w1": (width, latent), "w3": (width, latent), "w2": (latent, width)}
    gen, stored = torch.Generator().manual_seed(7), {}
    for name, keys in mappings.items():
        if isinstance(keys, str):
            shape, dtype = params[name].shape, params[name].dtype
            stored[keys] = (0.02 * torch.randn(shape, generator=gen)).to(dtype)
            continue
        for key in keys:
            w, kind = key.split(".")[-2:]
            rows, cols = stored_shape[w]
            div, low, high = (2, 0, 256) if kind == "weight_packed" else (32, 120, 123)
            stored[key] = torch.randint(low, high, (rows, cols // div), generator=gen,
                                        dtype=torch.uint8)
    save_file(stored, str(root / "model.safetensors"))


def _same_bytes(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    """True when both hold data of one dtype and shape, with the same bytes."""
    if a is None or b is None or a.is_meta or b.is_meta:
        return False
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def test_full_load_after_a_lite_load_equals_a_full_load_alone(tmp_path):
    """After a lite load and then the full load, each kernel tensor equals a full load's bytes."""
    config = _small_config()
    with torch.device("meta"):
        writer, alone, relaid = (m.KimiK3ForConditionalGeneration.from_configs(config)
                                 for _ in range(3))
    _write_checkpoint(writer, tmp_path)
    alone.load_weights(str(tmp_path), torch.device("cpu"))
    relaid.load_weights_lite(str(tmp_path), torch.device("cpu"), None)
    relaid.load_weights(str(tmp_path), torch.device("cpu"))
    want, got = _kernel_tensors(alone), _kernel_tensors(relaid)
    assert set(want) == set(got)
    differ = {name: "meta" if t is not None and t.is_meta else "differs"
              for name, (_, _, t) in got.items() if not _same_bytes(want[name][2], t)}
    assert not differ, f"kernel tensors not equal to the full load alone's: {differ}"
