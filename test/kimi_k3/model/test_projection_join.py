# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 projection call sites against fp64 on the weights as loaded.

The normed sites run the fused norm-and-projection kernel on weights stacked at load,
and the unnormed prefill sites run the library's prefill output projection. Two
one-layer models run prefill to the logits: KDA with the dense MLP, and MLA with the
MoE block, whose attention and routed experts are fixed rows here. The decode sites run
on the same layers; the KDA decode step runs the decode kernel on a bound page buffer. Set
``KIMI_K3_CHECKPOINT`` to the checkpoint directory (default ``~/models/moonshotai/Kimi-K3``);
run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import copy
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_neuron.functional.attention import o_proj as output_projection
from vllm_neuron.model.kimi_k3 import model as m

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

TOKENS = 256
# One state slot's page as the runner lays it: conv rows, then the recurrent state.
PAGE, CONV_BYTES, STATE_END = 294_912, 6912, 203_520
EPS = 1e-5  # the checkpoint's rms_norm_eps
KDA_INPUTS = ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj")
ATTN, MLP, MOE = "input_layernorm", "post_attention_layernorm", "block_sparse_moe."
SHARED = [f"{MOE}shared_experts.gate_proj", f"{MOE}shared_experts.up_proj"]
# Each site: its model, the norm gain it applies (none, or the gain decode passes), and the
# [out, in] checkpoint weights it projects with, in output order.
SITES = {
    "kda_input_prefill": ("kda", ATTN, [f"self_attn.{p}" for p in KDA_INPUTS]),
    "kda_forget_prefill": ("kda", None, ["self_attn.f_b_proj"]),
    "mlp_gate_up_prefill": ("kda", MLP, ["mlp.gate_proj", "mlp.up_proj"]),
    "mlp_down_prefill": ("kda", None, ["mlp.down_proj"]),
    "lm_head": ("kda", "model.norm", ["lm_head"]),
    "kda_input_decode": ("kda", "decode", [f"self_attn.{p}" for p in KDA_INPUTS]),
    "mla_latent_prefill": ("mla", ATTN, ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"]),
    "mla_gate_prefill": ("mla", ATTN, ["self_attn.g_proj"]),
    "mla_output_prefill": ("mla", None, ["self_attn.o_proj"]),
    "moe_down_router_prefill": ("mla", MLP, [f"{MOE}routed_expert_down_proj", f"{MOE}gate"]),
    "moe_latent_up_prefill": ("mla", None, [f"{MOE}routed_expert_up_proj"]),
    "shared_gate_up_prefill": ("mla", MLP, SHARED),
    "shared_down_prefill": ("mla", None, [f"{MOE}shared_experts.down_proj"]),
    "mla_gate_decode": ("mla", "decode", ["self_attn.g_proj"]),
    "shared_gate_up_decode": ("mla", "decode", SHARED),
}


def _one_layer(hf_config: dict, kda: bool) -> m.KimiK3ForConditionalGeneration:
    """A one-layer model on meta at one rank's widths: KDA and dense MLP, or MLA and MoE."""
    config = copy.deepcopy(hf_config)
    text = config["text_config"]
    text.update(num_hidden_layers=1, num_attention_heads=3, intermediate_size=1056,
                moe_intermediate_size=96, routed_expert_hidden_size=112, num_experts=64,
                vocab_size=5120, first_k_dense_replace=1 if kda else 0)
    text["linear_attn_config"].update(
        num_heads=3, kda_layers=[1] if kda else [], full_attn_layers=[] if kda else [1]
    )
    with torch.device("meta"):
        return m.KimiK3ForConditionalGeneration.from_configs(config)


def _fill(model: nn.Module, gen: torch.Generator) -> dict[str, torch.Tensor]:
    """Give every parameter random values; returns fp64 copies of them as loaded."""
    for name, p in list(model.named_parameters()):
        path, _, leaf = name.rpartition(".")
        if not p.dtype.is_floating_point:
            value = torch.zeros(p.shape, dtype=p.dtype)
        elif "norm" in name:
            value = 1 + 0.1 * torch.randn(p.shape, generator=gen)
        else:
            scale = 0.5 if leaf in ("A_log", "dt_bias") or "conv" in name else 0.02
            value = scale * torch.randn(p.shape, generator=gen)
        param = nn.Parameter(value.to(p.dtype), requires_grad=False)
        setattr(model.get_submodule(path), leaf, param)
    loaded = model.named_parameters()
    return {n.removeprefix("model.layers.0."): p.detach().double() for n, p in loaded}


def _prefill_meta() -> dict:
    """One fresh sequence of real tokens in state slot 1."""
    return {
        "block_table_tensor": torch.tensor([[1]], dtype=torch.int32),
        "slot_mapping": torch.arange(TOKENS) + 1,
        "cached_seq_len": torch.tensor([[0]], dtype=torch.int32),
        "max_query_len": TOKENS,
        "decode_token_threshold": 1,
    }


def _no_fallback(*args, **kwargs):
    raise AssertionError("the prefill output projection computed in torch")


def _page_buffer(slots: int) -> list[torch.Tensor]:
    """A zeroed state page buffer and its conv and state views, as the runner binds them."""
    pages = torch.zeros(slots, PAGE, dtype=torch.int8)
    conv = pages[:, :CONV_BYTES].view(torch.bfloat16).view(slots, 3, 1152)
    state = pages[:, CONV_BYTES:STATE_END].view(torch.float32).view(slots, 3, 128, 128)
    return [conv, state, pages]


@pytest.fixture(scope="module")
def sites() -> dict:
    """Each site's input, output, fp64 norm gain (or None) and fp64 checkpoint weights."""
    from vllm.transformers_utils.config import get_config

    hf_config = get_config(str(CHECKPOINT), trust_remote_code=False).to_dict()
    gen, calls = torch.Generator().manual_seed(0), []

    def record(fn):
        def wrapper(x, weight, *norm):
            out = fn(x, weight, *norm)
            calls.append((x, out))
            return out
        return wrapper

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(output_projection, "_torch_o_proj_impl", _no_fallback)
        mp.setattr(m, "_project", record(m._project))
        mp.setattr(m, "_prefill_projection", record(m._prefill_projection))
        mp.setattr(m, "router_select", lambda logits, bias, **kwargs: (None, None))
        x = (0.5 * torch.randn(4, 7168, generator=gen)).bfloat16()
        gain = (1 + 0.1 * torch.randn(7168, generator=gen)).bfloat16()
        k_slice = torch.randn(128, 4, 56, 1, generator=gen)

        a = _one_layer(hf_config, kda=True)
        weights = {"kda": _fill(a, gen)}
        a.prepare_after_load()
        kda = a.model.layers[0].self_attn
        a.bind_kv_cache({"layers.0.self_attn": _page_buffer(2)})
        ids = torch.randint(0, 5120, (TOKENS,), generator=gen)
        logits = a(ids, None, {"layers.0.self_attn": _prefill_meta()})
        meta = {"block_table_tensor": torch.tensor([[1], [0], [0], [0]], dtype=torch.int32)}
        kda.forward_decode(x, None, gain, meta)
        kda_calls = calls[:]

        calls.clear()
        b = _one_layer(hf_config, kda=False)
        weights["mla"] = _fill(b, gen)
        b.prepare_after_load()
        mla, moe = b.model.layers[0].self_attn, b.model.layers[0].block_sparse_moe
        attended = (0.5 * torch.randn(TOKENS, 384, generator=gen)).bfloat16()
        mla._attend = lambda latent, gate, meta: attended[: latent.shape[0]]
        combined = torch.randn(TOKENS, 112, generator=gen)
        moe._select_and_run = lambda latent, *args: combined[: latent.shape[0]]
        moe.tail_decode = lambda combined, gate_up: None
        b(ids, None, {"layers.0.self_attn": _prefill_meta()})
        mla._attend = lambda latent, gate, meta: m._decode_layout(attended[: gate.shape[0]])
        mla.forward_decode(x, k_slice, gain, None)
        moe.forward_decode(x, k_slice, gain)
        mla_calls = calls[:]

    kda_in, forget, gate_up, down, head, _, kda_decode = kda_calls
    head = (head[0], logits)
    *mla_prefill, _, _, gate_decode, shared_decode = mla_calls
    found = dict(zip(SITES, [kda_in, forget, gate_up, down, head, kda_decode, *mla_prefill,
                             gate_decode, shared_decode], strict=True))
    for loaded in weights.values():
        loaded["decode.weight"] = gain.double()
    return {site: (*found[site], norm and weights[model][f"{norm}.weight"],
                   [weights[model][f"{w}.weight"] for w in names])
            for site, (model, norm, names) in SITES.items()}


def _reference(x, gain, weights):
    """fp64 RMSNorm with ``gain`` (unless None), then times the ``[out, in]`` weights."""
    x = x.double()
    if gain is not None:
        x = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * gain
    return x @ torch.cat(weights).t()


@pytest.mark.parametrize("site", SITES)
def test_site_matches_fp64(sites, site):
    """Row cosines with fp64 are at least 0.9999; the largest error is at most 2^-6 of the peak."""
    x, out, gain, weights = sites[site]
    ref, out = _reference(x, gain, weights), out.double()
    cosine = F.cosine_similarity(out, ref, dim=-1).min().item()
    error = ((out - ref).abs().max() / ref.abs().max()).item()
    assert cosine >= 0.9999 and error <= 2**-6, f"lowest row cosine {cosine:.6f}, error {error:.1e}"


def test_prefill_projection_raises_outside_the_kernel_limits():
    """A width past the kernel's column groups raises and names that limit."""
    rows = torch.zeros(8, 2304, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="18 column groups of 128, at most 17"):
        m._prefill_projection(rows, torch.zeros(2304, 384, dtype=torch.bfloat16))
