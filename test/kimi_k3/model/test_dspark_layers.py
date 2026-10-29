# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter: the block pass over its three layers, on two TP ranks.

Each rank loads its TP 2 slices of the draft checkpoint and upcasts the drafter to fp32. Its
latent caches are bf16, the dtype the attention kernel's window mode reads. Each of 4
sequences writes 64 context rows, then one block pass runs 4 rows per sequence: the embedding
rows of a random anchor and of 3 mask tokens. The fused norm-projection kernel has no CPU path,
so an fp32 torch form stands in for it. The rows after each layer are read against upstream's
draft forward in fp64, rounded to bf16 where the attention kernel rounds, within 1e-3 of each
sequence's largest value. Each rank runs on one thread. Set
``KIMI_K3_DSPARK_CHECKPOINT`` to the draft directory (default
``~/models/Inferact/Kimi-K3-DSpark-Block5``). Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import json
import math
import os
import socket
import warnings
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors import safe_open
from vllm.model_executor.layers.rotary_embedding.common import (
    rotate_gptj,
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
)
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import yarn_get_mscale

DRAFT = Path(
    os.environ.get(
        "KIMI_K3_DSPARK_CHECKPOINT",
        Path.home() / "models" / "Inferact" / "Kimi-K3-DSpark-Block5",
    )
)
WEIGHTS = DRAFT / "model.safetensors"

pytestmark = pytest.mark.skipif(
    not WEIGHTS.is_file(), reason=f"DSpark checkpoint not found at {DRAFT}"
)

TP, SEQS, ROWS, CONTEXT, BLOCK = 2, 4, 4, 64, 128
HIDDEN, LATENT, ROPE, NOPE, HEADS = 7168, 512, 64, 128, 96
# One cache block per sequence, out of order; block 0 is the null block.
TABLE = torch.tensor([[3], [1], [4], [2]], dtype=torch.int32)
CONTEXT_POSITIONS = torch.arange(CONTEXT).repeat(SEQS)
BLOCK_POSITIONS = (CONTEXT + torch.arange(ROWS)).repeat(SEQS)


def _config() -> dict:
    """The draft checkpoint's config with the fp32 dtype."""
    with open(DRAFT / "config.json") as f:
        config = json.load(f)
    config.update(dtype="float32", torch_dtype="float32")
    return config


def _slots(positions: torch.Tensor) -> torch.Tensor:
    """Each row's cache slot: its sequence's block, then its position in the block."""
    seqs = torch.arange(len(positions)) * SEQS // len(positions)
    return TABLE[seqs, 0].long() * BLOCK + positions


def _norm_projection(hidden, weight, norm_weight, eps):
    """Stand-in fused norm and projection, in fp32: ``[B, S, I]`` in the input's dtype."""
    x = hidden.float()
    rows = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm_weight.float()
    return (rows @ weight.float()).to(hidden.dtype)


def _rank(rank: int, port: int, inputs: dict, out: str) -> None:
    """One TP rank: the block pass on the clean rows after each layer, then with row j changed."""
    import torch.distributed as dist

    from vllm_neuron.model.kimi_k3 import dspark
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state,
        stamp_spmd_rank_tensors,
    )

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank)
    stamp_spmd_rank_tensors("cpu")
    with torch.device("meta"):
        draft = dspark.K3DSparkForCausalLM.from_configs(
            config=_config(), start_layer_idx=93, max_positions=CONTEXT + ROWS
        )
    draft.load_weights(str(DRAFT), torch.device("cpu"))
    draft.float()
    meta = {
        "slot_mapping": _slots(BLOCK_POSITIONS),
        "block_size": BLOCK,
        "block_table_tensor": TABLE,
        "max_blocks_per_seq": 1,
        "context_lens": torch.full((SEQS,), CONTEXT, dtype=torch.int32),
        "is_causal": True,
    }

    def block_pass(embedded):
        caches = {
            layer.cache_name: [torch.full((5, 1, BLOCK, LATENT + ROPE), float("nan"),
                                          dtype=torch.bfloat16)]
            for layer in draft.layers
        }
        draft.bind_kv_cache(caches)
        # One context write per sequence, 64 rows each, as the decode-size projection takes.
        slots = _slots(CONTEXT_POSITIONS)
        for rows in torch.arange(SEQS * CONTEXT).split(CONTEXT):
            draft.precompute_and_store_context_kv(
                inputs["taps"][rows], CONTEXT_POSITIONS[rows], slots[rows]
            )
        return draft.block_pass(
            embedded, BLOCK_POSITIONS, {layer.cache_name: meta for layer in draft.layers}
        )

    seen, attend = [], dspark.K3DSparkForCausalLM._attention

    def attention(self, layer, x, *args):
        seen.append(x)
        return attend(self, layer, x, *args)

    with pytest.MonkeyPatch.context() as patch, torch.no_grad():
        patch.setattr(dspark, "qkv_proj_fp32acc", _norm_projection)
        patch.setattr(dspark.K3DSparkForCausalLM, "_attention", attention)
        clean = block_pass(inputs["embedded"])
        # Layer k's output is layer k + 1's attention input.
        layers = [x.view(clean.shape) for x in seen[1 : len(draft.layers)]] + [clean]
        state = {"clean": clean, "layers": layers, "changed": {}}
        for j in range(1, ROWS):
            rows = inputs["embedded"].clone()
            rows[j::ROWS] = inputs["replacements"]
            state["changed"][j] = block_pass(rows)
    torch.save(state, os.path.join(out, f"rank{rank}.pt"))
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def inputs() -> dict:
    """Random taps, the block's embedding rows, and other tokens' rows to put at row j."""
    gen = torch.Generator().manual_seed(0)
    taps = torch.randn(SEQS * CONTEXT, 3 * HIDDEN, generator=gen)
    anchors = torch.randint(0, _config()["vocab_size"], (SEQS,), generator=gen).tolist()
    others = torch.randint(0, _config()["vocab_size"], (SEQS,), generator=gen).tolist()
    mask = _config()["mask_token_id"]
    ids = [token for anchor in anchors for token in [anchor] + [mask] * (ROWS - 1)]
    with safe_open(WEIGHTS, framework="pt") as f:
        table = f.get_slice("embed_tokens.weight")
        embedded = torch.stack([table[i : i + 1][0] for i in ids]).float()
        replacements = torch.stack([table[i : i + 1][0] for i in others]).float()
    return {"taps": taps, "embedded": embedded, "replacements": replacements}


@pytest.fixture(scope="module")
def ranks(tmp_path_factory, inputs) -> list[dict]:
    """Each rank's block-pass outputs, from one two-rank run."""
    out = tmp_path_factory.mktemp("ranks")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cwd, base = os.getcwd(), os.environ.get("VLLM_RPC_BASE_PATH")
    # vLLM's broadcaster binds an IPC socket under this path, which must stay short.
    os.environ["VLLM_RPC_BASE_PATH"] = "."
    os.chdir(out)
    try:
        mp.spawn(_rank, args=(port, inputs, str(out)), nprocs=TP, join=True)
    finally:
        os.chdir(cwd)
        if base is None:
            os.environ.pop("VLLM_RPC_BASE_PATH")
        else:
            os.environ["VLLM_RPC_BASE_PATH"] = base
    return [torch.load(out / f"rank{r}.pt", weights_only=False) for r in range(TP)]


def _yarn(positions: torch.Tensor) -> torch.Tensor:
    """Upstream's DeepSeek YaRN cos | sin for the draft config, evaluated in fp64."""
    rope = _config()["rope_parameters"]
    base, factor = rope["rope_theta"], rope["factor"]
    pos_freqs = base ** (torch.arange(0, ROPE, 2, dtype=torch.float64) / ROPE)
    low, high = yarn_find_correction_range(
        rope["beta_fast"], rope["beta_slow"], ROPE, base, rope["original_max_position_embeddings"]
    )
    mask = 1 - yarn_linear_ramp_mask(low, high, ROPE // 2, dtype=torch.float64)
    inv_freq = 1 / (factor * pos_freqs) * (1 - mask) + 1 / pos_freqs * mask
    mscale = (
        yarn_get_mscale(factor, rope["mscale"])
        / yarn_get_mscale(factor, rope["mscale_all_dim"])
        * rope["attn_factor"]
    )
    angles = positions.double()[:, None] * inv_freq
    return torch.cat([angles.cos(), angles.sin()], -1) * mscale


def _rotate(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Upstream's interleaved rotary of ``x`` ``[T, ..., 64]`` at each row's position."""
    cos, sin = (t.repeat_interleave(2, -1) for t in _yarn(positions).chunk(2, -1))
    shape = (len(positions),) + (1,) * (x.dim() - 2) + (ROPE,)
    return x * cos.view(shape) + rotate_gptj(x) * sin.view(shape)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """RMSNorm at the draft config's eps."""
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + _config()["rms_norm_eps"]) * weight


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """``x`` rounded to bf16, back in fp64."""
    return x.to(torch.bfloat16).double()


def _attention(w, prefix: str, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Upstream's MLA of one draft layer, absorbed, in fp64 but where the attention kernel rounds.

    The kernel holds the query lift, the cached rows, the window rows and its output in bf16.
    """
    attn = f"{prefix}.self_attn"
    h = _rms_norm(x, w(f"{prefix}.input_layernorm.weight"))
    q = _rms_norm(h @ w(f"{attn}.q_a_proj.weight").t(), w(f"{attn}.q_a_layernorm.weight"))
    q = (q @ w(f"{attn}.q_b_proj.weight").t()).view(len(x), HEADS, -1)
    kv_b = w(f"{attn}.kv_b_proj.weight").view(HEADS, -1, LATENT)
    lift = _bf16(torch.cat([torch.einsum("thn,hnl->thl", q[..., :NOPE], kv_b[:, :NOPE]),
                            _rotate(q[..., NOPE:], BLOCK_POSITIONS)], -1))

    def latent_rows(rows, positions):
        kv = rows @ w(f"{attn}.kv_a_proj_with_mqa.weight").t()
        normed = _rms_norm(kv[:, :LATENT], w(f"{attn}.kv_a_layernorm.weight"))
        return _bf16(torch.cat([normed, _rotate(kv[:, LATENT:], positions)], -1))

    cached = latent_rows(context, CONTEXT_POSITIONS).view(SEQS, CONTEXT, -1)
    window = latent_rows(h, BLOCK_POSITIONS).view(SEQS, ROWS, -1)
    rope = _config()["rope_parameters"]
    mscale = 0.1 * rope["mscale_all_dim"] * math.log(rope["factor"]) + 1.0
    scale = (NOPE + ROPE) ** -0.5 * mscale * mscale
    causal = torch.ones(ROWS, CONTEXT + ROWS, dtype=torch.bool).tril(CONTEXT)
    out = []
    for s in range(SEQS):
        keys = torch.cat([cached[s], window[s]])
        scores = torch.einsum("ihl,nl->hin", lift[s * ROWS : (s + 1) * ROWS], keys) * scale
        probs = scores.masked_fill(~causal, float("-inf")).softmax(-1)
        latent = torch.einsum("hin,nl->ihl", probs, keys[:, :LATENT])
        out.append(_bf16(torch.einsum("ihl,hvl->ihv", latent, kv_b[:, NOPE:])).reshape(ROWS, -1))
    return torch.cat(out) @ w(f"{attn}.o_proj.weight").t()


def _mlp(w, prefix: str, x: torch.Tensor) -> torch.Tensor:
    """Upstream's dense SiLU MLP of one draft layer."""
    h = _rms_norm(x, w(f"{prefix}.post_attention_layernorm.weight"))
    gate, up = (h @ w(f"{prefix}.mlp.{name}.weight").t() for name in ("gate_proj", "up_proj"))
    return (F.silu(gate) * up) @ w(f"{prefix}.mlp.down_proj.weight").t()


@pytest.fixture(scope="module")
def reference(inputs) -> list[torch.Tensor]:
    """Upstream's draft forward from the whole stored weights, after each layer: ``[B, 4, H]``."""
    with safe_open(WEIGHTS, framework="pt") as f:
        def w(name):
            return f.get_tensor(name).double()

        context = _rms_norm(inputs["taps"].double() @ w("context_proj.weight").t(),
                            w("context_norm.weight"))
        x, layers = inputs["embedded"].double(), []
        for i in range(_config()["num_hidden_layers"]):
            x = x + _attention(w, f"layers.{i}", x, context)
            x = x + _mlp(w, f"layers.{i}", x)
            layers.append(x.view(SEQS, ROWS, HIDDEN))
    return layers


def _bf16_unit(magnitude: float) -> float:
    """One bf16 unit in the last place at ``magnitude``."""
    return 2.0 ** (math.frexp(magnitude)[1] - 8)


def test_block_rows_stay_within_1e_3_of_the_reference_after_each_layer(ranks, reference):
    units, relative = {}, {}
    for rank, state in enumerate(ranks):
        for layer, (got, ref) in enumerate(zip(state["layers"], reference), 1):
            assert tuple(got.shape) == (SEQS, ROWS, HIDDEN), (rank, layer, tuple(got.shape))
            error = (got.double() - ref).abs().amax((1, 2))
            peak = ref.abs().amax((1, 2))
            units[(rank, layer)] = [float(e) / _bf16_unit(float(p)) for e, p in zip(error, peak)]
            relative[(rank, layer)] = [float(e / p) for e, p in zip(error, peak)]
    warnings.warn(f"worst element in bf16 units by rank and layer, printed, not graded: {units}")
    warnings.warn(f"relative error by rank and layer: {relative}")
    read = [(r, layer) for r in range(TP) for layer in (1, 2, 3)]
    assert sorted(relative) == read, sorted(relative)
    # NaN <= 1e-3 is false, so a non-finite row fails here.
    assert all(e <= 1e-3 for row in relative.values() for e in row), relative


def test_changing_block_row_j_changes_no_earlier_row(ranks):
    readings = {}
    for rank, state in enumerate(ranks):
        clean = state["clean"]
        for j, changed in state["changed"].items():
            earlier = changed[:, :j].view(torch.int32) != clean[:, :j].view(torch.int32)
            reached = bool((changed[:, j] != clean[:, j]).any(-1).all())
            readings[(rank, j)] = (int(earlier.sum()), reached)
    warnings.warn(f"changed bits before row j, and row j moved, by rank and j: {readings}")
    assert sorted(j for _, j in readings) == sorted(list(range(1, ROWS)) * TP)
    assert all(reached for _, reached in readings.values()), f"not adjudicable: {readings}"
    assert [bits for bits, _ in readings.values()] == [0] * len(readings), readings
