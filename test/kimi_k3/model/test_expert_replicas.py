# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 expert replica slots: the loaded copies, the replica map and the decode MoE block.

With ``r`` slots, slot ``j`` of rank ``k``'s experts has a second home on rank
``(k + 1 + j * (R // (r + 1))) mod R``, ``R`` the expert ranks, after that rank's own experts.
The first item maps layer 1 of the checkpoint on each of the served form's 64 expert ranks
(TP 32 by DP 2, 14 experts per rank) at 1 slot and loads each replica and its home copy
through the bank loader. The other two run 4 gloo ranks, TP4 with the experts over its 4
ranks, at hidden 1,024, latent 512 and 64 random MXFP4 experts of width 3,072, 16 per rank.
Each rank builds the MoE layer at 0 and at 1 slot and runs 32 decode steps of 4 random tokens
through it. The shared-expert projection and the tail kernel run only on a device or the NKI
simulator, so torch forms of their math stand in for both, in both runs and in the reference.
The reference runs the expert call on the dispatched rows on one rank that holds all 64
experts and composes the rest in fp64. The error is the largest difference over the largest
reference magnitude. Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory; run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import contextlib
import copy
import json
import os
import types
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open
from torch import nn

from vllm_neuron.model.kimi_k3 import model as m
from vllm_neuron.model.kimi_k3 import weight_loaders_mxfp4
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.weight_loader import SliceView, get_weight_loader

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

SERVED_TP, SERVED_DP = 32, 2
TP, H, K, E, I = 4, 1024, 512, 64, 3072
LOCAL = E // TP
T, STEPS = 4, 32
MOE = "model.layers.1.block_sparse_moe."
BANK = MOE + "experts."
PARTS = weight_loaders_mxfp4.EXPERT_PARTS
BOUND = 1e-4


def _second_home(rank: int, slot: int, ranks: int, slots: int) -> int:
    """The rank that holds the replica of slot ``slot`` of ``rank``'s experts."""
    return (rank + 1 + slot * (ranks // (slots + 1))) % ranks


def _expected_banks(ranks: int, local: int, slots: int) -> list[list[int]]:
    """Each rank's bank by the placement rule: its own experts, then its replicas by slot."""
    return [list(range(k * local, (k + 1) * local))
            + [((k - _second_home(0, j, ranks, slots)) % ranks) * local + j for j in range(slots)]
            for k in range(ranks)]


def _bank_ids(keys: list[str], entries: int) -> list[int]:
    """The expert id of each bank entry, read from its checkpoint keys."""
    width = len(keys) // entries
    return [int(key.partition(".experts.")[2].split(".")[0]) for key in keys[::width]]


def _config(**text) -> dict:
    """The checkpoint's HF config as a dict, with ``text`` set in its text config."""
    from vllm.transformers_utils.config import get_config

    config = copy.deepcopy(get_config(str(CHECKPOINT), trust_remote_code=False).to_dict())
    config["text_config"]["linear_attn_config"].update(text.pop("linear_attn_config", {}))
    config["text_config"].update(text)
    return config


def _build(config: dict, slots: int) -> m.KimiK3ForConditionalGeneration:
    """The model on meta at ``slots`` replica slots and this process's groups."""
    neuron = NeuronConfig(on_device_sampling_config=None, expert_replica_slots=slots)
    with torch.device("meta"):
        return m.KimiK3ForConditionalGeneration.from_configs(config, neuron)


@contextlib.contextmanager
def _served_rank(rank: int):
    """Make the model see rank ``rank`` of the served form: TP 32 by DP 2, experts over 64."""
    tp = types.SimpleNamespace(world_size=SERVED_TP, rank_in_group=rank % SERVED_TP,
                               device_group=None)
    world = types.SimpleNamespace(world_size=SERVED_TP * SERVED_DP, rank_in_group=rank,
                                  device_group=None)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m, "_tp_group", lambda: tp)
        patch.setattr(m, "get_world_group", lambda: world)
        patch.setattr(m, "get_wide_ep_group", lambda: world)
        yield


def test_layer1_replicas_equal_their_home_copies():
    """At 1 slot each replica in layer 1's banks holds its home copy's bytes."""
    slots, ranks = 1, SERVED_TP * SERVED_DP
    config = _config(num_hidden_layers=2,
                     linear_attn_config={"kda_layers": [1, 2], "full_attn_layers": []})
    local = config["text_config"]["num_experts"] // ranks
    wanted = {}
    for rank in range(ranks):
        with _served_rank(rank):
            model = _build(config, slots)
            mappings, _ = weight_loaders_mxfp4.checkpoint_mappings(
                model, model.text_config, model.local_expert_ids())
        params = dict(model.named_parameters())
        for part in PARTS:
            keys, bank = mappings[BANK + part], params[BANK + part]
            assert bank.shape[0] == local + slots and len(keys) % bank.shape[0] == 0
            width = len(keys) // bank.shape[0]
            for entry in [*range(slots), *range(local, local + slots)]:
                wanted[(rank, entry, part)] = (keys[entry * width:(entry + 1) * width],
                                               get_weight_loader(bank))
    with open(CHECKPOINT / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    mismatched, replicas = 0, 0
    with contextlib.ExitStack() as stack:
        files = {}

        def read(key):
            name = weight_map[key]
            if name not in files:
                files[name] = stack.enter_context(safe_open(CHECKPOINT / name, framework="pt"))
            return files[name].get_tensor(key)

        def load(rank, entry, part):
            keys, loader = wanted[(rank, entry, part)]
            return loader.load([SliceView(read(key)) for key in keys], rank)[0]

        for home in range(ranks):
            for j in range(slots):
                holder = _second_home(home, j, ranks, slots)
                for part in PARTS:
                    replica, original = load(holder, local + j, part), load(home, j, part)
                    assert (replica.shape, replica.dtype) == (original.shape, original.dtype)
                    mismatched += int((replica.contiguous().view(torch.uint8)
                                       != original.contiguous().view(torch.uint8)).sum())
                replicas += 1
    assert (replicas, mismatched) == (ranks * slots, 0)


def _norm_projection(hidden, weight, norm_weight, eps):
    """Torch form of the fused RMSNorm and projection: fp32 norm and product, stored bf16."""
    x = hidden.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm_weight.float()
    return (x @ weight.float()).to(hidden.dtype)


def _tail(combined, shared_gate_up, latent_norm_weight, latent_up_weight, shared_down_weight,
          beta, linear_beta):
    """Torch form of the fused tail's pack ``[up | shared | sum of squares]``: fp64, stored fp32."""
    gate, up = shared_gate_up.double().chunk(2, -1)
    up_rows = (combined.double() * latent_norm_weight.double().t()) @ latent_up_weight.double()
    shared = m.situ_and_mul(gate, up, beta, linear_beta) @ shared_down_weight.double()
    squares = combined.double().square().sum(-1, keepdim=True)
    return torch.cat([up_rows, shared, squares], -1).float()


@contextlib.contextmanager
def _stand_ins():
    """The torch forms in place of the two kernels the CPU venue cannot run."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m, "qkv_proj_fp32acc", _norm_projection)
        patch.setattr(m, "moe_fused_tail", _tail)
        yield patch


def _small_model(slots: int) -> m.KimiK3ForConditionalGeneration:
    """A 4-layer model on meta at the test widths and ``slots`` replica slots."""
    return _build(_config(num_hidden_layers=4, hidden_size=H, num_attention_heads=8,
                          intermediate_size=2112, moe_intermediate_size=I,
                          routed_expert_hidden_size=K, num_experts=E, vocab_size=10240,
                          first_k_dense_replace=1,
                          linear_attn_config={"num_heads": 8, "kda_layers": [1, 2, 3],
                                              "full_attn_layers": [4]}), slots)


def _weights(model) -> dict[str, torch.Tensor]:
    """Random whole weights for the MoE layer's parameters, its experts aside."""
    gen, out = torch.Generator().manual_seed(7), {}
    for name, p in model.named_parameters():
        if not name.startswith(MOE) or name.startswith(BANK):
            continue
        if "norm" in name:
            value = 1 + 0.1 * torch.randn(p.shape, generator=gen)
        else:
            scale = 0.01 if name.endswith("e_score_correction_bias") else 0.02
            value = scale * torch.randn(p.shape, generator=gen)
        out[name] = value.to(p.dtype)
    return out


def _experts() -> list[dict[str, torch.Tensor]]:
    """Each expert's stored MXFP4 tensors: random fp4 codes, E8M0 scales near 2^-6."""
    gen = torch.Generator().manual_seed(13)

    def stored(rows, cols):
        packed = torch.randint(0, 256, (rows, cols // 2), generator=gen, dtype=torch.uint8)
        scale = torch.randint(120, 123, (rows, cols // 32), generator=gen, dtype=torch.uint8)
        return packed, scale

    out = []
    for _ in range(E):
        parts = {}
        for w, (rows, cols) in (("w1", (I, K)), ("w3", (I, K)), ("w2", (K, I))):
            parts[f"{w}.weight_packed"], parts[f"{w}.weight_scale"] = stored(rows, cols)
        out.append(parts)
    return out


def _load(model, weights: dict, experts: list, tp_rank: int) -> dict:
    """Load the MoE layer: this rank's shards and its bank; returns the checkpoint mappings."""
    mappings, _ = weight_loaders_mxfp4.checkpoint_mappings(
        model, model.text_config, model.local_expert_ids())
    for name, p in list(model.named_parameters()):
        if not name.startswith(MOE):
            continue
        if name.startswith(BANK):
            keys = [key.partition(".experts.")[2].split(".", 1) for key in mappings[name]]
            slices = [SliceView(experts[int(e)][src]) for e, src in keys]
        else:
            slices = [SliceView(weights[name])]
        path, _, leaf = name.rpartition(".")
        value = get_weight_loader(p).load(slices, tp_rank)
        setattr(model.get_submodule(path), leaf, nn.Parameter(value, requires_grad=False))
    model.model.layers[1].block_sparse_moe.prepare_after_load()
    return mappings


def _block(pack: torch.Tensor, eps: float) -> torch.Tensor:
    """The fp64 block output of a summed pack: the norm-scaled up part plus the shared part."""
    up, shared, squares = pack.double().split([H, H, 1], -1)
    return up * torch.rsqrt(squares / K + eps) + shared


def _rank(rank: int, root: Path) -> None:
    """One gloo rank: the MoE layer's block output at 0 and at 1 slot for every step."""
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state, stamp_spmd_rank_tensors)

    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv", rank=rank, world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank)
    stamped = stamp_spmd_rank_tensors("cpu")
    weights, experts, steps = (torch.load(root / f"{n}.pt")
                               for n in ("weights", "experts", "steps"))
    tp_rank, width = torch.tensor([rank], dtype=torch.int32), H // TP
    cols = slice(rank * width, (rank + 1) * width)
    out, expert_sum = {"routed": [], "blocks": {}}, m._expert_sum

    def dispatched(x):
        out["routed"].append(expert_sum(x))
        return out["routed"][-1]

    with _stand_ins() as patch:
        patch.setattr(m, "_expert_sum", dispatched)
        for slots in (0, 1):
            model = _small_model(slots)
            mappings = _load(model, weights, experts, rank)
            moe = model.model.layers[1].block_sparse_moe
            out["blocks"][slots] = [
                _block(moe.forward_decode(mix, m._decode_layout(rows[:, cols].contiguous()),
                                          steps["gain"], tp_rank), moe.eps)
                for rows, mix in zip(steps["rows"], steps["mix"])]
    entries = moe.experts.banks()[0].shape[0]
    out["map"] = moe.replica_map
    out["bank"] = {part: _bank_ids(mappings[BANK + part], entries) for part in PARTS}
    torch.save(out, root / f"rank{rank}.pt")
    del stamped
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    """The four ranks' records, and the one-rank model on the same weights and experts."""
    root = tmp_path_factory.mktemp("replicas")
    one = _small_model(0)
    weights, experts, gen = _weights(one), _experts(), torch.Generator().manual_seed(21)
    steps = {"rows": torch.randn(STEPS, T, H, generator=gen),
             "mix": (0.5 * torch.randn(STEPS, T, H, generator=gen)).bfloat16(),
             "gain": (1 + 0.1 * torch.randn(H, generator=gen)).bfloat16()}
    for name, value in (("weights", weights), ("experts", experts), ("steps", steps)):
        torch.save(value, root / f"{name}.pt")
    mp.spawn(_rank, args=(root,), nprocs=TP, join=True)
    _load(one, weights, experts, 0)
    return {"one": one, "weights": weights, "steps": steps,
            "ranks": [torch.load(root / f"rank{r}.pt") for r in range(TP)]}


def test_replicated_experts_have_two_homes_and_one_map(world):
    """At 1 slot each replicated expert sits in exactly 2 ranks' banks, by one map on every rank."""
    slots, ranks = 1, world["ranks"]
    expected = _expected_banks(TP, LOCAL, slots)
    want = {(k, i, e) for k, row in enumerate(expected) for i, e in enumerate(row)}
    differences, homes = 0, {}
    for rank, out in enumerate(ranks):
        differences += len(want ^ {(k, i, e) for k, row in enumerate(out["map"])
                                   for i, e in enumerate(row)})
        differences += sum(ids != expected[rank] for ids in out["bank"].values())
        for e in out["bank"]["gate_up_weight"]:
            homes.setdefault(e, []).append(rank)
    replicated = {k * LOCAL + j for k in range(TP) for j in range(slots)}
    wrong = {e: homes.get(e, []) for e in range(E)
             if len(homes.get(e, [])) != (2 if e in replicated else 1)}
    assert (len(replicated), differences, wrong) == (TP * slots, 0, {})


def _error(out: torch.Tensor, ref: torch.Tensor) -> float:
    """The largest difference over the largest reference magnitude."""
    return ((out.double() - ref.double()).abs().max() / ref.double().abs().max()).item()


def _shard(weight: torch.Tensor, dim: int, tp: int) -> torch.Tensor:
    """Tensor-parallel rank ``tp``'s contiguous part of ``weight`` along ``dim``."""
    return weight.chunk(TP, dim)[tp]


def _reference(world) -> list[torch.Tensor]:
    """Each step's fp64 block output: one rank's expert call on the dispatched rows, the tail."""
    moe, w = world["one"].model.layers[1].block_sparse_moe, world["weights"]
    shared = moe.shared_experts
    weight = {n: w[MOE + n + ".weight"] for n in ("routed_expert_norm", "routed_expert_up_proj",
              "shared_experts.gate_proj", "shared_experts.up_proj", "shared_experts.down_proj")}
    blocks = []
    with _stand_ins():
        for routed, mix in zip(world["ranks"][0]["routed"], world["steps"]["mix"]):
            experts = m.routed_experts_mx(
                routed, moe.gate.e_score_correction_bias, torch.zeros(1, 1, dtype=torch.int32),
                *moe.experts.banks(), beta=shared.beta, linear_beta=shared.linear_beta,
                scaling=moe.scaling, replicas=1, chunks=TP)
            pack = 0
            for tp in range(TP):
                gate_up = torch.cat([_shard(weight[f"shared_experts.{n}_proj"], 0, tp)
                                     for n in ("gate", "up")]).t().contiguous()
                gate_up = m._project(mix, gate_up, world["steps"]["gain"], moe.eps)
                pack = pack + _tail(
                    experts[0, tp], gate_up,
                    _shard(weight["routed_expert_norm"].float(), 0, tp).reshape(-1, 1),
                    _shard(weight["routed_expert_up_proj"], 1, tp).t().contiguous(),
                    _shard(weight["shared_experts.down_proj"], 1, tp).t().contiguous(),
                    shared.beta, shared.linear_beta).double()
            blocks.append(_block(pack, moe.eps))
    return blocks


def test_one_slot_block_output_equals_zero_slots(world):
    """Over 32 routings the 1-slot block output meets the 0-slot run and the fp64 reference."""
    reference, readings = _reference(world), {}
    for rank, out in enumerate(world["ranks"]):
        assert len(out["blocks"][0]) == len(out["blocks"][1]) == len(reference) == STEPS
        for step, ref in enumerate(reference):
            one, zero = out["blocks"][1][step], out["blocks"][0][step]
            readings[(rank, step)] = max(_error(one, ref), _error(zero, ref), _error(one, zero))
    over = {key: f"{value:.2e}" for key, value in readings.items() if not value <= BOUND}
    assert not over, f"over {BOUND}, by rank and step: {over}"
