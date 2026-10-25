# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter: the context projection and the context KV write, on two TP ranks.

Each rank loads its TP 2 slices of the draft checkpoint, upcasts the drafter to fp32 and
writes 20 verify rows into fp32 latent caches twice: with every row on every rank, and
with the rows split over a decode-context group of 2 by the runner's slot rule. Set
``KIMI_K3_DSPARK_CHECKPOINT`` to the draft directory (default
``~/models/Inferact/Kimi-K3-DSpark-Block5``). Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
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

TP = 2
BLOCK = 256
LATENT, ROPE = 512, 64
MAX_POSITIONS = 32768
# Rotary rows past the last row's position, as the served table holds.
TABLE_ROWS = 2 * MAX_POSITIONS
# Four sequences of 5 verify rows: at the start, across the first block edge, across
# position 8,192 and at the end of the original 32,768-token context.
STARTS = (0, 252, 8188, 32763)
WINDOW = 5
POSITIONS = torch.tensor([start + i for start in STARTS for i in range(WINDOW)])
TABLE_POSITIONS = torch.tensor([0, 1, 8191, 32767])


def _config() -> dict:
    """The draft checkpoint's config with the fp32 dtype."""
    with open(DRAFT / "config.json") as f:
        config = json.load(f)
    config.update(dtype="float32", torch_dtype="float32")
    return config


def _slots(cp_size: int, cp_rank: int) -> tuple[torch.Tensor, int]:
    """Each row's slot on ``cp_rank`` of a ``cp_size`` group by the runner's rule, and blocks."""
    from vllm_neuron.vllm.worker.neuron_model_runner import _compute_slot_mapping_cpu

    virtual = BLOCK * cp_size
    table = np.zeros((len(STARTS), MAX_POSITIONS // virtual), dtype=np.int64)
    seqs = np.repeat(np.arange(len(STARTS)), WINDOW)
    # One block per sequence and virtual block the rows reach; block 0 is the null block.
    reached = sorted({(int(s), int(p) // virtual) for s, p in zip(seqs, POSITIONS)})
    for block, (seq, index) in enumerate(reached, start=1):
        table[seq, index] = block
    slots = np.zeros(len(POSITIONS), dtype=np.int64)
    _compute_slot_mapping_cpu(
        table, slots, POSITIONS.numpy(), seqs, BLOCK, cp_size, cp_rank, BLOCK
    )
    return torch.from_numpy(slots), len(reached) + 1


def _rank(rank: int, port: int, taps: torch.Tensor, out: str) -> None:
    """One TP rank: the fp32 drafter writes the rows at DCP 1 and at DCP 2."""
    import torch.distributed as dist

    from vllm_neuron.model.kimi_k3.dspark import K3DSparkForCausalLM
    from vllm_neuron.parallel.neuron_parallel_state import initialize_neuron_parallel_state

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank)
    with torch.device("meta"):
        draft = K3DSparkForCausalLM.from_configs(
            config=_config(), start_layer_idx=93, max_positions=TABLE_ROWS
        )
    draft.load_weights(str(DRAFT), torch.device("cpu"))
    draft.float()
    state = {"table": draft.rope_table[TABLE_POSITIONS].clone()}
    for cp_size in (1, 2):
        slots, blocks = _slots(cp_size, rank % cp_size)
        caches = {
            layer.cache_name: [torch.full((blocks, 1, BLOCK, LATENT + ROPE), float("nan"))]
            for layer in draft.layers
        }
        draft.bind_kv_cache(caches)
        draft.precompute_and_store_context_kv(taps, POSITIONS, slots)
        state[cp_size] = {
            "slots": slots,
            "caches": torch.stack([c[0].view(-1, LATENT + ROPE) for c in caches.values()]),
        }
    torch.save(state, os.path.join(out, f"rank{rank}.pt"))


@pytest.fixture(scope="module")
def taps() -> torch.Tensor:
    """Random taps for the 20 rows: three 7,168-wide target states each."""
    return torch.randn(len(POSITIONS), 3 * 7168, generator=torch.Generator().manual_seed(0))


@pytest.fixture(scope="module")
def ranks(tmp_path_factory, taps) -> list[dict]:
    """Each rank's rotary rows and caches, from one two-rank run."""
    out = tmp_path_factory.mktemp("ranks")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cwd, base = os.getcwd(), os.environ.get("VLLM_RPC_BASE_PATH")
    # vLLM's broadcaster binds an IPC socket under this path, which must stay short.
    os.environ["VLLM_RPC_BASE_PATH"] = "."
    os.chdir(out)
    try:
        mp.spawn(_rank, args=(port, taps, str(out)), nprocs=TP, join=True)
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


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """RMSNorm at the draft config's eps."""
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + _config()["rms_norm_eps"]) * weight


@pytest.fixture(scope="module")
def reference(taps) -> torch.Tensor:
    """Upstream's context KV rows in fp64 from the whole stored weights: ``[rows, 3, 576]``."""
    cos, sin = (t.repeat_interleave(2, -1) for t in _yarn(POSITIONS).chunk(2, -1))
    with safe_open(WEIGHTS, framework="pt") as f:
        def w(name):
            return f.get_tensor(name).double()

        hidden = _rms_norm(taps.double() @ w("context_proj.weight").t(), w("context_norm.weight"))
        layers = []
        for i in range(3):
            attn = f"layers.{i}.self_attn"
            kv = hidden @ w(f"{attn}.kv_a_proj_with_mqa.weight").t()
            latent, k_pe = kv.split([LATENT, ROPE], -1)
            latent = _rms_norm(latent, w(f"{attn}.kv_a_layernorm.weight"))
            layers.append(torch.cat([latent, k_pe * cos + rotate_gptj(k_pe) * sin], -1))
    return torch.stack(layers, 1)


def _stored(state: dict) -> torch.Tensor:
    """The rows at their slots in each layer's cache: ``[rows, 3, 576]``."""
    return state["caches"][:, state["slots"]].transpose(0, 1)


def test_stored_rows_of_all_layers_equal_the_fp64_reference(ranks, reference):
    for rank, state in enumerate(ranks):
        slots = state[1]["slots"]
        assert len(set(slots.tolist())) == len(POSITIONS) and int(slots.min()) >= BLOCK
        stored = _stored(state[1]).double()
        for part in (slice(0, LATENT), slice(LATENT, None)):
            error = (stored[..., part] - reference[..., part]).abs().amax((0, 2))
            relative = error / reference[..., part].abs().amax((0, 2))
            assert bool((relative <= 1e-5).all()), (rank, part, relative.tolist())


def test_rotary_table_equals_upstream_yarn_at_four_positions(ranks):
    expected = _yarn(TABLE_POSITIONS)
    for rank, state in enumerate(ranks):
        table = state["table"]
        assert (table.dtype, tuple(table.shape)) == (torch.float32, (4, ROPE)), rank
        # NaN > 0 is false, so a non-finite entry fails here rather than count as inside.
        differ = (~torch.isfinite(table) != ~torch.isfinite(expected)).any(-1)
        positions = TABLE_POSITIONS[differ].tolist()
        assert positions == [], (
            rank, f"positions {positions}: non-finite where the reference is not, or the reverse"
        )
        finite = torch.isfinite(table) & torch.isfinite(expected)
        excess = ((table.double() - expected).abs() - 2**-20 * expected.abs())[finite]
        assert int((excess > 0).sum()) == 0, (rank, excess.amax().item())


def test_dcp2_stores_each_row_on_exactly_its_owner_rank(ranks, reference):
    scale = reference.abs().amax(-1)
    held = torch.zeros(TP, len(POSITIONS), dtype=torch.bool)
    for rank, state in enumerate(ranks):
        caches = state[2]["caches"][:, BLOCK:].double()  # the null block's slots are dropped
        written = (~caches.isnan().all(-1)).any(0).nonzero().flatten()
        for slot in written.tolist():
            # The reference row this slot holds in all 3 layers; NaN in any layer matches none.
            error = (caches[:, slot][None] - reference).abs().amax(-1) / scale
            matches = (error <= 1e-5).all(-1).nonzero().flatten().tolist()
            assert len(matches) == 1, (rank, slot, matches)
            held[rank, matches[0]] = True
    owners = (POSITIONS // BLOCK) % TP
    on_both = int(held.all(0).sum())
    on_neither = int((~held.any(0)).sum())
    off_owner = int((~held[owners, torch.arange(len(POSITIONS))]).sum())
    assert (on_both, on_neither, off_owner) == (0, 0, 0)
