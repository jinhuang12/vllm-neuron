# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the decode router and decode expert tests.

The 5938748 snapshot under ``test/hardware/baselines/moe_5938748`` is the
"old" side of every comparison here. Realistic router inputs come from the
GLM-5.3-Flash checkpoint when it is on this host and are skipped otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import nki
import nki.simulator
import pytest
import torch

REPO = Path(__file__).resolve().parents[4]
BASELINE = REPO / "test" / "hardware" / "baselines" / "moe_5938748"
CHECKPOINT = Path(
    "/home/ubuntu/glm53f-campaign/lane-serve/models/GLM-5.3-Flash-04c4e9e9"
)

#: GLM-5.3-Flash routing constants (``glm5_next/config.py``).
HIDDEN = 4096
EXPERTS = 288
TOP_K = 8
SCALING = 2.5
EPS = 1e-5
#: One rank's expert shard at EP=16, TP=64: 18 experts, I = 2048 / 4.
LOCAL_EXPERTS = 18
LOCAL_INTERMEDIATE = 512
EP_GROUPS = EXPERTS // LOCAL_EXPERTS
SWIGLU_LIMIT = 10.0


def baseline():
    """The 5938748 pipeline module (router, packed experts, combine)."""
    import importlib.util
    import sys

    if "moe_5938748" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "moe_5938748", BASELINE / "__init__.py",
            submodule_search_locations=[str(BASELINE)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["moe_5938748"] = module
        spec.loader.exec_module(module)
    import importlib

    return importlib.import_module("moe_5938748.pipeline")


class SimulatorCounter:
    """Records every ``nki.simulator.simulate_kernel`` call by kernel name."""

    def __init__(self) -> None:
        self.kernels: list[str] = []
        self._real = None

    def __enter__(self) -> "SimulatorCounter":
        self._real = real = nki.simulator.simulate_kernel

        def counting(kernel, *args, **kwargs):
            func = getattr(kernel, "func", kernel)
            self.kernels.append(getattr(func, "__name__", repr(func)))
            return real(kernel, *args, **kwargs)

        nki.simulator.simulate_kernel = counting
        return self

    def __exit__(self, *exc_info) -> None:
        nki.simulator.simulate_kernel = self._real


def random_router_inputs(tokens: int, seed: int = 7, hidden: int = HIDDEN):
    """Gaussian activations and weights at the decode shapes."""
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(tokens, hidden, generator=gen) * 0.8).to(torch.bfloat16)
    gamma = (1.0 + 0.1 * torch.randn(hidden, generator=gen)).to(torch.bfloat16)
    weights = (torch.randn(hidden, EXPERTS, generator=gen) / hidden ** 0.5).to(
        torch.bfloat16
    )
    bias = (torch.rand(EXPERTS, generator=gen) - 0.5) * 0.1
    return x, gamma, weights, bias.to(torch.float32)


def _checkpoint_tensor(name: str) -> torch.Tensor:
    from safetensors import safe_open

    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())
    shard = CHECKPOINT / index["weight_map"][name]
    with safe_open(str(shard), framework="pt") as handle:
        return handle.get_tensor(name)


def realistic_router_inputs(tokens: int, layer: int = 10, seed: int = 11):
    """The checkpoint's own router weight, correction bias and FFN-norm gain.

    Activations are synthetic but shaped like a decoder residual: Gaussian with a
    few large outlier channels. Skips when the checkpoint is not on this host.
    """
    if not (CHECKPOINT / "model.safetensors.index.json").exists():
        pytest.skip("GLM-5.3-Flash checkpoint not on this host")
    prefix = f"model.language_model.layers.{layer}"
    weight = _checkpoint_tensor(f"{prefix}.mlp.gate.weight")  # [E, H]
    bias = _checkpoint_tensor(f"{prefix}.mlp.gate.e_score_correction_bias")
    gamma = _checkpoint_tensor(f"{prefix}.post_attention_layernorm.weight")
    gen = torch.Generator().manual_seed(seed)
    scale = torch.ones(HIDDEN)
    scale[torch.randperm(HIDDEN, generator=gen)[:24]] = 20.0
    x = (torch.randn(tokens, HIDDEN, generator=gen) * scale).to(torch.bfloat16)
    return (
        x,
        gamma.to(torch.bfloat16),
        weight.to(torch.bfloat16).t().contiguous(),
        bias.to(torch.float32),
    )


def tie_rows(logits: torch.Tensor, bias: torch.Tensor, margin: float) -> torch.Tensor:
    """Rows whose 8th and 9th corrected scores are closer than ``margin``.

    The selection is discrete, so a last-bit difference in a logit may swap the
    8th and 9th expert of such a row. Those rows are the "ties" the exactness
    claim sets aside; every other row must select the identical index set.
    """
    choice = torch.sigmoid(logits.to(torch.float32)) + bias.reshape(1, -1)
    top = torch.topk(choice, TOP_K + 1, dim=-1).values
    return (top[:, TOP_K - 1] - top[:, TOP_K]) < margin


def index_sets_equal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row set equality of two ``[T, K]`` index tensors."""
    return (a.sort(dim=-1).values == b.sort(dim=-1).values).all(dim=-1)


# ---- Routed-expert fixtures ------------------------------------------------ #

#: The EP group the expert tests compute: any of the 16 works; 5 keeps the rank
#: offset (5 * 18 = 90) away from zero so a dropped offset reads the wrong slice.
RANK = 5


def packed_expert_bank(experts: int = LOCAL_EXPERTS, hidden: int = HIDDEN,
                       intermediate: int = LOCAL_INTERMEDIATE, seed: int = 23):
    """A packed fp8 bank with decoder-like magnitudes.

    Stored weights are Gaussian fp8 inside the Trn2 range; block scales put the
    gate/up pre-activations at a standard deviation near 3 for unit activations,
    so the +-10 SwiGLU clamps engage on a few channels, and the down projection
    at unit scale.
    """
    from vllm_neuron.functional.moe.fused_fp8_pack import pack_experts

    gen = torch.Generator().manual_seed(seed)
    nh, ni = hidden // 128, intermediate // 128

    def fp8(*shape):
        return (torch.randn(*shape, generator=gen) * 48).clamp(-240, 240).to(
            torch.float8_e4m3fn)

    gate_up = fp8(experts, hidden, 2 * intermediate)
    down = fp8(experts, intermediate, hidden)
    gate_up_scales = (0.5 + torch.rand(experts, nh, 2, ni, generator=gen)) * (
        3.0 / (48 * hidden ** 0.5))
    down_scales = (0.5 + torch.rand(experts, ni, nh, generator=gen)) * (
        1.0 / (48 * intermediate ** 0.5))
    return pack_experts(gate_up, down, gate_up_scales, down_scales)


def decode_hidden(tokens: int, hidden: int = HIDDEN, seed: int = 29):
    """Post-norm expert inputs: unit Gaussian, bf16."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(tokens, hidden, generator=gen).to(torch.bfloat16)


def routed_affinities(local_hits, *, experts: int = EXPERTS,
                      local_experts: int = LOCAL_EXPERTS, rank: int = RANK,
                      top_k: int = TOP_K, seed: int = 31):
    """Scattered ``[T, E]`` router output with chosen hits on ``rank``'s slice.

    ``local_hits[t]`` lists the local expert ids token ``t`` selects in this
    rank's group; the rest of its ``top_k`` picks land on other groups. Each row
    has ``top_k`` positive weights that sum to ``SCALING``, as ``route_tokens``
    emits with ``norm_topk_prob`` and ``routed_scaling_factor``.
    """
    gen = torch.Generator().manual_seed(seed)
    first = rank * local_experts
    others = [e for e in range(experts) if not first <= e < first + local_experts]
    rows = []
    for hits in local_hits:
        assert len(set(hits)) == len(hits) <= top_k
        picks = [first + h for h in hits]
        rest = torch.randperm(len(others), generator=gen)[: top_k - len(picks)]
        picks += [others[i] for i in rest.tolist()]
        weight = torch.rand(top_k, generator=gen) + 0.1
        row = torch.zeros(experts)
        row[torch.tensor(picks)] = weight / weight.sum() * SCALING
        rows.append(row)
    return torch.stack(rows).to(torch.float32)
