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
