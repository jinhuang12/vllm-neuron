# SPDX-License-Identifier: Apache-2.0
"""Stage 4 dense MLP and MoE component gates for GLM-5.2."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig

from vllm_neuron.model.glm_moe_dsa import block_fp8_moe
from vllm_neuron.model.glm_moe_dsa.block_fp8 import dequantize_block_fp8
from vllm_neuron.model.glm_moe_dsa.block_fp8_moe import (
    selective_block_fp8_moe_nki,
    selective_block_fp8_moe_reference,
)
from vllm_neuron.model.glm_moe_dsa.config import GlmMoeDsaConfig
from vllm_neuron.model.glm_moe_dsa.mlp import GlmMoeDsaSwiGLUMLP
from vllm_neuron.model.glm_moe_dsa.moe import (
    GlmMoeDsaMoE,
    GlmMoeDsaNoAuxRouter,
    GlmMoeDsaRoutedExperts,
)

MODEL_PATH_VALUE = os.environ.get("GLM52_MODEL_PATH")
MODEL_PATH = Path(MODEL_PATH_VALUE or ".")


def _block_fp8_weight(
    rows: int, columns: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, columns, generator=generator) * 0.25
    scales = torch.rand(
        (rows + 127) // 128,
        (columns + 127) // 128,
        generator=generator,
    )
    scales = scales * 0.04 + 0.01
    row_blocks = torch.arange(rows) // 128
    column_blocks = torch.arange(columns) // 128
    expanded = scales[row_blocks[:, None], column_blocks[None, :]]
    quantized = (weight / expanded).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scales


def _selective_fp8_fixture(
    token_count: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    hidden_size = 128
    intermediate_size = 128
    hidden = torch.randn(
        token_count,
        hidden_size,
        generator=torch.Generator().manual_seed(3),
    )
    gate = []
    gate_scales = []
    up = []
    up_scales = []
    down = []
    down_scales = []
    for expert_id in range(4):
        tensor, scale = _block_fp8_weight(
            intermediate_size, hidden_size, seed=101 + expert_id
        )
        gate.append(tensor)
        gate_scales.append(scale)
        tensor, scale = _block_fp8_weight(
            intermediate_size, hidden_size, seed=201 + expert_id
        )
        up.append(tensor)
        up_scales.append(scale)
        tensor, scale = _block_fp8_weight(
            hidden_size, intermediate_size, seed=301 + expert_id
        )
        down.append(tensor)
        down_scales.append(scale)
    return (
        hidden,
        torch.stack(gate),
        torch.stack(gate_scales),
        torch.stack(up),
        torch.stack(up_scales),
        torch.stack(down),
        torch.stack(down_scales),
    )


@torch.no_grad()
def _selective_block_fp8_kernel_order_reference(
    hidden: torch.Tensor,
    local_affinities: torch.Tensor,
    gate: torch.Tensor,
    gate_scales: torch.Tensor,
    up: torch.Tensor,
    up_scales: torch.Tensor,
    down: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    round_gate_up: bool,
) -> torch.Tensor:
    """Mirror post-matmul scales with selectable gate/up BF16 boundaries."""

    num_experts, intermediate_size, hidden_size = gate.shape
    output = torch.zeros_like(hidden)
    for expert_id in range(num_experts):
        token_ids = torch.nonzero(
            local_affinities[:, expert_id] != 0, as_tuple=False
        ).flatten()
        if token_ids.numel() == 0:
            continue
        expert_input = hidden.index_select(0, token_ids)
        gate_accumulated = torch.zeros(
            token_ids.numel(), intermediate_size, dtype=torch.float32
        )
        up_accumulated = torch.zeros_like(gate_accumulated)
        for intermediate_start in range(0, intermediate_size, 128):
            intermediate_slice = slice(intermediate_start, intermediate_start + 128)
            for hidden_start in range(0, hidden_size, 128):
                hidden_slice = slice(hidden_start, hidden_start + 128)
                input_tile = expert_input[:, hidden_slice].float()
                gate_partial = F.linear(
                    input_tile,
                    gate[expert_id, intermediate_slice, hidden_slice].float(),
                )
                up_partial = F.linear(
                    input_tile, up[expert_id, intermediate_slice, hidden_slice].float()
                )
                gate_accumulated[:, intermediate_slice] += (
                    gate_partial
                    * gate_scales[
                        expert_id, intermediate_start // 128, hidden_start // 128
                    ]
                )
                up_accumulated[:, intermediate_slice] += (
                    up_partial
                    * up_scales[
                        expert_id, intermediate_start // 128, hidden_start // 128
                    ]
                )
        gate_for_activation = (
            gate_accumulated.to(hidden.dtype) if round_gate_up else gate_accumulated
        )
        up_for_multiply = (
            up_accumulated.to(hidden.dtype) if round_gate_up else up_accumulated
        )
        intermediate = (F.silu(gate_for_activation) * up_for_multiply).to(hidden.dtype)
        down_accumulated = torch.zeros(
            token_ids.numel(), hidden_size, dtype=torch.float32
        )
        for hidden_start in range(0, hidden_size, 128):
            hidden_slice = slice(hidden_start, hidden_start + 128)
            for intermediate_start in range(0, intermediate_size, 128):
                intermediate_slice = slice(intermediate_start, intermediate_start + 128)
                down_partial = F.linear(
                    intermediate[:, intermediate_slice].float(),
                    down[expert_id, hidden_slice, intermediate_slice].float(),
                )
                down_accumulated[:, hidden_slice] += (
                    down_partial
                    * down_scales[
                        expert_id, hidden_start // 128, intermediate_start // 128
                    ]
                )
        down_bf16 = down_accumulated.to(hidden.dtype)
        scaled = (
            down_bf16.float()
            * local_affinities[token_ids, expert_id : expert_id + 1].float()
        ).to(hidden.dtype)
        output[token_ids] = (output[token_ids].float() + scaled.float()).to(
            hidden.dtype
        )
    return output


@pytest.mark.parametrize("token_count", [1, 8, 16, 32, 128, 512])
def test_selective_block_fp8_reference_covers_static_token_buckets(
    token_count: int,
) -> None:
    hidden, gate, gate_scales, up, up_scales, down, down_scales = (
        _selective_fp8_fixture(token_count)
    )
    affinities = torch.zeros(token_count, 4)
    affinities[::2, 0] = 0.25
    affinities[::3, 1] = 0.5
    affinities[1::4, 2] = 0.75
    affinities[-1, 3] = 1.0

    actual = selective_block_fp8_moe_reference(
        hidden,
        affinities,
        gate,
        gate_scales,
        up,
        up_scales,
        down,
        down_scales,
    )

    expected = torch.zeros_like(hidden)
    for expert_id in range(4):
        expert = GlmMoeDsaRoutedExperts(
            hidden_size=128,
            intermediate_size=128,
            num_experts=4,
            top_k=4,
            expert_parallel_size=1,
            fp8_weights=True,
        ).experts[expert_id]
        with torch.no_grad():
            expert.gate_proj.weight.copy_(gate[expert_id])
            expert.gate_proj.weight_scale_inv.copy_(gate_scales[expert_id])
            expert.up_proj.weight.copy_(up[expert_id])
            expert.up_proj.weight_scale_inv.copy_(up_scales[expert_id])
            expert.down_proj.weight.copy_(down[expert_id])
            expert.down_proj.weight_scale_inv.copy_(down_scales[expert_id])
        expected += expert(hidden) * affinities[:, expert_id : expert_id + 1]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_selective_block_fp8_reference_skips_unselected_poisoned_experts() -> None:
    hidden, gate, gate_scales, up, up_scales, down, down_scales = (
        _selective_fp8_fixture(32)
    )
    affinities = torch.zeros(32, 4)
    affinities[:, 1] = 0.4
    affinities[::2, 2] = 0.6

    expected = selective_block_fp8_moe_reference(
        hidden,
        affinities,
        gate,
        gate_scales,
        up,
        up_scales,
        down,
        down_scales,
    )
    gate[0].fill_(float("nan"))
    gate_scales[0].fill_(float("nan"))
    up[0].fill_(float("nan"))
    up_scales[0].fill_(float("nan"))
    down[0].fill_(float("nan"))
    down_scales[0].fill_(float("nan"))
    gate[3].fill_(float("nan"))
    gate_scales[3].fill_(float("nan"))
    up[3].fill_(float("nan"))
    up_scales[3].fill_(float("nan"))
    down[3].fill_(float("nan"))
    down_scales[3].fill_(float("nan"))
    actual = selective_block_fp8_moe_reference(
        hidden,
        affinities,
        gate,
        gate_scales,
        up,
        up_scales,
        down,
        down_scales,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_selective_block_fp8_reference_zero_local_routes_returns_zero() -> None:
    hidden, gate, gate_scales, up, up_scales, down, down_scales = (
        _selective_fp8_fixture(16)
    )
    affinities = torch.zeros(16, 4)
    output = selective_block_fp8_moe_reference(
        hidden,
        affinities,
        gate,
        gate_scales,
        up,
        up_scales,
        down,
        down_scales,
    )
    torch.testing.assert_close(output, torch.zeros_like(hidden), rtol=0, atol=0)


@pytest.mark.parametrize("active_experts", [(0,), (3,), (0, 3)])
def test_selective_block_fp8_healthy_boot_recovery_routes(
    active_experts: tuple[int, ...],
) -> None:
    hidden, gate, gate_scales, up, up_scales, down, down_scales = (
        _selective_fp8_fixture(16)
    )
    affinities = torch.zeros(16, 4)
    for expert_id in active_experts:
        affinities[:, expert_id] = 0.25 + 0.1 * expert_id

    actual = selective_block_fp8_moe_reference(
        hidden,
        affinities,
        gate,
        gate_scales,
        up,
        up_scales,
        down,
        down_scales,
    )
    expected = torch.zeros_like(hidden)
    for expert_id in active_experts:
        expert_affinity = torch.zeros_like(affinities)
        expert_affinity[:, expert_id] = affinities[:, expert_id]
        expected += selective_block_fp8_moe_reference(
            hidden,
            expert_affinity,
            gate,
            gate_scales,
            up,
            up_scales,
            down,
            down_scales,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_selective_block_fp8_kernel_freezes_lnc2_contract() -> None:
    source = Path(block_fp8_moe.__file__).read_text()
    assert "nl.num_programs(axes=0) == 2" in source
    assert "shard_id = nl.program_id(axis=0)" in source
    assert "output_rows_per_shard = BLOCK_ROWS // 2" in source
    assert "(2, intermediate_size, block_size)" in source
    assert source.count("nisa.core_barrier(") == 1
    assert source.index("nisa.core_barrier(") < source.index("nl.dynamic_range(")
    assert "nisa.tensor_copy(dst=gate_rounded, src=gate_accumulated)" in source
    assert "nisa.tensor_copy(dst=up_rounded, src=up_accumulated)" in source
    assert "nisa.activation(dst=gate_activated, data=gate_rounded" in source
    assert "data2=up_rounded" in source
    assert "_wrapped_selective_block_fp8_moe[2]" in source


def test_selective_block_fp8_production_dispatch_has_no_runtime_stacks() -> None:
    source = Path("vllm_neuron/model/glm_moe_dsa/moe.py").read_text()
    dispatch_source = source.split("    def _block_fp8_nki(", 1)[1].split(
        "    def _decode_nki(", 1
    )[0]

    assert "torch.stack" not in dispatch_source
    assert "_block_fp8_weights" not in source
    assert "expert_0.gate_proj.weight" in dispatch_source
    assert "expert_3.down_proj.weight_scale_inv" in dispatch_source


def test_selective_block_fp8_kernel_has_no_scalar_indirect_expert_weights() -> None:
    source = Path(block_fp8_moe.__file__).read_text()
    kernel_source = source.split("def _selective_block_fp8_moe_nki(", 1)[1].split(
        "_wrapped_selective_block_fp8_moe", 1
    )[0]

    assert "scalar_offset=expert_id" not in kernel_source
    assert "gate_weights.ap(" not in kernel_source
    assert "up_weights.ap(" not in kernel_source
    assert "down_weights.ap(" not in kernel_source
    assert "expert_gate_weight.ap(" in kernel_source
    assert "expert_gate_scale.ap(" in kernel_source
    assert "expert_up_weight.ap(" in kernel_source
    assert "expert_up_scale.ap(" in kernel_source
    assert "expert_down_weight.ap(" in kernel_source
    assert "expert_down_scale.ap(" in kernel_source
    for expert_id in range(4):
        assert f"expert_gate_weight = gate_weight_{expert_id}" in kernel_source
        assert f"expert_gate_scale = gate_scale_{expert_id}" in kernel_source
        assert f"expert_up_weight = up_weight_{expert_id}" in kernel_source
        assert f"expert_up_scale = up_scale_{expert_id}" in kernel_source
        assert f"expert_down_weight = down_weight_{expert_id}" in kernel_source
        assert f"expert_down_scale = down_scale_{expert_id}" in kernel_source


def test_selective_block_fp8_kernel_uses_compile_time_expert_phases() -> None:
    source = Path(block_fp8_moe.__file__).read_text()
    kernel_source = source.split("def _selective_block_fp8_moe_nki(", 1)[1].split(
        "_wrapped_selective_block_fp8_moe", 1
    )[0]

    assert "direct_experts" not in kernel_source
    assert "for expert_phase in range(num_experts):" in kernel_source
    assert "op0=nl.equal" in kernel_source
    assert "expert_block_count_register" in kernel_source
    assert "nl.dynamic_range(0, expert_block_count_register)" in kernel_source
    assert "expert_id_register" not in kernel_source
    assert "if expert_id_register" not in kernel_source


def test_selective_block_fp8_compile_time_phases_preserve_block_order() -> None:
    block_to_expert = torch.tensor([0, 0, 1, 2, 2, 3, 3, 3])
    conditions = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0])
    active_block_ids = torch.nonzero(conditions, as_tuple=False).flatten()

    phase_counts = [
        torch.count_nonzero(
            (block_to_expert == expert_id) & conditions.to(torch.bool)
        ).item()
        for expert_id in range(4)
    ]
    phase_offsets = [0]
    for count in phase_counts:
        phase_offsets.append(phase_offsets[-1] + count)
    phased_block_ids = torch.cat(
        [
            torch.arange(phase_offsets[expert_id], phase_offsets[expert_id + 1])
            for expert_id in range(4)
        ]
    )

    assert phase_counts == [2, 1, 2, 1]
    assert torch.equal(phased_block_ids, active_block_ids)


class _SelectiveBlockFP8CompileProbe(nn.Module):
    """Compile/numeric discriminator using the same standalone NKI kernel."""

    def __init__(
        self,
        token_count: int,
        *,
        hidden_size: int = 6144,
        intermediate_size: int = 2048,
    ) -> None:
        super().__init__()
        num_local_experts = 4
        block_size = 128
        self.block_size = block_size
        local_affinities = torch.zeros(token_count, num_local_experts)
        local_affinities[:, 0] = 0.25
        local_affinities[0, 1] = 0.75
        self.register_buffer(
            "expert_affinities_masked",
            local_affinities.reshape(-1, 1),
        )
        token_ids = torch.arange(token_count, dtype=torch.int32)
        token_position_to_id = torch.full(
            (num_local_experts, block_size), -1, dtype=torch.int32
        )
        token_position_to_id[0, :token_count] = token_ids
        token_position_to_id[1, 0] = 0
        self.register_buffer("token_position_to_id", token_position_to_id.flatten())
        self.register_buffer(
            "block_to_expert",
            torch.arange(num_local_experts, dtype=torch.int32),
        )
        self.register_buffer(
            "conditions", torch.tensor([1, 1, 0, 0], dtype=torch.int32)
        )
        if hidden_size == 128 and intermediate_size == 128:
            _, gate, gate_scales, up, up_scales, down, down_scales = (
                _selective_fp8_fixture(token_count)
            )
        else:
            gate = torch.full(
                (num_local_experts, intermediate_size, hidden_size),
                0.03125,
                dtype=torch.float8_e4m3fn,
            )
            up = gate.clone()
            down = torch.full(
                (num_local_experts, hidden_size, intermediate_size),
                0.03125,
                dtype=torch.float8_e4m3fn,
            )
            gate_scales = torch.full(
                (
                    num_local_experts,
                    intermediate_size // 128,
                    hidden_size // 128,
                ),
                0.01,
            )
            up_scales = gate_scales.clone()
            down_scales = torch.full(
                (
                    num_local_experts,
                    hidden_size // 128,
                    intermediate_size // 128,
                ),
                0.01,
            )
        for tensor in (
            gate,
            gate_scales,
            up,
            up_scales,
            down,
            down_scales,
        ):
            tensor[2:].fill_(float("nan"))
        expert_tensors = {
            "gate_weight": gate,
            "gate_scale": gate_scales,
            "up_weight": up,
            "up_scale": up_scales,
            "down_weight": down,
            "down_scale": down_scales,
        }
        for name, tensor in expert_tensors.items():
            for expert_id in range(num_local_experts):
                self.register_buffer(f"{name}_{expert_id}", tensor[expert_id].clone())

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_position_to_id: torch.Tensor,
        block_to_expert: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        return selective_block_fp8_moe_nki(
            hidden_states,
            self.expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
            self.gate_weight_0,
            self.gate_scale_0,
            self.up_weight_0,
            self.up_scale_0,
            self.down_weight_0,
            self.down_scale_0,
            self.gate_weight_1,
            self.gate_scale_1,
            self.up_weight_1,
            self.up_scale_1,
            self.down_weight_1,
            self.down_scale_1,
            self.gate_weight_2,
            self.gate_scale_2,
            self.up_weight_2,
            self.up_scale_2,
            self.down_weight_2,
            self.down_scale_2,
            self.gate_weight_3,
            self.gate_scale_3,
            self.up_weight_3,
            self.up_scale_3,
            self.down_weight_3,
            self.down_scale_3,
            block_size=self.block_size,
        )


def _stack_expert_buffers(
    module: _SelectiveBlockFP8CompileProbe, name: str
) -> torch.Tensor:
    """Stack direct buffers for CPU references outside compiled forward."""

    return torch.stack(
        [getattr(module, f"{name}_{expert_id}") for expert_id in range(4)]
    )


def test_selective_block_fp8_probe_uses_direct_expert_buffers() -> None:
    source = Path(__file__).read_text()
    class_source = source.split("class _SelectiveBlockFP8CompileProbe", 1)[1].split(
        "def _stack_expert_buffers", 1
    )[0]
    forward_source = class_source.split("    def forward(", 1)[1]
    direct_names = [
        f"{projection}_{kind}_{expert_id}"
        for expert_id in range(4)
        for projection in ("gate", "up", "down")
        for kind in ("weight", "scale")
    ]
    module = _SelectiveBlockFP8CompileProbe(1, hidden_size=128, intermediate_size=128)

    for stacked_name in (
        "gate_weights",
        "gate_scales",
        "up_weights",
        "up_scales",
        "down_weights",
        "down_scales",
    ):
        assert f"self.{stacked_name}[" not in forward_source
        assert stacked_name not in module._buffers
    assert len(direct_names) == 24
    assert all(name in module._buffers for name in direct_names)
    assert all(forward_source.count(f"self.{name}") == 1 for name in direct_names)


def _expert_0_3_numeric_probe(token_count: int) -> _SelectiveBlockFP8CompileProbe:
    """Build the 128-wide probe with valid routes for experts 0 and 3."""

    module = _SelectiveBlockFP8CompileProbe(
        token_count,
        hidden_size=128,
        intermediate_size=128,
    ).eval()
    _, gate, gate_scales, up, up_scales, down, down_scales = _selective_fp8_fixture(
        token_count
    )
    fixture_tensors = {
        "gate_weight": gate,
        "gate_scale": gate_scales,
        "up_weight": up,
        "up_scale": up_scales,
        "down_weight": down,
        "down_scale": down_scales,
    }
    for name, fixture in fixture_tensors.items():
        for expert_id in range(4):
            tensor = getattr(module, f"{name}_{expert_id}")
            if expert_id in (1, 2):
                tensor.fill_(float("nan"))
            else:
                tensor.copy_(fixture[expert_id])

    affinities = module.expert_affinities_masked.reshape(token_count, 4)
    affinities.zero_()
    affinities[:, 0] = 0.25
    affinities[:, 3] = 0.55
    return module


def _expert_0_3_routes(
    token_count: int,
) -> dict[str, tuple[tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor]]:
    def route(
        active_experts: tuple[int, ...],
    ) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
        mapping = torch.full((4, 128), -1, dtype=torch.int32)
        token_ids = torch.arange(token_count, dtype=torch.int32)
        for block_id in range(len(active_experts)):
            mapping[block_id, :token_count] = token_ids
        block_to_expert = torch.tensor(
            (*active_experts, *(0 for _ in range(4 - len(active_experts)))),
            dtype=torch.int32,
        )
        conditions = torch.tensor(
            (1,) * len(active_experts) + (0,) * (4 - len(active_experts)),
            dtype=torch.int32,
        )
        return active_experts, mapping.flatten(), block_to_expert, conditions

    return {
        "expert0": route((0,)),
        "expert3": route((3,)),
        "expert0_expert3": route((0, 3)),
    }


def _expert_0_3_cpu_reference(
    module: _SelectiveBlockFP8CompileProbe,
    hidden: torch.Tensor,
    active_experts: tuple[int, ...],
) -> torch.Tensor:
    affinities = module.expert_affinities_masked.reshape(hidden.shape[0], 4)
    active_affinities = torch.zeros_like(affinities)
    for expert_id in active_experts:
        active_affinities[:, expert_id] = affinities[:, expert_id]
    return _selective_block_fp8_kernel_order_reference(
        hidden,
        active_affinities,
        _stack_expert_buffers(module, "gate_weight"),
        _stack_expert_buffers(module, "gate_scale"),
        _stack_expert_buffers(module, "up_weight"),
        _stack_expert_buffers(module, "up_scale"),
        _stack_expert_buffers(module, "down_weight"),
        _stack_expert_buffers(module, "down_scale"),
        round_gate_up=True,
    ).float()


def test_selective_block_fp8_expert_0_3_numeric_probe_contract() -> None:
    token_count = 8
    module = _expert_0_3_numeric_probe(token_count)
    routes = _expert_0_3_routes(token_count)
    token_ids = torch.arange(token_count, dtype=torch.int32)

    for active_experts, mapping, block_to_expert, conditions in routes.values():
        mapping = mapping.reshape(4, 128)
        active_count = len(active_experts)
        assert conditions.tolist() == [1] * active_count + [0] * (4 - active_count)
        assert block_to_expert[:active_count].tolist() == list(active_experts)
        for block_id in range(active_count):
            assert torch.equal(mapping[block_id, :token_count], token_ids)
        assert mapping[:active_count, token_count:].eq(-1).all()
        assert mapping[active_count:].eq(-1).all()
    for name in (
        "gate_weight",
        "gate_scale",
        "up_weight",
        "up_scale",
        "down_weight",
        "down_scale",
    ):
        assert torch.isfinite(getattr(module, f"{name}_0").float()).all()
        assert torch.isnan(getattr(module, f"{name}_1").float()).all()
        assert torch.isnan(getattr(module, f"{name}_2").float()).all()
        assert torch.isfinite(getattr(module, f"{name}_3").float()).all()

    hidden = torch.randn(
        token_count,
        128,
        generator=torch.Generator().manual_seed(911),
        dtype=torch.bfloat16,
    )
    references = {
        "expert0": _expert_0_3_cpu_reference(module, hidden, (0,)),
        "expert3": _expert_0_3_cpu_reference(module, hidden, (3,)),
        "expert0_expert3": _expert_0_3_cpu_reference(module, hidden, (0, 3)),
    }
    assert all(torch.isfinite(reference).all() for reference in references.values())
    assert all(torch.count_nonzero(reference) > 0 for reference in references.values())
    assert not torch.equal(references["expert0"], references["expert3"])
    assert not torch.equal(references["expert0_expert3"], references["expert0"])
    assert not torch.equal(references["expert0_expert3"], references["expert3"])


@pytest.mark.skipif(
    os.getenv("GLM_STAGE4_SELECTIVE_FP8") != "1",
    reason="explicit selective block-FP8 MoE compile discriminator",
)
def test_neuron_compile_selective_block_fp8_moe_production_shape() -> None:
    token_count = int(os.getenv("GLM_STAGE4_SELECTIVE_FP8_TOKENS", "16"))
    if token_count not in {1, 8, 16, 32, 128}:
        pytest.fail("compile discriminator token count must fit one 128-token block")
    module = _SelectiveBlockFP8CompileProbe(token_count).eval().to("neuron:0")
    hidden = torch.ones(token_count, 6144, dtype=torch.bfloat16, device="neuron:0")
    compiled = torch.compile(
        module,
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={"compiler_workdir": os.environ["GLM_STAGE4_COMPILE_DIR"]},
    )
    output = compiled(
        hidden,
        module.token_position_to_id,
        module.block_to_expert,
        module.conditions,
    ).cpu()
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) > 0


@pytest.mark.skipif(
    os.getenv("GLM_STAGE4_SELECTIVE_FP8_EXPERT3_NUMERIC") != "1",
    reason="explicit expert-0/expert-3 selective block-FP8 numeric discriminator",
)
def test_neuron_selective_block_fp8_moe_expert_0_3_matches_cpu() -> None:
    token_count = 8
    module = _expert_0_3_numeric_probe(token_count)
    hidden = torch.randn(
        token_count,
        128,
        generator=torch.Generator().manual_seed(911),
        dtype=torch.bfloat16,
    )
    routes = _expert_0_3_routes(token_count)
    expected = {
        name: _expert_0_3_cpu_reference(module, hidden, active_experts)
        for name, (active_experts, _, _, _) in routes.items()
    }

    compiled = torch.compile(
        module.to("neuron:0"),
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={"compiler_workdir": os.environ["GLM_STAGE4_EXPERT3_COMPILE_DIR"]},
    )
    hidden_neuron = hidden.to("neuron:0")
    actual = {
        name: compiled(
            hidden_neuron,
            mapping.to("neuron:0"),
            block_to_expert.to("neuron:0"),
            conditions.to("neuron:0"),
        )
        .cpu()
        .float()
        for name, (_, mapping, block_to_expert, conditions) in routes.items()
    }

    numeric_evidence = {}
    for name in routes:
        absolute_error = (actual[name] - expected[name]).abs()
        numeric_evidence[name] = {
            "max_abs": absolute_error.max().item(),
            "mean_abs": absolute_error.mean().item(),
            "cosine": F.cosine_similarity(
                actual[name].flatten(), expected[name].flatten(), dim=0
            ).item(),
        }
    print(f"selective block-FP8 expert-0/expert-3 evidence: {numeric_evidence}")

    for name, metrics in numeric_evidence.items():
        assert torch.isfinite(actual[name]).all(), name
        assert torch.count_nonzero(actual[name]) > 0, name
        assert metrics["max_abs"] <= 0.125, {"route": name, **metrics}
        assert metrics["mean_abs"] <= 0.002, {"route": name, **metrics}
        assert metrics["cosine"] >= 0.99999, {"route": name, **metrics}
    assert not torch.equal(actual["expert0"], actual["expert3"])
    assert not torch.equal(actual["expert0_expert3"], actual["expert0"])
    assert not torch.equal(actual["expert0_expert3"], actual["expert3"])


@pytest.mark.skipif(
    os.getenv("GLM_STAGE4_SELECTIVE_FP8_NUMERIC") != "1",
    reason="explicit selective block-FP8 MoE numeric discriminator",
)
def test_neuron_selective_block_fp8_moe_matches_cpu_with_poisoned_experts() -> None:
    token_count = 8
    module = _SelectiveBlockFP8CompileProbe(
        token_count,
        hidden_size=128,
        intermediate_size=128,
    ).eval()
    cpu_reference_tensors = {
        plural_name: _stack_expert_buffers(module, singular_name).detach().clone()
        for plural_name, singular_name in (
            ("gate_weights", "gate_weight"),
            ("gate_scales", "gate_scale"),
            ("up_weights", "up_weight"),
            ("up_scales", "up_scale"),
            ("down_weights", "down_weight"),
            ("down_scales", "down_scale"),
        )
    }
    assert all(tensor.device.type == "cpu" for tensor in cpu_reference_tensors.values())
    hidden = torch.randn(
        token_count,
        128,
        generator=torch.Generator().manual_seed(911),
        dtype=torch.bfloat16,
    )
    local_affinities = module.expert_affinities_masked.reshape(token_count, 4)
    framework_expected_two = selective_block_fp8_moe_reference(
        hidden,
        local_affinities,
        cpu_reference_tensors["gate_weights"],
        cpu_reference_tensors["gate_scales"],
        cpu_reference_tensors["up_weights"],
        cpu_reference_tensors["up_scales"],
        cpu_reference_tensors["down_weights"],
        cpu_reference_tensors["down_scales"],
    ).float()
    one_expert_affinities = local_affinities.clone()
    one_expert_affinities[:, 1:] = 0
    framework_expected_one = selective_block_fp8_moe_reference(
        hidden,
        one_expert_affinities,
        cpu_reference_tensors["gate_weights"],
        cpu_reference_tensors["gate_scales"],
        cpu_reference_tensors["up_weights"],
        cpu_reference_tensors["up_scales"],
        cpu_reference_tensors["down_weights"],
        cpu_reference_tensors["down_scales"],
    ).float()
    expected_two = _selective_block_fp8_kernel_order_reference(
        hidden,
        local_affinities,
        cpu_reference_tensors["gate_weights"],
        cpu_reference_tensors["gate_scales"],
        cpu_reference_tensors["up_weights"],
        cpu_reference_tensors["up_scales"],
        cpu_reference_tensors["down_weights"],
        cpu_reference_tensors["down_scales"],
        round_gate_up=True,
    ).float()
    expected_one = _selective_block_fp8_kernel_order_reference(
        hidden,
        one_expert_affinities,
        cpu_reference_tensors["gate_weights"],
        cpu_reference_tensors["gate_scales"],
        cpu_reference_tensors["up_weights"],
        cpu_reference_tensors["up_scales"],
        cpu_reference_tensors["down_weights"],
        cpu_reference_tensors["down_scales"],
        round_gate_up=True,
    ).float()
    unrounded_expected_two = _selective_block_fp8_kernel_order_reference(
        hidden,
        local_affinities,
        cpu_reference_tensors["gate_weights"],
        cpu_reference_tensors["gate_scales"],
        cpu_reference_tensors["up_weights"],
        cpu_reference_tensors["up_scales"],
        cpu_reference_tensors["down_weights"],
        cpu_reference_tensors["down_scales"],
        round_gate_up=False,
    ).float()
    unrounded_expected_one = _selective_block_fp8_kernel_order_reference(
        hidden,
        one_expert_affinities,
        cpu_reference_tensors["gate_weights"],
        cpu_reference_tensors["gate_scales"],
        cpu_reference_tensors["up_weights"],
        cpu_reference_tensors["up_scales"],
        cpu_reference_tensors["down_weights"],
        cpu_reference_tensors["down_scales"],
        round_gate_up=False,
    ).float()
    expected_zero = torch.zeros_like(expected_two)
    assert torch.isfinite(framework_expected_two).all()
    assert torch.isfinite(framework_expected_one).all()
    assert torch.isfinite(expected_two).all()
    assert torch.isfinite(expected_one).all()
    assert torch.count_nonzero(expected_two) > 0
    assert torch.count_nonzero(expected_one) > 0
    assert torch.isnan(cpu_reference_tensors["gate_weights"][2:]).all()
    assert torch.isnan(cpu_reference_tensors["gate_scales"][2:]).all()
    assert module.conditions.tolist() == [1, 1, 0, 0]
    mapping = module.token_position_to_id.reshape(4, 128)
    assert mapping[0, token_count:].eq(-1).all()
    assert mapping[1, 1:].eq(-1).all()
    for active_mapping in mapping[:2]:
        valid_token_ids = active_mapping[active_mapping >= 0]
        assert valid_token_ids.unique().numel() == valid_token_ids.numel()
    assert local_affinities[0, :2].ne(0).all()

    active_two = module.conditions.clone()
    active_one = torch.tensor([1, 0, 0, 0], dtype=torch.int32)
    active_zero = torch.zeros(4, dtype=torch.int32)
    compiled = torch.compile(
        module.to("neuron:0"),
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={"compiler_workdir": os.environ["GLM_STAGE4_NUMERIC_COMPILE_DIR"]},
    )
    hidden_neuron = hidden.to("neuron:0")
    active_two_neuron = active_two.to("neuron:0")
    active_one_neuron = active_one.to("neuron:0")
    active_zero_neuron = active_zero.to("neuron:0")
    mapping_neuron = module.token_position_to_id
    block_to_expert_neuron = module.block_to_expert
    actual_two = (
        compiled(
            hidden_neuron, mapping_neuron, block_to_expert_neuron, active_two_neuron
        )
        .cpu()
        .float()
    )
    actual_one = (
        compiled(
            hidden_neuron, mapping_neuron, block_to_expert_neuron, active_one_neuron
        )
        .cpu()
        .float()
    )
    actual_zero = (
        compiled(
            hidden_neuron, mapping_neuron, block_to_expert_neuron, active_zero_neuron
        )
        .cpu()
        .float()
    )

    def numeric_metrics(
        actual: torch.Tensor, expected: torch.Tensor
    ) -> dict[str, float]:
        absolute_error = (actual - expected).abs()
        return {
            "max_abs": absolute_error.max().item(),
            "mean_abs": absolute_error.mean().item(),
            "cosine": F.cosine_similarity(
                actual.flatten(), expected.flatten(), dim=0
            ).item(),
        }

    def lnc2_metrics(
        actual: torch.Tensor, expected: torch.Tensor
    ) -> dict[str, dict[str, float]]:
        return {
            "whole": numeric_metrics(actual, expected),
            "columns_0_63": numeric_metrics(actual[:, :64], expected[:, :64]),
            "columns_64_127": numeric_metrics(actual[:, 64:], expected[:, 64:]),
        }

    def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
        return {
            "max_abs": tensor.abs().max().item(),
            "l2_norm": torch.linalg.vector_norm(tensor).item(),
        }

    def run_to_run_differences(
        actual: torch.Tensor, baseline: torch.Tensor
    ) -> dict[str, dict[str, float | int | bool]]:
        def region_difference(
            current: torch.Tensor, first: torch.Tensor
        ) -> dict[str, float | int | bool]:
            difference = (current - first).abs()
            return {
                "exact_equal": torch.equal(current, first),
                "unequal_elements": torch.count_nonzero(current != first).item(),
                "max_abs": difference.max().item(),
                "mean_abs": difference.mean().item(),
            }

        return {
            "whole": region_difference(actual, baseline),
            "columns_0_63": region_difference(actual[:, :64], baseline[:, :64]),
            "columns_64_127": region_difference(actual[:, 64:], baseline[:, 64:]),
        }

    def mismatch_diagnostics(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        column_offset: int = 0,
    ) -> dict[str, object]:
        actual_bf16 = actual.to(torch.bfloat16)
        expected_bf16 = expected.to(torch.bfloat16)
        error = actual_bf16.float() - expected_bf16.float()
        nonzero_error = error[error != 0]
        error_histogram = []
        if nonzero_error.numel() > 0:
            values, counts = torch.unique(nonzero_error, return_counts=True)
            entries = sorted(zip(counts.tolist(), values.tolist()), reverse=True)[:16]
            error_histogram = [
                {"error_value": value, "count": count} for count, value in entries
            ]

        def bf16_ordered_bits(tensor: torch.Tensor) -> torch.Tensor:
            bits = tensor.contiguous().view(torch.int16).to(torch.int32)
            return torch.where(bits < 0, -32768 - bits, bits)

        ulp_distance = (
            bf16_ordered_bits(actual_bf16) - bf16_ordered_bits(expected_bf16)
        ).abs()
        mismatch_ulp = ulp_distance[error != 0]
        ulp_histogram = []
        if mismatch_ulp.numel() > 0:
            values, counts = torch.unique(mismatch_ulp, return_counts=True)
            entries = sorted(zip(counts.tolist(), values.tolist()), reverse=True)[:16]
            ulp_histogram = [{"ulp": value, "count": count} for count, value in entries]

        flat_error = error.abs().flatten()
        mismatch_indices = torch.nonzero(flat_error != 0, as_tuple=False).flatten()
        top_indices = sorted(
            mismatch_indices.tolist(),
            key=lambda index: (-flat_error[index].item(), index),
        )[:16]
        column_count = actual.shape[-1]
        actual_flat = actual_bf16.flatten()
        expected_flat = expected_bf16.flatten()
        ulp_flat = ulp_distance.flatten()
        top_mismatches = [
            {
                "flat": index,
                "column": index % column_count + column_offset,
                "actual": actual_flat[index].float().item(),
                "expected": expected_flat[index].float().item(),
                "abs": flat_error[index].item(),
                "ulp": ulp_flat[index].item(),
            }
            for index in top_indices
        ]
        return {
            "metrics": numeric_metrics(actual, expected),
            "exact_equal": torch.equal(actual, expected),
            "mismatch_count": nonzero_error.numel(),
            "element_count": error.numel(),
            "max_ulp": ulp_distance.max().item(),
            "error_value_histogram": error_histogram,
            "ulp_histogram": ulp_histogram,
            "top_mismatches": top_mismatches,
        }

    def routed_region_diagnostics(
        actual: torch.Tensor, expected: torch.Tensor
    ) -> dict[str, dict[str, object]]:
        return {
            "whole": mismatch_diagnostics(actual, expected),
            "columns_0_63": mismatch_diagnostics(actual[:, :64], expected[:, :64]),
            "columns_64_127": mismatch_diagnostics(
                actual[:, 64:], expected[:, 64:], column_offset=64
            ),
        }

    if os.getenv("GLM_STAGE4_SELECTIVE_FP8_DIAGNOSTIC") == "1":
        diagnostic_runs = [
            {
                "active_two": actual_two,
                "active_one": actual_one,
                "active_zero": actual_zero,
            }
        ]
        for _ in range(4):
            diagnostic_runs.append(
                {
                    "active_two": compiled(
                        hidden_neuron,
                        mapping_neuron,
                        block_to_expert_neuron,
                        active_two_neuron,
                    )
                    .cpu()
                    .float(),
                    "active_one": compiled(
                        hidden_neuron,
                        mapping_neuron,
                        block_to_expert_neuron,
                        active_one_neuron,
                    )
                    .cpu()
                    .float(),
                    "active_zero": compiled(
                        hidden_neuron,
                        mapping_neuron,
                        block_to_expert_neuron,
                        active_zero_neuron,
                    )
                    .cpu()
                    .float(),
                }
            )
        framework_references = {
            "active_two": framework_expected_two,
            "active_one": framework_expected_one,
            "active_zero": expected_zero,
        }
        kernel_order_references = {
            "active_two": unrounded_expected_two,
            "active_one": unrounded_expected_one,
            "active_zero": expected_zero,
        }
        production_sibling_order_references = {
            "active_two": expected_two,
            "active_one": expected_one,
            "active_zero": expected_zero,
        }
        reproducibility_evidence = {
            "reference_stats": {
                "framework": {
                    name: tensor_stats(reference)
                    for name, reference in framework_references.items()
                },
                "kernel_order": {
                    name: tensor_stats(reference)
                    for name, reference in kernel_order_references.items()
                },
                "production_sibling_order": {
                    name: tensor_stats(reference)
                    for name, reference in production_sibling_order_references.items()
                },
            },
            "oracle_order_deltas": {
                "kernel_order_vs_framework": {
                    name: lnc2_metrics(
                        kernel_order_references[name], framework_references[name]
                    )
                    for name in framework_references
                },
                "production_sibling_order_vs_framework": {
                    name: lnc2_metrics(
                        production_sibling_order_references[name],
                        framework_references[name],
                    )
                    for name in framework_references
                },
                "kernel_order_vs_production_sibling_order": {
                    name: lnc2_metrics(
                        kernel_order_references[name],
                        production_sibling_order_references[name],
                    )
                    for name in framework_references
                },
            },
            "runs": [],
        }
        active_two_localization = {
            "per_token": [
                {
                    "token_id": token_id,
                    "routing": "double" if token_id == 0 else "single",
                    "regions": routed_region_diagnostics(
                        actual_two[token_id : token_id + 1],
                        expected_two[token_id : token_id + 1],
                    ),
                }
                for token_id in range(token_count)
            ],
            "route_classes": {
                "token_0_double_route": routed_region_diagnostics(
                    actual_two[:1], expected_two[:1]
                ),
                "tokens_1_7_single_route": routed_region_diagnostics(
                    actual_two[1:], expected_two[1:]
                ),
                "token_0_expert_0_only_control": routed_region_diagnostics(
                    actual_one[:1], expected_one[:1]
                ),
                "token_0_second_route_increment": routed_region_diagnostics(
                    actual_two[:1] - actual_one[:1],
                    expected_two[:1] - expected_one[:1],
                ),
                "single_route_cross_condition_stability": routed_region_diagnostics(
                    actual_two[1:], actual_one[1:]
                ),
            },
        }
        swapped_module = _SelectiveBlockFP8CompileProbe(
            token_count,
            hidden_size=128,
            intermediate_size=128,
        ).eval()
        swapped_mapping = swapped_module.token_position_to_id.reshape(4, 128)
        original_row_0 = swapped_mapping[0].clone()
        original_row_1 = swapped_mapping[1].clone()
        swapped_mapping[0].copy_(original_row_1)
        swapped_mapping[1].copy_(original_row_0)
        swapped_module.block_to_expert[:2].copy_(torch.tensor([1, 0]))
        assert swapped_module.block_to_expert.tolist() == [1, 0, 2, 3]
        assert swapped_mapping[0, 0].item() == 0
        assert swapped_mapping[0, 1:].eq(-1).all()
        assert torch.equal(swapped_mapping[1, :token_count], torch.arange(token_count))
        assert swapped_mapping[1, token_count:].eq(-1).all()
        for active_mapping in swapped_mapping[:2]:
            valid_token_ids = active_mapping[active_mapping >= 0]
            assert valid_token_ids.unique().numel() == valid_token_ids.numel()
        for name, singular_name in (
            ("gate_weights", "gate_weight"),
            ("gate_scales", "gate_scale"),
            ("up_weights", "up_weight"),
            ("up_scales", "up_scale"),
            ("down_weights", "down_weight"),
            ("down_scales", "down_scale"),
        ):
            tensor = _stack_expert_buffers(swapped_module, singular_name)
            assert torch.equal(tensor[:2], cpu_reference_tensors[name][:2])
            assert torch.isnan(tensor[2:]).all()

        expert_1_affinities = torch.zeros_like(local_affinities)
        expert_1_affinities[:, 1] = local_affinities[:, 1]
        expert_1_only = _selective_block_fp8_kernel_order_reference(
            hidden,
            expert_1_affinities,
            cpu_reference_tensors["gate_weights"],
            cpu_reference_tensors["gate_scales"],
            cpu_reference_tensors["up_weights"],
            cpu_reference_tensors["up_scales"],
            cpu_reference_tensors["down_weights"],
            cpu_reference_tensors["down_scales"],
            round_gate_up=True,
        )
        expert_0_only = expected_one.to(torch.bfloat16)
        # dma_compute performs the add internally in FP32 and casts the result
        # to the BF16 output HBM tensor after each completed expert block.
        reverse_two = (expert_1_only.float() + expert_0_only.float()).to(torch.bfloat16)

        swapped_count_0 = torch.zeros(4, dtype=torch.int32)
        swapped_count_1 = torch.tensor([1, 0, 0, 0], dtype=torch.int32)
        swapped_count_2 = torch.tensor([1, 1, 0, 0], dtype=torch.int32)
        swapped_compiled = torch.compile(
            swapped_module.to("neuron:0"),
            backend="vllm_neuron",
            fullgraph=True,
            dynamic=False,
            options={"compiler_workdir": os.environ["GLM_STAGE4_SWAPPED_COMPILE_DIR"]},
        )
        swapped_conditions = {
            "count_0": swapped_count_0.to("neuron:0"),
            "count_1": swapped_count_1.to("neuron:0"),
            "count_2": swapped_count_2.to("neuron:0"),
        }
        swapped_runs = []
        for _ in range(5):
            swapped_runs.append(
                {
                    name: swapped_compiled(
                        hidden_neuron,
                        swapped_module.token_position_to_id,
                        swapped_module.block_to_expert,
                        condition,
                    ).cpu()
                    for name, condition in swapped_conditions.items()
                }
            )
        reverse_two_device = (
            swapped_runs[0]["count_1"].float() + actual_one.float()
        ).to(torch.bfloat16)
        swapped_evidence = {
            "count_0": routed_region_diagnostics(
                swapped_runs[0]["count_0"].float(), expected_zero
            ),
            "count_1_expert_1_only": routed_region_diagnostics(
                swapped_runs[0]["count_1"].float(), expert_1_only.float()
            ),
            "count_2_reverse_order": routed_region_diagnostics(
                swapped_runs[0]["count_2"].float(), reverse_two.float()
            ),
            "count_2_device_contribution_order": routed_region_diagnostics(
                swapped_runs[0]["count_2"].float(), reverse_two_device.float()
            ),
            "count_2_per_token": [
                {
                    "token_id": token_id,
                    "regions": routed_region_diagnostics(
                        swapped_runs[0]["count_2"][token_id : token_id + 1].float(),
                        reverse_two[token_id : token_id + 1].float(),
                    ),
                }
                for token_id in range(token_count)
            ],
            "count_2_device_per_token": [
                {
                    "token_id": token_id,
                    "regions": routed_region_diagnostics(
                        swapped_runs[0]["count_2"][token_id : token_id + 1].float(),
                        reverse_two_device[token_id : token_id + 1].float(),
                    ),
                }
                for token_id in range(token_count)
            ],
            "repeatability": [
                {
                    name: run_to_run_differences(
                        output.float(), swapped_runs[0][name].float()
                    )
                    for name, output in run.items()
                }
                for run in swapped_runs
            ],
        }
        for run_index, run in enumerate(diagnostic_runs):
            reproducibility_evidence["runs"].append(
                {
                    "run": run_index,
                    "routes": {
                        name: {
                            "framework_reference_error": lnc2_metrics(
                                output, framework_references[name]
                            ),
                            "kernel_order_reference_error": lnc2_metrics(
                                output, kernel_order_references[name]
                            ),
                            "production_sibling_order_reference_error": lnc2_metrics(
                                output, production_sibling_order_references[name]
                            ),
                            "output_stats": tensor_stats(output),
                            "versus_run_0": run_to_run_differences(
                                output, diagnostic_runs[0][name]
                            ),
                        }
                        for name, output in run.items()
                    },
                }
            )
        print(
            "selective block-FP8 five-run reproducibility evidence: "
            f"{reproducibility_evidence}"
        )
        print(
            "selective block-FP8 active-two localization evidence: "
            f"{active_two_localization}"
        )
        print("selective block-FP8 swapped-prefix evidence: " f"{swapped_evidence}")
        assert torch.equal(actual_two[1:], expected_two[1:]), {
            "tokens_1_7": active_two_localization["route_classes"][
                "tokens_1_7_single_route"
            ]
        }
        for run_index in range(1, len(swapped_runs)):
            for count in ("count_0", "count_1", "count_2"):
                assert torch.equal(
                    swapped_runs[run_index][count], swapped_runs[0][count]
                ), {
                    "run": run_index,
                    "count": count,
                    "versus_run_0": swapped_evidence["repeatability"][run_index][count],
                }
        assert torch.equal(swapped_runs[0]["count_0"], expected_zero.to(torch.bfloat16))
        for region, evidence in swapped_evidence["count_1_expert_1_only"].items():
            metrics = evidence["metrics"]
            context = {"region": region, **evidence}
            assert metrics["max_abs"] <= 0.125, context
            assert metrics["mean_abs"] <= 0.0015, context
            assert metrics["cosine"] >= 0.99999, context
            assert evidence["max_ulp"] <= 8, context
        assert torch.equal(swapped_runs[0]["count_2"][1:], reverse_two_device[1:]), {
            "count_2_tokens_1_7": swapped_evidence["count_2_device_per_token"][1:]
        }
        assert torch.equal(swapped_runs[0]["count_2"], reverse_two_device), {
            "count_2_device_contribution_order": swapped_evidence[
                "count_2_device_contribution_order"
            ]
        }
        for run_index in range(1, len(diagnostic_runs)):
            for route in ("active_two", "active_one", "active_zero"):
                differences = reproducibility_evidence["runs"][run_index]["routes"][
                    route
                ]["versus_run_0"]
                assert torch.equal(
                    diagnostic_runs[run_index][route], diagnostic_runs[0][route]
                ), {
                    "run": run_index,
                    "route": route,
                    "versus_run_0": differences,
                }

    two_metrics = lnc2_metrics(actual_two, expected_two)
    one_metrics = lnc2_metrics(actual_one, expected_one)
    unrounded_two_metrics = lnc2_metrics(actual_two, unrounded_expected_two)
    unrounded_one_metrics = lnc2_metrics(actual_one, unrounded_expected_one)
    numeric_evidence = {
        "active_two": two_metrics,
        "active_one": one_metrics,
    }
    old_kernel_negative_control = {
        "active_two": unrounded_two_metrics,
        "active_one": unrounded_one_metrics,
    }
    print(f"selective block-FP8 LNC2 numeric evidence: {numeric_evidence}")
    print(
        "selective block-FP8 old-kernel negative control: "
        f"{old_kernel_negative_control}"
    )
    assert torch.isfinite(actual_two).all()
    assert torch.isfinite(actual_one).all()
    assert torch.isfinite(actual_zero).all()

    # The acceptance oracle mirrors post-matmul FP32 block scaling and the
    # production sibling's BF16 gate/up boundaries. CPU matmul and Trainium
    # Tensor Engine rounding can still differ near zero, so bound the absolute
    # distribution and require the output direction to match.
    for active_prefix, region_metrics in numeric_evidence.items():
        for region, metrics in region_metrics.items():
            context = {"active_prefix": active_prefix, "region": region, **metrics}
            assert metrics["max_abs"] <= 0.125, context
            assert metrics["mean_abs"] <= 0.002, context
            assert metrics["cosine"] >= 0.99999, context
            unrounded_metrics = old_kernel_negative_control[active_prefix][region]
            assert metrics["mean_abs"] < unrounded_metrics["mean_abs"], {
                "active_prefix": active_prefix,
                "region": region,
                "production_sibling_order": metrics,
                "old_unrounded_order": unrounded_metrics,
            }
    assert torch.equal(actual_one, expected_one)
    assert torch.equal(actual_zero, expected_zero)
    assert not torch.allclose(actual_two, actual_one)


@pytest.fixture(scope="module")
def config() -> GlmMoeDsaConfig:
    if MODEL_PATH_VALUE is None:
        pytest.skip("GLM52_MODEL_PATH is required for pinned-model tests")
    hf_config = AutoConfig.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False
    )
    return GlmMoeDsaConfig.from_configs(hf_config)


def test_config_freezes_glm52_moe_contract(config: GlmMoeDsaConfig) -> None:
    assert config.hidden_act == "silu"
    assert config.n_routed_experts == 256
    assert config.n_shared_experts == 1
    assert config.num_experts_per_tok == 8
    assert config.norm_topk_prob is True
    assert config.routed_scaling_factor == 2.5
    assert config.n_group == 1
    assert config.topk_group == 1
    assert config.topk_method == "noaux_tc"
    assert config.scoring_func == "sigmoid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_shared_experts", 2),
        ("norm_topk_prob", False),
        ("routed_scaling_factor", 1.0),
        ("topk_method", "greedy"),
        ("scoring_func", "softmax"),
        ("hidden_act", "gelu"),
    ],
)
def test_config_rejects_wrong_moe_contract(
    config: GlmMoeDsaConfig, field: str, value: object
) -> None:
    raw = AutoConfig.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False
    ).to_dict()
    raw[field] = value
    with pytest.raises(ValueError, match=field):
        GlmMoeDsaConfig.from_configs(raw)


def test_dense_swiglu_matches_reference() -> None:
    torch.manual_seed(11)
    module = GlmMoeDsaSwiGLUMLP(
        hidden_size=16,
        intermediate_size=24,
        tensor_parallel_size=2,
        dtype=torch.float32,
    )
    hidden = torch.randn(7, 16)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(
            torch.nn.functional.linear(hidden, module.gate_proj.weight)
        )
        * torch.nn.functional.linear(hidden, module.up_proj.weight),
        module.down_proj.weight,
    )
    torch.testing.assert_close(module(hidden), expected, rtol=0, atol=0)


def test_production_tp64_and_ep64_parameter_shapes(
    config: GlmMoeDsaConfig,
) -> None:
    dense = GlmMoeDsaSwiGLUMLP.dense_from_config(
        config, tensor_parallel_size=64, device="meta"
    )
    assert dense.gate_proj.weight.shape == (192, 6144)
    assert dense.up_proj.weight.shape == (192, 6144)
    assert dense.down_proj.weight.shape == (6144, 192)

    moe = GlmMoeDsaMoE(
        config,
        tensor_parallel_size=64,
        expert_parallel_size=64,
        expert_parallel_rank=63,
        device="meta",
    )
    assert moe.router.gate.weight.shape == (256, 6144)
    assert moe.router.correction_bias.shape == (256,)
    assert moe.router.selection_identity.dtype is torch.float32
    assert moe.experts.num_local_experts == 4
    assert moe.experts.global_expert_ids == (252, 253, 254, 255)
    assert moe.experts.experts[0].gate_proj.weight.shape == (2048, 6144)
    assert moe.experts.experts[0].down_proj.weight.shape == (6144, 2048)
    assert moe.shared_experts.gate_proj.weight.shape == (32, 6144)
    assert moe.shared_experts.down_proj.weight.shape == (6144, 32)


def test_noaux_router_bias_changes_selection_not_weight() -> None:
    router = GlmMoeDsaNoAuxRouter(
        hidden_size=4,
        num_experts=4,
        top_k=2,
        routed_scaling_factor=2.5,
    )
    with torch.no_grad():
        router.gate.weight.zero_()
        router.correction_bias.copy_(torch.tensor([-0.2, 0.1, 0.4, 0.8]))
    affinities, selected = router(torch.ones(3, 4))

    assert {2, 3} == set(selected[0].tolist())
    torch.testing.assert_close(
        affinities[:, 2:], torch.full((3, 2), 1.25), rtol=0, atol=0
    )
    torch.testing.assert_close(affinities[:, :2], torch.zeros(3, 2), rtol=0, atol=0)
    torch.testing.assert_close(
        affinities.sum(dim=-1), torch.full((3,), 2.5), rtol=0, atol=0
    )


def test_local_routed_experts_match_explicit_reference() -> None:
    torch.manual_seed(23)
    experts = GlmMoeDsaRoutedExperts(
        hidden_size=8,
        intermediate_size=12,
        num_experts=8,
        top_k=3,
        expert_parallel_size=2,
        expert_parallel_rank=1,
        dtype=torch.float32,
    )
    hidden = torch.randn(5, 8)
    affinities = torch.zeros(5, 8)
    affinities[:, 4] = 0.3
    affinities[:, 6] = 0.7
    selected = torch.tensor([[4, 6, 0]] * 5, dtype=torch.int32)

    expected = (
        experts.experts[0](hidden) * affinities[:, 4:5]
        + experts.experts[2](hidden) * affinities[:, 6:7]
    )
    actual = experts(hidden, affinities, selected, is_decode=True)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def _production_block_fp8_routed_experts() -> GlmMoeDsaRoutedExperts:
    return GlmMoeDsaRoutedExperts(
        hidden_size=6144,
        intermediate_size=2048,
        num_experts=256,
        top_k=8,
        expert_parallel_size=64,
        expert_parallel_rank=0,
        fp8_weights=True,
        device="meta",
    )


def test_selective_block_fp8_production_contract_accepts_checkpoint_shapes() -> None:
    experts = _production_block_fp8_routed_experts()
    hidden = torch.empty(512, 6144, dtype=torch.bfloat16, device="meta")
    affinities = torch.empty(512, 256, dtype=torch.float32, device="meta")

    assert experts._block_fp8_contract_violations(hidden, affinities) == []


def test_selective_block_fp8_neuron_dispatch_reuses_blockwise_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = _production_block_fp8_routed_experts()
    hidden = torch.empty(16, 6144, dtype=torch.bfloat16, device="meta")
    affinities = torch.empty(16, 256, dtype=torch.float32, device="meta")
    selected = torch.empty(16, 8, dtype=torch.int32, device="meta")
    sentinel = torch.empty_like(hidden)
    calls = []

    monkeypatch.setattr(
        "vllm_neuron.model.glm_moe_dsa.moe.can_run_kernel", lambda _: True
    )
    monkeypatch.setenv("GLM_ENABLE_EXPERIMENTAL_SELECTIVE_FP8_MOE", "1")
    monkeypatch.setattr(
        experts,
        "_block_fp8_contract_violations",
        lambda hidden_states, expert_affinities: [],
    )

    def record_dispatch(hidden_states, expert_affinities):
        calls.append((hidden_states, expert_affinities))
        return sentinel

    monkeypatch.setattr(experts, "_block_fp8_nki", record_dispatch)
    actual = experts(hidden, affinities, selected, is_decode=True)

    assert actual is sentinel
    assert len(calls) == 1
    assert calls[0][0] is hidden
    assert calls[0][1] is affinities


def test_selective_block_fp8_production_dispatch_builds_local_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = _production_block_fp8_routed_experts()
    hidden = torch.empty(16, 6144, dtype=torch.bfloat16, device="meta")
    affinities = torch.empty(16, 256, dtype=torch.float32, device="meta")
    mapping_calls = []
    kernel_calls = []
    mapping = (
        torch.empty(128 * 4, 1, device="meta"),
        torch.empty(4 * 128, dtype=torch.int32, device="meta"),
        torch.empty(4, dtype=torch.int32, device="meta"),
        torch.empty(4, dtype=torch.int32, device="meta"),
    )

    def record_mapping(**kwargs):
        mapping_calls.append(kwargs)
        return mapping

    def record_kernel(*args, **kwargs):
        kernel_calls.append((args, kwargs))
        return torch.empty(128, 6144, dtype=torch.bfloat16, device="meta")

    monkeypatch.setattr(
        "vllm_neuron.model.glm_moe_dsa.moe.NF.build_blockwise_mapping",
        record_mapping,
    )
    monkeypatch.setattr(
        "vllm_neuron.model.glm_moe_dsa.moe.selective_block_fp8_moe_nki",
        record_kernel,
    )
    output = experts._block_fp8_nki(hidden, affinities)

    assert output.shape == hidden.shape
    assert len(mapping_calls) == 1
    mapping_call = mapping_calls[0]
    assert mapping_call["expert_affinities"].shape == (128, 4)
    assert mapping_call["num_local_experts"] == 4
    assert mapping_call["num_experts_per_token"] == 4
    assert mapping_call["block_size"] == 128
    assert mapping_call["tp_degree"] == 1
    assert mapping_call["moe_group"].world_size == 1
    assert len(kernel_calls) == 1
    kernel_args, kernel_kwargs = kernel_calls[0]
    assert kernel_args[0].shape == (128, 6144)
    assert all(
        actual is expected for actual, expected in zip(kernel_args[1:5], mapping)
    )
    expected_expert_parameters = []
    for expert in experts.experts:
        expected_expert_parameters.extend(
            [
                expert.gate_proj.weight,
                expert.gate_proj.weight_scale_inv,
                expert.up_proj.weight,
                expert.up_proj.weight_scale_inv,
                expert.down_proj.weight,
                expert.down_proj.weight_scale_inv,
            ]
        )
    assert len(kernel_args[5:]) == 24
    assert all(
        actual is expected
        for actual, expected in zip(kernel_args[5:], expected_expert_parameters)
    )
    assert [tensor.shape for tensor in kernel_args[5:11]] == [
        (2048, 6144),
        (16, 48),
        (2048, 6144),
        (16, 48),
        (6144, 2048),
        (48, 16),
    ]
    assert kernel_kwargs == {"block_size": 128}


def test_selective_block_fp8_cpu_uses_reference_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = GlmMoeDsaRoutedExperts(
        hidden_size=128,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        expert_parallel_size=1,
        fp8_weights=True,
    )
    hidden = torch.randn(8, 128, dtype=torch.bfloat16)
    affinities = torch.zeros(8, 4)
    selected = torch.zeros(8, 2, dtype=torch.int32)
    sentinel = torch.empty_like(hidden)

    monkeypatch.setattr(
        "vllm_neuron.model.glm_moe_dsa.moe.can_run_kernel", lambda _: False
    )
    monkeypatch.setattr(experts, "_torch_forward", lambda *_: sentinel)
    monkeypatch.setattr(
        experts,
        "_block_fp8_nki",
        lambda *_: pytest.fail("CPU fallback called the selective NKI kernel"),
    )

    assert experts(hidden, affinities, selected, is_decode=False) is sentinel


def test_selective_block_fp8_neuron_defaults_to_reference_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = _production_block_fp8_routed_experts()
    hidden = torch.empty(16, 6144, dtype=torch.bfloat16, device="meta")
    affinities = torch.empty(16, 256, dtype=torch.float32, device="meta")
    selected = torch.empty(16, 8, dtype=torch.int32, device="meta")
    sentinel = torch.empty_like(hidden)

    monkeypatch.delenv("GLM_ENABLE_EXPERIMENTAL_SELECTIVE_FP8_MOE", raising=False)
    monkeypatch.setattr(
        "vllm_neuron.model.glm_moe_dsa.moe.can_run_kernel", lambda _: True
    )
    monkeypatch.setattr(experts, "_torch_forward", lambda *_: sentinel)
    monkeypatch.setattr(
        experts,
        "_block_fp8_nki",
        lambda *_: pytest.fail("default dispatch called the experimental kernel"),
    )

    assert experts(hidden, affinities, selected, is_decode=False) is sentinel


def test_selective_block_fp8_unsupported_neuron_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = GlmMoeDsaRoutedExperts(
        hidden_size=128,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        expert_parallel_size=1,
        fp8_weights=True,
        device="meta",
    )
    hidden = torch.empty(8, 128, dtype=torch.bfloat16, device="meta")
    affinities = torch.empty(8, 4, dtype=torch.float32, device="meta")
    selected = torch.empty(8, 2, dtype=torch.int32, device="meta")

    monkeypatch.setattr(
        "vllm_neuron.model.glm_moe_dsa.moe.can_run_kernel", lambda _: True
    )
    monkeypatch.setenv("GLM_ENABLE_EXPERIMENTAL_SELECTIVE_FP8_MOE", "1")
    monkeypatch.setattr(
        experts,
        "_block_fp8_nki",
        lambda *_: pytest.fail("invalid production contract reached NKI"),
    )

    with pytest.raises(
        RuntimeError, match="Selective GLM-5.2 block-FP8 MoE contract violation"
    ):
        experts(hidden, affinities, selected, is_decode=False)


def test_shared_expert_is_separate_tp_contribution(config: GlmMoeDsaConfig) -> None:
    shared = GlmMoeDsaSwiGLUMLP.shared_from_config(
        config, tensor_parallel_size=64, device="meta"
    )
    assert shared.intermediate_size == 2048
    assert shared.local_intermediate_size == 32


class _HardwareMoE(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, *, is_decode: bool) -> None:
        super().__init__()
        self.is_decode = is_decode
        self.moe = GlmMoeDsaMoE(
            config,
            tensor_parallel_size=64,
            expert_parallel_size=64,
            expert_parallel_rank=0,
        )
        with torch.no_grad():
            self.moe.router.gate.weight.zero_()
            correction = torch.linspace(
                1.0, 0.0, config.n_routed_experts, dtype=torch.float32
            )
            self.moe.router.correction_bias.copy_(correction)
            for expert_id, expert in enumerate(self.moe.experts.experts):
                for projection_id, projection in enumerate(
                    (expert.gate_proj, expert.up_proj, expert.down_proj)
                ):
                    raw_value = 0.03125 * (1 + expert_id + projection_id)
                    projection.weight.fill_(raw_value)
                    scale_rows, scale_columns = projection.weight_scale_inv.shape
                    row = torch.arange(scale_rows, dtype=torch.float32).unsqueeze(1)
                    column = torch.arange(scale_columns, dtype=torch.float32).unsqueeze(
                        0
                    )
                    scales = (
                        0.004
                        + 0.0005 * expert_id
                        + 0.00025 * projection_id
                        + 0.00001 * row
                        + 0.000001 * column
                    )
                    projection.weight_scale_inv.copy_(scales)
            for parameter_id, (name, parameter) in enumerate(
                self.moe.shared_experts.named_parameters()
            ):
                value = 0.001 * (parameter_id + 1)
                if parameter.dtype is torch.float8_e4m3fn:
                    parameter.fill_(value)
                elif name.endswith("weight_scale_inv"):
                    parameter.fill_(0.004 + value)
                elif parameter.is_floating_point():
                    parameter.fill_(value)
                else:
                    raise TypeError(
                        f"unsupported shared-expert test dtype {parameter.dtype}"
                    )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.moe(hidden_states, is_decode=self.is_decode)


class _HardwareMoEComponents(nn.Module):
    """Expose the production NKI component boundaries for numeric comparison."""

    def __init__(self, module: _HardwareMoE) -> None:
        super().__init__()
        self.module = module

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        moe = self.module.moe
        affinities, selected_experts = moe.router(hidden_states)
        routed = moe.experts(
            hidden_states,
            affinities,
            selected_experts,
            is_decode=self.module.is_decode,
        )
        shared = moe.shared_experts(hidden_states)
        return selected_experts, routed, shared, routed + shared


@torch.no_grad()
def _production_shape_reference(
    module: _HardwareMoE, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent torch reference using the exact initialized module tensors."""
    moe = module.moe
    scores = torch.sigmoid(F.linear(hidden_states, moe.router.gate.weight).float())
    selection_scores = scores + moe.router.correction_bias
    selected_experts = torch.topk(
        selection_scores,
        moe.router.top_k,
        dim=-1,
        sorted=False,
    ).indices.to(torch.int32)
    selected_weights = torch.gather(scores, -1, selected_experts.to(torch.long))
    selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)
    selected_weights = selected_weights * moe.router.routed_scaling_factor
    affinities = torch.zeros_like(scores).scatter(
        -1, selected_experts, selected_weights
    )

    routed = torch.zeros_like(hidden_states)
    for local_id, global_id in enumerate(moe.experts.global_expert_ids):
        expert = moe.experts.experts[local_id]
        gate_weight = dequantize_block_fp8(
            expert.gate_proj.weight,
            expert.gate_proj.weight_scale_inv,
        ).to(hidden_states.dtype)
        up_weight = dequantize_block_fp8(
            expert.up_proj.weight,
            expert.up_proj.weight_scale_inv,
        ).to(hidden_states.dtype)
        down_weight = dequantize_block_fp8(
            expert.down_proj.weight,
            expert.down_proj.weight_scale_inv,
        ).to(hidden_states.dtype)
        expert_output = F.linear(
            F.silu(F.linear(hidden_states, gate_weight))
            * F.linear(hidden_states, up_weight),
            down_weight,
        )
        routed = routed + expert_output * affinities[:, global_id : global_id + 1].to(
            hidden_states.dtype
        )

    shared_mlp = moe.shared_experts
    shared_gate_weight = dequantize_block_fp8(
        shared_mlp.gate_proj.weight,
        shared_mlp.gate_proj.weight_scale_inv,
    ).to(hidden_states.dtype)
    shared_up_weight = dequantize_block_fp8(
        shared_mlp.up_proj.weight,
        shared_mlp.up_proj.weight_scale_inv,
    ).to(hidden_states.dtype)
    shared_down_weight = dequantize_block_fp8(
        shared_mlp.down_proj.weight,
        shared_mlp.down_proj.weight_scale_inv,
    ).to(hidden_states.dtype)
    shared = F.linear(
        F.silu(F.linear(hidden_states, shared_gate_weight))
        * F.linear(hidden_states, shared_up_weight),
        shared_down_weight,
    )
    return selected_experts, routed, shared, routed + shared


def _assert_bf16_component_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Check BF16 kernel output against the independent reference."""
    actual_fp32 = actual.to(torch.float32)
    expected_fp32 = expected.to(torch.float32)
    # Eight percent allows BF16 rounding and different accelerator accumulation
    # order. The 3e-5 floor covers values near zero while remaining over 10x
    # smaller than the smallest component maximum in this deterministic gate.
    torch.testing.assert_close(
        actual_fp32,
        expected_fp32,
        rtol=0.08,
        atol=3e-5,
    )


@pytest.mark.skipif(
    os.getenv("GLM_STAGE4_HARDWARE") != "1",
    reason="explicit scoped Neuron Stage 4 compile smoke",
)
@pytest.mark.parametrize(
    ("token_count", "is_decode"),
    [(16, False), (512, False), (1, True), (32, True)],
)
def test_neuron_compile_and_activate_stage4_moe(
    config: GlmMoeDsaConfig,
    token_count: int,
    is_decode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GLM_ENABLE_EXPERIMENTAL_SELECTIVE_FP8_MOE", "1")
    torch.manual_seed(29 + token_count)
    hidden = torch.randn(token_count, config.hidden_size, dtype=torch.bfloat16)
    device = torch.device("neuron:0")
    module = _HardwareMoE(config, is_decode=is_decode).eval().to(device)
    compile_root = Path(os.environ["GLM_STAGE4_COMPILE_DIR"])
    variant = f"{'decode' if is_decode else 'prefill'}-{token_count}"
    compiled = torch.compile(
        module,
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={"compiler_workdir": str(compile_root / variant)},
    )
    actual = compiled(hidden.to(device)).cpu()
    assert actual.shape == hidden.shape
    assert torch.isfinite(actual).all()
    assert actual.abs().max().item() > 0


@pytest.mark.skipif(
    os.getenv("GLM_STAGE4_HARDWARE") != "1",
    reason="explicit scoped Neuron Stage 4 numeric comparison",
)
@pytest.mark.parametrize(
    ("token_count", "is_decode"),
    [(1, True), (16, False)],
    ids=["decode1", "prefill16"],
)
def test_neuron_stage4_moe_matches_production_shape_reference(
    config: GlmMoeDsaConfig,
    token_count: int,
    is_decode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GLM_ENABLE_EXPERIMENTAL_SELECTIVE_FP8_MOE", "1")
    torch.manual_seed(29 + token_count)
    hidden = torch.randn(token_count, config.hidden_size, dtype=torch.bfloat16)
    module = _HardwareMoE(config, is_decode=is_decode).eval()
    for expert in module.moe.experts.experts:
        for projection in (expert.gate_proj, expert.up_proj, expert.down_proj):
            assert projection.weight.dtype is torch.float8_e4m3fn
            assert torch.isfinite(projection.weight.float()).all()
            assert torch.isfinite(projection.weight_scale_inv).all()
            assert torch.all(projection.weight_scale_inv > 0)
    with torch.no_grad():
        # Exercise the production router matmul and sigmoid. The correction
        # remains large enough to include all four experts local to EP rank zero
        # while the nonzero logits vary the remaining selected global experts.
        torch.manual_seed(1031 + token_count)
        module.moe.router.gate.weight.normal_(mean=0.0, std=0.0005)

    expected = _production_shape_reference(module, hidden)
    expected_selected = torch.sort(expected[0].to(torch.long), dim=-1).values
    for local_expert in module.moe.experts.global_expert_ids:
        assert torch.any(expected_selected == local_expert)
    assert torch.any(expected_selected >= module.moe.experts.num_local_experts)

    device = torch.device("neuron:0")
    compiled_module = _HardwareMoEComponents(module).eval().to(device)
    compile_root = Path(os.environ["GLM_STAGE4_COMPILE_DIR"])
    variant = f"numeric-{'decode' if is_decode else 'prefill'}-{token_count}"
    compiled = torch.compile(
        compiled_module,
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={"compiler_workdir": str(compile_root / variant)},
    )
    actual = tuple(tensor.cpu() for tensor in compiled(hidden.to(device)))

    actual_selected = torch.sort(actual[0].to(torch.long), dim=-1).values
    assert torch.equal(actual_selected, expected_selected)
    assert expected[1].abs().max().item() > 0
    assert actual[1].abs().max().item() > 0
    for actual_component, expected_component in zip(
        actual[1:], expected[1:], strict=True
    ):
        assert actual_component.shape == expected_component.shape
        assert torch.isfinite(actual_component).all()
        _assert_bf16_component_close(actual_component, expected_component)
