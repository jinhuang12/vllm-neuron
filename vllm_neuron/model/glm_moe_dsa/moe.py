# SPDX-License-Identifier: Apache-2.0
"""Direct vLLM-Neuron GLM-5.2 routed and shared expert components."""

from __future__ import annotations

import os

import nki
import nki.language as nl
import torch
import torch.nn.functional as F
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.router_topk.router_topk import router_topk
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)
from torch import nn

import vllm_neuron.functional as NF
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

from .block_fp8 import BlockFP8Linear, RowFP8Linear, dequantize_block_fp8
from .block_fp8_moe import selective_block_fp8_moe_nki
from .config import GlmMoeDsaConfig
from .mlp import GlmMoeDsaSwiGLUMLP

_SELECTIVE_BLOCK_FP8_ENV = "GLM_ENABLE_EXPERIMENTAL_SELECTIVE_FP8_MOE"


class _SingleRankMoEGroup:
    """The pure-EP local mapping has no intermediate-dimension collective."""

    rank_in_group = 0
    world_size = 1


_router_topk_nki = nki.jit(router_topk)


class GlmMoeDsaNoAuxRouter(nn.Module):
    """Pinned sigmoid noaux_tc routing.

    Correction bias affects expert selection only. The selected uncorrected
    sigmoid scores are normalized and then multiplied by the routed scale.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        routed_scaling_factor: float = 2.5,
        norm_topk_prob: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not 0 < top_k <= num_experts:
            raise ValueError(f"top_k={top_k} must be in 1..{num_experts}")
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "selection_identity",
            torch.eye(num_experts, dtype=torch.float32, device=device),
            persistent=False,
        )

    @property
    def correction_bias(self) -> torch.Tensor:
        return self.gate.e_score_correction_bias

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(self.gate(hidden_states).float())
        selection_scores = scores + self.correction_bias
        if can_run_kernel(scores):
            selected_experts = self._select_nki(scores)
        else:
            _, selected_experts = torch.topk(
                selection_scores, self.top_k, dim=-1, sorted=False
            )
            selected_experts = selected_experts.to(torch.int32)
        selected_weights = torch.gather(scores, -1, selected_experts.to(torch.long))
        if self.norm_topk_prob:
            denominator = selected_weights.sum(dim=-1, keepdim=True)
            denominator = denominator + (denominator == 0).to(denominator.dtype)
            selected_weights = selected_weights / denominator
        routed_scale = scores.new_full((), self.routed_scaling_factor)
        selected_weights = selected_weights * routed_scale
        affinities = torch.zeros_like(scores).scatter(
            -1, selected_experts, selected_weights
        )
        return affinities, selected_experts

    def _select_nki(self, scores: torch.Tensor) -> torch.Tensor:
        """Select on sigmoid scores plus correction bias without HLO sort."""
        token_count = scores.shape[0]
        router_logits = torch.zeros_like(scores)
        scratch_affinities = torch.zeros_like(scores)
        expert_index = torch.zeros(
            token_count,
            self.top_k,
            dtype=torch.int32,
            device=scores.device,
        )
        wrapped = wrap_nki(_router_topk_nki)
        _, expert_index, _ = wrapped[2](
            x=scores,
            w=self.selection_identity,
            w_bias=self.correction_bias.unsqueeze(0),
            router_logits=router_logits,
            expert_affinities=scratch_affinities,
            expert_index=expert_index,
            act_fn=RouterActFnType.SIGMOID,
            k=self.top_k,
            x_hbm_layout=1,
            x_sb_layout=0,
            router_pre_norm=False,
            norm_topk_prob=False,
            use_indirect_dma_scatter=True,
            use_column_tiling=False,
            shard_on_tokens=token_count >= 128,
            skip_store_router_logits=True,
            skip_store_expert_index=False,
            use_PE_broadcast_w_bias=False,
        )
        return expert_index


class GlmMoeDsaExpertMLP(nn.Module):
    """One full-width routed expert owned by one EP64 rank."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        fp8_weights: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if fp8_weights:
            self.gate_proj = RowFP8Linear(hidden_size, intermediate_size, device=device)
            self.up_proj = RowFP8Linear(hidden_size, intermediate_size, device=device)
            self.down_proj = RowFP8Linear(intermediate_size, hidden_size, device=device)
        else:
            self.gate_proj = nn.Linear(
                hidden_size,
                intermediate_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.up_proj = nn.Linear(
                hidden_size,
                intermediate_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.down_proj = nn.Linear(
                intermediate_size,
                hidden_size,
                bias=False,
                dtype=dtype,
                device=device,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class GlmMoeDsaRoutedExperts(nn.Module):
    """Four full local experts for one rank of the frozen EP64 topology."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        num_experts: int = 256,
        top_k: int = 8,
        expert_parallel_size: int = 64,
        expert_parallel_rank: int = 0,
        fp8_weights: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        block_size: int = 256,
    ) -> None:
        super().__init__()
        if num_experts % expert_parallel_size:
            raise ValueError(
                f"num_experts={num_experts} is not divisible by "
                f"EP={expert_parallel_size}"
            )
        if not 0 <= expert_parallel_rank < expert_parallel_size:
            raise ValueError(
                f"expert_parallel_rank={expert_parallel_rank} is outside "
                f"EP={expert_parallel_size}"
            )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_parallel_size = expert_parallel_size
        self.expert_parallel_rank = expert_parallel_rank
        self.num_local_experts = num_experts // expert_parallel_size
        self.local_expert_start = expert_parallel_rank * self.num_local_experts
        self.block_size = block_size
        self.register_buffer(
            "expert_parallel_rank_tensor",
            torch.tensor([[expert_parallel_rank]], dtype=torch.int32, device=device),
            persistent=False,
        )
        self.experts = nn.ModuleList(
            GlmMoeDsaExpertMLP(
                hidden_size,
                intermediate_size,
                fp8_weights=fp8_weights,
                dtype=dtype,
                device=device,
            )
            for _ in range(self.num_local_experts)
        )

    @property
    def global_expert_ids(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.local_expert_start,
                self.local_expert_start + self.num_local_experts,
            )
        )

    def _kernel_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._uses_row_fp8:
            gate_up = torch.stack(
                [
                    torch.stack(
                        [expert.gate_proj.weight.T, expert.up_proj.weight.T], dim=1
                    )
                    for expert in self.experts
                ],
                dim=0,
            )
            down = torch.stack(
                [expert.down_proj.weight.T for expert in self.experts], dim=0
            )
            return gate_up, down
        if self._uses_block_fp8:
            gate_up = torch.stack(
                [
                    torch.stack(
                        [
                            dequantize_block_fp8(
                                expert.gate_proj.weight,
                                expert.gate_proj.weight_scale_inv,
                            )
                            .to(torch.bfloat16)
                            .T,
                            dequantize_block_fp8(
                                expert.up_proj.weight,
                                expert.up_proj.weight_scale_inv,
                            )
                            .to(torch.bfloat16)
                            .T,
                        ],
                        dim=1,
                    )
                    for expert in self.experts
                ],
                dim=0,
            )
            down = torch.stack(
                [
                    dequantize_block_fp8(
                        expert.down_proj.weight,
                        expert.down_proj.weight_scale_inv,
                    )
                    .to(torch.bfloat16)
                    .T
                    for expert in self.experts
                ],
                dim=0,
            )
            return gate_up, down
        gate_up = torch.stack(
            [
                torch.stack([expert.gate_proj.weight.T, expert.up_proj.weight.T], dim=1)
                for expert in self.experts
            ],
            dim=0,
        )
        down = torch.stack(
            [expert.down_proj.weight.T for expert in self.experts], dim=0
        )
        return gate_up, down

    def _torch_forward(
        self, hidden_states: torch.Tensor, affinities: torch.Tensor
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for local_id, global_id in enumerate(self.global_expert_ids):
            weight = affinities[:, global_id : global_id + 1].to(hidden_states.dtype)
            output = output + self.experts[local_id](hidden_states) * weight
        return output

    @property
    def _uses_block_fp8(self) -> bool:
        return isinstance(self.experts[0].gate_proj, BlockFP8Linear)

    @property
    def _uses_row_fp8(self) -> bool:
        return isinstance(self.experts[0].gate_proj, RowFP8Linear)

    def _row_fp8_kernel_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Trn2 token-generation MoE row-scale layouts."""

        gate_up = torch.stack(
            [
                torch.stack(
                    [
                        expert.gate_proj.weight_scale_inv,
                        expert.up_proj.weight_scale_inv,
                    ],
                    dim=0,
                )
                for expert in self.experts
            ],
            dim=0,
        )
        down = torch.stack(
            [expert.down_proj.weight_scale_inv for expert in self.experts], dim=0
        )
        return gate_up, down

    def _row_fp8_nki(
        self,
        hidden_states: torch.Tensor,
        affinities: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Run fused row-FP8 MoE in static token chunks supported by Trn2."""

        gate_up, down = self._kernel_weights()
        gate_up_scale, down_scale = self._row_fp8_kernel_scales()
        outputs = []
        for start in range(0, hidden_states.shape[0], 128):
            stop = min(start + 128, hidden_states.shape[0])
            outputs.append(
                NF.moe_tkg(
                    hidden_input=hidden_states[start:stop],
                    expert_gate_up_weights=gate_up,
                    expert_down_weights=down,
                    expert_affinities=affinities[start:stop],
                    expert_index=selected_experts[start:stop],
                    is_all_expert=True,
                    rank_id=self.expert_parallel_rank_tensor,
                    expert_gate_up_weights_scale=gate_up_scale,
                    expert_down_weights_scale=down_scale,
                    mask_unselected_experts=True,
                    expert_affinities_scaling_mode=(ExpertAffinityScaleMode.POST_SCALE),
                    activation_fn=ActFnType.SiLU,
                    output_dtype=hidden_states.dtype,
                )
            )
        return torch.cat(outputs, dim=0)

    def _block_fp8_contract_violations(
        self,
        hidden_states: torch.Tensor,
        affinities: torch.Tensor,
    ) -> list[str]:
        """Return reasons the production selective kernel cannot run."""

        violations = []
        if self.num_experts != 256:
            violations.append(f"num_experts={self.num_experts}, expected 256")
        if self.top_k != 8:
            violations.append(f"top_k={self.top_k}, expected 8")
        if self.expert_parallel_size != 64:
            violations.append(
                f"expert_parallel_size={self.expert_parallel_size}, expected 64"
            )
        if self.num_local_experts != 4:
            violations.append(f"num_local_experts={self.num_local_experts}, expected 4")
        if self.hidden_size != 6144:
            violations.append(f"hidden_size={self.hidden_size}, expected 6144")
        if self.intermediate_size != 2048:
            violations.append(
                f"intermediate_size={self.intermediate_size}, expected 2048"
            )
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            violations.append(
                f"hidden_states.shape={tuple(hidden_states.shape)}, expected [T,6144]"
            )
        elif not 1 <= hidden_states.shape[0] <= 512:
            violations.append(
                f"token_count={hidden_states.shape[0]}, expected a value in 1..512"
            )
        if hidden_states.dtype is not torch.bfloat16:
            violations.append(
                f"hidden_states.dtype={hidden_states.dtype}, expected torch.bfloat16"
            )
        if tuple(affinities.shape) != (hidden_states.shape[0], self.num_experts):
            violations.append(
                f"affinities.shape={tuple(affinities.shape)}, expected "
                f"[{hidden_states.shape[0]},256]"
            )

        for local_id, expert in enumerate(self.experts):
            projections = {
                "gate": expert.gate_proj,
                "up": expert.up_proj,
                "down": expert.down_proj,
            }
            all_block_fp8 = True
            for name, projection in projections.items():
                if not isinstance(projection, BlockFP8Linear):
                    violations.append(f"expert {local_id} {name} is not block FP8")
                    all_block_fp8 = False
                    continue
                if projection.weight.dtype is not torch.float8_e4m3fn:
                    violations.append(
                        f"expert {local_id} {name} weight dtype is "
                        f"{projection.weight.dtype}, expected float8_e4m3fn"
                    )
                if projection.weight_scale_inv.dtype is not torch.float32:
                    violations.append(
                        f"expert {local_id} {name} scale dtype is "
                        f"{projection.weight_scale_inv.dtype}, expected float32"
                    )
                if projection.row_offset != 0 or projection.col_offset != 0:
                    violations.append(
                        f"expert {local_id} {name} uses nonzero block offsets"
                    )
            if not all_block_fp8:
                continue
            expected_gate_shape = (self.intermediate_size, self.hidden_size)
            expected_down_shape = (self.hidden_size, self.intermediate_size)
            expected_gate_scale_shape = (
                self.intermediate_size // 128,
                self.hidden_size // 128,
            )
            expected_down_scale_shape = (
                self.hidden_size // 128,
                self.intermediate_size // 128,
            )
            if tuple(expert.gate_proj.weight.shape) != expected_gate_shape:
                violations.append(f"expert {local_id} gate weight shape mismatch")
            if tuple(expert.up_proj.weight.shape) != expected_gate_shape:
                violations.append(f"expert {local_id} up weight shape mismatch")
            if tuple(expert.down_proj.weight.shape) != expected_down_shape:
                violations.append(f"expert {local_id} down weight shape mismatch")
            if (
                tuple(expert.gate_proj.weight_scale_inv.shape)
                != expected_gate_scale_shape
            ):
                violations.append(f"expert {local_id} gate scale grid mismatch")
            if (
                tuple(expert.up_proj.weight_scale_inv.shape)
                != expected_gate_scale_shape
            ):
                violations.append(f"expert {local_id} up scale grid mismatch")
            if (
                tuple(expert.down_proj.weight_scale_inv.shape)
                != expected_down_scale_shape
            ):
                violations.append(f"expert {local_id} down scale grid mismatch")
        return violations

    def _block_fp8_nki(
        self,
        hidden_states: torch.Tensor,
        affinities: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch only routed local experts using checkpoint FP8 blocks."""

        token_count = hidden_states.shape[0]
        block_size = 128
        padded_count = max(block_size, token_count)
        padded_count = ((padded_count + block_size - 1) // block_size) * block_size
        if padded_count != token_count:
            hidden_states = torch.cat(
                [
                    hidden_states,
                    torch.zeros(
                        padded_count - token_count,
                        self.hidden_size,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ),
                ],
                dim=0,
            )
            affinities = torch.cat(
                [
                    affinities,
                    torch.zeros(
                        padded_count - token_count,
                        self.num_experts,
                        dtype=affinities.dtype,
                        device=affinities.device,
                    ),
                ],
                dim=0,
            )

        local_affinities = affinities[
            :,
            self.local_expert_start : self.local_expert_start + self.num_local_experts,
        ].contiguous()
        (
            affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=local_affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=min(self.top_k, self.num_local_experts),
            block_size=block_size,
            moe_group=_SingleRankMoEGroup(),
            tp_degree=1,
        )
        expert_0, expert_1, expert_2, expert_3 = self.experts
        output = selective_block_fp8_moe_nki(
            hidden_states,
            affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
            expert_0.gate_proj.weight,
            expert_0.gate_proj.weight_scale_inv,
            expert_0.up_proj.weight,
            expert_0.up_proj.weight_scale_inv,
            expert_0.down_proj.weight,
            expert_0.down_proj.weight_scale_inv,
            expert_1.gate_proj.weight,
            expert_1.gate_proj.weight_scale_inv,
            expert_1.up_proj.weight,
            expert_1.up_proj.weight_scale_inv,
            expert_1.down_proj.weight,
            expert_1.down_proj.weight_scale_inv,
            expert_2.gate_proj.weight,
            expert_2.gate_proj.weight_scale_inv,
            expert_2.up_proj.weight,
            expert_2.up_proj.weight_scale_inv,
            expert_2.down_proj.weight,
            expert_2.down_proj.weight_scale_inv,
            expert_3.gate_proj.weight,
            expert_3.gate_proj.weight_scale_inv,
            expert_3.up_proj.weight,
            expert_3.up_proj.weight_scale_inv,
            expert_3.down_proj.weight,
            expert_3.down_proj.weight_scale_inv,
            block_size=block_size,
        )
        return output[:token_count]

    def _decode_nki(
        self,
        hidden_states: torch.Tensor,
        affinities: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        gate_up, down = self._kernel_weights()
        return NF.moe_tkg(
            hidden_input=hidden_states,
            expert_gate_up_weights=gate_up,
            expert_down_weights=down,
            expert_affinities=affinities,
            expert_index=selected_experts,
            is_all_expert=True,
            rank_id=self.expert_parallel_rank_tensor,
            mask_unselected_experts=True,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
            output_dtype=hidden_states.dtype,
        )

    def _prefill_nki(
        self, hidden_states: torch.Tensor, affinities: torch.Tensor
    ) -> torch.Tensor:
        token_count = hidden_states.shape[0]
        padded_count = max(self.block_size, token_count)
        padded_count = (
            (padded_count + self.block_size - 1) // self.block_size
        ) * self.block_size
        if padded_count != token_count:
            hidden_states = torch.cat(
                [
                    hidden_states,
                    torch.zeros(
                        padded_count - token_count,
                        self.hidden_size,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ),
                ],
                dim=0,
            )
            affinities = torch.cat(
                [
                    affinities,
                    torch.zeros(
                        padded_count - token_count,
                        self.num_experts,
                        dtype=affinities.dtype,
                        device=affinities.device,
                    ),
                ],
                dim=0,
            )

        local_affinities = affinities[
            :,
            self.local_expert_start : self.local_expert_start + self.num_local_experts,
        ].contiguous()
        (
            affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=local_affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=min(self.top_k, self.num_local_experts),
            block_size=self.block_size,
            moe_group=_SingleRankMoEGroup(),
            tp_degree=1,
        )
        gate_up, down = self._kernel_weights()
        gate_up_scale = None
        down_scale = None
        if self._uses_row_fp8:
            gate_up_scale, down_scale = self._row_fp8_kernel_scales()
        output = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=hidden_states,
            expert_affinities_masked=affinities_masked,
            gate_up_proj_weight=gate_up,
            down_proj_weight=down,
            gate_up_proj_scale=gate_up_scale,
            down_proj_scale=down_scale,
            activation_function=ActFnType.SiLU,
            block_size=self.block_size,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )
        return output[:token_count]

    def forward(
        self,
        hidden_states: torch.Tensor,
        affinities: torch.Tensor,
        selected_experts: torch.Tensor,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        if self._uses_row_fp8:
            if not can_run_kernel(hidden_states):
                return self._torch_forward(hidden_states, affinities)
            return self._row_fp8_nki(
                hidden_states,
                affinities,
                selected_experts,
            )
        if self._uses_block_fp8:
            if not can_run_kernel(hidden_states):
                return self._torch_forward(hidden_states, affinities)
            if os.getenv(_SELECTIVE_BLOCK_FP8_ENV) == "1":
                violations = self._block_fp8_contract_violations(
                    hidden_states, affinities
                )
                if violations:
                    raise RuntimeError(
                        "Selective GLM-5.2 block-FP8 MoE contract violation: "
                        + "; ".join(violations)
                    )
                return self._block_fp8_nki(hidden_states, affinities)
        if not can_run_kernel(hidden_states):
            return self._torch_forward(hidden_states, affinities)
        if is_decode:
            return self._decode_nki(hidden_states, affinities, selected_experts)
        return self._prefill_nki(hidden_states, affinities)


class GlmMoeDsaMoE(nn.Module):
    """One rank-local GLM-5.2 MoE contribution.

    Routed experts use pure EP64. The shared expert uses TP64. The decoder
    sums this local result across the common 64-rank group.
    """

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        *,
        tensor_parallel_size: int = 64,
        expert_parallel_size: int = 64,
        expert_parallel_rank: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("GLM-5.2 supports only hidden_act='silu'")
        if config.scoring_func != "sigmoid" or config.topk_method != "noaux_tc":
            raise ValueError("GLM-5.2 requires sigmoid noaux_tc routing")
        if config.n_group != 1 or config.topk_group != 1:
            raise ValueError("GLM-5.2 grouped routing is frozen to one group")
        self.router = GlmMoeDsaNoAuxRouter(
            config.hidden_size,
            config.n_routed_experts,
            config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            norm_topk_prob=config.norm_topk_prob,
            dtype=config.torch_dtype,
            device=device,
        )
        self.experts = GlmMoeDsaRoutedExperts(
            config.hidden_size,
            config.moe_intermediate_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            expert_parallel_size=expert_parallel_size,
            expert_parallel_rank=expert_parallel_rank,
            fp8_weights=bool(config.quantization_config),
            dtype=config.torch_dtype,
            device=device,
        )
        self.shared_experts = GlmMoeDsaSwiGLUMLP.shared_from_config(
            config,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=expert_parallel_rank,
            device=device,
        )

    def forward(self, hidden_states: torch.Tensor, *, is_decode: bool) -> torch.Tensor:
        affinities, selected_experts = self.router(hidden_states)
        routed = self.experts(
            hidden_states,
            affinities,
            selected_experts,
            is_decode=is_decode,
        )
        shared = self.shared_experts(hidden_states)
        return routed + shared
