# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Optional, Union, Tuple, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
import nki

from nkilib.core.router_topk.router_topk import router_topk
from nkilib.core.utils.common_types import RouterActFnType

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

from dataclasses import dataclass

import nki.isa as nisa
import nki.language as nl

from nkilib.core.moe_block.moe_block_tkg_utils import _pmax
from nkilib.core.router_topk.router_topk import XSBLayout_tp102__0
from nkilib.core.router_topk.router_topk import router_topk as _substrate_router_topk
from nkilib.core.subkernels.rmsnorm_tkg import _rmsnorm_tkg_dloc
from nkilib.core.utils.common_types import QuantizationType
# The same sharding query nkilib's router uses to choose its token split, so the
# `noaux_tc` stage and the router cannot disagree about which core owns which rows.
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from vllm_neuron.functional.moe.rmsnorm_router_topk_tkg import (
    _can_use_kernel as _substrate_can_use_kernel,
)
from vllm_neuron.functional.moe.rmsnorm_router_topk_tkg import (
    _validate_inputs as _substrate_validate_inputs,
)

router_topk_jit = nki.jit(router_topk)


class RouterComputationOrder(Enum):
    """
    Enum specifying the computation order for MoE router operations.

    This enum determines the sequence of operations applied during routing:

    - ``PRENORM_LINEAR_TOPK_ACT_SCATTER``: RMSNorm (optional) → Linear → TopK → Activation → Scatter
        - Default behavior, applies optional RMSNorm to hidden states first
        - Projects to router logits, selects top-k experts
        - Applies activation (softmax/sigmoid) only to selected top-k values
        - Scatters activated values to full expert affinity matrix

    - ``PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER``: RMSNorm (optional) → Linear → Activation → TopK → Renorm → Scatter
        - Applies optional RMSNorm to hidden states first
        - Projects to router logits
        - Applies activation to ALL expert logits before selection
        - Selects top-k experts from activated values
        - L1 renormalizes selected values so they sum to 1.0
        - Scatters to full expert affinity matrix

    - ``PRENORM_LINEAR_TOPK_SCATTER_ACT``: RMSNorm (optional) → Linear → TopK → Scatter → Activation
        - Applies optional RMSNorm to hidden states first
        - Projects to router logits
        - Selects top-k experts based on raw logits
        - Scatters raw logit values to full matrix (zeros elsewhere)
        - Applies activation to the full sparse matrix

    Usage Examples:
        >>> from vllm_neuron.functional.moe.router import router, RouterComputationOrder
        >>>
        >>> # Default computation order (PRENORM_LINEAR_TOPK_ACT_SCATTER)
        >>> affinities = router(hidden_states, router_weights, top_k=2)
        >>>
        >>> # Activation before TopK selection with optional RMSNorm
        >>> affinities = router(
        ...     hidden_states, router_weights, top_k=2,
        ...     router_computation_order=RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER,
        ...     gamma=gamma,  # Optional RMSNorm
        ... )
        >>>
        >>> # Scatter before activation with optional RMSNorm
        >>> affinities = router(
        ...     hidden_states, router_weights, top_k=2,
        ...     router_computation_order=RouterComputationOrder.PRENORM_LINEAR_TOPK_SCATTER_ACT,
        ...     gamma=gamma  # Optional RMSNorm
        ... )
    """

    PRENORM_LINEAR_TOPK_ACT_SCATTER = "prenorm_linear_topk_act_scatter"
    PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER = "prenorm_linear_act_topk_renorm_scatter"
    PRENORM_LINEAR_TOPK_SCATTER_ACT = "prenorm_linear_topk_scatter_act"


def router(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor] = None,
    activation: Union[str, Callable[[Tensor], Tensor]] = "softmax",
    return_logits: bool = False,
    gamma: Optional[Tensor] = None,
    eps: float = 1e-6,
    computation_dtype: torch.dtype = torch.float32,
    router_computation_order: RouterComputationOrder = RouterComputationOrder.PRENORM_LINEAR_TOPK_ACT_SCATTER,
    shard_on_tokens: Optional[bool] = None,
    transposed_hidden_states: bool = False,
    x_sb_layout: Optional[int] = None,
    use_column_tiling: Optional[bool] = None,
    use_indirect_dma_scatter: Optional[bool] = None,
    use_PE_broadcast_w_bias: Optional[bool] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Router API for Mixture of Experts (MoE) expert selection.

    This function computes routing probabilities for selecting experts in MoE layers.
    It performs top-k expert selection and returns routing affinities. It supports optional RMSNorm
    preprocessing and flexible activation functions for different routing strategies.

    The function uses an optimized NKI kernel when constraints are met.
    Falls back to PyTorch implementation otherwise.

    Args:
        hidden_states: Input hidden states tensor with shape [T, H]
            where T is the number of tokens and H is the hidden dimension.
            Can be [H, T] if transposed_hidden_states is set to True
        router_weights: Router projection weights with shape [H, E]
            where E is the number of experts
        top_k: Number of top experts to select per token (typically 1 or 2)
        router_bias: Optional router projection bias with shape [E]
            Default: None (no bias)
        activation: Activation function to apply to router scores.
            Can be either a string or a callable function:
            - String options: "softmax" (default) for standard MoE routing,
              "sigmoid" for alternative routing strategies
            - Callable: Any function that takes a Tensor and returns a Tensor
              Examples: F.softmax, torch.sigmoid, or custom functions
        return_logits: Whether to return raw router logits in addition to affinities
            Default: False (return only affinities)
        gamma: Optional RMSNorm weights with shape [H] for input preprocessing.
            If provided, applies RMSNorm before router computation for all
            computation orders.
            Default: None (no normalization)
        eps: Epsilon value for RMSNorm numerical stability
            Default: 1e-6
        computation_dtype: Data type for computation (float32, float16, or bfloat16)
            Default: torch.float32
        router_computation_order: Specifies the order of operations in routing computation.
            See RouterComputationOrder enum for details.
            - PRENORM_LINEAR_TOPK_ACT_SCATTER: RMSNorm (optional) → Linear → TopK → Activation → Scatter (default)
            - PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER: RMSNorm (optional) → Linear → Activation → TopK → L1 Renorm → Scatter
            - PRENORM_LINEAR_TOPK_SCATTER_ACT: RMSNorm (optional) → Linear → TopK → Scatter → Activation
            Default: RouterComputationOrder.PRENORM_LINEAR_TOPK_ACT_SCATTER
        shard_on_tokens: [Kernel only arg] Enable LNC sharding across token dimension
        transposed_hidden_states: If True, hidden_states should be [H, T] instead of [T, H]
        x_sb_layout: [Kernel only arg] Layout of input x in SBUF (0, 1, or 2)
        use_column_tiling: [Kernel only arg] Enable PE array column tiling for small T
        use_indirect_dma_scatter: [Kernel only arg] Use indirect DMA for expert affinity scatter
        use_PE_broadcast_w_bias: [Kernel only arg] Use tensor engine for bias broadcast

    Returns:
        If return_logits=False:
            expert_affinities: Tensor with shape [T, E] containing routing probabilities.
                              Non-zero values for selected experts, zeros elsewhere.

        If return_logits=True:
            Tuple containing:
            - expert_affinities: [T, E] routing probabilities as above
            - router_logits: [T, E] raw router logits before top-k selection

    Raises:
        ValueError: If activation is not a valid string ("softmax" or "sigmoid")
                   or a callable function, or if other input parameters are invalid.

    Usage Examples:
        >>> # Basic router usage with softmax activation (default PRENORM_LINEAR_TOPK_ACT_SCATTER order)
        >>> hidden_states = torch.randn(128, 768)  # 128 tokens, 768 hidden dim
        >>> router_weights = torch.randn(768, 8)   # 8 experts
        >>>
        >>> affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     activation="softmax"
        ... )
        >>> print(affinities.shape)  # torch.Size([128, 8])
        >>> print((affinities > 0).sum(dim=1))  # Each token routes to exactly 2 experts

        >>> # Router with RMSNorm preprocessing and bias
        >>> gamma = torch.ones(768)
        >>> router_bias = torch.zeros(8)
        >>>
        >>> affinities, logits = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     router_bias=router_bias,
        ...     gamma=gamma,
        ...     eps=1e-5,
        ...     return_logits=True
        ... )
        >>> print(affinities.shape, logits.shape)  # torch.Size([128, 8]) torch.Size([128, 8])

        >>> # Router with PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER computation order
        >>> # Applies activation to ALL logits before TopK selection, then L1 renormalizes
        >>> affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     activation="softmax",
        ...     router_computation_order=RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER,
        ...     gamma=gamma,  # Optional RMSNorm preprocessing
        ... )
        >>> print(affinities.shape)  # torch.Size([128, 8])

        >>> # Router with PRENORM_LINEAR_TOPK_SCATTER_ACT computation order
        >>> # Scatters to full matrix first, then applies activation
        >>> affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     activation="softmax",
        ...     router_computation_order=RouterComputationOrder.PRENORM_LINEAR_TOPK_SCATTER_ACT,
        ...     gamma=gamma  # Optional RMSNorm preprocessing
        ... )
        >>> print(affinities.shape)  # torch.Size([128, 8])

        >>> # Router with callable activation functions
        >>> import torch.nn.functional as F
        >>>
        >>> # Using F.softmax as callable (equivalent to "softmax" string)
        >>> affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     activation=lambda x: F.softmax(x, dim=-1)
        ... )
        >>> print(affinities.shape)  # torch.Size([128, 8])

        >>> # Using torch.sigmoid as callable (equivalent to "sigmoid" string)
        >>> affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     activation=torch.sigmoid
        ... )
        >>> print(affinities.shape)  # torch.Size([128, 8])

        >>> # Router with custom computation dtype for reduced precision
        >>> affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     computation_dtype=torch.float16
        ... )
        >>> print(affinities.dtype)  # torch.float16
        >>> print(affinities.shape)  # torch.Size([128, 8])
    """
    # Validate inputs
    _validate_router_inputs(
        hidden_states,
        router_weights,
        top_k,
        router_bias,
        gamma,
        computation_dtype,
        router_computation_order,
        transposed_hidden_states,
    )

    # Check if kernel can be used
    can_use_kernel = _can_use_kernel(
        hidden_states,
        router_weights,
        top_k,
        activation,
        gamma,
        router_bias,
        router_computation_order,
        transposed_hidden_states,
    )

    hidden_states = hidden_states.to(computation_dtype)

    if can_use_kernel:
        expert_affinities, router_logits = _nki_router_impl(
            hidden_states=hidden_states,
            router_weights=router_weights,
            top_k=top_k,
            router_bias=router_bias,
            activation=activation,
            computation_dtype=computation_dtype,
            router_computation_order=router_computation_order,
            skip_store_router_logits=not return_logits,
            shard_on_tokens=shard_on_tokens,
            x_hbm_layout=0 if transposed_hidden_states else 1,
            x_sb_layout=x_sb_layout,
            use_column_tiling=use_column_tiling,
            use_indirect_dma_scatter=use_indirect_dma_scatter,
            use_PE_broadcast_w_bias=use_PE_broadcast_w_bias,
        )
    else:
        # PyTorch fallback implementation
        expert_affinities, router_logits = _torch_router_impl(
            hidden_states=hidden_states.T
            if transposed_hidden_states
            else hidden_states,
            router_weights=router_weights,
            top_k=top_k,
            router_bias=router_bias,
            activation=activation,
            gamma=gamma,
            eps=eps,
            computation_dtype=computation_dtype,
            router_computation_order=router_computation_order,
        )

    if return_logits:
        return expert_affinities, router_logits
    else:
        return expert_affinities


def _torch_router_impl(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor],
    activation: Union[str, Callable[[Tensor], Tensor]],
    gamma: Optional[Tensor],
    eps: float,
    computation_dtype: torch.dtype,
    router_computation_order: RouterComputationOrder,
) -> Tuple[Tensor, Tensor]:
    """
    PyTorch implementation of router computation with configurable computation order.

    Dispatches to the appropriate implementation based on router_computation_order:
    - PRENORM_LINEAR_TOPK_ACT_SCATTER: RMSNorm (optional) → Linear → TopK → Activation → Scatter
    - PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER: RMSNorm (optional) → Linear → Activation → TopK → L1 Renorm → Scatter
    - PRENORM_LINEAR_TOPK_SCATTER_ACT: RMSNorm (optional) → Linear → TopK → Scatter → Activation

    Args:
        hidden_states: Input tensor [T, H]
        router_weights: Router projection weights [H, E]
        top_k: Number of experts per token
        router_bias: Optional router bias [E]
        activation: Activation function ("softmax", "sigmoid", or callable)
        gamma: Optional RMSNorm weights [H]
        eps: RMSNorm epsilon
        computation_dtype: Data type for computation
        router_computation_order: Specifies the order of operations

    Returns:
        Tuple containing:
        - expert_affinities: [T, E] routing probabilities with zeros for non-selected experts
        - router_logits: [T, E] raw router logits
    """
    if (
        router_computation_order
        == RouterComputationOrder.PRENORM_LINEAR_TOPK_ACT_SCATTER
    ):
        return _torch_router_impl_prenorm_linear_topk_act_scatter(
            hidden_states=hidden_states,
            router_weights=router_weights,
            top_k=top_k,
            router_bias=router_bias,
            activation=activation,
            gamma=gamma,
            eps=eps,
            computation_dtype=computation_dtype,
        )
    elif (
        router_computation_order
        == RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER
    ):
        return _torch_router_impl_prenorm_linear_act_topk_renorm_scatter(
            hidden_states=hidden_states,
            router_weights=router_weights,
            top_k=top_k,
            router_bias=router_bias,
            activation=activation,
            gamma=gamma,
            eps=eps,
            computation_dtype=computation_dtype,
        )
    elif (
        router_computation_order
        == RouterComputationOrder.PRENORM_LINEAR_TOPK_SCATTER_ACT
    ):
        return _torch_router_impl_prenorm_linear_topk_scatter_act(
            hidden_states=hidden_states,
            router_weights=router_weights,
            top_k=top_k,
            router_bias=router_bias,
            activation=activation,
            gamma=gamma,
            eps=eps,
            computation_dtype=computation_dtype,
        )
    else:
        raise ValueError(
            f"Unknown router_computation_order: {router_computation_order}"
        )


def _torch_router_impl_prenorm_linear_topk_act_scatter(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor],
    activation: Union[str, Callable[[Tensor], Tensor]],
    gamma: Optional[Tensor],
    eps: float,
    computation_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """
    PyTorch implementation: RMSNorm (optional) → Linear → TopK → Activation → Scatter

    This is the default computation order. Applies optional RMSNorm to hidden states,
    projects to router logits, selects top-k experts, applies activation only to
    the selected top-k values, then scatters to the full affinity matrix.

    Args:
        hidden_states: Input tensor [T, H]
        router_weights: Router projection weights [H, E]
        top_k: Number of experts per token
        router_bias: Optional router bias [E]
        activation: Activation function ("softmax", "sigmoid", or callable)
        gamma: Optional RMSNorm weights [H]
        eps: RMSNorm epsilon
        computation_dtype: Data type for computation

    Returns:
        Tuple of (expert_affinities [T, E], router_logits [T, E])
    """
    T, H = hidden_states.shape
    E = router_weights.shape[1]
    device = hidden_states.device

    # Step 1: Optional RMSNorm preprocessing
    if gamma is not None:
        hidden_states = _torch_rms_norm(hidden_states, gamma, eps)

    # Step 2: Router linear projection
    router_logits = F.linear(
        hidden_states.to(computation_dtype),
        router_weights.T.to(computation_dtype),
        router_bias.to(computation_dtype) if router_bias is not None else None,
    )  # [T, E]

    # Step 3: Top-k expert selection
    router_top_values, router_indices = torch.topk(
        router_logits, top_k, dim=-1
    )  # [T, top_k]

    # Step 4: Apply activation function to top-k values only
    router_top_probs = _apply_activation(activation, router_top_values)  # [T, top_k]

    # Step 5: Scatter to full expert affinity matrix
    expert_affinities = torch.zeros(T, E, device=device, dtype=computation_dtype)
    expert_affinities.scatter_(1, router_indices, router_top_probs)  # [T, E]

    return expert_affinities, router_logits


def _torch_router_impl_prenorm_linear_act_topk_renorm_scatter(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor],
    activation: Union[str, Callable[[Tensor], Tensor]],
    gamma: Optional[Tensor],
    eps: float,
    computation_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """
    PyTorch implementation: RMSNorm (optional) → Linear → Activation → TopK → L1 Renorm → Scatter

    Applies optional RMSNorm to hidden states, projects to router logits, applies activation
    to ALL expert logits (not just top-k), selects top-k from activated values, L1-normalizes
    the selected values so they sum to 1.0, then scatters.

    Args:
        hidden_states: Input tensor [T, H]
        router_weights: Router projection weights [H, E]
        top_k: Number of experts per token
        router_bias: Optional router bias [E]
        activation: Activation function ("softmax", "sigmoid", or callable)
        gamma: Optional RMSNorm weights [H]
        eps: RMSNorm epsilon
        computation_dtype: Data type for computation

    Returns:
        Tuple of (expert_affinities [T, E], router_logits [T, E])
    """
    T, H = hidden_states.shape
    E = router_weights.shape[1]
    device = hidden_states.device

    # Step 1: Optional RMSNorm preprocessing
    if gamma is not None:
        hidden_states = _torch_rms_norm(hidden_states, gamma, eps)

    # Step 2: Router linear projection
    router_logits = F.linear(
        hidden_states.to(computation_dtype),
        router_weights.T.to(computation_dtype),
        router_bias.to(computation_dtype) if router_bias is not None else None,
    )  # [T, E]

    # Step 3: Apply activation function to ALL logits
    router_probs = _apply_activation(activation, router_logits)  # [T, E]

    # Step 4: Top-k selection from activated values
    router_top_probs, router_indices = torch.topk(
        router_probs, top_k, dim=-1
    )  # [T, top_k]

    # Step 5: L1 renormalization of top-k probabilities (always applied for this computation order)
    router_top_probs = router_top_probs / router_top_probs.sum(dim=-1, keepdim=True)

    # Step 6: Scatter to full expert affinity matrix
    expert_affinities = torch.zeros(T, E, device=device, dtype=computation_dtype)
    expert_affinities.scatter_(1, router_indices, router_top_probs)  # [T, E]

    return expert_affinities, router_logits


def _torch_router_impl_prenorm_linear_topk_scatter_act(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor],
    activation: Union[str, Callable[[Tensor], Tensor]],
    gamma: Optional[Tensor],
    eps: float,
    computation_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """
    PyTorch implementation: RMSNorm (optional) → Linear → TopK → Scatter → Activation

    Applies optional RMSNorm to hidden states, projects to router logits, selects top-k
    based on raw logits, scatters raw logit values to the full matrix (zeros elsewhere),
    then applies activation to the full sparse matrix.

    Args:
        hidden_states: Input tensor [T, H]
        router_weights: Router projection weights [H, E]
        top_k: Number of experts per token
        router_bias: Optional router bias [E]
        activation: Activation function ("softmax", "sigmoid", or callable)
        gamma: Optional RMSNorm weights [H]
        eps: RMSNorm epsilon
        computation_dtype: Data type for computation

    Returns:
        Tuple of (expert_affinities [T, E], router_logits [T, E])
    """
    T, H = hidden_states.shape
    E = router_weights.shape[1]
    device = hidden_states.device

    # Step 1: Optional RMSNorm preprocessing
    if gamma is not None:
        hidden_states = _torch_rms_norm(hidden_states, gamma, eps)

    # Step 2: Router linear projection
    router_logits = F.linear(
        hidden_states.to(computation_dtype),
        router_weights.T.to(computation_dtype),
        router_bias.to(computation_dtype) if router_bias is not None else None,
    )  # [T, E]

    # Step 3: Top-k selection based on raw logits
    router_top_values, router_indices = torch.topk(
        router_logits, top_k, dim=-1
    )  # [T, top_k]

    # Step 4: Scatter raw logit values to matrix initialized with -inf (not zeros)
    expert_affinities = torch.full(
        (T, E), float("-inf"), device=device, dtype=computation_dtype
    )
    expert_affinities.scatter_(1, router_indices, router_top_values)  # [T, E]

    # Step 5: Apply activation function to the full sparse matrix
    expert_affinities = _apply_activation(activation, expert_affinities)  # [T, E]

    return expert_affinities, router_logits


def _apply_activation(
    activation: Union[str, Callable[[Tensor], Tensor]], x: Tensor
) -> Tensor:
    """
    Apply activation function to input tensor.

    Args:
        activation: Activation function ("softmax", "sigmoid", or callable)
        x: Input tensor

    Returns:
        Activated tensor

    Raises:
        ValueError: If activation is not a valid string or callable
    """
    if isinstance(activation, str):
        if activation == "softmax":
            return F.softmax(x, dim=-1)
        elif activation == "sigmoid":
            return torch.sigmoid(x)
        else:
            raise ValueError(
                f"Unsupported activation function: {activation}. Use 'softmax' or 'sigmoid'."
            )
    elif callable(activation):
        return activation(x)
    else:
        raise ValueError(
            f"Activation must be either a string ('softmax' or 'sigmoid') or a callable function. Got: {type(activation)}"
        )


def _validate_router_inputs(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor] = None,
    gamma: Optional[Tensor] = None,
    computation_dtype: torch.dtype = torch.float32,
    router_computation_order: RouterComputationOrder = RouterComputationOrder.PRENORM_LINEAR_TOPK_ACT_SCATTER,
    transposed_hidden_states: bool = False,
) -> None:
    """
    Validate input parameters for router function.

    This function performs comprehensive input validation for the router function,
    ensuring all tensors have correct dimensions and shapes, and that parameters
    are within valid ranges.

    Args:
        hidden_states: Input hidden states tensor, expected shape [T, H]
        router_weights: Router projection weights, expected shape [H, E]
        top_k: Number of top experts to select per token
        router_bias: Optional router projection bias, expected shape [E]
        gamma: Optional RMSNorm weights, expected shape [H]
        computation_dtype: Data type for computation
        router_computation_order: Specifies the order of operations

    Raises:
        ValueError: If any input parameter has invalid shape or value
    """
    # Validate hidden_states dimensions
    if hidden_states.dim() != 2:
        raise ValueError(
            f"Expected hidden_states to be 2D [T, H], got shape {hidden_states.shape}"
        )

    # Validate router_weights dimensions
    if router_weights.dim() != 2:
        raise ValueError(
            f"Expected router_weights to be 2D [H, E], got shape {router_weights.shape}"
        )

    if transposed_hidden_states:
        H, T = hidden_states.shape
    else:
        T, H = hidden_states.shape
    H_w, E = router_weights.shape

    # Validate dimension compatibility
    if H != H_w:
        raise ValueError(
            f"Hidden dimension mismatch: hidden_states has {H}, router_weights has {H_w}"
        )

    # Validate optional router_bias shape
    if router_bias is not None and router_bias.shape != (E,):
        raise ValueError(f"Expected router_bias shape [E], got {router_bias.shape}")

    # Validate optional gamma shape
    if gamma is not None and gamma.shape != (H,):
        raise ValueError(f"Expected gamma shape [H], got {gamma.shape}")

    # Validate top_k range
    if top_k < 1 or top_k > E:
        raise ValueError(f"top_k must be between 1 and {E}, got {top_k}")

    # Validate computation_dtype
    supported_dtypes = {torch.float32, torch.float16, torch.bfloat16}
    if computation_dtype not in supported_dtypes:
        raise ValueError(
            f"computation_dtype must be one of {supported_dtypes}, got {computation_dtype}"
        )

    # Validate router_computation_order type
    if not isinstance(router_computation_order, RouterComputationOrder):
        raise ValueError(
            f"router_computation_order must be a RouterComputationOrder enum, got {type(router_computation_order)}"
        )


def _torch_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm implementation in PyTorch."""
    original_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    x_normed = x_normed * weight
    return x_normed.to(original_dtype)


def _nki_router_impl(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor],
    activation: str,
    computation_dtype: torch.dtype,
    router_computation_order: RouterComputationOrder,
    skip_store_router_logits: bool,
    shard_on_tokens: Optional[bool],
    x_hbm_layout: int,
    x_sb_layout: Optional[int],
    use_column_tiling: Optional[bool],
    use_indirect_dma_scatter: Optional[bool],
    use_PE_broadcast_w_bias: Optional[bool],
) -> Tuple[Tensor, Tensor]:
    """
    NKI kernel implementation of router computation.

    Currently only supports PRENORM_LINEAR_TOPK_ACT_SCATTER and PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER computation orders.

    Args:
        hidden_states: [T, H] input tensor
        router_weights: [H, E] weight tensor
        top_k: Number of experts per token
        router_bias: Optional [E] bias tensor
        activation: "softmax" or "sigmoid"
        computation_dtype: Computation dtype
        router_computation_order: Specifies the order of operations
        skip_store_router_logits: Skips storing router logits to HBM
        shard_on_tokens: Enable LNC sharding across token dimension
        x_hbm_layout: Layout of input x in HBM (0=[H,T], 1=[T,H])
        x_sb_layout: Layout of input x in SBUF
        use_column_tiling: Enable PE array column tiling for small T
        use_indirect_dma_scatter: Use indirect DMA for expert affinity scatter
        use_PE_broadcast_w_bias: Use tensor engine for bias broadcast

    Returns:
        Tuple of (expert_affinities [T, E], router_logits [T, E])
    """
    # HBM layout dictates the expected hidden states shape
    if x_hbm_layout == 0:
        H, T = hidden_states.shape
    else:
        T, H = hidden_states.shape
    E = router_weights.shape[1]
    device = hidden_states.device

    # Set kernel args to reasonable defaults if not provided
    if shard_on_tokens is None:
        shard_on_tokens = T >= 128  # Enable LNC sharding when using a high token count
    if x_sb_layout is None:
        x_sb_layout = 0
    if use_column_tiling is None:
        # TODO: Default to True once NKILIB-584 is resolved
        use_column_tiling = False
    if use_indirect_dma_scatter is None:
        # TODO: Default to False once NKILIB-615 is resolved
        use_indirect_dma_scatter = True
    if use_PE_broadcast_w_bias is None:
        use_PE_broadcast_w_bias = False

    act_fn = (
        RouterActFnType.SOFTMAX if activation == "softmax" else RouterActFnType.SIGMOID
    )

    router_logits = torch.zeros(T, E, dtype=computation_dtype, device=device)
    expert_affinities = torch.zeros(T, E, dtype=computation_dtype, device=device)
    expert_index = torch.zeros(T, top_k, dtype=torch.int32, device=device)

    w_bias = router_bias.unsqueeze(0) if router_bias is not None else None

    # Map computation order to kernel's router_pre_norm parameter
    # router_pre_norm=True -> PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER (activation before topk)
    # router_pre_norm=False -> PRENORM_LINEAR_TOPK_ACT_SCATTER (activation after topk)
    router_pre_norm = (
        router_computation_order
        == RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER
    )

    # For PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER, always apply L1 renormalization
    norm_topk_prob = router_pre_norm

    router_topk_nki = wrap_nki(router_topk_jit)

    router_logits, expert_index, expert_affinities = router_topk_nki[2](
        x=hidden_states,
        w=router_weights,
        w_bias=w_bias,
        router_logits=router_logits,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=act_fn,
        k=top_k,
        x_hbm_layout=x_hbm_layout,
        x_sb_layout=x_sb_layout,
        router_pre_norm=router_pre_norm,
        norm_topk_prob=norm_topk_prob,
        use_indirect_dma_scatter=use_indirect_dma_scatter,
        use_column_tiling=use_column_tiling,
        shard_on_tokens=shard_on_tokens,
        skip_store_router_logits=skip_store_router_logits,
        skip_store_expert_index=True,
        use_PE_broadcast_w_bias=use_PE_broadcast_w_bias,
    )

    return expert_affinities, router_logits


def _can_use_kernel(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    activation: Union[str, Callable],
    gamma: Optional[Tensor],
    router_bias: Optional[Tensor] = None,
    router_computation_order: RouterComputationOrder = RouterComputationOrder.PRENORM_LINEAR_TOPK_ACT_SCATTER,
    transposed_hidden_states: bool = False,
) -> bool:
    """
    Check if the NKI kernel can be used for router computation.

    Kernel constraints from router_topk_kernel_nki:
    - K <= 8
    - T <= 128 or (T <= 2048 and T % 128 == 0)
    - E <= 512
    - (H % 128) == 0
    - Activation must be "softmax" or "sigmoid" (string only)
    - No RMSNorm support (gamma must be None)
    - Device must be XLA (Neuron)
    - router_bias must be None or shape [E]
    - Only PRENORM_LINEAR_TOPK_ACT_SCATTER and PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER computation orders supported

    Returns:
        bool: True if kernel can be used, False otherwise
    """

    # TODO: Remove this after debugging compilation issue on TRN3
    return False

    if not can_run_kernel(hidden_states):
        return False

    if transposed_hidden_states:
        H, T = hidden_states.shape
    else:
        T, H = hidden_states.shape
    E = router_weights.shape[1]

    if top_k > 8 or E > 512 or H % 128 != 0:
        return False

    # TODO: Remove T <= 2048 requirements when NKILIB-618 is resolved
    if T > 128 and (T > 2048 or T % 128 != 0):
        return False

    if not isinstance(activation, str) or activation not in ["softmax", "sigmoid"]:
        return False

    if gamma is not None:
        return False

    # Bias shape validation
    if router_bias is not None and router_bias.shape != (E,):
        return False

    # Only PRENORM_LINEAR_TOPK_ACT_SCATTER and PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER are supported by kernel
    # PRENORM_LINEAR_TOPK_SCATTER_ACT requires PyTorch fallback
    if (
        router_computation_order
        == RouterComputationOrder.PRENORM_LINEAR_TOPK_SCATTER_ACT
    ):
        return False

    return True


#: Tokens per tile: `nisa.max8`/`nisa.nc_find_index8` work one token per partition.
NOAUX_TC_TILE = 128

#: `nisa.max8` emits, and `nisa.nc_find_index8` consumes, exactly 8 values per token.
NOAUX_TC_K = 8

#: Guard term in the L1 denominator, verbatim from the reference implementation.
NOAUX_TC_DENOM_EPS = 1e-20

#: nkilib's router caps E at its gemm moving free-dim maximum.
_NOAUX_TC_F_MAX = 512

#: Token multiple for the fused two-core launch: a whole 128-row tile per core.
_NOAUX_TC_T_MULTIPLE = 256

_NOAUX_TC_TORCH_TO_NKI_DTYPE = {
    torch.bfloat16: nl.bfloat16,
    torch.float16: nl.float16,
    torch.float32: nl.float32,
}


class NoauxTcRouterError(ValueError):
    """Raised for a `top_k` or expert count the NKI `noaux_tc` stage cannot serve.

    Raised rather than returned as `False`: nkilib's admission gate answers `E > 512`
    with `False` and its caller then takes the torch path, so a bad extent would run
    torch silently. Here the same case raises, and the kernel path cannot be skipped
    by accident.
    """


@dataclass
class _NoauxTcCounters:
    """Entries into the NKI path and into the torch fallback, counted separately."""

    nki_dispatch: int = 0
    torch_fallback: int = 0


_NOAUX_TC_COUNTERS = _NoauxTcCounters()


def reset_noaux_tc_counters() -> None:
    """Zero both dispatch counters."""
    _NOAUX_TC_COUNTERS.nki_dispatch = 0
    _NOAUX_TC_COUNTERS.torch_fallback = 0


def noaux_tc_dispatch_counters() -> Tuple[int, int]:
    """Return ``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _NOAUX_TC_COUNTERS.nki_dispatch, _NOAUX_TC_COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch outside the trace, so the store is not a guard."""
    _NOAUX_TC_COUNTERS.nki_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry outside the trace."""
    _NOAUX_TC_COUNTERS.torch_fallback += 1


def _require_noaux_tc_extents(num_experts: int, top_k: int) -> None:
    """Raise `NoauxTcRouterError` for a `top_k` or `E` the ISA top-k path cannot serve.

    The token extent is not checked here: both entry points pad it to a whole tile.
    """
    if top_k != NOAUX_TC_K:
        raise NoauxTcRouterError(
            f"top_k must be exactly {NOAUX_TC_K}: `nisa.max8` emits 8 values "
            f"per partition and `nisa.nc_find_index8` consumes exactly 8, and "
            f"nkilib refuses k > 8 (router_topk.py:582-583). got top_k={top_k}"
        )
    if num_experts < NOAUX_TC_K:
        raise NoauxTcRouterError(
            f"E must be >= {NOAUX_TC_K} for the ISA top-K members "
            f"(router_topk.py:312 pads E to at least 8). got E={num_experts}"
        )
    if num_experts > _NOAUX_TC_F_MAX:
        raise NoauxTcRouterError(
            f"E must be <= {_NOAUX_TC_F_MAX}, the gemm moving free-dim cap the "
            f"substrate applies at rmsnorm_router_topk_tkg.py:201,206. "
            f"got E={num_experts}"
        )


def can_run_noaux_tc_router(
    reference: Tensor, num_experts: int, top_k: int
) -> bool:
    """True when a Neuron device or simulator can run the kernel for `reference`.

    Extents are checked first and a bad `top_k` or `E` raises `NoauxTcRouterError`;
    only the absence of a device sends this path to the torch reference. Independent
    of `_can_use_kernel` above, which currently returns False unconditionally.
    """
    _require_noaux_tc_extents(num_experts, top_k)
    return can_run_kernel(reference)


def _noaux_tc_pad_target(num_tokens: int, multiple: int) -> int:
    """Round `num_tokens` up to a whole `multiple`.

    Each entry point passes its own multiple: 128 (one tile) for `noaux_tc_correct`,
    which launches without a grid, and 256 for the `[2]`-launched fused entry, whose
    two cores each need a whole 128-row tile. With 128 there, a core could receive
    64 rows, the tile loop would run zero times, and the kernel would return its
    uninitialised output buffers.
    """
    return -(-num_tokens // multiple) * multiple


def _noaux_tc_pad_tokens(x: Tensor, t_pad: int) -> Tensor:
    """Pad the token axis (`dim=-2`) of `x` to `t_pad` rows; always contiguous.

    The pad repeats the last real row instead of writing zeros. Pad rows cannot reach
    the real ones (the stage never reduces across tokens), but an all-zero row gives
    `sigmoid(0) == 0.5` at every expert: an exact 8-way tie for `nisa.max8` and
    `nisa.nc_find_index8` that real rows never produce. Contiguous because the kernel
    reads the buffer from HBM, and a non-contiguous view is not the same bytes.
    """
    t_real = x.shape[-2]
    if t_pad == t_real:
        return x.contiguous()
    last = x.narrow(-2, t_real - 1, 1)
    reps = [1] * x.dim()
    reps[-2] = t_pad - t_real
    return torch.cat([x, last.repeat(*reps)], dim=-2).contiguous()


def _noaux_tc_shard_range(num_tokens: int, n_prgs: int, prg_id: int):
    """Return `(t_offset, t_local)`: the token rows program `prg_id` owns.

    Same split as nkilib's `router_topk`: `T // n_prgs` rows to program 0 and the
    remainder to program 1, so this stage reads exactly the logit rows the router
    wrote on the same core. Only two programs are defined, as
    `get_verified_program_sharding_info(..., (0, 1), 2)` admits; `n_prgs == 1`
    returns the whole range. Callers pad `num_tokens` so `t_local` is a whole number
    of `NOAUX_TC_TILE` rows; an unpadded extent would floor-divide to too few tiles.
    """
    t_first = num_tokens // n_prgs
    if prg_id == 0:
        return 0, t_first
    return t_first, num_tokens - t_first


def _noaux_tc_stage(
    router_logits_hbm,
    correction_bias_hbm,
    expert_index_hbm,
    expert_affinities_hbm,
    num_tokens: int,
    num_experts: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
    shard_on_tokens: bool,
):
    """The `noaux_tc` numerics in NKI: a plain subkernel both jit entry points inline.

    Reads `[T, E]` logits and a `[1, E]` correction bias from HBM; writes the `[T, K]`
    selected indices and the `[T, E]` scattered gate weights. Per token, following
    `Glm5NextTextTopkRouter.forward`::

        scores            = sigmoid(logits)
        scores_for_choice = scores + correction_bias
        topk_indices      = topk(scores_for_choice, k)   # nisa.max8 + nc_find_index8
        topk_weights      = scores[topk_indices]         # one-hot mask, below
        if norm_topk_prob:
            topk_weights /= topk_weights.sum() + 1e-20
        topk_weights     *= routed_scaling_factor

    The gather is a one-hot mask over the selected indices, not a DMA gather: the MoE
    block consumes the scattered `[T, E]` form anyway, and an index mask selects
    exactly the eight reported columns where a value-equality mask against the top-8
    values would select nine when two experts tie on the corrected score.

    When `shard_on_tokens` is true the launch has two programs and this program
    covers only the token rows the nkilib router wrote on this core; when false one
    program covers the whole extent.
    """
    bias_sb = nl.load(correction_bias_hbm)  # [1, E]

    # Cover only the token rows the nkilib router wrote on this core. The router
    # shards its stores by this same query, so no cross-core read remains and no
    # barrier is needed. `shard_on_tokens` is passed in, not inferred from the grid:
    # the router shards only when asked (never at T == 1), and the two must agree.
    if shard_on_tokens:
        _grid_ndim, n_prgs, prg_id = get_verified_program_sharding_info(
            "noaux_tc_stage", (0, 1), 2
        )
    else:
        n_prgs, prg_id = 1, 0
    t_offset, t_local = _noaux_tc_shard_range(num_tokens, n_prgs, prg_id)

    for t_tile in range(t_local // NOAUX_TC_TILE):
        t0 = t_offset + t_tile * NOAUX_TC_TILE
        rows = NOAUX_TC_TILE

        logits_sb = nl.load(
            router_logits_hbm[t0 : t0 + rows, :], dtype=nl.float32
        )

        # fp32 throughout: the selection is discrete, and a bf16 score would decide
        # near-tie experts by round-off rather than by value.
        scores = nl.sigmoid(logits_sb, dtype=nl.float32)

        # The bias broadcast is along the partition axis; `nl.broadcast_to` does
        # that, `tensor_scalar` does not.
        choice = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        bias_b = nl.broadcast_to(bias_sb, (rows, num_experts))
        nisa.tensor_tensor(dst=choice, data1=scores, data2=bias_b, op=nl.add)

        # Top-k on the corrected score, with the same two ISA instructions nkilib's
        # router uses.
        top8 = nl.ndarray((rows, NOAUX_TC_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=top8, src=choice)
        idx8 = nl.ndarray((rows, NOAUX_TC_K), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=idx8, data=choice, vals=top8)

        # One-hot over the selected indices; the indices are cast to fp32 to compare
        # against the fp32 iota.
        idx_f32 = nl.ndarray((rows, NOAUX_TC_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idx_f32, src=idx8)
        col = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=col, pattern=[[1, num_experts]], offset=0, channel_multiplier=0)

        mask = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=mask, value=0.0)
        hit = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        for k in range(NOAUX_TC_K):
            # `tensor_scalar` broadcasts a [par, 1] operand along the free dim.
            nisa.tensor_scalar(
                dst=hit, data=col, op0=nl.equal, operand0=idx_f32[:, k : k + 1]
            )
            nisa.tensor_tensor(dst=mask, data1=mask, data2=hit, op=nl.add)

        # Gate weights from the unbiased scores, masked to the selected experts.
        sel = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sel, data1=scores, data2=mask, op=nl.multiply)

        out = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        if norm_topk_prob:
            # Normalise, then scale, in one pass.
            row_sum = nl.sum(sel, axis=1, keepdims=True, dtype=nl.float32)
            denom = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=denom, data=row_sum, op0=nl.add, operand0=NOAUX_TC_DENOM_EPS
            )
            recip = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=recip, data=denom)
            nisa.tensor_scalar(
                dst=out,
                data=sel,
                op0=nl.multiply,
                operand0=recip,
                op1=nl.multiply,
                operand1=float(routed_scaling_factor),
            )
        else:
            # Scale only.
            nisa.tensor_scalar(
                dst=out,
                data=sel,
                op0=nl.multiply,
                operand0=float(routed_scaling_factor),
            )

        nl.store(expert_affinities_hbm[t0 : t0 + rows, :], value=out)
        nl.store(expert_index_hbm[t0 : t0 + rows, :], value=idx8)


@nki.jit
def _noaux_tc_correct_nki(
    router_logits,
    correction_bias,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
):
    """Run the `noaux_tc` stage alone: `[T, E]` logits in, indices and weights out."""
    t_extent, e_extent = router_logits.shape
    expert_index = nl.ndarray(
        (t_extent, NOAUX_TC_K), dtype=nl.uint32, buffer=nl.shared_hbm
    )
    expert_affinities = nl.ndarray(
        (t_extent, e_extent), dtype=nl.float32, buffer=nl.shared_hbm
    )
    _noaux_tc_stage(
        router_logits_hbm=router_logits,
        correction_bias_hbm=correction_bias,
        expert_index_hbm=expert_index,
        expert_affinities_hbm=expert_affinities,
        num_tokens=t_extent,
        num_experts=e_extent,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        # Launched without a grid: one program owns every token.
        shard_on_tokens=False,
    )
    return expert_index, expert_affinities


@nki.jit
def _noaux_tc_rmsnorm_router_topk_nki(
    hidden_states,
    gamma,
    router_weights,
    correction_bias,
    eps: float = 1e-6,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    router_mm_dtype=nl.bfloat16,
):
    """RMSNorm, router matmul and the `noaux_tc` stage in one kernel.

    Follows `rmsnorm_router_topk_tkg._rmsnorm_router_topk_tkg_nki` with two changes:
    the router logits are stored (the `noaux_tc` stage consumes them, and they are
    returned), and that stage runs after `_router_topk` in the same kernel body.

    The nkilib router's own uncorrected top-k (`substrate_index`) is returned too.
    `noaux_tc` differs from it exactly by the correction bias, so the two selections
    should differ on some rows; identical selections mean the bias was ignored.

    `shard_on_tokens` is decided once and handed to both the nkilib router and the
    `noaux_tc` stage, so they agree on which core owns which token rows.
    """
    b_extent, s_extent, h_extent = hidden_states.shape
    t_extent = b_extent * s_extent
    _, e_extent = router_weights.shape
    h_free = h_extent // _pmax

    # One shard decision for the router and the `noaux_tc` stage.
    shard_on_tokens = t_extent > 1

    router_logits = nl.ndarray((t_extent, e_extent), dtype=nl.float32,
                               buffer=nl.shared_hbm)
    norm_output = nl.ndarray((t_extent, h_extent), dtype=router_mm_dtype,
                             buffer=nl.shared_hbm)
    # The nkilib router's own uncorrected outputs.
    substrate_index = nl.ndarray((t_extent, NOAUX_TC_K), dtype=nl.int32,
                                 buffer=nl.shared_hbm)
    substrate_affinities = nl.ndarray((t_extent, e_extent), dtype=nl.bfloat16,
                                      buffer=nl.shared_hbm)
    # The corrected outputs.
    expert_index = nl.ndarray((t_extent, NOAUX_TC_K), dtype=nl.uint32,
                              buffer=nl.shared_hbm)
    expert_affinities = nl.ndarray((t_extent, e_extent), dtype=nl.float32,
                                   buffer=nl.shared_hbm)

    norm_sb = nl.ndarray((_pmax, t_extent, h_free), dtype=router_mm_dtype,
                         buffer=nl.sbuf)

    # Stage 1: RMSNorm, as `rmsnorm_router_topk_tkg` calls it.
    _rmsnorm_tkg_dloc(
        input_hbm=hidden_states,
        gamma=gamma,
        output_hbm=norm_output,
        output_sb=norm_sb,
        eps=eps,
        hidden_actual=None,
        sync_output=True,
    )

    # Stage 2: router matmul and its own top-k, with `router_logits` stored.
    # `w_bias=None`: the correction bias is not a projection bias and must not be
    # added to the logits; it enters the selection score after the sigmoid.
    _substrate_router_topk(
        x=norm_sb,
        w=router_weights,
        w_bias=None,
        router_logits=router_logits,
        expert_affinities=substrate_affinities,
        expert_index=substrate_index,
        act_fn=RouterActFnType.SIGMOID,
        k=NOAUX_TC_K,
        x_hbm_layout=0,
        x_sb_layout=XSBLayout_tp102__0,
        router_pre_norm=False,
        norm_topk_prob=False,
        use_column_tiling=True,
        use_indirect_dma_scatter=True,
        use_PE_broadcast_w_bias=True,
        shard_on_tokens=shard_on_tokens,
        skip_store_expert_index=False,
        skip_store_router_logits=False,
    )

    # Stage 3: the `noaux_tc` split, in the same kernel.
    _noaux_tc_stage(
        router_logits_hbm=router_logits,
        correction_bias_hbm=correction_bias,
        expert_index_hbm=expert_index,
        expert_affinities_hbm=expert_affinities,
        num_tokens=t_extent,
        num_experts=e_extent,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        # Same shard decision as the router above.
        shard_on_tokens=shard_on_tokens,
    )

    return router_logits, expert_index, expert_affinities, substrate_index


def noaux_tc_correct(
    router_logits: Tensor,
    correction_bias: Tensor,
    top_k: int = NOAUX_TC_K,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """`noaux_tc` selection and gate weights from precomputed router logits.

    Args:
        router_logits: `[T, E]` raw router logits, any `T >= 1`; the token axis is
            padded to a whole tile before the launch and the outputs sliced back.
        correction_bias: `[E]` or `[1, E]` `e_score_correction_bias`.
        top_k: must equal `NOAUX_TC_K`; a mismatch raises rather than reshapes.
        norm_topk_prob: L1-normalise the selected weights.
        routed_scaling_factor: final multiplier on the weights.

    Returns:
        `(expert_index [T, K] int32, expert_affinities [T, E] float32)`.
        `expert_affinities` is scattered: the gate weight at each selected expert's
        column, zero elsewhere.

    Raises:
        NoauxTcRouterError: for a `top_k` or `E` the NKI stage cannot serve.
    """
    if router_logits.dim() != 2:
        raise NoauxTcRouterError(
            f"router_logits must be 2D [T, E], got shape {tuple(router_logits.shape)}"
        )
    num_tokens, num_experts = router_logits.shape
    bias = _legalize_correction_bias(correction_bias, num_experts)

    if not can_run_noaux_tc_router(router_logits, num_experts, top_k):
        _count_torch_fallback()
        return noaux_tc_correct_torch_oracle(
            router_logits, bias, norm_topk_prob, routed_scaling_factor
        )

    # Launched without a grid: one program owns every token, so one 128-row tile
    # is the pad unit (the fused entry needs 256).
    t_pad = _noaux_tc_pad_target(num_tokens, NOAUX_TC_TILE)

    _count_nki_dispatch()
    index, affinities = wrap_nki(_noaux_tc_correct_nki)(
        router_logits=_noaux_tc_pad_tokens(router_logits.to(torch.float32), t_pad),
        correction_bias=bias,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=float(routed_scaling_factor),
    )
    # Drop the pad rows; nothing in the stage reduces across tokens.
    return index[:num_tokens].to(torch.int32), affinities[:num_tokens]


def noaux_tc_rmsnorm_router_topk(
    hidden_states: Tensor,
    gamma: Tensor,
    router_weights: Tensor,
    correction_bias: Tensor,
    top_k: int = NOAUX_TC_K,
    eps: float = 1e-6,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    router_mm_dtype: torch.dtype = torch.bfloat16,
    quantization_type: QuantizationType = QuantizationType.NONE,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fused RMSNorm, router matmul and `noaux_tc` top-8 in one dispatch.

    This is the form the MoE block calls. Args mirror `rmsnorm_router_topk_tkg`, with
    `correction_bias` in place of `router_bias`: a projection bias is added to the
    logits, while `e_score_correction_bias` is added to the sigmoid scores for
    selection only.

    Returns:
        `(router_logits [T, E], expert_index [T, K] int32,
          expert_affinities [T, E] float32, substrate_index [T, K] int32)`.
        `substrate_index` is the nkilib router's own uncorrected selection.

    Raises:
        AssertionError: from nkilib's `_validate_inputs`, run first on the caller's
            real shapes, so a wrong `H` raises on the NKI and torch routes alike.
        NoauxTcRouterError: for a `top_k` or `E` the NKI stage cannot serve.
    """
    # nkilib's own validation first; `gamma` is legalised to [1, H] as nkilib does.
    if gamma.ndim == 1:
        gamma = gamma.unsqueeze(0)
    _substrate_validate_inputs(
        hidden_states,
        gamma,
        router_weights,
        None,
        top_k,
        None,
        quantization_type,
        RouterActFnType.SIGMOID,
    )

    b_extent, s_extent, h_extent = hidden_states.shape
    num_tokens = b_extent * s_extent
    num_experts = router_weights.shape[1]
    bias = _legalize_correction_bias(correction_bias, num_experts)

    # Pad to a multiple of 256: the `[2]` launch splits the tokens and each core
    # needs a whole 128-row tile. Padding runs after validation (which must see the
    # caller's real shapes) and before the admission gate below (which returns
    # False on a token extent that is not a multiple of 256). Reshaping to
    # `[1, T, H]` first puts the pad rows after all real tokens, so one slice
    # recovers them; padding `S` per batch would interleave them.
    t_pad = _noaux_tc_pad_target(num_tokens, _NOAUX_TC_T_MULTIPLE)
    hidden_padded = _noaux_tc_pad_tokens(
        hidden_states.reshape(1, num_tokens, h_extent), t_pad
    )

    # Both gates read the padded tensor. nkilib's gate returns False on anything it
    # would refuse; this module's gate raises on the same extents, so nothing can
    # be admitted here and refused there.
    substrate_admits = _substrate_can_use_kernel(
        hidden_padded, router_weights, router_mm_dtype, quantization_type
    )
    seam_admits = can_run_noaux_tc_router(hidden_padded, num_experts, top_k)
    if not (substrate_admits and seam_admits):
        _count_torch_fallback()
        return noaux_tc_rmsnorm_router_topk_torch_oracle(
            hidden_states,
            gamma,
            router_weights,
            bias,
            eps,
            norm_topk_prob,
            routed_scaling_factor,
            router_mm_dtype,
        )

    _count_nki_dispatch()
    # `[2]` is the SPMD launch grid: the nkilib subkernels shard over two logical
    # cores (LNC=2).
    wrapped = wrap_nki(_noaux_tc_rmsnorm_router_topk_nki)
    logits, index, affinities, substrate_index = wrapped[2](
        hidden_states=hidden_padded,
        gamma=gamma,
        router_weights=router_weights,
        correction_bias=bias,
        eps=eps,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=float(routed_scaling_factor),
        router_mm_dtype=_NOAUX_TC_TORCH_TO_NKI_DTYPE[router_mm_dtype],
    )
    # All four outputs are `[t_pad, ...]`; slice every one back to the caller's
    # extent.
    return (
        logits[:num_tokens],
        index[:num_tokens].to(torch.int32),
        affinities[:num_tokens],
        substrate_index[:num_tokens].to(torch.int32),
    )


def _legalize_correction_bias(correction_bias: Tensor, num_experts: int) -> Tensor:
    """Accept `[E]` or `[1, E]`; return a contiguous fp32 `[1, E]`.

    fp32, not the model dtype: the bias decides a discrete selection, and a bf16 bias
    would quantise the correction to about three decimal digits and merge experts the
    checkpoint separates.
    """
    if correction_bias.dim() == 1:
        correction_bias = correction_bias.unsqueeze(0)
    if correction_bias.shape != (1, num_experts):
        raise NoauxTcRouterError(
            f"correction_bias must be [E] or [1, E] with E={num_experts}, "
            f"got shape {tuple(correction_bias.shape)}"
        )
    return correction_bias.to(torch.float32).contiguous()


def noaux_tc_correct_torch_oracle(
    router_logits: Tensor,
    correction_bias: Tensor,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Torch reference for `noaux_tc`; used only when no Neuron device is available.

    Follows `transformers` 5.16.1 `Glm5NextTextTopkRouter.forward`. Its group-routing
    stage is omitted: the model has `n_group == 1`, which makes that stage an
    identity.
    """
    logits = router_logits.to(torch.float32)
    bias = correction_bias.to(torch.float32).reshape(-1)
    scores = logits.sigmoid()
    scores_for_choice = scores + bias
    topk_indices = torch.topk(
        scores_for_choice, k=NOAUX_TC_K, dim=-1, sorted=False
    )[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + NOAUX_TC_DENOM_EPS
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * routed_scaling_factor

    affinities = torch.zeros_like(scores)
    affinities.scatter_(1, topk_indices, topk_weights)
    return topk_indices.to(torch.int32), affinities


def noaux_tc_rmsnorm_router_topk_torch_oracle(
    hidden_states: Tensor,
    gamma: Tensor,
    router_weights: Tensor,
    correction_bias: Tensor,
    eps: float = 1e-6,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    router_mm_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Torch reference for the fused form; used only when no Neuron device is available.

    RMSNorm and matmul follow `rmsnorm_router_topk_tkg._torch_impl`: operands cast to
    `router_mm_dtype`, accumulation in fp32, matching the tensor engine. A true
    bf16-accumulation matmul over H flips near-tie expert selections.
    """
    b_extent, s_extent, h_extent = hidden_states.shape
    num_tokens = b_extent * s_extent

    hidden_f32 = hidden_states.to(torch.float32).reshape(num_tokens, h_extent)
    gamma_f32 = gamma.to(torch.float32)
    inv_rms = torch.rsqrt(
        torch.mean(hidden_f32**2, dim=-1, keepdim=True) + eps
    )
    norm = (hidden_f32 * inv_rms * gamma_f32).to(router_mm_dtype)

    logits = norm.to(router_mm_dtype).float() @ router_weights.to(
        router_mm_dtype
    ).float()

    index, affinities = noaux_tc_correct_torch_oracle(
        logits, correction_bias, norm_topk_prob, routed_scaling_factor
    )
    # The nkilib router's uncorrected selection: top-k on the raw logits.
    substrate_index = torch.topk(logits, k=NOAUX_TC_K, dim=-1)[1].to(torch.int32)
    return logits, index, affinities, substrate_index
