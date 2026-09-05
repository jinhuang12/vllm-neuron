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

# --- inc-glm53f-032 additions. Pure additions: no line above is edited. ---
from dataclasses import dataclass

import nki.isa as nisa
import nki.language as nl

from nkilib.core.moe_block.moe_block_tkg_utils import _pmax
from nkilib.core.router_topk.router_topk import XSBLayout_tp102__0
from nkilib.core.router_topk.router_topk import router_topk as _substrate_router_topk
from nkilib.core.subkernels.rmsnorm_tkg import _rmsnorm_tkg_dloc
from nkilib.core.utils.common_types import QuantizationType
# The substrate's OWN sharding query -- the same call the vendor router uses to
# decide its `T_local`/`T_offset` (`router_topk.py:217`). Imported rather than
# reimplemented so the authored stage below and the vendor producer cannot
# disagree about which core owns which tokens. Repair round 1 of
# `inc-glm53f-032`, finding `M-B20-1`.
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from vllm_neuron.functional.moe.rmsnorm_router_topk_tkg import (
    _can_use_kernel as _substrate_can_use_kernel,
)
from vllm_neuron.functional.moe.rmsnorm_router_topk_tkg import (
    _validate_inputs as _substrate_validate_inputs,
)
# --- end inc-glm53f-032 import additions ---

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


# ===========================================================================
# inc-glm53f-032 -- WP7 router: top-8 sigmoid with `noaux_tc`.
#
# EVERYTHING BELOW THIS BANNER IS A PURE ADDITION. No line above it is edited:
# `router()`, `_can_use_kernel()`, `_nki_router_impl()`, `_torch_router_impl()`
# and their validators are the pin's (last touched by `ed3580d`, "Release
# 0.24.0.1.1.0") and stay byte-identical. In particular the pin's
# `_can_use_kernel` opens with an unconditional `return False` ("TODO: Remove
# this after debugging compilation issue on TRN3"), so the pin's `router()`
# never reaches NKI. That line is NOT removed here -- removing it would change
# landed pin behaviour for every existing caller, which is a design question
# and not this increment's. This increment brings its OWN gate,
# `can_run_noaux_tc_router()`, which consults `can_run_kernel` directly.
#
# WHAT THE SUBSTRATE PROVIDES AND WHAT IS AUTHORED (P13, plan D6 and the
# Substrate bullet at increment-plan.md L929).
#
#   PROVIDED, reused, not authored:
#     * RMSNorm            -- `nkilib.core.subkernels.rmsnorm_tkg._rmsnorm_tkg_dloc`
#     * router matmul      -- `nkilib.core.router_topk.router_topk`
#     * top-K SELECTION    -- `nisa.max8` + `nisa.nc_find_index8`, the SAME two
#       ISA members `router_topk` itself uses (`router_topk.py:598`, `:609`).
#       No sort, no scan and no comparison network is authored here.
#
#   AUTHORED, in NKI, and it is the only authored numerics:
#     * the `noaux_tc` SPLIT. `noaux_tc` selects on `sigmoid(logits) + bias`
#       but takes its gate weight from the UNBIASED `sigmoid(logits)`. No
#       argument of the substrate kernel can separate the selection score from
#       the weight score: `router_topk` adds `w_bias` to the LOGITS, before the
#       activation, and with `router_pre_norm=False` it activates only the
#       already-selected values (`rmsnorm_router_topk_tkg.py:260-268`). So the
#       split is the gap, and closing it in torch beside the kernel would be
#       exactly the P13 violation the plan forbids -- hence it is closed INSIDE
#       the same dispatch, below.
#
# THE REFERENCE IS NOT FROM MEMORY. `transformers` 5.16.1 ships
# `models/glm5_next/modeling_glm5_next.py::Glm5NextTextTopkRouter.forward`
# (:158-183). Its group-routing stage (:163-176) is an IDENTITY for this
# checkpoint, and that is read from the campaign's own pinned bytes rather than
# assumed: `test/vllm_neuron/model/glm5_next/fixtures/config.json` carries
# `n_group = 1`, so `group_idx` is always group 0, `group_mask` is all ones and
# `masked_fill(~all_ones)` masks nothing. `test_router.py` measures that
# identity against the verbatim upstream function instead of asserting it.
# ===========================================================================

#: The partition-dim tile the ISA top-K members work over. `nisa.max8` and
#: `nisa.nc_find_index8` are per-partition instructions, so one token per
#: partition and at most 128 tokens per tile.
NOAUX_TC_TILE = 128

#: `nisa.max8` emits exactly 8 values per partition and `nisa.nc_find_index8`
#: consumes exactly 8 -- both are fixed by the instructions, not by a choice
#: here (see their docstrings, and `router_topk.py:582-583` which refuses k > 8).
#: This checkpoint's `num_experts_per_tok` is 8 (`glm5_next/config.py:187`), so
#: the model sits exactly on the instruction width and no masking pass is owed.
NOAUX_TC_K = 8

#: `modeling_glm5_next.py:180`, verbatim: the L1 denominator's guard term. Kept
#: as the upstream constant rather than a rounder one, because the fork's output
#: is compared against upstream's and a different guard is a different function.
NOAUX_TC_DENOM_EPS = 1e-20
#: The substrate's own caps, restated at their source lines rather than
#: re-derived. `_F_MAX` is a local in `rmsnorm_router_topk_tkg._can_use_kernel`
#: (`:201`) so it cannot be imported; the value is carried here with its cite.
_NOAUX_TC_F_MAX = 512
_NOAUX_TC_T_MULTIPLE = 256
#: Same map as `rmsnorm_router_topk_tkg._TORCH_TO_NKI_DTYPE`, rebuilt here rather
#: than imported, because importing a private name for a three-entry dtype table
#: would couple this seam to that module's internals for no gain.
_NOAUX_TC_TORCH_TO_NKI_DTYPE = {
    torch.bfloat16: nl.bfloat16,
    torch.float16: nl.float16,
    torch.float32: nl.float32,
}


class NoauxTcRouterError(ValueError):
    """A geometry this seam refuses, named rather than coerced.

    Raised in preference to a silent `False`. The distinction is the whole
    reason this class exists: the substrate's own admission gate answers the
    `E > 512` question by RETURNING FALSE
    (`rmsnorm_router_topk_tkg.py:206`, inside `_can_use_kernel`), and its caller
    then computes the torch reference (`:94`, `:117-124`). A test whose oracle
    is also torch would pass green with no kernel executed -- the false green
    plan section 4b exists to block. On this seam the same rule is a NAMED
    RAISE, so that outcome is unreachable rather than merely unlikely.

    THE EXAMPLE USED TO BE `T % 256`, and it is `E > 512` now because the token
    extent is no longer refused here at all: `inc-glm53f-088` pads the token axis
    at each entry point instead (`_noaux_tc_pad_target` below). The argument is
    unchanged and the substrate still returns a silent `False` on `E`, so the
    class keeps its warrant with a clause that is still live.
    """


@dataclass
class _NoauxTcCounters:
    """What route actually ran, counted rather than inferred.

    ``nki_dispatch`` counts entries into this module's ``wrap_nki`` seams;
    ``torch_fallback`` counts entries into the torch reference. Two counters
    rather than one flag, so "the kernel ran" and "the fallback did not run" are
    independent readings and a test can require both.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: MODULE-LEVEL, and that is a contract rather than an implementation detail:
#: the `-025`/`-026` precedent is that a sibling increment counts this seam from
#: its OWN test module (form R-2), so the counter must be resettable and
#: readable from outside this module and outside this increment's test.
_NOAUX_TC_COUNTERS = _NoauxTcCounters()


def reset_noaux_tc_counters() -> None:
    """Zero both counters. Called at the start of each declared test case."""
    _NOAUX_TC_COUNTERS.nki_dispatch = 0
    _NOAUX_TC_COUNTERS.torch_fallback = 0


def noaux_tc_dispatch_counters() -> Tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _NOAUX_TC_COUNTERS.nki_dispatch, _NOAUX_TC_COUNTERS.torch_fallback


def _require_noaux_tc_extents(num_experts: int, top_k: int) -> None:
    """Refuse, by name, every extent the reused members cannot serve.

    Each clause cites the line that imposes it, so a refusal message is
    actionable and a future extent change can be checked against its source
    rather than against this function's memory of it.

    THE TOKEN EXTENT IS NOT ONE OF THEM, since `inc-glm53f-088`. It used to be:
    a fourth clause refused any `T` that was not a multiple of 256. The two
    public entry points now pad the token axis to their own tile multiple and
    slice the outputs back, so every `T >= 1` is servable and there is nothing
    left to refuse. `num_tokens` is gone from this signature rather than kept and
    ignored -- a parameter named for a screen this function no longer performs
    would make the signature claim more than the body does.
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
    """Two independent conditions, deliberately not merged.

    ``_require_noaux_tc_extents`` answers "does this seam accept these extents"
    and RAISES when it does not; ``can_run_kernel`` answers "is there a device
    or a simulator" and is the only thing allowed to send this seam to the torch
    reference. A geometry the reused members cannot serve must raise rather than
    fall back, because falling back would ship a torch path for kernel-class
    work (P13, plan D6).

    This function does NOT consult the pin's ``_can_use_kernel`` above, whose
    first statement is an unconditional ``return False``.

    ``num_tokens`` left this signature with `inc-glm53f-088`, for the reason
    given in ``_require_noaux_tc_extents``: the token extent no longer bears on
    the answer, and an argument that does not bear on the answer would advertise
    a screen that is not here.
    """
    _require_noaux_tc_extents(num_experts, top_k)
    return can_run_kernel(reference)


def _noaux_tc_pad_target(num_tokens: int, multiple: int) -> int:
    """Round ``num_tokens`` UP to a whole ``multiple``. Returns it unchanged when
    it already is one.

    ``multiple`` IS AN ARGUMENT ON PURPOSE. The two public entry points have
    different pad targets and each must read its own from the constant that
    imposes it:

      * ``noaux_tc_correct`` launches with NO grid, so one program owns every
        token and the stage's tile loop needs whole ``NOAUX_TC_TILE`` (128) rows.
      * ``noaux_tc_rmsnorm_router_topk`` launches ``[2]`` and the stage binds
        ``shard_on_tokens = t_extent > 1``, so the extent splits in two and EACH
        HALF must be a whole tile -- ``_NOAUX_TC_T_MULTIPLE`` (256).

    A helper that picked the multiple itself would have to know which caller it
    served. Copying one entry's target to the other is the specific defect this
    signature makes visible rather than possible: at 128 the fused entry would
    hand each core 64 rows, ``64 // 128`` is 0 tiles, the loop body would never
    run, and the kernel would return its uninitialised output buffers while the
    seam counted a successful NKI dispatch. A silent all-zero result behind a
    green route reading is exactly the false green this campaign counts.
    """
    return -(-num_tokens // multiple) * multiple


def _noaux_tc_pad_tokens(x: Tensor, t_pad: int) -> Tensor:
    """Extend ``x``'s token axis (``dim=-2``) to ``t_pad`` rows, contiguous.

    THE PAD REPEATS THE LAST REAL ROW rather than writing zeros, and that is a
    choice with a reason. The pad rows cannot affect the real ones -- nothing in
    ``_noaux_tc_stage`` reduces across the token axis, every load and store there
    is ``[t0 : t0 + rows, :]``, and the only reduction is ``nl.sum(..., axis=1)``
    over EXPERTS -- so this is not a correctness argument. It is about not
    inventing an input class: ``sigmoid(0)`` is 0.5 at every expert, so an
    all-zero row hands ``nisa.max8`` and ``nisa.nc_find_index8`` an exact 8-way
    tie, and ``nc_find_index8`` documents "the first occurrence of each value".
    Repeating a real row keeps the padded region inside the value distribution
    the measured rows come from.

    Always contiguous, including on the no-pad path: the kernel reads this buffer
    from HBM, and a caller's non-contiguous view is not the same bytes.
    """
    t_real = x.shape[-2]
    if t_pad == t_real:
        return x.contiguous()
    last = x.narrow(-2, t_real - 1, 1)
    reps = [1] * x.dim()
    reps[-2] = t_pad - t_real
    return torch.cat([x, last.repeat(*reps)], dim=-2).contiguous()


def _noaux_tc_shard_range(num_tokens: int, n_prgs: int, prg_id: int):
    """This core's token range, by the substrate's own formula. Plain python.

    Returns ``(t_offset, t_local)``: the first global token row this core owns
    and how many it owns. Transliterated line for line from the vendor router's
    own split (``router_topk.py:221-224``)::

        T_first_shard = T // n_prgs
        T_second_shard = T - T_first_shard
        T_local = T_first_shard if prg_id == 0 else T_second_shard
        T_offset = 0 if prg_id == 0 else T_first_shard

    The remainder goes to the SECOND core, and that detail is copied rather than
    simplified. An even split written as ``T // n_prgs`` for both cores agrees
    with the vendor on every extent this stage can receive, because the fused
    entry pads the extent to a multiple of 256 before it launches -- but it would
    silently drop the tail on any other extent, and a private helper that
    disagrees with the producer on an unreachable input is the same defect
    ``M-B20-1`` is about, one input away. The vendor's split is only defined for
    two programs, which is all
    ``get_verified_program_sharding_info(..., (0, 1), 2)`` admits.

    Factored OUT of the kernel body on purpose: the arithmetic that decides
    which core writes which rows is the whole content of finding ``M-B20-1``,
    and a plain function can be read exactly by a test as well as by the kernel.
    ``n_prgs == 1`` returns the whole range at offset 0, which is the unsharded
    case and the only case the standalone entry point ever sees.

    ``t_local`` is always a whole number of ``NOAUX_TC_TILE`` rows here, and that
    is not an assumption -- but since `inc-glm53f-088` the guarantee comes from
    the CALLER rather than from a refusal. Each public entry point pads
    ``num_tokens`` up to its own multiple before it launches
    (``_noaux_tc_pad_target``): 128 for the unsharded correct-only entry, 256 for
    the fused ``[2]`` entry so that ``T // 2`` is still a whole tile. Whoever
    calls this with an unpadded extent gets a floor-divided loop that covers less
    than ``t_local``, which is why the pad and the launch grid are chosen together
    at each entry and never separately.
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
    """The AUTHORED `noaux_tc` numerics, in NKI. A plain subkernel, not a jit.

    Factored as a plain function -- the same shape nkilib uses for
    `_rmsnorm_tkg_dloc` and `_router_topk` -- so both jit entry points below
    inline ONE implementation. Two copies would be two things to keep true.

    Reads `[T, E]` logits and a `[1, E]` correction bias from HBM; writes the
    `[T, K]` selected indices and the `[T, E]` scattered gate weights. Stage by
    stage, against `modeling_glm5_next.py`:

        :161  scores            = sigmoid(logits)
        :162  scores_for_choice = scores + e_score_correction_bias
        :177  topk_indices      = topk(scores_for_choice, k)      <- ISA members
        :178  topk_weights      = scores.gather(topk_indices)     <- mask, below
        :179  if norm_topk_prob:
        :180      denominator   = topk_weights.sum() + 1e-20
        :181      topk_weights /= denominator
        :182  topk_weights     *= routed_scaling_factor

    WHY THE MASK RATHER THAN A DMA GATHER. `:178` gathers `scores` at the
    selected indices, and the downstream MoE consumes the SCATTERED `[T, E]`
    form anyway (`rmsnorm_router_topk_tkg.py:71-73`: "expert_affinities: [T, E]
    masked expert affinities, zero outside top-K positions"). So gathering and
    re-scattering would be two DMAs to land where a one-hot mask lands in one
    pass of cheap elementwise work. The mask is built from the INDICES, not from
    value equality against the top-8 values: two experts sharing a corrected
    score would make a value-equality mask select nine columns while the
    reported index set held eight, and the two readings must not be able to
    disagree.

    THIS STAGE IS GRID-AWARE. `shard_on_tokens` says whether the caller split the
    tokens across the logical cores. When it is true, the stage asks the substrate
    which program it is and covers only that program's token rows, which are the
    rows whose logits the producer wrote. When it is false the launch has one
    program and the stage covers the whole extent. The long comment below the
    signature records why this is the producer's own shard and not a new one; it
    is the repair for finding `M-B20-1`.
    """
    bias_sb = nl.load(correction_bias_hbm)  # [1, E]

    # ---- WHICH TOKENS THIS CORE OWNS. ------------------------------------- #
    # Until repair round 1 this loop ran over ALL the tokens on EVERY core,
    # while the producer wrote only its own half. On a `[2]` launch each core
    # then loaded router-logit rows the OTHER core was still writing, with
    # nothing ordering the two, and both cores stored the whole `[T, E]` and
    # `[T, K]` buffers -- so which core's values survived was undefined. Finding
    # `M-B20-1`.
    #
    # The fix is the producer's own shard, not a new one. The vendor router
    # computes its `T_local` and `T_offset` at `router_topk.py:221-224` and then
    # writes the logits, the affinities and the indices only at that offset --
    # each through `_hbm_tiled_store_view(..., T_offset, ...)`, at `:486` for the
    # logits, `:562` for the affinities and `:713` for the indices.
    # This stage asks the SAME question through the SAME call and covers only its
    # own rows, so each core reads exactly the logits it produced and writes
    # exactly the rows it read. No barrier is needed because no cross-core read
    # remains.
    #
    # `shard_on_tokens` is passed in rather than inferred: the vendor shards only
    # when its caller asks it to, so a stage that inferred sharding from the grid
    # alone would shard at `T == 1` where the producer did not. The fused caller
    # passes the SAME variable it hands the vendor.
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

        # :161 -- scores = sigmoid(logits). fp32 throughout: the selection is
        # discrete, so a bf16 score would collide near-neighbour experts and
        # decide the top-8 by round-off rather than by value.
        scores = nl.sigmoid(logits_sb, dtype=nl.float32)

        # :162 -- the bias is a [1, E] row, so its broadcast is on the PARTITION
        # axis, which `nl.broadcast_to` does and `tensor_scalar` does not.
        choice = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        bias_b = nl.broadcast_to(bias_sb, (rows, num_experts))
        nisa.tensor_tensor(dst=choice, data1=scores, data2=bias_b, op=nl.add)

        # :177 -- top-K on the CORRECTED score, through the substrate's own two
        # ISA members (`router_topk.py:598`, `:609`). Not authored here.
        top8 = nl.ndarray((rows, NOAUX_TC_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=top8, src=choice)
        idx8 = nl.ndarray((rows, NOAUX_TC_K), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=idx8, data=choice, vals=top8)

        # :178 -- the one-hot over the selected indices. `router_topk.py:619-622`
        # casts its own index tile to fp32 for exactly this purpose.
        idx_f32 = nl.ndarray((rows, NOAUX_TC_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idx_f32, src=idx8)
        col = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=col, pattern=[[1, num_experts]], offset=0, channel_multiplier=0)

        mask = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=mask, value=0.0)
        hit = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        for k in range(NOAUX_TC_K):
            # `tensor_scalar` broadcasts a [par, 1] operand along the FREE dim.
            nisa.tensor_scalar(
                dst=hit, data=col, op0=nl.equal, operand0=idx_f32[:, k : k + 1]
            )
            nisa.tensor_tensor(dst=mask, data1=mask, data2=hit, op=nl.add)

        # The gather, as a masked scatter: UNBIASED scores, which is the half of
        # `noaux_tc` the substrate cannot express.
        sel = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sel, data1=scores, data2=mask, op=nl.multiply)

        out = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        if norm_topk_prob:
            # :180-182 -- normalise then scale, fused into one pass.
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
            # :182 alone.
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
    """The authored stage alone: `[T, E]` logits in, selection + weights out.

    This entry point exists because it is the honest unit for the increment's
    declared acceptance. The declared comparison is a SET EQUALITY over selected
    expert indices -- discrete, with no tolerance to absorb anything -- so the
    kernel and its oracle must consume BYTE-IDENTICAL logits. Feeding the oracle
    a torch recomputation of the router matmul instead would measure the
    substrate's bf16 matmul precision, not this increment's numerics, and a
    single flipped near-tie would fail the arm for a reason that is not the
    implementation's. The `-025` conditioning carry is the same lesson one level
    up.
    """
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
        # This entry point is launched with NO grid (`noaux_tc_correct` below
        # calls `wrap_nki(...)` without a `[n]`), so there is one program, it owns
        # every token, and there is no other core to race. Stated rather than
        # left to the default, because the stage now has no default.
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
    """RMSNorm + router matmul + `noaux_tc`, all inside ONE dispatch.

    An ADAPT of `rmsnorm_router_topk_tkg._rmsnorm_router_topk_tkg_nki`
    (`:345-451`): the same two substrate members in the same order and with the
    same arguments, with two deliberate differences.

      1. `skip_store_router_logits=False` and a REAL `router_logits` buffer,
         where the substrate passes `None` (`:434`). The logits are what the
         authored stage consumes, and returning them is what lets a caller's
         oracle read the same bytes the kernel read.
      2. The authored `noaux_tc` stage runs after `_router_topk`, in this same
         kernel body, so the correction is INSIDE the dispatch the seam counts.
         A correction applied in torch beside the kernel would leave the
         dispatch count at 1 while changing every number -- which is why the
         seam's fallback counter reading is part of the route predicate and not
         decoration.

    The substrate's OWN (uncorrected) selection is computed and returned too.
    It is not dead work: it is the increment's non-vacuity control (plan D1.5).
    `noaux_tc` differs from the substrate's routing exactly by the correction, so
    an implementation that ignored the bias would return the two selections
    EQUAL, and the test measures how many rows they differ on.

    THE TOKEN SPLIT IS DECIDED ONCE, HERE. `shard_on_tokens` is bound below and
    handed BOTH to the vendor router, which writes the logits only at its own
    token offset, and to the authored stage, which now reads only at that same
    offset. Two separate expressions could drift apart, and that drift is
    precisely finding `M-B20-1`.
    """
    b_extent, s_extent, h_extent = hidden_states.shape
    t_extent = b_extent * s_extent
    _, e_extent = router_weights.shape
    h_free = h_extent // _pmax

    # ONE shard decision for the producer and the authored consumer. It is bound
    # here, handed to the vendor router below, and handed to the authored stage
    # below that -- so the two cannot disagree about whether the tokens were
    # split. Splitting this expression in two is the defect `M-B20-1` reports.
    shard_on_tokens = t_extent > 1

    router_logits = nl.ndarray((t_extent, e_extent), dtype=nl.float32,
                               buffer=nl.shared_hbm)
    norm_output = nl.ndarray((t_extent, h_extent), dtype=router_mm_dtype,
                             buffer=nl.shared_hbm)
    # The substrate's own uncorrected outputs -- the non-vacuity control.
    substrate_index = nl.ndarray((t_extent, NOAUX_TC_K), dtype=nl.int32,
                                 buffer=nl.shared_hbm)
    substrate_affinities = nl.ndarray((t_extent, e_extent), dtype=nl.bfloat16,
                                      buffer=nl.shared_hbm)
    # This increment's corrected outputs.
    expert_index = nl.ndarray((t_extent, NOAUX_TC_K), dtype=nl.uint32,
                              buffer=nl.shared_hbm)
    expert_affinities = nl.ndarray((t_extent, e_extent), dtype=nl.float32,
                                   buffer=nl.shared_hbm)

    norm_sb = nl.ndarray((_pmax, t_extent, h_free), dtype=router_mm_dtype,
                         buffer=nl.sbuf)

    # Substrate stage 1 -- RMSNorm. Same call as `rmsnorm_router_topk_tkg.py:413`.
    _rmsnorm_tkg_dloc(
        input_hbm=hidden_states,
        gamma=gamma,
        output_hbm=norm_output,
        output_sb=norm_sb,
        eps=eps,
        hidden_actual=None,
        sync_output=True,
    )

    # Substrate stage 2 -- router matmul + its own top-K. Same call as
    # `rmsnorm_router_topk_tkg.py:430-449`, with `router_logits` stored.
    # `w_bias=None`: the `noaux_tc` correction bias is NOT a router projection
    # bias and must not be added to the logits here. It enters the SELECTION
    # score after the sigmoid, in the authored stage below.
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

    # AUTHORED stage 3 -- the `noaux_tc` split, same dispatch.
    _noaux_tc_stage(
        router_logits_hbm=router_logits,
        correction_bias_hbm=correction_bias,
        expert_index_hbm=expert_index,
        expert_affinities_hbm=expert_affinities,
        num_tokens=t_extent,
        num_experts=e_extent,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        # The SAME variable the vendor router was handed above. Each core now
        # corrects only the tokens whose logits it produced.
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
        router_logits: `[T, E]` raw router logits, any `T >= 1`. The token axis
            is padded up to this entry's own tile multiple before the launch and
            the outputs are sliced back, so `T` carries no divisibility rule.
        correction_bias: `[1, E]` or `[E]` `e_score_correction_bias`.
        top_k: must be `NOAUX_TC_K`; present so a caller's intent is explicit
            and a mismatch is a named refusal rather than a silent reshape.
        norm_topk_prob: L1-normalise the selected weights (`glm5_next/config.py:191`).
        routed_scaling_factor: final multiplier (`glm5_next/config.py:192`).

    Returns:
        `(expert_index [T, K] int32, expert_affinities [T, E] float32)`.
        `expert_affinities` is the SCATTERED form: the gate weight at each
        selected expert's column and zero elsewhere.

    Raises:
        NoauxTcRouterError: on any extent the reused ISA members cannot serve.
    """
    if router_logits.dim() != 2:
        raise NoauxTcRouterError(
            f"router_logits must be 2D [T, E], got shape {tuple(router_logits.shape)}"
        )
    num_tokens, num_experts = router_logits.shape
    bias = _legalize_correction_bias(correction_bias, num_experts)

    if not can_run_noaux_tc_router(router_logits, num_experts, top_k):
        _NOAUX_TC_COUNTERS.torch_fallback += 1
        return noaux_tc_correct_torch_oracle(
            router_logits, bias, norm_topk_prob, routed_scaling_factor
        )

    # THIS ENTRY'S OWN PAD TARGET, read from the constant that imposes it. The
    # launch below has no grid, so one program owns every token and the stage's
    # loop consumes whole `NOAUX_TC_TILE` rows. 128, not the fused entry's 256.
    t_pad = _noaux_tc_pad_target(num_tokens, NOAUX_TC_TILE)

    _NOAUX_TC_COUNTERS.nki_dispatch += 1
    index, affinities = wrap_nki(_noaux_tc_correct_nki)(
        router_logits=_noaux_tc_pad_tokens(router_logits.to(torch.float32), t_pad),
        correction_bias=bias,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=float(routed_scaling_factor),
    )
    # SLICE BACK to the caller's extent. The pad rows were computed and are
    # discarded; nothing in the stage reduces across tokens, so they cannot have
    # reached these rows.
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
    """Fused RMSNorm + router + `noaux_tc` top-8, in ONE dispatch.

    This is the form the model's MoE block calls. Args mirror
    `rmsnorm_router_topk_tkg` so the two read as one family, with
    `correction_bias` replacing that function's `router_bias`: they are
    different tensors with different jobs, and conflating them is the defect
    this signature makes impossible. `router_bias` is a projection bias added to
    the LOGITS; `correction_bias` is `e_score_correction_bias`, added to the
    SIGMOID SCORES for selection only.

    Returns:
        `(router_logits [T, E], expert_index [T, K] int32,
          expert_affinities [T, E] float32, substrate_index [T, K] int32)`.
        `router_logits` is returned so a caller's oracle can consume the same
        bytes the kernel consumed. `substrate_index` is the substrate's own
        UNCORRECTED selection, carried for the non-vacuity control.

    Raises:
        AssertionError: from the substrate's `_validate_inputs`, called here
            unconditionally and BEFORE the admission branch, exactly as the
            substrate calls it (`rmsnorm_router_topk_tkg.py:79`, before the
            `:94` branch). So a wrong `H` raises on the NKI route and on the
            torch route alike -- there is no `H` value that produces a silent
            fall-through, which is why `H` carries no acceptance value here.
        NoauxTcRouterError: on any extent the reused ISA members cannot serve.
    """
    # The substrate's own validation, unconditionally and first. `gamma` is
    # legalised to [1, H] the way the substrate does (`:91-92`).
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

    # THIS ENTRY'S OWN PAD TARGET, and WHERE it is applied is as load-bearing as
    # the value. It is 256, read from `_NOAUX_TC_T_MULTIPLE` rather than copied
    # from the correct-only entry above: the launch below is `[2]`, the stage
    # binds `shard_on_tokens = t_extent > 1` and splits the tokens, so each core
    # must receive a whole 128-row tile and the full extent must be twice that.
    #
    # It runs AFTER `_substrate_validate_inputs` above, so that validation still
    # sees the caller's real extents and a wrong `H` raises on the shape the
    # caller passed. It runs BEFORE the substrate's admission gate below, which
    # reads `hidden_states.shape` and returns False on a token extent that is not
    # a multiple of 256 (`rmsnorm_router_topk_tkg.py:209`) -- padding after that
    # gate would leave the seam falling back to torch on exactly the extents this
    # increment exists to serve.
    #
    # The reshape to `[1, T, H]` flattens `B` and `S` into the single token axis
    # the kernel already computes (`t_extent = b_extent * s_extent`), so the pad
    # rows land after ALL the real tokens and one slice recovers them. Padding
    # `S` per batch instead would interleave pad rows between batches.
    t_pad = _noaux_tc_pad_target(num_tokens, _NOAUX_TC_T_MULTIPLE)
    hidden_padded = _noaux_tc_pad_tokens(
        hidden_states.reshape(1, num_tokens, h_extent), t_pad
    )

    # BOTH gates, and the order matters. The substrate's own admission gate is
    # consulted so this seam cannot admit a case the substrate would refuse;
    # this seam's gate then RAISES on the same extents rather than returning
    # False, so no admitted-here / refused-there gap can exist. Both read the
    # PADDED tensor, so neither can disagree with the other about the extent the
    # kernel will actually receive.
    substrate_admits = _substrate_can_use_kernel(
        hidden_padded, router_weights, router_mm_dtype, quantization_type
    )
    seam_admits = can_run_noaux_tc_router(hidden_padded, num_experts, top_k)
    if not (substrate_admits and seam_admits):
        _NOAUX_TC_COUNTERS.torch_fallback += 1
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

    _NOAUX_TC_COUNTERS.nki_dispatch += 1
    # `[2]` is the SPMD launch grid, not an output arity: the reused subkernels
    # shard over the two logical cores, and the substrate enters its own kernel
    # the same way (`rmsnorm_router_topk_tkg.py:97-98`, docstring `:378`
    # "Requires LNC=2 sharding").
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
    # SLICE ALL FOUR back to the caller's extent. Every output the kernel
    # allocates is `[t_extent, ...]` with `t_extent = b_extent * s_extent` of the
    # tensor it received, so all four are `[t_pad, ...]` and all four are sliced
    # -- including `substrate_index`, the non-vacuity control, which would
    # otherwise be compared row-for-row against a shorter corrected selection.
    return (
        logits[:num_tokens],
        index[:num_tokens].to(torch.int32),
        affinities[:num_tokens],
        substrate_index[:num_tokens].to(torch.int32),
    )



def _legalize_correction_bias(correction_bias: Tensor, num_experts: int) -> Tensor:
    """Accept `[E]` or `[1, E]`, return a contiguous fp32 `[1, E]`.

    fp32 rather than the model dtype: the bias decides a DISCRETE selection, and
    a bf16 bias would quantise the correction to ~3 decimal digits and merge
    experts the checkpoint separated.
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
    """Torch reference for `noaux_tc`. THE CPU ORACLE, NEVER THE SHIPPED PATH.

    Plan D6: "where a `functional/` module also carries a torch path, that path
    is the CPU oracle and the constraint-violation fallback, never the shipped
    kernel-class implementation." This one is reached only when
    `can_run_kernel()` is False, and every entry into it increments
    `torch_fallback`, which every declared case asserts is 0.

    Transliterated from `transformers` 5.16.1
    `models/glm5_next/modeling_glm5_next.py::Glm5NextTextTopkRouter.forward`
    (:161-182), with the group-routing stage (:163-176) omitted because
    `n_group == 1` makes it an identity -- read from the campaign's own pinned
    `fixtures/config.json`, and measured against the verbatim upstream function
    in `test_router.py` rather than asserted here.
    """
    logits = router_logits.to(torch.float32)
    bias = correction_bias.to(torch.float32).reshape(-1)
    scores = logits.sigmoid()  # :161
    scores_for_choice = scores + bias  # :162
    topk_indices = torch.topk(
        scores_for_choice, k=NOAUX_TC_K, dim=-1, sorted=False
    )[1]  # :177
    topk_weights = scores.gather(1, topk_indices)  # :178
    if norm_topk_prob:  # :179
        denominator = topk_weights.sum(dim=-1, keepdim=True) + NOAUX_TC_DENOM_EPS
        topk_weights = topk_weights / denominator  # :180-181
    topk_weights = topk_weights * routed_scaling_factor  # :182

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
    """Torch reference for the fused form. THE CPU ORACLE, NEVER THE SHIPPED PATH.

    The RMSNorm and matmul halves follow
    `rmsnorm_router_topk_tkg._torch_impl` (`:213-262`) rather than being
    re-derived, including its recorded reason for casting the matmul operands to
    `router_mm_dtype` and accumulating in fp32: "matching the tensor engine
    (bf16 inputs, fp32 accumulation). A true bf16-accumulation matmul over H
    loses far more precision and flips near-tie expert selections" (`:252-256`).
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
    # The substrate's UNCORRECTED selection, for the same non-vacuity control the
    # kernel path returns: top-K on the RAW logits (`rmsnorm_router_topk_tkg.py:263`).
    substrate_index = torch.topk(logits, k=NOAUX_TC_K, dim=-1)[1].to(torch.int32)
    return logits, index, affinities, substrate_index
