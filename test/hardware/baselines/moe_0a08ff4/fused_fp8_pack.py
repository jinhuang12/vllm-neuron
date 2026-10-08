# SPDX-License-Identifier: Apache-2.0
"""Offline, lossless weight packing for the GLM narrow expert kernel."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedExperts:
    """Packed FP8 expert weights and their matching block scales."""

    weights: torch.Tensor
    scales: torch.Tensor


def pack_experts(gate_up, down, gate_up_scales, down_scales):
    """Pack local expert shards without dequantization or rounding.

    Inputs are the model loader's prepared weights and matching compensated
    scales, not raw checkpoint values. Trn2 requires finite stored weights
    within [-240,240]. This function does not rescale them. Layouts: ``gate_up`` has
    shape [E,H,2*I], ``down`` [E,I,H], and scales have shapes
    [E,H/128,2,I/128] and [E,I/128,H/128]. H and I are positive multiples
    of 128. Meta inputs build only shapes; real CPU inputs also check values.
    Run once during weight preparation.
    """
    if any(t.device.type not in ("cpu", "meta") or t.device != gate_up.device
           for t in (gate_up, down, gate_up_scales, down_scales)):
        raise ValueError("Pack prepared CPU weights, or meta shapes, before device transfer")
    if gate_up.ndim != 3:
        raise ValueError("gate_up must have shape [E,H,2*I]")
    experts, hidden, twice_i = gate_up.shape
    intermediate = twice_i // 2
    if experts < 1 or hidden < 1 or intermediate < 1 or hidden % 128 or twice_i % 256:
        raise ValueError("Require E > 0 and positive H/I multiples of 128")
    nh, ni = hidden // 128, intermediate // 128
    shapes = ((gate_up, (experts, hidden, 2 * intermediate)),
              (down, (experts, intermediate, hidden)),
              (gate_up_scales, (experts, nh, 2, ni)), (down_scales, (experts, ni, nh)))
    for tensor, expected in shapes:
        if tuple(tensor.shape) != expected:
            raise ValueError(f"Expected shape {expected}, got {tuple(tensor.shape)}")
    if gate_up.dtype != torch.float8_e4m3fn or down.dtype != torch.float8_e4m3fn:
        raise TypeError("Weights must use torch.float8_e4m3fn")
    if gate_up_scales.dtype != torch.float32 or down_scales.dtype != torch.float32:
        raise TypeError("Block scales must use torch.float32")
    for name, weight in (("gate_up", gate_up), ("down", down)):
        # Exponent 15 holds the E4M3FN values 256..448 and NaN. Trn2
        # interprets those bytes as legacy E4M3 non-finite values.
        if weight.device.type != "meta" and bool(((weight.view(torch.uint8) & 0x78) == 0x78).any()):
            raise ValueError(
                f"{name} contains values outside finite Trn2 E4M3 range [-240,240]; "
                "use the model loader prepared weights and matching compensated scales"
            )
    # uint8 views make the exact byte permutation explicit and avoid unsupported
    # FP8 copy/cat operators on CPU. The resulting tensor retains FP8 values.
    gate = gate_up.view(torch.uint8).reshape(experts, nh, 128, 2 * ni, 128)
    gate = gate.permute(0, 3, 2, 1, 4).contiguous()
    down_tiles = down.view(torch.uint8).reshape(experts, ni, 128, nh, 128)
    weights = torch.cat((gate, down_tiles), dim=1).view(torch.float8_e4m3fn)
    gate_scale = gate_up_scales.reshape(experts, nh, 2 * ni).transpose(1, 2)
    scales = torch.cat((gate_scale, down_scales), dim=1).contiguous()
    return PackedExperts(weights, scales)


def unpack_experts(packed):
    """Undo packing exactly; useful for checkpoint and layout checks."""
    if packed.weights.ndim != 5:
        raise ValueError("Invalid packed weight shape")
    experts, panels, contraction, nh, output = packed.weights.shape
    if panels < 3 or panels % 3 or contraction != 128 or output != 128 or nh < 1:
        raise ValueError("Invalid packed weight geometry")
    ni = panels // 3
    if tuple(packed.scales.shape) != (experts, panels, nh):
        raise ValueError("Invalid packed scale shape")
    raw = packed.weights.view(torch.uint8)
    gate = raw[:, :2 * ni].permute(0, 3, 2, 1, 4).contiguous().reshape(experts, nh * 128, 2 * ni * 128)
    down = raw[:, 2 * ni:].contiguous().reshape(experts, ni * 128, nh * 128)
    gate_scale = packed.scales[:, :2 * ni].transpose(1, 2).contiguous().reshape(experts, nh, 2, ni)
    down_scale = packed.scales[:, 2 * ni:].contiguous()
    return gate.view(torch.float8_e4m3fn), down.view(torch.float8_e4m3fn), gate_scale, down_scale
