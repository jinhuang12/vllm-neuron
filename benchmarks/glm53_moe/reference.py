"""CPU oracle for the three FP8 MoE limbs at PR 4, c0a7e539.

The oracle reads checkpoint scale grids. It does not import production code,
packed scale operands, or NKI. Matmul operands are BF16, block products and
scale accumulation are FP32, and each scale applies to one 128 by 128 block.
"""

from dataclasses import dataclass

import torch

BLOCK = 128
ATOL = 1e-5
RTOL = 3e-2


@dataclass
class Inputs:
    hidden: torch.Tensor             # [T + 1, H], last row is padding
    gate_weight: torch.Tensor        # [E, H, 2, I], FP8
    gate_scales: torch.Tensor        # [E, H/128, 2, I/128], FP32
    down_weight: torch.Tensor        # [E, I, H], FP8
    down_scales: torch.Tensor        # [E, I/128, H/128], FP32
    affinity: torch.Tensor           # [T + 1, E], FP32
    row_index: torch.Tensor          # [P], -1 selects padding
    expert_index: torch.Tensor       # [P/block]
    block: int = 256
    gate_upper: float | None = 10.0
    up_upper: float | None = 10.0


def block_matmul(x, weight, scales):
    """[Q,K] times [K,N]; multiply each FP32 block product by its scale."""
    k, n = weight.shape
    if x.device.type != "cpu" or weight.device.type != "cpu":
        raise ValueError("The reference must run on CPU")
    if k % BLOCK or n % BLOCK or tuple(scales.shape) != (k // BLOCK, n // BLOCK):
        raise ValueError("Expected a 128 by 128 checkpoint scale grid")
    # FP8 values are exactly representable in BF16. The conversion is explicit
    # because BF16 operands at the two matmuls are part of the kernel contract.
    lhs = x.to(torch.bfloat16).float()
    rhs = weight.to(torch.bfloat16).float()
    out = torch.empty((x.shape[0], n), dtype=torch.float32)
    for j in range(n // BLOCK):
        col = slice(j * BLOCK, (j + 1) * BLOCK)
        acc = None
        for i in range(k // BLOCK):
            row = slice(i * BLOCK, (i + 1) * BLOCK)
            scaled = (lhs[:, row] @ rhs[row, col]) * scales[i, j].float()
            acc = scaled if acc is None else acc + scaled
        out[:, col] = acc
    return out


def swiglu(gate_up, gate_upper=None, up_upper=None):
    """FP32 gate-only upper clamp, symmetric up clamp, then SiLU * up."""
    gate, up = gate_up.float().chunk(2, dim=-1)
    if gate_upper is not None:
        gate = gate.clamp(max=gate_upper)
    if up_upper is not None:
        up = up.clamp(min=-up_upper, max=up_upper)
    return (gate * torch.sigmoid(gate)) * up


def compact_reference(hidden, gate_weight, gate_scales, down_weight,
                      down_scales, affinity, gate_upper=10.0, up_upper=10.0):
    """One expert, real rows only. Return all three stage boundaries."""
    h, _, intermediate = gate_weight.shape
    gate_up = block_matmul(
        hidden, gate_weight.reshape(h, 2 * intermediate),
        gate_scales.reshape(h // BLOCK, 2 * intermediate // BLOCK),
    )
    activated = swiglu(gate_up, gate_upper, up_upper)
    down = block_matmul(activated.to(torch.bfloat16), down_weight, down_scales)
    return gate_up, activated.T.contiguous(), down * affinity.float().reshape(-1, 1)


def routed_reference(case):
    """Return padded contribution rows; no scatter or final BF16 rounding."""
    gates, activations, outputs = [], [], []
    for block_id, expert in enumerate(case.expert_index.tolist()):
        rows = case.row_index[block_id * case.block:(block_id + 1) * case.block]
        resolved = torch.where(rows >= 0, rows, case.hidden.shape[0] - 1).long()
        stages = compact_reference(
            case.hidden[resolved], case.gate_weight[expert], case.gate_scales[expert],
            case.down_weight[expert], case.down_scales[expert],
            case.affinity[resolved, expert], case.gate_upper, case.up_upper,
        )
        gates.append(stages[0])
        activations.append(stages[1])
        outputs.append(stages[2])
    return torch.cat(gates), torch.cat(activations, dim=1), torch.cat(outputs)


def routed_down_reference(intermediate_t, case):
    """Down-stage oracle from a supplied FP32 activation boundary."""
    outputs = []
    for block_id, expert in enumerate(case.expert_index.tolist()):
        positions = slice(block_id * case.block, (block_id + 1) * case.block)
        rows = case.row_index[positions]
        resolved = torch.where(rows >= 0, rows, case.hidden.shape[0] - 1).long()
        down = block_matmul(intermediate_t[:, positions].T.to(torch.bfloat16),
                            case.down_weight[expert], case.down_scales[expert])
        outputs.append(down * case.affinity[resolved, expert, None])
    return torch.cat(outputs)


def metrics(actual, expected):
    """Fixed allclose gate plus outlier, norm, and direction diagnostics."""
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    difference = (actual - expected).double()
    a, b = actual.double().flatten(), expected.double().flatten()
    norm_a, norm_b = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if norm_a == 0 or norm_b == 0:
        cosine = 1.0 if norm_a == norm_b else 0.0
    else:
        cosine = float(torch.dot(a, b) / (norm_a * norm_b))
    limit = ATOL + RTOL * expected.abs()
    return {
        "allclose": bool(torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)),
        "atol": ATOL, "rtol": RTOL,
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(difference.abs().max()),
        "l2": float(torch.linalg.vector_norm(difference)),
        "relative_l2": float(torch.linalg.vector_norm(difference) / norm_b) if norm_b else 0.0 if norm_a == 0 else None,
        "cosine": cosine,
        "outside_tolerance": int(((actual - expected).abs() > limit).sum()),
    }


def make_fixture(q=32, hidden=4096, intermediate=512, experts=1,
                 block=256, kind="random", seed=530917):
    """Deterministic signed FP8 bytes, nonuniform scales, and routed tails.

    Each expert receives q real tokens in a different order plus padding. The
    same token can have several expert contributions, as in top-k routing.
    """
    if not 1 <= q <= block or block <= 0 or block % BLOCK:
        raise ValueError("Require 1 <= q <= block, with block a positive multiple of 128")
    if kind not in ("random", "cancellation", "zeros", "clamp", "routing"):
        raise ValueError(f"Unknown fixture: {kind}")
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((q + 1, hidden), generator=generator).to(torch.bfloat16)
    x[-1].zero_()
    # Small finite FP8 values keep the random fixture away from blanket clamp.
    wg = (torch.randn((experts, hidden, 2, intermediate), generator=generator) * 0.25)
    wd = (torch.randn((experts, intermediate, hidden), generator=generator) * 0.25)
    gs_count = experts * (hidden // BLOCK) * 2 * (intermediate // BLOCK)
    ds_count = experts * (intermediate // BLOCK) * (hidden // BLOCK)
    # Both mantissa and exponent vary. The grid is deliberately non-separable.
    gs = (0.015625 + ((torch.arange(gs_count) * 37) % 127).float() / 4096).reshape(experts, hidden // BLOCK, 2, intermediate // BLOCK)
    ds = (0.03125 + ((torch.arange(ds_count) * 53) % 113).float() / 2048).reshape(experts, intermediate // BLOCK, hidden // BLOCK)
    if kind == "zeros":
        x.zero_()
    elif kind == "clamp":
        x[:-1].mul_(16)
        gs.mul_(16)
    elif kind == "cancellation":
        # Paired signs cancel within a contraction block; the small remainder
        # keeps the allclose absolute threshold material near zero.
        x[:-1, 1::2] = x[:-1, 0::2]
        wg[:, 1::2] = -wg[:, 0::2]
        wg[:, 1::2, :, 0] *= 0.9375
    affinity = torch.rand((q + 1, experts), generator=generator, dtype=torch.float32)
    affinity[:-1] *= 0.93751
    affinity[-1].zero_()
    if kind == "routing" and q > 1:
        affinity[0].zero_()
        x[1].zero_()
    positions = []
    for expert in range(experts):
        rows = torch.full((block,), -1, dtype=torch.int32)
        # Scatter real positions through the block to exercise internal holes.
        where = torch.arange(q) if kind != "routing" else torch.linspace(0, block - 1, q).long()
        rows[where] = torch.roll(torch.arange(q, dtype=torch.int32), expert)
        positions.append(rows)
    return Inputs(x, wg.to(torch.float8_e4m3fn), gs, wd.to(torch.float8_e4m3fn), ds,
                  affinity, torch.cat(positions), torch.arange(experts, dtype=torch.int32), block)
