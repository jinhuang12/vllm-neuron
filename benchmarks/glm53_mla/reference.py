"""Independent CPU sparse MLA reference. This file does not import NKI.

Tolerances come from test/vllm_neuron/functional/attention/test_mla_sparse.py
at c0a7e539. They are not adjusted for these measurements.
"""

from dataclasses import dataclass

import torch

ATOL = 1e-5
RTOL = 1e-2


@dataclass
class Inputs:
    q_lift: torch.Tensor
    cache: torch.Tensor
    indices: torch.Tensor
    scale: float
    q_pe: torch.Tensor | None = None
    k_pe: torch.Tensor | None = None


def sparse_reference(case):
    """FP32 attention over the ordered multiset of non-sentinel selected rows.

    Selecting a row twice gives it two softmax terms. The -1 sentinel contributes
    no term. A query with no terms returns exact zeros. Cache values, including
    row zero, do not affect sentinel-only output.
    """
    for value in vars(case).values():
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            raise ValueError("Reference inputs must be on CPU")
    s, h, latent = case.q_lift.shape
    if (case.q_pe is None) != (case.k_pe is None):
        raise ValueError("Both RoPE operands must be present or absent")
    if bool((case.indices < -1).any()) or bool((case.indices >= case.cache.shape[0]).any()):
        raise ValueError("Indices must name a cache row or the -1 sentinel")
    output = torch.zeros((s, h, latent), dtype=torch.float32)
    for row in range(s):
        selected = case.indices[row]
        selected = selected[selected != -1].long()
        if selected.numel() == 0:
            continue
        values = case.cache.float().index_select(0, selected)
        logits = torch.mm(case.q_lift[row].float(), values.T)
        if case.q_pe is not None:
            keys = case.k_pe.float().index_select(0, selected)
            logits += torch.mm(case.q_pe[row].float(), keys.T)
        probabilities = torch.softmax(logits * case.scale, dim=1)
        output[row] = torch.mm(probabilities, values)
    return output


def make_fixture(seq=1, heads=1, latent=512, cache_rows=4096, topk=2176,
                 rope=0, dtype=torch.bfloat16, kind="random", seed=530917,
                 scale=None):
    """Create reproducible input tensors without consulting kernel helpers."""
    if min(seq, heads, latent, cache_rows, topk) < 1 or rope < 0:
        raise ValueError("All extents must be positive, except optional RoPE")
    rng = torch.Generator().manual_seed(seed)
    q = torch.randn((seq, heads, latent), generator=rng).to(dtype)
    cache = torch.randn((cache_rows, latent), generator=rng).to(dtype)
    indices = torch.randint(cache_rows, (seq, topk), generator=rng, dtype=torch.int32)
    if kind == "sentinel":
        indices[:, ::3] = -1
        indices[0] = -1
        if seq > 1:
            indices[1] = -1
            indices[1, topk // 2] = cache_rows - 1
    elif kind == "duplicates":
        indices[:, 1::2] = indices[:, :topk // 2]
    elif kind == "zeros":
        q.zero_()
    elif kind != "random":
        raise ValueError(f"Unknown fixture kind: {kind}")
    q_pe = torch.randn((seq, heads, rope), generator=rng).to(dtype) if rope else None
    k_pe = torch.randn((cache_rows, rope), generator=rng).to(dtype) if rope else None
    return Inputs(q, cache, indices, float(scale if scale is not None else
                  (latent + rope) ** -0.5), q_pe, k_pe)


def metrics(actual, expected):
    """Keep the fixed allclose gate plus magnitude and direction diagnostics."""
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    a, b = actual.double().flatten(), expected.double().flatten()
    difference = a - b
    norm_a, norm_b = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    cosine = float(torch.dot(a, b) / (norm_a * norm_b)) if norm_a and norm_b else float(norm_a == norm_b)
    return {
        "allclose": bool(torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)),
        "atol": ATOL, "rtol": RTOL, "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(difference.abs().max()),
        "l2": float(torch.linalg.vector_norm(difference)),
        "relative_l2": float(torch.linalg.vector_norm(difference) / norm_b) if norm_b else None,
        "cosine": cosine,
        "outside_tolerance": int(((actual - expected).abs() > ATOL + RTOL * expected.abs()).sum()),
    }
