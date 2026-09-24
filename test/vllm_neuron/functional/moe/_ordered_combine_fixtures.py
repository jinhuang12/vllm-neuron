# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures and strict comparisons for ordered-combine regression tests.

The function bodies and case constants retain the reviewed campaign coverage.
These helpers come from PR #9; only their test-package import changed.
"""
from __future__ import annotations

import torch

from test.vllm_neuron.functional.moe import _ordered_combine_reference as baseline


CASES = {
    "prefill": (1024, 4096), "t3-h130": (3, 130), "t127": (127, 128),
    "t128": (128, 384), "t129": (129, 640), "t256": (256, 1024),
    "t257": (257, 1154), "adversarial": (3, 130),
    "decode-fallback": (1, 128), "odd-hidden-fallback": (3, 129),
    "t128-h1024": (128, 1024), "t257-h2048": (257, 2048),
    "t128-top6": (128, 1024), "t384": (384, 1024),
}


NATIVE_CASES = {"prefill", "t256", "t257-h2048", "t128-top6", "t384"}


DENSE_CASES = NATIVE_CASES | {"t128-h1024"}


def compare(actual, expected):
    """Require finite FP32 bits, including signed zero; diagnostics use float64."""
    if actual.shape != expected.shape or actual.dtype != torch.float32 or expected.dtype != torch.float32:
        raise ValueError("Output shape or FP32 dtype changed")
    actual, expected = actual.contiguous(), expected.contiguous()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    bits = int(torch.count_nonzero(actual.view(torch.int32) != expected.view(torch.int32)))
    result = {"finite": finite, "different_bits_elements": bits,
              "bitwise_equal": bits == 0, "pass": finite and bits == 0}
    if finite:
        a, e = actual.double().flatten(), expected.double().flatten()
        difference = a - e
        anorm, enorm = torch.linalg.vector_norm(a), torch.linalg.vector_norm(e)
        cosine = float(torch.dot(a, e) / (anorm * enorm)) if anorm > 0 and enorm > 0 else float(anorm == enorm)
        result.update(max_abs=float(difference.abs().max()), difference_l2=float(torch.linalg.vector_norm(difference)),
                      reference_l2=float(enorm), cosine_similarity=cosine,
                      within_cpu_bounds=bool(torch.allclose(actual, expected, rtol=0, atol=0)))
    return result


def preservation_pass(checks, expected_native):
    """Keep every native baseline bit; changed NKI routes also require CPU bits."""
    return (checks["candidate_vs_baseline"]["pass"]
            and (not expected_native or (checks["baseline_vs_cpu"]["pass"]
                                        and checks["candidate_vs_cpu"]["pass"])))


def prove_map(ids, tokens):
    if ids.dtype != torch.int32 or ids.ndim != 2 or not ids.is_contiguous():
        raise ValueError("Expected contiguous int32 block map")
    counts = []
    for block in ids:
        valid = block[block >= 0]
        if bool((valid >= tokens).any()) or valid.unique().numel() != valid.numel():
            raise ValueError("Fixture violates the generated-ID contract")
        counts.append(valid.numel())
    return {"valid_id_uniqueness_per_block": True, "valid_id_range": [0, tokens],
            "valid_rows_per_block": counts, "negative_ids_are_padding": True}


def concentrated_fixture(template, source_root, seed, routing="first-local"):
    """Build concentrated, rotating or sparse routes with the frozen mapping."""
    tokens, global_experts = template["global_affinities"].shape
    local = template["local_affinities"].shape[1]
    top_k = template["selected_global_experts"].shape[1]
    block_size = template["full_mapping_blocks"].shape[1]
    hidden = template["contribution"].shape[1]
    if top_k > local:
        raise ValueError("Concentrated fixture requires enough local experts")
    selected = torch.arange(top_k).expand(tokens, top_k).clone()
    if routing == "rotating-local":
        selected = (selected + torch.arange(tokens)[:, None]) % local
    elif routing in ("outside-local", "one-local"):
        selected += local
        if routing == "one-local":
            selected[-1, 0] = 0
    elif routing != "first-local":
        raise ValueError("Unknown diagnostic routing")
    affinities = torch.zeros(tokens, global_experts, dtype=torch.float32)
    affinities.scatter_(1, selected, 1.0 / top_k)
    local_affinities = affinities[:, :local].contiguous()
    scope = baseline.cpu_source_functions(source_root / baseline.MAPPING,
                                         ["_build_blockwise_mapping_torch", "_cumsum_matmul"])
    ids, experts, blocks = scope["_build_blockwise_mapping_torch"](
        expert_mask=(local_affinities != 0).float(), num_local_experts=local,
        num_experts_per_token=top_k, block_size=block_size, total_tokens=tokens,
        tp_degree=1, moe_group=None)
    full_ids = ids.int().reshape(blocks, block_size)
    rows = full_ids[:, :min(tokens, block_size)].contiguous()
    if bool((full_ids[:, rows.shape[1]:] >= 0).any()):
        raise ValueError("Compact mapping would discard a valid row")
    pairs = [(token, expert) for row, expert in zip(rows, experts.tolist())
             for token in row[row >= 0].tolist()]
    expected_pairs = {tuple(pair) for pair in (local_affinities != 0).nonzero().tolist()}
    if len(pairs) != len(expected_pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("Concentrated mapping does not emit each selected pair once")
    if set(pairs) != expected_pairs:
        raise ValueError("Concentrated mapping changed the selected expert set")
    fixture = {"contribution": torch.randn(rows.numel(), hidden, generator=torch.Generator().manual_seed(seed)),
               "mapping_blocks": rows, "token_position_to_id": rows.reshape(-1),
               "full_mapping_blocks": full_ids, "block_to_expert": experts.int(),
               "local_affinities": local_affinities, "global_affinities": affinities,
               "selected_global_experts": selected}
    proof = {"route": "exact source CPU mapping", "routing": routing,
             "selected_pairs_emitted_once": True, "valid_rows": len(pairs),
             "active_blocks": int((rows >= 0).any(dim=1).sum()),
             "valid_rows_in_second_flat_half": int((rows.reshape(-1)[rows.numel() // 2:] >= 0).sum()),
             **prove_map(rows, tokens)}
    return fixture, proof


def fixtures(name, source_root, seed):
    tokens, hidden = CASES[name]
    if name != "adversarial":
        shape = dict(tokens=tokens, hidden=hidden, global_experts=288, local_experts=18,
                     top_k=6 if name == "t128-top6" else 8, block_size=256, expert_parallel_rank=0)
        for variant, offset in (("original", 0), ("changed-routes-and-data", 1)):
            fixture, proof = baseline.mapping_fixture(shape, seed + offset, source_root)
            yield variant, fixture, proof
        if name == "prefill":
            fixture, proof = concentrated_fixture(fixture, source_root, seed + 2)
            yield "concentrated-local-routes-and-data", fixture, proof
        if name in DENSE_CASES:
            fixture, proof = concentrated_fixture(fixture, source_root, seed + 3, "rotating-local")
            if proof["valid_rows_in_second_flat_half"] == 0:
                raise ValueError("Admitted rotating fixture does not exercise the second flat half")
            yield "rotating-local-routes-and-data", fixture, proof
            rows = fixture["mapping_blocks"]
            data = torch.zeros_like(fixture["contribution"])
            scale = torch.ones(hidden, dtype=torch.float32)
            scale[1::4], scale[2::4], scale[3::4] = 2**36, 2**-60, -1
            for token in range(tokens):
                positions = (rows.reshape(-1) == token).nonzero().flatten()
                data[positions[0]] = scale * 2**24
                data[positions[-2]] = scale
                data[positions[-1]] = -scale * 2**24
            yield "full-shape-cancellation-and-scales", {**fixture, "contribution": data}, proof
            data = fixture["contribution"].clone()
            padding = rows.reshape(-1) < 0
            data[padding, 0::3], data[padding, 1::3], data[padding, 2::3] = float("nan"), float("inf"), -float("inf")
            yield "full-shape-padding-nonfinite", {**fixture, "contribution": data}, proof
        if name == "prefill":
            for variant, routing in (("full-shape-padding-only", "outside-local"), ("full-shape-one-active-row", "one-local")):
                fixture, proof = concentrated_fixture(fixture, source_root, seed + 4, routing)
                data, rows = fixture["contribution"], fixture["mapping_blocks"]
                data.fill_(1.25)
                padding = rows.reshape(-1) < 0
                data[padding, 0::3], data[padding, 1::3], data[padding, 2::3] = float("nan"), float("inf"), -float("inf")
                yield variant, fixture, proof
        return
    ids = torch.tensor([[0, 1, -1], [0, 1, -1], [0, 1, -1], [-1, -1, -1]], dtype=torch.int32)
    signed = torch.tensor([2**24, -(2**24), 0, 1, -1, 0, -(2**24), 2**24, 0, 0, 0, 0], dtype=torch.float32)
    values = signed[:, None].expand(-1, hidden).contiguous()
    for variant in ("cross-block-cancellation", "changed-routes-and-data", "scale-extremes",
                    "padding-nonfinite", "padding-only", "one-active-row"):
        rows, data = ids.clone(), values.clone()
        if variant == "changed-routes-and-data":
            rows = rows.flip(0).contiguous()
            data = -data
        elif variant == "scale-extremes":
            data[:, 0::2] *= 2**36
            data[:, 1::2] *= 2**-60
        elif variant in ("padding-nonfinite", "padding-only", "one-active-row"):
            if variant != "padding-nonfinite":
                rows.fill_(-1)
            if variant == "one-active-row":
                rows[0, 0] = tokens - 1
                data[0].fill_(1.25)
            padding = rows.reshape(-1) < 0
            data[padding, 0::3] = float("nan")
            data[padding, 1::3] = float("inf")
            data[padding, 2::3] = -float("inf")
        fixture = {"contribution": data, "mapping_blocks": rows}
        yield variant, fixture, {"route": "source-contract adversarial fixture", **prove_map(rows, tokens)}
