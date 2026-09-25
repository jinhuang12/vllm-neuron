# SPDX-License-Identifier: Apache-2.0
"""Source-derived CPU oracles for ordered-combine regression tests.

The function bodies are copied unchanged from the reviewed campaign helpers.
These helpers come from PR #9; only their test-package location changed.
The mapping and scatter expression are extracted from this checkout at runtime.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import math
from pathlib import Path
from typing import Optional

import torch


MODEL = "vllm_neuron/model/glm5_next/model_fp8.py"


MAPPING = "vllm_neuron/functional/moe/moe_blockwise.py"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_tree(path):
    return ast.parse(Path(path).read_text(), filename=str(path))


def cpu_source_functions(path, names, namespace=None):
    """Execute exact CPU function bodies, without the module's device imports."""
    selected = [node for node in source_tree(path).body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Missing CPU source functions in {path}: {names}")
    scope = {"torch": torch, "Tensor": torch.Tensor, "Optional": Optional, "math": math}
    scope.update(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


def scatter_expression(source_root, output):
    """Extract the original FP32 expression; omit only the final model-dtype cast."""
    tree = source_tree(source_root / MODEL)
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        tail = node.body[-3:]
        if len(tail) == 3 and all(isinstance(item, ast.Assign) for item in tail[:2]):
            names = [ast.unparse(item.targets[0]) for item in tail[:2]]
            if names == ["wanted", "accumulated"]:
                matches.append((node, tail))
    if len(matches) != 1:
        raise ValueError("The original post-expert scatter source no longer has one matching tail")
    method, tail = matches[0]
    if ast.unparse(tail[-1]) != "return accumulated[:tokens].to(hidden_states.dtype)":
        raise ValueError("The model's final scatter cast changed")
    if not (isinstance(tail[1].value, ast.Call)
            and isinstance(tail[1].value.func, ast.Attribute)
            and tail[1].value.func.attr == "index_add"):
        raise ValueError("The baseline source no longer calls index_add")

    class DeviceOperand(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if ast.unparse(node) == "hidden_states.device":
                return ast.copy_location(ast.parse("contribution.device", mode="eval").body, node)
            return self.generic_visit(node)

    shell = ast.parse(
        "def baseline_scatter(contribution, token_position_to_id, tokens):\n"
        "    hidden = contribution.shape[1]\n"
    )
    shell.body[0].body += [DeviceOperand().visit(copy.deepcopy(node)) for node in tail[:2]]
    shell.body[0].body += [ast.parse("return accumulated[:tokens]").body[0]]
    ast.fix_missing_locations(shell)
    path = output / "scatter_expression.py"
    path.write_text(ast.unparse(shell) + "\n")
    scope = {"torch": torch}
    exec(compile(path.read_text(), str(path), "exec"), scope)
    return scope["baseline_scatter"], {
        "source": MODEL, "method": method.name,
        "first_line": tail[0].lineno, "last_line": tail[-1].end_lineno,
        "changes": ["Use contribution.device for the same device", "Return FP32 before the model cast"],
        "extracted_sha256": digest(path),
    }


def mapping_fixture(shape, seed, source_root):
    rng = torch.Generator().manual_seed(seed)
    t, e, local, k = (shape[key] for key in ("tokens", "global_experts", "local_experts", "top_k"))
    start = shape["expert_parallel_rank"] * local
    logits = torch.randn(t, e, generator=rng, dtype=torch.float32)
    values, selected = logits.topk(k, dim=1)
    # Give decode a nonzero local signal. This is a synthetic valid route, not a
    # claim about the checkpoint's measured router distribution.
    selected[0] = (torch.arange(k) + start) % e
    affinities = torch.zeros(t, e, dtype=torch.float32)
    affinities.scatter_(1, selected, torch.softmax(values, dim=1))
    local_affinities = affinities[:, start:start + local].contiguous()
    scope = cpu_source_functions(source_root / MAPPING, ["_build_blockwise_mapping_torch", "_cumsum_matmul"])
    token_ids, experts, blocks = scope["_build_blockwise_mapping_torch"](
        expert_mask=(local_affinities != 0).float(), num_local_experts=local,
        num_experts_per_token=k, block_size=shape["block_size"], total_tokens=t,
        tp_degree=1, moe_group=None,
    )
    full_ids = token_ids.to(torch.int32).reshape(blocks, shape["block_size"])
    rows = min(t, shape["block_size"])
    compact = full_ids[:, :rows].contiguous()
    if torch.any(full_ids[:, rows:] != -1):
        raise ValueError("Compact rows would discard a valid token")
    expected_pairs = {(token, expert) for token, expert in (local_affinities != 0).nonzero().tolist()}
    actual_pairs = []
    per_block = []
    for ids, expert in zip(compact, experts.tolist()):
        valid = ids[ids >= 0]
        if torch.any(ids < -1) or torch.any(valid >= t) or len(valid.unique()) != len(valid):
            raise ValueError("Mapping violates its valid-ID range or per-block uniqueness")
        if torch.any(ids[len(valid):] != -1):
            raise ValueError("Valid IDs do not form a block prefix")
        actual_pairs.extend((token, expert) for token in valid.tolist())
        per_block.append(len(valid))
    if len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != expected_pairs:
        raise ValueError("Mapping does not emit each selected local (token, expert) pair exactly once")
    contributions = torch.randn(blocks * rows, shape["hidden"], generator=rng, dtype=torch.float32)
    fixture = {
        "contribution": contributions, "token_position_to_id": compact.reshape(-1),
        "mapping_blocks": compact, "full_mapping_blocks": full_ids,
        "block_to_expert": experts.to(torch.int32), "local_affinities": local_affinities,
        "selected_global_experts": selected, "global_affinities": affinities,
    }
    proof = {
        "route": "exact source CPU mapping functions, tp_degree=1, no collectives",
        "native_public_mapping_crosscheck": "pending",
        "valid_id_uniqueness_per_block": True, "selected_pairs_emitted_once": True,
        "valid_ids_are_prefixes": True, "valid_rows_per_block": per_block,
        "valid_rows": len(actual_pairs), "padding_rows": compact.numel() - len(actual_pairs),
        "blocks": blocks, "rows_per_block": rows,
        "contribution_shape": list(contributions.shape), "target_shape": [t + 1, shape["hidden"]],
        "synthetic_distribution": "Random global top-k; first token selects the local expert prefix",
    }
    return fixture, proof
