# SPDX-License-Identifier: Apache-2.0
"""Per-graph kernel digests in the Neuron compile cache keys.

``vllm_neuron.compile_cache_key`` folds into a compiled graph's key the digest
of only the kernel source files that graph can reach: the module of each NKI
kernel the graph calls, plus the transitive closure of that module's imports
inside the digested kernel sources. An edit to kernel X then recompiles only
the graphs that call X, and a reference the resolver cannot map folds the
whole-package digest instead (never a stale hit).

The graphs here are real ``torch.fx`` graphs whose NKI kernel nodes name
kernels registered in the library's real kernel registry; the keys come from
the library's real key functions with the per-graph fold installed.
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import vllm_neuron
from vllm_neuron import compile_cache_key as ck

REPO_ROOT = Path(vllm_neuron.__file__).resolve().parent.parent
PACKAGE_ROOT = Path(vllm_neuron.__file__).resolve().parent
KNOB = "VLLM_NEURON_DISABLE_KERNEL_DIGEST_KEY"
LOGGER = "vllm_neuron.compile_cache_key"

# Kernel modules of the served decode graphs (stack-1b cache root, graph.hlo).
SINKHORN = "vllm_neuron.functional.mhc.sinkhorn"
ROUTER = "vllm_neuron.functional.moe.router_decode"
DECODE_BATCH = "vllm_neuron.functional.dsa.decode_batch"
CAUSAL_BOUND = "vllm_neuron.functional.dsa.causal_bound"


def _rel(module: str) -> str:
    return module[len("vllm_neuron.") :].replace(".", "/") + ".py"


# ------------------------------------------------------------------ helpers


@pytest.fixture
def lib():
    """The two library modules whose key functions the digest wraps."""
    from libtorch_neuronx_lite.compile import cache
    from libtorch_neuronx_lite.nki import nki_cache

    return cache, nki_cache


@pytest.fixture
def restore_keys(lib, monkeypatch):
    """Put back whatever key functions the process had before the test."""
    cache, nki_cache = lib
    monkeypatch.setattr(cache, "create_cache_hash", cache.create_cache_hash)
    monkeypatch.setattr(
        nki_cache, "create_nki_cache_key", nki_cache.create_nki_cache_key
    )
    monkeypatch.delenv(KNOB, raising=False)


@pytest.fixture
def no_kernel_compile(lib, monkeypatch):
    """Skip the library's NKI compile inside the graph key.

    ``create_cache_hash`` compiles every NKI kernel node to write its backend
    config into the hashed graph text. The kernel nodes here take a symbolic
    argument list that is not a real launch, so that one step is replaced; the
    rest of the library key (graph text, inputs, versions, flags) runs as is.
    """
    cache, _ = lib
    monkeypatch.setattr(cache, "_add_kernel_backend_config_for_hashing", lambda gm: gm)


def _copy_tree(dst_parent: Path) -> Path:
    """Copy every file the digest covers into ``dst_parent/vllm_neuron``."""
    dst = dst_parent / "vllm_neuron"
    for rel in ck.digest_files(PACKAGE_ROOT):
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PACKAGE_ROOT / rel, target)
    return dst


def _write_tree(dst_parent: Path, files: dict[str, str]) -> Path:
    """Write a synthetic ``vllm_neuron`` package tree; return its root."""
    root = dst_parent / "vllm_neuron"
    for rel, source in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


def _append_byte(path: Path) -> None:
    path.write_bytes(path.read_bytes() + b"\n")


def _kernel_entry(module: str, name: str):
    """The kernel object the package hands to ``wrap_nki`` (an ``nki.jit`` kernel)."""
    import importlib

    return getattr(importlib.import_module(module), name)


def _raw_kernel(module: str, name: str):
    """The raw function the library registers and keys for ``module.name``."""
    obj = _kernel_entry(module, name)
    return getattr(obj, "func", obj)


def _stand_in_kernel(module: str, name: str = "kernel"):
    """A plain function that reports ``module`` as its defining module."""

    def kernel(x_hbm):
        return x_hbm

    kernel.__module__ = module
    kernel.__qualname__ = name
    kernel.__name__ = name
    return kernel


def _register(kernel) -> int:
    """Register ``kernel`` the way the package does, so the shared registry
    entry (argument names, defaults) is the one a later trace reuses."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(kernel).kernel_idx


def _kernel_graph(*kernels, extra_args=()) -> torch.fx.GraphModule:
    """A graph that calls each kernel once through the library's NKI HOP.

    ``extra_args`` are non-tensor objects each kernel also takes; the graph
    holds them as attributes it fetches, as Dynamo does for an object it cannot
    rebuild from constants.
    """
    from libtorch_neuronx_lite.nki.nki_hop import nki_kernel_wrapper

    root = torch.nn.Module()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    fetched = []
    for i, obj in enumerate(extra_args):
        setattr(root, f"const_{i}", obj)
        fetched.append(graph.get_attr(f"const_{i}"))
    extra_args = tuple(fetched)
    outs = []
    for kernel in kernels:
        outs.append(
            graph.call_function(
                nki_kernel_wrapper,
                (),
                {
                    "kernel_idx": _register(kernel),
                    "grid": [],
                    "backend_config": "",
                    "operand_output_aliases": {},
                    "args": (x, *extra_args),
                    "arg_names": ["x_hbm"] + [f"c{i}" for i in range(len(extra_args))],
                    "constant_args_key": -1,
                },
            )
        )
    graph.output(tuple(outs) if outs else (x,))
    return torch.fx.GraphModule(root, graph)


def _call_graph(*targets) -> torch.fx.GraphModule:
    """A graph that calls each target once with the input."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    outs = [graph.call_function(t, (x,)) for t in targets]
    graph.output(tuple(outs) if outs else (x,))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


_INPUTS = [torch.empty(4, 8, device="meta")]


def _graph_keys(lib, root: Path, *graphs) -> list[str]:
    """Install per-graph keys for the tree at ``root``; key each graph."""
    cache, _ = lib
    assert ck.install_per_graph_digest_key(ck.KernelDigestResolver(root)) is not None
    return [cache.create_cache_hash(gm, _INPUTS, {}) for gm in graphs]


def _library_graph_key(lib, gm) -> str:
    cache, _ = lib
    library = getattr(cache.create_cache_hash, ck.LIBRARY_FN_ATTR, cache.create_cache_hash)
    return library(gm, _INPUTS, {})


def _module_files(resolver, module: str) -> frozenset[str]:
    resolution = resolver.resolve([ck.Reference(ck.RefKind.MODULE, module)])
    assert resolution.per_graph, resolution.reasons
    return resolution.files


# ------------------------------------------- (1) one kernel edit, one graph


def test_editing_one_kernel_file_changes_only_the_graph_that_calls_it(
    tmp_path, lib, restore_keys, no_kernel_compile
):
    """Graph A calls the Sinkhorn kernel, graph B the decode router kernel."""
    copy = _copy_tree(tmp_path)
    graph_a = _kernel_graph(_kernel_entry(SINKHORN, "sinkhorn_blocks_kernel"))
    graph_b = _kernel_graph(_kernel_entry(ROUTER, "noaux_router_decode_kernel"))

    before = _graph_keys(lib, copy, graph_a, graph_b)
    _append_byte(copy / _rel(SINKHORN))
    after_a = _graph_keys(lib, copy, graph_a, graph_b)
    _append_byte(copy / _rel(ROUTER))
    after_b = _graph_keys(lib, copy, graph_a, graph_b)

    assert after_a[0] != before[0] and after_a[1] == before[1]
    assert after_b[0] == after_a[0] and after_b[1] != after_a[1]


def test_the_per_graph_key_is_the_library_key_with_the_closure_digest_folded(
    tmp_path, lib, restore_keys, no_kernel_compile
):
    """(5) The folded key keeps the library's 32-hex format."""
    copy = _copy_tree(tmp_path)
    resolver = ck.KernelDigestResolver(copy)
    gm = _kernel_graph(_kernel_entry(SINKHORN, "sinkhorn_blocks_kernel"))
    (key,) = _graph_keys(lib, copy, gm)

    files = _module_files(resolver, SINKHORN)
    assert key == ck.fold_key(_library_graph_key(lib, gm), resolver.digest_of(files))
    assert len(key) == 32
    int(key, 16)
    assert resolver.digest_of(files) != resolver.package_digest


# ---------------------------------------- (2) transitive imports under functional/


def test_editing_a_shared_import_changes_every_graph_reaching_it_and_no_other(
    tmp_path, lib, restore_keys, no_kernel_compile
):
    """decode_batch imports causal_bound; the Sinkhorn kernel does not."""
    copy = _copy_tree(tmp_path)
    graphs = (
        _kernel_graph(_kernel_entry(DECODE_BATCH, "dsa_decode_scores_kernel")),
        _kernel_graph(_kernel_entry(CAUSAL_BOUND, "_causal_sentinel_nki")),
        _kernel_graph(_kernel_entry(SINKHORN, "sinkhorn_blocks_kernel")),
    )
    resolver = ck.KernelDigestResolver(copy)
    assert _rel(CAUSAL_BOUND) in _module_files(resolver, DECODE_BATCH)
    assert _rel(CAUSAL_BOUND) not in _module_files(resolver, SINKHORN)

    before = _graph_keys(lib, copy, *graphs)
    _append_byte(copy / _rel(CAUSAL_BOUND))
    after = _graph_keys(lib, copy, *graphs)

    assert after[0] != before[0]
    assert after[1] != before[1]
    assert after[2] == before[2]


def test_each_digested_file_moves_exactly_the_graphs_that_reach_it(tmp_path):
    """Every file in a graph's closure moves its digest; no file outside does."""
    copy = _copy_tree(tmp_path)
    modules = (DECODE_BATCH, SINKHORN, ROUTER)
    base = ck.KernelDigestResolver(copy)
    closures = {m: _module_files(base, m) for m in modules}
    digests = {m: base.digest_of(closures[m]) for m in modules}
    assert all(closures.values())

    wrong = []
    for rel in base.files:
        path = copy / rel
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        try:
            edited = ck.KernelDigestResolver(copy)
            for m in modules:
                moved = edited.digest_of(_module_files(edited, m)) != digests[m]
                if moved != (rel in closures[m]):
                    wrong.append((rel, m, moved))
        finally:
            path.write_bytes(original)

    assert wrong == []


def test_editing_a_torch_only_file_changes_no_graph_key(
    tmp_path, lib, restore_keys, no_kernel_compile
):
    """A file no kernel reaches (torch code around the kernels) keys nothing."""
    copy = _copy_tree(tmp_path)
    graphs = (
        _kernel_graph(_kernel_entry(DECODE_BATCH, "dsa_decode_scores_kernel")),
        _kernel_graph(_kernel_entry(SINKHORN, "sinkhorn_blocks_kernel")),
        _kernel_graph(),
    )
    resolver = ck.KernelDigestResolver(copy)
    reached = _module_files(resolver, DECODE_BATCH) | _module_files(resolver, SINKHORN)
    torch_only = "functional/sampling.py"
    assert torch_only in resolver.files and torch_only not in reached

    before = _graph_keys(lib, copy, *graphs)
    _append_byte(copy / torch_only)
    after = _graph_keys(lib, copy, *graphs)

    assert after == before


def test_a_graph_without_kernels_folds_the_empty_file_set(
    tmp_path, lib, restore_keys
):
    copy = _copy_tree(tmp_path)
    resolver = ck.KernelDigestResolver(copy)
    gm = _call_graph(torch.relu)
    (key,) = _graph_keys(lib, copy, gm)

    assert resolver.graph_resolution(gm).files == frozenset()
    assert key == ck.fold_key(_library_graph_key(lib, gm), resolver.digest_of(()))


# ------------------------------------------------- import-closure rules


RULES_TREE = {
    "functional/__init__.py": "from .pkg.impl import reexported\nfrom .shared import helper\n",
    "functional/kernel_a.py": (
        "from vllm_neuron.functional.shared import helper\n\n"
        "def kernel_a(x):\n    return helper(x)\n"
    ),
    "functional/kernel_b.py": (
        "from . import leaf_b\nfrom ..utils.neuron_utils import TILE\n\n"
        "def kernel_b(x):\n    return leaf_b.f(x) * TILE\n"
    ),
    "functional/shared.py": "import math\n\ndef helper(x):\n    return math.floor(x)\n",
    "functional/leaf_b.py": "def f(x):\n    return x\n",
    "functional/torch_only.py": "import torch\n\ndef g(x):\n    return torch.relu(x)\n",
    "functional/via_package.py": "from vllm_neuron.functional import helper\n",
    "functional/pkg/__init__.py": "from .impl import reexported\n\nLOCAL = 1\n",
    "functional/pkg/impl.py": "def reexported():\n    return 1\n",
    "functional/via_pkg_local.py": "from vllm_neuron.functional.pkg import LOCAL\n",
    "functional/star.py": "from vllm_neuron.functional.pkg import *\n",
    "functional/inner_import.py": (
        "def kernel(x):\n    from vllm_neuron.functional.leaf_b import f\n    return f(x)\n"
    ),
    "functional/plain_import.py": "import vllm_neuron.functional.leaf_b as lb\n",
    "functional/dynamic.py": (
        "import importlib\n\nmod = importlib.import_module('vllm_neuron.functional.leaf_b')\n"
    ),
    "functional/uses_dynamic.py": "from vllm_neuron.functional import dynamic\n",
    "functional/uses_envs.py": "from vllm_neuron import envs\n",
    "utils/neuron_utils.py": "TILE = 128\n",
    "envs.py": "FLAG = 1\n",
    "model/kernel_outside.py": "def k(x):\n    return x\n",
}

F = "functional/"


@pytest.fixture
def rules(tmp_path):
    return ck.KernelDigestResolver(_write_tree(tmp_path, RULES_TREE))


@pytest.mark.parametrize(
    "module, expected",
    [
        # A kernel module reaches the modules it imports, not its package init.
        ("functional.kernel_a", {"kernel_a.py", "shared.py"}),
        # Relative imports, and an allowlisted module outside functional/.
        ("functional.kernel_b", {"kernel_b.py", "leaf_b.py", "../utils/neuron_utils.py"}),
        # A name re-exported by a package init follows only its own source.
        ("functional.via_package", {"via_package.py", "__init__.py", "shared.py"}),
        # A name the init defines itself takes the whole init and its imports.
        ("functional.via_pkg_local", {"via_pkg_local.py", "pkg/__init__.py", "pkg/impl.py"}),
        ("functional.star", {"star.py", "pkg/__init__.py", "pkg/impl.py"}),
        # Imports inside a function body count.
        ("functional.inner_import", {"inner_import.py", "leaf_b.py"}),
        ("functional.plain_import", {"plain_import.py", "leaf_b.py"}),
        # envs.py is outside the digested sources: not followed (as before).
        ("functional.uses_envs", {"uses_envs.py"}),
        ("functional.torch_only", {"torch_only.py"}),
    ],
)
def test_the_closure_follows_static_imports(rules, module, expected):
    files = _module_files(rules, f"vllm_neuron.{module}")

    expected_rel = {
        os.path.normpath(F + rel).replace(os.sep, "/") for rel in expected
    }
    assert files == expected_rel


def test_a_dynamic_import_in_the_closure_falls_back(rules):
    resolution = rules.resolve(
        [ck.Reference(ck.RefKind.MODULE, "vllm_neuron.functional.uses_dynamic")]
    )

    assert not resolution.per_graph
    assert any("functional/dynamic.py" in r for r in resolution.reasons)
    assert rules.digest(resolution) == rules.package_digest


@pytest.mark.parametrize(
    "module, fragment",
    [
        ("vllm_neuron.model.kernel_outside", "outside the digested kernel sources"),
        ("vllm_neuron.functional.missing", "no source file"),
    ],
)
def test_a_package_kernel_module_the_digest_cannot_cover_falls_back(
    rules, module, fragment
):
    resolution = rules.resolve([ck.Reference(ck.RefKind.MODULE, module)])

    assert not resolution.per_graph
    assert any(module in r and fragment in r for r in resolution.reasons)
    assert rules.digest(resolution) == rules.package_digest


def test_a_kernel_from_another_package_reaches_no_package_file(rules):
    """nkilib kernels: their source is outside vllm_neuron (library-versioned)."""
    resolution = rules.resolve(
        [ck.Reference(ck.RefKind.MODULE, "nkilib.core.attention.attention_cte")]
    )

    assert resolution.per_graph
    assert resolution.files == frozenset()


def test_a_qualified_kernel_name_resolves_to_its_defining_module(rules):
    """HLO backend configs carry ``<module>.<qualname>``; nested names too."""
    for name in (
        "vllm_neuron.functional.kernel_a.kernel_a",
        "vllm_neuron.functional.kernel_a.factory.<locals>.kernel_a",
    ):
        resolution = rules.resolve([ck.Reference(ck.RefKind.QUALIFIED_NAME, name)])
        assert resolution.files == _module_files(rules, "vllm_neuron.functional.kernel_a")


def test_the_file_set_digest_matches_the_package_digest_format(rules):
    """Digest of every file == the whole-package digest (same byte stream)."""
    assert rules.digest_of(rules.files) == rules.package_digest
    assert rules.package_digest == ck.kernel_digest(rules.root)


# --------------------------------------------- references a graph carries


def test_a_kernel_node_resolves_through_the_kernel_registry(tmp_path):
    root = _write_tree(tmp_path, RULES_TREE)
    resolver = ck.KernelDigestResolver(root)
    gm = _kernel_graph(_stand_in_kernel("vllm_neuron.functional.kernel_a"))

    resolution = resolver.graph_resolution(gm)

    assert resolution.files == _module_files(resolver, "vllm_neuron.functional.kernel_a")
    assert resolution.modules == frozenset({"vllm_neuron.functional.kernel_a"})


def test_a_kernel_index_missing_from_the_registry_falls_back(tmp_path):
    from libtorch_neuronx_lite.nki.nki_hop import kernel_registry, nki_kernel_wrapper

    resolver = ck.KernelDigestResolver(_write_tree(tmp_path, RULES_TREE))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    missing = max(kernel_registry.funcs, default=0) + 10_000
    graph.output(
        graph.call_function(
            nki_kernel_wrapper,
            (),
            {
                "kernel_idx": missing,
                "grid": [],
                "backend_config": "",
                "operand_output_aliases": {},
                "args": (x,),
                "arg_names": ["x_hbm"],
                "constant_args_key": -1,
            },
        )
    )
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    resolution = resolver.graph_resolution(gm)

    assert not resolution.per_graph
    assert any(str(missing) in r for r in resolution.reasons)


class _PackageConfig:
    """A compile-time config object of a type a kernel module defines."""

    def __init__(self, n):
        self.n = n

    def __hash__(self):
        return hash(self.n)


def test_a_package_object_passed_to_a_kernel_joins_the_closure(tmp_path, monkeypatch):
    resolver = ck.KernelDigestResolver(_write_tree(tmp_path, RULES_TREE))
    monkeypatch.setattr(_PackageConfig, "__module__", "vllm_neuron.functional.leaf_b")
    gm = _kernel_graph(
        _stand_in_kernel("vllm_neuron.functional.kernel_a"), extra_args=(_PackageConfig(3),)
    )

    files = resolver.graph_resolution(gm).files

    assert files == _module_files(resolver, "vllm_neuron.functional.kernel_a") | {
        "functional/leaf_b.py"
    }


def test_a_package_callable_in_the_graph_joins_the_closure(tmp_path):
    """The served graphs construct rotational-top-k configs in the graph."""
    resolver = ck.KernelDigestResolver(_copy_tree(tmp_path))
    from vllm_neuron.functional.vendored_kernels.rotational_topk.rotational_topk_utils import (
        RotationalTopkConfig,
    )

    resolution = resolver.graph_resolution(_call_graph(RotationalTopkConfig))

    assert resolution.per_graph
    assert "functional/vendored_kernels/rotational_topk/rotational_topk_utils.py" in (
        resolution.files
    )


@torch.library.custom_op("w60test::opaque", mutates_args=())
def _opaque_op(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@torch.library.custom_op("w60test::mapped", mutates_args=())
def _mapped_op(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def test_an_unmapped_custom_op_falls_back_with_one_info_log(
    tmp_path, lib, restore_keys, caplog
):
    """(3) An op absent from CUSTOM_OP_KERNEL_MODULES keys the whole package."""
    copy = _copy_tree(tmp_path)
    resolver = ck.KernelDigestResolver(copy)
    gm = _call_graph(torch.ops.w60test.opaque.default)
    cache, _ = lib
    caplog.set_level(logging.INFO, logger=LOGGER)

    ck.install_per_graph_digest_key(resolver)
    first = cache.create_cache_hash(gm, _INPUTS, {})
    second = cache.create_cache_hash(gm, _INPUTS, {})

    assert first == second == ck.fold_key(_library_graph_key(lib, gm), resolver.package_digest)
    naming = [
        r for r in caplog.records
        if r.name == LOGGER and r.levelno == logging.INFO and "w60test::opaque" in r.getMessage()
    ]
    assert len(naming) == 1
    assert "CUSTOM_OP_KERNEL_MODULES" in naming[0].getMessage()


def test_a_mapped_custom_op_keys_its_kernel_modules(tmp_path, monkeypatch):
    resolver = ck.KernelDigestResolver(_copy_tree(tmp_path))
    monkeypatch.setitem(ck.CUSTOM_OP_KERNEL_MODULES, "w60test::mapped", (SINKHORN,))

    resolution = resolver.graph_resolution(_call_graph(torch.ops.w60test.mapped.default))

    assert resolution.per_graph
    assert resolution.files == _module_files(resolver, SINKHORN)


def test_torch_and_collective_ops_carry_no_kernel_source(tmp_path):
    resolver = ck.KernelDigestResolver(_copy_tree(tmp_path))
    gm = _call_graph(torch.ops.aten.relu.default, torch.ops.aten.add, torch.sigmoid)

    resolution = resolver.graph_resolution(gm)

    assert resolution.per_graph and resolution.files == frozenset()


# --------------------------------------------------- (3) the per-kernel key


def test_the_nki_key_folds_only_its_kernels_closure(tmp_path, lib, restore_keys):
    """An edit to the Sinkhorn module misses only the Sinkhorn NKI entries."""
    _, nki_cache = lib
    copy = _copy_tree(tmp_path)
    sinkhorn = _raw_kernel(SINKHORN, "sinkhorn_blocks_kernel")
    router = _raw_kernel(ROUTER, "noaux_router_decode_kernel")
    args = {"x_hbm": torch.empty(4, 8, device="meta")}

    def keys():
        ck.install_per_graph_digest_key(ck.KernelDigestResolver(copy))
        return [nki_cache.create_nki_cache_key(f, args, (1,)) for f in (sinkhorn, router)]

    before = keys()
    _append_byte(copy / _rel(SINKHORN))
    after = keys()

    assert None not in before
    assert after[0] != before[0] and after[1] == before[1]
    library = getattr(nki_cache.create_nki_cache_key, ck.LIBRARY_FN_ATTR)
    resolver = ck.KernelDigestResolver(copy)
    assert after[0] == ck.fold_key(
        library(sinkhorn, args, (1,)), resolver.digest_of(_module_files(resolver, SINKHORN))
    )


def test_a_package_object_argument_joins_the_kernel_closure(tmp_path, monkeypatch):
    resolver = ck.KernelDigestResolver(_write_tree(tmp_path, RULES_TREE))
    monkeypatch.setattr(_PackageConfig, "__module__", "vllm_neuron.functional.leaf_b")

    resolution = resolver.kernel_resolution(
        _stand_in_kernel("vllm_neuron.functional.kernel_a"), {"cfg": _PackageConfig(2)}
    )

    assert "functional/leaf_b.py" in resolution.files


def test_an_uncacheable_nki_call_stays_uncacheable_per_graph(tmp_path, lib, restore_keys):
    _, nki_cache = lib
    ck.install_per_graph_digest_key(ck.KernelDigestResolver(_copy_tree(tmp_path)))

    kernel = _stand_in_kernel(SINKHORN)
    assert nki_cache.create_nki_cache_key(kernel, {"x": bytearray(b"no")}, (1,)) is None


# ------------------------------------------------------------ (4) the knob


def test_the_knob_restores_the_library_keys_in_per_graph_mode(
    tmp_path, lib, restore_keys, monkeypatch
):
    cache, nki_cache = lib
    resolver = ck.KernelDigestResolver(_copy_tree(tmp_path))
    ck.install_per_graph_digest_key(resolver)
    library_graph = getattr(cache.create_cache_hash, ck.LIBRARY_FN_ATTR)
    library_nki = getattr(nki_cache.create_nki_cache_key, ck.LIBRARY_FN_ATTR)

    monkeypatch.setenv(KNOB, "1")

    assert ck.install_per_graph_digest_key(resolver) is None
    assert cache.create_cache_hash is library_graph
    assert nki_cache.create_nki_cache_key is library_nki


def _subprocess_report(extra_env: dict[str, str]) -> dict:
    code = (
        "import json\n"
        "import vllm_neuron\n"
        "from vllm_neuron import compile_cache_key as ck\n"
        "from libtorch_neuronx_lite.compile import cache\n"
        "from libtorch_neuronx_lite.nki import nki_cache\n"
        "def mode(fn):\n"
        "    r = getattr(fn, ck.RESOLVER_ATTR, None)\n"
        "    return None if r is None else str(r.root)\n"
        "print('REPORT ' + json.dumps({\n"
        "    'file': vllm_neuron.__file__,\n"
        "    'graph': mode(cache.create_cache_hash),\n"
        "    'nki': mode(nki_cache.create_nki_cache_key),\n"
        "}))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), **extra_env}
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("REPORT ")]
    assert len(lines) == 1, out.stdout + out.stderr
    return json.loads(lines[0][len("REPORT ") :])


def test_importing_vllm_neuron_installs_per_graph_keys():
    report = _subprocess_report({})

    assert Path(report["file"]).resolve() == PACKAGE_ROOT / "__init__.py"
    assert report["graph"] == report["nki"] == str(PACKAGE_ROOT)


def test_the_knob_keeps_a_fresh_process_off_per_graph_keys():
    report = _subprocess_report({KNOB: "1"})

    assert report["graph"] is None and report["nki"] is None


# ------------------------------------------------- startup cost and logging


def test_the_default_resolver_is_built_once_per_process():
    assert ck.default_resolver() is ck.default_resolver()
    assert ck.default_resolver().package_digest == ck.kernel_digest()


def test_startup_logs_the_kernel_file_count_once(restore_keys, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)

    ck.install_at_startup()

    lines = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert len(lines) == 1
    assert f"files={len(ck.default_resolver().files)}" in lines[0]
    assert "keys=per-graph" in lines[0]


def test_each_new_graph_logs_its_file_count_and_the_running_totals(
    tmp_path, lib, restore_keys, no_kernel_compile, caplog
):
    cache, _ = lib
    resolver = ck.KernelDigestResolver(_copy_tree(tmp_path))
    caplog.set_level(logging.INFO, logger=LOGGER)
    ck.install_per_graph_digest_key(resolver)
    graph_a = _kernel_graph(_kernel_entry(SINKHORN, "sinkhorn_blocks_kernel"))
    graph_b = _call_graph(torch.ops.w60test.opaque.default)

    for gm in (graph_a, graph_a, graph_b):
        cache.create_cache_hash(gm, _INPUTS, {})

    graph_lines = [
        r.getMessage() for r in caplog.records
        if r.name == LOGGER and "graph key" in r.getMessage()
    ]
    assert len(graph_lines) == 2
    n = len(_module_files(resolver, SINKHORN))
    assert f"files={n}" in graph_lines[0] and "per-graph=1 fallback=0" in graph_lines[0]
    assert "whole-package" in graph_lines[1] and "per-graph=1 fallback=1" in graph_lines[1]


# ----------------------------------------- offline: FX text and HLO files


def _b64_config(func_name: str) -> str:
    return base64.b64encode(json.dumps({"func_name": func_name}).encode()).decode()


def test_fx_text_names_kernels_through_their_backend_config():
    gm = _kernel_graph(_stand_in_kernel(SINKHORN))
    (node,) = [n for n in gm.graph.nodes if n.op == "call_function"]
    node.kwargs = {**node.kwargs, "backend_config": _b64_config(f"{SINKHORN}.sinkhorn_blocks_kernel")}

    scan = ck.fx_text_references(str(gm.graph))

    assert scan.unnamed_kernel_calls == 0
    assert ck.Reference(
        ck.RefKind.QUALIFIED_NAME, f"{SINKHORN}.sinkhorn_blocks_kernel"
    ) in scan.references


def test_fx_text_counts_kernel_calls_it_cannot_name():
    """The persisted fxgraph.txt holds kernel_idx only (backend_config empty)."""
    gm = _kernel_graph(_stand_in_kernel(SINKHORN), _stand_in_kernel(ROUTER))

    scan = ck.fx_text_references(str(gm.graph))

    assert scan.unnamed_kernel_calls == 2


def test_fx_text_reports_custom_ops_and_package_callables():
    from vllm_neuron.functional.vendored_kernels.rotational_topk.rotational_topk_utils import (
        RotationalTopkConfig,
    )

    text = str(_call_graph(torch.ops.w60test.opaque.default, RotationalTopkConfig, torch.relu).graph)
    refs = ck.fx_text_references(text).references

    assert ck.Reference(ck.RefKind.CUSTOM_OP, "w60test::opaque") in refs
    assert ck.Reference(
        ck.RefKind.QUALIFIED_NAME,
        "vllm_neuron.functional.vendored_kernels.rotational_topk.rotational_topk_utils"
        ".RotationalTopkConfig",
    ) in refs


def _hlo_module(*custom_calls: tuple[str, str]) -> bytes:
    from libtorch_neuronx_lite.pyhlo import hlo_pb2

    module = hlo_pb2.HloModuleProto(name="m")
    comp = module.computations.add(name="main")
    for i, (target, config) in enumerate(custom_calls):
        ins = comp.instructions.add(name=f"cc.{i}", opcode="custom-call", id=i)
        ins.custom_call_target = target
        ins.backend_config = config.encode()
    return module.SerializeToString()


def test_hlo_custom_calls_name_their_kernels():
    data = _hlo_module(
        ("AwsNeuronCustomNativeKernel", _b64_config(f"{SINKHORN}.sinkhorn_blocks_kernel")),
        ("AwsNeuronArgMax", ""),
    )

    refs = ck.hlo_references(data)

    assert refs == [ck.Reference(ck.RefKind.QUALIFIED_NAME, f"{SINKHORN}.sinkhorn_blocks_kernel")]


def test_an_unknown_hlo_custom_call_or_config_is_unresolved():
    data = _hlo_module(("SomeVendorCall", ""), ("AwsNeuronCustomNativeKernel", "not-base64"))

    refs = ck.hlo_references(data)

    assert [r.kind for r in refs] == [ck.RefKind.UNRESOLVED, ck.RefKind.UNRESOLVED]
    assert "SomeVendorCall" in refs[0].name


def _cache_entry(tmp_path: Path, with_hlo: bool) -> Path:
    entry = tmp_path / "0123456789abcdef0123456789abcdef"
    entry.mkdir()
    gm = _kernel_graph(_stand_in_kernel(SINKHORN))
    (entry / "fxgraph.txt").write_text(str(gm.graph))
    if with_hlo:
        (entry / "graph.hlo").write_bytes(
            _hlo_module(
                ("AwsNeuronCustomNativeKernel", _b64_config(f"{SINKHORN}.sinkhorn_blocks_kernel"))
            )
        )
    return entry


def test_a_cache_entry_names_its_kernels_from_graph_hlo(tmp_path):
    resolver = ck.KernelDigestResolver(_copy_tree(tmp_path / "tree"))
    entry = _cache_entry(tmp_path, with_hlo=True)

    for path in (entry, entry / "fxgraph.txt", entry / "graph.hlo"):
        resolution = resolver.resolve(ck.graph_file_references(path))
        assert resolution.per_graph, (path, resolution.reasons)
        assert resolution.files == _module_files(resolver, SINKHORN)


def test_fx_text_without_graph_hlo_falls_back(tmp_path):
    resolver = ck.KernelDigestResolver(_copy_tree(tmp_path / "tree"))
    entry = _cache_entry(tmp_path, with_hlo=False)

    resolution = resolver.resolve(ck.graph_file_references(entry / "fxgraph.txt"))

    assert not resolution.per_graph
    assert any("graph.hlo" in r for r in resolution.reasons)


def test_the_cli_prints_the_per_graph_file_set(tmp_path, capsys):
    entry = _cache_entry(tmp_path, with_hlo=True)

    assert ck.main(["--graph", str(entry)]) == 0

    out = capsys.readouterr().out
    assert out.startswith("VLLM_NEURON_KERNEL_DIGEST=")
    assert f"GRAPH {entry} per-graph" in out
    for rel in _module_files(ck.default_resolver(), SINKHORN):
        assert f"  {rel}\n" in out
    assert f"modules={SINKHORN}" in out


def test_the_cli_module_entry_point_still_lists_files_and_takes_graph(tmp_path):
    entry = _cache_entry(tmp_path, with_hlo=True)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}

    out = subprocess.run(
        [sys.executable, "-m", "vllm_neuron.compile_cache_key", "--files", "--graph", str(entry)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    ).stdout

    listing, _, graph_part = out.partition(f"GRAPH {entry} ")
    closure = sorted(_module_files(ck.default_resolver(), SINKHORN))
    assert "functional/mhc/sinkhorn.py\n" in listing
    # Run as __main__, the module must still map package kernels to files.
    assert graph_part.startswith(f"per-graph files={len(closure)} ")
    assert [ln.strip() for ln in graph_part.splitlines()[1:]] == closure


# ------------------------------------------------- (6) resolver completeness


def _module_name(path: Path) -> str:
    rel = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_nki_jit(node: ast.AST) -> bool:
    """``nki.jit`` / ``nki.jit(...)`` / bare ``jit`` imported from nki."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr == "jit" and isinstance(node.value, ast.Name) and node.value.id == "nki"
    return False


def _kernel_entry_points(path: Path) -> set[str]:
    """Module-level names in ``path`` that are NKI kernel entry points."""
    tree = ast.parse(path.read_text())
    top_defs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    imported = {
        (a.asname or a.name)
        for n in tree.body
        if isinstance(n, ast.ImportFrom)
        for a in n.names
    }
    assigned = {
        t.id
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    module_level = top_defs | imported | assigned
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(_is_nki_jit(d) for d in node.decorator_list) and node.name in top_defs:
                names.add(node.name)
        elif isinstance(node, ast.Call):
            wrapped = None
            if isinstance(node.func, ast.Name) and node.func.id == "wrap_nki" and node.args:
                wrapped = node.args[0]
            elif isinstance(node.func, ast.Call) and _is_nki_jit(node.func.func) and node.args:
                wrapped = node.args[0]  # nki.jit(...)(kernel)
            elif _is_nki_jit(node) and node.args:
                wrapped = node.args[0]  # nki.jit(kernel)
            if isinstance(wrapped, ast.Name) and wrapped.id in module_level:
                names.add(wrapped.id)
    return names


def test_every_nki_entry_point_under_functional_resolves_to_a_file_set():
    """(6a) Static scan: each kernel the package can register maps per-graph."""
    import importlib

    resolver = ck.default_resolver()
    checked, external, failures = 0, set(), []
    for path in sorted((PACKAGE_ROOT / "functional").rglob("*.py")):
        names = _kernel_entry_points(path)
        if not names:
            continue
        module = importlib.import_module(_module_name(path))
        for name in sorted(names):
            obj = getattr(module, name)
            func = getattr(obj, "func", obj)
            defining = func.__module__
            for ref in (
                ck.Reference(ck.RefKind.MODULE, defining),
                ck.Reference(ck.RefKind.QUALIFIED_NAME, f"{defining}.{func.__qualname__}"),
            ):
                resolution = resolver.resolve([ref])
                if not resolution.per_graph:
                    failures.append((path.name, name, ref, resolution.reasons))
                elif defining.startswith("vllm_neuron."):
                    if _rel(defining) not in resolution.files and (
                        _rel(defining).replace(".py", "/__init__.py") not in resolution.files
                    ):
                        failures.append((path.name, name, ref, "defining file missing"))
                else:
                    external.add(defining.split(".")[0])
            checked += 1

    assert failures == []
    assert checked >= 60  # the scan sees the package's kernels, not nothing
    assert external <= {"nkilib"}


def test_no_nki_entry_point_lives_outside_the_digested_sources():
    """(6b) A kernel under model/ or vllm/ would fall back on every graph."""
    covered = set(ck.default_resolver().files)
    outside = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        if rel in covered or rel == "compile_cache_key.py":
            continue
        if _kernel_entry_points(path):
            outside.append(rel)
    assert outside == []


_OP_REGISTRARS = {"custom_op", "define", "impl", "register_fake", "register_kernel"}


def _registered_ops(source: str) -> set[str]:
    """Op names ``source`` registers through torch.library or vLLM's helper."""
    ops = set()
    libraries = {}  # variable name -> namespace of a torch.library.Library(...)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "Library" and node.value.args:
                ns = node.value.args[0]
                for t in node.targets:
                    if isinstance(t, ast.Name) and isinstance(ns, ast.Constant):
                        libraries[t.id] = ns.value
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        first = node.args[0] if node.args else None
        literal = first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None
        if name == "direct_register_custom_op" and literal:
            ops.add(f"vllm::{literal}")
        elif name in _OP_REGISTRARS and literal:
            owner = func.value.id if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) else None
            if owner in libraries:
                ops.add(f"{libraries[owner]}::{literal.split('(')[0].split('.')[0]}")
            elif "::" in literal:
                ops.add(literal.split("(")[0].split(".")[0])
    return ops


def test_the_op_registration_scan_detects_each_registration_form():
    source = (
        "import torch\n"
        "from vllm.utils import direct_register_custom_op\n"
        "lib = torch.library.Library('nsA', 'FRAGMENT')\n"
        "lib.define('op_a(Tensor x) -> Tensor')\n"
        "@torch.library.custom_op('nsB::op_b', mutates_args=())\n"
        "def op_b(x): ...\n"
        "torch.library.define('nsC::op_c', '(Tensor x) -> Tensor')\n"
        "@torch.library.register_fake('nsD::op_d')\n"
        "def _(x): ...\n"
        "direct_register_custom_op('op_e', op_e)\n"
    )

    assert _registered_ops(source) == {
        "nsA::op_a", "nsB::op_b", "nsC::op_c", "nsD::op_d", "vllm::op_e"
    }


def test_every_custom_op_the_package_registers_is_mapped():
    """(6c) An unmapped op here is a failure, not a fallback."""
    sources = [PACKAGE_ROOT / "model" / "glm5_next" / "model_fp8.py"]
    sources += sorted(PACKAGE_ROOT.rglob("*.py"))
    registered = set()
    for path in sources:
        registered |= _registered_ops(path.read_text())

    unmapped = sorted(op for op in registered if op not in ck.CUSTOM_OP_KERNEL_MODULES)
    assert unmapped == []
    resolver = ck.default_resolver()
    for op, modules in ck.CUSTOM_OP_KERNEL_MODULES.items():
        resolution = resolver.resolve([ck.Reference(ck.RefKind.CUSTOM_OP, op)])
        assert resolution.per_graph, (op, modules, resolution.reasons)


def test_the_library_key_signatures_match_what_the_wrappers_read():
    """The per-graph wrappers read the graph and the kernel by position."""
    import inspect

    from libtorch_neuronx_lite.compile import cache
    from libtorch_neuronx_lite.nki import nki_cache

    graph_fn = getattr(cache.create_cache_hash, ck.LIBRARY_FN_ATTR, cache.create_cache_hash)
    nki_fn = getattr(
        nki_cache.create_nki_cache_key, ck.LIBRARY_FN_ATTR, nki_cache.create_nki_cache_key
    )

    assert list(inspect.signature(graph_fn).parameters)[:2] == ["gm", "example_inputs"]
    assert list(inspect.signature(nki_fn).parameters) == ["func", "args", "grid"]
