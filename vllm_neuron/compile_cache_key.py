# SPDX-License-Identifier: Apache-2.0
"""Per-graph digests of the NKI kernel sources, folded into the Neuron compile cache keys.

``libtorch_neuronx_lite`` keys a compiled graph on the FX graph text, the input
signatures, the tool versions and the compiler flags
(``compile/cache.py:create_cache_hash``), and keys a compiled NKI kernel on that
kernel function's own source (``nki/nki_cache.py:create_nki_cache_key``). Neither
key sees an edit to a helper the kernel calls, a module constant or a tile table,
so a warm ``NEURON_LIBTORCH_CACHE_ROOT`` serves the old kernel binary.

This module snapshots the kernel sources -- every ``*.py`` under
``vllm_neuron/functional`` plus the short allowlist of other modules kernel files
import (:data:`KERNEL_SOURCE_FILES`) -- and folds into each library key the digest
of only the files that key's kernels can reach:

* a graph key folds the files reached from every NKI kernel the graph calls,
  every package callable the graph calls (a kernel config constructor) and every
  custom op it calls (through :data:`CUSTOM_OP_KERNEL_MODULES`);
* an NKI kernel key folds the files reached from that kernel.

A graph with no kernel node, package callable or custom op folds the empty file
set: the key is the graph's dependency set, and such a graph compiles from its
FX text alone, which the library key already holds. No kernel edit can change
what it compiles to, so none recompiles it.

"Reached" is static: the kernel's defining module, then the transitive closure
of its ``import`` / ``from ... import`` statements (read with :mod:`ast`, never
by importing) that stay inside the snapshot, plus every snapshot file that
patches a module the closure uses. A kernel of another package (``nkilib``)
reaches the closures of the snapshot files that import it. An edit to kernel X
therefore recompiles only the graphs that call X, and an edit to a file no
kernel reaches (torch code around the kernels, whose effect is in the FX text)
recompiles no graph for kernel reasons. A reference the resolver cannot map --
a custom op missing from the table, a package module outside the snapshot, a
dynamic import, another package's kernel no snapshot file imports -- folds the
whole-package digest instead and is logged once at INFO, so a fallback costs a
recompile and never serves a stale kernel.

A *patch* is what a snapshot file does to an object it imported: rebinding or
deleting an attribute or item (``mod.X = v``, ``mod.TABLE[k] = v``,
``del mod.X``, ``setattr(mod, ...)``, ``delattr``), also through an alias
(``cfg = mod.cfg; cfg.X = v``), or calling a mutating method (``append``,
``extend``, ``insert``, ``pop``, ``popitem``, ``clear``, ``remove``,
``discard``, ``add``, ``update``, ``setdefault``, ``sort``, ``reverse`` and the
``__setitem__`` family) on an imported object or a module attribute
(``mod.TABLE.update(...)``). Imports and patches under ``if TYPE_CHECKING:`` do
not count. A snapshot file that may patch a module the scan cannot name makes
every key that folds files fall back, wherever the file sits: a dynamic import
(``importlib``, ``importlib.reload``, ``__import__``), a store through
``vars(mod)`` or ``globals()``, a store into ``sys.modules``, or a decorator
rooted in another package outside :data:`NON_REGISTERING_DECORATOR_ROOTS`
(``nki``, ``torch``, ``functools``, ``dataclasses``, ``typing``, ``abc``,
``contextlib``, ``enum``), which may register into that package. Not detected:
a mutation inside a function the file calls (``mod.set_mode(1)``) or of an
object a call returns (``mod.get_table().append(x)``; ``torch.where(...).sort()``
sorts a new tensor and patches nothing), a store through a function parameter
(``def f(m): m.X = v``), ``unittest.mock`` patching, a mutating-method name
called directly on a module bound by ``import`` (``nl.add`` is a kernel op, not
a mutation), and ``exec`` / ``eval``.
``test_compile_cache_key_pergraph.py`` pins the patches the package makes today
and the decorator allowlist, so a new one fails a test until it is reviewed.

The library looks both key functions up through their module at every call
(``cache.create_cache_hash(...)``; ``from .nki_cache import create_nki_cache_key``
inside ``compile_nki``), so replacing the module attribute is the whole hook. The
library offers no key-component hook of its own, and a digest-named cache
sub-root would miss ``NEURON_LIBTORCH_REMOTE_CACHE`` and an explicit
``compiler_workdir``; folding the digest into the key covers both.

Set ``VLLM_NEURON_DISABLE_KERNEL_DIGEST_KEY=1`` to keep the library's own keys.
``python -m vllm_neuron.compile_cache_key [--files] [--graph PATH ...]`` prints
the package digest, the digested files, and the files a cached graph folds.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import enum
import functools
import hashlib
import inspect
import json
import logging
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Optional

logger = logging.getLogger(__name__)

#: The ``vllm_neuron`` package directory the default digest reads.
PACKAGE_ROOT = Path(__file__).resolve().parent

#: The top-level package name; module names under it map to files under a root.
#: Read from ``__package__``: under ``python -m`` this module's ``__name__`` is
#: ``__main__``.
PACKAGE_NAME = (__package__ or __name__).partition(".")[0]

#: Every ``*.py`` under this package-relative directory is kernel source.
KERNEL_SOURCE_DIR = "functional"

#: Package-relative modules outside :data:`KERNEL_SOURCE_DIR` that kernel files
#: import: dispatch predicates, dtype constants and the KV segment table. Over-
#: inclusion costs one recompile; omission serves a stale kernel.
KERNEL_SOURCE_FILES = (
    "parallel/neuron_parallel_state.py",
    "utils/bucket_utils.py",
    "utils/dtype_utils.py",
    "utils/neuron_utils.py",
)

#: Custom ops whose implementation launches NKI kernels: qualified op name
#: (``"namespace::op"``) -> the modules that define those kernels. A
#: ``torch.library`` op is opaque in the FX graph, so its kernels are named here.
#: An op in a namespace outside :data:`TORCH_OP_NAMESPACES` and absent from this
#: table folds the whole-package digest and is logged. The package registers no
#: custom op today; ``test_compile_cache_key_pergraph.py`` fails when one appears
#: without an entry.
CUSTOM_OP_KERNEL_MODULES: dict[str, tuple[str, ...]] = {}

#: ``torch.ops`` namespaces whose ops carry no package kernel source: ATen and
#: prims lower through the library, functional collectives lower to HLO
#: collectives, and higher-order ops keep their bodies in subgraphs the resolver
#: walks (NKI kernel calls are resolved per kernel).
TORCH_OP_NAMESPACES = frozenset(
    {"aten", "prims", "higher_order", "_c10d_functional", "c10d_functional"}
)

#: Files of a compile-cache entry the offline resolver reads.
FX_GRAPH_FILE = "fxgraph.txt"
HLO_GRAPH_FILE = "graph.hlo"

#: FX text target of the library's NKI kernel call (``nki_hop.nki_kernel_wrapper``).
NKI_HOP_TARGET = "torch.ops.higher_order.nki_kernel_wrapper"

#: Modules whose use makes a file's imports invisible to a static scan.
DYNAMIC_IMPORT_MODULES = frozenset({"importlib"})

#: The environment knob that keeps the library's own keys.
KNOB = "VLLM_NEURON_DISABLE_KERNEL_DIGEST_KEY"

#: Attribute on an installed key function holding the whole-package digest: the
#: digest every key folds under :func:`install_kernel_digest_key`, and the
#: fallback digest under :func:`install_per_graph_digest_key`.
DIGEST_ATTR = "vllm_neuron_kernel_digest"

#: Attribute on a per-graph key function holding its :class:`KernelDigestResolver`.
RESOLVER_ATTR = "vllm_neuron_kernel_resolver"

#: Attribute on an installed key function holding the library's own function.
LIBRARY_FN_ATTR = "__wrapped__"


# ------------------------------------------------------------- the snapshot


def digest_files(root: Path | str = PACKAGE_ROOT) -> list[str]:
    """Sorted package-relative POSIX paths of every file the digest covers."""
    root = Path(root)
    files = {
        path.relative_to(root).as_posix()
        for path in (root / KERNEL_SOURCE_DIR).rglob("*.py")
    }
    files.update(rel for rel in KERNEL_SOURCE_FILES if (root / rel).is_file())
    return sorted(files)


def _digest_sources(sources: Mapping[str, bytes], rels: Iterable[str]) -> str:
    """sha256 over ``rels`` in sorted order: path, length, then the bytes."""
    digest = hashlib.sha256()
    for rel in sorted(set(rels)):
        data = sources[rel]
        digest.update(f"{rel}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


def _compute_digest(root: Path) -> str:
    return _digest_sources(
        {rel: (root / rel).read_bytes() for rel in digest_files(root)},
        digest_files(root),
    )


def kernel_digest(root: Path | str | None = None) -> str:
    """The 64-hex sha256 of every kernel source under ``root`` (default: this package).

    Relative paths and contents feed the hash, so a rename or a one-byte edit
    changes it and two checkouts of one commit at different paths share it. The
    default root is read once per process (:func:`default_resolver`).
    """
    if root is None:
        return default_resolver().package_digest
    return _compute_digest(Path(root))


def fold_key(base: str, digest: str) -> str:
    """The library's 32-hex key with the kernel digest folded in, still 32 hex."""
    return hashlib.sha256(f"{base}|kernel_digest:{digest}".encode()).hexdigest()[:32]


# --------------------------------------------------------------- references


class RefKind(enum.Enum):
    """What a :class:`Reference` names."""

    #: A module that defines a kernel, or an object a kernel or graph uses.
    MODULE = "module"
    #: ``<module>.<qualname>`` as an HLO backend config or FX text spells it.
    QUALIFIED_NAME = "qualified_name"
    #: A ``torch.library`` op, ``"namespace::op"``.
    CUSTOM_OP = "custom_op"
    #: A reason the references cannot be mapped to files.
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Reference:
    """One thing a graph or kernel call refers to that may carry kernel source."""

    kind: RefKind
    name: str


@dataclass(frozen=True)
class KernelResolution:
    """The kernel files a set of references reaches.

    ``files`` is None when a reference could not be mapped; the key then folds
    the whole-package digest and ``reasons`` says why.
    """

    files: Optional[frozenset[str]]
    #: Package kernel modules, and the dotted names of other packages' kernels.
    modules: frozenset[str]
    reasons: tuple[str, ...]

    @property
    def per_graph(self) -> bool:
        """Whether the key folds only the reached files."""
        return self.files is not None


def _unresolved(reason: str) -> Reference:
    return Reference(RefKind.UNRESOLVED, reason)


def _in_package(module: str) -> bool:
    return module == PACKAGE_NAME or module.startswith(PACKAGE_NAME + ".")


def _op_references(qualified: str) -> list[Reference]:
    if qualified.partition("::")[0] in TORCH_OP_NAMESPACES:
        return []
    return [Reference(RefKind.CUSTOM_OP, qualified)]


def _unwrapped(obj: Any) -> list[Any]:
    """``obj`` and every callable it wraps (``.func``, ``__wrapped__``)."""
    chain, seen = [], set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        chain.append(obj)
        if not callable(obj):
            break
        inner = getattr(obj, "func", None)  # nki.jit kernels, functools.partial
        obj = inner if callable(inner) else getattr(obj, "__wrapped__", None)
    return chain


def _defining_modules(obj: Any) -> list[str]:
    """The modules of ``obj`` and of every function it wraps.

    A wrapper's own module counts too: a package decorator around a torch
    function runs package code. A non-callable gives its type's module.
    """
    modules = []
    for item in _unwrapped(obj):
        module = getattr(item, "__module__", None) if callable(item) else type(item).__module__
        if isinstance(module, str) and module not in modules:
            modules.append(module)
    return modules


def _kernel_references(func: Any) -> list[Reference]:
    """``<module>.<qualname>`` of an NKI kernel and of every function it wraps.

    The dotted name is the HLO ``func_name`` spelling. It tells a package kernel
    from a namesake in another package (``cumsum``, ``rotational_topk``), and it
    lets another package's kernel be matched to the package files importing it.
    """
    refs = []
    for item in _unwrapped(func):
        module = getattr(item, "__module__", None)
        qualname = getattr(item, "__qualname__", None)
        if isinstance(module, str) and isinstance(qualname, str):
            refs.append(Reference(RefKind.QUALIFIED_NAME, f"{module}.{qualname}"))
    return refs or [_unresolved(f"cannot tell which module defines NKI kernel {func!r}")]


def _object_references(obj: Any) -> list[Reference]:
    """References for a callable or object a graph node or kernel argument holds.

    Only package modules count: another package's callable in the graph is torch
    code, whose effect is in the FX text.
    """
    from torch._ops import HigherOrderOperator, OpOverload, OpOverloadPacket

    if isinstance(obj, OpOverload):
        return _op_references(obj._schema.name)
    if isinstance(obj, OpOverloadPacket):
        return _op_references(obj._qualified_op_name)
    if isinstance(obj, HigherOrderOperator):
        return []
    return [Reference(RefKind.MODULE, m) for m in _defining_modules(obj) if _in_package(m)]


def _value_references(value: Any) -> list[Reference]:
    """References for the non-tensor constants inside node arguments."""
    import torch

    leaves = (
        bool, int, float, complex, str, bytes, type(None), type(Ellipsis),
        torch.fx.Node, torch.Tensor, torch.dtype, torch.device, torch.layout,
        torch.memory_format,
    )
    refs, stack, seen = [], [value], set()
    while stack:
        item = stack.pop()
        if isinstance(item, leaves) or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, (tuple, list, set, frozenset)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, slice):
            stack.extend((item.start, item.stop, item.step))
        else:
            refs.extend(_object_references(item))
    return refs


def _kernel_node_references(node, registry) -> list[Reference]:
    idx = node.kwargs.get("kernel_idx")
    try:
        func = registry.get_func(idx)
    except KeyError:
        return [_unresolved(f"NKI kernel_idx {idx} is not in this process's kernel registry")]
    refs = _kernel_references(func)
    constant_args_key = node.kwargs.get("constant_args_key", -1)
    if constant_args_key != -1:
        try:
            refs.extend(_value_references(registry.get_constant_args(constant_args_key)))
        except KeyError:
            refs.append(_unresolved(f"NKI constant_args_key {constant_args_key} is not registered"))
    return refs


def graph_module_references(gm) -> list[Reference]:
    """Every kernel-carrying reference in ``gm`` and its subgraph modules.

    NKI kernel nodes resolve through the library's kernel registry (the same
    process traced the graph and computes its key); other call targets, called
    submodule classes, fetched attributes and argument constants yield the
    package modules that define them, and custom ops yield their op names.
    """
    import torch
    from libtorch_neuronx_lite.nki.nki_hop import NKIKernelWrapper, kernel_registry

    refs = []
    for owner in gm.modules():
        if not isinstance(owner, torch.fx.GraphModule):
            continue
        for node in owner.graph.nodes:
            if node.op == "call_function" and isinstance(node.target, NKIKernelWrapper):
                refs.extend(_kernel_node_references(node, kernel_registry))
            elif node.op == "call_function":
                refs.extend(_object_references(node.target))
            elif node.op == "call_module":
                sub = owner.get_submodule(node.target)
                if not isinstance(sub, torch.fx.GraphModule):
                    refs.extend(_object_references(type(sub)))
            elif node.op == "get_attr":
                try:
                    value = functools.reduce(getattr, node.target.split("."), owner)
                except AttributeError:
                    refs.append(_unresolved(f"FX get_attr target {node.target} is missing"))
                else:
                    if not isinstance(value, torch.nn.Module):
                        refs.extend(_value_references(value))
            refs.extend(_value_references((node.args, node.kwargs)))
    return refs


def kernel_call_references(func: Callable, args: Mapping[str, Any]) -> list[Reference]:
    """References for one NKI kernel compile: the kernel and its argument objects."""
    refs = _kernel_references(func)
    refs.extend(_value_references(list(args.values())))
    return refs


# ------------------------------------------------- offline: FX text and HLO


class FxScan(NamedTuple):
    """References read from FX graph text."""

    references: list[Reference]
    #: NKI kernel calls whose backend config is empty (the persisted
    #: ``fxgraph.txt`` holds only the process-local ``kernel_idx``).
    unnamed_kernel_calls: int


_FX_CALL = re.compile(r"= (call_function|call_module)\[target=([^\]]+)\]")
_FX_BACKEND_CONFIG = re.compile(r"backend_config: ([A-Za-z0-9+/=]*)")


def _kernel_name_from_config(config: str | bytes) -> Optional[str]:
    """The ``func_name`` of a base64 JSON NKI backend config, or None."""
    try:
        name = json.loads(base64.b64decode(config, validate=True)).get("func_name")
    except (binascii.Error, ValueError, AttributeError):
        return None
    return name if isinstance(name, str) and name else None


def fx_text_references(text: str) -> FxScan:
    """References in ``str(gm.graph)`` text, as the library persists it."""
    refs, unnamed = [], 0
    for line in text.splitlines():
        match = _FX_CALL.search(line)
        if match is None:
            continue
        op, target = match.groups()
        if op == "call_module":
            refs.append(_unresolved(f"FX text calls module {target}; its class is not in the text"))
        elif target == NKI_HOP_TARGET:
            config = _FX_BACKEND_CONFIG.search(line)
            name = _kernel_name_from_config(config.group(1)) if config else None
            if name is None:
                unnamed += 1
            else:
                refs.append(Reference(RefKind.QUALIFIED_NAME, name))
        elif target.startswith("torch.ops."):
            parts = target.split(".")
            if len(parts) >= 4:
                refs.extend(_op_references(f"{parts[2]}::{parts[3]}"))
        elif _in_package(target):
            refs.append(Reference(RefKind.QUALIFIED_NAME, target))
    return FxScan(refs, unnamed)


def hlo_references(data: bytes) -> list[Reference]:
    """References in a serialized ``HloModuleProto`` (a cache entry's ``graph.hlo``).

    Each NKI custom call's backend config names its kernel; a custom call the
    library lowers itself carries no package source; any other target is
    unresolved.
    """
    from libtorch_neuronx_lite.pyhlo import hlo_pb2
    from libtorch_neuronx_lite.xla_impl import custom_call_targets as targets

    nki_target = targets.AwsNeuronNkiKernel
    library_lowerings = {
        value
        for name, value in vars(targets).items()
        if isinstance(value, str) and not name.startswith("_")
    } - {nki_target, targets.AwsNeuronCustomOp}
    module = hlo_pb2.HloModuleProto()
    module.ParseFromString(data)
    refs = []
    for computation in module.computations:
        for ins in computation.instructions:
            if ins.opcode != "custom-call":
                continue
            target = ins.custom_call_target
            if target == nki_target:
                name = _kernel_name_from_config(ins.backend_config)
                refs.append(
                    Reference(RefKind.QUALIFIED_NAME, name)
                    if name
                    else _unresolved(f"HLO NKI custom call {ins.name} has no readable func_name")
                )
            elif target not in library_lowerings:
                refs.append(
                    _unresolved(f"HLO custom-call target {target} ({ins.name}) is not a library lowering")
                )
    return refs


def graph_file_references(path: Path | str) -> list[Reference]:
    """References of a compile-cache entry: its directory, ``fxgraph.txt`` or ``graph.hlo``.

    The FX text gives op names and package callables; the HLO gives kernel
    names (the persisted FX text carries only process-local kernel indices), read
    from the ``graph.hlo`` beside an FX file.
    """
    path = Path(path)
    if path.is_dir():
        fx, hlo = path / FX_GRAPH_FILE, path / HLO_GRAPH_FILE
        if not fx.is_file() and not hlo.is_file():
            raise ValueError(f"{path} holds neither {FX_GRAPH_FILE} nor {HLO_GRAPH_FILE}")
    elif path.read_bytes().startswith(b"graph("):
        fx, hlo = path, path.parent / HLO_GRAPH_FILE
    else:
        fx, hlo = None, path
    refs, unnamed = [], 0
    if fx is not None and fx.is_file():
        scan = fx_text_references(fx.read_text())
        refs.extend(scan.references)
        unnamed = scan.unnamed_kernel_calls
    if hlo.is_file():
        refs.extend(hlo_references(hlo.read_bytes()))
    elif unnamed:
        refs.append(
            _unresolved(
                f"{unnamed} NKI kernel calls in {fx} carry no backend config and no "
                f"{HLO_GRAPH_FILE} sits beside it"
            )
        )
    return refs


# ------------------------------------------------------------ import closure


@dataclass(frozen=True)
class _ImportEntry:
    """One imported name: ``from module import name`` (``name`` None: ``import module``)."""

    module: str
    name: Optional[str]


@dataclass(frozen=True)
class _ModuleImports:
    entries: tuple[_ImportEntry, ...]
    #: Module-level names bound only by import statements -> those imports.
    imported_names: Mapping[str, tuple[_ImportEntry, ...]]
    #: Module-level names bound any other way (def, class, assignment, global).
    defined_names: frozenset[str]
    dynamic: Optional[str]
    #: Dotted names of imported objects whose attributes or items the file
    #: rebinds or deletes (``mod.ATTR = v``, ``mod.TABLE[k] = v``,
    #: ``setattr(mod, ...)``), at any depth.
    patched: tuple[str, ...]
    #: The first mutation whose target the scan cannot name (``vars(mod)[k] = v``,
    #: ``globals()[k] = v``, ``sys.modules[k] = m``, a decorator of another
    #: package outside :data:`NON_REGISTERING_DECORATOR_ROOTS`), if any.
    unknown_patch: Optional[str]


class _Patchers(NamedTuple):
    """Snapshot files that patch modules, by what they patch."""

    #: Patched snapshot file -> the other snapshot files that patch its module.
    by_file: Mapping[str, frozenset[str]]
    #: Top-level name of another package -> the snapshot files that patch it.
    by_package: Mapping[str, frozenset[str]]
    #: Snapshot file -> why it may patch any module (a dynamic import or a
    #: mutation of unknown target). Every key that folds files falls back
    #: while such a file is in the snapshot.
    unknown: Mapping[str, str]


def _package_of(rel: str) -> str:
    """The package a module file's relative imports start from."""
    parts = rel[: -len(".py")].split("/")[:-1]
    return ".".join([PACKAGE_NAME, *parts])


def _absolute_module(node: ast.ImportFrom, package: str) -> Optional[str]:
    if node.level == 0:
        return node.module
    parts = package.split(".")
    if node.level - 1 >= len(parts):
        return None
    base = ".".join(parts[: len(parts) - node.level + 1])
    return f"{base}.{node.module}" if node.module else base


def _module_level_bindings(tree: ast.Module) -> tuple[list[ast.stmt], set[str]]:
    """Module-level import statements, and names bound at module level otherwise.

    Function and class bodies are not module level, except that a ``global``
    statement anywhere rebinds a module name.
    """
    imports, defined, stack = [], set(), list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Lambda):
            continue
        else:
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                defined.add(node.id)
            elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
                defined.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                defined.add(node.rest)
            stack.extend(ast.iter_child_nodes(node))
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            defined.update(node.names)
    return imports, defined


#: Builtins that rebind or delete an attribute of their first argument.
_ATTRIBUTE_SETTERS = frozenset({"setattr", "delattr"})

#: Methods that mutate their receiver in place (list, dict, set, attribute hooks).
_MUTATING_METHODS = frozenset(
    {
        "append", "extend", "insert", "pop", "popitem", "clear", "remove", "discard",
        "add", "update", "setdefault", "sort", "reverse",
        "__setitem__", "__delitem__", "__setattr__", "__delattr__",
    }
)


#: Builtins whose result is a module's namespace: a store through it can
#: rebind any name of any module.
_NAMESPACE_BUILTINS = frozenset({"vars", "globals"})

#: The namespace of every loaded module (``sys.modules[name] = shim``).
_MODULE_TABLE = "sys.modules"

#: Top-level packages whose decorators wrap or describe the function they
#: decorate and register it nowhere another file reads. A decorator from any
#: other package may register into that package's state, a mutation of
#: unknown target.
NON_REGISTERING_DECORATOR_ROOTS = frozenset(
    {"nki", "torch", "functools", "dataclasses", "typing", "abc", "contextlib", "enum"}
)


def _namespace_store(expr: ast.expr) -> Optional[str]:
    """The builtin (``vars`` / ``globals``) whose result ``expr`` goes through."""
    while isinstance(expr, (ast.Attribute, ast.Subscript)):
        expr = expr.value
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id in _NAMESPACE_BUILTINS
    ):
        return expr.func.id
    return None


def _decorator_root(expr: ast.expr) -> Optional[str]:
    """The name a decorator expression starts from (``@a.b(c)`` -> ``a``)."""
    while isinstance(expr, (ast.Call, ast.Attribute, ast.Subscript)):
        expr = expr.func if isinstance(expr, ast.Call) else expr.value
    return expr.id if isinstance(expr, ast.Name) else None


def _mutated_object(expr: ast.expr) -> Optional[tuple[str, tuple[str, ...]]]:
    """``(base name, attribute path)`` of the object ``expr`` evaluates to.

    The path stops at the first item (``a.b[0].c`` -> ``("a", ("b",))``), so it
    names the module-level object the mutation goes through. What a call returns
    names nothing (``None``): ``torch.where(...).sort()`` sorts a new tensor, not
    ``torch.where``; a call that hands out module state is the documented gap of
    a mutation inside a called function.
    """
    attrs: list[str] = []
    while not isinstance(expr, ast.Name):
        if isinstance(expr, ast.Attribute):
            attrs.append(expr.attr)
            expr = expr.value
        elif isinstance(expr, ast.Subscript):
            attrs.clear()
            expr = expr.value
        else:
            return None
    return expr.id, tuple(reversed(attrs))


#: The ``typing`` flag whose ``if`` body never runs.
_TYPE_CHECKING = "TYPE_CHECKING"


def _type_checking_only(tree: ast.Module) -> set[int]:
    """Ids of the nodes inside ``if TYPE_CHECKING:`` bodies (they never run)."""
    skipped = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        name = test.id if isinstance(test, ast.Name) else getattr(test, "attr", None)
        if name == _TYPE_CHECKING:
            for stmt in node.body:
                skipped.update(id(n) for n in ast.walk(stmt))
    return skipped


def _parse_imports(rel: str, source: bytes) -> _ModuleImports:
    tree = ast.parse(source, filename=rel)
    package = _package_of(rel)
    entries_by_node, dynamic = {}, None
    bound_to: dict[str, list[str]] = {}  # name an import binds -> dotted object names
    module_names: set[str] = set()  # names bound by ``import`` (always modules)
    aliases: list[tuple[str, ast.expr]] = []  # ``name = <attribute chain>``
    mutated: list[ast.expr] = []
    method_receivers: list[ast.expr] = []  # ``<receiver>.append(...)`` and the like
    decorators: list[ast.expr] = []
    unknown: list[str] = []  # mutations whose target the scan cannot name
    type_only = _type_checking_only(tree)
    for node in ast.walk(tree):
        if id(node) in type_only:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorators.extend(node.decorator_list)
        if isinstance(node, ast.Import):
            entries_by_node[id(node)] = [
                (alias.asname or alias.name.partition(".")[0], _ImportEntry(alias.name, None))
                for alias in node.names
            ]
            for alias in node.names:
                name = alias.asname or alias.name.partition(".")[0]
                module_names.add(name)
                bound_to.setdefault(name, []).append(
                    alias.name if alias.asname else alias.name.partition(".")[0]
                )
            used = [a.name for a in node.names if a.name.partition(".")[0] in DYNAMIC_IMPORT_MODULES]
        elif isinstance(node, ast.ImportFrom):
            module = _absolute_module(node, package)
            if module is None:
                continue
            entries_by_node[id(node)] = [
                (None if a.name == "*" else (a.asname or a.name), _ImportEntry(module, a.name))
                for a in node.names
            ]
            for alias in node.names:
                if alias.name != "*":
                    bound_to.setdefault(alias.asname or alias.name, []).append(
                        f"{module}.{alias.name}"
                    )
            used = [module] if module.partition(".")[0] in DYNAMIC_IMPORT_MODULES else []
        elif isinstance(node, ast.Name) and node.id == "__import__":
            used = ["__import__"]
        else:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
                node.value, (ast.Name, ast.Attribute)
            ):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                aliases.extend((t.id, node.value) for t in targets if isinstance(t, ast.Name))
            elif isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                mutated.append(node.value)
                if (builtin := _namespace_store(node.value)) is not None:
                    unknown.append(f"line {node.lineno} stores through {builtin}()")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _ATTRIBUTE_SETTERS
                and node.args
            ):
                mutated.append(node.args[0])
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _MUTATING_METHODS
            ):
                method_receivers.append(node.func.value)
                if (builtin := _namespace_store(node.func.value)) is not None:
                    unknown.append(f"line {node.lineno} mutates {builtin}()")
            continue
        if used and dynamic is None:
            dynamic = f"line {node.lineno} uses {used[0]}"
    def objects(expr: ast.expr) -> list[str]:
        """Dotted names of the imported objects ``expr`` can evaluate to."""
        found = _mutated_object(expr)
        if found is None:
            return []
        base, attrs = found
        return [".".join((dotted, *attrs)) for dotted in bound_to.get(base, ())]

    # An alias of an imported object (``cfg = pkg.mod.cfg``) patches it too;
    # each pass resolves one more link of an alias chain.
    for _ in aliases:
        grew = False
        for name, expr in aliases:
            for dotted in objects(expr):
                if dotted not in bound_to.setdefault(name, []):
                    bound_to[name].append(dotted)
                    grew = True
        if not grew:
            break
    patched = [dotted for expr in mutated for dotted in objects(expr)]
    for expr in method_receivers:
        found = _mutated_object(expr)
        # ``mod.add(...)`` on a module bound by ``import`` is a module function
        # call (``nl.add``), not a mutation; ``mod.TABLE.update(...)`` is one.
        if found is not None and not (found[0] in module_names and not found[1]):
            patched.extend(objects(expr))
    # ``sys.modules[name] = shim`` replaces a whole module: its target is a
    # run-time string, so the scan cannot name the module it patches.
    if any(t == _MODULE_TABLE or t.startswith(_MODULE_TABLE + ".") for t in patched):
        unknown.append(f"mutates {_MODULE_TABLE}")
        patched = [
            t for t in patched if not (t == _MODULE_TABLE or t.startswith(_MODULE_TABLE + "."))
        ]
    for expr in decorators:
        root = _decorator_root(expr)
        for dotted in bound_to.get(root, ()) if root is not None else ():
            if not _in_package(dotted) and (
                dotted.partition(".")[0] not in NON_REGISTERING_DECORATOR_ROOTS
            ):
                unknown.append(f"line {expr.lineno} decorates with {dotted} of another package")
    module_imports, defined = _module_level_bindings(tree)
    imported: dict[str, list[_ImportEntry]] = {}
    for node in module_imports:
        for bound, entry in entries_by_node.get(id(node), ()):
            if bound is not None:
                imported.setdefault(bound, []).append(entry)
    return _ModuleImports(
        entries=tuple(e for pairs in entries_by_node.values() for _, e in pairs),
        imported_names={k: tuple(v) for k, v in imported.items()},
        defined_names=frozenset(defined),
        dynamic=dynamic,
        patched=tuple(dict.fromkeys(patched)),
        unknown_patch=unknown[0] if unknown else None,
    )


def _imports_name(entry: _ImportEntry, name: str) -> bool:
    """Whether ``entry`` imports ``name`` (a dotted module or kernel name of
    another package), a module or package that holds it, something from inside
    it, or a parent package's re-export of it."""
    target = entry.module if entry.name in (None, "*") else f"{entry.module}.{entry.name}"
    if name == target or name.startswith(target + ".") or target.startswith(name + "."):
        return True
    if entry.name in (None, "*") or not name.startswith(entry.module + "."):
        return False
    return entry.name in name[len(entry.module) + 1 :].split(".")


class KernelDigestResolver:
    """Static map from kernel references to the kernel source files they reach.

    Snapshots the digested sources under ``root`` once, so every key a process
    computes reads the same bytes. A module reaches its own file and, through
    its ``import`` statements (at any depth, read with :mod:`ast`), the files of
    the modules it imports that lie inside the snapshot. ``from M import name``
    follows only the statements that bind ``name`` in M when M binds it by
    import alone (a package ``__init__`` re-export); a name M defines itself, a
    star import or an unbound name follows all of M. A package's ``__init__`` is
    reached only when imported from, and imports leaving the snapshot (another
    package, ``vllm_neuron.envs``) are not followed, as the whole-package digest
    never covered them.

    A snapshot file that patches a module (rebinds or deletes one of its
    attributes or items) changes what that module's code does wherever it runs,
    so the patcher joins every closure that reaches the patched package module,
    and every closure that imports from the patched other package. A kernel of
    another package (``nkilib``) reaches the closures of the snapshot files that
    import it: they wrap it, choose its arguments and may patch its package.
    """

    def __init__(self, root: Path | str = PACKAGE_ROOT):
        self.root = Path(root)
        #: Every digested file, package-relative and sorted.
        self.files = tuple(digest_files(self.root))
        self._sources = {rel: (self.root / rel).read_bytes() for rel in self.files}
        #: The digest over every file: what a fallback key folds.
        self.package_digest = _digest_sources(self._sources, self.files)
        self._imports: dict[str, _ModuleImports] = {}
        self._closures: dict[str, tuple[frozenset[str], tuple[str, ...]]] = {}
        self._digests: dict[frozenset[str], str] = {}
        self._importers: dict[str, tuple[str, ...]] = {}
        self._patchers: Optional[_Patchers] = None

    def digest_of(self, rels: Iterable[str]) -> str:
        """The digest of a set of snapshot files, in the package digest's format."""
        key = frozenset(rels)
        if key not in self._digests:
            unknown = sorted(key - set(self._sources))
            if unknown:
                raise ValueError(f"not digested kernel sources under {self.root}: {unknown}")
            self._digests[key] = _digest_sources(self._sources, key)
        return self._digests[key]

    def digest(self, resolution: KernelResolution) -> str:
        """What a key folds: the reached files' digest, or the package digest."""
        if resolution.files is None:
            return self.package_digest
        return self.digest_of(resolution.files)

    def graph_resolution(self, gm) -> KernelResolution:
        """The kernel files graph ``gm`` reaches."""
        return self.resolve(graph_module_references(gm))

    def kernel_resolution(self, func: Callable, args: Mapping[str, Any]) -> KernelResolution:
        """The kernel files one NKI kernel compile reaches."""
        return self.resolve(kernel_call_references(func, args))

    def resolve(self, refs: Iterable[Reference]) -> KernelResolution:
        """Map references to files; any unmappable reference makes it a fallback."""
        files, modules, reasons = set(), set(), []
        for ref in refs:
            if ref.kind in (RefKind.MODULE, RefKind.QUALIFIED_NAME) and not _in_package(ref.name):
                self._add_external(ref.name, files, modules, reasons)
            elif ref.kind is RefKind.MODULE:
                self._add_module(ref.name, files, modules, reasons)
            elif ref.kind is RefKind.QUALIFIED_NAME:
                module = self._defining_module(ref.name)
                if module is None:
                    reasons.append(f"kernel {ref.name} has no source file under {self.root}")
                else:
                    self._add_module(module, files, modules, reasons)
            elif ref.kind is RefKind.CUSTOM_OP:
                kernel_modules = CUSTOM_OP_KERNEL_MODULES.get(ref.name)
                if kernel_modules is None:
                    reasons.append(
                        f"custom op {ref.name} has no entry in CUSTOM_OP_KERNEL_MODULES "
                        f"({__name__})"
                    )
                else:
                    for module in kernel_modules:
                        self._add_module(module, files, modules, reasons)
            else:
                reasons.append(ref.name)
        if reasons:
            return KernelResolution(None, frozenset(modules), tuple(dict.fromkeys(reasons)))
        return KernelResolution(frozenset(files), frozenset(modules), ())

    # -- module and file mapping

    def _module_file(self, module: str) -> Optional[str]:
        """The package-relative file defining ``module``, on disk or in the snapshot."""
        parts = module.split(".")[1:]
        base = "/".join(parts)
        candidates = (f"{base}.py", f"{base}/__init__.py") if parts else ("__init__.py",)
        for rel in candidates:
            if rel in self._sources or (self.root / rel).is_file():
                return rel
        return None

    def _defining_module(self, qualified: str) -> Optional[str]:
        """The longest package-module prefix of a package ``<module>.<qualname>``."""
        parts = qualified.split(".")
        for end in range(len(parts), 1, -1):
            module = ".".join(parts[:end])
            if self._module_file(module) is not None:
                return module
        return None

    def _add_external(self, name, files, modules, reasons) -> None:
        """Another package's kernel (or kernel module): the files importing it."""
        modules.add(name)
        importers = self._importers_of(name)
        if not importers:
            reasons.append(
                f"kernel {name} of package {name.partition('.')[0]} is imported by no "
                f"digested file under {self.root}"
            )
        for rel in importers:
            closure, why = self._closure(rel)
            files.update(closure)
            reasons.extend(why)

    def _add_module(self, module, files, modules, reasons) -> None:
        modules.add(module)
        rel = self._module_file(module)
        if rel is None:
            reasons.append(f"module {module} has no source file under {self.root}")
        elif rel not in self._sources:
            reasons.append(f"module {module} ({rel}) is outside the digested kernel sources")
        else:
            closure, why = self._closure(rel)
            files.update(closure)
            reasons.extend(why)

    # -- the import closure

    def _parsed(self, rel: str) -> _ModuleImports:
        if rel not in self._imports:
            self._imports[rel] = _parse_imports(rel, self._sources[rel])
        return self._imports[rel]

    def _targets(self, entry: _ImportEntry) -> list[tuple[str, Optional[str]]]:
        """Snapshot nodes ``(file, name or None for the whole file)`` an import reaches."""
        if not _in_package(entry.module):
            return []
        package = self._module_file(entry.module)
        in_snapshot = package in self._sources
        if entry.name is None or entry.name == "*":
            return [(package, None)] if in_snapshot else []
        targets = []
        sub = self._module_file(f"{entry.module}.{entry.name}")
        if sub is not None:
            if sub in self._sources:
                targets.append((sub, None))
            # A package init may rebind its submodule's name to another object.
            if in_snapshot:
                info = self._parsed(package)
                if entry.name in info.defined_names or entry.name in info.imported_names:
                    targets.append((package, entry.name))
        elif in_snapshot:
            targets.append((package, entry.name))
        return targets

    def _importers_of(self, name: str) -> tuple[str, ...]:
        """Snapshot files with an import of another package's ``name`` (any depth)."""
        if name not in self._importers:
            self._importers[name] = tuple(
                rel
                for rel in self.files
                if any(_imports_name(e, name) for e in self._parsed(rel).entries)
            )
        return self._importers[name]

    def _patch_index(self) -> _Patchers:
        """Which snapshot files patch which modules, over every snapshot file."""
        if self._patchers is None:
            by_file: dict[str, set[str]] = {}
            by_package: dict[str, set[str]] = {}
            unknown: dict[str, str] = {}
            for rel in self.files:
                info = self._parsed(rel)
                if info.dynamic is not None:
                    unknown[rel] = f"imports modules dynamically ({info.dynamic})"
                elif info.unknown_patch is not None:
                    unknown[rel] = f"patches an object the scan cannot name ({info.unknown_patch})"
                for target in info.patched:
                    if not _in_package(target):
                        by_package.setdefault(target.partition(".")[0], set()).add(rel)
                        continue
                    module = self._defining_module(target)
                    patched = self._module_file(module) if module else None
                    # A patch of a module outside the snapshot (envs) is not
                    # kernel source, as the module itself is not.
                    if patched in self._sources and patched != rel:
                        by_file.setdefault(patched, set()).add(rel)
            self._patchers = _Patchers(
                {k: frozenset(v) for k, v in by_file.items()},
                {k: frozenset(v) for k, v in by_package.items()},
                unknown,
            )
        return self._patchers

    def _closure(self, start: str) -> tuple[frozenset[str], tuple[str, ...]]:
        if start not in self._closures:
            patchers = self._patch_index()
            # A file that may patch any module reaches every closure.
            reasons = [f"{rel} {why}" for rel, why in sorted(patchers.unknown.items())]
            files, seen, work = set(), set(), [(start, None)]
            while work:
                node = work.pop()
                if node in seen:
                    continue
                seen.add(node)
                rel, name = node
                files.add(rel)
                work.extend((p, None) for p in patchers.by_file.get(rel, ()))
                info = self._parsed(rel)
                if name is None or name in info.defined_names or name not in info.imported_names:
                    entries = info.entries
                else:
                    entries = info.imported_names[name]
                for entry in entries:
                    if _in_package(entry.module):
                        work.extend(self._targets(entry))
                    else:
                        package = entry.module.partition(".")[0]
                        work.extend((p, None) for p in patchers.by_package.get(package, ()))
            self._closures[start] = (frozenset(files), tuple(dict.fromkeys(reasons)))
        return self._closures[start]


@functools.lru_cache(maxsize=1)
def default_resolver() -> KernelDigestResolver:
    """The resolver over this package, built once per process."""
    return KernelDigestResolver(PACKAGE_ROOT)


# ------------------------------------------------------------- installation


def disabled() -> bool:
    """Whether :data:`KNOB` asks for the library's own keys."""
    from vllm_neuron import envs

    return bool(envs.VLLM_NEURON_DISABLE_KERNEL_DIGEST_KEY)


def _library_fn(current: Callable) -> Callable:
    if hasattr(current, DIGEST_ATTR):
        return getattr(current, LIBRARY_FN_ATTR)
    return current


def _install(
    module,
    name: str,
    digest: str,
    digest_for_call: Optional[Callable[[str, list], str]] = None,
    resolver: Optional[KernelDigestResolver] = None,
) -> None:
    """Wrap ``module.name`` to fold ``digest``, or ``digest_for_call(base, args)``."""
    library = _library_fn(getattr(module, name))
    signature = inspect.signature(library)
    params = list(signature.parameters)

    @functools.wraps(library)
    def keyed(*args, **kwargs):
        base = library(*args, **kwargs)
        if base is None:  # an uncacheable NKI call stays uncacheable
            return None
        if digest_for_call is None:
            return fold_key(base, digest)
        bound = signature.bind(*args, **kwargs).arguments
        return fold_key(base, digest_for_call(base, [bound.get(p) for p in params]))

    setattr(keyed, DIGEST_ATTR, digest)
    if resolver is not None:
        setattr(keyed, RESOLVER_ATTR, resolver)
    setattr(module, name, keyed)


def _restore(module, name: str) -> None:
    setattr(module, name, _library_fn(getattr(module, name)))


def _key_targets():
    from libtorch_neuronx_lite.compile import cache
    from libtorch_neuronx_lite.nki import nki_cache

    return ((cache, "create_cache_hash"), (nki_cache, "create_nki_cache_key"))


def install_kernel_digest_key(digest: str) -> Optional[str]:
    """Fold one ``digest`` into every key of both library functions; return it, or None.

    This is the whole-package scheme (any kernel edit moves every key); startup
    installs :func:`install_per_graph_digest_key`. Installing again replaces the
    previous fold rather than stacking on it. With :data:`KNOB` set, both module
    attributes are put back to the library's own functions.
    """
    targets = _key_targets()
    if disabled():
        for module, name in targets:
            _restore(module, name)
        return None
    for module, name in targets:
        _install(module, name, digest)
    return digest


class _KeyLog:
    """Per-install bookkeeping: one INFO line per new graph and per fallback reason."""

    def __init__(self, resolver: KernelDigestResolver):
        self.resolver = resolver
        self.graphs: dict[str, bool] = {}
        self.reasons: set[str] = set()

    def _log_reasons(self, resolution: KernelResolution) -> None:
        for reason in resolution.reasons:
            if reason not in self.reasons:
                self.reasons.add(reason)
                logger.info(
                    "kernel digest fallback: %s; keys that reach it fold the "
                    "whole-package digest",
                    reason,
                )

    def _resolve(self, refs: Callable[[], list[Reference]], what: str) -> KernelResolution:
        try:
            return self.resolver.resolve(refs())
        except Exception as exc:  # a scan bug must cost a recompile, not a stale hit
            logger.warning("kernel digest reference scan failed for %s: %r", what, exc)
            return KernelResolution(None, frozenset(), (f"reference scan failed for {what}: {exc!r}",))

    def graph_digest(self, base: str, args: list) -> str:
        started = time.perf_counter()
        resolution = self._resolve(lambda: graph_module_references(args[0]), "a graph")
        digest = self.resolver.digest(resolution)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        self._log_reasons(resolution)
        if base not in self.graphs:
            self.graphs[base] = resolution.per_graph
            per_graph = sum(self.graphs.values())
            totals = f"graphs per-graph={per_graph} fallback={len(self.graphs) - per_graph}"
            if resolution.per_graph:
                logger.info(
                    "kernel digest: graph key %s folds files=%d modules=%s resolve_ms=%.1f (%s)",
                    fold_key(base, digest),
                    len(resolution.files),
                    ",".join(sorted(resolution.modules)) or "-",
                    elapsed_ms,
                    totals,
                )
            else:
                logger.info(
                    "kernel digest: graph key %s folds the whole-package digest (%s)",
                    fold_key(base, digest),
                    totals,
                )
        return digest

    def kernel_digest(self, base: str, args: list) -> str:
        func, kernel_args = args[0], args[1]
        resolution = self._resolve(
            lambda: kernel_call_references(func, kernel_args),
            f"NKI kernel {getattr(func, '__qualname__', func)!r}",
        )
        self._log_reasons(resolution)
        return self.resolver.digest(resolution)


def install_per_graph_digest_key(resolver: KernelDigestResolver) -> Optional[str]:
    """Fold per-graph and per-kernel closure digests into both library keys.

    A graph key folds the digest of the files its kernels reach, an NKI kernel
    key the files that kernel reaches; an unresolvable reference folds
    ``resolver.package_digest``. Returns the package digest, or None when
    :data:`KNOB` keeps the library's own keys (both attributes restored).
    """
    targets = _key_targets()
    if disabled():
        for module, name in targets:
            _restore(module, name)
        return None
    log = _KeyLog(resolver)
    (cache, graph_fn), (nki_cache, nki_fn) = targets
    _install(cache, graph_fn, resolver.package_digest, log.graph_digest, resolver)
    _install(nki_cache, nki_fn, resolver.package_digest, log.kernel_digest, resolver)
    return resolver.package_digest


def install_at_startup() -> Optional[str]:
    """Snapshot the package's kernel sources, install per-graph keys, log once at INFO."""
    started = time.perf_counter()
    resolver = default_resolver()
    elapsed_ms = (time.perf_counter() - started) * 1e3
    installed = install_per_graph_digest_key(resolver)
    if installed is None:
        logger.info(
            "VLLM_NEURON_KERNEL_DIGEST=%s not folded into compile cache keys: %s=1",
            resolver.package_digest,
            KNOB,
        )
    else:
        logger.info(
            "VLLM_NEURON_KERNEL_DIGEST=%s files=%d digest_ms=%.1f keys=per-graph",
            resolver.package_digest,
            len(resolver.files),
            elapsed_ms,
        )
    return installed


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """Print the package digest; ``--files`` lists the inputs, ``--graph`` a graph's files."""
    parser = argparse.ArgumentParser(
        prog="python -m vllm_neuron.compile_cache_key",
        description="Kernel-source digests folded into the Neuron compile cache keys.",
    )
    parser.add_argument("--files", action="store_true", help="list every digested file")
    parser.add_argument(
        "--graph",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            f"a compile-cache entry directory, its {FX_GRAPH_FILE} or its "
            f"{HLO_GRAPH_FILE}: print the kernel files that graph's key folds "
            "(repeatable)"
        ),
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    resolver = default_resolver()
    elapsed_ms = (time.perf_counter() - started) * 1e3
    print(
        f"VLLM_NEURON_KERNEL_DIGEST={resolver.package_digest} "
        f"files={len(resolver.files)} digest_ms={elapsed_ms:.1f}"
    )
    if args.files:
        print("\n".join(resolver.files))
    for path in args.graph:
        started = time.perf_counter()
        resolution = resolver.resolve(graph_file_references(path))
        elapsed_ms = (time.perf_counter() - started) * 1e3
        modules = ",".join(sorted(resolution.modules)) or "-"
        digest = resolver.digest(resolution)
        if resolution.per_graph:
            print(
                f"GRAPH {path} per-graph files={len(resolution.files)} digest={digest} "
                f"resolve_ms={elapsed_ms:.1f} modules={modules}"
            )
            for rel in sorted(resolution.files):
                print(f"  {rel}")
        else:
            print(f"GRAPH {path} whole-package digest={digest} modules={modules}")
            for reason in resolution.reasons:
                print(f"  fallback: {reason}")
    return 0


if __name__ == "__main__":
    # Run the imported module, not this __main__ copy: importing vllm_neuron
    # already loaded it, and its resolver is the one built once per process.
    from vllm_neuron.compile_cache_key import main as _main

    sys.exit(_main())
