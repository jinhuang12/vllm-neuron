# SPDX-License-Identifier: Apache-2.0
"""``mla_decode_masked.py`` reaches only the compile cache keys of graphs that call it.

``vllm_neuron.compile_cache_key`` folds into a graph's key the digest of the kernel
files the graph's kernels reach: each kernel's defining module and the closure of its
imports, plus every file that patches a module in that closure. ``mla_decode_masked``
reuses :mod:`mla_decode`'s helpers, and every decode graph reaches :mod:`mla_decode`
(the bypass step's kernels live there). So the new file stays out of every other
graph's key only while:

1. no other digested file reaches it, at any depth, through an import or a patch;
2. it patches nothing it imports (a patcher joins the closure of every graph that
   reaches the patched module);
3. it imports nothing dynamically (that makes every key fold the whole-package digest)
   and nothing of ``nkilib`` (that joins it to the closure of every graph that calls the
   ``nkilib`` kernel).

Each test reads the resolver's own scan of the files, so the rules are the key's rules.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import vllm_neuron
from vllm_neuron import compile_cache_key as ck

PACKAGE_ROOT = Path(vllm_neuron.__file__).resolve().parent
MODULE = "vllm_neuron.functional.attention.mla_decode_masked"
REL = "functional/attention/mla_decode_masked.py"


def test_the_file_is_kernel_source():
    assert REL in ck.digest_files(PACKAGE_ROOT)


def test_no_other_digested_file_reaches_it():
    resolver = ck.KernelDigestResolver(PACKAGE_ROOT)
    reaching = [rel for rel in resolver.files
                if rel != REL and REL in resolver._closure(rel)[0]]
    assert reaching == []


def test_it_patches_nothing_it_imports():
    resolver = ck.KernelDigestResolver(PACKAGE_ROOT)
    scan = resolver._parsed(REL)
    assert scan.patched == ()
    assert scan.unknown_patch is None
    patchers = resolver._patch_index()
    assert REL not in patchers.unknown
    assert not any(REL in files for files in patchers.by_file.values())
    assert not any(REL in files for files in patchers.by_package.values())


def test_it_imports_nothing_dynamically_and_nothing_of_nkilib():
    resolver = ck.KernelDigestResolver(PACKAGE_ROOT)
    assert resolver._parsed(REL).dynamic is None
    assert REL not in resolver._importers_of("nkilib")


def test_a_graph_that_calls_its_kernels_resolves_per_graph():
    resolver = ck.KernelDigestResolver(PACKAGE_ROOT)
    resolution = resolver.resolve([ck.Reference(ck.RefKind.MODULE, MODULE)])
    assert resolution.per_graph, resolution.reasons
    assert REL in resolution.files


def test_every_other_kernel_module_folds_the_digest_it_folds_without_the_file(tmp_path):
    """The keys of graphs whose kernels are defined elsewhere are those of a tree
    without this file: every other module's closure digest is unchanged."""
    trees = {}
    for name, keep in (("with", True), ("without", False)):
        root = tmp_path / name / "vllm_neuron"
        for rel in ck.digest_files(PACKAGE_ROOT):
            if rel == REL and not keep:
                continue
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PACKAGE_ROOT / rel, root / rel)
        trees[name] = ck.KernelDigestResolver(root)
    with_file, without = trees["with"], trees["without"]
    assert set(with_file.files) - set(without.files) == {REL}
    moved = []
    for rel in without.files:
        closure, reasons = with_file._closure(rel)
        before = without.digest_of(without._closure(rel)[0])
        if reasons or with_file.digest_of(closure) != before:
            moved.append(rel)
    assert moved == []
