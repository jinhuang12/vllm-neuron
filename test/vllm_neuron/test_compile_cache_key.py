# SPDX-License-Identifier: Apache-2.0
"""The kernel source digest folded into the Neuron compile cache keys.

``libtorch_neuronx_lite`` keys a compiled graph on its FX text, its inputs and
the tool versions, and keys a compiled NKI kernel on that kernel function's own
source. An edit to a helper, a module constant or a tile table under
``vllm_neuron/functional`` changes neither key, so a warm cache root serves the
old kernel. ``vllm_neuron.compile_cache_key`` folds one digest of the kernel
sources into both keys. These tests read the digest, the cache directory the
real capture backend writes for a tiny FX graph, and the key the real compile
backend looks up.
"""

from __future__ import annotations

import ast
import inspect
import json
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

#: Modules that files under ``functional/`` import but that hold no kernel
#: source: they run on the host around a kernel call, never inside one.
NOT_KERNEL_SOURCE = {
    # Debug tensor capture, called from torch code in functional/sampling.py.
    "vllm_neuron.accuracy.tensor_capture",
    # Environment knobs: a kernel branches on the value at trace time, and the
    # branch taken is in the FX graph text the library already keys on. The
    # file itself holds no kernel code.
    "vllm_neuron.envs",
}


class _Graph(torch.nn.Module):
    """A tiny graph the real capture backend lowers to HLO without a device."""

    def forward(self, x):
        return (x * 2 + 1,)


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
def capture_env(tmp_path, monkeypatch):
    """Let the real capture backend run on meta inputs into a private cache root."""
    root = tmp_path / "cache-root"
    monkeypatch.setenv("NEURON_LIBTORCH_CACHE_ROOT", str(root))
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "0")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_COMPILE", "1")
    return root / "neuron" / "compile_cache"


def _copy_tree(dst_parent: Path) -> Path:
    """Copy every file the digest covers into ``dst_parent/vllm_neuron``."""
    dst = dst_parent / "vllm_neuron"
    for rel in ck.digest_files(PACKAGE_ROOT):
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PACKAGE_ROOT / rel, target)
    return dst


def _capture_dirs(cache_dir: Path) -> set[str]:
    """Cache directory names the capture backend wrote a graph into."""
    if not cache_dir.is_dir():
        return set()
    return {p.name for p in cache_dir.iterdir() if (p / "graph.hlo").is_file()}


def _capture_once(cache_dir: Path) -> str:
    """Run the real capture backend on the tiny graph; return its new cache dir."""
    from libtorch_neuronx_lite.compile.capture_backend import capture

    before = _capture_dirs(cache_dir)
    gm = torch.fx.symbolic_trace(_Graph())
    capture(gm, [torch.empty(4, 8, device="meta")], {})
    new = _capture_dirs(cache_dir) - before
    assert len(new) == 1, f"capture wrote {sorted(new)} under {cache_dir}"
    (name,) = new
    meta = json.loads((cache_dir / name / ".artifact_metadata_v0.json").read_text())
    assert meta["cache_key"] == name
    return name


def _subprocess_report(extra_env: dict[str, str]) -> dict:
    """Import vllm_neuron in a fresh process and report the digest it installed."""
    code = (
        "import json\n"
        "import vllm_neuron\n"
        "from vllm_neuron import compile_cache_key as ck\n"
        "from libtorch_neuronx_lite.compile import cache\n"
        "from libtorch_neuronx_lite.nki import nki_cache\n"
        "print('REPORT ' + json.dumps({\n"
        "    'digest': ck.kernel_digest(),\n"
        "    'graph': getattr(cache.create_cache_hash, ck.DIGEST_ATTR, None),\n"
        "    'nki': getattr(nki_cache.create_nki_cache_key, ck.DIGEST_ATTR, None),\n"
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


# ---------------------------------------------------------------- the digest


def test_digest_is_stable_across_processes():
    """(a) Two fresh processes on one tree compute and install the same digest."""
    first = _subprocess_report({})
    second = _subprocess_report({})

    assert first["digest"] == second["digest"] == ck.kernel_digest()
    assert len(first["digest"]) == 64
    # Importing vllm_neuron is what installs the digest into both library keys.
    assert first["graph"] == first["nki"] == first["digest"]


def test_digest_covers_functional_and_the_allowlist():
    files = ck.digest_files(PACKAGE_ROOT)
    on_disk = sorted(
        p.relative_to(PACKAGE_ROOT).as_posix()
        for p in (PACKAGE_ROOT / "functional").rglob("*.py")
    )

    assert on_disk and set(on_disk) <= set(files)
    for rel in ck.KERNEL_SOURCE_FILES:
        assert rel in files
    assert files == sorted(files)


def test_digest_does_not_depend_on_where_the_tree_lives(tmp_path):
    """Two checkouts of one commit at different paths share their cache keys."""
    copy = _copy_tree(tmp_path)

    assert ck.kernel_digest(copy) == ck.kernel_digest(PACKAGE_ROOT)


def test_one_appended_byte_in_any_covered_file_changes_the_digest(tmp_path):
    """(b) Every file under functional/ and in the allowlist feeds the digest."""
    copy = _copy_tree(tmp_path)
    base = ck.kernel_digest(copy)
    files = ck.digest_files(copy)
    assert any(rel.startswith("functional/") for rel in files)

    unchanged = []
    for rel in files:
        path = copy / rel
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        try:
            if ck.kernel_digest(copy) == base:
                unchanged.append(rel)
        finally:
            path.write_bytes(original)

    assert unchanged == []
    assert ck.kernel_digest(copy) == base


def test_bytecode_and_non_python_files_do_not_change_the_digest(tmp_path):
    copy = _copy_tree(tmp_path)
    base = ck.kernel_digest(copy)
    cached = copy / "functional" / "__pycache__"
    cached.mkdir(exist_ok=True)
    (cached / "norm.cpython-312.pyc").write_bytes(b"\x00stale")
    (copy / "functional" / "notes.txt").write_text("scratch")

    assert ck.kernel_digest(copy) == base


def test_a_renamed_kernel_file_changes_the_digest(tmp_path):
    copy = _copy_tree(tmp_path)
    base = ck.kernel_digest(copy)
    src = copy / "functional" / "norm.py"
    src.rename(src.with_name("norm_renamed.py"))

    assert ck.kernel_digest(copy) != base


def test_every_vllm_neuron_module_a_kernel_file_imports_is_digested():
    """A new import from functional/ into the package must join the allowlist."""
    covered = set(ck.digest_files(PACKAGE_ROOT))
    missing = set()
    for path in (PACKAGE_ROOT / "functional").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                if not name.startswith("vllm_neuron."):
                    continue
                if name.startswith("vllm_neuron.functional"):
                    continue
                rel = name[len("vllm_neuron.") :].replace(".", "/")
                as_module = f"{rel}.py"
                as_package = f"{rel}/__init__.py"
                if not (PACKAGE_ROOT / as_module).is_file() and not (
                    PACKAGE_ROOT / as_package
                ).is_file():
                    continue  # an imported name, not a module
                if name in NOT_KERNEL_SOURCE:
                    continue
                if as_module not in covered and as_package not in covered:
                    missing.add(f"{name} (from {path.relative_to(PACKAGE_ROOT)})")
    assert sorted(missing) == []


def test_the_fold_keeps_the_library_key_format():
    folded = ck.fold_key("0123456789abcdef0123456789abcdef", "a" * 64)

    assert len(folded) == 32
    int(folded, 16)
    assert folded != ck.fold_key("0123456789abcdef0123456789abcdef", "b" * 64)
    assert folded == ck.fold_key("0123456789abcdef0123456789abcdef", "a" * 64)


# ------------------------------------------------------- the graph cache key


def test_capture_cache_dir_name_changes_with_the_kernel_digest(
    tmp_path, lib, restore_keys, capture_env
):
    """(c) The same FX graph lands in a different cache dir when a kernel changes.

    The real capture backend lowers the tiny graph to HLO in CPU-compile mode
    and names its cache directory with the key; nothing is stubbed.
    """
    copy = _copy_tree(tmp_path)
    digest_a = ck.kernel_digest(copy)
    kernel = copy / "functional" / "dsa" / "decode_batch.py"
    kernel.write_bytes(kernel.read_bytes() + b"\n")
    digest_b = ck.kernel_digest(copy)
    assert digest_a != digest_b

    assert ck.install_kernel_digest_key(digest_a) == digest_a
    dir_a = _capture_once(capture_env)
    assert ck.install_kernel_digest_key(digest_b) == digest_b
    dir_b = _capture_once(capture_env)

    assert dir_a != dir_b
    # Each name is the library's own key with the digest folded in.
    cache, _ = lib
    library_key = getattr(cache.create_cache_hash, ck.LIBRARY_FN_ATTR)
    gm = torch.fx.symbolic_trace(_Graph())
    from libtorch_neuronx_lite.compile.backend import (
        _apply_platform_compiler_args,
        preprocess_and_validate_inputs,
    )

    gm, inputs = preprocess_and_validate_inputs(
        gm, [torch.empty(4, 8, device="meta")], {}
    )
    base = library_key(gm, inputs, _apply_platform_compiler_args({}))
    assert dir_a == ck.fold_key(base, digest_a)
    assert dir_b == ck.fold_key(base, digest_b)


def test_compile_backend_looks_up_the_dir_capture_wrote(
    lib, restore_keys, capture_env, monkeypatch
):
    """The registered compile backend and the capture backend agree on the key.

    The compile backend is the one vLLM's torch.compile resolves by name. Only
    its miss path is replaced: past the lookup it would run neuronx-cc and load
    the NEFF on a device.
    """
    import torch._dynamo.backends.registry as registry

    cache, _ = lib
    ck.install_kernel_digest_key(ck.kernel_digest())
    captured = _capture_once(capture_env)

    class _LookedUp(Exception):
        pass

    def miss(hash_key, local_cache_dir, *args, **kwargs):
        raise _LookedUp(hash_key, local_cache_dir)

    monkeypatch.setattr(cache, "fetch_remote_or_compile", miss)
    backend = registry.lookup_backend("neuron_libtorch")
    gm = torch.fx.symbolic_trace(_Graph())
    with pytest.raises(_LookedUp) as looked_up:
        backend(gm, [torch.empty(4, 8, device="meta")])

    hash_key, local_cache_dir = looked_up.value.args
    assert hash_key == captured
    assert Path(local_cache_dir) == capture_env


# --------------------------------------------------------- the NKI cache key


def _kernel_stand_in(x_hbm, w_hbm):
    return x_hbm


def test_nki_cache_key_changes_with_the_kernel_digest(lib, restore_keys):
    """A helper-only edit must miss the per-kernel NKI result cache too.

    That cache hashes only the kernel function's own source; a stale entry
    hands a freshly compiled graph the old kernel's binary path.
    """
    _, nki_cache = lib
    args = {"x_hbm": torch.empty(4, 8), "w_hbm": torch.empty(8, 8)}

    ck.install_kernel_digest_key("a" * 64)
    key_a = nki_cache.create_nki_cache_key(_kernel_stand_in, args, (1,))
    ck.install_kernel_digest_key("b" * 64)
    key_b = nki_cache.create_nki_cache_key(_kernel_stand_in, args, (1,))
    library_key = getattr(nki_cache.create_nki_cache_key, ck.LIBRARY_FN_ATTR)
    base = library_key(_kernel_stand_in, args, (1,))

    assert base is not None
    assert key_a == ck.fold_key(base, "a" * 64)
    assert key_b == ck.fold_key(base, "b" * 64)
    assert key_a != key_b


def test_an_uncacheable_nki_call_stays_uncacheable(lib, restore_keys):
    """The library returns None for an unhashable argument; the fold keeps it None."""
    _, nki_cache = lib
    args = {"x": bytearray(b"not hashable")}
    assert nki_cache.create_nki_cache_key(_kernel_stand_in, args, (1,)) is None

    ck.install_kernel_digest_key("a" * 64)

    assert nki_cache.create_nki_cache_key(_kernel_stand_in, args, (1,)) is None


# ------------------------------------------------------------------ the knob


def test_the_knob_restores_the_library_keys(
    tmp_path, lib, restore_keys, capture_env, monkeypatch
):
    """(d) With the knob set, keys and cache dir names are the library's own."""
    cache, nki_cache = lib
    library_graph = getattr(cache.create_cache_hash, ck.LIBRARY_FN_ATTR, None)
    library_graph = library_graph or cache.create_cache_hash
    library_nki = getattr(nki_cache.create_nki_cache_key, ck.LIBRARY_FN_ATTR, None)
    library_nki = library_nki or nki_cache.create_nki_cache_key

    ck.install_kernel_digest_key("a" * 64)
    keyed = _capture_once(capture_env)
    shutil.rmtree(capture_env)

    monkeypatch.setenv(KNOB, "1")
    assert ck.install_kernel_digest_key("a" * 64) is None
    assert cache.create_cache_hash is library_graph
    assert nki_cache.create_nki_cache_key is library_nki
    plain = _capture_once(capture_env)

    assert plain != keyed
    assert keyed == ck.fold_key(plain, "a" * 64)


def test_the_knob_keeps_a_fresh_process_on_the_library_keys():
    report = _subprocess_report({KNOB: "1"})

    assert report["graph"] is None
    assert report["nki"] is None


def test_installing_twice_folds_the_digest_once(lib, restore_keys):
    _, nki_cache = lib
    args = {"x_hbm": torch.empty(4, 8)}
    ck.install_kernel_digest_key("a" * 64)
    once = nki_cache.create_nki_cache_key(_kernel_stand_in, args, (1,))
    ck.install_kernel_digest_key("a" * 64)

    assert nki_cache.create_nki_cache_key(_kernel_stand_in, args, (1,)) == once


# ----------------------------------------------- seams in the installed library


def test_the_library_reads_both_key_functions_at_call_time():
    """The wrap only takes effect where the library looks the function up late.

    A library release that binds either name at import time would bypass the
    wrap and bring back stale kernels; this test fails first.
    """
    import libtorch_neuronx_lite
    from libtorch_neuronx_lite.compile import backend, capture_backend
    from libtorch_neuronx_lite.nki import nki_compile

    assert "cache.create_cache_hash(" in inspect.getsource(backend.compile)
    assert "cache.create_cache_hash(" in inspect.getsource(
        capture_backend.setup_workdir_common
    )
    compile_nki = inspect.getsource(nki_compile.compile_nki)
    assert "from .nki_cache import" in compile_nki
    assert "create_nki_cache_key(" in compile_nki
    assert not hasattr(nki_compile, "create_nki_cache_key")

    lib_dir = Path(libtorch_neuronx_lite.__file__).parent
    early_binders = []
    for path in lib_dir.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:  # module level only: names bound at import
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in ("create_cache_hash", "create_nki_cache_key"):
                        early_binders.append(f"{path.relative_to(lib_dir)}")
    assert early_binders == []
