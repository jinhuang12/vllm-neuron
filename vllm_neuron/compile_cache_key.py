# SPDX-License-Identifier: Apache-2.0
"""One digest of the NKI kernel sources, folded into the Neuron compile cache keys.

``libtorch_neuronx_lite`` keys a compiled graph on the FX graph text, the input
signatures, the tool versions and the compiler flags
(``compile/cache.py:create_cache_hash``), and keys a compiled NKI kernel on that
kernel function's own source (``nki/nki_cache.py:create_nki_cache_key``). Neither
key sees an edit to a helper the kernel calls, a module constant or a tile table,
so a warm ``NEURON_LIBTORCH_CACHE_ROOT`` serves the old kernel binary.

This module computes one sha256 over every ``*.py`` under ``vllm_neuron/functional``
plus the short allowlist of other modules kernel files import
(:data:`KERNEL_SOURCE_FILES`), and folds it into both library keys. The library
looks both functions up through their module at every call
(``cache.create_cache_hash(...)``; ``from .nki_cache import create_nki_cache_key``
inside ``compile_nki``), so replacing the module attribute is the whole hook. The
library offers no key-component hook of its own, and a digest-named cache
sub-root would miss ``NEURON_LIBTORCH_REMOTE_CACHE`` and an explicit
``compiler_workdir`` and would leave the directory names unchanged; folding the
digest into the key covers all three and a kernel edit lands in a new directory.

Set ``VLLM_NEURON_DISABLE_KERNEL_DIGEST_KEY=1`` to keep the library's own keys.
``python -m vllm_neuron.compile_cache_key [--files]`` prints the digest.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: The ``vllm_neuron`` package directory the default digest reads.
PACKAGE_ROOT = Path(__file__).resolve().parent

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

#: The environment knob that keeps the library's own keys.
KNOB = "VLLM_NEURON_DISABLE_KERNEL_DIGEST_KEY"

#: Attribute on an installed key function holding the digest it folds in.
DIGEST_ATTR = "vllm_neuron_kernel_digest"

#: Attribute on an installed key function holding the library's own function.
LIBRARY_FN_ATTR = "__wrapped__"


def digest_files(root: Path | str = PACKAGE_ROOT) -> list[str]:
    """Sorted package-relative POSIX paths of every file the digest covers."""
    root = Path(root)
    files = {
        path.relative_to(root).as_posix()
        for path in (root / KERNEL_SOURCE_DIR).rglob("*.py")
    }
    files.update(rel for rel in KERNEL_SOURCE_FILES if (root / rel).is_file())
    return sorted(files)


def _compute_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for rel in digest_files(root):
        data = (root / rel).read_bytes()
        digest.update(f"{rel}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def _default_digest() -> str:
    return _compute_digest(PACKAGE_ROOT)


def kernel_digest(root: Path | str | None = None) -> str:
    """The 64-hex sha256 of the kernel sources under ``root`` (default: this package).

    Relative paths and contents feed the hash, so a rename or a one-byte edit
    changes it and two checkouts of one commit at different paths share it. The
    default root is computed once per process.
    """
    if root is None:
        return _default_digest()
    return _compute_digest(Path(root))


def fold_key(base: str, digest: str) -> str:
    """The library's 32-hex key with the kernel digest folded in, still 32 hex."""
    return hashlib.sha256(f"{base}|kernel_digest:{digest}".encode()).hexdigest()[:32]


def disabled() -> bool:
    """Whether :data:`KNOB` asks for the library's own keys."""
    from vllm_neuron import envs

    return bool(envs.VLLM_NEURON_DISABLE_KERNEL_DIGEST_KEY)


def _library_fn(current: Callable) -> Callable:
    if hasattr(current, DIGEST_ATTR):
        return getattr(current, LIBRARY_FN_ATTR)
    return current


def _install(module, name: str, digest: str) -> None:
    library = _library_fn(getattr(module, name))

    @functools.wraps(library)
    def keyed(*args, **kwargs):
        base = library(*args, **kwargs)
        if base is None:  # an uncacheable NKI call stays uncacheable
            return None
        return fold_key(base, digest)

    setattr(keyed, DIGEST_ATTR, digest)
    setattr(module, name, keyed)


def _restore(module, name: str) -> None:
    setattr(module, name, _library_fn(getattr(module, name)))


def install_kernel_digest_key(digest: str) -> Optional[str]:
    """Fold ``digest`` into both library key functions; return it, or None if disabled.

    Installing again replaces the previous fold rather than stacking on it. With
    :data:`KNOB` set, both module attributes are put back to the library's own
    functions.
    """
    from libtorch_neuronx_lite.compile import cache
    from libtorch_neuronx_lite.nki import nki_cache

    targets = ((cache, "create_cache_hash"), (nki_cache, "create_nki_cache_key"))
    if disabled():
        for module, name in targets:
            _restore(module, name)
        return None
    for module, name in targets:
        _install(module, name, digest)
    return digest


def install_at_startup() -> Optional[str]:
    """Compute the package digest, install it, and log it once at INFO."""
    started = time.perf_counter()
    digest = kernel_digest()
    elapsed_ms = (time.perf_counter() - started) * 1e3
    installed = install_kernel_digest_key(digest)
    if installed is None:
        logger.info(
            "VLLM_NEURON_KERNEL_DIGEST=%s not folded into compile cache keys: %s=1",
            digest,
            KNOB,
        )
    else:
        logger.info(
            "VLLM_NEURON_KERNEL_DIGEST=%s files=%d digest_ms=%.1f",
            digest,
            len(digest_files()),
            elapsed_ms,
        )
    return installed


def main(argv: list[str] | None = None) -> int:
    """Print ``VLLM_NEURON_KERNEL_DIGEST=<hex>``; with ``--files`` list the inputs."""
    argv = sys.argv[1:] if argv is None else argv
    started = time.perf_counter()
    digest = kernel_digest()
    elapsed_ms = (time.perf_counter() - started) * 1e3
    files = digest_files()
    print(f"VLLM_NEURON_KERNEL_DIGEST={digest} files={len(files)} digest_ms={elapsed_ms:.1f}")
    if "--files" in argv:
        print("\n".join(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
