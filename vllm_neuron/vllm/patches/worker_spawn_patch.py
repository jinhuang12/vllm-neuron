# SPDX-License-Identifier: Apache-2.0
"""Deliver a spawned process's start-up payload without making the parent wait.

``vllm.v1.executor.multiproc_executor.MultiprocExecutor._init_executor`` starts
its workers one at a time, calling ``WorkerProc.make_worker_process`` and so
``Process.start()`` once per local rank. With the ``spawn`` start method (vLLM's
CLI default), ``multiprocessing.popen_spawn_posix.Popen._launch`` pickles the
child's start-up payload, starts a fresh interpreter, and then writes the payload
into the child's pipe with ``open(parent_w, 'wb', closefd=False).write(...)``.
That write blocks once the payload is larger than the pipe, until the child reads
the rest -- and the child reads its process object only after it has imported
its parent's main module (``vllm.entrypoints.openai.api_server`` for
``python -m``). So every start waits for one child's interpreter start and
imports, and a TP=64 engine pays that 64 times in a row before the first worker
can initialise. The served GLM-5.3-Flash payload is 71,320 bytes (the
``VllmConfig`` is 69,139 of them); a pipe holds 64 KiB, and only 8 KiB once the
user is over ``/proc/sys/fs/pipe-user-pages-soft``.

This patch gives ``popen_spawn_posix`` its own ``open``, which ``_launch`` looks
up as a module global before the builtin. For exactly upstream's call, a
write-only open of a pipe descriptor with ``closefd=False``, it returns a writer
that:

* writes a payload that fits in the empty pipe at once, as upstream does (such a
  write never blocks, so nothing changes for it), and
* hands a larger payload to a background thread that writes it into a duplicate
  of the descriptor and closes the duplicate, so ``start()`` returns without
  waiting and the children read their payloads, and import, concurrently.

Every other ``open`` call reaching the module is passed to the builtin
unchanged. The pipe is not resized: ``F_SETPIPE_SZ`` is refused (``EPERM``) for an
unprivileged user over the per-user pipe budget, which a busy host reaches, and a
larger pipe per worker would draw on that budget for the life of the engine.

What a child sees is unchanged: the same bytes on the same pipe, read the same
way. What the parent sees differs in two edge cases. A child that exits before
reading surfaces as a WARNING from the writer and as the child's own exit, instead
of a ``BrokenPipeError`` out of ``start()``. And a parent that ends with
``os._exit`` straight after ``start()`` can cut the delivery short; a normal exit
does not, because the writer thread is not a daemon.

The patch must be live in the process that starts the workers, vLLM's EngineCore
subprocess, which never calls ``NeuronPlatform.check_and_update_config``; it
imports ``vllm_neuron``, hence the call from ``vllm_neuron/__init__.py``.
"""

from __future__ import annotations

import builtins
import fcntl
import os
import stat
import threading

from vllm.logger import init_logger

logger = init_logger(__name__)

_TARGET_MODULE = "multiprocessing.popen_spawn_posix"
_TARGET_NAME = "open"

#: Names ``Popen._launch`` must reference for this patch to sit on its payload
#: write: it creates the pipe, opens its write end, and writes the pickle buffer.
_LAUNCH_NAMES = ("pipe", "open", "write", "getbuffer")

#: Name prefix of the background writer threads, one per oversized payload.
WRITER_THREAD_PREFIX = "SpawnPayloadWriter"

_applied = False
# The first oversized payload in a process is reported at INFO, the rest at DEBUG.
_reported_background_write = False


class WorkerSpawnPatchTargetError(RuntimeError):
    """The spawn launcher is not shaped the way this patch expects.

    Raised at apply time rather than skipped: without the patch every worker
    start waits on the previous child's imports again, which shows up only as a
    start-up several minutes longer, with nothing pointing back here.
    """


class SpawnPayloadDeliveryError(RuntimeError):
    """A payload larger than its pipe could not be handed to a background writer.

    Raised instead of writing it in the foreground, which would wait on the
    child's imports again, the serialised start this patch exists to remove.
    """


def _write_all(fd: int, data: bytes) -> int:
    """Write all of ``data`` to the pipe ``fd``, blocking until it fits; return its length."""
    view = memoryview(data)
    written = 0
    while written < len(view):
        written += os.write(fd, view[written:])
    return written


def _write_in_background(fd: int, payload: bytes) -> None:
    """Thread body: deliver ``payload`` into the pipe ``fd``, then close ``fd``."""
    try:
        _write_all(fd, payload)
    except OSError as exc:
        logger.warning(
            "Neuron: a spawned process closed its start-up pipe before reading its "
            "%d-byte payload (%s); it exited before it could start.",
            len(payload),
            exc,
        )
    finally:
        os.close(fd)


class _SpawnPayloadWriter:
    """The writer ``Popen._launch`` gets for its payload pipe.

    Used as ``with open(fd, 'wb', closefd=False) as f: f.write(payload)``. The
    descriptor stays owned by ``_launch`` (its finalizer closes it), so neither
    ``close`` nor ``__exit__`` touches it.
    """

    def __init__(self, fd: int):
        self._fd = fd
        self._handed_off = False

    def __enter__(self) -> _SpawnPayloadWriter:
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def write(self, payload) -> int:
        """Write ``payload`` now if it fits in the pipe, else from a background thread.

        Returns ``len(payload)`` in bytes, as a file's ``write`` would.

        Raises:
            SpawnPayloadDeliveryError: the payload does not fit and no background
                writer could be started for it, or a payload was already handed
                off (``_launch`` writes once; a second write could overtake the
                first and corrupt the stream).
        """
        global _reported_background_write

        data = bytes(payload)
        if self._handed_off:
            raise SpawnPayloadDeliveryError(
                f"a second write ({len(data)} bytes) to a spawn pipe whose payload "
                "is still being delivered from a background thread would land out "
                "of order; Popen._launch is expected to write its payload once."
            )
        capacity = fcntl.fcntl(self._fd, fcntl.F_GETPIPE_SZ)
        if len(data) <= capacity:
            # The pipe is empty (``_launch`` created it for this write), so this
            # never blocks: upstream's behaviour, byte for byte.
            return _write_all(self._fd, data)

        fd = None
        try:
            fd = os.dup(self._fd)
            threading.Thread(
                target=_write_in_background,
                args=(fd, data),
                name=f"{WRITER_THREAD_PREFIX}-{fd}",
                daemon=False,
            ).start()
        except (OSError, RuntimeError) as exc:
            if fd is not None:
                os.close(fd)
            raise SpawnPayloadDeliveryError(
                f"cannot hand a {len(data)}-byte start-up payload over a "
                f"{capacity}-byte pipe to a background writer ({exc}); writing it "
                "in the foreground would wait for the child's imports and "
                "serialise worker starts again."
            ) from exc

        self._handed_off = True
        log = logger.debug if _reported_background_write else logger.info
        log(
            "Neuron: spawn start-up payload of %d bytes exceeds its %d-byte pipe; "
            "delivering it from a background thread so Process.start() does not "
            "wait for the child's imports.",
            len(data),
            capacity,
        )
        _reported_background_write = True
        return len(data)


def _open_spawn_payload_pipe(file, mode="r", *args, **kwargs):
    """``open`` as ``popen_spawn_posix`` sees it.

    Upstream's payload open -- ``open(fd, 'wb', closefd=False)`` on a pipe --
    returns a :class:`_SpawnPayloadWriter`; any other call is the builtin's.
    """
    if (
        isinstance(file, int)
        and mode == "wb"
        and not args
        and kwargs == {"closefd": False}
        and stat.S_ISFIFO(os.fstat(file).st_mode)
    ):
        return _SpawnPayloadWriter(file)
    return builtins.open(file, mode, *args, **kwargs)


_open_spawn_payload_pipe.__wrapped__ = builtins.open


def _install(launcher_module) -> None:
    """Shadow ``open`` in ``launcher_module`` after checking ``Popen._launch``'s shape.

    Raises:
        WorkerSpawnPatchTargetError: ``Popen._launch`` is missing, does not
            reference every name in :data:`_LAUNCH_NAMES`, ``fcntl`` cannot read
            a pipe's capacity, or the module already defines an ``open`` of its
            own.
    """
    popen = getattr(launcher_module, "Popen", None)
    launch = getattr(popen, "_launch", None)
    code = getattr(launch, "__code__", None)
    if code is None:
        raise WorkerSpawnPatchTargetError(
            f"{_TARGET_MODULE}.Popen._launch is missing or not a Python function "
            f"(got {launch!r}); the spawn payload write this patch takes off the "
            "parent's critical path is not where it expects. Check the Python "
            "version's multiprocessing."
        )
    missing = [name for name in _LAUNCH_NAMES if name not in code.co_names]
    if missing:
        raise WorkerSpawnPatchTargetError(
            f"{_TARGET_MODULE}.Popen._launch no longer references "
            f"{', '.join(repr(n) for n in missing)}; this patch relies on it "
            "opening the pipe's write end with open() and writing the pickle "
            "buffer to it. Check the Python version's multiprocessing."
        )
    if not hasattr(fcntl, "F_GETPIPE_SZ"):
        raise WorkerSpawnPatchTargetError(
            "fcntl.F_GETPIPE_SZ is not available (Linux >= 2.6.35 with Python >= "
            "3.10 provides it); the patch reads each spawn pipe's capacity with it."
        )
    current = launcher_module.__dict__.get(_TARGET_NAME)
    if current is _open_spawn_payload_pipe:
        return
    if current is not None:
        raise WorkerSpawnPatchTargetError(
            f"{_TARGET_MODULE} already defines {_TARGET_NAME!r} ({current!r}); "
            "refusing to shadow another patch's open."
        )
    setattr(launcher_module, _TARGET_NAME, _open_spawn_payload_pipe)


def apply_worker_spawn_patch() -> None:
    """Install the background payload writer into the spawn launcher.

    Idempotent: repeated calls leave exactly one shim installed.

    Raises:
        WorkerSpawnPatchTargetError: the launcher's shape changed (see
            :func:`_install`).
    """
    global _applied

    if _applied:
        return
    from multiprocessing import popen_spawn_posix

    _install(popen_spawn_posix)
    _applied = True
    logger.info(
        "Neuron: worker spawn patch applied: %s writes a start-up payload larger "
        "than its pipe from a background thread, so worker starts do not wait on "
        "each child's imports.",
        _TARGET_MODULE,
    )
