# SPDX-License-Identifier: Apache-2.0
"""The worker spawn patch: start-up payloads delivered without blocking the parent.

``vllm_neuron/vllm/patches/worker_spawn_patch.py`` gives
``multiprocessing.popen_spawn_posix`` its own ``open``, the call ``Popen._launch``
uses to write a spawned child's pickled start-up payload into the child's pipe.
A payload that fits in the empty pipe is written there and then, exactly as
upstream does; a larger one is handed to a background thread, so
``Process.start()`` returns without waiting for the child to read it.

Covered here: importing the plugin installs the patch, applying it twice leaves
one layer, an upstream launcher of a different shape (or a platform that cannot
read a pipe's capacity) is refused by name, a real spawn whose child reads late
no longer holds ``start()`` and still receives its payload intact, a payload
that fits takes no thread, a writer that cannot start is refused by name without
leaking its descriptor, a child that dies before reading is logged, a write that
could overtake one in flight is refused, other ``open`` calls pass through, and
the production import order emits one apply record.

The payload sizes below are chosen against the capacity of a pipe created on this
host at test time (``F_GETPIPE_SZ``), never against a fixed number, because that
capacity drops from 64 KiB to 8 KiB once a user exceeds
``/proc/sys/fs/pipe-user-pages-soft``.
"""

from __future__ import annotations

import builtins
import fcntl
import hashlib
import logging
import multiprocessing
import os
import subprocess
import sys
import textwrap
import threading
import time
import types

import pytest

PATCH_MODULE = "vllm_neuron.vllm.patches.worker_spawn_patch"
LAUNCHER_MODULE = "multiprocessing.popen_spawn_posix"

#: How long the test child's main-module import takes, before it reads its process
#: object. Upstream's ``start()`` waits at least this long; the patched one must
#: return in well under half of it.
CHILD_READ_DELAY_S = 3.0
START_RETURN_BOUND_S = CHILD_READ_DELAY_S / 2


def _patch():
    import vllm_neuron  # noqa: F401  -- the import applies the patch

    from vllm_neuron.vllm.patches import worker_spawn_patch

    return worker_spawn_patch


def _launcher():
    from multiprocessing import popen_spawn_posix

    return popen_spawn_posix


def _fresh_pipe_capacity() -> int:
    """Capacity of a pipe created now, as Popen._launch would get it."""
    r, w = os.pipe()
    try:
        return fcntl.fcntl(w, fcntl.F_GETPIPE_SZ)
    finally:
        os.close(r)
        os.close(w)


def _open_fds() -> set[str]:
    return set(os.listdir("/proc/self/fd"))


def _writer_threads() -> list[threading.Thread]:
    patch = _patch()
    return [t for t in threading.enumerate() if t.name.startswith(patch.WRITER_THREAD_PREFIX)]


def test_importing_the_plugin_installs_the_payload_writer():
    """Importing ``vllm_neuron`` gives the spawn launcher this patch's ``open``."""
    patch = _patch()
    launcher = _launcher()

    installed = launcher.__dict__.get("open")
    assert installed is patch._open_spawn_payload_pipe, (
        f"{LAUNCHER_MODULE}.open is not this patch's shim: {installed!r}"
    )
    assert installed.__wrapped__ is builtins.open, (
        f"the shim's __wrapped__ is not the builtin open: {installed.__wrapped__!r}"
    )


def test_applying_twice_leaves_exactly_one_layer():
    """Re-applying installs nothing new and wraps nothing twice."""
    patch = _patch()
    launcher = _launcher()

    before = launcher.__dict__["open"]
    patch.apply_worker_spawn_patch()
    patch.apply_worker_spawn_patch()
    after = launcher.__dict__["open"]

    assert before is after is patch._open_spawn_payload_pipe
    depth, inner = 0, after
    while hasattr(inner, "__wrapped__"):
        inner = inner.__wrapped__
        depth += 1
    assert depth == 1 and inner is builtins.open, (
        f"expected one layer over builtins.open, measured {depth} over {inner!r}"
    )


def _fake_launcher(launch_source: str, extra_globals: dict | None = None):
    """A stand-in for popen_spawn_posix whose Popen._launch has the given body."""
    module = types.ModuleType("fake_popen_spawn_posix")
    namespace: dict = {"os": os}
    exec(textwrap.dedent(launch_source), namespace)  # noqa: S102 -- test fixture code
    module.Popen = type("Popen", (), {"_launch": namespace["_launch"]})
    for name, value in (extra_globals or {}).items():
        setattr(module, name, value)
    return module


UPSTREAM_SHAPED_LAUNCH = """
    def _launch(self, process_obj):
        fp = None
        parent_r, child_w = os.pipe()
        with open(child_w, 'wb', closefd=False) as f:
            f.write(fp.getbuffer())
"""


def test_a_launcher_that_writes_without_open_is_refused_by_name():
    """If upstream stops opening the pipe with ``open``, applying raises, naming the gap."""
    patch = _patch()
    fake = _fake_launcher(
        """
        def _launch(self, process_obj):
            fp = None
            parent_r, child_w = os.pipe()
            os.write(child_w, fp.getbuffer())
        """
    )

    with pytest.raises(patch.WorkerSpawnPatchTargetError, match=r"'open'"):
        patch._install(fake)
    assert "open" not in fake.__dict__, "a refused install still shadowed open"


def test_a_launcher_without_popen_launch_is_refused_by_name():
    patch = _patch()
    fake = types.ModuleType("fake_popen_spawn_posix")

    with pytest.raises(patch.WorkerSpawnPatchTargetError, match=r"Popen\._launch"):
        patch._install(fake)


def test_a_foreign_open_in_the_launcher_module_is_refused_by_name():
    """Someone else already shadows ``open`` there: refuse rather than stack on it."""
    patch = _patch()

    def foreign_open(*args, **kwargs):
        return builtins.open(*args, **kwargs)

    fake = _fake_launcher(UPSTREAM_SHAPED_LAUNCH, {"open": foreign_open})

    with pytest.raises(patch.WorkerSpawnPatchTargetError, match="already defines"):
        patch._install(fake)
    assert fake.open is foreign_open


def test_a_platform_without_pipe_capacity_reads_is_refused_by_name(monkeypatch):
    """Without ``F_GETPIPE_SZ`` the writer could not tell a payload that fits; refuse at apply."""
    patch = _patch()
    monkeypatch.delattr(patch.fcntl, "F_GETPIPE_SZ")
    fake = _fake_launcher(UPSTREAM_SHAPED_LAUNCH)

    with pytest.raises(patch.WorkerSpawnPatchTargetError, match="F_GETPIPE_SZ"):
        patch._install(fake)
    assert "open" not in fake.__dict__, "a refused install still shadowed open"


def test_an_upstream_shaped_launcher_is_accepted():
    """The shape check passes on a launcher shaped like upstream's (guards the fakes above)."""
    patch = _patch()
    fake = _fake_launcher(UPSTREAM_SHAPED_LAUNCH)

    patch._install(fake)

    assert fake.open is patch._open_spawn_payload_pipe


CHILD_TARGET_SOURCE = """
import hashlib
import sys


def check_payload(blob, expected_sha256):
    sys.exit(0 if hashlib.sha256(blob).hexdigest() == expected_sha256 else 3)
"""

#: The child's main module. A spawned child runs its parent's main module (here:
#: this file, standing in for vLLM's API server) after reading only the small
#: preparation pickle and before reading the process object, so a slow import here
#: is exactly the window in which upstream's parent sits blocked in the write.
SLOW_MAIN_SOURCE = f"""
import time

time.sleep({CHILD_READ_DELAY_S})
"""


def test_start_returns_before_the_child_reads_an_oversized_payload(tmp_path, monkeypatch):
    """A real spawn: ``start()`` no longer waits for a child that reads its payload late."""
    _patch()
    from multiprocessing import spawn

    (tmp_path / "w76_spawn_child.py").write_text(CHILD_TARGET_SOURCE)
    slow_main = tmp_path / "w76_slow_main.py"
    slow_main.write_text(SLOW_MAIN_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    import w76_spawn_child

    upstream_preparation_data = spawn.get_preparation_data

    def preparation_data_with_slow_main(name):
        data = upstream_preparation_data(name)
        data.pop("init_main_from_name", None)
        data["init_main_from_path"] = str(slow_main)
        return data

    monkeypatch.setattr(spawn, "get_preparation_data", preparation_data_with_slow_main)

    capacity = _fresh_pipe_capacity()
    blob = os.urandom(4 * capacity)
    assert len(blob) > capacity, "fixture payload fits in the pipe; nothing to test"

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=w76_spawn_child.check_payload,
        args=(blob, hashlib.sha256(blob).hexdigest()),
        daemon=True,
    )
    t0 = time.monotonic()
    proc.start()
    start_s = time.monotonic() - t0
    proc.join(60)
    total_s = time.monotonic() - t0

    assert proc.exitcode == 0, (
        f"the child did not receive its payload intact (exitcode {proc.exitcode})"
    )
    assert total_s >= CHILD_READ_DELAY_S, (
        f"the child finished in {total_s:.2f} s, under its {CHILD_READ_DELAY_S} s main "
        "import, so the payload was never in flight while start() returned"
    )
    assert start_s < START_RETURN_BOUND_S, (
        f"start() took {start_s:.2f} s with a {len(blob)} B payload over a "
        f"{capacity} B pipe: it waited for the child to read, so starts still serialise"
    )


def test_a_payload_that_fits_is_written_at_once_without_a_thread(monkeypatch):
    patch = _patch()

    def no_thread(*args, **kwargs):
        raise AssertionError("a payload that fits must not start a writer thread")

    monkeypatch.setattr(patch.threading, "Thread", no_thread)
    r, w = os.pipe()
    try:
        payload = os.urandom(_fresh_pipe_capacity() // 2)
        with patch._open_spawn_payload_pipe(w, "wb", closefd=False) as f:
            assert f.write(memoryview(payload)) == len(payload)
        os.set_blocking(r, False)
        received = os.read(r, len(payload) + 1)
    finally:
        os.close(r)
        os.close(w)

    assert received == payload


def test_a_writer_that_cannot_start_is_refused_by_name(monkeypatch):
    patch = _patch()

    def cannot_start(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(patch.threading.Thread, "start", cannot_start)
    r, w = os.pipe()
    capacity = fcntl.fcntl(w, fcntl.F_GETPIPE_SZ)
    payload = os.urandom(2 * capacity)
    fds_before = _open_fds()
    try:
        with pytest.raises(patch.SpawnPayloadDeliveryError) as excinfo:
            with patch._open_spawn_payload_pipe(w, "wb", closefd=False) as f:
                f.write(payload)
        fds_after = _open_fds()
    finally:
        os.close(r)
        os.close(w)

    message = str(excinfo.value)
    assert str(len(payload)) in message and str(capacity) in message, message
    assert fds_after == fds_before, f"descriptors leaked: {fds_after - fds_before}"


def test_a_child_that_dies_before_reading_is_logged(caplog):
    patch = _patch()
    caplog.set_level(logging.WARNING, logger=PATCH_MODULE)
    r, w = os.pipe()
    payload = os.urandom(2 * fcntl.fcntl(w, fcntl.F_GETPIPE_SZ))
    os.close(r)  # the "child" is gone before it read anything
    fds_before = _open_fds()
    try:
        with patch._open_spawn_payload_pipe(w, "wb", closefd=False) as f:
            f.write(payload)
        for thread in _writer_threads():
            thread.join(10)
        fds_after = _open_fds()
    finally:
        os.close(w)

    assert not _writer_threads(), "the writer thread did not finish"
    assert fds_after == fds_before, f"the writer's descriptor leaked: {fds_after - fds_before}"
    warnings = [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == PATCH_MODULE and rec.levelno == logging.WARNING
    ]
    assert len(warnings) == 1 and f"{len(payload)}-byte payload" in warnings[0], warnings


def test_a_second_write_after_a_hand_off_is_refused_by_name():
    """A write that could overtake a payload still in flight is refused, not reordered."""
    patch = _patch()
    r, w = os.pipe()
    capacity = fcntl.fcntl(w, fcntl.F_GETPIPE_SZ)
    first = os.urandom(2 * capacity)
    received = bytearray()

    def drain():
        # Reads until every write end is closed, so a regression that lets the
        # second write through fails this test instead of deadlocking it.
        while chunk := os.read(r, capacity):
            received.extend(chunk)

    reader = threading.Thread(target=drain)
    reader.start()
    try:
        with patch._open_spawn_payload_pipe(w, "wb", closefd=False) as f:
            f.write(first)
            with pytest.raises(patch.SpawnPayloadDeliveryError, match="out of order"):
                f.write(b"x")
        for thread in _writer_threads():
            thread.join(10)
    finally:
        os.close(w)
        reader.join(10)
        os.close(r)

    assert bytes(received) == first, "the handed-off payload was not delivered intact"


def test_other_opens_in_the_launcher_module_pass_through(tmp_path):
    patch = _patch()
    path = tmp_path / "plain.txt"

    with patch._open_spawn_payload_pipe(path, "w") as f:
        f.write("text")
    with patch._open_spawn_payload_pipe(path, "rb") as f:
        assert f.read() == b"text"
    fd = os.open(path, os.O_WRONLY)
    try:
        f = patch._open_spawn_payload_pipe(fd, "wb", closefd=False)
        assert f.__class__.__module__ in ("io", "_io"), (
            f"a regular file got the payload writer: {f!r}"
        )
        f.close()
    finally:
        os.close(fd)


def test_production_import_order_emits_one_apply_record():
    """``import vllm`` first, then the plugin loads: patch installed, one INFO record."""
    # Records are counted, not output lines: vllm_neuron/logging_config.py sends
    # every vllm_neuron record to both stdout (root handler) and stderr (its own).
    child = textwrap.dedent(
        f"""
        import logging
        import sys


        class _ApplyRecords(logging.Handler):
            def __init__(self):
                super().__init__()
                self.levels = []

            def emit(self, record):
                if "worker spawn patch applied" in record.getMessage():
                    self.levels.append(record.levelname)


        counter = _ApplyRecords()
        logging.getLogger("{PATCH_MODULE}").addHandler(counter)

        import vllm  # noqa: F401  -- production order: vllm first
        from vllm.platforms import current_platform

        current_platform.device_type  # resolves the platform, loading the plugin
        from multiprocessing import popen_spawn_posix
        from vllm_neuron.vllm.patches import worker_spawn_patch as p

        p.apply_worker_spawn_patch()  # a re-apply logs nothing
        print("PLUGIN_LOADED=%s" % ("vllm_neuron" in sys.modules))
        print("INSTALLED=%s" % (popen_spawn_posix.__dict__.get("open") is p._open_spawn_payload_pipe))
        print("APPLY_RECORDS=%s" % ",".join(counter.levels))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=300
    )
    readings = dict(
        line.split("=", 1) for line in completed.stdout.splitlines() if line.count("=") == 1
    )

    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-3000:]
    assert readings.get("PLUGIN_LOADED") == "True", readings
    assert readings.get("INSTALLED") == "True", readings
    assert readings.get("APPLY_RECORDS") == "INFO", readings
