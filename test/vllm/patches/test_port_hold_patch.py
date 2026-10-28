# SPDX-License-Identifier: Apache-2.0
"""The loopback rendezvous store waits as long as the caller's process-group timeout.

Workers start one after another, so at large world sizes the last one reaches rank 0's store
late. The patch computes the timeout (torch's default when the caller gives none) and rank 0's
store must wait that long, not TCPStore's own 300 s. No process group is formed: TCPStore and
the original init_process_group are replaced. Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import os
import socket
from datetime import timedelta

import torch.distributed as dist
from torch.distributed.constants import default_pg_timeout

from vllm_neuron.vllm.patches import port_hold_patch as patch


class RecordingStore:
    """Stands in for TCPStore with its constructor keywords and defaults; keeps what it got."""

    def __init__(
        self,
        host_name,
        port,
        world_size=None,
        is_master=False,
        timeout=timedelta(seconds=300),
        wait_for_workers=True,
        multi_tenant=False,
        master_listen_fd=None,
        use_libuv=True,
    ):
        self.timeout = timeout
        self.master_listen_fd = master_listen_fd


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_rank0_store_gets_the_timeout_it_receives(tmp_path, monkeypatch):
    """The store rank 0 creates waits for the timeout the function was given."""
    monkeypatch.setattr(patch, "_RENDEZVOUS_DIR", str(tmp_path))
    monkeypatch.setattr(dist, "TCPStore", RecordingStore)
    wanted = timedelta(seconds=1234)

    store = patch._rank0_create_store("127.0.0.1", _free_port(), 2, wanted)
    os.close(store.master_listen_fd)

    assert store.timeout == wanted


def test_default_process_group_timeout_reaches_the_store_creator(tmp_path, monkeypatch):
    """With no timeout from the caller, rank 0's store creator gets torch's default."""
    monkeypatch.setattr(patch, "_RENDEZVOUS_DIR", str(tmp_path))
    given = {}

    def create_store(host, port, world_size, timeout):
        given["timeout"] = timeout
        return object()

    monkeypatch.setattr(patch, "_rank0_create_store", create_store)
    monkeypatch.setattr(patch, "_original_init_process_group", lambda **kwargs: None)

    patch._init_process_group_patched(
        backend="gloo", init_method=f"tcp://127.0.0.1:{_free_port()}", rank=0, world_size=2
    )

    assert given["timeout"] == default_pg_timeout
