# SPDX-License-Identifier: Apache-2.0
"""A captured step's forward runs on ``meta``, where no value can be read off a tensor."""

from __future__ import annotations

import importlib
import os
import sys
import traceback

import pytest
import torch

from vllm_neuron.functional.attention import mla_sparse as seam
from vllm_neuron.utils import neuron_utils

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_capture_sites as sites
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as tiny_e2e

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The two legs a captured step is extracted for, with the entry point each one drives.
LEGS = ("prefill", "decode")

#: The functional package whose kernel routes this test holds, and the two modules the layer
#: reaches before the seam, which both treatments are checked on afterwards.
FUNCTIONAL = "vllm_neuron.functional"
BEFORE_THE_SEAM = (
    f"{FUNCTIONAL}.attention.mla_projections",
    f"{FUNCTIONAL}.attention.mla_absorb",
)

#: The one module whose seam raises on an unavailable route instead of taking a torch path
#: (``mhc/sinkhorn.py``), so its predicate is the one the stand-down leaves alone.
RAISES_RATHER_THAN_FALLING_BACK = f"{FUNCTIONAL}.mhc.sinkhorn"

#: One module on this path that is gated and does carry a torch oracle
#: (``mhc/hyper_connection.py``), where standing the predicate down is what routes around
#: the vendor. The premise reads the stand-down there, and nowhere else: at a site that never
#: consults the predicate, the rebound name says nothing about what ran.
STOOD_DOWN_ON_THE_PATH = f"{FUNCTIONAL}.mhc.hyper_connection"

#: The dispatch sites on this path that no stand-down can reach -- they never consult their own
#: predicate, or they refuse instead of falling back -- and the kernel each one enters. The
#: seam's own declared stand-in is the last site, counted by its own list. Both legs reach all
#: of them, so the premise reads them site by site.
#:
#: the expert site is the fused one, because that is the dispatch this model's packed bank takes:
#: one entry replaces the three blockwise kernels the unpacked bank used to enter, and those
#: three are the non-packed path's, covered where that path is driven. A site named here that
#: the path no longer enters would fail this test for a route change rather than a defect.
DECLARED_SITES = {
    "mhc/sinkhorn.py": "sinkhorn_blocks_kernel",
    "attention/mla_projections.py": "mla_projection_kernel",
    "attention/mla_absorb.py": "mla_absorb_kernel",
    "moe/fused_fp8.py": "moe_fused_fp8_kernel",
}

#: The vendor's dispatch wrapper and the route predicate as they are before anything here
#: patches them: what a held dispatch calls, and what the teardown hands back.
NKI_HOP = "libtorch_neuronx_lite.nki.nki_hop"
REAL_WRAP_NKI = importlib.import_module(NKI_HOP).wrap_nki
REAL_CAN_RUN_KERNEL = neuron_utils.can_run_kernel


def _meta_root_and_runner():
    """The tiny root, its banks and a runner shell, all on ``meta``. """
    tiny_e2e._require_cpu_mode()
    root = tiny_e2e._fixture()["root"]
    caches = {
        name: [tensor.to("meta") for tensor in tensors]
        for name, tensors in tiny_e2e._runner_shaped_caches(root).items()
    }
    root.to("meta")
    with torch.device("meta"):
        for module in root.modules():
            if hasattr(type(module), "bind_hyper_connection_sites"):
                module.bind_hyper_connection_sites(root.text_config, torch.device("meta"))
    root.bind_kv_cache(caches)
    # ``.to(device)`` visits parameters, buffers and submodules and nothing else, so a
    # tensor a module stashes as a plain attribute stays where the load put it. A stash
    # left on the host is invisible inside a held dispatch and fatal in a torch route,
    # which requires one device.
    _every_stash_to_meta(root)
    runner = sites._runner(root)
    runner.device = torch.device("meta")
    return root, runner


def _on_meta(value):
    """``value`` with every tensor leaf on ``meta``, or ``value`` itself if none was elsewhere."""
    if isinstance(value, torch.Tensor):
        return value if value.device.type == "meta" else value.to("meta")
    if isinstance(value, dict):
        moved = {key: _on_meta(item) for key, item in value.items()}
        return moved if any(moved[key] is not value[key] for key in value) else value
    if isinstance(value, (list, tuple)):
        moved = [_on_meta(item) for item in value]
        if all(new is old for new, old in zip(moved, value)):
            return value
        return moved if isinstance(value, list) else tuple(moved)
    return value


def _extract(runner, leg: str):
    """Drive one capture entry point through the stand-in backend."""
    backend = sites._StandInBackend(runner)
    runner.capture_backend_model = backend
    if leg == "prefill":
        runner.extract_prefill_graphs(sites.PREFILL_BUCKET, 0)
    else:
        runner.extract_decode_graphs(sites.DECODE_BATCH)
    return backend


def _reads_a_value(error: BaseException) -> bool:
    """True when this failure is a host read of a tensor value."""
    text = str(error)
    return (
        "cannot be called on meta tensors" in text
        or "Cannot access data pointer" in text
        or "data-dependent" in text
    )


def _site_of(error: BaseException) -> str:
    """The last frame inside the plugin, as ``file:line``, or the word ``unknown``."""
    frames = [
        frame
        for frame in traceback.extract_tb(error.__traceback__)
        if "/vllm_neuron/" in frame.filename
    ]
    if not frames:
        return "unknown"
    last = frames[-1]
    return f"{last.filename.split('/vllm_neuron/')[-1]}:{last.lineno}"


def _refuses_every_route(*_args, **_kwargs) -> bool:
    """The stood-down kernel route: this module has no NKI route for this call."""
    return False


def _drop_the_module_that_takes_its_dispatch_at_import(monkeypatch) -> None:
    """Drop the one module that takes its dispatch at import, leaving the real one to put
    back.
    """
    name = f"{FUNCTIONAL}.moe.fused_fp8"
    at_the_real_boundary = importlib.import_module(name)
    package, _, leaf = name.rpartition(".")
    # Set to the value it already holds: what this records is the put-back of the package
    # attribute, which the forward's own import rebinds to the module that holds the arm.
    monkeypatch.setattr(sys.modules[package], leaf, at_the_real_boundary)
    monkeypatch.delitem(sys.modules, name)


def _stand_down_every_route_but_the_seam(monkeypatch) -> list[str]:
    """Refuse the NKI route in every functional module but the seam, and name those
    patched.
    """
    spared = importlib.import_module(RAISES_RATHER_THAN_FALLING_BACK)
    patched = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith(FUNCTIONAL) or module is seam or module is spared:
            continue
        if not hasattr(module, "can_run_kernel"):
            continue
        monkeypatch.setattr(module, "can_run_kernel", _refuses_every_route)
        patched.append(name)
    monkeypatch.setattr(neuron_utils, "can_run_kernel", _refuses_every_route)
    return patched


class _HeldBoundary:
    """One module's NKI dispatch, crossed on host operands and handed back on ``meta``.
    """

    def __init__(self, kernel, module: str, crossed: list[str]) -> None:
        self.kernel, self.module, self.crossed = kernel, module, crossed
        self.grid = None

    def __getitem__(self, grid) -> _HeldBoundary:
        """``wrap_nki(k)[n]`` is an spmd launch grid, not an output arity: it is kept."""
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs):
        name = getattr(self.kernel, "__name__", type(self.kernel).__name__)
        real = REAL_WRAP_NKI(self.kernel)
        returned = (real if self.grid is None else real[self.grid])(
            *(_on_the_host(value) for value in args),
            **{key: _on_the_host(value) for key, value in kwargs.items()},
        )
        # Every tensor leaf goes back, whatever shape the return has: a bare tensor, a tuple, a
        # list or a dict of them. The row records what was handed back, leaf by leaf, so a leak
        # here is named by the row rather than found in a traceback two modules later.
        handed = _on_meta(returned)
        self.crossed.append(f"{self.module}.{name}->{_leaves_of(handed)}")
        return handed


def _on_the_host(value):
    """One operand as host data of its own shape, or unchanged if it is not a meta tensor."""
    if not isinstance(value, torch.Tensor) or value.device.type != "meta":
        return value
    make = torch.ones if value.dtype.is_floating_point else torch.zeros
    return make(tuple(value.shape), dtype=value.dtype)


def _every_stash_to_meta(root) -> list[str]:
    """Move what a module keeps outside its parameters and buffers, and name what moved.

    A stash is found by the attribute it is kept under, never by line, so the reference
    survives any edit that moves the lines around it.
    """
    moved = []
    for name, module in root.named_modules():
        for field, value in list(vars(module).items()):
            if field in ("_parameters", "_buffers", "_modules"):
                continue
            replaced = _on_meta(value)
            if replaced is not value:
                setattr(module, field, replaced)
                moved.append(f"{name}.{field}")
    return moved


def _leaves_of(value) -> str:
    """A returned value as ``dtype:device`` per tensor leaf, for a row that must name a leak."""
    if isinstance(value, torch.Tensor):
        return f"{value.dtype}:{value.device.type}".replace("torch.", "")
    if isinstance(value, dict):
        return "{" + ",".join(_leaves_of(item) for item in value.values()) + "}"
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(_leaves_of(item) for item in value) + ")"
    return type(value).__name__


def _hold_every_dispatch_but_the_seam(monkeypatch, crossed: list[str]) -> list[str]:
    """Hold the vendor dispatch in every functional module but the seam, and name them.
    """
    held = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith(FUNCTIONAL) or module is seam:
            continue
        if not hasattr(module, "wrap_nki"):
            continue
        monkeypatch.setattr(module, "wrap_nki", _make_holder(name, crossed))
        held.append(name)
    monkeypatch.setattr(importlib.import_module(NKI_HOP), "wrap_nki",
                        _make_holder(NKI_HOP, crossed))
    return held


def _make_holder(module: str, crossed: list[str]):
    """The ``wrap_nki`` one module binds while its dispatch is held."""

    def hold(kernel) -> _HeldBoundary:
        return _HeldBoundary(kernel, module, crossed)

    return hold


def _where_the_arm_still_stands() -> list[tuple]:
    """``(module, field, what it should hold, module name)`` for each of this arm's objects."""
    standing = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith(FUNCTIONAL):
            continue
        if getattr(module, "can_run_kernel", None) is _refuses_every_route:
            standing.append((module, "can_run_kernel", REAL_CAN_RUN_KERNEL, name))
        dispatch = getattr(module, "wrap_nki", None)
        if dispatch is not None and dispatch is not REAL_WRAP_NKI:
            standing.append((module, "wrap_nki", REAL_WRAP_NKI, name))
    return standing


def _hand_back_what_a_late_import_took() -> list[str]:
    """Undo this arm in the modules ``monkeypatch`` cannot reach, and name what was undone.
    """
    handed_back = []
    for module, field, real, name in _where_the_arm_still_stands():
        setattr(module, field, real)
        handed_back.append(f"{name}.{field}")
    return handed_back


# ══════════════════════════════════════════════════════════════════════════════
# The forward completes, reads no value, and reaches the attention dispatch.
# ══════════════════════════════════════════════════════════════════════════════
def test_a_captured_forward_completes_and_reads_no_value_off_a_tensor(
    monkeypatch,
) -> None:
    """Both legs, on ``meta``, with the one declared stand-in at the NKI dispatch."""
    dispatched: list[tuple] = []
    crossed: list[str] = []

    def stand_in(entry):
        """The seam's declared return, ``[S, H, L]`` float32 (``mla_sparse.py``)."""

        def call(q_lift, *rest):
            dispatched.append(tuple(int(d) for d in q_lift.shape))
            return torch.zeros(
                tuple(q_lift.shape), dtype=torch.float32, device=q_lift.device
            )

        return call

    monkeypatch.setattr(seam, "wrap_nki", stand_in)
    _drop_the_module_that_takes_its_dispatch_at_import(monkeypatch)
    _stand_down_every_route_but_the_seam(monkeypatch)
    held = _hold_every_dispatch_but_the_seam(monkeypatch, crossed)

    try:
        for leg in LEGS:
            _, runner = _meta_root_and_runner()
            seam.reset_mla_sparse_dispatch_counters()
            completed, error = True, None
            try:
                _extract(runner, leg)
            except Exception as caught:  # noqa: BLE001 -- classified below, not swallowed
                completed, error = False, caught
            counters = seam.mla_sparse_dispatch_counters()
            # A device mismatch is attributed where it happened, not left to a traceback: the
            # row names the site and the last dispatch this arm handed back, with its leaves.
            if error is not None:
                assert not _reads_a_value(error), (
                    f"the {leg} forward read a value off a tensor at "
                    f"{_site_of(error)}: {error}"
                )
            # The premise, checked before the outcome is judged, and read per site.
            # Every declared dispatch was entered, so nothing on this path decided the leg's
            # outcome by running a kernel on meta. A total would pass with one site missing.
            entered = [held.split("->", 1)[0].rsplit(".", 1)[-1] for held in crossed]
            for site, kernel in DECLARED_SITES.items():
                assert kernel in entered, (
                    f"the {leg} forward never entered the declared dispatch at {site}, so "
                    f"nothing stood in for {kernel} and its own answer to meta inputs "
                    f"decided this leg; entered: {sorted(set(entered))}"
                    f"; held and never crossed: "
                    f"{sorted(set(held) - {c.split('->', 1)[0] for c in crossed})}"
                    + (f". It stopped at {_site_of(error)}: {error}" if error else "")
                )
            # And the stand-down is read where the predicate is actually consulted. At a site
            # that never reads it, the rebound name would say nothing about what ran.
            gated = sys.modules.get(STOOD_DOWN_ON_THE_PATH)
            assert gated is not None and gated.can_run_kernel is _refuses_every_route, (
                f"{STOOD_DOWN_ON_THE_PATH} kept its own kernel route, so the {leg} outcome "
                f"at every gated site is the vendor's answer to meta inputs"
            )
            # The two modules the layer reaches before the seam are read as well: they must
            # have been imported, and their dispatch -- which they enter whatever the
            # predicate says (``mla_projections.py``) -- must be held.
            for module_name in BEFORE_THE_SEAM:
                reached = sys.modules.get(module_name)
                assert reached is not None, (
                    f"the {leg} forward never imported {module_name}, so nothing was "
                    f"measured against the layer that runs before the seam; it stopped at "
                    f"{_site_of(error) if error else 'no failure'}: {error}"
                )
                assert reached.wrap_nki is not REAL_WRAP_NKI, (
                    f"{module_name} kept the vendor dispatch, which it enters whatever the "
                    f"route predicate says, so the {leg} outcome is the vendor's and not "
                    f"this candidate's"
                )
            assert completed, (
                f"the {leg} forward did not complete on meta; it stopped at "
                f"{_site_of(error)}: {error}"
            )
            assert counters[0] >= 1, (
                f"the {leg} forward never reached the attention seam's dispatch"
                + (f"; it stopped at {_site_of(error)}: {error}" if error else "")
            )
            assert dispatched, f"the {leg} forward never entered the declared stand-in"
            dispatched.clear()
            crossed.clear()
    finally:
        # What monkeypatch cannot give back, this test gives back itself, pass or fail:
        # the modules the forward imported while the source was patched outlive this
        # test, and no teardown reaches them.
        _hand_back_what_a_late_import_took()


# ══════════════════════════════════════════════════════════════════════════════
# The same two legs at the real vendor boundary, reported and not judged.
# ══════════════════════════════════════════════════════════════════════════════
