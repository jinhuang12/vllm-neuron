# SPDX-License-Identifier: Apache-2.0
"""A captured step's forward runs on ``meta``, where no value can be read off a tensor.

THE DECLARED ACCEPTANCE:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest \\
      test/vllm_neuron/model/glm5_next/test_meta_forward_119.py -s -rA \\
      -p no:randomly -p no:cacheprovider

Two items, one test each, no ``parametrize``. The pins are the two the tiny harness this
file imports declares, and no platform target: the fixture, the cache dict and the runner
shell come from the landed tiny files, so this file measures the tree those items measure.

WHAT THIS FILE IS ABOUT. Graph extraction builds the whole batch on ``meta``
(``neuron_worker.py:500-501``) and calls the model. A precondition that reads a VALUE off a
tensor cannot run there, and two of them were on the path: the converter's geometry reads,
which ``test_host_geometry_119.py`` covers, and the attention seam's selected-row range
refusal (``mla_sparse.py:1411``), which every MLA layer of every step reaches
(``model_fp8.py:6850``, unconditionally).

THE TWO ITEMS

* A07 -- the whole tiny root forward on ``meta``, at the prefill bucket and at decode,
  through the runner's own capture entry points and its own kwargs builder. It requires
  that the forward COMPLETES, that NO host read of a tensor value happens anywhere on that
  path, and that it reached the attention seam's dispatch. THE ARM HOLDS EVERY KERNEL ROUTE
  IT DOES NOT MEASURE, and those routes differ in three ways that need three treatments:

  - the seam under test keeps its real route, and ONE STAND-IN IS DECLARED at its NKI
    dispatch, which returns the seam's declared output shape on ``meta``
    (``mla_sparse.py:1331``) instead of entering the vendor kernel. That is the landed
    capture-sites pattern for a backend boundary
    (``tiny/test_tiny_glm5next_capture_sites.py:120-139``);
  - EVERY OTHER ROUTE PREDICATE IS STOOD DOWN, so each module takes its own torch path --
    every module but one. The batched sinkhorn RAISES when the route is unavailable instead
    of falling back (``mhc/sinkhorn.py:996-1002``), because kernel-class work ships no torch
    path, so standing that one down stops the forward at the first mHC layer;
  - AND EVERY OTHER VENDOR DISPATCH IS HELD BY THE ARM, because a seam that never consults
    the predicate cannot be stood down at all: the MLA projections, the absorb and the three
    MoE limbs dispatch unconditionally (``mla_projections.py:271-272``,
    ``mla_absorb.py:321-322``, ``moe_blockwise_fp8.py:1353-1354``). A held dispatch crosses
    the REAL boundary on operands the arm makes at the shapes it was handed -- a ``meta``
    tensor carries no value to hand over -- and returns the result on ``meta`` for the
    forward to carry on with. The kernel still decides the output's shape, so this file
    declares no kernel's output shape but the seam's own.
* D01 -- THE DIAGNOSTIC, and it is not a criterion. The same two legs with the REAL vendor
  boundary, reported and never asserted: one row per leg saying whether the forward
  completed and, if it did not, the first line of the failure and the site. It passes
  whatever the answer is, because what a vendor entry point does with ``meta`` inputs is a
  campaign finding for the lead and not this increment's acceptance. Its row also names
  whether anything of the arm was still standing when it ran, because that is the one way
  its rows could report the arm instead of the vendor.

WHAT A07 DOES NOT MEASURE. Any kernel's own answer to ``meta`` inputs: the stand-in replaces
that for the seam, the stand-down replaces it wherever a torch path exists, and the held
dispatch supplies the operands everywhere else. What A07 drives is the candidate's own host
code -- the runner's kwargs builder, the converter's geometry reads and each module's torch
route -- all of which is device-independent and readable in this repository. D01 is where the
vendor's answer is reported, without a verdict attached.

THE BASE ARM, DECLARED: A07 FAILS and D01 PASSES.
"""

from __future__ import annotations

import importlib
import sys
import traceback

import pytest
import torch

from vllm_neuron.functional.attention import mla_sparse as seam
from vllm_neuron.utils import neuron_utils

# The landed tiny files and the landed capture-sites file, imported rather than
# re-implemented: the fixture and its dials, the runner-shaped cache dict, the CPU-lane gate
# and the runner shell all come from items that already measure this tree.
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_capture_sites as sites
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as landed

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The two legs a captured step is extracted for, with the entry point each one drives.
LEGS = ("prefill", "decode")

#: The functional package whose kernel routes A07 holds, and the two modules the layer
#: reaches before the seam, which both treatments are checked on afterwards.
FUNCTIONAL = "vllm_neuron.functional"
BEFORE_THE_SEAM = (
    f"{FUNCTIONAL}.attention.mla_projections",
    f"{FUNCTIONAL}.attention.mla_absorb",
)

#: The one module whose seam RAISES on an unavailable route instead of taking a torch path
#: (``mhc/sinkhorn.py:996-1002``), so its predicate is the one the stand-down leaves alone.
RAISES_RATHER_THAN_FALLING_BACK = f"{FUNCTIONAL}.mhc.sinkhorn"

#: The vendor's dispatch wrapper and the route predicate as they are BEFORE anything here
#: patches them: what a held dispatch calls, and what the teardown hands back.
NKI_HOP = "libtorch_neuronx_lite.nki.nki_hop"
REAL_WRAP_NKI = importlib.import_module(NKI_HOP).wrap_nki
REAL_CAN_RUN_KERNEL = neuron_utils.can_run_kernel


def _meta_root_and_runner():
    """The tiny root, its banks and a runner shell, all on ``meta``.

    The caches are the runner-shaped dict the landed items build, moved to ``meta`` BEFORE
    they are bound: ``bind_kv_cache`` keeps them as plain entries rather than buffers, so
    moving the module afterwards would leave the banks behind on CPU and the forward would
    mix two devices.

    THE HYPER-CONNECTION SITES NEED THE SAME TREATMENT, AND ``.to()`` CANNOT GIVE IT. Each
    layer holds its two mHC sites in a plain dict rather than as submodules, so no
    ``.to(device)`` visits them and their weights stay wherever the load put them -- the
    load's own refusal says exactly that (``model_fp8.py:8084-8089``). They are re-bound here
    at the device the rest of the model now holds, through the same method the load calls
    (``model_fp8.py:8830``), or the first mHC layer meets a CPU weight with ``meta``
    activations.

    THE BIND RUNS UNDER A DEFAULT-DEVICE CONTEXT, and that is not decoration. The method
    builds a fresh site object and assigns each loaded weight onto that object's own
    parameter (``model_fp8.py:8101``); the object is allocated wherever the default device
    points, so a bind that targets any other device assigns across two tensor types and
    ``set_data`` refuses. The context makes the site the method builds land on the device
    the bind was asked for.
    """
    landed._require_cpu_mode()
    root = landed._fixture()["root"]
    caches = {
        name: [tensor.to("meta") for tensor in tensors]
        for name, tensors in landed._runner_shaped_caches(root).items()
    }
    root.to("meta")
    with torch.device("meta"):
        for module in root.modules():
            if hasattr(type(module), "bind_hyper_connection_sites"):
                module.bind_hyper_connection_sites(root.text_config, torch.device("meta"))
    root.bind_kv_cache(caches)
    runner = sites._runner(root)
    runner.device = torch.device("meta")
    return root, runner


def _extract(runner, leg: str):
    """Drive one capture entry point through the landed stand-in backend."""
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


def _stand_down_every_route_but_the_seam(monkeypatch) -> list[str]:
    """Refuse the NKI route in every functional module but the seam, and name those patched.

    ``can_run_kernel`` is true in CPU mode with the simulator on ANY device
    (``neuron_utils.py:17-24``), so a route it guards is taken on ``meta`` too. Two limbs
    reach every module: each module binds the name at its own import, so a module already
    imported is patched by name here; a module imported later inside a method
    (``model_fp8.py:4748`` is one) reads the source, which is patched too. The seam under
    test keeps the real route, so the forward still reaches the dispatch the stand-in holds.

    AND ONE MODULE IS SPARED, because standing its route down stops the forward instead of
    routing around it: the batched sinkhorn raises when the route is unavailable rather than
    taking a torch path. It is imported here so that it holds the real predicate BEFORE the
    source is patched -- the second limb would otherwise reach it at the mHC layer's own
    import and put back what this exemption exists to avoid.
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

    A ``meta`` tensor carries no value, so the vendor kernel cannot be given the operands the
    forward built. This makes them at the shapes it was handed -- ones where the operand is
    floating point, because a seam of this family requires its affinities to be positive
    (``mhc/sinkhorn.py:967-968``), and zeros where it is an index, which every table this
    forward addresses holds -- crosses the REAL boundary, and returns the result on ``meta``.
    The kernel decides the output's shape, so no shape is declared here.
    """

    def __init__(self, kernel, module: str, crossed: list[str]) -> None:
        self.kernel, self.module, self.crossed = kernel, module, crossed
        self.grid = None

    def __getitem__(self, grid) -> _HeldBoundary:
        """``wrap_nki(k)[n]`` is an SPMD launch grid, not an output arity: it is kept."""
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs):
        name = getattr(self.kernel, "__name__", type(self.kernel).__name__)
        self.crossed.append(f"{self.module}.{name}")
        real = REAL_WRAP_NKI(self.kernel)
        returned = (real if self.grid is None else real[self.grid])(
            *(_on_the_host(value) for value in args),
            **{key: _on_the_host(value) for key, value in kwargs.items()},
        )
        if isinstance(returned, tuple):
            return tuple(_back_on_meta(value) for value in returned)
        return _back_on_meta(returned)


def _on_the_host(value):
    """One operand as host data of its own shape, or unchanged if it is not a meta tensor."""
    if not isinstance(value, torch.Tensor) or value.device.type != "meta":
        return value
    make = torch.ones if value.dtype.is_floating_point else torch.zeros
    return make(tuple(value.shape), dtype=value.dtype)


def _back_on_meta(value):
    """One returned operand back where the forward is running."""
    return value.to("meta") if isinstance(value, torch.Tensor) else value


def _hold_every_dispatch_but_the_seam(monkeypatch, crossed: list[str]) -> list[str]:
    """Hold the vendor dispatch in every functional module but the seam, and name them.

    A SEAM THAT NEVER CONSULTS THE ROUTE PREDICATE CANNOT BE STOOD DOWN, and several on this
    path do not: the MLA projections, the absorb and the MoE limbs call ``wrap_nki``
    unconditionally, so the stand-down above decides nothing for them. Holding the dispatch
    is what covers those, and it covers them without this file naming a single kernel: the
    same two limbs as the stand-down reach every module, by name where the module is already
    imported and through the source where it is imported later.
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


def _standing_arm() -> list[str]:
    """The same, as ``module.field`` names a row can carry."""
    return [f"{name}.{field}" for _m, field, _real, name in _where_the_arm_still_stands()]


def _hand_back_what_a_late_import_took() -> list[str]:
    """Undo this arm in the modules ``monkeypatch`` cannot reach, and name what was undone.

    ``monkeypatch`` restores what it SET. A module first imported DURING the forward was never
    set: it read the patched source at its own import and kept the arm's objects for the life
    of the process, so the next item in this file would report the arm and not the vendor.
    """
    handed_back = []
    for module, field, real, name in _where_the_arm_still_stands():
        setattr(module, field, real)
        handed_back.append(f"{name}.{field}")
    return handed_back


# ══════════════════════════════════════════════════════════════════════════════
# A07. The forward completes, reads no value, and reaches the attention dispatch.
# ══════════════════════════════════════════════════════════════════════════════
def test_a07_a_captured_forward_completes_and_reads_no_value_off_a_tensor(
    monkeypatch,
) -> None:
    """Both legs, on ``meta``, with the one declared stand-in at the NKI dispatch."""
    dispatched: list[tuple] = []
    crossed: list[str] = []

    def stand_in(entry):
        """The seam's declared return, ``[S, H, L]`` float32 (``mla_sparse.py:1331``)."""

        def call(q_lift, *rest):
            dispatched.append(tuple(int(d) for d in q_lift.shape))
            return torch.zeros(
                tuple(q_lift.shape), dtype=torch.float32, device=q_lift.device
            )

        return call

    monkeypatch.setattr(seam, "wrap_nki", stand_in)
    stood_down = _stand_down_every_route_but_the_seam(monkeypatch)
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
            print(
                f"A07|{leg}|completed={'yes' if completed else 'no'}"
                f"|seam_dispatch={counters[0]}|stand_in_calls={len(dispatched)}"
                f"|stood_down={len(stood_down)}|held={len(held)}|crossed={len(crossed)}"
                f"|error={'none' if error is None else str(error).splitlines()[0]}"
            )
            if error is not None:
                assert not _reads_a_value(error), (
                    f"the {leg} forward read a value off a tensor at "
                    f"{_site_of(error)}: {error}"
                )
            # THE ITEM'S OWN PREMISE, CHECKED BEFORE ITS OUTCOME IS JUDGED. Both limbs of
            # the stand-down are visible here: a module imported before the patch was
            # patched by name, and one imported during the forward read the patched source.
            # THE PREDICATE IS NOT WHAT HOLDS THESE TWO, though -- neither seam consults it
            # (``mla_projections.py:271-272``) -- so each one's dispatch is read as well.
            for module_name in BEFORE_THE_SEAM:
                reached = sys.modules.get(module_name)
                assert reached is not None, (
                    f"the {leg} forward never imported {module_name}, so nothing was "
                    f"measured against the layer that runs before the seam; it stopped at "
                    f"{_site_of(error) if error else 'no failure'}: {error}"
                )
                assert reached.can_run_kernel is _refuses_every_route, (
                    f"{module_name} kept its own kernel route, so the {leg} outcome is the "
                    f"vendor's answer to meta inputs and not this candidate's"
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
            assert crossed, f"the {leg} forward crossed no held dispatch"
            dispatched.clear()
            crossed.clear()
    finally:
        # WHAT MONKEYPATCH CANNOT GIVE BACK, THIS ITEM GIVES BACK ITSELF, pass or fail: the
        # modules the forward imported while the source was patched outlive this item, and
        # the diagnostic below would report this arm instead of the vendor.
        print(f"A07|handed_back|{len(_hand_back_what_a_late_import_took())}")


# ══════════════════════════════════════════════════════════════════════════════
# D01. The same two legs at the real vendor boundary, reported and not judged.
# ══════════════════════════════════════════════════════════════════════════════
def test_d01_reports_whether_a_meta_forward_completes_at_the_real_boundary() -> None:
    """One row per leg. This item passes whatever the rows say.

    What the vendor's NKI entry point does with ``meta`` inputs is not readable off this
    repository -- the package is installed on the host and nowhere else -- so it is reported
    from the venue that can see it and left to the lead. A ``completed=no`` here is a
    finding, not a failure: A07 above already carries the criterion this increment declares.

    AND THE ROW SAYS WHOSE ROUTE IT READ. ``monkeypatch`` restores what it set, so A07's
    treatments come off the modules that existed when it ran -- but a module A07 imported
    DURING its own forward read the patched source and kept those objects, and no teardown
    reaches that. A07 hands them back itself; this row names whatever is still standing,
    because a row reporting the arm's own refusal as the vendor's answer is a finding about
    nothing.
    """
    for leg in LEGS:
        standing = _standing_arm()
        _, runner = _meta_root_and_runner()
        seam.reset_mla_sparse_dispatch_counters()
        completed, error = True, None
        try:
            _extract(runner, leg)
        except Exception as caught:  # noqa: BLE001 -- reported, never asserted on
            completed, error = False, caught
        counters = seam.mla_sparse_dispatch_counters()
        print(
            f"DIAG|meta_forward|leg={leg}|completed={'yes' if completed else 'no'}"
            f"|route={'real' if not standing else 'arm:' + ','.join(standing)}"
            f"|seam_dispatch={counters[0]}"
            f"|error={'none' if error is None else str(error).splitlines()[0]}"
            f"|site={'none' if error is None else _site_of(error)}"
        )
