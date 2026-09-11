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
(``model_fp8.py:6873``, unconditionally).

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

SO EVERY KERNEL-CLASS DISPATCH ON THE PATH IS EITHER STOOD IN -- the seven declared sites
below -- OR STOOD DOWN onto its own torch oracle, and each of the seven is read per site
rather than as a total, because a module that never consults the predicate does not care what
the predicate was rebound to.

WHAT A07 DOES NOT MEASURE, AND WHOSE QUESTION THAT IS. Any kernel's own answer to ``meta``
inputs is D01's question, not this item's. What A07 measures is the torch orchestration
BETWEEN the kernels, on ``meta``, for host reads of a value: the runner's kwargs builder, the
converter's geometry reads, the carriers, the mHC affinity and combine oracle, the norms and
residuals, the DSA oracles, the MoE mapping flow and the seam's own dispatch. That is the
defect class this file exists for, and it is a criterion.

THE BASE ARM, DECLARED: A07 FAILS and D01 PASSES.
"""

from __future__ import annotations

import importlib
import os
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

#: One module on this path that IS gated and DOES carry a torch oracle
#: (``mhc/hyper_connection.py:423``), where standing the predicate down is what routes around
#: the vendor. The premise reads the stand-down there, and nowhere else: at a site that never
#: consults the predicate, the rebound name says nothing about what ran.
STOOD_DOWN_ON_THE_PATH = f"{FUNCTIONAL}.mhc.hyper_connection"

#: The six dispatch sites on this path that no stand-down can reach -- five that never consult
#: their own predicate and one that refuses instead of falling back -- and the kernel each one
#: enters. The seam's own declared stand-in is the seventh site, counted by its own list. Both
#: legs reach all seven, so the premise reads them site by site.
DECLARED_SITES = {
    "mhc/sinkhorn.py:1005": "sinkhorn_blocks_kernel",
    "attention/mla_projections.py:272": "mla_projection_kernel",
    "attention/mla_absorb.py:322": "mla_absorb_kernel",
    "moe/moe_blockwise_fp8.py:1354": "moe_gate_up_blockwise_fp8_kernel",
    "moe/moe_blockwise_fp8.py:1638": "moe_swiglu_transposed_kernel",
    "moe/moe_blockwise_fp8.py:1865": "moe_down_blockwise_fp8_kernel",
}

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
    load's own refusal says exactly that (``model_fp8.py:8107-8112``). They are re-bound here
    at the device the rest of the model now holds, through the same method the load calls
    (``model_fp8.py:8989``), or the first mHC layer meets a CPU weight with ``meta``
    activations.

    THE BIND RUNS UNDER A DEFAULT-DEVICE CONTEXT, and that is not decoration. The method
    builds a fresh site object and assigns each loaded weight onto that object's own
    parameter (``model_fp8.py:8130``); the object is allocated wherever the default device
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
    print(f"META|stash_to_meta|moved={len(_every_stash_to_meta(root))}")
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


def _every_stash_to_meta(root) -> list[str]:
    """Move what a module keeps OUTSIDE its parameters and buffers, and name what moved.

    ``.to(device)`` visits parameters, buffers and submodules and nothing else. A module that
    stashes a tensor as a plain attribute keeps it where the load put it: the kernel scale
    operands are built once at load time and stashed exactly so
    (``PREPARED_SCALE_OPERANDS_ATTR``, read back by ``_prepared_scale_operand``, which refuses
    rather than building one per step), and the mHC sites above are the same shape of thing. The
    reference is by NAME and not by line: a stash is found by the attribute it is kept under, and
    that name survives a fold that moves every line around it. A stash
    left on the host is INVISIBLE inside a held dispatch, which makes its own operands at the
    shapes it is handed, and fatal in a torch route, which requires one device -- which is how
    a load-time scale operand ends a forward inside a fallback oracle rather than at a kernel.
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
    (``model_fp8.py:4771`` is one) reads the source, which is patched too. The seam under
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
        real = REAL_WRAP_NKI(self.kernel)
        returned = (real if self.grid is None else real[self.grid])(
            *(_on_the_host(value) for value in args),
            **{key: _on_the_host(value) for key, value in kwargs.items()},
        )
        # EVERY TENSOR LEAF GOES BACK, whatever shape the return has: a bare tensor, a tuple, a
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
            # A DEVICE MISMATCH IS ATTRIBUTED WHERE IT HAPPENED, not left to a traceback: the
            # row names the site and the last dispatch this arm handed back, with its leaves.
            if error is not None and "expected device" in str(error):
                print(
                    f"A07|{leg}|device_mismatch|site={_site_of(error)}"
                    f"|last_held={crossed[-1] if crossed else 'none'}"
                )
            if error is not None:
                assert not _reads_a_value(error), (
                    f"the {leg} forward read a value off a tensor at "
                    f"{_site_of(error)}: {error}"
                )
            # THE ITEM'S OWN PREMISE, CHECKED BEFORE ITS OUTCOME IS JUDGED, AND READ PER SITE.
            # Every declared dispatch was entered, so nothing on this path decided the leg's
            # outcome by running a kernel on meta. A total would pass with one site missing.
            entered = [held.split("->", 1)[0].rsplit(".", 1)[-1] for held in crossed]
            for site, kernel in DECLARED_SITES.items():
                assert kernel in entered, (
                    f"the {leg} forward never entered the declared dispatch at {site}, so "
                    f"nothing stood in for {kernel} and its own answer to meta inputs "
                    f"decided this leg; entered: {sorted(set(entered))}"
                    + (f". It stopped at {_site_of(error)}: {error}" if error else "")
                )
            # AND THE STAND-DOWN IS READ WHERE THE PREDICATE IS ACTUALLY CONSULTED. At a site
            # that never reads it, the rebound name would say nothing about what ran.
            gated = sys.modules.get(STOOD_DOWN_ON_THE_PATH)
            assert gated is not None and gated.can_run_kernel is _refuses_every_route, (
                f"{STOOD_DOWN_ON_THE_PATH} kept its own kernel route, so the {leg} outcome "
                f"at every gated site is the vendor's answer to meta inputs"
            )
            # The two modules the layer reaches before the seam are read as well: they must
            # have been imported, and their dispatch -- which they enter whatever the
            # predicate says (``mla_projections.py:271-272``) -- must be held.
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

    AND THE ROW NAMES THE VENUE IT ASKED. On this lane the dispatch goes to the simulator
    rather than to a device (``neuron_utils.py:17-24``), so what a ``completed=no`` row reports
    is the SIMULATOR's answer to ``meta`` inputs. A device may answer differently, and nothing
    here claims otherwise.

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
            f"|venue={'nki_simulator' if os.environ.get('NKI_SIMULATOR') == '1' else 'device'}"
            f"|seam_dispatch={counters[0]}"
            f"|error={'none' if error is None else str(error).splitlines()[0]}"
            f"|site={'none' if error is None else _site_of(error)}"
        )
