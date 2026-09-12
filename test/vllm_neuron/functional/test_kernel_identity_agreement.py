"""Does each module's first-hop identity function name the kernel the seam really dispatched?

Increment ``inc-glm53f-092``, plan revision 94.

THE QUESTION THIS FILE ANSWERS
------------------------------
Six modules under ``vllm_neuron/functional/`` expose an identity function that reports which
NKI member the module talks to. Five read a **module-level name** -- an import in
``depthwise_conv1d``, a same-file ``@nki.jit def`` in four more. ``moe_blockwise_fp8`` reads
no such name: it takes the name from its seam's source. No claim watches a dispatch.

So each test here takes an INDEPENDENT reading -- the first positional argument handed to
``nki.simulator.simulate_kernel``, which is the kernel the CPU dispatch actually ran -- and
compares it with what the module claims. The claim is a string; the reading is an event. A
claim that agrees with a reading derived from the claim would certify nothing, which is why
the reading never touches the module's imports.

FIRST HOP, AND WHY ONE MODULE USES A DIFFERENT FUNCTION
-------------------------------------------------------
Five seams wrap their kernel directly, so ``kernel_identity()`` IS the first hop for them.
``moe_blockwise_fp8`` is the only one of the six that puts a **shim** between seam and kernel:
its own diagram at ``vllm_neuron/functional/moe/moe_blockwise_fp8.py:545-547`` shows
``blockwise_fp8_moe --wrap_nki--> the shim --return--> the kernel``, with
``seam_identity()`` naming the first hop and
``kernel_identity()`` naming the second. The simulator dispatch is the FIRST hop, because
``wrap_nki`` registers the shim. So that module's first-hop function is ``seam_identity()``,
and its ``kernel_identity()`` is deliberately NOT compared here -- it reports a different hop
by design, and increment ``-077`` already certifies it.

THE THREE INSTRUMENTS, AND WHY THREE
------------------------------------
1. ``nki_dispatch`` / ``torch_fallback`` counters inside each module's seam.
2. ``can_run_kernel()`` -- the gate that decides kernel versus torch oracle.
3. real ``nki.simulator.simulate_kernel`` invocations, counted and ATTRIBUTED to the seam
   under test.

Instrument 3 counts the vendor entry point, so a bug in instrument 1 cannot fake it. This file
uses instruments 2 and 3, because the question is which kernel ran and a reading taken with no
dispatch at all would be the self-referential certificate the increment exists to close.

ATTRIBUTION IS BY (FILE, ENCLOSING FUNCTION), NOT BY FILE
---------------------------------------------------------
``kda/chunked_recurrence.py`` carries TWO seams -- ``kda_intra_chunk`` and
``kda_inter_chunk`` -- each with its own kernel and counters. A frame walk that matched the
filename alone could not tell them apart, and would read a dispatch from the wrong seam as
belonging to the one under test. Matching the enclosing function name too removes that.

THE SEVENTH ITEM CARRIES A CONTROL, AND WHY THE CONTROL PATCHES THE SEAM
------------------------------------------------------------------------
Item 7 counts how many of the six disagree and requires exactly ``0``. A counted zero proves
nothing unless the same count can be shown to move, so the control substitutes what the SEAM
dispatches -- the module's own ``wrap_nki`` reference -- and the count must read exactly ``1``.

The control must NOT patch the module global instead. ``wrap_nki`` applies the same
``.func`` unwrap to the same object the identity function reads, so patching the global moves
BOTH readings together and the count would stay ``0`` while measuring nothing.
"""

from __future__ import annotations

import importlib
import os
import sys
import types

import nki.simulator

# Modules are imported by ``importlib`` rather than ``from ... import name`` on purpose.
# ``blockwise_fp8_mm`` and ``depthwise_conv1d`` each name BOTH a module and a function inside
# it, so the ``from`` form silently binds the function and every later ``__file__`` lookup
# fails on an object that looks close enough to be missed.
_SOURCE = {
    "blockwise_fp8_mm": "vllm_neuron.functional.blockwise_fp8_mm",
    "chunked_recurrence": "vllm_neuron.functional.kda.chunked_recurrence",
    "depthwise_conv1d": "vllm_neuron.functional.kda.depthwise_conv1d",
    "hyper_connection": "vllm_neuron.functional.mhc.hyper_connection",
    "sinkhorn": "vllm_neuron.functional.mhc.sinkhorn",
    "moe_blockwise_fp8": "vllm_neuron.functional.moe.moe_blockwise_fp8",
}
_OWNER_TEST = {
    "blockwise_fp8_mm": "test.vllm_neuron.functional.test_blockwise_fp8_mm",
    "chunked_recurrence": "test.vllm_neuron.functional.kda.test_chunked_recurrence",
    "depthwise_conv1d": "test.vllm_neuron.functional.kda.test_depthwise_conv1d",
    "hyper_connection": "test.vllm_neuron.functional.mhc.test_hyper_connection",
    "sinkhorn": "test.vllm_neuron.functional.mhc.test_sinkhorn",
    "moe_blockwise_fp8": "test.vllm_neuron.functional.moe.test_moe_blockwise_fp8",
}
#: The seam whose dispatches are attributed to it. Two of these live in one file.
_SEAM = {
    "blockwise_fp8_mm": "blockwise_fp8_mm",
    "chunked_recurrence": "kda_intra_chunk",
    "depthwise_conv1d": "depthwise_conv1d",
    "hyper_connection": "hyper_connection_combine",
    "sinkhorn": "sinkhorn_normalise",
    "moe_blockwise_fp8": "blockwise_fp8_moe",
}
#: The FIRST-HOP identity function. One module differs, for the reason the docstring gives.
_FIRST_HOP = {
    "blockwise_fp8_mm": "kernel_identity",
    "chunked_recurrence": "kernel_identity",
    "depthwise_conv1d": "kernel_identity",
    "hyper_connection": "kernel_identity",
    "sinkhorn": "kernel_identity",
    "moe_blockwise_fp8": "seam_identity",
}

_M = {name: importlib.import_module(path) for name, path in _SOURCE.items()}
_T = {name: importlib.import_module(path) for name, path in _OWNER_TEST.items()}


# --------------------------------------------------------------------------- #
# instruments
# --------------------------------------------------------------------------- #
class _DispatchRecorder:
    """Records every simulator dispatch: WHAT ran, and whether the seam under test ran it.

    Follows the ``_SimulatorCounter`` idiom the landed test files use -- swap
    ``nki.simulator.simulate_kernel``, delegate to the real one -- and adds two things the
    plain counter does not have: the identity of the dispatched callable, and attribution by
    walking the live frame stack up to the seam under test.
    """

    def __init__(self, seam_file: str, seam_func: str) -> None:
        self.entries: list[tuple[tuple[str | None, str | None], bool]] = []
        self._file = os.path.realpath(seam_file)
        self._func = seam_func
        self._real = None

    def __enter__(self) -> "_DispatchRecorder":
        self._real = nki.simulator.simulate_kernel
        real, entries = self._real, self.entries
        seam_file, seam_func = self._file, self._func

        def recording(*args, **kwargs):
            func = args[0] if args else kwargs.get("func")
            ident = (getattr(func, "__module__", None), getattr(func, "__qualname__", None))
            frame, attributed = sys._getframe(1), False
            while frame is not None:
                if (os.path.realpath(frame.f_code.co_filename) == seam_file
                        and frame.f_code.co_name == seam_func):
                    attributed = True
                    break
                frame = frame.f_back
            entries.append((ident, attributed))
            return real(*args, **kwargs)

        nki.simulator.simulate_kernel = recording
        return self

    def __exit__(self, *exc_info) -> None:
        nki.simulator.simulate_kernel = self._real

    @property
    def attributed(self) -> list[tuple[str | None, str | None]]:
        return [ident for ident, was_ours in self.entries if was_ours]


def _drive(name: str):
    """Drive one seam once, with the OWNER'S OWN smallest declared case.

    Each module's own test file already owns a minimal fixture that the seam accepts. Reusing
    it means this file invents no shapes, so a shape the owner later changes cannot leave this
    file asserting against a case the seam no longer supports.

    Returns the seam's output and the tensor its gate is asked about.
    """
    mod, owner = _M[name], _T[name]
    if name == "blockwise_fp8_mm":
        case = owner._build_case()
        out = mod.blockwise_fp8_mm(case["x"], case["weight"], case["weight_scale"])
        return out, case["x"]
    if name == "chunked_recurrence":
        q, k, v, beta, gk = owner._inputs(32)
        return mod.kda_intra_chunk(q, k, v, beta, gk), q
    if name == "depthwise_conv1d":
        img, filt = owner._image(), owner._filter()
        return mod.depthwise_conv1d(img, filt), img
    if name == "hyper_connection":
        x, residual, plm, crm = owner._inputs()
        return mod.hyper_connection_combine(x, residual, plm, crm), residual
    if name == "sinkhorn":
        affinity = owner._affinity()
        return mod.sinkhorn_normalise(affinity), affinity
    if name == "moe_blockwise_fp8":
        case = owner._build_case()
        inputs = case["kernel_inputs"]
        return mod.blockwise_fp8_moe(**inputs), inputs["hidden_states"]
    raise AssertionError(f"no driver declared for {name!r}")


def _measure(name: str) -> dict:
    """Drive one seam and return its claimed identity beside the dispatched one."""
    mod = _M[name]
    recorder = _DispatchRecorder(mod.__file__, _SEAM[name])
    with recorder:
        out, gate_probe = _drive(name)
    claimed = getattr(mod, _FIRST_HOP[name])()
    return {
        "claimed": tuple(claimed),
        "attributed": recorder.attributed,
        "dispatches": len(recorder.attributed),
        "gate": mod.can_run_kernel(gate_probe),
        "out": out,
    }


def _assert_first_hop_agrees(name: str) -> dict:
    """The whole per-module conjunct: a live dispatch, an open gate, and the two names equal.

    The route predicate is asserted FIRST. If no dispatch reached the simulator, the identity
    comparison would be comparing a claim against nothing, and it would pass by vacuity.
    """
    r = _measure(name)
    assert r["dispatches"] == 1, (
        f"{name}: expected exactly 1 simulator dispatch attributed to seam "
        f"{_SEAM[name]!r}, got {r['dispatches']}: {r['attributed']}"
    )
    assert r["gate"] is True, f"{name}: can_run_kernel read {r['gate']!r}, expected True"
    assert r["claimed"] == r["attributed"][0], (
        f"{name}: {_FIRST_HOP[name]}() reports {r['claimed']} but the seam dispatched "
        f"{r['attributed'][0]}"
    )
    return r


def _count_disagreements() -> tuple[int, dict]:
    """How many of the six report a first-hop identity the seam did not dispatch."""
    readings, disagreeing = {}, 0
    for name in _SOURCE:
        r = _measure(name)
        agrees = r["dispatches"] == 1 and r["claimed"] == r["attributed"][0]
        readings[name] = (r["claimed"], r["attributed"], agrees)
        if not agrees:
            disagreeing += 1
    return disagreeing, readings


def _decoy_of(kernel):
    """A kernel that RUNS the same code but READS as a different symbol.

    Built from the real kernel's own code object, so the arithmetic is identical and the seam
    still produces a real result -- a control that crashed the kernel would move the reading
    for the wrong reason. Only ``__module__`` and ``__qualname__`` differ, and those are
    exactly what the identity comparison comes down to.
    """
    inner = getattr(kernel, "func", kernel)
    decoy = types.FunctionType(
        inner.__code__, inner.__globals__, "decoy_kernel",
        inner.__defaults__, inner.__closure__,
    )
    decoy.__module__ = "test.control"
    decoy.__qualname__ = "decoy_kernel"
    decoy.__kwdefaults__ = inner.__kwdefaults__
    return decoy


# --------------------------------------------------------------------------- #
# (1)-(6) one first-hop comparison per module, six named functions.
# Deliberately NOT parametrised: the collected item count stays readable from the source, and
# a failure names its module in the test id rather than in a case index.
# --------------------------------------------------------------------------- #
def test_blockwise_fp8_mm_first_hop_identity_matches_the_dispatch():
    """``blockwise_fp8_mm``: ``kernel_identity()`` names the kernel the seam dispatched."""
    _assert_first_hop_agrees("blockwise_fp8_mm")


def test_chunked_recurrence_first_hop_identity_matches_the_dispatch():
    """``kda_intra_chunk``: the reading is attributed to this seam, not its file-mate."""
    _assert_first_hop_agrees("chunked_recurrence")


def test_depthwise_conv1d_first_hop_identity_matches_the_dispatch():
    """``depthwise_conv1d``: the one module of the six whose claim reads a real import."""
    _assert_first_hop_agrees("depthwise_conv1d")


def test_hyper_connection_first_hop_identity_matches_the_dispatch():
    """``hyper_connection_combine``: ``kernel_identity()`` names the dispatched kernel."""
    _assert_first_hop_agrees("hyper_connection")


def test_sinkhorn_first_hop_identity_matches_the_dispatch():
    """``sinkhorn_normalise``: ``kernel_identity()`` names the dispatched kernel."""
    _assert_first_hop_agrees("sinkhorn")


def test_moe_blockwise_fp8_first_hop_identity_matches_the_dispatch():
    """``blockwise_fp8_moe``: the first hop is the SHIM, so ``seam_identity()`` is compared.

    ``kernel_identity()`` names the second hop by design and is not compared here. Increment
    ``-077`` certifies that hop; this test would fail against it for a correct module.
    """
    r = _assert_first_hop_agrees("moe_blockwise_fp8")
    mod = _M["moe_blockwise_fp8"]
    # ASSERTED, not merely recorded: the two hops must differ, or this file's premise moved.
    assert tuple(mod.kernel_identity()) != r["claimed"], (
        "moe_blockwise_fp8 no longer has two distinct hops; seam_identity() and "
        "kernel_identity() now report the same symbol, so this file's premise moved"
    )


# --------------------------------------------------------------------------- #
# (7) the counted zero, with the control that makes it move
# --------------------------------------------------------------------------- #
def test_no_module_disagrees_and_the_control_makes_exactly_one_disagree():
    """Zero of six disagree -- and the same count reads 1 when one seam is substituted.

    Without the control a six-way agreement would prove only that the comparison cannot tell
    the difference. The control patches the module's ``wrap_nki`` reference, which is what the
    SEAM dispatches; patching the module global instead would move the claim and the reading
    together and leave the count at zero.

    Restoration is asserted inside the test rather than left to ``monkeypatch`` teardown, so a
    control that leaked would fail here instead of failing some later file.
    """
    disagreeing, readings = _count_disagreements()
    assert disagreeing == 0, f"modules whose two first-hop readings disagree: {readings}"

    mod = _M["sinkhorn"]
    real_wrap_nki = mod.wrap_nki

    def wrap_nki_with_decoy(kernel):
        return real_wrap_nki(_decoy_of(kernel))

    try:
        mod.wrap_nki = wrap_nki_with_decoy
        under_control, control_readings = _count_disagreements()
        claim_held = tuple(mod.kernel_identity())
        dispatched = control_readings["sinkhorn"][1]
    finally:
        mod.wrap_nki = real_wrap_nki

    assert mod.wrap_nki is real_wrap_nki, "the control leaked: wrap_nki was not restored"
    assert under_control == 1, (
        f"the control substituted one seam, so exactly 1 module must disagree, "
        f"got {under_control}: {control_readings}"
    )
    assert control_readings["sinkhorn"][2] is False, "the substituted module still agreed"
    assert dispatched == [("test.control", "decoy_kernel")], (
        f"the control did not reach the dispatch; it read {dispatched}"
    )
    assert claim_held == ("vllm_neuron.functional.mhc.sinkhorn", "sinkhorn_kernel"), (
        f"the control moved the CLAIM as well as the reading, which proves nothing: "
        f"kernel_identity() read {claim_held}"
    )

    # And the count returns to zero once the seam is restored, so the 1 was the control's
    # doing rather than a state change that outlived it.
    restored, restored_readings = _count_disagreements()
    assert restored == 0, f"disagreements after restoring the seam: {restored_readings}"
