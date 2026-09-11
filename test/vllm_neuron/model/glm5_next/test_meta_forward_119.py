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
(``model_fp8.py:6895``, unconditionally).

THE TWO ITEMS

* A07 -- the whole tiny root forward on ``meta``, at the prefill bucket and at decode,
  through the runner's own capture entry points and its own kwargs builder. It requires
  that the forward COMPLETES, that NO host read of a tensor value happens anywhere on that
  path, and that it reached the attention seam's dispatch. ONE STAND-IN IS DECLARED: the
  NKI dispatch inside the seam returns the seam's declared output shape on ``meta``
  instead of entering the vendor kernel. That is the landed capture-sites pattern for a
  backend boundary (``tiny/test_tiny_glm5next_capture_sites.py:120-139``). AND EVERY OTHER
  NKI ROUTE IS STOOD DOWN: the same layer enters the real vendor boundary in
  ``mla_projections`` and in ``mla_absorb`` before it ever reaches this seam, so without
  that stand-down the item's outcome would be the vendor's answer to ``meta`` inputs
  rather than the candidate's.
* D01 -- THE DIAGNOSTIC, and it is not a criterion. The same two legs with the REAL vendor
  boundary, reported and never asserted: one row per leg saying whether the forward
  completed and, if it did not, the first line of the failure and the site. It passes
  whatever the answer is, because what a vendor entry point does with ``meta`` inputs is a
  campaign finding for the lead and not this increment's acceptance.

WHAT A07 DOES NOT MEASURE. Any kernel's own behaviour on ``meta``: the stand-in replaces
that for the seam and the stand-down replaces it everywhere else, so what A07 drives outside
the seam is each module's torch route, which is device-independent and readable in this
repository. D01 is where the vendor's answer is reported, without a verdict attached.

THE BASE ARM, DECLARED: A07 FAILS and D01 PASSES.
"""

from __future__ import annotations

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

#: The functional package whose kernel routes A07 stands down, and the two modules the
#: layer reaches before the seam, which the stand-down is checked on afterwards.
FUNCTIONAL = "vllm_neuron.functional"
BEFORE_THE_SEAM = (
    f"{FUNCTIONAL}.attention.mla_projections",
    f"{FUNCTIONAL}.attention.mla_absorb",
)


def _meta_root_and_runner():
    """The tiny root, its banks and a runner shell, all on ``meta``.

    The caches are the runner-shaped dict the landed items build, moved to ``meta`` BEFORE
    they are bound: ``bind_kv_cache`` keeps them as plain entries rather than buffers, so
    moving the module afterwards would leave the banks behind on CPU and the forward would
    mix two devices.
    """
    landed._require_cpu_mode()
    root = landed._fixture()["root"]
    caches = {
        name: [tensor.to("meta") for tensor in tensors]
        for name, tensors in landed._runner_shaped_caches(root).items()
    }
    root.to("meta")
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
    (``model_fp8.py:4906`` is one) reads the source, which is patched too. The seam under
    test keeps the real route, so the forward still reaches the dispatch the stand-in holds.
    """
    patched = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith(FUNCTIONAL) or module is seam:
            continue
        if not hasattr(module, "can_run_kernel"):
            continue
        monkeypatch.setattr(module, "can_run_kernel", _refuses_every_route)
        patched.append(name)
    monkeypatch.setattr(neuron_utils, "can_run_kernel", _refuses_every_route)
    return patched


# ══════════════════════════════════════════════════════════════════════════════
# A07. The forward completes, reads no value, and reaches the attention dispatch.
# ══════════════════════════════════════════════════════════════════════════════
def test_a07_a_captured_forward_completes_and_reads_no_value_off_a_tensor(
    monkeypatch,
) -> None:
    """Both legs, on ``meta``, with the one declared stand-in at the NKI dispatch."""
    dispatched: list[tuple] = []

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
            f"|stood_down={len(stood_down)}"
            f"|error={'none' if error is None else str(error).splitlines()[0]}"
        )
        if error is not None:
            assert not _reads_a_value(error), (
                f"the {leg} forward read a value off a tensor at {_site_of(error)}: {error}"
            )
        # THE ITEM'S OWN PREMISE, CHECKED BEFORE ITS OUTCOME IS JUDGED. Both limbs of the
        # stand-down are visible here: a module imported before the patch was patched by
        # name, and one imported during the forward read the patched source.
        for module_name in BEFORE_THE_SEAM:
            reached = sys.modules.get(module_name)
            assert reached is not None, (
                f"the {leg} forward never imported {module_name}, so nothing was measured "
                f"against the layer that runs before the seam; it stopped at "
                f"{_site_of(error) if error else 'no failure'}: {error}"
            )
            assert reached.can_run_kernel is _refuses_every_route, (
                f"{module_name} kept its own kernel route, so the {leg} outcome is the "
                f"vendor's answer to meta inputs and not this candidate's"
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


# ══════════════════════════════════════════════════════════════════════════════
# D01. The same two legs at the real vendor boundary, reported and not judged.
# ══════════════════════════════════════════════════════════════════════════════
def test_d01_reports_whether_a_meta_forward_completes_at_the_real_boundary() -> None:
    """One row per leg. This item passes whatever the rows say.

    What the vendor's NKI entry point does with ``meta`` inputs is not readable off this
    repository -- the package is installed on the host and nowhere else -- so it is reported
    from the venue that can see it and left to the lead. A ``completed=no`` here is a
    finding, not a failure: A07 above already carries the criterion this increment declares.
    """
    for leg in LEGS:
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
            f"|seam_dispatch={counters[0]}"
            f"|error={'none' if error is None else str(error).splitlines()[0]}"
            f"|site={'none' if error is None else _site_of(error)}"
        )
