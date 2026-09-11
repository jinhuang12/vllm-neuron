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
refusal (``mla_sparse.py:1400``), which every MLA layer of every step reaches
(``model_fp8.py:6895``, unconditionally).

THE TWO ITEMS

* A07 -- the whole tiny root forward on ``meta``, at the prefill bucket and at decode,
  through the runner's own capture entry points and its own kwargs builder. It requires that
  NO host read of a tensor value happens anywhere on that path, and that the forward reached
  the attention seam's dispatch. ONE STAND-IN IS DECLARED: the NKI dispatch inside the seam
  returns the seam's declared output shape on ``meta`` instead of entering the vendor
  kernel. That is the landed capture-sites pattern for a backend boundary
  (``tiny/test_tiny_glm5next_capture_sites.py:120-139``).
* D01 -- THE DIAGNOSTIC, and it is not a criterion. The same two legs with the REAL vendor
  boundary, reported and never asserted: one row per leg saying whether the forward
  completed and, if it did not, the first line of the failure and the site. It passes
  whatever the answer is, because what a vendor entry point does with ``meta`` inputs is a
  campaign finding for the lead and not this increment's acceptance.

WHAT A07 DOES NOT MEASURE. The kernel's own behaviour on ``meta``: the stand-in replaces
exactly that. D01 is where that question is answered, without a verdict attached.

THE BASE ARM, DECLARED: A07 FAILS and D01 PASSES.
"""

from __future__ import annotations

import traceback

import pytest
import torch

from vllm_neuron.functional.attention import mla_sparse as seam

# The landed tiny files and the landed capture-sites file, imported rather than
# re-implemented: the fixture and its dials, the runner-shaped cache dict, the CPU-lane gate
# and the runner shell all come from items that already measure this tree.
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_capture_sites as sites
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as landed

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The two legs a captured step is extracted for, with the entry point each one drives.
LEGS = ("prefill", "decode")


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


# ══════════════════════════════════════════════════════════════════════════════
# A07. The forward reads no value, and it reaches the attention dispatch.
# ══════════════════════════════════════════════════════════════════════════════
def test_a07_a_captured_forward_reads_no_value_off_a_tensor(monkeypatch) -> None:
    """Both legs, on ``meta``, with the one declared stand-in at the NKI dispatch."""
    dispatched: list[tuple] = []

    def stand_in(entry):
        """The seam's declared return, ``[S, H, L]`` float32 (``mla_sparse.py:1345``)."""

        def call(q_lift, *rest):
            dispatched.append(tuple(int(d) for d in q_lift.shape))
            return torch.zeros(
                tuple(q_lift.shape), dtype=torch.float32, device=q_lift.device
            )

        return call

    monkeypatch.setattr(seam, "wrap_nki", stand_in)

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
            f"|error={'none' if error is None else str(error).splitlines()[0]}"
        )
        if error is not None:
            assert not _reads_a_value(error), (
                f"the {leg} forward read a value off a tensor at {_site_of(error)}: {error}"
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
