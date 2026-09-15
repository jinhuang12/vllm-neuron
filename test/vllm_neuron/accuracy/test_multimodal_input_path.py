"""``inc-glm53f-006`` -- G15 closure: the instrument's multimodal-input path.

The increment measures a pair of counts, and the pair selects one of three
declared outcomes -- so the branches are distinguished by numbers, not narrative.
**(A)** *"the count of ``accuracy/`` parameters/branches accepting non-text
input, enumerated file-by-file with ``file:line``"*. **(B)** *"the count of
synthetic multimodal samples for which the instrument returns an
``AssertCloseResult``"*. Outcome 1 is ``A >= 1 and B == 1/1``; outcome 2 is
``A >= 1 and B == 0/1`` with the failure captured verbatim; outcome 3 is
``A == 0``. Nothing here adjudicates which outcome holds -- these tests **pin
what was measured**, so the state cannot silently change class later.

Counting rule for (A), declared rather than left implicit: a screened line counts
iff the identifier **on that line** names non-text-modality data and the line is
either **P**, a formal parameter in a ``def`` signature, or **B**, a control-flow
predicate (``if`` statement or conditional expression) testing that data.
Comments, docstrings, format strings, assignments, call-site pass-throughs,
assertion messages and import/export lines are non-counting.

(A) is re-derived from anchor *patterns*, never line numbers: the plan's own cites
into ``logit_validation.py`` are ``+28`` stale after a landed increment moved
them, so a line-pinned test would encode a number already known to drift. Line
numbers are reported here and pinned by nothing.

Nothing loads a checkpoint, opens a socket or touches a device: the live
multimodal branch calls a **caller-supplied** ``generate_fn``, so the single
synthetic sample is driven with a fake and the measurement stays on CPU.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

import vllm_neuron.accuracy as _accuracy_pkg
import vllm_neuron.accuracy.testing as _testing
from vllm_neuron.accuracy.logit_validation import logit_validation
from vllm_neuron.accuracy.testing import AssertCloseResult, assert_close

ACCURACY_DIR = Path(_accuracy_pkg.__file__).resolve().parent

#: The screen the enumeration is derived from. Case-insensitive, as run.
SCREEN = re.compile(
    r"multimodal|multi_modal|\bmm_|pixel_values|image|vision|video|audio",
    re.IGNORECASE,
)

#: The screen widened once, to test the base screen for misses. Its only new
#: counting hit is ``**visual_kwargs``; every other added line is
#: "visuali[sz]ation"/plotting noise or an import/export line.
SCREEN_WIDE = re.compile(
    r"multimodal|multi_modal|\bmm_|pixel_values|image|vision|video|audio"
    r"|visual|modality|grid_thw|non_text|encoder_cache",
    re.IGNORECASE,
)

#: (A)'s enumeration, as anchor patterns matched against stripped source lines.
#: ``kind`` is the classification the counting rule assigns. Each anchor must
#: match **exactly once** in its file, so a duplicated line cannot inflate (A)
#: and a deleted line cannot silently shrink it.
_LV = "logit_validation.py"
_ECA = "encoder_cache_analysis.py"
A_ANCHORS: tuple[tuple[str, str, str, str], ...] = (
    (_LV, "P", r"^multimodal_inputs: Optional\[List\[dict\]\] = None,$",
     "logit_validation() param -- per-batch mm dicts"),
    (_LV, "B", r"^if multimodal_inputs is not None:$",
     "dispatch to the multimodal generate_fn call"),
    (_LV, "P", r"^prompts_multimodal_inputs: Optional\[List\[List\[dict\]\]\] = None,$",
     "multi_prompt_logit_validation() param"),
    (_LV, "B", r"^if prompts_multimodal_inputs is not None$",
     "conditional expression slicing one prompt's mm inputs"),
    (_LV, "B", r"^if mm_inputs is not None:$",
     "dispatch replicating mm inputs to batch size"),
    (_ECA, "P", r"^pixel_values: torch\.Tensor,$",
     "extract_hf_encoder_outputs() param -- pixel values"),
    (_ECA, "P", r"^image_grid_thw: torch\.Tensor = None,$",
     "extract_hf_encoder_outputs() param -- per-image grid"),
    (_ECA, "P", r"^\*\*visual_kwargs,$",
     "extract_hf_encoder_outputs() param -- vision kwargs"),
    (_ECA, "B", r"^if visual is None:$",
     "predicate on the vision-submodule lookup"),
    (_ECA, "B", r"^if image_grid_thw is not None:$",
     "dispatch building the vision encoder's grid_thw kwarg"),
)


def _acr_producers() -> dict[str, Any]:
    """Every ``testing.py`` function annotated ``-> AssertCloseResult``.

    Discovered by screening the module's own annotations, not named by hand: a
    producer added later is picked up without editing this file.
    """
    found: dict[str, Any] = {}
    for name, obj in vars(_testing).items():
        if not inspect.isfunction(obj) or obj.__module__ != _testing.__name__:
            continue
        annotation = inspect.signature(obj).return_annotation
        rendered = annotation if isinstance(annotation, str) else getattr(
            annotation, "__name__", repr(annotation)
        )
        if rendered == "AssertCloseResult":
            found[name] = obj
    return found

_SINK = Path(
    os.environ.get("VLLM_NEURON_INC006_G15_JSON")
    or Path(tempfile.gettempdir()) / "vllm_neuron_inc006_g15_readings.json"
)
_RECORD: dict[str, Any] = {}
_SINK.write_text("{}\n")  # truncate a stale run's values


def _rec(**values: Any) -> None:
    """Persist as we go, so a failing conjunct still leaves its readings behind."""
    _RECORD.update(values)
    _SINK.write_text(json.dumps(_RECORD, indent=2, sort_keys=True, default=str) + "\n")


def _py_sources() -> list[Path]:
    return sorted(ACCURACY_DIR.rglob("*.py"))


def _screen_hits(pattern: re.Pattern[str]) -> int:
    return sum(
        1
        for path in _py_sources()
        for line in path.read_text().splitlines()
        if pattern.search(line)
    )


def _synthetic_sample() -> dict[str, torch.Tensor]:
    """One synthetic multimodal sample. Zero checkpoints, zero network."""
    return {"pixel_values": torch.zeros(1, 3, 4, 4)}


def test_a_non_text_parameters_and_branches_are_enumerated() -> None:
    """(A) -- re-derived at HEAD, reported with ``file:line``, never pinned.

    The assertion is the outcome-selecting predicate ``A >= 1`` plus per-anchor
    uniqueness. A hand that deletes ``multimodal_inputs`` from the instrument
    fails this test; a hand that moves it by 28 lines does not.
    """
    enumeration: list[str] = []
    missing: list[str] = []
    duplicated: list[str] = []

    for filename, kind, pattern, why in A_ANCHORS:
        path = ACCURACY_DIR / filename
        rx = re.compile(pattern)
        found = [
            lineno
            for lineno, line in enumerate(path.read_text().splitlines(), start=1)
            if rx.match(line.strip())
        ]
        if not found:
            missing.append(f"{filename}: /{pattern}/")
            continue
        if len(found) > 1:
            duplicated.append(f"{filename}:{found} /{pattern}/")
        enumeration.append(f"{kind} {filename}:{found[0]} -- {why}")

    a_count = len(enumeration)
    _rec(
        interpreter=sys.executable,
        accuracy_dir=str(ACCURACY_DIR),
        accuracy_py_files=len(_py_sources()),
        a_count=a_count,
        a_enumeration=enumeration,
        a_missing=missing,
        a_duplicated=duplicated,
        screen_hits_base=_screen_hits(SCREEN),
        screen_hits_wide=_screen_hits(SCREEN_WIDE),
    )
    print("(A) enumeration, re-derived at HEAD:")
    for row in enumeration:
        print("   ", row)
    print(f"(A) = {a_count}")

    assert not missing, f"(A) anchors that no longer match: {missing}"
    assert not duplicated, f"(A) anchors matching more than once: {duplicated}"
    assert a_count >= 1, "(A) == 0 -- this is outcome 3, RULED OUT"


def test_control_the_assertcloseresult_detector_fires_on_a_real_one() -> None:
    """Control for (B). Without it, ``B == 0`` could be a broken import.

    ``assert_close`` on a tensor against itself is the cheapest object that must
    yield an ``AssertCloseResult``. If this arm ever fails, (B)'s zero says
    nothing about the multimodal path.
    """
    tensor = torch.zeros(4)
    control = assert_close(tensor, tensor, name="inc006-control")
    _rec(
        control_type=f"{type(control).__module__}.{type(control).__qualname__}",
        control_is_assertcloseresult=isinstance(control, AssertCloseResult),
    )
    assert isinstance(control, AssertCloseResult), (
        f"control: assert_close returned {type(control)!r}, not AssertCloseResult -- "
        "(B)'s detector is broken and (B) is uninterpretable"
    )


def test_b_one_synthetic_multimodal_sample_result_type() -> None:
    """(B) -- one synthetic sample, driven through the live multimodal branch.

    The captured signature is PINNED: type ``builtins.bool``, and not an
    ``AssertCloseResult``. That is the middle branch's own requirement -- if the
    instrument later starts returning a result object, this test fails and the
    class change is visible instead of silent.
    """
    torch.manual_seed(0)
    vocab = 1024
    expected_logits = torch.randn(1, 1, vocab)
    sample = _synthetic_sample()
    calls: list[dict[str, Any]] = []

    def fake_generate_fn(input_ids, multimodal_inputs=None):
        calls.append(
            {
                "mm_kwarg_seen": multimodal_inputs is not None,
                "mm_is_the_same_object": multimodal_inputs is not None
                and multimodal_inputs[0]["pixel_values"] is sample["pixel_values"],
            }
        )
        return expected_logits.clone()

    returned = logit_validation(
        [[1, 2, 3]],
        fake_generate_fn,
        expected_logits,
        multimodal_inputs=[sample],
        test_device="cpu",
        colorize=False,
        visualize=False,
    )

    returned_type = f"{type(returned).__module__}.{type(returned).__qualname__}"
    is_acr = isinstance(returned, AssertCloseResult)
    b_count = 1 if is_acr else 0
    _rec(
        b_samples_driven=1,
        b_count=b_count,
        b_verbatim_repr=repr(returned),
        b_returned_type=returned_type,
        b_is_assertcloseresult=is_acr,
        b_generate_fn_calls=calls,
    )
    print(f"(B) = {b_count}/1 -- returned {returned_type} repr={returned!r}")

    assert calls and calls[0]["mm_kwarg_seen"] is True, (
        f"the multimodal branch was not taken: generate_fn calls={calls}"
    )
    assert calls[0]["mm_is_the_same_object"] is True, (
        "the synthetic sample did not reach generate_fn unchanged"
    )
    assert returned_type == "builtins.bool", (
        f"captured signature changed: the instrument returned {returned_type}, "
        "not the measured builtins.bool"
    )
    assert not is_acr, (
        "captured signature changed: the instrument now returns an "
        "AssertCloseResult -- (B) is no longer 0/1 and G15 needs re-adjudicating"
    )


def test_negative_control_text_only_input_skips_the_multimodal_branch() -> None:
    """The dispatch at the multimodal branch is real, not always-on.

    With ``multimodal_inputs=None`` the instrument must call ``generate_fn``
    *without* the keyword. A branch that passed the keyword unconditionally
    would make the (B) reading say nothing about non-text handling.
    """
    torch.manual_seed(0)
    expected_logits = torch.randn(1, 1, 1024)
    seen: list[dict[str, Any]] = []

    def fake_generate_fn(input_ids, **kwargs):
        seen.append(dict(kwargs))
        return expected_logits.clone()

    logit_validation(
        [[1, 2, 3]],
        fake_generate_fn,
        expected_logits,
        test_device="cpu",
        colorize=False,
        visualize=False,
    )
    _rec(negative_control_kwargs=seen)
    assert seen == [{}], (
        f"text-only path passed keyword arguments to generate_fn: {seen}"
    )


def test_assertcloseresult_producers_accept_no_non_text_parameter() -> None:
    """What outcome 2's "instrument work" actually is, stated as a measurement.

    Every producer of an ``AssertCloseResult`` is a bare two-way tensor
    comparator: no parameter of any of them names non-text data. So the
    multimodal path and the ``AssertCloseResult`` path do not meet anywhere in
    ``accuracy/`` at this HEAD.
    """
    producers = _acr_producers()
    readings: dict[str, list[str]] = {}
    offenders: list[str] = []
    for name, fn in producers.items():
        params = list(inspect.signature(fn).parameters)
        readings[name] = params
        offenders += [f"{name}({p})" for p in params if SCREEN_WIDE.search(p)]

    _rec(acr_producer_parameters=readings, acr_producer_non_text_params=offenders)
    print(f"AssertCloseResult producers: {readings}")
    assert producers, (
        "no function in testing.py is annotated -> AssertCloseResult; the "
        "discovery screen found nothing and this reading is uninterpretable"
    )
    assert not offenders, (
        "an AssertCloseResult producer now takes a non-text parameter "
        f"({offenders}) -- the two paths have met and G15 needs re-adjudicating"
    )
