"""``inc-glm53f-073`` -- G15 / M5: the multimodal-to-``AssertCloseResult`` join.

``inc-glm53f-006`` measured its pair of counts as **(A >= 1, B == 0/1)** -- the
multimodal path exists and accepts an image-bearing input, but driving one
synthetic sample through it yields ``builtins.bool``, not an
``AssertCloseResult``. That is outcome 2, *"PATH PRESENT, COMPARISON DOES NOT
YIELD A RESULT"*, and the two result families were disjoint by construction:
nothing turned a multimodal validation run into a result object. This increment
builds the missing adapter, and these five conjuncts measure it.

**What this file is NOT.** This increment builds an **instrument**. It authors
no criterion, no tolerance and no threshold, and it satisfies and claims **no
registered acceptance type**. ``G15`` is registered as a **precondition, not a
criterion** (RG-P5), so no conjunct here is a gate reading. Conjunct 4 makes the
no-tolerance boundary mechanical rather than a promise.

**Five conjuncts, five tests, no decorator-generated cases** -- so the collected
item count is derivable before the run rather than guessed: **5**.

1. The join returns the right family, by class identity, not by name string.
2. ``inc-glm53f-006``'s own predicate B flips to ``1/1`` **through the join**,
   while the **direct** path's ``builtins.bool`` reading stays exactly as
   ``-006`` measured it.
3. A counted **negative control**: the existing return contract is unchanged.
   The gap was closed by *adding* an adapter, never by widening a union every
   caller unpacks.
4. A counted **zero**: no numeric literal in the adapter's own source, and its
   tolerances demonstrably come from the registered dtype pair.
5. An uncomparable pair **raises**, carrying a verbatim signature and the
   offending type, instead of answering with a bare falsey value.

Nothing loads a checkpoint, opens a socket, or touches a device: the synthetic
image-bearing sample is built in-test and driven through a caller-supplied
``generate_fn`` fake, exactly as ``-006``'s landed screen does. No module here
is imported for its side effects, and nothing outside ``accuracy/`` is imported
at all.
"""

import ast
import inspect
import textwrap
import typing
from typing import Any, Dict, List, Tuple

import pytest
import torch

import vllm_neuron.accuracy.logit_validation as _logit_validation
import vllm_neuron.accuracy.testing as _testing
from vllm_neuron.accuracy.logit_validation import (
    logit_validation,
    multi_prompt_logit_validation,
)
from vllm_neuron.accuracy.testing import (
    AssertCloseResult,
    assert_close_logit_pair,
    resolve_dtype_tolerance,
)

#: Vocabulary width for the synthetic logits. Small enough to stay instant.
VOCAB = 1024

#: The one sample index the single-element batch exposes.
SAMPLE_INDEX = 0


def _synthetic_image_bearing_sample() -> Dict[str, torch.Tensor]:
    """One synthetic image-bearing sample. Zero checkpoints, zero network.

    Deliberately the same shape ``-006``'s landed screen uses, so conjunct 2 is
    re-evaluating *the same predicate over the same sample*, not a friendlier
    one.
    """
    return {"pixel_values": torch.zeros(1, 3, 4, 4)}


def _drive_one_sample_through_the_join() -> Dict[str, Any]:
    """Drive one synthetic image-bearing sample and route the pair to the adapter.

    Returns every reading both conjunct 1 and conjunct 2 need, measured in one
    pass so the two tests cannot disagree about what the instrument did:

    * ``returned`` / ``returned_type`` -- what the **direct** call answered.
    * ``exposed`` -- the per-sample pairs the sink received.
    * ``joined`` -- what the adapter produced from the exposed pair.
    * ``generate_calls`` -- proof the live non-text branch was taken.
    """
    torch.manual_seed(0)
    expected_logits = torch.randn(1, 1, VOCAB)
    sample = _synthetic_image_bearing_sample()
    generate_calls: List[Dict[str, Any]] = []
    exposed: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def fake_generate_fn(input_ids, **kwargs):
        seen = kwargs.get("multimodal_inputs")
        generate_calls.append(
            {
                "kwarg_seen": seen is not None,
                "is_the_same_object": seen is not None
                and seen[SAMPLE_INDEX]["pixel_values"] is sample["pixel_values"],
            }
        )
        return expected_logits.clone()

    def collect(sample_index, actual, expected):
        exposed[sample_index] = (actual, expected)

    returned = logit_validation(
        [[1, 2, 3]],
        fake_generate_fn,
        expected_logits,
        multimodal_inputs=[sample],
        logit_pair_sink=collect,
        test_device="cpu",
        colorize=False,
        visualize=False,
    )

    actual, expected = exposed[SAMPLE_INDEX]
    joined = assert_close_logit_pair(actual, expected, name="inc073-join")
    return {
        "returned": returned,
        "returned_type": f"{type(returned).__module__}.{type(returned).__qualname__}",
        "exposed": exposed,
        "actual": actual,
        "expected": expected,
        "joined": joined,
        "generate_calls": generate_calls,
    }


def _annotation_tree(annotation: Any) -> List[Any]:
    """Every node of an annotation's type tree, root included.

    ``typing.get_args`` only peels one layer, and the union this control guards
    nests ``AssertCloseResult`` could hide inside a ``Tuple`` member. So the
    whole tree is walked rather than the top level screened.
    """
    nodes = [annotation]
    for arg in typing.get_args(annotation):
        nodes.extend(_annotation_tree(arg))
    return nodes


def _numeric_literals(function: Any) -> List[Any]:
    """Every numeric literal in ``function``'s own source.

    ``bool`` is excluded deliberately and explicitly: ``True``/``False`` are
    ``int`` subclasses in Python, and a boolean flag default is not a tolerance.
    Everything else numeric counts, so the conjunct cannot be satisfied by
    dressing a tolerance up as an ``int``.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ]


def test_c1_the_join_returns_the_assertcloseresult_family() -> None:
    """Conjunct 1 -- ``1/1``. Asserted by class identity, never by name string.

    A name-string comparison would pass against any class that happens to be
    called ``AssertCloseResult``, including a look-alike defined in this file.
    So the assertion is ``isinstance`` against the imported class object, plus
    an identity check that the class the adapter returned is the very object
    ``testing.py`` defines.
    """
    reading = _drive_one_sample_through_the_join()
    joined = reading["joined"]

    print(f"(C1) adapter returned {type(joined)!r}")
    print(f"(C1) exposed pair shapes: {tuple(reading['actual'].shape)}")

    assert isinstance(joined, AssertCloseResult), (
        f"the adapter returned {type(joined)!r}, not an AssertCloseResult -- "
        "the join does not produce the required family"
    )
    assert type(joined) is _testing.AssertCloseResult, (
        f"the returned class {type(joined)!r} is not the identical class object "
        "testing.py defines -- a same-named look-alike would not close G15"
    )
    assert joined.allclose is True, (
        "the synthetic pair is a tensor against its own clone and must compare "
        f"equal; got allclose={joined.allclose!r}"
    )


def test_c2_predicate_b_flips_to_one_of_one_through_the_join() -> None:
    """Conjunct 2 -- ``1/1``. The increment's reason for existing.

    ``-006`` measured B as ``0/1`` on the **direct** path. This re-evaluates the
    same predicate -- *"the count of synthetic multimodal samples for which the
    instrument returns an ``AssertCloseResult``"* -- **through the join**, and it
    must now read ``1/1``.

    The direct reading is asserted to be **unchanged** in the same breath. Both
    hold at once, and that is the point: the join is a new path, not a mutation
    of the measured one. ``-006``'s landed screen pins the direct reading in a
    file this increment does not touch, and this conjunct agrees with it here.
    """
    reading = _drive_one_sample_through_the_join()
    calls = reading["generate_calls"]
    exposed = reading["exposed"]

    b_samples_driven = 1
    b_count_through_the_join = sum(
        1
        for pair in exposed.values()
        if isinstance(assert_close_logit_pair(*pair, name="inc073-b"), AssertCloseResult)
    )
    print(
        f"(C2) B through the join = {b_count_through_the_join}/{b_samples_driven}; "
        f"direct path still returns {reading['returned_type']}"
    )

    assert calls and calls[SAMPLE_INDEX]["kwarg_seen"] is True, (
        f"the live non-text branch was not taken: generate_fn calls={calls}"
    )
    assert calls[SAMPLE_INDEX]["is_the_same_object"] is True, (
        "the synthetic sample did not reach generate_fn unchanged, so this "
        "reading would not be about the multimodal path"
    )
    assert len(exposed) == b_samples_driven, (
        f"expected the pair of exactly {b_samples_driven} sample(s) to be "
        f"exposed; got {sorted(exposed)}"
    )
    assert b_count_through_the_join == b_samples_driven, (
        f"(B) through the join = {b_count_through_the_join}/{b_samples_driven}, "
        "not 1/1 -- the join does not close G15's instrument gap"
    )
    assert reading["returned_type"] == "builtins.bool", (
        "the DIRECT path's captured signature moved: it returned "
        f"{reading['returned_type']}, not the builtins.bool -006 measured. The "
        "join must add a path, never mutate the measured one"
    )


def test_c3_the_existing_return_contract_is_unchanged() -> None:
    """Conjunct 3 -- ``1/1``, a counted NEGATIVE CONTROL, not ceremony.

    The cheapest wrong fix is to make ``logit_validation`` return an
    ``AssertCloseResult`` in the non-text case. That would satisfy conjuncts 1
    and 2 and silently break every existing caller, which unpacks the current
    union. So the control is mechanical, in three arms:

    1. ``AssertCloseResult`` is absent from ``logit_validation``'s whole return
       type tree, read via ``typing.get_type_hints`` on the **live** function
       object -- the module has no ``from __future__ import annotations``, so
       the hints are live classes and the check is against real types.
    2. The same for the multi-prompt entry point, a second result family that
       must not have been widened either.
    3. **No function in the validation module is annotated
       ``-> AssertCloseResult`` at all** -- the mirror of the adapter-side
       discovery screen, applied to the producing side. This forbids widening
       anywhere in the module, not merely at the two cited entry points.
    """
    hints = typing.get_type_hints(logit_validation)
    tree = _annotation_tree(hints["return"])
    multi_hints = typing.get_type_hints(multi_prompt_logit_validation)
    multi_tree = _annotation_tree(multi_hints["return"])

    module_source = ast.parse(
        inspect.getsource(_logit_validation)
    )
    acr_annotated = [
        node.name
        for node in ast.walk(module_source)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.returns is not None
        and "AssertCloseResult" in ast.unparse(node.returns)
    ]

    print(f"(C3) logit_validation return: {hints['return']}")
    print(f"(C3) multi_prompt return: {multi_hints['return']}")
    print(f"(C3) functions annotated -> AssertCloseResult in the module: {acr_annotated}")

    assert not any(node is AssertCloseResult for node in tree), (
        "AssertCloseResult entered logit_validation()'s return type tree: "
        f"{hints['return']}. The union every existing caller unpacks was "
        "widened -- forbidden, not merely undesirable"
    )
    assert not any(node is AssertCloseResult for node in multi_tree), (
        "AssertCloseResult entered multi_prompt_logit_validation()'s return "
        f"type tree: {multi_hints['return']}"
    )
    assert acr_annotated == [], (
        "the validation module now declares AssertCloseResult producer(s) "
        f"{acr_annotated} -- the gap must be closed by the adapter, not by "
        "turning the validation path into a producer"
    )


def test_c4_the_adapter_carries_zero_tolerance_literals() -> None:
    """Conjunct 4 -- a counted **ZERO**, with provenance measured behaviourally.

    A tolerance literal in this adapter would be this increment authoring a
    comparator, which is exactly what it must not do. Two arms:

    1. The count of numeric literals in the adapter's own source is **0**. Not
       "no float that looks like a tolerance" -- zero numeric literals at all,
       which leaves nothing to argue about.
    2. Provenance is a **measurement with a falsification arm**, not a text
       match. A text-only check would pass on a source that *calls*
       ``resolve_dtype_tolerance`` and then ignores the answer. So the pair
       compared here is deliberately **not bit-equal** -- a bit-equal pair
       short-circuits before any tolerance is consulted, which would make this
       arm agree vacuously -- and three readings are taken: defaulted, the
       registered pair passed explicitly, and a deliberately **wrong** pair.
       The first two must agree and the third must disagree. Without the third
       reading, agreement would prove nothing.
    """
    literals = _numeric_literals(assert_close_logit_pair)
    source = inspect.getsource(assert_close_logit_pair)

    torch.manual_seed(0)
    expected = torch.randn(4, VOCAB)
    # Not a clone: a bit-equal pair never reaches tolerance resolution at all.
    actual = expected + torch.ones_like(expected)
    registered_rtol, registered_atol = resolve_dtype_tolerance(expected.dtype)
    defaulted = assert_close_logit_pair(actual, expected, name="inc073-defaulted")
    explicit = assert_close_logit_pair(
        actual,
        expected,
        rtol=registered_rtol,
        atol=registered_atol,
        name="inc073-explicit",
    )
    wrong_rtol = assert_close_logit_pair(
        actual, expected, rtol=1.0e3, atol=registered_atol, name="inc073-wrong-rtol"
    )
    wrong_atol = assert_close_logit_pair(
        actual, expected, rtol=registered_rtol, atol=2.0, name="inc073-wrong-atol"
    )

    print(f"(C4) numeric literals in the adapter's source = {len(literals)} {literals}")
    print(
        f"(C4) registered pair for {expected.dtype}: "
        f"rtol={registered_rtol} atol={registered_atol}"
    )
    print(
        f"(C4) allclose defaulted={defaulted.allclose} explicit={explicit.allclose} "
        f"wrong_rtol={wrong_rtol.allclose}"
    )
    print(
        f"(C4) max_rel_error defaulted={defaulted.max_rel_error} "
        f"explicit={explicit.max_rel_error} wrong_atol={wrong_atol.max_rel_error}"
    )

    assert len(literals) == 0, (
        f"the adapter's source carries {len(literals)} numeric literal(s) "
        f"{literals} -- a tolerance literal here is this increment authoring a "
        "comparator (P9)"
    )
    assert "resolve_dtype_tolerance(" in source, (
        "the adapter no longer names resolve_dtype_tolerance, so its tolerance "
        "provenance is no longer readable in its own source"
    )
    assert defaulted.allclose == explicit.allclose, (
        "the adapter's defaulted rtol does not match the registered value for "
        f"{expected.dtype}: defaulted allclose={defaulted.allclose}, "
        f"explicit={explicit.allclose}"
    )
    assert defaulted.max_rel_error == explicit.max_rel_error, (
        "the adapter's defaulted atol does not match the registered value for "
        f"{expected.dtype}: defaulted max_rel_error={defaulted.max_rel_error}, "
        f"explicit={explicit.max_rel_error}"
    )
    # Falsification arms: if the readings above could not disagree, their
    # agreement would be worth nothing.
    assert wrong_rtol.allclose != defaulted.allclose, (
        "a deliberately wrong rtol produced the same verdict as the registered "
        "one, so this conjunct's agreement is vacuous and proves no provenance"
    )
    assert wrong_atol.max_rel_error != defaulted.max_rel_error, (
        "a deliberately wrong atol produced the same max_rel_error as the "
        "registered one, so this conjunct's agreement is vacuous"
    )


def test_c5_an_uncomparable_pair_raises_instead_of_answering_falsey() -> None:
    """Conjunct 5 -- ``1/1``. It fails loudly, never silently.

    The whole defect this increment repairs was *a comparison that answered a
    plain ``bool``*. An adapter that answered ``False`` for a pair it could not
    compare would reintroduce the same ambiguity one layer up: the caller could
    not tell "compared, and they differ" from "could not compare at all".

    So every uncomparable arm must raise, carrying the verbatim signature and
    the offending type. The final arm is the contrast that gives the conjunct
    its meaning: a pair that IS comparable but compares badly returns a result
    with ``allclose`` false -- it does not raise.
    """
    signature = "uncomparable logit pair"
    torch.manual_seed(0)
    reference = torch.randn(4, VOCAB)

    with pytest.raises(TypeError) as not_a_tensor:
        assert_close_logit_pair([reference], reference, name="inc073-c5-type")
    with pytest.raises(ValueError) as shape_mismatch:
        assert_close_logit_pair(reference[:2], reference, name="inc073-c5-shape")
    with pytest.raises(ValueError) as dtype_mismatch:
        assert_close_logit_pair(
            reference.to(torch.bfloat16), reference, name="inc073-c5-dtype"
        )

    raised = {
        "not_a_tensor": str(not_a_tensor.value),
        "shape_mismatch": str(shape_mismatch.value),
        "dtype_mismatch": str(dtype_mismatch.value),
    }
    for arm, message in raised.items():
        print(f"(C5) {arm}: {message}")
        assert message.startswith(signature), (
            f"the {arm} arm raised without the verbatim signature "
            f"{signature!r}: {message!r}"
        )
        assert "offending_type=" in message, (
            f"the {arm} arm raised without naming the offending type: {message!r}"
        )
    assert "<class 'list'>" in raised["not_a_tensor"], (
        "the non-tensor arm did not carry the offending type verbatim: "
        f"{raised['not_a_tensor']!r}"
    )
    assert "torch.bfloat16" in raised["dtype_mismatch"], (
        "the dtype arm did not carry the offending dtype verbatim: "
        f"{raised['dtype_mismatch']!r}"
    )

    comparable_but_failing = assert_close_logit_pair(
        reference + torch.full_like(reference, float(VOCAB)),
        reference,
        name="inc073-c5-contrast",
    )
    print(f"(C5) comparable-but-failing pair returned {comparable_but_failing!r}")
    assert isinstance(comparable_but_failing, AssertCloseResult), (
        "a comparable pair must still return a result, even when it compares "
        f"badly; got {type(comparable_but_failing)!r}"
    )
    assert comparable_but_failing.allclose is False, (
        "the contrast arm was supposed to compare badly, so the raise-versus-"
        "return distinction is untested; got "
        f"allclose={comparable_but_failing.allclose!r}"
    )
