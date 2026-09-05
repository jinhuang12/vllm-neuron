# SPDX-License-Identifier: Apache-2.0
"""``platform.py`` hybrid-cache configuration (``inc-glm53f-020``).

Subject: the hybrid branch of ``NeuronPlatform.check_and_update_config``
(``vllm_neuron/vllm/platform.py``). The increment block declares the range
**308-443** against the PIN; at the parent ``7ca17aa4`` the same method reads
``@classmethod`` **316** / ``def`` **317**, bound ``validate_request`` **453**
(the ``+8`` that ``-075``/``-074`` landed above it). The shift is NOT uniform
across the block's four declared ranges -- ``-019``'s hunk landed between the
pairs, so ``get_attn_backend_cls`` and ``support_hybrid_kv_cache`` read at
``+52``. Every cite here was re-measured at the parent rather than shifted by a
single constant.

**Five items, one per counted conjunct, no ``parametrize``** (test-layout rule
6), so the item count stays derivable before the run. The block's conjuncts are
numbered ``1``, ``2``, ``4``, ``5`` and ``6``: **number 3 is RETIRED at plan
revision 28 and deliberately NOT reused**, so every label here keeps resolving
to the criterion it was measured against.

* item 1 -- conjunct 1: the hybrid path is ENGAGED, a counted differential,
  ``1/1`` engaged and ``1/1`` not-engaged, read off the branch's own resolved
  output rather than off a mock.
* item 2 -- conjunct 2: the resolved hybrid block size equals the declared
  value **128**, exact equality, ``1/1``, **in the knob-UNSET case** (revision
  28 scopes the domain, because conjunct 6 makes the operator knob a second
  source for the number), **and the value SURVIVES to the allocator** -- the
  latch is codified by the block, not chosen here.
* item 3 -- conjunct 4: **0** of the pin's **5** existing archs change resolved
  config, with the D1.5 control that moves the zero.
* item 4 -- conjunct 5: **2/2** counted negative arms -- ``TP != 64`` and
  ``KV dtype != bf16`` each raise, each raise happens BEFORE any block-size
  assignment, and each message names the OPERATIVE value, which is the decided
  ``128`` in both arms because both are declared against the unset knob.
* item 5 -- conjunct 6: the operator override is HONOURED only within the
  constraints ``DECISIONS.md`` section 6 registers -- **2/2** arms,
  one in each direction, with the D1.5 control that drops the validation and
  moves arm A's raise count to ``0/1``.

**``inc-glm53f-081`` ADDS FOUR ITEMS AT THE END OF THIS FILE, one per NEW arm,
so the derivable collected count for this file is now NINE.** `-020`'s five
items above, the helpers they call and the constants they read are **re-run
whole and not edited** -- they are `-081`'s own acceptance, so `-081` cannot
redden one without reddening itself. The four new items are conjunct **(a)** the
automatic engagement at the registered TP degree, conjunct **(b) arm 2** the
bfloat16 guard reached on the DEFAULT path, and conjuncts **(d2)** and **(d3)**
the quiet-and-unchanged and the audible halves of the same non-engagement. They
bring their own level-aware log context because ``_RecordingHandler`` above
keeps messages without their level and ``(d3)`` counts records BY LEVEL.
`-081`'s subject is the decision ABOVE the limb -- whether to enter it -- so it
moves no registered value, no guard, no message and no derivation, and it adds
no ``parametrize``.

**THE ``inc-glm53f-081`` REPAIR (finding ``F-B29-01``) ADDS A TENTH ITEM at the
end, so the derivable collected count for this file is now TEN.** ``(d3)`` above
reads the page out of the landed method, but every config in this file supplied
no block size, so it only ever read the unlatched half of the domain -- and the
warning named ``32`` unconditionally, which is wrong for an operator who passed
``--block-size``. The tenth item builds the latched half and reads the same
agreement there. It adds one input constant of its own and no registered value.

**Conjunct 3 (the vision limb) is DEFERRED to M5 at plan revision 28, and this
file adds NO item for the retired number.** The deferral rests on measurements,
not on preference, and they are recorded once rather than twice: the ground is
``increments/evidence-020.md`` section 5, the disposition is the plan's block
plus ``RG-20``, and the limb needs no ``platform.py`` change at all -- what is
missing is model-side.

**Parent readings, measured on the instrument BEFORE a source line was written**
(``increments/probe-020-parent-readings.py``, run at the unmodified parent
``7ca17aa4``, ``PROBE_EXIT_CODE=0``):

* Conjuncts 1 and 2 are **FALSE at the parent**: across the declared range the
  parent carries **0** ``block_size`` assignments and exactly **1**
  ``block_size`` occurrence -- the comment at ``platform.py:365`` -- and the
  resolved block size reads the ``CacheConfig`` default **16**, not 128.
  (The block's own non-vacuity control for that zero: the same pattern returns
  **4** assignment sites file-wide, measured at ``:187``, ``:228``, ``:244``,
  ``:661``.)
* Conjunct 5's two arms are **FALSE at the parent**: ``TP=32`` returns without
  raising and an ``fp16`` KV cache returns without raising.
* Conjunct 4 reads **0 of 5** changed at the parent -- green at the parent and
  declared so, because its role is a regression guard on the archs this
  increment does not target.
* Conjunct 6's two arms are **FALSE at the parent** as well
  (``increments/probe-020b-conjunct6-parent-readings.py``, run at the same
  unmodified parent, ``PROBE_EXIT_CODE=0``): the parent assigns no block size
  and validates nothing, so an operator value below the floor is neither
  refused (it resolves to the default **16**, no raise) nor is a legal
  non-default operator value honoured (it also resolves to **16**).

**The downstream clobber, why items 2 and 5 both measure survival** (probe
reading R6). ``check_and_update_config`` runs from
``VllmConfig.__post_init__``; the fork's ``update_block_size_for_backend`` runs
**later**, from the executor, and hard-sets **32** unless
``cache_config.user_specified_block_size`` is set -- measured at the parent, a
bare ``128`` reads back **32** with the latch clear and **128** with it set. The
latch is codified by the block at revision 28. That method itself is
byte-unchanged, and nothing here calls ``Platform._align_hybrid_block_size`` or
sets ``cache_config.mamba_page_size_padded``.

The value **128** is cited to ``DECISIONS.md`` section 6 (verbatim
user answer "128 (Recommended)"), never to the round-trip sentinel in
``test/vllm_neuron/model/test_neuron_config_glm5next.py:42`` -- the numeric
equality with that sentinel is a disclosed coincidence and is not corroboration.
**Section 6's two constraint values live in exactly ONE place, the branch, each
cited there to section 6.** This file restates neither: item 5 asserts
behaviour and message content, so a re-derivation moves section 6 and not this
file. The one number item 5 carries is its own arm-B input.

No item constructs an engine, loads a checkpoint, reaches a network or touches a
device. Nothing under any compile cache is read, written or relocated (P2).
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import textwrap
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.vllm import platform as platform_module
from vllm_neuron.vllm.platform import NeuronPlatform

# The user-decided hybrid block size. Authority: DECISIONS.md section
# 6 -- NOT the round-trip sentinel in test_neuron_config_glm5next.py, whose
# numeric equality with this value is a disclosed coincidence.
DECIDED_HYBRID_BLOCK_SIZE = 128

# The two registered preconditions the value is valid under.
REGISTERED_TP_DEGREE = 64
REGISTERED_KV_DTYPE = torch.bfloat16

# One off-value per negative arm, recorded so the arm says which it used.
OFF_VALUE_TP_DEGREE = 32
OFF_VALUE_MODEL_DTYPE = torch.float16

# Conjunct 6's two arm inputs. Both are INPUTS, not registered values: the
# constraints themselves live in the branch, cited to DECISIONS section 6, and
# are restated nowhere here.
#
# Arm A's off-value is chosen to isolate ONE constraint -- it satisfies the
# granularity multiple and violates only the state-page floor, so the arm cannot
# pass by tripping the wrong check.
OFF_VALUE_OPERATOR_BLOCK_SIZE_BELOW_FLOOR = 64
# Arm B's value is the smallest legal operator value that is NOT the decided
# one, so "honoured" is proven on a value the branch could not have produced by
# falling back.
LEGAL_NON_DEFAULT_OPERATOR_BLOCK_SIZE = 192
# Recorded, adding no conjunct: above the floor, off the granularity multiple.
OFF_VALUE_OPERATOR_BLOCK_SIZE_OFF_GRANULARITY = 200

# The branch delimits its section-6 validation with these two markers so the
# D1.5 control can remove exactly that region from the REAL source instead of
# re-implementing the method.
VALIDATION_REGION_BEGIN = "-- BEGIN section-6 constraint validation --"
VALIDATION_REGION_END = "-- END section-6 constraint validation --"
# Anchors unique to the two REFUSAL sentences. The constraint names themselves
# occur in the branch's explanatory comment as well, so anchoring on them would
# fire on prose the control keeps on purpose.
FLOOR_REFUSAL_ANCHOR = "is below the registered KDA"
GRANULARITY_REFUSAL_ANCHOR = "is not a multiple of the "

HYBRID_MARKER = "Hybrid KDA/DSA KV cache enabled"

# The campaign's pinned checkpoint config, landed by inc-glm53f-008. Resolved
# off __file__ so the read cannot depend on the invocation's cwd.
FIXTURE_CONFIG = (
    Path(__file__).resolve().parent / "model" / "glm5_next" / "fixtures" / "config.json"
)

# Class attributes check_and_update_config writes. Saved and restored around
# every item so no item can read another item's residue.
_MUTATED_CLASS_ATTRS = (
    "_enable_structured_outputs",
    "_max_embeds_per_image",
    "_termination_timeout_patched",
)


def _record(label: str, value: object) -> None:
    """Emit an instrument-world reading into the run's own transcript."""
    warnings.warn(f"RECORDED {label}={value!r}", stacklevel=2)


@contextlib.contextmanager
def _isolated_platform_class_state():
    saved = {name: getattr(NeuronPlatform, name) for name in _MUTATED_CLASS_ATTRS}
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(NeuronPlatform, name, value)


class _RecordingHandler(logging.Handler):
    """Collects formatted records off the module's own logger object."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _capture_platform_log():
    """Attach to ``platform.logger`` directly.

    Directly, not through ``caplog``: ``caplog``'s handler lives on the root
    logger, so a propagation setting anywhere in vLLM's logging configuration
    would make the differential read 0 for a reason that has nothing to do with
    the code under test.
    """
    handler = _RecordingHandler()
    target = platform_module.logger
    previous_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)


def _hybrid_records(handler: _RecordingHandler) -> int:
    return sum(1 for message in handler.messages if HYBRID_MARKER in message)


def _fixture_hf_config(*, with_vision: bool):
    """An ``hf_config`` stand-in built from the PINNED fixture bytes.

    ``with_vision=False`` strips ``vision_config`` so the hybrid items resolve
    through the hybrid branch alone and never enter
    ``_resolve_vision_auto_config`` -- which keeps each item's certifying
    component the one its conjunct names.
    """
    assert FIXTURE_CONFIG.is_file(), f"pinned fixture unreachable: {FIXTURE_CONFIG}"
    raw = json.loads(FIXTURE_CONFIG.read_text())
    hf = SimpleNamespace(
        architectures=list(raw["architectures"]),
        quantization_config=dict(raw["quantization_config"]),
        text_config=SimpleNamespace(**raw["text_config"]),
    )
    if with_vision:
        hf.vision_config = SimpleNamespace(**raw["vision_config"])
    return hf


def _build_config(
    *,
    arch: str | None = None,
    neuron_config: dict | None = None,
    tp: int = REGISTERED_TP_DEGREE,
    model_dtype: torch.dtype = REGISTERED_KV_DTYPE,
    cache_dtype: str = "auto",
    max_model_len: int = 8192,
    block_size: int | None = None,
):
    """A config carrying REAL vLLM ``CacheConfig``/``ParallelConfig``.

    The three fields the conjuncts read (``block_size``,
    ``user_specified_block_size``, ``cache_dtype``) are the vendor's own, with
    the vendor's own defaults and validators, so conjunct 2's exact equality is
    measured against the real field rather than against a stand-in attribute.
    ``model_config`` stays a stand-in: a real ``ModelConfig`` needs a
    checkpoint, and no item here may reach a network or a device.

    ``block_size`` left as ``None`` is the operator supplying none, which is the
    state every item in this file used to build. Passing a value is how the
    operator's own ``--block-size`` is reproduced: the vendor's ``CacheConfig``
    latches ``user_specified_block_size`` on an explicit value, which is the
    half of the domain ``F-B29-01`` found unmeasured.
    """
    from vllm.config.cache import CacheConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.config.scheduler import SchedulerConfig

    hf = _fixture_hf_config(with_vision=False)
    if arch is not None:
        hf.architectures = [arch]

    model_config = SimpleNamespace(
        hf_config=hf,
        multimodal_config=None,
        dtype=model_dtype,
        is_moe=True,
        model_arch_config=SimpleNamespace(num_experts=128),
        max_model_len=max_model_len,
    )
    return SimpleNamespace(
        model_config=model_config,
        additional_config=(
            {} if neuron_config is None else {"neuron_config": dict(neuron_config)}
        ),
        parallel_config=ParallelConfig(tensor_parallel_size=tp),
        scheduler_config=SchedulerConfig(
            max_model_len=max_model_len, is_encoder_decoder=False
        ),
        cache_config=CacheConfig(
            cache_dtype=cache_dtype,
            **({} if block_size is None else {"block_size": block_size}),
        ),
        kv_transfer_config=None,
    )


def _cache_snapshot(cfg) -> dict[str, str]:
    """Every ``cache_config`` field, as the block's before/after dict."""
    return {k: repr(v) for k, v in sorted(cfg.cache_config.__dict__.items())}


def _pin_archs() -> list[str]:
    """The pin's five existing archs, READ from the registry rather than typed.

    D1.3's preferred form: the population is a set the instrument produces, so
    a sixth pin arch widens the denominator instead of being absorbed by a
    literal list. This campaign's own arch is excluded by name.
    """
    from vllm_neuron.model.registry import get_models

    return [
        name
        for name, _ in get_models()
        if name != "Glm5NextForConditionalGeneration"
    ]


def _validation_stripped_variant():
    """The REAL method with ONLY its section-6 validation region removed.

    Conjunct 6's D1.5 control needs the counterfactual "the override is still
    read, but its validation is dropped". This builds that counterfactual from
    the method's own source -- ``inspect.getsource`` of the live function, minus
    the lines between the two markers the branch delimits the region with -- so
    the control measures the real code path with one region excised rather than
    a re-implementation of it, which would only ever test the copy.

    The strip is asserted to be exactly one region, and the result is asserted to
    have lost exactly the two ``raise`` statements and both refusal texts, so a
    control that silently stops stripping fails loudly instead of reporting a
    pass. The anchors are the REFUSAL sentences rather than the constraint names,
    because the constraint names also appear in the branch's explanatory comment
    above the read -- which this control deliberately keeps, since the override
    read is the half that must survive.
    """
    source = textwrap.dedent(
        inspect.getsource(NeuronPlatform.check_and_update_config.__func__)
    )
    lines = source.splitlines(keepends=True)
    begins = [i for i, line in enumerate(lines) if VALIDATION_REGION_BEGIN in line]
    ends = [i for i, line in enumerate(lines) if VALIDATION_REGION_END in line]
    assert len(begins) == 1 and len(ends) == 1 and ends[0] > begins[0], (
        "the section-6 validation region is no longer delimited exactly once, "
        "so this control is not stripping what it claims to strip"
    )
    kept = lines[: begins[0]] + lines[ends[0] + 1 :]
    while kept and kept[0].lstrip().startswith("@"):
        kept.pop(0)
    stripped = "".join(kept)
    assert VALIDATION_REGION_BEGIN not in stripped
    assert stripped.count("raise ValueError") == source.count("raise ValueError") - 2, (
        "the strip did not remove exactly the two section-6 refusals"
    )
    assert FLOOR_REFUSAL_ANCHOR not in stripped
    assert GRANULARITY_REFUSAL_ANCHOR not in stripped
    # The override READ survives -- the control is "validation dropped", not
    # "override ignored".
    assert "hybrid_kv_block_size" in stripped

    namespace = dict(vars(platform_module))
    exec(compile(stripped, "<section-6 validation stripped>", "exec"), namespace)
    return namespace["check_and_update_config"]


def test_hybrid_path_is_engaged_counted_differential() -> None:
    """Conjunct 1 -- 1/1 engaged, 1/1 not engaged, on the resolved output."""
    engaged_cfg = _build_config(neuron_config={"enable_hybrid_kv_cache": True})
    not_engaged_cfg = _build_config(neuron_config={"enable_hybrid_kv_cache": False})

    default_block_size = engaged_cfg.cache_config.block_size
    _record("cache_config_default_block_size", default_block_size)

    with _isolated_platform_class_state(), _capture_platform_log() as handler:
        NeuronPlatform.check_and_update_config(engaged_cfg)
        engaged_records = _hybrid_records(handler)

    with _isolated_platform_class_state(), _capture_platform_log() as handler:
        NeuronPlatform.check_and_update_config(not_engaged_cfg)
        not_engaged_records = _hybrid_records(handler)

    _record("engaged_block_size", engaged_cfg.cache_config.block_size)
    _record("engaged_latch", engaged_cfg.cache_config.user_specified_block_size)
    _record("engaged_hybrid_log_records", engaged_records)
    _record("not_engaged_block_size", not_engaged_cfg.cache_config.block_size)
    _record("not_engaged_latch", not_engaged_cfg.cache_config.user_specified_block_size)
    _record("not_engaged_hybrid_log_records", not_engaged_records)

    # ENGAGED, 1/1 -- observed on the branch's own resolved output.
    assert engaged_cfg.cache_config.block_size == DECIDED_HYBRID_BLOCK_SIZE
    assert engaged_cfg.cache_config.user_specified_block_size is True
    assert engaged_records == 1

    # NOT ENGAGED, 1/1 -- the same config with the opt-in off. The parent
    # reading for BOTH sides is 16 / False / 0, so this side is the parent's
    # behaviour preserved and the engaged side is this increment's own work.
    assert not_engaged_cfg.cache_config.block_size == default_block_size
    assert not_engaged_cfg.cache_config.user_specified_block_size is False
    assert not_engaged_records == 0


def test_hybrid_block_size_is_the_decided_value() -> None:
    """Conjunct 2 -- exact equality against the DECISIONS section 6 value, 1/1.

    Scoped to the KNOB-UNSET case, which the block declares at revision 28: the
    operator override is conjunct 6's domain, and the assertion below is about
    what the branch resolves when nothing overrides it.
    """
    cfg = _build_config(neuron_config={"enable_hybrid_kv_cache": True})
    # The domain of this conjunct, asserted rather than assumed.
    assert "hybrid_kv_block_size" not in cfg.additional_config["neuron_config"]
    with _isolated_platform_class_state():
        NeuronPlatform.check_and_update_config(cfg)

    resolved = cfg.cache_config.block_size
    _record("resolved_hybrid_block_size", resolved)
    assert resolved == DECIDED_HYBRID_BLOCK_SIZE
    assert isinstance(resolved, int) and not isinstance(resolved, bool)

    # RECORDED SURVIVAL DIFFERENTIAL, adding no conjunct. The fork's landed
    # update_block_size_for_backend runs after this method, from the executor,
    # and hard-sets 32 unless the latch is set. Measured against the LANDED
    # method, so the resolved value is shown to reach the allocator.
    NeuronPlatform.update_block_size_for_backend(cfg)
    resolved_after = cfg.cache_config.block_size
    _record("block_size_after_landed_update_block_size_for_backend", resolved_after)
    assert resolved_after == DECIDED_HYBRID_BLOCK_SIZE

    # Control for that differential: the identical sequence with the latch
    # cleared reads 32, so the survival tracks the latch and not the assignment.
    control_cfg = _build_config(neuron_config={"enable_hybrid_kv_cache": True})
    with _isolated_platform_class_state():
        NeuronPlatform.check_and_update_config(control_cfg)
    control_cfg.cache_config.user_specified_block_size = False
    NeuronPlatform.update_block_size_for_backend(control_cfg)
    _record(
        "control_block_size_with_latch_cleared",
        control_cfg.cache_config.block_size,
    )
    assert control_cfg.cache_config.block_size == 32

    # The padded-page prohibition, asserted rather than trusted: this path sets
    # no padded mamba page and calls no base alignment.
    _record("mamba_page_size_padded", cfg.cache_config.mamba_page_size_padded)
    assert cfg.cache_config.mamba_page_size_padded is None


def test_pin_archs_resolve_with_unchanged_cache_config() -> None:
    """Conjunct 4 -- 0 of 5 pin archs change, and the zero is shown to move."""
    archs = _pin_archs()
    _record("pin_archs", archs)
    assert len(archs) == 5, f"the pin arch population changed size: {archs}"

    changed: dict[str, dict] = {}
    for arch in archs:
        cfg = _build_config(arch=arch)
        before = _cache_snapshot(cfg)
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(cfg)
        after = _cache_snapshot(cfg)
        diff = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
        if diff:
            changed[arch] = diff

    _record("pin_archs_with_changed_cache_config", changed)
    assert changed == {}, f"pin archs whose resolved cache config moved: {changed}"

    # D1.5 CONTROL -- the zero MOVES. The property this zero tests is that the
    # hybrid branch is reached only by the opt-in, so the falsification is the
    # same five archs with the opt-in ON at the registered preconditions: every
    # one of them then changes, and the count reads 5 rather than 0.
    control_changed: dict[str, dict] = {}
    for arch in archs:
        cfg = _build_config(
            arch=arch, neuron_config={"enable_hybrid_kv_cache": True}
        )
        before = _cache_snapshot(cfg)
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(cfg)
        after = _cache_snapshot(cfg)
        diff = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
        if diff:
            control_changed[arch] = diff

    _record("control_changed_count_with_opt_in_on", len(control_changed))
    _record(
        "control_changed_fields",
        sorted({k for d in control_changed.values() for k in d}),
    )
    assert len(control_changed) == 5, (
        "the control did not move the zero, so this conjunct would read 0 "
        "whether or not the opt-in gates the branch"
    )
    assert all(
        set(d) == {"block_size", "user_specified_block_size"}
        for d in control_changed.values()
    )


def test_both_registered_preconditions_fail_loudly() -> None:
    """Conjunct 5 -- 2/2 negative arms, each raising BEFORE any assignment."""
    # ARM 1 -- TP != 64. The off-value is recorded, not implied.
    _record("arm1_off_value_tensor_parallel_size", OFF_VALUE_TP_DEGREE)
    tp_cfg = _build_config(
        neuron_config={"enable_hybrid_kv_cache": True}, tp=OFF_VALUE_TP_DEGREE
    )
    tp_before = _cache_snapshot(tp_cfg)
    with pytest.raises(ValueError) as tp_exc:
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(tp_cfg)
    tp_message = str(tp_exc.value)
    _record("arm1_message", tp_message)
    assert "tensor_parallel_size" in tp_message
    assert str(DECIDED_HYBRID_BLOCK_SIZE) in tp_message
    assert str(OFF_VALUE_TP_DEGREE) in tp_message
    # The raise happened BEFORE any block-size assignment.
    assert _cache_snapshot(tp_cfg) == tp_before
    assert tp_cfg.cache_config.user_specified_block_size is False

    # ARM 2 -- KV dtype != bf16, reached through cache_dtype="auto" so the
    # resolved dtype follows the model dtype, exactly as vLLM's own hybrid
    # alignment resolves it.
    _record("arm2_off_value_model_dtype", str(OFF_VALUE_MODEL_DTYPE))
    dt_cfg = _build_config(
        neuron_config={"enable_hybrid_kv_cache": True},
        model_dtype=OFF_VALUE_MODEL_DTYPE,
    )
    dt_before = _cache_snapshot(dt_cfg)
    with pytest.raises(ValueError) as dt_exc:
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(dt_cfg)
    dt_message = str(dt_exc.value)
    _record("arm2_message", dt_message)
    assert "bfloat16" in dt_message
    assert str(DECIDED_HYBRID_BLOCK_SIZE) in dt_message
    assert str(OFF_VALUE_MODEL_DTYPE) in dt_message
    assert _cache_snapshot(dt_cfg) == dt_before
    assert dt_cfg.cache_config.user_specified_block_size is False

    # The arms read the preconditions and assert NO value for them: the
    # matching positive arm is conjunct 2's 1/1 at TP=64 / bf16, which is what
    # makes this pair a differential rather than two independent hopes.
    positive_cfg = _build_config(neuron_config={"enable_hybrid_kv_cache": True})
    with _isolated_platform_class_state():
        NeuronPlatform.check_and_update_config(positive_cfg)
    _record("positive_arm_block_size", positive_cfg.cache_config.block_size)
    assert positive_cfg.cache_config.block_size == DECIDED_HYBRID_BLOCK_SIZE

    # An explicitly-elected non-bf16 KV cache reaches the same refusal through
    # the STR_DTYPE_TO_TORCH_DTYPE branch rather than through the model dtype,
    # recorded so the guard is not passing only on one of its two routes.
    explicit_cfg = _build_config(
        neuron_config={"enable_hybrid_kv_cache": True}, cache_dtype="fp8"
    )
    with pytest.raises(ValueError) as explicit_exc:
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(explicit_cfg)
    _record("explicit_cache_dtype_fp8_message", str(explicit_exc.value))
    assert explicit_cfg.cache_config.user_specified_block_size is False


def test_operator_override_is_honoured_only_within_registered_constraints() -> None:
    """Conjunct 6 -- 2/2 arms, one in each direction, plus the D1.5 control.

    The policy is section 6's, read at realization and asserted here only by
    behaviour and by message content: an operator value is honoured when it
    satisfies both registered constraints, and refused with the SAME error class
    conjunct 5 uses when it does not.
    """
    # ---- ARM A, negative, 1/1 -- below the registered floor ----------------
    # The off-value is recorded, not implied, and it violates ONLY the floor.
    _record(
        "arm_a_off_value_operator_block_size",
        OFF_VALUE_OPERATOR_BLOCK_SIZE_BELOW_FLOOR,
    )
    low_cfg = _build_config(
        neuron_config={
            "enable_hybrid_kv_cache": True,
            "hybrid_kv_block_size": OFF_VALUE_OPERATOR_BLOCK_SIZE_BELOW_FLOOR,
        }
    )
    low_before = _cache_snapshot(low_cfg)
    with pytest.raises(ValueError) as low_exc:
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(low_cfg)
    low_message = str(low_exc.value)
    _record("arm_a_message", low_message)
    # The message names the violated constraint and the offending value.
    assert "floor" in low_message
    assert str(OFF_VALUE_OPERATOR_BLOCK_SIZE_BELOW_FLOOR) in low_message
    # The raise happened BEFORE any block-size assignment -- conjunct 5's
    # assertion shape, for conjunct 5's reason: at the pin nothing else catches
    # an under-sized page.
    assert _cache_snapshot(low_cfg) == low_before
    assert low_cfg.cache_config.user_specified_block_size is False

    # ---- ARM B, positive, 1/1 -- a legal non-default value is HONOURED -----
    _record(
        "arm_b_legal_operator_block_size",
        LEGAL_NON_DEFAULT_OPERATOR_BLOCK_SIZE,
    )
    assert LEGAL_NON_DEFAULT_OPERATOR_BLOCK_SIZE != DECIDED_HYBRID_BLOCK_SIZE
    legal_cfg = _build_config(
        neuron_config={
            "enable_hybrid_kv_cache": True,
            "hybrid_kv_block_size": LEGAL_NON_DEFAULT_OPERATOR_BLOCK_SIZE,
        }
    )
    with _isolated_platform_class_state():
        NeuronPlatform.check_and_update_config(legal_cfg)
    resolved = legal_cfg.cache_config.block_size
    _record("arm_b_resolved_block_size", resolved)
    # Resolves to ITSELF -- neither refused nor silently overwritten with the
    # decided value.
    assert resolved == LEGAL_NON_DEFAULT_OPERATOR_BLOCK_SIZE
    assert legal_cfg.cache_config.user_specified_block_size is True

    # Recorded, adding no conjunct: the operator value also reaches the
    # allocator, so "honoured" is not undone one call later.
    NeuronPlatform.update_block_size_for_backend(legal_cfg)
    _record(
        "arm_b_block_size_after_update_block_size_for_backend",
        legal_cfg.cache_config.block_size,
    )
    assert legal_cfg.cache_config.block_size == LEGAL_NON_DEFAULT_OPERATOR_BLOCK_SIZE

    # Recorded, adding no conjunct: the second registered constraint refuses on
    # its own, so arm A is not the only route into the refusal.
    off_grain_cfg = _build_config(
        neuron_config={
            "enable_hybrid_kv_cache": True,
            "hybrid_kv_block_size": OFF_VALUE_OPERATOR_BLOCK_SIZE_OFF_GRANULARITY,
        }
    )
    with pytest.raises(ValueError) as off_grain_exc:
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(off_grain_cfg)
    off_grain_message = str(off_grain_exc.value)
    _record("off_granularity_value", OFF_VALUE_OPERATOR_BLOCK_SIZE_OFF_GRANULARITY)
    _record("off_granularity_message", off_grain_message)
    assert "granularity" in off_grain_message
    assert str(OFF_VALUE_OPERATOR_BLOCK_SIZE_OFF_GRANULARITY) in off_grain_message
    assert off_grain_cfg.cache_config.user_specified_block_size is False

    # ---- D1.5 CONTROL -- arm A's raise count MOVES to 0/1 ------------------
    # Keeping the override read while dropping the validation: the same input
    # then reaches cache_config unrefused, so the arm reddens exactly when the
    # property it tests is false.
    variant = _validation_stripped_variant()
    control_cfg = _build_config(
        neuron_config={
            "enable_hybrid_kv_cache": True,
            "hybrid_kv_block_size": OFF_VALUE_OPERATOR_BLOCK_SIZE_BELOW_FLOOR,
        }
    )
    control_raises = 0
    with _isolated_platform_class_state():
        try:
            variant(NeuronPlatform, control_cfg)
        except ValueError:
            control_raises = 1
    _record("control_arm_a_raise_count_without_validation", control_raises)
    _record(
        "control_block_size_without_validation",
        control_cfg.cache_config.block_size,
    )
    assert control_raises == 0, (
        "the control still raised, so arm A cannot distinguish the validation "
        "from the override read"
    )
    assert (
        control_cfg.cache_config.block_size
        == OFF_VALUE_OPERATOR_BLOCK_SIZE_BELOW_FLOOR
    )


# =========================================================================
# inc-glm53f-081 -- the platform ENGAGES the limb above for this architecture
# =========================================================================
# Four items, one per NEW arm, appended so that not one byte of `-020` above is
# edited. The subject is the three-question decision immediately above the limb
# in ``check_and_update_config``, never the limb itself.

# The non-engagement marker the decision emits. Deliberately NOT a substring of
# HYBRID_MARKER in either direction, so the landed engagement differential
# cannot be inflated by it; both directions are asserted in item (d3).
OFF_MARKER = "Hybrid KDA/DSA KV cache left OFF"

# The campaign's architecture string, the decision's own question-1 predicate.
CAMPAIGN_ARCH = "Glm5NextForConditionalGeneration"

# Conjunct (b) arm 2's match: the substring the RENDERED landed message carries,
# quoted from HEAD 3e53891c and edited nowhere. Arm 1's counterpart substring is
# NOT declared here -- arm 1 is `-020`'s landed item, which this file re-runs
# whole, and its rendered message is read off the transcript's own recording
# rather than re-asserted by a constant no item would use.
BF16_REFUSAL_SUBSTRING = "registered ONLY for a bfloat16 KV cache"


class _LevelRecordingHandler(logging.Handler):
    """``_RecordingHandler``'s shape, keeping each record's LEVEL as well."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelname, record.getMessage()))


@contextlib.contextmanager
def _capture_platform_records():
    """``_capture_platform_log``'s shape, level-aware.

    A separate context rather than an edit to `-020`'s helper, for `-020`'s own
    reason -- the handler attaches to ``platform.logger`` directly, never
    through ``caplog`` -- and because the five landed items are this block's
    acceptance and are left byte-identical.
    """
    handler = _LevelRecordingHandler()
    target = platform_module.logger
    previous_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)


def _marker_records(
    handler: _LevelRecordingHandler, marker: str, level: str | None = None
) -> list[str]:
    """The messages carrying ``marker``, optionally narrowed to one level."""
    return [
        message
        for levelname, message in handler.records
        if marker in message and (level is None or levelname == level)
    ]


def test_engagement_is_automatic_at_the_registered_tp_degree() -> None:
    """Conjunct (a) -- 1/1. No knob set, TP=64, bf16: the limb is ENTERED.

    The reading the repair exists for. At the parent the same input leaves the
    page at the vendor default and ``update_block_size_for_backend`` then hard
    sets 32; the parent reading is measured in the acceptance transcript's own
    parent row rather than restated here.
    """
    cfg = _build_config()
    # The domain, asserted rather than assumed: NOTHING sets the knob.
    assert cfg.additional_config == {}
    default_block_size = cfg.cache_config.block_size
    _record("a_default_block_size_before_call", default_block_size)

    with _isolated_platform_class_state(), _capture_platform_records() as handler:
        NeuronPlatform.check_and_update_config(cfg)
        engaged = _marker_records(handler, HYBRID_MARKER)
        off = _marker_records(handler, OFF_MARKER)

    _record("a_block_size", cfg.cache_config.block_size)
    _record("a_latch", cfg.cache_config.user_specified_block_size)
    _record("a_engagement_records", len(engaged))
    _record("a_off_records", len(off))
    _record("a_additional_config", cfg.additional_config)

    assert cfg.cache_config.block_size == DECIDED_HYBRID_BLOCK_SIZE
    assert cfg.cache_config.user_specified_block_size is True
    assert len(engaged) == 1
    # The decision is READABLE DOWNSTREAM: the write-back, not just the local
    # mapping the limb read. Without it a later NeuronConfig construction would
    # see no decision at all.
    assert cfg.additional_config["neuron_config"]["enable_hybrid_kv_cache"] is True
    # The engagement path stays quiet -- the OFF warning is (d3)'s alone.
    assert off == []
    # The value the engagement produced is NOT the value it started from, so
    # this 1/1 cannot be read off a config that already carried 128.
    assert default_block_size != DECIDED_HYBRID_BLOCK_SIZE

    # Recorded, adding no conjunct: the resolved page SURVIVES to the allocator,
    # which is the consequence the repair is for. Measured against the LANDED
    # method, unchanged here.
    NeuronPlatform.update_block_size_for_backend(cfg)
    _record(
        "a_block_size_after_update_block_size_for_backend",
        cfg.cache_config.block_size,
    )
    assert cfg.cache_config.block_size == DECIDED_HYBRID_BLOCK_SIZE


def test_bfloat16_guard_is_reached_on_the_default_path() -> None:
    """Conjunct (b) arm 2 -- the knob UNSET at TP=64 with fp16 still RAISES.

    Why this arm is not vacuous: the bfloat16 precondition is deliberately left
    INSIDE the limb rather than pre-checked by the decision above it, so a
    DEFAULT run at the registered degree with a non-bf16 KV cache must still
    reach `-020`'s loud guard. Arm 1 of this conjunct is `-020`'s landed
    operator-explicit item, re-run by this file and not duplicated here.
    """
    cfg = _build_config(model_dtype=OFF_VALUE_MODEL_DTYPE)
    # The domain: no knob, registered TP degree, off-value model dtype.
    assert cfg.additional_config == {}
    assert cfg.parallel_config.tensor_parallel_size == REGISTERED_TP_DEGREE
    before = _cache_snapshot(cfg)

    with pytest.raises(ValueError) as exc:
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(cfg)
    message = str(exc.value)
    _record("b_arm2_message", message)

    # Matched on the substring the RENDERED landed message carries.
    assert BF16_REFUSAL_SUBSTRING in message
    assert str(OFF_VALUE_MODEL_DTYPE) in message
    assert str(DECIDED_HYBRID_BLOCK_SIZE) in message
    # The raise happened BEFORE any block-size assignment.
    assert _cache_snapshot(cfg) == before
    assert cfg.cache_config.user_specified_block_size is False

    # D1.5 CONTROL -- the raise MOVES to 0 when the engagement declines. The
    # same off-value dtype at the off-value TP degree raises nothing, because
    # there the decision never enters the limb. That is what makes this arm's
    # raise attributable to the engagement rather than to the dtype alone.
    quiet_cfg = _build_config(
        model_dtype=OFF_VALUE_MODEL_DTYPE, tp=OFF_VALUE_TP_DEGREE
    )
    control_raises = 0
    with _isolated_platform_class_state():
        try:
            NeuronPlatform.check_and_update_config(quiet_cfg)
        except ValueError:
            control_raises = 1
    _record("b_arm2_control_raise_count", control_raises)
    _record("b_arm2_control_block_size", quiet_cfg.cache_config.block_size)
    assert control_raises == 0, (
        "the control still raised, so this arm cannot distinguish the "
        "engagement from the dtype"
    )
    assert quiet_cfg.cache_config.user_specified_block_size is False


def test_default_run_at_off_value_tp_is_quiet_and_unchanged() -> None:
    """Conjunct (d2) -- no knob, TP != 64: nothing raises and nothing moves."""
    cfg = _build_config(tp=OFF_VALUE_TP_DEGREE)
    assert cfg.additional_config == {}
    before = _cache_snapshot(cfg)
    block_size_before = cfg.cache_config.block_size
    _record("d2_off_value_tensor_parallel_size", OFF_VALUE_TP_DEGREE)
    _record("d2_block_size_before_call", block_size_before)

    raised = None
    with _isolated_platform_class_state(), _capture_platform_records() as handler:
        try:
            NeuronPlatform.check_and_update_config(cfg)
        except BaseException as exc:  # recorded, then adjudicated below
            raised = f"{type(exc).__name__}: {exc}"
        engaged = _marker_records(handler, HYBRID_MARKER)

    _record("d2_raised", raised)
    _record("d2_block_size", cfg.cache_config.block_size)
    _record("d2_latch", cfg.cache_config.user_specified_block_size)
    _record("d2_engagement_records", len(engaged))

    assert raised is None, f"the non-engagement path raised: {raised}"
    assert cfg.cache_config.block_size == block_size_before
    assert cfg.cache_config.user_specified_block_size is False
    assert len(engaged) == 0
    assert _cache_snapshot(cfg) == before
    # The write-back is engagement-only, so nothing downstream can read a
    # decision this path never took.
    assert "neuron_config" not in cfg.additional_config

    # D1.5 CONTROL -- all four readings MOVE at the registered degree, so this
    # quiet tracks the TP degree rather than a decision that never fires.
    control_cfg = _build_config()
    with _isolated_platform_class_state(), _capture_platform_records() as handler:
        NeuronPlatform.check_and_update_config(control_cfg)
        control_engaged = _marker_records(handler, HYBRID_MARKER)
    _record("d2_control_block_size", control_cfg.cache_config.block_size)
    _record("d2_control_latch", control_cfg.cache_config.user_specified_block_size)
    _record("d2_control_engagement_records", len(control_engaged))
    assert control_cfg.cache_config.block_size == DECIDED_HYBRID_BLOCK_SIZE
    assert control_cfg.cache_config.user_specified_block_size is True
    assert len(control_engaged) == 1
    assert "neuron_config" in control_cfg.additional_config


def test_default_run_at_off_value_tp_warns_exactly_once() -> None:
    """Conjunct (d3) -- the SAME construction says so, once, at WARNING.

    The half of (d2) that keeps the non-engagement AUDIBLE: a reading of 0
    warning records here fails the block, because a silent TP!=64 run is
    review 36's defect coming back.
    """
    cfg = _build_config(tp=OFF_VALUE_TP_DEGREE)
    assert cfg.additional_config == {}

    with _isolated_platform_class_state(), _capture_platform_records() as handler:
        NeuronPlatform.check_and_update_config(cfg)
        at_warning = _marker_records(handler, OFF_MARKER, level="WARNING")
        at_any_level = _marker_records(handler, OFF_MARKER)
        all_records = list(handler.records)

    _record("d3_all_platform_records", all_records)
    _record("d3_marker_records_any_level", len(at_any_level))
    _record("d3_marker_records_at_warning", len(at_warning))

    assert len(at_any_level) == 1, f"expected exactly 1 record, got {at_any_level}"
    assert len(at_warning) == 1, f"the record was not at WARNING: {all_records}"
    message = at_warning[0]
    _record("d3_message", message)

    # It names BOTH degrees -- what this run resolved and what is registered.
    # Matched on the RENDERED key=value pair rather than on the bare number,
    # because the bare 32 also occurs in the page the sentence promises and a
    # bare-number match would pass with the resolved degree missing entirely.
    assert f"tensor_parallel_size={OFF_VALUE_TP_DEGREE}" in message
    assert f"tensor_parallel_size={REGISTERED_TP_DEGREE}" in message
    # ... and the architecture, so one line tells the operator which model lost
    # the page.
    assert CAMPAIGN_ARCH in message
    # ... and what re-enabling it elsewhere costs.
    assert "re-derive" in message
    assert "enable_hybrid_kv_cache" in message
    # Neither marker is a substring of the other, in EITHER direction, so the
    # landed engagement differential cannot be inflated by this warning.
    assert HYBRID_MARKER not in message
    assert OFF_MARKER not in HYBRID_MARKER
    assert HYBRID_MARKER not in OFF_MARKER

    # The page the warning PROMISES is the page the run actually gets, measured
    # rather than restated: the landed update_block_size_for_backend's own
    # output has to appear in the sentence. This is what pins the one number the
    # warning duplicates, so a drift in either place reddens a test.
    NeuronPlatform.update_block_size_for_backend(cfg)
    delivered = cfg.cache_config.block_size
    _record("d3_delivered_block_size", delivered)
    assert f"{delivered}-token page" in message

    # D1.5 CONTROL -- the 1 MOVES to 0 at the registered degree, so the count
    # tracks the declined engagement and not an unconditional warning.
    control_cfg = _build_config()
    with _isolated_platform_class_state(), _capture_platform_records() as handler:
        NeuronPlatform.check_and_update_config(control_cfg)
        control_off = _marker_records(handler, OFF_MARKER)
    _record("d3_control_off_records", len(control_off))
    assert len(control_off) == 0

    # SECOND CONTROL -- a non-campaign architecture at the SAME off-value degree
    # is silent, so question 1 ends the decision and the warning is scoped to
    # this architecture instead of firing for every model on the pin.
    other_arch = _pin_archs()[0]
    other_cfg = _build_config(arch=other_arch, tp=OFF_VALUE_TP_DEGREE)
    with _isolated_platform_class_state(), _capture_platform_records() as handler:
        NeuronPlatform.check_and_update_config(other_cfg)
        other_off = _marker_records(handler, OFF_MARKER)
    _record("d3_other_arch", other_arch)
    _record("d3_other_arch_off_records", len(other_off))
    assert len(other_off) == 0

    # THIRD CONTROL -- an operator who set the knob is not lectured: question 2
    # ends the decision before the warning. Explicit False is measured here;
    # explicit True at this degree is `-020`'s landed arm 1, which raises inside
    # the limb and so never reaches this warning either.
    explicit_cfg = _build_config(
        neuron_config={"enable_hybrid_kv_cache": False}, tp=OFF_VALUE_TP_DEGREE
    )
    with _isolated_platform_class_state(), _capture_platform_records() as handler:
        NeuronPlatform.check_and_update_config(explicit_cfg)
        explicit_off = _marker_records(handler, OFF_MARKER)
    _record("d3_explicit_false_off_records", len(explicit_off))
    assert len(explicit_off) == 0


#: The operator's own page, for the latched half of the domain. Distinct from
#: every page any other item here uses, so a match on the rendered
#: ``64-token page`` cannot be satisfied by a number the code did not read.
OPERATOR_LATCHED_BLOCK_SIZE = 64


def test_off_warning_reports_the_page_a_latched_run_actually_gets() -> None:
    """The declined engagement names the OPERATOR's page, not the default.

    ADDED BY THE `inc-glm53f-081` REPAIR, for finding `F-B29-01`. The item above
    establishes that the sentence matches the delivered page, and it does so in
    one half of the domain: every config it builds supplies no block size, so
    ``user_specified_block_size`` is ``False`` throughout and the delivered page
    is always 32. The sentence used to say 32 unconditionally, so that item
    passed while the claim was false for the other half.

    THIS IS THE OTHER HALF. The operator passes a block size, the vendor latches
    it, ``update_block_size_for_backend`` returns without touching the page, and
    the run allocates the operator's value. A sentence that still said 32 would
    be telling someone sizing KV memory the wrong number.

    Widening the item above was not enough and the finding says so: it passes
    today on a message that names 32 unconditionally, so only a new reading in
    the latched state can move.
    """
    cfg = _build_config(
        tp=OFF_VALUE_TP_DEGREE, block_size=OPERATOR_LATCHED_BLOCK_SIZE
    )
    # The premise, asserted rather than assumed: the vendor really did latch.
    assert cfg.cache_config.user_specified_block_size is True
    assert cfg.cache_config.block_size == OPERATOR_LATCHED_BLOCK_SIZE
    _record("repair_latched_block_size", cfg.cache_config.block_size)
    _record("repair_latch", cfg.cache_config.user_specified_block_size)

    with _isolated_platform_class_state(), _capture_platform_records() as handler:
        NeuronPlatform.check_and_update_config(cfg)
        at_warning = _marker_records(handler, OFF_MARKER, level="WARNING")

    assert len(at_warning) == 1, f"expected exactly 1 record, got {at_warning}"
    message = at_warning[0]
    _record("repair_latched_message", message)

    # The delivered page, from the landed method itself rather than from a
    # literal. Called AFTER the decision, which is the order a real run has.
    NeuronPlatform.update_block_size_for_backend(cfg)
    delivered = cfg.cache_config.block_size
    _record("repair_delivered_block_size", delivered)
    assert delivered == OPERATOR_LATCHED_BLOCK_SIZE, (
        f"the operator's page did not survive: delivered {delivered}"
    )

    # THE READING THIS ITEM EXISTS FOR: the sentence names the delivered page.
    assert f"{delivered}-token page" in message
    # ... and does NOT name the default, which is what it used to say here.
    # This is the assertion that reddens on the unrepaired code.
    assert f"{NeuronPlatform.UNIFORM_NEURON_PAGE}-token page" not in message
    # ... and says where the number came from, so the operator can tell the two
    # cases apart without reading this file.
    assert "supplied on the command line" in message

    # The two readers of one number are pinned against each other, in BOTH latch
    # states, so neither can drift alone. This is the part that keeps the fix
    # honest: `resolved_uniform_page` is what the warning reads and
    # `update_block_size_for_backend` is what the run allocates.
    latched = _build_config(
        tp=OFF_VALUE_TP_DEGREE, block_size=OPERATOR_LATCHED_BLOCK_SIZE
    )
    predicted_latched = NeuronPlatform.resolved_uniform_page(latched)
    NeuronPlatform.update_block_size_for_backend(latched)
    unlatched = _build_config(tp=OFF_VALUE_TP_DEGREE)
    predicted_unlatched = NeuronPlatform.resolved_uniform_page(unlatched)
    NeuronPlatform.update_block_size_for_backend(unlatched)
    _record("repair_predicted_latched", predicted_latched)
    _record("repair_predicted_unlatched", predicted_unlatched)
    assert predicted_latched == latched.cache_config.block_size
    assert predicted_unlatched == unlatched.cache_config.block_size
    # And the prediction MOVES with the latch, so agreement is not two constants
    # that happen to match.
    assert predicted_latched != predicted_unlatched, (
        f"both latch states predicted {predicted_latched}; the reader is not "
        f"reading the latch"
    )
    assert predicted_unlatched == NeuronPlatform.UNIFORM_NEURON_PAGE
