# SPDX-License-Identifier: Apache-2.0
"""Hybrid KDA/DSA cache configuration in ``NeuronPlatform.check_and_update_config``.

For ``Glm5NextForConditionalGeneration`` the platform resolves the KV cache
block size at config time: it engages the hybrid path by default at the
supported tensor-parallel degree, honours an operator ``hybrid_kv_block_size``
that satisfies the KDA state-page floor and the granularity multiple, refuses
one that does not, and refuses a non-bfloat16 KV cache. When it declines to
engage it leaves the cache config untouched and says so once, at WARNING,
naming the page the run will actually get.

The block size it resolves has to survive ``update_block_size_for_backend``,
which runs later from the executor and hard-sets the uniform page unless
``cache_config.user_specified_block_size`` is set.

No test constructs an engine, loads a checkpoint, reaches a network or touches a
device.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.vllm import platform as platform_module
from vllm_neuron.vllm.platform import NeuronPlatform

# The hybrid block size the platform resolves when nothing overrides it.
HYBRID_BLOCK_SIZE = 128

# The two preconditions that block size is valid under.
REQUIRED_TP_DEGREE = 64
REQUIRED_KV_DTYPE = torch.bfloat16

UNSUPPORTED_TP_DEGREE = 32
UNSUPPORTED_MODEL_DTYPE = torch.float16

# An operator block size that satisfies the granularity multiple and violates
# only the KDA state-page floor, so the refusal cannot come from the other
# check.
BELOW_FLOOR_BLOCK_SIZE = 64
# The smallest legal operator block size that is not the resolved default, so
# "honoured" is shown on a value the platform could not have produced itself.
ALTERNATE_BLOCK_SIZE = 192
# Above the floor, off the granularity multiple.
OFF_GRANULARITY_BLOCK_SIZE = 200
# The operator's own page for the latched half of the domain.
OPERATOR_BLOCK_SIZE = 64

HYBRID_MARKER = "Hybrid KDA/DSA KV cache enabled"
# The non-engagement marker. Deliberately not a substring of HYBRID_MARKER in
# either direction, so counting one cannot pick up the other.
OFF_MARKER = "Hybrid KDA/DSA KV cache left OFF"

GLM5_NEXT_ARCH = "Glm5NextForConditionalGeneration"

BF16_REFUSAL_SUBSTRING = "supported only with a bfloat16 KV cache"

# Resolved off ``__file__`` so the read cannot depend on the invocation's cwd.
FIXTURE_CONFIG = (
    Path(__file__).resolve().parent / "model" / "glm5_next" / "fixtures" / "config.json"
)

# Class attributes check_and_update_config writes. Saved and restored around
# every call so no test can read another test's residue.
_MUTATED_CLASS_ATTRS = (
    "_enable_structured_outputs",
    "_max_embeds_per_image",
    "_termination_timeout_patched",
)


@contextlib.contextmanager
def _isolated_platform_class_state():
    saved = {name: getattr(NeuronPlatform, name) for name in _MUTATED_CLASS_ATTRS}
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(NeuronPlatform, name, value)


class _RecordingHandler(logging.Handler):
    """Collects each record's level and formatted message."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelname, record.getMessage()))


@contextlib.contextmanager
def _capture_platform_log():
    """Attach to ``platform.logger`` directly.

    Not through ``caplog``: its handler lives on the root logger, so a
    propagation setting anywhere in vLLM's logging configuration would make a
    record count read 0 for a reason unrelated to the code under test.
    """
    handler = _RecordingHandler()
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
    handler: _RecordingHandler, marker: str, level: str | None = None
) -> list[str]:
    """The messages carrying ``marker``, optionally narrowed to one level."""
    return [
        message
        for levelname, message in handler.records
        if marker in message and (level is None or levelname == level)
    ]


def _fixture_hf_config(*, with_vision: bool):
    """An ``hf_config`` stand-in built from the fixture checkpoint's bytes.

    ``with_vision=False`` strips ``vision_config`` so the call resolves through
    the hybrid branch alone and never enters ``_resolve_vision_auto_config``.
    """
    assert FIXTURE_CONFIG.is_file(), f"fixture unreachable: {FIXTURE_CONFIG}"
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
    tp: int = REQUIRED_TP_DEGREE,
    model_dtype: torch.dtype = REQUIRED_KV_DTYPE,
    cache_dtype: str = "auto",
    max_model_len: int = 8192,
    block_size: int | None = None,
):
    """A config carrying real vLLM ``CacheConfig``/``ParallelConfig``.

    The three fields under test (``block_size``,
    ``user_specified_block_size``, ``cache_dtype``) are the vendor's own, with
    the vendor's defaults and validators. ``model_config`` stays a stand-in: a
    real ``ModelConfig`` needs a checkpoint.

    ``block_size=None`` is the operator supplying none. Passing a value is how
    the operator's own ``--block-size`` is reproduced: the vendor's
    ``CacheConfig`` latches ``user_specified_block_size`` on an explicit value.
    """
    from vllm.config.cache import CacheConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.config.scheduler import SchedulerConfig

    hf = _fixture_hf_config(with_vision=False)
    if arch is not None:
        hf.architectures = [arch]

    model_config = SimpleNamespace(
        hf_config=hf,
        # vLLM's own ModelConfig.architectures is a property over hf_config, and
        # the platform reads it there, so the stand-in mirrors both spellings.
        architectures=hf.architectures,
        model_impl="auto",
        # Named by vLLM's registry on the Transformers fallback path, which no
        # architecture here takes; a fixed name is enough to keep it reachable.
        _get_transformers_backend_cls=lambda: "TransformersForCausalLM",
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
    """Every ``cache_config`` field, for a before/after comparison."""
    return {k: repr(v) for k, v in sorted(cfg.cache_config.__dict__.items())}


def _other_archs() -> list[str]:
    """The registry's other architectures, read rather than typed."""
    from vllm_neuron.model.registry import get_models

    return [name for name, _ in get_models() if name != GLM5_NEXT_ARCH]


def test_hybrid_path_engages_only_when_opted_in() -> None:
    """The opt-in resolves the hybrid page; the opt-out leaves the default."""
    engaged_cfg = _build_config(neuron_config={"enable_hybrid_kv_cache": True})
    not_engaged_cfg = _build_config(neuron_config={"enable_hybrid_kv_cache": False})

    default_block_size = engaged_cfg.cache_config.block_size

    with _isolated_platform_class_state(), _capture_platform_log() as handler:
        NeuronPlatform.check_and_update_config(engaged_cfg)
        engaged_records = _marker_records(handler, HYBRID_MARKER)

    with _isolated_platform_class_state(), _capture_platform_log() as handler:
        NeuronPlatform.check_and_update_config(not_engaged_cfg)
        not_engaged_records = _marker_records(handler, HYBRID_MARKER)

    assert engaged_cfg.cache_config.block_size == HYBRID_BLOCK_SIZE
    assert engaged_cfg.cache_config.user_specified_block_size is True
    assert len(engaged_records) == 1

    assert not_engaged_cfg.cache_config.block_size == default_block_size
    assert not_engaged_cfg.cache_config.user_specified_block_size is False
    assert not_engaged_records == []


def test_hybrid_block_size_survives_to_the_allocator() -> None:
    """The resolved page is 128 and the latch keeps it through the executor."""
    cfg = _build_config(neuron_config={"enable_hybrid_kv_cache": True})
    # The domain of this test, asserted rather than assumed: no operator value.
    assert "hybrid_kv_block_size" not in cfg.additional_config["neuron_config"]
    with _isolated_platform_class_state():
        NeuronPlatform.check_and_update_config(cfg)

    resolved = cfg.cache_config.block_size
    assert resolved == HYBRID_BLOCK_SIZE
    assert isinstance(resolved, int) and not isinstance(resolved, bool)

    # update_block_size_for_backend runs later, from the executor, and hard-sets
    # the uniform page unless the latch is set.
    NeuronPlatform.update_block_size_for_backend(cfg)
    assert cfg.cache_config.block_size == HYBRID_BLOCK_SIZE

    # The same sequence with the latch cleared reads the uniform page, so the
    # survival is the latch's doing and not the assignment's.
    control_cfg = _build_config(neuron_config={"enable_hybrid_kv_cache": True})
    with _isolated_platform_class_state():
        NeuronPlatform.check_and_update_config(control_cfg)
    control_cfg.cache_config.user_specified_block_size = False
    NeuronPlatform.update_block_size_for_backend(control_cfg)
    assert control_cfg.cache_config.block_size == 32

    # This path sets no padded mamba page and calls no base alignment.
    assert cfg.cache_config.mamba_page_size_padded is None


def test_other_archs_resolve_with_unchanged_cache_config() -> None:
    """No other architecture's resolved cache config moves."""
    archs = _other_archs()

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

    assert changed == {}, f"archs whose resolved cache config moved: {changed}"

    # With the opt-in on, every one of them does change, and only in the two
    # block-size fields -- so the quiet above is the opt-in gating the branch
    # and not a call that never reaches it.
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

    assert set(control_changed) == set(archs)
    assert all(
        set(d) == {"block_size", "user_specified_block_size"}
        for d in control_changed.values()
    )


def test_unsupported_tp_degree_and_kv_dtype_raise() -> None:
    """Both preconditions raise, before any block-size assignment."""
    tp_cfg = _build_config(
        neuron_config={"enable_hybrid_kv_cache": True}, tp=UNSUPPORTED_TP_DEGREE
    )
    tp_before = _cache_snapshot(tp_cfg)
    with pytest.raises(ValueError) as tp_exc:
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(tp_cfg)
    tp_message = str(tp_exc.value)
    assert "tensor_parallel_size" in tp_message
    assert str(HYBRID_BLOCK_SIZE) in tp_message
    assert str(UNSUPPORTED_TP_DEGREE) in tp_message
    assert _cache_snapshot(tp_cfg) == tp_before
    assert tp_cfg.cache_config.user_specified_block_size is False

    # The KV dtype is reached through cache_dtype="auto", so the resolved dtype
    # follows the model dtype the way vLLM's own hybrid alignment resolves it.
    dt_cfg = _build_config(
        neuron_config={"enable_hybrid_kv_cache": True},
        model_dtype=UNSUPPORTED_MODEL_DTYPE,
    )
    dt_before = _cache_snapshot(dt_cfg)
    with pytest.raises(ValueError) as dt_exc:
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(dt_cfg)
    dt_message = str(dt_exc.value)
    assert "bfloat16" in dt_message
    assert str(HYBRID_BLOCK_SIZE) in dt_message
    assert str(UNSUPPORTED_MODEL_DTYPE) in dt_message
    assert _cache_snapshot(dt_cfg) == dt_before
    assert dt_cfg.cache_config.user_specified_block_size is False

    # The matching positive case, so the pair is a differential rather than two
    # independent refusals: at the supported degree with bf16 it resolves.
    positive_cfg = _build_config(neuron_config={"enable_hybrid_kv_cache": True})
    with _isolated_platform_class_state():
        NeuronPlatform.check_and_update_config(positive_cfg)
    assert positive_cfg.cache_config.block_size == HYBRID_BLOCK_SIZE

    # An explicitly elected non-bf16 KV cache reaches the same refusal through
    # the STR_DTYPE_TO_TORCH_DTYPE branch rather than through the model dtype.
    explicit_cfg = _build_config(
        neuron_config={"enable_hybrid_kv_cache": True}, cache_dtype="fp8"
    )
    with pytest.raises(ValueError):
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(explicit_cfg)
    assert explicit_cfg.cache_config.user_specified_block_size is False


def test_operator_block_size_override_is_validated() -> None:
    """An operator block size is honoured when legal and refused when not."""
    # Below the state-page floor: refused, naming the constraint and the value.
    low_cfg = _build_config(
        neuron_config={
            "enable_hybrid_kv_cache": True,
            "hybrid_kv_block_size": BELOW_FLOOR_BLOCK_SIZE,
        }
    )
    low_before = _cache_snapshot(low_cfg)
    with pytest.raises(ValueError) as low_exc:
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(low_cfg)
    low_message = str(low_exc.value)
    assert "floor" in low_message
    assert str(BELOW_FLOOR_BLOCK_SIZE) in low_message
    # Nothing else catches an under-sized page, so the refusal has to happen
    # before any block-size assignment.
    assert _cache_snapshot(low_cfg) == low_before
    assert low_cfg.cache_config.user_specified_block_size is False

    # A legal non-default value resolves to itself: neither refused nor
    # silently overwritten with the default.
    assert ALTERNATE_BLOCK_SIZE != HYBRID_BLOCK_SIZE
    legal_cfg = _build_config(
        neuron_config={
            "enable_hybrid_kv_cache": True,
            "hybrid_kv_block_size": ALTERNATE_BLOCK_SIZE,
        }
    )
    with _isolated_platform_class_state():
        NeuronPlatform.check_and_update_config(legal_cfg)
    assert legal_cfg.cache_config.block_size == ALTERNATE_BLOCK_SIZE
    assert legal_cfg.cache_config.user_specified_block_size is True

    # ... and it reaches the allocator, so "honoured" is not undone one call
    # later.
    NeuronPlatform.update_block_size_for_backend(legal_cfg)
    assert legal_cfg.cache_config.block_size == ALTERNATE_BLOCK_SIZE

    # The granularity multiple refuses on its own, so the floor is not the only
    # route into the refusal.
    off_grain_cfg = _build_config(
        neuron_config={
            "enable_hybrid_kv_cache": True,
            "hybrid_kv_block_size": OFF_GRANULARITY_BLOCK_SIZE,
        }
    )
    with pytest.raises(ValueError) as off_grain_exc:
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(off_grain_cfg)
    off_grain_message = str(off_grain_exc.value)
    assert "granularity" in off_grain_message
    assert str(OFF_GRANULARITY_BLOCK_SIZE) in off_grain_message
    assert off_grain_cfg.cache_config.user_specified_block_size is False


def test_hybrid_cache_engages_by_default_at_the_supported_tp_degree() -> None:
    """With no knob set, at TP=64 and bf16, the hybrid path is entered."""
    cfg = _build_config()
    # The domain, asserted rather than assumed: nothing sets the knob.
    assert cfg.additional_config == {}
    default_block_size = cfg.cache_config.block_size

    with _isolated_platform_class_state(), _capture_platform_log() as handler:
        NeuronPlatform.check_and_update_config(cfg)
        engaged = _marker_records(handler, HYBRID_MARKER)
        off = _marker_records(handler, OFF_MARKER)

    assert cfg.cache_config.block_size == HYBRID_BLOCK_SIZE
    assert cfg.cache_config.user_specified_block_size is True
    assert len(engaged) == 1
    # The decision is written back, so a later NeuronConfig construction can
    # read it instead of seeing no decision at all.
    assert cfg.additional_config["neuron_config"]["enable_hybrid_kv_cache"] is True
    assert off == []
    # The resolved page is not the page the config started from, so this reading
    # cannot come from a config that already carried 128.
    assert default_block_size != HYBRID_BLOCK_SIZE

    NeuronPlatform.update_block_size_for_backend(cfg)
    assert cfg.cache_config.block_size == HYBRID_BLOCK_SIZE


def test_bfloat16_guard_is_reached_on_the_default_path() -> None:
    """With no knob set, at TP=64, an fp16 KV cache still raises."""
    cfg = _build_config(model_dtype=UNSUPPORTED_MODEL_DTYPE)
    assert cfg.additional_config == {}
    assert cfg.parallel_config.tensor_parallel_size == REQUIRED_TP_DEGREE
    before = _cache_snapshot(cfg)

    with pytest.raises(ValueError) as exc:
        with _isolated_platform_class_state():
            NeuronPlatform.check_and_update_config(cfg)
    message = str(exc.value)

    assert BF16_REFUSAL_SUBSTRING in message
    assert str(UNSUPPORTED_MODEL_DTYPE) in message
    assert str(HYBRID_BLOCK_SIZE) in message
    assert _cache_snapshot(cfg) == before
    assert cfg.cache_config.user_specified_block_size is False

    # The same dtype at an unsupported degree raises nothing, because the guard
    # lives inside the hybrid path and that path is not entered there.
    quiet_cfg = _build_config(
        model_dtype=UNSUPPORTED_MODEL_DTYPE, tp=UNSUPPORTED_TP_DEGREE
    )
    with _isolated_platform_class_state():
        NeuronPlatform.check_and_update_config(quiet_cfg)
    assert quiet_cfg.cache_config.user_specified_block_size is False


def test_unsupported_tp_degree_leaves_the_cache_config_unchanged() -> None:
    """With no knob set at TP != 64, nothing raises and nothing moves."""
    cfg = _build_config(tp=UNSUPPORTED_TP_DEGREE)
    assert cfg.additional_config == {}
    before = _cache_snapshot(cfg)

    raised = None
    with _isolated_platform_class_state(), _capture_platform_log() as handler:
        try:
            NeuronPlatform.check_and_update_config(cfg)
        except BaseException as exc:  # reported by the assertion below
            raised = f"{type(exc).__name__}: {exc}"
        engaged = _marker_records(handler, HYBRID_MARKER)

    assert raised is None, f"the non-engagement path raised: {raised}"
    assert _cache_snapshot(cfg) == before
    assert cfg.cache_config.user_specified_block_size is False
    assert engaged == []
    # The hybrid write-back happens only on engagement, so nothing downstream can
    # read a decision this path never took. Other keys in ``neuron_config`` are
    # written on every path, so the check names the hybrid one.
    assert "enable_hybrid_kv_cache" not in cfg.additional_config.get(
        "neuron_config", {}
    )

    # All four readings move at the supported degree, so the quiet tracks the
    # degree rather than a decision that never fires at all.
    control_cfg = _build_config()
    with _isolated_platform_class_state(), _capture_platform_log() as handler:
        NeuronPlatform.check_and_update_config(control_cfg)
        control_engaged = _marker_records(handler, HYBRID_MARKER)
    assert control_cfg.cache_config.block_size == HYBRID_BLOCK_SIZE
    assert control_cfg.cache_config.user_specified_block_size is True
    assert len(control_engaged) == 1
    assert "neuron_config" in control_cfg.additional_config


def test_unsupported_tp_degree_warns_once() -> None:
    """The declined engagement is audible: one WARNING, naming what was lost."""
    cfg = _build_config(tp=UNSUPPORTED_TP_DEGREE)
    assert cfg.additional_config == {}

    with _isolated_platform_class_state(), _capture_platform_log() as handler:
        NeuronPlatform.check_and_update_config(cfg)
        at_warning = _marker_records(handler, OFF_MARKER, level="WARNING")
        at_any_level = _marker_records(handler, OFF_MARKER)
        all_records = list(handler.records)

    assert len(at_any_level) == 1, f"expected exactly 1 record, got {at_any_level}"
    assert len(at_warning) == 1, f"the record was not at WARNING: {all_records}"
    message = at_warning[0]

    # It names both degrees -- what this run resolved and what is supported.
    # Matched on the rendered key=value pair, because the bare 32 also occurs in
    # the page the sentence promises.
    assert f"tensor_parallel_size={UNSUPPORTED_TP_DEGREE}" in message
    assert f"tensor_parallel_size={REQUIRED_TP_DEGREE}" in message
    # ... and the architecture, so one line tells the operator which model lost
    # the page.
    assert GLM5_NEXT_ARCH in message
    # ... and what re-enabling it elsewhere costs.
    assert "derive a block size" in message
    assert "enable_hybrid_kv_cache" in message
    assert HYBRID_MARKER not in message

    # The page the warning promises is the page the run gets, read from
    # update_block_size_for_backend rather than restated as a literal.
    NeuronPlatform.update_block_size_for_backend(cfg)
    delivered = cfg.cache_config.block_size
    assert f"{delivered}-token page" in message

    # Nothing warns at the supported degree, where the engagement happens.
    control_cfg = _build_config()
    with _isolated_platform_class_state(), _capture_platform_log() as handler:
        NeuronPlatform.check_and_update_config(control_cfg)
        control_off = _marker_records(handler, OFF_MARKER)
    assert control_off == []

    # Another architecture at the same unsupported degree is silent, so the
    # warning is scoped to this model instead of firing for every model.
    other_arch = _other_archs()[0]
    other_cfg = _build_config(arch=other_arch, tp=UNSUPPORTED_TP_DEGREE)
    with _isolated_platform_class_state(), _capture_platform_log() as handler:
        NeuronPlatform.check_and_update_config(other_cfg)
        other_off = _marker_records(handler, OFF_MARKER)
    assert other_off == []

    # An operator who set the knob is not lectured: an explicit False ends the
    # decision before the warning.
    explicit_cfg = _build_config(
        neuron_config={"enable_hybrid_kv_cache": False}, tp=UNSUPPORTED_TP_DEGREE
    )
    with _isolated_platform_class_state(), _capture_platform_log() as handler:
        NeuronPlatform.check_and_update_config(explicit_cfg)
        explicit_off = _marker_records(handler, OFF_MARKER)
    assert explicit_off == []


def test_off_warning_reports_the_operator_supplied_page() -> None:
    """With ``--block-size`` latched, the warning names the operator's page."""
    # The vendor latches an explicit block size, update_block_size_for_backend
    # then leaves it alone and the run allocates the operator's value, so a
    # sentence naming the default page would be the wrong number for anyone
    # sizing KV memory.
    cfg = _build_config(tp=UNSUPPORTED_TP_DEGREE, block_size=OPERATOR_BLOCK_SIZE)
    # The premise, asserted rather than assumed: the vendor really did latch.
    assert cfg.cache_config.user_specified_block_size is True
    assert cfg.cache_config.block_size == OPERATOR_BLOCK_SIZE

    with _isolated_platform_class_state(), _capture_platform_log() as handler:
        NeuronPlatform.check_and_update_config(cfg)
        at_warning = _marker_records(handler, OFF_MARKER, level="WARNING")

    assert len(at_warning) == 1, f"expected exactly 1 record, got {at_warning}"
    message = at_warning[0]

    # The delivered page comes from the method itself, called after the
    # decision, which is the order a real run has.
    NeuronPlatform.update_block_size_for_backend(cfg)
    delivered = cfg.cache_config.block_size
    assert delivered == OPERATOR_BLOCK_SIZE, (
        f"the operator's page did not survive: delivered {delivered}"
    )

    assert f"{delivered}-token page" in message
    assert f"{NeuronPlatform.UNIFORM_NEURON_PAGE}-token page" not in message
    # ... and says where the number came from, so the operator can tell the two
    # cases apart without reading this file.
    assert "supplied on the command line" in message

    # The warning's reader and the allocator's writer are pinned against each
    # other in both latch states, so neither can drift alone.
    latched = _build_config(tp=UNSUPPORTED_TP_DEGREE, block_size=OPERATOR_BLOCK_SIZE)
    predicted_latched = NeuronPlatform.resolved_uniform_page(latched)
    NeuronPlatform.update_block_size_for_backend(latched)
    unlatched = _build_config(tp=UNSUPPORTED_TP_DEGREE)
    predicted_unlatched = NeuronPlatform.resolved_uniform_page(unlatched)
    NeuronPlatform.update_block_size_for_backend(unlatched)
    assert predicted_latched == latched.cache_config.block_size
    assert predicted_unlatched == unlatched.cache_config.block_size
    assert predicted_latched != predicted_unlatched, (
        f"both latch states predicted {predicted_latched}; the prediction does "
        "not read the latch"
    )
    assert predicted_unlatched == NeuronPlatform.UNIFORM_NEURON_PAGE
