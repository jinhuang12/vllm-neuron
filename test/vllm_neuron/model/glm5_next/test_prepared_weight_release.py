# SPDX-License-Identifier: Apache-2.0
"""The checkpoint tensors a load-time prep replaces are released once its operands exist.

Each prep copies its inputs into kernel orientation and the forward reads the copies
only, so the originals would otherwise stay resident beside them. These items load the
routed miniature on the CPU through the real loader and read what the tree holds before
and after the release: one resident copy per replaced weight, the same operand objects
and bytes the pre-release path built, and nothing left referencing what was freed.
"""
import pytest
import torch

from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    DSA_SCALED_PROJECTIONS,
    FP8_SCALE_SUFFIX,
)

from .test_load_weights import (  # noqa: F401 -- the fixture is used by name
    _mappings_for,
    _stacked_checkpoint,
    _stacked_config,
    _stacked_model,
    single_rank_process_group,
)

SENT = "RELEASE"

BANK_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
MLA_PROJECTIONS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
INDEXER_PARAMETERS = (
    "wq_b_weight",
    "wk_weight",
    "weights_proj_weight",
    "index_kpool_compress_gate",
)

BANK_RELEASED = {f"{name}_weight" for name in BANK_PROJECTIONS} | {
    f"{name}_{FP8_SCALE_SUFFIX}" for name in BANK_PROJECTIONS
}
MLA_RELEASED = {f"{name}_weight" for name in MLA_PROJECTIONS} | {
    f"{name}_{FP8_SCALE_SUFFIX}" for name in DSA_SCALED_PROJECTIONS
}
INDEXER_RELEASED = set(INDEXER_PARAMETERS)


def say(*parts: object) -> None:
    """Print a reading. The suite runs under ``-s``, so these reach the transcript."""
    print(f"{SENT}|" + "|".join(str(p) for p in parts), flush=True)


def _impl():
    """Import the implementation module inside a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _held(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Every tensor this module's own attributes hold, keyed by where it hangs."""
    held: dict[str, torch.Tensor] = {}
    for name, value in module._parameters.items():
        if value is not None:
            held[name] = value
    for name, value in module._buffers.items():
        if value is not None:
            held[name] = value
    for name, value in vars(module).items():
        if name in ("_parameters", "_buffers", "_modules"):
            continue
        if isinstance(value, torch.Tensor):
            held[name] = value
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, torch.Tensor):
                    held[f"{name}[{key}]"] = item
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, torch.Tensor):
                    held[f"{name}[{index}]"] = item
    return held


def _count(held: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Distinct tensors and their bytes, so a tensor hung under two names counts once."""
    unique = {id(tensor): tensor for tensor in held.values()}
    return len(unique), sum(t.numel() * t.element_size() for t in unique.values())


def _prepared_attributes(module: torch.nn.Module) -> list[str]:
    """The attribute names under which this module's class stores prepared operands."""
    names = []
    for constant in (
        "PREPARED_KERNEL_OPERANDS_ATTR",
        "PREPARED_SCALE_OPERANDS_ATTR",
        "PREPARED_WEIGHTS_ATTR",
        "ABSORB_WEIGHTS_ATTR",
    ):
        attribute = getattr(type(module), constant, None)
        if attribute and isinstance(getattr(module, attribute, None), dict):
            names.append(attribute)
    return names


class _Recorder:
    """Wrap the release: read the module just before it frees, and keep what it freed."""

    def __init__(self, original) -> None:
        self.original = original
        self.before: dict[int, tuple[int, int]] = {}
        self.freed: dict[int, dict[str, torch.Tensor]] = {}
        self.operands: dict[int, dict[str, dict[str, torch.Tensor]]] = {}

    def __call__(self, module: torch.nn.Module, *names: str) -> int:
        key = id(module)
        self.before.setdefault(key, _count(_held(module)))
        freed = self.freed.setdefault(key, {})
        for name in names:
            tensor = getattr(module, name, None)
            if tensor is not None:
                freed[name] = tensor
        self.operands[key] = {
            attribute: dict(getattr(module, attribute))
            for attribute in _prepared_attributes(module)
        }
        return self.original(module, *names)


def _expected_released(module: torch.nn.Module) -> set[str]:
    """The names each class is expected to release, spelled out."""
    impl = _impl()
    if isinstance(module, impl.Glm5NextRoutedExperts):
        return BANK_RELEASED
    if isinstance(module, impl.Glm5NextMLAAttention):
        return MLA_RELEASED
    if isinstance(module, impl.Glm5NextDSAIndexer):
        return INDEXER_RELEASED
    raise AssertionError(
        f"{type(module).__name__} released parameters, and no item expects it to"
    )


def _written_checkpoint(tmp_path):
    """The routed miniature's checkpoint on disk, written once for every model here."""
    directory = tmp_path / "stacked"
    model = _stacked_model()
    written = _stacked_checkpoint(directory, _mappings_for(_stacked_config()), model)
    say("fixture", f"tensors_written={written}|declared={len(model.declared_parameter_names())}")
    return directory


def _loaded(directory):
    """A fresh routed miniature loaded from ``directory`` on the CPU."""
    model = _stacked_model()
    model.load_weights(str(directory), torch.device("cpu"), None)
    return model


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-identical, compared as bytes so fp8 operands compare like any other."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def test_each_replaced_weight_is_resident_once_after_the_preps(
    tmp_path, single_rank_process_group, monkeypatch
) -> None:
    """(t1) After the load, every replaced weight is held once: as its prepared operand."""
    impl = _impl()
    recorder = _Recorder(impl._release_replaced_parameters)
    monkeypatch.setattr(impl, "_release_replaced_parameters", recorder)
    model = _loaded(_written_checkpoint(tmp_path))

    released = model.released_parameters()
    by_module: dict[str, set[str]] = {}
    for dotted in released:
        path, _, leaf = dotted.rpartition(".")
        by_module.setdefault(path, set()).add(leaf)
    say("released", f"parameters={len(released)}|modules={len(by_module)}")
    assert by_module, "no module released anything after its prep"

    classes = set()
    total_before = total_after = total_freed = 0
    for path, module in model.named_modules():
        after_count, after_bytes = _count(_held(module))
        total_after += after_bytes
        if id(module) not in recorder.before:
            assert path not in by_module, f"{path} released without passing the release"
            assert getattr(module, impl.RELEASED_PARAMETERS_ATTR, {}) == {}
            total_before += after_bytes
            continue
        before_count, before_bytes = recorder.before[id(module)]
        freed = recorder.freed[id(module)]
        if not freed:
            say("module", type(module).__name__, path, "freed=0:0")
            assert path not in by_module, f"{path} recorded a release it did not make"
            total_before += after_bytes
            continue
        freed_bytes = sum(t.numel() * t.element_size() for t in freed.values())
        total_before += before_bytes
        total_freed += freed_bytes
        say(
            "module",
            type(module).__name__,
            path,
            f"before={before_count}:{before_bytes}",
            f"after={after_count}:{after_bytes}",
            f"freed={len(freed)}:{freed_bytes}",
        )
        assert set(freed) == by_module[path] == _expected_released(module), (
            f"{path} freed {sorted(freed)}, recorded {sorted(by_module[path])}, "
            f"expected {sorted(_expected_released(module))}"
        )
        one_copy_each = (before_count - len(freed), before_bytes - freed_bytes)
        assert (after_count, after_bytes) == one_copy_each, (
            f"{path} holds {after_count} tensors / {after_bytes} bytes after the release; "
            f"{one_copy_each[0]} / {one_copy_each[1]} would be one copy each"
        )
        for name in freed:
            assert getattr(module, name) is None, f"{path}.{name} is still bound"
        for attribute in _prepared_attributes(module):
            assert getattr(module, attribute), f"{path}.{attribute} holds no operand"
        classes.add(type(module).__name__)

    dense = [
        (path, module)
        for path, module in model.named_modules()
        if hasattr(type(module), "retile_checkpoint_scale_grids")
        and not hasattr(type(module), "prepare_scale_operands")
    ]
    say(
        "control",
        f"dense_mlps={len(dense)}",
        f"dense_weights_kept={sum(1 for _, m in dense if m.gate_proj_weight is not None)}",
    )
    assert dense and all(m.gate_proj_weight is not None for _, m in dense), (
        "the dense MLP reads its raw weights in the forward, so the release must not touch it"
    )
    say(
        "total",
        f"resident_before={total_before}|resident_after={total_after}|freed={total_freed}",
    )
    assert total_after == total_before - total_freed
    assert classes == {
        "Glm5NextRoutedExperts",
        "Glm5NextMLAAttention",
        "Glm5NextDSAIndexer",
    }, classes


def test_the_forward_reads_the_operands_the_prep_stored(
    tmp_path, single_rank_process_group, monkeypatch
) -> None:
    """(t2) The forward reads the objects the prep stored, byte-equal to the pre-release path's."""
    impl = _impl()
    directory = _written_checkpoint(tmp_path)
    with monkeypatch.context() as held:
        held.setattr(impl, "_release_replaced_parameters", lambda module, *names: 0)
        reference = _loaded(directory)
    assert reference.released_parameters() == {}, "the pre-release path released something"

    recorder = _Recorder(impl._release_replaced_parameters)
    monkeypatch.setattr(impl, "_release_replaced_parameters", recorder)
    model = _loaded(directory)

    twins = dict(reference.named_modules())
    compared = identical = through_accessor = 0
    for path, module in model.named_modules():
        if id(module) not in recorder.operands:
            continue
        for attribute, stored in recorder.operands[id(module)].items():
            live = getattr(module, attribute)
            twin = getattr(twins[path], attribute)
            assert set(live) == set(stored) == set(twin), (path, attribute)
            for key in stored:
                assert live[key] is stored[key], f"{path}.{attribute}[{key}] was rebuilt"
                identical += 1
                assert _same_bytes(live[key], twin[key]), (
                    f"{path}.{attribute}[{key}] differs from the pre-release path"
                )
                compared += 1
        accessor = getattr(module, "_prepared_kernel_operand", None) or getattr(
            module, "_prepared_weight", None
        )
        primary = getattr(type(module), "PREPARED_KERNEL_OPERANDS_ATTR", None) or getattr(
            type(module), "PREPARED_WEIGHTS_ATTR", None
        )
        for key, tensor in recorder.operands[id(module)].get(primary, {}).items():
            assert accessor(key) is tensor, f"{path} reads {key} from somewhere else"
            through_accessor += 1
    say(
        "operands",
        f"compared={compared}|identical={identical}|through_accessor={through_accessor}",
    )
    assert compared > 0 and identical == compared and through_accessor > 0


def test_nothing_in_the_tree_references_a_released_tensor(
    tmp_path, single_rank_process_group, monkeypatch
) -> None:
    """(t3) Nothing in the tree still holds a freed tensor, and the dense MLP keeps its raw weights."""
    impl = _impl()
    directory = _written_checkpoint(tmp_path)
    recorder = _Recorder(impl._release_replaced_parameters)
    monkeypatch.setattr(impl, "_release_replaced_parameters", recorder)
    model = _loaded(directory)

    freed = [tensor for held in recorder.freed.values() for tensor in held.values()]
    freed_ids = {id(tensor) for tensor in freed}
    freed_storages = {tensor.untyped_storage().data_ptr() for tensor in freed}
    referencing = []
    for path, module in model.named_modules():
        for where, tensor in _held(module).items():
            if id(tensor) in freed_ids or tensor.untyped_storage().data_ptr() in freed_storages:
                referencing.append(f"{path}.{where}")

    released = model.released_parameters()
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    still_named = [name for name in released if name in parameters or name in buffers]
    still_bound = []
    for dotted in released:
        path, _, leaf = dotted.rpartition(".")
        if getattr(model.get_submodule(path), leaf) is not None:
            still_bound.append(dotted)

    with monkeypatch.context() as held:
        held.setattr(impl, "_release_replaced_parameters", lambda module, *names: 0)
        reference = _loaded(directory)
    bound_in_reference = 0
    for dotted in released:
        path, _, leaf = dotted.rpartition(".")
        if getattr(reference.get_submodule(path), leaf) is not None:
            bound_in_reference += 1

    dense = [
        (path, module)
        for path, module in model.named_modules()
        if hasattr(type(module), "retile_checkpoint_scale_grids")
        and not hasattr(type(module), "prepare_scale_operands")
    ]
    raw_names = [f"{name}_weight" for name in BANK_PROJECTIONS]
    dense_released = [path for path, _ in dense if any(f"{path}.{n}" in released for n in raw_names)]
    dense_unbound = [
        path for path, module in dense if any(getattr(module, n) is None for n in raw_names)
    ]
    twin_dense = reference.get_submodule(dense[0][0]) if dense else None
    if twin_dense is not None:
        recorder.original(twin_dense, "gate_proj_weight")
    control_reads_unbound = twin_dense is not None and twin_dense.gate_proj_weight is None

    say(
        "references",
        f"freed={len(freed)}|released={len(released)}|referencing={len(referencing)}",
        f"still_named={len(still_named)}|still_bound={len(still_bound)}",
        f"control_bound_without_the_release={bound_in_reference}",
    )
    say(
        "dense_mlp",
        f"modules={len(dense)}|released={len(dense_released)}|unbound={len(dense_unbound)}",
        f"control_release_on_the_twin_reads_unbound={control_reads_unbound}",
    )
    assert dense and dense_released == [] and dense_unbound == [], (
        f"the dense MLP reads its raw weights in the forward and must keep them: "
        f"released={dense_released} unbound={dense_unbound}"
    )
    assert control_reads_unbound, "the control did not move: a release on the twin's dense MLP was not read"
    assert freed and len(freed) == len(released)
    assert (referencing, still_named, still_bound) == ([], [], []), (
        f"referencing={referencing[:4]} still_named={still_named[:4]} still_bound={still_bound[:4]}"
    )
    assert bound_in_reference == len(released), (
        "the control did not move: without the release the same names should all be bound"
    )
