# SPDX-License-Identifier: Apache-2.0
"""The checkpoint tensors a load-time prep replaces are released once its operands exist.

Each class that prepares operands declares which loaded tensors its forward stops
reading; the load releases those and nothing else. These items load a miniature that
carries every class the load visits -- routed bank, shared expert, dense MLP, MLA and
indexer -- through the real loader on the CPU and read what the tree holds before and
after the release: one resident copy per declared tensor, the declaring classes' forwards
byte-equal to a twin loaded with the release held off, and every class that declares
nothing still holding what it loaded.
"""
import pytest
import torch

from vllm_neuron.model.glm5_next.config import Glm5NextConfig, Glm5NextTextConfig
from vllm_neuron.model.glm5_next.quantization import DEFAULT_WEIGHT_BLOCK_SIZE
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    DSA_SCALED_PROJECTIONS,
    FP8_SCALE_SUFFIX,
    block_grid_shape,
    scale_keys,
)

from .test_load_weights import (  # noqa: F401 -- the fixture is used by name
    MINI_FIRST_K_DENSE,
    MINI_LAYERS,
    MINI_MLA_WIDTHS,
    MINI_SHARED_EXPERTS,
    _keys_of,
    _mappings_for,
    _write_miniature_checkpoint,
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

#: The three MLP families' weights and grids, which the two MLP classes that declare
#: no release keep and the bank releases.
MLP_LEAVES = {f"{name}_weight" for name in BANK_PROJECTIONS} | {
    f"{name}_{FP8_SCALE_SUFFIX}" for name in BANK_PROJECTIONS
}

#: The miniature's forward geometry. The checkpoint loader admits whole 256 blocks,
#: so the residual width is 256; every MLP intermediate is 512 so each grid has more than
#: one block along both axes; the router selects exactly eight experts per token, so the
#: bank holds eight.
HIDDEN = 256
INTERMEDIATE = 512
ROUTED_EXPERTS = 8
TOP_K = 8
TOKENS = 256
SEED = 157


def say(*parts: object) -> None:
    """Print a reading. The suite runs under ``-s``, so these reach the transcript."""
    print(f"{SENT}|" + "|".join(str(p) for p in parts), flush=True)


def _impl():
    """Import the implementation module inside a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _forward_config() -> Glm5NextConfig:
    """The miniature with every class the load visits, at widths its forwards admit."""
    widths = dict(MINI_MLA_WIDTHS, hidden_size=HIDDEN)
    return Glm5NextConfig(
        text_config=Glm5NextTextConfig(
            num_hidden_layers=MINI_LAYERS,
            n_routed_experts=ROUTED_EXPERTS,
            n_shared_experts=MINI_SHARED_EXPERTS,
            num_experts_per_tok=TOP_K,
            first_k_dense_replace=MINI_FIRST_K_DENSE,
            moe_intermediate_size=INTERMEDIATE,
            intermediate_size=INTERMEDIATE,
            tie_word_embeddings=False,
            **widths,
        )
    )


def _fp8_values(generator: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    """Distinct finite fp8 bytes, so a forward over them is not a forward over ones."""
    bytes_ = torch.randint(8, 127, shape, generator=generator, dtype=torch.int32)
    signs = torch.randint(0, 2, shape, generator=generator, dtype=torch.int32) * 128
    return (bytes_ + signs).to(torch.uint8).view(torch.float8_e4m3fn)


def _pow2_grid(generator: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    """Block scales that are exact powers of two, one per block."""
    exponents = torch.randint(-3, 1, shape, generator=generator, dtype=torch.int32)
    return torch.pow(2.0, exponents.to(torch.float32))


def _mlp_overrides(
    generator: torch.Generator, path: str, mappings: dict
) -> dict[str, torch.Tensor]:
    """One MLP family's three weights and grids at the forward geometry, per key."""
    overrides: dict[str, torch.Tensor] = {}
    for name in BANK_PROJECTIONS:
        param = f"{path}.{name}_weight"
        if param not in mappings:
            continue
        shape = (HIDDEN, INTERMEDIATE) if name == "down_proj" else (INTERMEDIATE, HIDDEN)
        grid = block_grid_shape(shape, DEFAULT_WEIGHT_BLOCK_SIZE)
        keys = _keys_of(mappings, param)
        grids = set(scale_keys(keys))
        for key in keys:
            overrides[key] = (
                _pow2_grid(generator, grid)
                if key in grids
                else _fp8_values(generator, shape)
            )
    return overrides


def _forward_overrides(model: torch.nn.Module, mappings: dict) -> dict[str, torch.Tensor]:
    """Every tensor the miniature's forwards read, at the shapes those forwards take.

    The writer types every other key at an arbitrary placeholder shape, which the load
    accepts and a forward cannot. Nothing here spells a checkpoint key: the keys come
    from the map, the shapes from the config's own dials.
    """
    impl = _impl()
    generator = torch.Generator().manual_seed(SEED)
    overrides: dict[str, torch.Tensor] = {}
    for path, module in model.named_modules():
        if isinstance(
            module,
            (impl.Glm5NextRoutedExperts, impl.Glm5NextSharedExperts, impl.Glm5NextDenseMLP),
        ):
            overrides.update(_mlp_overrides(generator, path, mappings))
        if isinstance(module, impl.Glm5NextRoutedExperts):
            experts = int(module.num_routed_experts)
            for key in _keys_of(mappings, f"{path}.router_weight"):
                overrides[key] = (
                    torch.randn(experts, HIDDEN, generator=generator) * 0.1
                ).to(torch.bfloat16)
            for key in _keys_of(mappings, f"{path}.router_bias"):
                overrides[key] = torch.randn(experts, generator=generator) * 0.01
        if hasattr(type(module), "projection_widths"):
            sites = getattr(type(module), "PROJECTION_PARAMETERS", {})
            for name, idim, odim in module.projection_widths():
                quantised = name in DSA_SCALED_PROJECTIONS
                weight = f"{path}.{sites.get(name, f'{name}_weight')}"
                for key in _keys_of(mappings, weight) if weight in mappings else ():
                    overrides[key] = (
                        _fp8_values(generator, (odim, idim))
                        if quantised
                        else (torch.randn(odim, idim, generator=generator) * 0.05).to(
                            torch.bfloat16
                        )
                    )
                scale = f"{path}.{name}_{FP8_SCALE_SUFFIX}"
                for key in _keys_of(mappings, scale) if scale in mappings else ():
                    overrides[key] = _pow2_grid(
                        generator, block_grid_shape((odim, idim), DEFAULT_WEIGHT_BLOCK_SIZE)
                    )
        if isinstance(module, impl.Glm5NextMLAAttention):
            for leaf, width in (
                ("q_a_layernorm_weight", module.q_lora_rank),
                ("kv_a_layernorm_weight", module.kv_lora_rank),
            ):
                for key in _keys_of(mappings, f"{path}.{leaf}"):
                    overrides[key] = torch.ones(width, dtype=torch.bfloat16)
        if isinstance(module, impl.Glm5NextDSAIndexer):
            width = module.index_head_dim
            for key in _keys_of(mappings, f"{path}.k_norm_weight"):
                overrides[key] = torch.ones(width, dtype=torch.bfloat16)
            for key in _keys_of(mappings, f"{path}.k_norm_bias"):
                overrides[key] = torch.zeros(width, dtype=torch.bfloat16)
    return overrides


def _written_checkpoint(tmp_path):
    """The miniature's checkpoint on disk, written once for every model here."""
    directory = tmp_path / "forward"
    config = _forward_config()
    model = _impl().Glm5NextForConditionalGeneration(config)
    mappings = _mappings_for(config)
    written = _write_miniature_checkpoint(
        directory, mappings, model, extra_overrides=_forward_overrides(model, mappings)
    )
    say("fixture", f"tensors_written={written}|declared={len(model.declared_parameter_names())}")
    return directory


def _loaded(directory, monkeypatch=None):
    """A fresh miniature loaded from ``directory`` on the CPU.

    A ``monkeypatch`` holds the release off, which is the pre-release path.
    """
    impl = _impl()
    if monkeypatch is not None:
        monkeypatch.setattr(impl, "_release_replaced_parameters", lambda module, *names: 0)
    model = impl.Glm5NextForConditionalGeneration(_forward_config())
    model.load_weights(str(directory), torch.device("cpu"), None)
    return model


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


def _expected_released(module: torch.nn.Module) -> set[str] | None:
    """What each class is expected to release: a set, empty for a class whose forward
    reads what it loaded, ``None`` for a class no item expects to release anything."""
    impl = _impl()
    if isinstance(module, impl.Glm5NextRoutedExperts):
        return BANK_RELEASED
    if isinstance(module, impl.Glm5NextMLAAttention):
        return MLA_RELEASED
    if isinstance(module, impl.Glm5NextDSAIndexer):
        return INDEXER_RELEASED
    if isinstance(module, (impl.Glm5NextSharedExperts, impl.Glm5NextDenseMLP)):
        return set()
    return None


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-identical, compared as bytes so fp8 operands compare like any other."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _first(model: torch.nn.Module, cls) -> tuple[str, torch.nn.Module]:
    """The first module of ``cls`` in tree order, with its path."""
    for path, module in model.named_modules():
        if isinstance(module, cls):
            return path, module
    raise AssertionError(f"the miniature built no {cls.__name__}")


def _forward_inputs() -> dict[str, torch.Tensor]:
    """The activations every forward here takes, the same on both twins."""
    generator = torch.Generator().manual_seed(SEED + 1)
    hidden = torch.randn(TOKENS, HIDDEN, generator=generator).to(torch.bfloat16)
    x = hidden.to(torch.float32)
    normed = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)).to(torch.bfloat16)
    return {"hidden": hidden, "normed": normed, "gamma": torch.ones(HIDDEN, dtype=torch.bfloat16)}


def _forwards(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """One forward per class the load visits, on the loaded tree, keyed by what it produced."""
    impl = _impl()
    quant_config = impl.Glm5NextQuantConfig.from_model_config(model.config)
    _, block = _first(model, impl.Glm5NextMoEBlock)
    _, dense = _first(model, impl.Glm5NextDenseMLP)
    _, mla = _first(model, impl.Glm5NextMLAAttention)
    _, indexer = _first(model, impl.Glm5NextDSAIndexer)
    out = {
        "moe_block": block.forward(
            inputs["hidden"],
            inputs["normed"],
            router_gamma=inputs["gamma"],
            text_config=model.text_config,
            quant_config=quant_config,
        ),
        "dense_mlp": dense.forward(inputs["normed"], quant_config=quant_config),
    }
    query, key, value = mla.project_qkv(inputs["normed"])
    out.update(mla_query=query, mla_key=key, mla_value=value)
    out["mla_output"] = mla.project_output(value)
    q_latent = mla.project_query_latent(inputs["normed"])
    stage = indexer.project_stage(inputs["normed"], q_latent)
    out.update(zip(("indexer_query", "indexer_key", "indexer_weights", "indexer_gate"), stage))
    return out


def test_each_replaced_weight_is_resident_once_after_the_preps(
    tmp_path, single_rank_process_group, monkeypatch
) -> None:
    """(t1) After the load, every declared tensor is held once, as its prepared operand,
    and every class that declares nothing still holds what it loaded."""
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

    released_classes: set[str] = set()
    kept_classes: dict[str, int] = {}
    total_before = total_after = total_freed = 0
    for path, module in model.named_modules():
        after_count, after_bytes = _count(_held(module))
        total_after += after_bytes
        expected = _expected_released(module)
        if not expected:
            assert path not in by_module, (
                f"{type(module).__name__} at {path} released {sorted(by_module[path])}, "
                f"and no item expects it to"
            )
            assert id(module) not in recorder.before, f"{path} passed through the release"
            assert getattr(module, impl.RELEASED_PARAMETERS_ATTR, {}) == {}
            total_before += after_bytes
            if expected is not None:
                kept = [n for n in MLP_LEAVES if getattr(module, n, None) is not None]
                assert len(kept) == len(MLP_LEAVES), (
                    f"{path} keeps {sorted(kept)} of {sorted(MLP_LEAVES)}; its forward reads "
                    f"every one of them"
                )
                kept_classes[type(module).__name__] = kept_classes.get(type(module).__name__, 0) + 1
            continue
        before_count, before_bytes = recorder.before[id(module)]
        freed = recorder.freed[id(module)]
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
        assert set(freed) == by_module[path] == expected, (
            f"{path} freed {sorted(freed)}, recorded {sorted(by_module[path])}, "
            f"expected {sorted(expected)}"
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
        released_classes.add(type(module).__name__)

    say("kept", *(f"{cls}={n}" for cls, n in sorted(kept_classes.items())))
    say(
        "total",
        f"resident_before={total_before}|resident_after={total_after}|freed={total_freed}",
    )
    assert total_after == total_before - total_freed
    assert released_classes == {
        "Glm5NextRoutedExperts",
        "Glm5NextMLAAttention",
        "Glm5NextDSAIndexer",
    }, released_classes
    assert set(kept_classes) == {"Glm5NextSharedExperts", "Glm5NextDenseMLP"}, kept_classes


def test_the_forward_reads_the_operands_the_prep_stored(
    tmp_path, single_rank_process_group, monkeypatch
) -> None:
    """(t2) The forward reads the objects the prep stored, byte-equal to the pre-release path's."""
    impl = _impl()
    directory = _written_checkpoint(tmp_path)
    with monkeypatch.context() as held:
        reference = _loaded(directory, held)
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
        reference = _loaded(directory, held)
    bound_in_reference = 0
    for dotted in released:
        path, _, leaf = dotted.rpartition(".")
        if getattr(reference.get_submodule(path), leaf) is not None:
            bound_in_reference += 1

    dense = [
        (path, module)
        for path, module in model.named_modules()
        if isinstance(module, impl.Glm5NextDenseMLP)
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


def test_every_forward_is_byte_equal_with_the_release_on_and_off(
    tmp_path, single_rank_process_group, monkeypatch
) -> None:
    """(t4) One MoE-block forward, the dense MLP, the MLA projections and the indexer stage,
    run on the released tree, are byte-equal to the same runs on a twin loaded with the
    release held off."""
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    impl = _impl()
    directory = _written_checkpoint(tmp_path)
    with monkeypatch.context() as held:
        reference = _loaded(directory, held)
    model = _loaded(directory)
    released = model.released_parameters()
    assert released and reference.released_parameters() == {}

    shared = [m for m in model.modules() if isinstance(m, impl.Glm5NextSharedExperts)]
    assert shared, "the miniature built no shared expert, so the MoE forward would not read one"
    assert all(getattr(m, n) is not None for m in shared for n in MLP_LEAVES), (
        "a shared expert lost a weight or grid its forward reads"
    )

    inputs = _forward_inputs()
    got = _forwards(model, inputs)
    want = _forwards(reference, inputs)
    identical = [name for name in got if _same_bytes(got[name], want[name])]
    finite = [name for name, t in got.items() if bool(torch.isfinite(t.to(torch.float32)).all())]
    nonzero = [name for name, t in got.items() if bool((t.to(torch.float32) != 0).any())]
    say(
        "forward",
        f"released={len(released)}|shared_experts={len(shared)}",
        f"outputs={len(got)}|identical={len(identical)}",
        f"finite={len(finite)}|nonzero={len(nonzero)}",
        f"moe_block={tuple(got['moe_block'].shape)}:{got['moe_block'].dtype}",
    )
    assert set(identical) == set(got), f"differ: {sorted(set(got) - set(identical))}"
    assert set(finite) == set(got), f"not finite: {sorted(set(got) - set(finite))}"
    assert set(nonzero) == set(got), f"all zero: {sorted(set(got) - set(nonzero))}"


def test_releasing_the_shared_expert_makes_its_forward_refuse(
    tmp_path, single_rank_process_group, monkeypatch
) -> None:
    """(t5, control) A declaration that releases the shared expert's six tensors makes
    the MoE-block forward raise from ``scale_route_operands``, which is why the class
    declares none."""
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    impl = _impl()
    directory = _written_checkpoint(tmp_path)
    monkeypatch.setattr(
        impl.Glm5NextSharedExperts,
        "RELEASED_AFTER_PREP",
        impl.Glm5NextRoutedExperts.RELEASED_AFTER_PREP,
    )
    model = _loaded(directory)
    shared = [
        (path, m) for path, m in model.named_modules() if isinstance(m, impl.Glm5NextSharedExperts)
    ]
    released = model.released_parameters()
    unbound = [
        f"{path}.{n}" for path, m in shared for n in MLP_LEAVES if getattr(m, n, None) is None
    ]
    assert shared and len(unbound) == len(shared) * len(MLP_LEAVES), unbound

    with pytest.raises(impl.Glm5NextSharedExpertRouteError) as info:
        _forwards(model, _forward_inputs())
    raised_from = info.traceback[-1].name
    say(
        "control",
        f"shared_experts={len(shared)}|released={len(released)}|unbound={len(unbound)}",
        f"raised={type(info.value).__name__}|from={raised_from}",
    )
    assert raised_from == "scale_route_operands", raised_from
    assert f"gate_proj_{FP8_SCALE_SUFFIX} is not on this module" in str(info.value)
