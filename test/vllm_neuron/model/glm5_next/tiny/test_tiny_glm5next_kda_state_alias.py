"""Every recurrent layer that shares one raw KV tensor keeps its own state bytes.

vLLM's hybrid allocator hands one raw tensor to one layer of every KV cache group
(``KVCacheTensor.shared_by``). The recurrent layers address their state by request slot, so
two recurrent layers over one raw tensor need distinct storage offsets, or the slot one
layer writes is the slot its sharer reads. Three readings over the KV cache configuration
vLLM builds from the runner's own specs: the storage identity of every recurrent state view;
a write through one layer's slot 0 read back through its sharer's slot 0; and a stack of two
real recurrent layers, driven by the runner's own carriers for a prefill and one decode
step, which must end with different states, each equal to the state that layer reaches when
it runs alone.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_kda_state_alias.py
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_uniform_page_size,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next import test_kda_layer as layer_half
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as first
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

# Two recurrent layers over one shared tensor is the smallest stack that can alias. The
# prompt is long enough to run the chunked prefill before the decode step.
RECURRENT_LAYERS = 2
PREFILL_TOKENS = layer_half.DECLARED_PREFILL_TOKENS


def _production_kv_cache_config(runner: NeuronModelRunner, vllm_config, specs) -> KVCacheConfig:
    """The configuration vLLM's allocator builds from the runner's specs."""
    groups = get_kv_cache_groups(vllm_config, specs)
    page_size = get_uniform_page_size([group.kv_cache_spec for group in groups])
    group_size = max(len(group.layer_names) for group in groups)
    return get_kv_cache_config_from_groups(vllm_config, groups, page_size * group_size * (e2e.E2E_BLOCKS + 1))


def _two_recurrent_then_attention(root) -> KVSpec:
    """The tiny root's spec with its first two layers recurrent.

    That makes the recurrent layers outnumber the attention layer, so the two families land
    in separate KV cache groups and the first raw tensor is shared by recurrent layers.
    """
    recurrent = e2e._recurrent_spec(root).layers
    original = root.get_kv_spec().layers
    return KVSpec(layers=list(recurrent[:2]) + list(original[2:]))


def _identity(view: torch.Tensor) -> tuple[int, int]:
    """Where a view's bytes start: its storage address and its element offset."""
    return view.untyped_storage().data_ptr(), view.storage_offset()


@pytest.fixture
def shared_tensor(tmp_path, monkeypatch):
    """The recurrent sharers of the first raw tensor, and the state views bound for them."""
    e2e._require_cpu_mode()
    config = first._engine_config()
    root = e2e._fixture()["root"]
    spec = _two_recurrent_then_attention(root)
    monkeypatch.setattr(root, "get_kv_spec", lambda: spec)
    with first._parallel_state(tmp_path, config):
        runner = NeuronModelRunner(config, device=torch.device("cpu"))
        runner.model = root
        runner.vocab_size = tiny.STACK_VOCAB_SIZE
        specs = runner.get_kv_cache_spec()
        kv_cache_config = _production_kv_cache_config(runner, config, specs)
        kv = runner.initialize_kv_cache(kv_cache_config)
    tensors = kv_cache_config.kv_cache_tensors
    sharers = [name for name in tensors[0].shared_by if isinstance(specs[name], MambaSpec)]
    assert len(sharers) >= 2, (
        f"the first raw tensor is shared by fewer than two recurrent layers: "
        f"{tensors[0].shared_by}"
    )
    return sharers, kv


def test_recurrent_sharers_of_one_raw_tensor_have_distinct_storage_or_offset(shared_tensor):
    """No two recurrent layers start their first state view at the same bytes."""
    sharers, kv = shared_tensor
    identities = {name: _identity(kv[name][0]) for name in sharers}
    aliased = sorted(
        name for name, identity in identities.items()
        if list(identities.values()).count(identity) > 1
    )
    assert not aliased, f"recurrent layers over one raw tensor start at the same storage and offset: {aliased}"


def test_a_slot_written_through_one_recurrent_layer_stays_zero_in_its_sharer(shared_tensor):
    """Filling slot 0 of one sharer's first state leaves its sharer's slot 0 zero."""
    sharers, kv = shared_tensor
    a, b = sharers[0], sharers[1]
    before = kv[b][0][0].abs().sum().item()
    kv[a][0][0].fill_(1)
    after = kv[b][0][0].abs().sum().item()
    assert before == 0, f"{b} slot 0 was not zero before the write: {before}"
    assert after == 0, f"a slot written through {a} is read back through {b}: {after} nonzero"


def _recurrent_layers(count: int) -> SimpleNamespace:
    """``count`` real recurrent layers at the served tensor-parallel degree, seeded."""
    text_config = Glm5NextTextConfig()
    linear_attn = text_config.linear_attn_config
    heads = int(linear_attn["num_heads"]) // layer_half.TP_WORLD_SIZE
    head_dim = int(linear_attn["head_dim"])
    kernel = int(linear_attn["short_conv_kernel_size"])
    hidden = int(text_config.hidden_size)
    torch.manual_seed(layer_half.SEED)
    weights = [layer_half._make_weights(hidden, heads, head_dim, kernel) for _ in range(count)]
    layers = []
    for index, layer_weights in enumerate(weights):
        layer = layer_half._impl().Glm5NextKDALayer(text_config, index, layer_half.TP_WORLD_SIZE)
        for name, tensor in layer_weights.items():
            target = layer if name == "input_layernorm_weight" else layer.attention
            setattr(target, name, torch.nn.Parameter(tensor.clone(), requires_grad=False))
        layers.append(layer)
    return SimpleNamespace(
        layers=layers, weights=weights, heads=heads, head_dim=head_dim, kernel=kernel, hidden=hidden,
        eps=float(text_config.rms_norm_eps),
    )


def _state_page_bytes(layer) -> int:
    """The bytes one request slot of a recurrent layer's two states takes."""
    attention = layer.attention
    return sum(
        math.prod(shape) * dtype.itemsize
        for shape, dtype in (
            (attention.kda_conv_state_shape, attention.kda_conv_state_dtype),
            (attention.kda_recurrent_state_shape, attention.kda_recurrent_state_dtype),
        )
    )


def _spec_with_real_recurrent_layers(root, layers, block_size: int) -> KVSpec:
    """The tiny root's spec carrying the real recurrent layers' own state geometry.

    The attention layers are widened until one attention page holds one recurrent slot,
    which is what the runner's page unification requires.
    """
    original = root.get_kv_spec().layers
    recurrent = [
        LayerSpec(
            name=spec.name,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=spec.dtype,
            sliding_window_size=None,
            chunk_size=None,
            kda_conv_state_shape=tuple(layer.attention.kda_conv_state_shape),
            kda_recurrent_state_shape=tuple(layer.attention.kda_recurrent_state_shape),
            kda_conv_state_dtype=layer.attention.kda_conv_state_dtype,
            kda_recurrent_state_dtype=layer.attention.kda_recurrent_state_dtype,
        )
        for spec, layer in zip(original, layers)
    ]
    state_page = max(_state_page_bytes(layer) for layer in layers)
    attention = []
    for spec in original[len(layers):]:
        per_token = block_size * int(spec.num_kv_heads) * spec.dtype.itemsize
        width = -(-(-(-state_page // per_token)) // 128) * 128
        attention.append(dataclasses.replace(spec, head_size=max(int(spec.head_size), width)))
    return KVSpec(layers=recurrent + attention)


def _reference_states(world: SimpleNamespace, rows: torch.Tensor) -> list[torch.Tensor]:
    """Each layer's recurrent state after the prefill and the decode step, every layer run alone."""
    histories = [
        torch.zeros(world.kernel - 1, 3 * world.heads * world.head_dim, dtype=torch.float32)
        for _ in world.layers
    ]
    carried: list = [None] * len(world.layers)
    conv_dtype = world.layers[0].attention.kda_conv_state_dtype
    for start, end in ((0, PREFILL_TOKENS), (PREFILL_TOKENS, PREFILL_TOKENS + 1)):
        out = rows[start:end]
        for index, weights in enumerate(world.weights):
            out, histories[index], carried[index] = layer_half._reference_layer(
                out, weights, histories[index], carried[index],
                heads=world.heads, head_dim=world.head_dim, eps=world.eps, conv_carrier_dtype=conv_dtype,
            )
    return [torch.stack(state) for state in carried]


def test_two_steps_leave_each_recurrent_sharer_its_own_state(tmp_path, monkeypatch):
    """A prefill and one decode leave each sharer its own state, equal to its solo run."""
    e2e._require_cpu_mode()
    world = _recurrent_layers(RECURRENT_LAYERS)
    config = first._engine_config()
    root = e2e._fixture()["root"]
    spec = _spec_with_real_recurrent_layers(root, world.layers, int(config.cache_config.block_size))
    monkeypatch.setattr(root, "get_kv_spec", lambda: spec)
    torch.manual_seed(layer_half.SEED + 1)
    rows = torch.randn(PREFILL_TOKENS + 1, world.hidden, dtype=torch.float32)
    blocks = list(range(1, 1 + e2e._blocks_for(PREFILL_TOKENS + 1)))
    with first._parallel_state(tmp_path, config):
        runner = NeuronModelRunner(config, device=torch.device("cpu"))
        runner.model = root
        runner.vocab_size = tiny.STACK_VOCAB_SIZE
        specs = runner.get_kv_cache_spec()
        kv_cache_config = _production_kv_cache_config(runner, config, specs)
        runner.initialize_kv_cache(kv_cache_config)
        monkeypatch.setattr(runner, "input_batch", SimpleNamespace(req_ids=[first.REQUEST]))
        banks = root.glm5next_layer_banks
        recurrent = [bank for bank in banks if bank["family"] == "linear_attn"]
        sharers = [
            name for name in kv_cache_config.kv_cache_tensors[0].shared_by if isinstance(specs[name], MambaSpec)
        ]
        assert [bank["name"] for bank in recurrent] == sharers and len(sharers) == RECURRENT_LAYERS, (
            f"every recurrent layer must share the first raw tensor, or nothing here can "
            f"alias: {sharers}"
        )

        def step(tokens: int, cached: int) -> list[dict]:
            metadata = e2e._grouped_metadata(
                banks, tokens=tokens, sparse_row=blocks, state_row=[blocks[0]],
                sparse_cached=cached, state_cached=cached,
            )
            kwargs = runner._glm5next_model_kwargs(
                e2e._generic(tokens=tokens, metadata=metadata, sampling_row=tokens - 1)
            )
            carriers = kwargs["layer_carriers"]
            # The runner pads a prefill to the widest configured bucket and builds every
            # carrier at that width, so the layers run on rows as wide as the input_ids it
            # returned. The real rows are still `tokens` of them, and `row_mask` is what
            # tells a layer which rows those are.
            out = rows[cached : cached + tokens]
            width = int(kwargs["input_ids"].shape[0])
            if width > tokens:
                out = torch.cat([out, out.new_zeros(width - tokens, world.hidden)])
            for layer, carrier in zip(world.layers, carriers):
                out = layer(out, **carrier, chunk_size=layer_half.DECLARED_CHUNK)
            return carriers

        prefill = step(PREFILL_TOKENS, 0)
        decode = step(1, PREFILL_TOKENS)
        slot = runner._glm5next_request_slot_table[first.REQUEST]
    assert all(carrier["is_prefill"] is True for carrier in prefill[:RECURRENT_LAYERS])
    assert all(carrier["is_prefill"] is False for carrier in decode[:RECURRENT_LAYERS])

    states = [bank["recurrent_state"][slot].float() for bank in recurrent]
    references = _reference_states(world, rows)
    tolerance = layer_half.DECLARED_ATOL + layer_half.DECLARED_RTOL * max(float(r.abs().max()) for r in references)
    reference_apart = float((references[0] - references[1]).abs().max())
    if reference_apart <= tolerance:
        raise tiny.VacuousControlError(
            f"the two reference layers end {reference_apart:.6g} apart, inside the tolerance {tolerance:.6g}"
        )
    apart = float((states[0] - states[1]).abs().max())
    assert apart > tolerance, f"the two recurrent sharers hold one state: apart by {apart:.6g}"
    for name, state, reference in zip(sharers, states, references):
        torch.testing.assert_close(
            state, reference, rtol=layer_half.DECLARED_RTOL, atol=layer_half.DECLARED_ATOL, msg=name
        )
