# SPDX-License-Identifier: Apache-2.0
"""Engine port: ports, sharding, shape propagation, FLOPs/bytes, roofline, graph order."""

from __future__ import annotations

import pytest

from test.kernel_ledger.engine.collectives import ConstantCollectiveModel
from test.kernel_ledger.engine.graph import ForwardPassGraph
from test.kernel_ledger.engine.node import (
    HBM_BW,
    PEAK_FLOPS,
    CollectiveDim,
    CollectiveNode,
    CollectiveType,
    KernelNode,
    MLPNode,
    OpType,
    Port,
    RooflineMode,
    ShardingSpec,
    WeightSharding,
)


def _root(name="x", shape=(1, 1, 4096), dtype="bf16"):
    return KernelNode(
        name,
        input_ports={"input": Port("x", shape, dtype)},
        op_type=OpType.NONE,
    )


def test_port_bytes_follow_dtype():
    assert Port("a", (2, 3, 4), "bf16").size_bytes == 48
    assert Port("a", (2, 3, 4), "fp32").size_bytes == 96
    assert Port("a", (2, 3, 4), "fp8").size_bytes == 24


def test_tp_head_sharding_gives_one_head_per_rank():
    node = KernelNode(
        "q",
        weights={"q": (64, 4096, 128)},
        sharding=ShardingSpec(weight=[(WeightSharding.TP_HEAD, 64)]),
        input_ports={"input": Port("x", (1, 1, 4096))},
    )
    assert node._apply_weight_sharding((64, 4096, 128)) == (1, 4096, 128)
    # input (B, S, H) x weight (heads, H, Dh) -> (B, S, heads, Dh); the source
    # engine names a matmul output after its weight.
    assert node.output_ports["q"].shape == (1, 1, 1, 128)


def test_one_input_many_weights_gives_one_output_per_weight():
    node = KernelNode(
        "proj",
        weights={"a": (4096, 1536), "b": (4096, 512)},
        input_ports={"input": Port("x", (4, 1, 4096))},
    )
    assert node.output_ports["a"].shape == (4, 1, 1536)
    assert node.output_ports["b"].shape == (4, 1, 512)


def test_contraction_mismatch_is_refused():
    node = KernelNode(
        "bad",
        weights={"w": (1024, 8)},
        input_ports={"input": Port("x", (1, 1, 4096))},
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        node.output_ports


def test_matmul_flops_and_weight_bytes():
    node = KernelNode(
        "gemv",
        weights={"w": (4096, 512)},
        weight_dtype="fp8",
        input_ports={"input": Port("x", (2, 1, 4096))},
    )
    assert node.compute_flops() == 2 * 2 * 4096 * 512
    assert node._total_memory_bytes() == 4096 * 512


def test_roofline_max_is_memory_bound_for_a_gemv():
    node = KernelNode(
        "gemv",
        weights={"w": (4096, 4096)},
        weight_dtype="bf16",
        input_ports={"input": Port("x", (1, 1, 4096))},
        roofline_mode=RooflineMode.MAX,
        min_latency_us=0.001,
    )
    memory_us = 4096 * 4096 * 2 / HBM_BW * 1e6
    compute_us = 2 * 4096 * 4096 / PEAK_FLOPS * 1e6
    assert memory_us > compute_us
    assert node.compute_roofline() == pytest.approx(memory_us)


def test_roofline_never_goes_below_the_node_floor():
    node = KernelNode(
        "tiny",
        weights={"w": (128, 8)},
        input_ports={"input": Port("x", (1, 1, 128))},
        roofline_mode=RooflineMode.MAX,
        min_latency_us=2.0,
    )
    assert node.compute_roofline() == pytest.approx(2.0)


def test_mlp_node_flops_use_the_local_intermediate():
    node = MLPNode(
        "mlp",
        weights={"gate_up": (4096, 2 * 12288), "down": (12288, 4096)},
        sharding=ShardingSpec(weight=[(WeightSharding.TP_HIDDEN, 64)]),
        weight_dtype="fp8",
    )
    node.set_input_port("input", Port("x", (1, 1, 4096)))
    assert node.compute_flops() == 2 * 3 * 1 * 4096 * 192


def test_all_gather_output_and_message_size():
    ag = CollectiveNode(
        "ag",
        collective_type=CollectiveType.ALL_GATHER,
        group_size=64,
        dim=CollectiveDim.HIDDEN,
        dim_idx=2,
    )
    ag.set_input_port("input", Port("logits", (1, 1, 2420), "bf16"))
    assert ag.output_ports["output"].shape == (1, 1, 154880)
    assert ag.message_size_bytes == 154880 * 2


def test_all_reduce_keeps_shape():
    ar = CollectiveNode(
        "ar",
        collective_type=CollectiveType.ALL_REDUCE,
        group_size=64,
        dim=CollectiveDim.HIDDEN,
    )
    ar.set_input_port("input", Port("y", (1, 1, 4096), "fp32"))
    assert ar.output_ports["output"].shape == (1, 1, 4096)
    assert ar.message_size_bytes == 16384


def test_constant_collective_model_is_latency_plus_bytes():
    model = ConstantCollectiveModel(latency_us=17.5, bandwidth_bytes_per_s=100e9)
    assert model.lookup_us("AllReduce", "g64_s1", 0) == pytest.approx(17.5)
    assert model.lookup_us("AllReduce", "g64_s1", 16384) == pytest.approx(17.5 + 0.16384)


def test_connect_propagates_shapes_downstream():
    g = ForwardPassGraph("t", "decode")
    x = g.add_node(_root())
    a = g.add_node(KernelNode("a", weights={"w": (4096, 512)}))
    b = g.add_node(KernelNode("b", weights={"w": (512, 4096)}))
    g.connect(x, "output", a, "input")
    g.connect(a, "w", b, "input")
    assert b.input_ports["input"].shape == (1, 1, 512)
    assert b.output_ports["w"].shape == (1, 1, 4096)


def test_connect_refuses_an_unknown_port():
    g = ForwardPassGraph("t", "decode")
    x = g.add_node(_root())
    a = g.add_node(KernelNode("a", weights={"w": (4096, 512)}))
    with pytest.raises(ValueError, match="no output port"):
        g.connect(x, "nope", a, "input")


def test_reshape_checks_element_count():
    g = ForwardPassGraph("t", "decode")
    x = g.add_node(_root(shape=(1, 1, 4, 4096)))
    r = g.reshape(x, "output", (1, 1, 16384))
    assert r.output_ports["output"].shape == (1, 1, 16384)
    with pytest.raises(ValueError, match="element count"):
        g.reshape(x, "output", (1, 1, 4096))


def test_merge_joins_same_shaped_branches_and_refuses_others():
    g = ForwardPassGraph("t", "decode")
    x = g.add_node(_root())
    a = g.add_node(KernelNode("a", weights={"w": (4096, 4096)}))
    b = g.add_node(KernelNode("b", weights={"w": (4096, 4096)}))
    c = g.add_node(KernelNode("c", weights={"w": (4096, 128)}))
    for n in (a, b, c):
        g.connect(x, "output", n, "input")
    m = g.merge([(a, "w"), (b, "w")], name="m")
    assert m.output_ports["output"].shape == (1, 1, 4096)
    with pytest.raises(ValueError, match="merge"):
        g.merge([(a, "w"), (c, "w")], name="bad")


def test_resolve_order_is_topological_and_detects_cycles():
    g = ForwardPassGraph("t", "decode")
    x = g.add_node(_root())
    a = g.add_node(KernelNode("a", weights={"w": (4096, 4096)}))
    b = g.add_node(KernelNode("b", weights={"w": (4096, 4096)}))
    g.connect(x, "output", b, "input")
    g.connect(b, "w", a, "input")
    names = [n.name for n in g.resolve_order()]
    assert names.index("x") < names.index("b") < names.index("a")
    g.connect(a, "w", b, "input")
    with pytest.raises(ValueError, match="cycle"):
        g.resolve_order()
