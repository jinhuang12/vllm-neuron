# SPDX-License-Identifier: Apache-2.0
"""The row-parallel all-reduce entry point and its two switches.

``collective_policy.reduce_row_parallel`` is the one call every row-parallel site makes.
Two switches, registered in ``vllm_neuron.envs`` and read when a site runs:

* ``VLLM_NEURON_TP_ALLREDUCE_DTYPE`` = ``fp32`` (default) | ``bf16``: the dtype the
  partial crosses the wire in. ``fp32`` is the as-built path, bit for bit: the
  coordinator is handed the very partial and reduces it in place. ``bf16`` reduces a
  bfloat16 copy and returns it.
* ``VLLM_NEURON_TP_ALLREDUCE_FUSE`` = ``0`` (default) | ``1``: names the neuronx-cc
  argument that keeps each all-reduce one collective, for the runner to append.

    VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/test_collective_policy.py
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import operator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

from vllm_neuron import envs
from vllm_neuron.model.glm5_next import collective_policy as policy
from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.glm5_next.config import Glm5NextConfig

pytestmark = [pytest.mark.fast]

DTYPE_ENV = "VLLM_NEURON_TP_ALLREDUCE_DTYPE"
FUSE_ENV = "VLLM_NEURON_TP_ALLREDUCE_FUSE"
SITE = policy.RowParallelSite.FFN


class _RecordingGroup:
    """A coordinator stand-in: records what it was handed and doubles it in place."""

    def __init__(self) -> None:
        self.handed: list[torch.Tensor] = []

    def all_reduce(self, tensor: torch.Tensor) -> None:
        # The statement form vLLM's device communicator implements: the reduction
        # lands in the tensor it was handed.
        self.handed.append(tensor)
        tensor.mul_(2)


def _partial(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(20261007)
    return torch.randn(8, 16, generator=gen, dtype=torch.float32).to(dtype)


# ── the switches ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name, default", [(DTYPE_ENV, "fp32"), (FUSE_ENV, False)])
def test_the_switches_are_registered_in_envs(monkeypatch, name, default):
    monkeypatch.delenv(name, raising=False)
    assert name in envs.environment_variables
    assert getattr(envs, name) == default


def test_the_policy_names_the_registered_switches():
    assert (policy.DTYPE_ENV, policy.FUSE_ENV) == (DTYPE_ENV, FUSE_ENV)


def test_the_defaults_are_the_as_built_path(monkeypatch):
    monkeypatch.delenv(DTYPE_ENV, raising=False)
    monkeypatch.delenv(FUSE_ENV, raising=False)
    assert policy.allreduce_wire_dtype() is torch.float32
    assert policy.allreduce_fuse_enabled() is False
    assert policy.fuse_compiler_args() == []


@pytest.mark.parametrize(
    "value, dtype",
    [("fp32", torch.float32), ("bf16", torch.bfloat16), (" BF16 ", torch.bfloat16)],
)
def test_the_dtype_switch_reads_its_two_values(monkeypatch, value, dtype):
    monkeypatch.setenv(DTYPE_ENV, value)
    assert policy.allreduce_wire_dtype() is dtype


@pytest.mark.parametrize("value", ["fp16", "float32", "1", ""])
def test_an_unknown_dtype_is_refused_by_name(monkeypatch, value):
    monkeypatch.setenv(DTYPE_ENV, value)
    with pytest.raises(policy.CollectivePolicyError, match=DTYPE_ENV):
        policy.allreduce_wire_dtype()


@pytest.mark.parametrize("value, enabled", [("0", False), ("1", True)])
def test_the_fuse_switch_reads_its_two_values(monkeypatch, value, enabled):
    monkeypatch.setenv(FUSE_ENV, value)
    assert policy.allreduce_fuse_enabled() is enabled
    expected = [policy.FUSE_COMPILER_ARG] if enabled else []
    assert policy.fuse_compiler_args() == expected


@pytest.mark.parametrize("value", ["yes", "true", ""])
def test_a_non_integer_fuse_value_is_refused_by_name(monkeypatch, value):
    monkeypatch.setenv(FUSE_ENV, value)
    with pytest.raises(policy.CollectivePolicyError, match=FUSE_ENV):
        policy.allreduce_fuse_enabled()


def test_the_fuse_argument_is_a_tensorizer_option():
    # The runner passes compiler_args to neuronx-cc unchanged; the pass lives in the
    # Tensorizer, whose options neuronx-cc takes as one --tensorizer-options value.
    flag, value = policy.FUSE_COMPILER_ARG.split("=", 1)
    assert flag == "--tensorizer-options"
    assert value == "--disable-tiling-allreduce"


# ── the runner appends the fuse flag ──────────────────────────────────────────


class _CompileReached(Exception):
    """Raised where ``load_model`` asks for its compile backend: the options are built."""


class _StandInModel(torch.nn.Module):
    """A model class with the one constructor ``load_model`` calls."""

    @classmethod
    def from_configs(cls, hf_config, neuron_config):
        return cls()


def _runner_compiler_args(monkeypatch, fuse: str | None) -> list[str]:
    """The ``compiler_args`` that ``NeuronModelRunner.load_model`` builds.

    The runner's own ``load_model`` runs on a stand-in model through its CPU-compile
    path (no weights), and stops where it asks for the compile backend, after the
    compile options are built.
    """
    from vllm_neuron.vllm.worker import neuron_model_runner as runner_module

    if fuse is None:
        monkeypatch.delenv(FUSE_ENV, raising=False)
    else:
        monkeypatch.setenv(FUSE_ENV, fuse)
    monkeypatch.setenv("VLLM_NEURON_CPU_COMPILE", "1")
    monkeypatch.delenv("VLLM_NEURON_DEBUG_MODE", raising=False)
    # load_model lifts dynamo's cache limit for the process; restore it afterwards.
    dynamo_config = torch._dynamo.config
    monkeypatch.setattr(
        dynamo_config, "cache_size_limit", dynamo_config.cache_size_limit
    )
    monkeypatch.setattr(
        runner_module.ModelRegistry,
        "resolve_model_cls",
        lambda *args, **kwargs: (_StandInModel, "StandIn"),
    )

    def compile_backend_name():
        raise _CompileReached

    monkeypatch.setattr(envs, "get_compile_backend_name", compile_backend_name)
    model_config = SimpleNamespace(
        architecture="StandIn",
        enforce_eager=False,
        hf_config=None,
        runner_type="generate",
        model="stand-in",
        quantization=None,
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=model_config,
            cache_config=SimpleNamespace(cache_dtype="auto"),
            load_config=SimpleNamespace(download_dir=None),
            optimization_level=SimpleNamespace(value=1),
        ),
        neuron_config=SimpleNamespace(),
        vision_neuron_config=None,
        is_eagle3_spec=False,
        device=torch.device("cpu"),
    )
    with pytest.raises(_CompileReached):
        runner_module.NeuronModelRunner.load_model(runner)
    return runner.compile_options["compiler_args"]


def test_the_runner_appends_the_fuse_flag_only_when_the_switch_is_on(monkeypatch):
    """Unset or ``0`` leaves the compiler arguments as they were; ``1`` adds the flag."""
    unset = _runner_compiler_args(monkeypatch, None)
    assert not any(arg.startswith("--tensorizer-options") for arg in unset)
    assert _runner_compiler_args(monkeypatch, "0") == unset
    assert _runner_compiler_args(monkeypatch, "1") == [
        *unset,
        policy.FUSE_COMPILER_ARG,
    ]


# ── the entry point ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["fp32", "bf16"])
def test_one_rank_is_untouched_in_either_setting(monkeypatch, value):
    """No coordinator: the same object comes back, no cast, no collective."""
    monkeypatch.setenv(DTYPE_ENV, value)
    partial = _partial()
    before = partial.clone()
    out = policy.reduce_row_parallel(partial, site=SITE, group=None)
    assert out is partial
    assert torch.equal(out, before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp32_hands_the_coordinator_the_partial_itself(monkeypatch, dtype):
    """The as-built path: the partial is reduced in place, in its own dtype."""
    monkeypatch.setenv(DTYPE_ENV, "fp32")
    group = _RecordingGroup()
    partial = _partial(dtype)
    expected = partial * 2
    out = policy.reduce_row_parallel(partial, site=SITE, group=group)
    assert len(group.handed) == 1
    assert group.handed[0] is partial
    assert out is partial
    assert torch.equal(out, expected)


def test_bf16_reduces_a_bf16_copy_and_leaves_the_partial_alone(monkeypatch):
    """The partial is cast once, the copy crosses the wire, the copy comes back."""
    monkeypatch.setenv(DTYPE_ENV, "bf16")
    group = _RecordingGroup()
    partial = _partial()
    before = partial.clone()
    out = policy.reduce_row_parallel(partial, site=SITE, group=group)
    assert len(group.handed) == 1
    handed = group.handed[0]
    assert handed.dtype is torch.bfloat16
    assert handed.shape == partial.shape
    assert out is handed
    # The partial is not written: a tap that holds it still holds this rank's share.
    assert torch.equal(partial, before)
    assert torch.equal(out, before.to(torch.bfloat16) * 2)


def test_bf16_reduces_a_bf16_partial_in_place(monkeypatch):
    """A partial already in bfloat16 (the CPU routed-expert sum) is not copied."""
    monkeypatch.setenv(DTYPE_ENV, "bf16")
    group = _RecordingGroup()
    partial = _partial(torch.bfloat16)
    out = policy.reduce_row_parallel(partial, site=SITE, group=group)
    assert group.handed == [partial]
    assert out is partial


def test_the_switch_is_read_per_call(monkeypatch):
    """A trace takes whichever value is set when it runs, not when the module loaded."""
    for value, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        monkeypatch.setenv(DTYPE_ENV, value)
        out = policy.reduce_row_parallel(_partial(), site=SITE, group=_RecordingGroup())
        assert out.dtype is dtype


def test_every_site_is_accepted(monkeypatch):
    monkeypatch.setenv(DTYPE_ENV, "fp32")
    for site in policy.RowParallelSite:
        group = _RecordingGroup()
        policy.reduce_row_parallel(_partial(), site=site, group=group)
        assert len(group.handed) == 1


@pytest.mark.parametrize("site", ["lm_head", SITE.value])
def test_a_site_that_is_not_a_member_is_refused(site):
    """Only a :class:`RowParallelSite` member names a site, not its string value."""
    with pytest.raises(policy.CollectivePolicyError, match="RowParallelSite"):
        policy.reduce_row_parallel(_partial(), site=site, group=_RecordingGroup())


@pytest.mark.parametrize("value", ["fp32", "bf16"])
@pytest.mark.parametrize("group", [None, _RecordingGroup()], ids=["one-rank", "group"])
def test_the_entry_point_traces_as_one_graph(monkeypatch, value, group):
    """The sites run inside the captured prefill and decode graphs: no graph break.

    The site is named as the layer code names it, by attribute inside the traced code.
    """
    monkeypatch.setenv(DTYPE_ENV, value)
    torch._dynamo.reset()

    def site(partial):
        return policy.reduce_row_parallel(
            partial.clone(), site=policy.RowParallelSite.FFN, group=group
        )

    partial = _partial()
    out = torch.compile(site, fullgraph=True, backend="eager")(partial)
    expected = policy.reduce_row_parallel(
        partial.clone(), site=SITE, group=None if group is None else _RecordingGroup()
    )
    assert out.dtype is expected.dtype
    assert torch.equal(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_a_partial_outside_the_contract_is_refused_by_site(dtype):
    with pytest.raises(policy.CollectivePolicyError, match=SITE.value):
        policy.reduce_row_parallel(_partial(dtype), site=SITE, group=None)


# ── what the entry point adds to a traced site ────────────────────────────────

#: The checkpoint's own HF config, the file ``test_load_weights`` reads as the real one.
CHECKPOINT_CONFIG = Path(__file__).parent / "fixtures" / "hf-config.json"
#: The carrier every site casts the reduced sum to on the device: the model dtype the
#: checkpoint names. Each site casts to its input's dtype (``hidden_states`` at the KDA
#: and FFN sites, the ``mla_absorb`` output at the MLA site), and both are this dtype.
CARRIER = Glm5NextConfig.from_configs(
    json.loads(CHECKPOINT_CONFIG.read_text())
).text_config.torch_dtype
CAST = "aten._to_copy.default"


# Defined at import. A second import defines it again with the same body, which torch
# accepts (``test_this_module_imports_twice_in_one_process``), so it needs no guard.
@torch.library.custom_op(
    "collective_policy_test::all_reduce_", mutates_args=("tensor",)
)
def _traced_all_reduce_(tensor: torch.Tensor) -> None:
    """A collective that stays one opaque in-place op in the traced graph."""
    tensor.mul_(2)


class _TracedGroup:
    """A coordinator whose ``all_reduce`` traces as :func:`_traced_all_reduce_`."""

    def all_reduce(self, tensor: torch.Tensor) -> None:
        torch.ops.collective_policy_test.all_reduce_(tensor)


def _as_built_site(partial: torch.Tensor) -> torch.Tensor:
    """A site before the entry point: reduce in place, then cast to the carrier."""
    _TracedGroup().all_reduce(partial)
    return partial.to(CARRIER)


def _policy_site(partial: torch.Tensor) -> torch.Tensor:
    """A site through the entry point, as the layer code writes it."""
    reduced = policy.reduce_row_parallel(
        partial, site=policy.RowParallelSite.FFN, group=_TracedGroup()
    )
    return reduced.to(CARRIER)


def _aten_ops(site, partial: torch.Tensor) -> list[tuple[str, torch.dtype | None]]:
    """The aten ops a ``fullgraph=True`` capture of ``site`` runs, in order.

    Each op is its target and, for a cast, the dtype it casts to. Tuple indexing
    (``getitem``) moves no data and is left out.
    """
    graphs = []

    def record(graph_module, example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    torch.compile(site, fullgraph=True, backend=aot_autograd(fw_compiler=record))(
        partial
    )
    (graph,) = graphs
    return [
        (str(node.target), node.kwargs.get("dtype"))
        for node in graph.graph.nodes
        if node.op == "call_function" and node.target is not operator.getitem
    ]


def test_this_module_imports_twice_in_one_process():
    """A second import (under another name, or in a second ``pytest.main``) defines the
    stand-in op again with the same body; torch replaces it without error, and the op
    still reduces."""
    spec = importlib.util.spec_from_file_location("collective_policy_again", __file__)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    tensor = torch.ones(2)
    torch.ops.collective_policy_test.all_reduce_(tensor)
    assert torch.equal(tensor, torch.full((2,), 2.0))


def test_the_fp32_wire_adds_no_op_to_a_traced_site(monkeypatch):
    """The default traces to the as-built site's ops, one for one."""
    monkeypatch.setenv(DTYPE_ENV, "fp32")
    as_built = _aten_ops(_as_built_site, _partial())
    assert _aten_ops(_policy_site, _partial()) == as_built
    assert as_built.count((CAST, CARRIER)) == 1


def test_the_bf16_wire_moves_the_carrier_cast_ahead_of_the_collective(monkeypatch):
    """The cast to bf16 runs before the collective instead of after it; the site's own
    cast to its bf16 carrier then traces to nothing, so the op count is unchanged."""
    monkeypatch.setenv(DTYPE_ENV, "bf16")
    # The move holds because the wire dtype is the model's own: the site's cast to its
    # carrier then has nothing left to do.
    assert CARRIER is policy.allreduce_wire_dtype()
    as_built = _aten_ops(_as_built_site, _partial())
    ops = _aten_ops(_policy_site, _partial())
    assert as_built[-1] == ops[0] == (CAST, CARRIER)
    assert ops[1:] == as_built[:-1]


# ── the layer code calls the entry point and nothing else ─────────────────────

#: Each row-parallel method and the one site it reduces at.
_SITES = {
    model_fp8.Glm5NextKDAAttention._gated_output: policy.RowParallelSite.KDA_O_PROJ,
    model_fp8.Glm5NextMLAAttention.project_output: policy.RowParallelSite.MLA_O_PROJ,
    model_fp8.Glm5NextModel._ffn_half: policy.RowParallelSite.FFN,
}


def test_each_site_makes_one_call_naming_its_site():
    for method, site in _SITES.items():
        source = inspect.getsource(method)
        assert source.count("reduce_row_parallel(") == 1, method.__qualname__
        assert f"site=RowParallelSite.{site.name}," in source, method.__qualname__
    assert set(_SITES.values()) == set(policy.RowParallelSite)


def test_the_model_makes_no_direct_all_reduce():
    """The acceptance grep: every collective of the layer code goes through the helper."""
    source = inspect.getsource(model_fp8)
    assert "all_reduce(" not in source
    assert source.count("reduce_row_parallel(") == len(_SITES)
