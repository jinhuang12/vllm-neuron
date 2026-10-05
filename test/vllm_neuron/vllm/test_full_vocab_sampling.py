# SPDX-License-Identifier: Apache-2.0
"""The full-vocabulary on-device sampler the GLM-5.3-Flash root hands its logits to.

Contract with the lm_head work: every rank holds the FULL ``[B, 154880]`` logits, so the
sampler reads no process group. Contract with the batch work: any B up to 64, one row of
``[top_k, top_p, temperature]`` per request.
"""

from __future__ import annotations

import operator

import pytest
import torch

from vllm_neuron.functional.full_vocab_sampling import (
    model_samples_on_device,
    sample_full_vocab,
)
from vllm_neuron.model.neuron_config import NeuronConfig, OnDeviceSamplingConfig

pytestmark = [pytest.mark.fast]

VOCAB = 154880


def _logits(batch: int, dtype: torch.dtype, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + batch)
    return (torch.randn(batch, VOCAB, generator=generator) * 4.0).to(dtype)


def _params(rows) -> torch.Tensor:
    """One ``[top_k, top_p, temperature]`` row per request, the runner's layout."""
    return torch.tensor(rows, dtype=torch.float32)


def _greedy(batch: int) -> torch.Tensor:
    return _params([[-1, 1.0, 0.0]] * batch)


@pytest.mark.parametrize("batch", [1, 4, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_greedy_rows_are_torch_argmax(batch, dtype):
    logits = _logits(batch, dtype)
    tokens = sample_full_vocab(logits, _greedy(batch), OnDeviceSamplingConfig())
    assert tokens.dtype == torch.int32 and tuple(tokens.shape) == (batch,)
    assert torch.equal(tokens, torch.argmax(logits, dim=-1).to(torch.int32))


def test_greedy_matches_argmax_on_a_bf16_tie():
    """A tie at the maximum resolves to the first index, as torch.argmax and vLLM do."""
    logits = torch.zeros(2, VOCAB, dtype=torch.bfloat16)
    logits[:, 900] = 3.0
    logits[:, 77] = 3.0
    tokens = sample_full_vocab(logits, _greedy(2), OnDeviceSamplingConfig())
    assert tokens.tolist() == [77, 77]


@pytest.mark.parametrize("batch", [1, 4])
def test_all_greedy_config_ignores_sampling_params(batch):
    logits = _logits(batch, torch.bfloat16)
    params = _params([[50, 0.9, 1.0]] * batch)
    tokens = sample_full_vocab(logits, params, OnDeviceSamplingConfig(all_greedy=True))
    assert torch.equal(tokens, torch.argmax(logits, dim=-1).to(torch.int32))


def test_per_request_parameters_in_one_batch():
    """Greedy, top-k, top-p and plain random rows share one call."""
    logits = _logits(4, torch.float32)
    params = _params([
        [-1, 1.0, 0.0],      # greedy
        [5, 1.0, 1.0],       # top-5
        [-1, 1e-6, 1.0],     # nucleus so tight that only the top token survives
        [-1, 1.0, 1.0],      # unrestricted (inside max_top_k)
    ])
    top5 = torch.topk(logits[1], 5).indices.tolist()
    top256 = torch.topk(logits[3], 256).indices.tolist()
    argmax = torch.argmax(logits, dim=-1)
    for seed in range(8):
        torch.manual_seed(seed)
        tokens = sample_full_vocab(logits, params, OnDeviceSamplingConfig())
        assert int(tokens[0]) == int(argmax[0])
        assert int(tokens[1]) in top5
        assert int(tokens[2]) == int(argmax[2])
        assert int(tokens[3]) in top256


def test_random_rows_spread_over_the_top_k():
    """top_k=3 at a flat distribution reaches more than one token: the row is not greedy."""
    logits = torch.zeros(1, VOCAB)
    logits[0, [10, 20, 30]] = 5.0
    seen = set()
    for seed in range(32):
        torch.manual_seed(seed)
        seen.add(int(sample_full_vocab(logits, _params([[3, 1.0, 1.0]]), OnDeviceSamplingConfig())[0]))
    assert seen <= {10, 20, 30} and len(seen) > 1, seen


def test_logit_mask_removes_the_argmax():
    logits = _logits(2, torch.bfloat16)
    best = torch.argmax(logits, dim=-1)
    mask = torch.ones(2, VOCAB, dtype=torch.bool)
    mask[0, best[0]] = False
    tokens = sample_full_vocab(logits, _greedy(2), OnDeviceSamplingConfig(), logit_mask=mask)
    expected = logits.clone()
    expected[0, best[0]] = float("-inf")
    assert int(tokens[0]) != int(best[0])
    assert torch.equal(tokens, torch.argmax(expected, dim=-1).to(torch.int32))


def test_a_mask_narrower_than_the_vocabulary_is_refused():
    logits = _logits(1, torch.bfloat16)
    shard_mask = torch.ones(1, VOCAB // 64, dtype=torch.bool)
    with pytest.raises(ValueError, match="full-vocabulary"):
        sample_full_vocab(logits, _greedy(1), OnDeviceSamplingConfig(), logit_mask=shard_mask)


def test_parameters_for_another_batch_are_refused():
    with pytest.raises(ValueError, match="one row per request"):
        sample_full_vocab(_logits(4, torch.float32), _greedy(1), OnDeviceSamplingConfig())


def test_no_sampling_config_is_refused():
    with pytest.raises(ValueError, match="on_device_sampling_config"):
        sample_full_vocab(_logits(1, torch.float32), _greedy(1), None)


def _model(config):
    text = type("TextConfig", (), {"neuron_config": config})()
    return type("Root", (), {"text_config": text})()


def test_model_samples_on_device_reads_the_models_own_config():
    assert model_samples_on_device(_model(NeuronConfig())) is True
    assert model_samples_on_device(_model(NeuronConfig(on_device_sampling_config=None))) is False
    assert model_samples_on_device(_model(None)) is False
    assert model_samples_on_device(object()) is False


def test_model_samples_on_device_looks_through_torch_compile():
    wrapper = type("Compiled", (), {"_orig_mod": _model(NeuronConfig())})()
    assert model_samples_on_device(wrapper) is True


_COMPARISONS = {operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne,
                torch.lt, torch.le, torch.gt, torch.ge, torch.eq, torch.ne}
_COMPARISON_METHODS = {"lt", "le", "gt", "ge", "eq", "ne",
                       "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"}


@pytest.mark.parametrize("all_greedy", [False, True])
def test_no_comparison_with_a_python_float_reaches_the_graph(all_greedy):
    """A tensor compared with a Python float lowers to an f64 convert and compare, which
    neuronx-cc refuses (``NCC_ESPP004 f64 dtype is not supported``); trn2 refused
    ``temperature < _SAMPLING_EPS`` this way. Every comparison in the traced sampler
    must be tensor against tensor (or against an integer)."""
    graphs = []

    def capture(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    config = OnDeviceSamplingConfig(all_greedy=all_greedy)
    compiled = torch.compile(
        lambda logits, params: sample_full_vocab(logits, params, config),
        backend=capture, fullgraph=True, dynamic=False)
    compiled(_logits(4, torch.bfloat16), _params([[VOCAB, 1.0, 0.0], [50, 0.9, 0.8]] * 2))
    assert graphs, "nothing was traced"
    offenders = [
        node.format_node()
        for graph in graphs for node in graph.graph.nodes
        if ((node.op == "call_function" and node.target in _COMPARISONS)
            or (node.op == "call_method" and node.target in _COMPARISON_METHODS))
        and any(isinstance(arg, float) for arg in node.args)
    ]
    assert offenders == [], offenders
