# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 serves text only: the served form builds through the worker and refuses images.

The served checkpoint's config carries a vision part. The DCP 4 form's serve flags, from
``test/kimi_k3/serving/cells.json``, build the engine config through ``vllm serve``'s parser and
the platform's config step. On 4 gloo CPU ranks, one replica at TP 4 with DCP 4, the real
NeuronWorker then creates the model runner and loads the model on the meta device. That is the
smallest world in which the form's 3 expert replica slots build; the served world of 64 ranks
does not fit one host's CPU test. The load stops at the weight read, so no weight is read. Set
``KIMI_K3_CHECKPOINT`` to the checkpoint directory (its config and tokenizer only); run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import json
import os
from pathlib import Path

import pytest
import torch.multiprocessing as mp
from vllm import SamplingParams
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.inputs.engine import mm_input, tokens_input
from vllm.multimodal.inputs import PlaceholderRange
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_neuron.model.kimi_k3 import model as kimi_k3_model
from vllm_neuron.parallel.neuron_parallel_state import init_neuron_distributed_environment
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)
CELLS = Path(__file__).resolve().parents[1] / "serving" / "cells.json"
WORLD = 4


class _WeightRead(Exception):
    """Raised where the model load would read the checkpoint's weights."""


@pytest.fixture(scope="module")
def served_config(tmp_path_factory):
    """The engine config ``vllm serve`` builds from the DCP 4 form's flags."""
    measurements = tmp_path_factory.mktemp("measurements")
    flags = [f.replace("{checkpoint}", str(CHECKPOINT)).replace("{measurements}", str(measurements))
             for f in json.loads(CELLS.read_text(encoding="utf-8"))["forms"]["dcp4"]["server_flags"]]
    args = make_arg_parser(FlexibleArgumentParser()).parse_args([str(CHECKPOINT), *flags])
    args.model = args.model_tag
    validate_parsed_serve_args(args)
    return AsyncEngineArgs.from_cli_args(args).create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER)


def _rank(rank: int, config, root: Path) -> None:
    """One rank: the worker creates the runner and loads the model up to the weight read."""

    def groups(self, *args, **kwargs):
        self._patch_in_same_node_as_function()
        with set_current_vllm_config(VllmConfig()):
            init_neuron_distributed_environment(
                world_size=WORLD, rank=rank, local_rank=rank, backend="gloo",
                tensor_parallel_size=WORLD, decode_context_parallel_size=4, ep_degree=WORLD,
                distributed_init_method=f"file://{root}/rendezvous")

    def weight_read(self, *args, **kwargs):
        raise _WeightRead

    NeuronWorker._init_neuron_distributed_environment_and_runtime = groups
    kimi_k3_model.KimiK3ForConditionalGeneration.load_weights = weight_read
    worker = NeuronWorker(vllm_config=config, local_rank=rank, rank=rank,
                          distributed_init_method="unused")
    with set_current_vllm_config(config):
        worker.init_device()
    with pytest.raises(_WeightRead):
        worker.load_model()
    runner = worker.model_runner
    cache = runner.encoder_cache
    (root / f"rank{rank}.json").write_text(json.dumps({
        "model": f"{type(runner.model).__module__}.{type(runner.model).__qualname__}",
        "devices": sorted({p.device.type for p in runner.model.parameters()}),
        "encoder_cache_bytes": None if cache is None else cache.buffer.numel() * cache.buffer.element_size(),
    }))


def test_served_form_builds_with_no_encoder_cache(served_config, tmp_path):
    """Every rank builds the model through the runner's own load call, and no rank sets up an
    encoder cache, though the checkpoint config carries a vision part."""
    assert hasattr(served_config.model_config.hf_config, "vision_config")
    mp.spawn(_rank, args=(served_config, tmp_path), nprocs=WORLD)
    model = kimi_k3_model.KimiK3ForConditionalGeneration
    for rank in range(WORLD):
        reading = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert reading["model"] == f"{model.__module__}.{model.__qualname__}", rank
        assert reading["devices"] == ["meta"], rank
        assert not reading["encoder_cache_bytes"], (rank, reading["encoder_cache_bytes"])


def test_image_request_is_refused_by_name(served_config):
    """A request with an image item is refused before scheduling, with an error that names
    vision as out of the served form's scope; a text request passes."""
    params = SamplingParams(max_tokens=1)
    current_platform.validate_request(tokens_input([1, 2, 3]), params)
    image = mm_input([1, 2, 3, 4], mm_kwargs={}, mm_hashes={"image": ["0"]},
                     mm_placeholders={"image": [PlaceholderRange(offset=1, length=2)]})
    with pytest.raises(ValueError, match="Vision input is out of the scope of this served form"):
        current_platform.validate_request(image, params)
