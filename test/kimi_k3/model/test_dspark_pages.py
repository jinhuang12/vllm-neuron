# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter: its forward traces under the device-free compile; its page view.

Both items run one world of two ranks over gloo. The trace item builds the proposer at TP 2 and
DCP 2 under the device-free compile; two ranks are the fewest the drafter's window attention
takes. The proposer's warm-up then traces the drafter's first graph as a trace lane does: on
meta tensors, inside the lane's compile-only scopes, so no graph runs and the runtime never
starts. The trace goes through dynamo and AOT autograd, and the item stops it where the native
backend starts the lowering. Each rank sets its own venue. The window kernel takes at most 128
head rows a sequence, so the item needs a draft config with few heads, such as a reduced
checkpoint's, and skips otherwise.

The page item loads each rank's TP 2 slices of the draft checkpoint and binds bf16 latent caches
in the runner's form: one dense ``[1, blocks, 1, block_size, 576]`` view of an int8 buffer per
layer. The forward's int8 page view of each cache must be the bound page buffer itself, and must
hold the buffer's bytes after the context write puts 24 random rows at random slots.

Set ``KIMI_K3_DSPARK_CHECKPOINT`` to the draft directory (default
``~/models/Inferact/Kimi-K3-DSpark-Block5``) and ``KIMI_K3_CHECKPOINT`` to the target directory
(default ``~/models/moonshotai/Kimi-K3``); the trace item reads their configs only. Run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import json
import os
import socket
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

MODELS = Path.home() / "models"
TARGET = Path(os.environ.get("KIMI_K3_CHECKPOINT", MODELS / "moonshotai" / "Kimi-K3"))
DRAFT = Path(
    os.environ.get("KIMI_K3_DSPARK_CHECKPOINT", MODELS / "Inferact" / "Kimi-K3-DSpark-Block5")
)

RANKS, TOKENS, K, SEQS, MAX_LEN, BLOCK = 2, 16, 4, 16, 4096, 256
BLOCKS, WIDTH, ROWS = 3, 576, 24
# The device-free compile; the package refuses it beside CPU mode.
VENUE = {"VLLM_NEURON_CPU_COMPILE": "1", "VLLM_NEURON_CPU_MODE": None, "NKI_SIMULATOR": None,
         "NEURON_PLATFORM_TARGET_OVERRIDE": os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE",
                                                           "trn3pre"),
         "VLLM_RPC_BASE_PATH": "."}


def _draft_config() -> dict:
    """The draft checkpoint's config."""
    with open(DRAFT / "config.json") as f:
        return json.load(f)


def _fits_the_window_kernel() -> bool:
    """Whether a rank's heads times the k block rows fit the window kernel's 128 rows."""
    return _draft_config()["num_attention_heads"] // RANKS * K <= 128


class _AtLowering(Exception):
    """Raised where the native backend starts the lowering: the trace before it completed."""


def _stop_at_lowering(*args, **kwargs):
    """Stand-in for the native backend's lowering entry."""
    raise _AtLowering


def _engine_config():
    """The engine config at TP 2 and DCP 2: DSpark with 4 drafts and 16 sequences."""
    from vllm.engine.arg_utils import EngineArgs

    spec = {"method": "dspark", "model": str(DRAFT), "num_speculative_tokens": K}
    config = EngineArgs(
        model=str(TARGET), tensor_parallel_size=RANKS, decode_context_parallel_size=RANKS,
        max_model_len=MAX_LEN, max_num_seqs=SEQS, skip_tokenizer_init=True,
        enable_prefix_caching=False, language_model_only=True, speculative_config=spec,
        async_scheduling=False,
    ).create_engine_config()
    config.additional_config.pop("encoder_neuron_config", None)
    return config


def _target_head(draft):
    """The target's embedding and LM-head halves on meta, as its own lite load leaves them."""
    from vllm_neuron.model.kimi_k3.model import _tp_group
    from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

    c = draft.config
    with torch.device("meta"):
        embed = VocabDimShardedEmbedding(c.vocab_size, c.hidden_size, dtype=c.torch_dtype,
                                         tp_group=_tp_group().device_group)
        halves = [torch.empty(c.hidden_size, c.vocab_size // RANKS // 2, dtype=c.torch_dtype)
                  for _ in range(2)]
    return SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), head_weights=halves)


def _trace(config) -> str:
    """The proposer's load and warm-up under the device-free compile, as a trace lane runs it."""
    import torch._functorch.config as functorch_config
    from libtorch_neuronx_lite._compiler import compiler
    from libtorch_neuronx_lite.compile.lite_compile_only import lite_compile_only
    from libtorch_neuronx_lite.compile.native_backend import native_precompile
    from vllm.config import set_current_vllm_config

    from vllm_neuron.vllm.spec_decode.dspark import DSparkProposer

    meta, columns = torch.device("meta"), MAX_LEN // BLOCK
    with pytest.MonkeyPatch.context() as patch, set_current_vllm_config(config), torch.no_grad():
        # A cached graph would skip the trace this item reads.
        patch.setattr(functorch_config, "enable_autograd_cache", False)
        patch.setattr(compiler, "fx_to_neff", _stop_at_lowering)
        proposer = DSparkProposer(config, meta)
        proposer.load_model(target_hidden_size=config.model_config.hf_text_config.hidden_size)
        draft = proposer.model._orig_mod
        latent = draft.config.kv_lora_rank + draft.config.qk_rope_head_dim
        draft.bind_kv_cache({layer.cache_name: [torch.empty(columns + 1, 1, BLOCK, latent,
                                                            dtype=torch.bfloat16, device=meta)]
                             for layer in draft.layers})
        proposer.alias_target_weights(_target_head(draft))
        table = torch.empty(1, columns, dtype=torch.int32, device=meta)
        metadata = {layer.cache_name: {"block_table_tensor": table, "block_size": BLOCK,
                                       "max_blocks_per_seq": columns}
                    for layer in draft.layers}
        try:
            with lite_compile_only(), native_precompile():
                proposer.warmup(TOKENS, 1, metadata)
        except Exception as e:
            if isinstance(getattr(e, "inner_exception", None), _AtLowering):
                return "at lowering"
            raise
    return "not stopped"


def _trace_rank(rank: int, port: int, root: str) -> None:
    """One trace rank: the trace, or the error that stopped it."""
    import torch.distributed as dist
    from vllm import ModelRegistry

    from vllm_neuron.model import registry
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state,
        stamp_spmd_rank_tensors,
    )

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=RANKS)
    initialize_neuron_parallel_state(dcp_size=RANKS, tp_global_ranks=list(range(RANKS)),
                                     local_rank=rank)
    try:
        # The config comes first: vLLM checks the registered class when it builds one.
        config = _engine_config()
        for arch, model_cls in registry.get_models():
            ModelRegistry.register_model(arch, model_cls)
        stamped = stamp_spmd_rank_tensors("meta")  # the rank tensors of meta inputs
        out = {"ok": _trace(config)}
        del stamped
    except Exception as e:  # the reading names the stop
        out = {"error": "".join(traceback.format_exception_only(e)).strip(),
               "where": traceback.format_exc(limit=-4)}
    torch.save(out, Path(root) / f"rank{rank}.pt")
    dist.destroy_process_group()


def _page_rank(rank: int, port: int, inputs: dict, root: str) -> None:
    """One page rank: each layer's page view and page buffer, before and after the write."""
    import torch.distributed as dist

    from vllm_neuron.model.kimi_k3.dspark import K3DSparkForCausalLM, _page_view
    from vllm_neuron.parallel.neuron_parallel_state import initialize_neuron_parallel_state

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=RANKS)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(RANKS)), local_rank=rank)
    config = _draft_config()
    with torch.device("meta"):
        draft = K3DSparkForCausalLM.from_configs(
            config=config, start_layer_idx=config["target_num_hidden_layers"],
            max_positions=MAX_LEN,
        )
    draft.load_weights(str(DRAFT), torch.device("cpu"))
    buffers = {layer.cache_name: torch.zeros(BLOCKS * BLOCK * WIDTH * 2, dtype=torch.int8)
               for layer in draft.layers}
    draft.bind_kv_cache({name: [raw.view(torch.bfloat16).view(1, BLOCKS, 1, BLOCK, WIDTH)[0]]
                         for name, raw in buffers.items()})
    attns = [layer.self_attn for layer in draft.layers]
    views = [_page_view(attn.kv_cache) for attn in attns]
    before = [attn.pages.clone() for attn in attns]
    draft.precompute_and_store_context_kv(
        inputs["taps"].to(draft.config.torch_dtype), inputs["positions"], inputs["slots"]
    )

    def form(t):
        return tuple(t.shape), t.stride(), t.data_ptr()

    state = {"layout": [(form(view), form(attn.pages)) for view, attn in zip(views, attns)],
             "views": [v.clone() for v in views], "pages": [a.pages.clone() for a in attns],
             "before": before}
    torch.save(state, Path(root) / f"rank{rank}.pt")
    dist.destroy_process_group()


def _world(root: Path, rank_fn, args: tuple, venue: dict) -> list[dict]:
    """Each rank's saved reading from one two-rank world in the given venue."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cwd = os.getcwd()
    with pytest.MonkeyPatch.context() as patch:
        for name, value in venue.items():
            if value is None:
                patch.delenv(name, raising=False)
            else:
                patch.setenv(name, value)
        os.chdir(root)
        try:
            mp.spawn(rank_fn, args=(port, *args, str(root)), nprocs=RANKS, join=True)
        finally:
            os.chdir(cwd)
    return [torch.load(root / f"rank{r}.pt", weights_only=False) for r in range(RANKS)]


@pytest.mark.skipif(
    not ((TARGET / "config.json").is_file() and (DRAFT / "config.json").is_file()),
    reason=f"Kimi K3 config at {TARGET} or DSpark config at {DRAFT} not found",
)
def test_draft_forward_traces_under_the_device_free_compile(tmp_path):
    if not _fits_the_window_kernel():
        pytest.skip(f"the DSpark config at {DRAFT} has too many heads for the window kernel")
    for rank, reading in enumerate(_world(tmp_path, _trace_rank, (), VENUE)):
        assert reading == {"ok": "at lowering"}, (rank, reading)


@pytest.mark.skipif(
    not (DRAFT / "model.safetensors").is_file(), reason=f"DSpark checkpoint not found at {DRAFT}"
)
def test_page_view_is_the_bound_page_buffer_and_holds_its_bytes(tmp_path):
    g = torch.Generator().manual_seed(0)
    config = _draft_config()
    inputs = {
        "taps": torch.randn(ROWS, config["num_target_layers"] * config["target_hidden_size"],
                            generator=g),
        "positions": torch.randint(0, MAX_LEN, (ROWS,), generator=g),
        "slots": BLOCK + torch.randperm((BLOCKS - 1) * BLOCK, generator=g)[:ROWS],
    }
    world = _world(tmp_path, _page_rank, (inputs,), {"VLLM_RPC_BASE_PATH": "."})
    for rank, state in enumerate(world):
        for layer, (view, pages) in enumerate(state["layout"]):
            assert view == pages, (rank, layer, view, pages)
        for layer, (view, pages, before) in enumerate(
            zip(state["views"], state["pages"], state["before"])
        ):
            assert not torch.equal(pages, before), (rank, layer, "the write changed no byte")
            assert torch.equal(view, pages), (rank, layer)
