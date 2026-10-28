# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter: its weight layout for the device-free compile, on two TP ranks.

The device-free compile loads no weight, so the drafter lays out its kernels' weights from the
meta parameters with ``load_weights_lite``. One TP 2 world over gloo runs three checks on the
draft checkpoint's config. After a lite load from a folder that does not exist, every layout
has a full load's shape and dtype. A lite load and then a full load equal a full load alone,
byte for byte, and leave no meta tensor. The proposer's load under the device-free flag, then
one draft step on meta inputs at 16 sequences and 4 drafts, completes.

At TP 2 each split width is 16 times its size in the served form, TP 32; the other widths
are the served ones. A served rank holds context_kv_weight [1728, 224] and latent_weight
[224, 2112] fp32, output_weight [384, 7168], w_qb_t [1536, 576] and w_uk [3, 128, 512] bf16,
w_uv_t [3, 512, 128] fp32, gate_up_weight [7168, 896] and down_weight [448, 7168] bf16, and
rope_table [max length + 4, 64] fp32. Set ``KIMI_K3_DSPARK_CHECKPOINT`` to the draft directory
(default ``~/models/Inferact/Kimi-K3-DSpark-Block5``) and ``KIMI_K3_CHECKPOINT`` to the target
directory (its config only; default ``~/models/moonshotai/Kimi-K3``). Run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import gc
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

pytestmark = pytest.mark.skipif(
    not ((TARGET / "config.json").is_file() and (DRAFT / "model.safetensors").is_file()),
    reason=f"Kimi K3 config at {TARGET} or DSpark checkpoint at {DRAFT} not found",
)

TP, K, SEQS, MAX_LEN, TARGET_LAYERS = 2, 4, 16, 32768, 93
# The served attention block size; the hybrid alignment at TP 2 gives a size not served.
BLOCK = 256
BF16, F32 = torch.bfloat16, torch.float32


def _expected(c) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Each layout's shape and dtype on one rank of the TP 2 world."""
    hidden, heads = c.hidden_size // TP, c.num_attention_heads // TP
    width = c.intermediate_size // TP
    q_rank, kv_rank, nope = c.q_lora_rank, c.kv_lora_rank, c.qk_nope_head_dim
    rope, v_dim, layers = c.qk_rope_head_dim, c.v_head_dim, c.num_hidden_layers
    out = {"context_kv_weight": ((layers * (kv_rank + rope), hidden), F32),
           "rope_table": ((MAX_LEN + K, rope), F32)}
    for i in range(layers):
        out.update({
            f"layers.{i}.self_attn.latent_weight": ((hidden, q_rank + kv_rank + rope), F32),
            f"layers.{i}.self_attn.output_weight": ((heads * v_dim, c.hidden_size), BF16),
            f"layers.{i}.self_attn.w_qb_t": ((q_rank, heads * (nope + rope)), BF16),
            f"layers.{i}.self_attn.w_uk": ((heads, nope, kv_rank), BF16),
            f"layers.{i}.self_attn.w_uv_t": ((heads, kv_rank, v_dim), F32),
            f"layers.{i}.mlp.gate_up_weight": ((c.hidden_size, 2 * width), BF16),
            f"layers.{i}.mlp.down_weight": ((width, c.hidden_size), BF16),
        })
    return out


def _layouts(draft) -> dict:
    """Each layout's shape and dtype by tree name, or None where it is not laid out."""
    tensors = dict(draft.named_buffers(remove_duplicate=False))
    return {name: None if tensors.get(name) is None else (tuple(tensors[name].shape),
                                                          tensors[name].dtype)
            for name in _expected(draft.config)}


def _tensors(draft) -> dict[str, torch.Tensor]:
    """Every parameter and buffer of the tree, by name."""
    return {**dict(draft.named_parameters(remove_duplicate=False)),
            **dict(draft.named_buffers(remove_duplicate=False))}


def _held_meta(draft) -> list[str]:
    """Each meta tensor any module of the drafter holds, inside its tree or outside it."""
    found = []
    todo = [(f"{name or 'model'}.{key}", value)
            for name, module in draft.named_modules() for key, value in vars(module).items()]
    while todo:
        path, value = todo.pop()
        if isinstance(value, torch.Tensor):
            found += [path] if value.is_meta else []
        elif isinstance(value, dict):
            todo += [(f"{path}.{k}", v) for k, v in value.items()]
        elif isinstance(value, (list, tuple)):
            todo += [(f"{path}.{i}", v) for i, v in enumerate(value)]
    return sorted(found)


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True when both hold data of one dtype and shape, with the same bytes."""
    if a.is_meta or b.is_meta or a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _draft():
    """The drafter on meta, built as the proposer builds it for the served length."""
    from vllm.transformers_utils.config import get_config

    from vllm_neuron.model.kimi_k3.dspark import K3DSparkForCausalLM

    with torch.device("meta"):
        return K3DSparkForCausalLM.from_configs(get_config(str(DRAFT), trust_remote_code=False),
                                                TARGET_LAYERS, max_positions=MAX_LEN + K)


def _lite_layouts(full, root: Path) -> dict:
    """The layouts after a lite load from a missing folder, beside the full load's."""
    lite = _draft()
    lite.load_weights_lite(str(root / "no-checkpoint"), torch.device("cpu"), None)
    return {"lite": _layouts(lite), "full": _layouts(full), "expected": _expected(lite.config)}


def _lite_then_full(full) -> dict:
    """A lite load then a full load, against the full load alone."""
    relaid = _draft()
    relaid.load_weights_lite(str(DRAFT), torch.device("cpu"), None)
    relaid.load_weights(str(DRAFT), torch.device("cpu"))
    want, got = _tensors(full), _tensors(relaid)
    differ = sorted(n for n in want.keys() & got.keys() if not _same_bytes(want[n], got[n]))
    return {"names": (sorted(want.keys() - got.keys()), sorted(got.keys() - want.keys())),
            "count": len(got), "differ": differ, "meta": _held_meta(relaid)}


def _block_pass(self, embedded, positions, attn_metadata):
    """Stand-in block pass: the block rows as ``[B, k, H]``; it reads no layer weight."""
    batch = attn_metadata[self.layers[0].cache_name]["block_table_tensor"].shape[0]
    return embedded.view(batch, -1, embedded.shape[-1])


def _qkv(hidden, weight, norm_weight, eps):
    """Stand-in fused norm and projection in fp32, returned in bf16."""
    x = hidden.float()
    rows = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm_weight.float()
    return (rows @ weight.float()).bfloat16()


def _engine_config():
    """The engine config at TP 2: DSpark with 4 drafts at the served length and 16 sequences."""
    from vllm.engine.arg_utils import EngineArgs

    spec = {"method": "dspark", "model": str(DRAFT), "num_speculative_tokens": K}
    config = EngineArgs(
        model=str(TARGET), tensor_parallel_size=TP, max_model_len=MAX_LEN, max_num_seqs=SEQS,
        enforce_eager=True, skip_tokenizer_init=True, enable_prefix_caching=False,
        language_model_only=True, speculative_config=spec, async_scheduling=False,
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
        halves = [torch.empty(c.hidden_size, c.vocab_size // TP // 2, dtype=c.torch_dtype)
                  for _ in range(2)]
    return SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), head_weights=halves)


def _device_free_step(config) -> dict:
    """The proposer's load under the device-free flag, then one draft step on meta inputs."""
    from vllm.config import set_current_vllm_config

    from vllm_neuron.model.kimi_k3 import dspark
    from vllm_neuron.vllm.spec_decode.dspark import DSparkProposer

    meta = torch.device("meta")
    rows, columns = SEQS * (K + 1), -(-MAX_LEN // BLOCK)
    with pytest.MonkeyPatch.context() as patch, set_current_vllm_config(config), torch.no_grad():
        patch.setattr(dspark.K3DSparkForCausalLM, "block_pass", _block_pass)
        patch.setattr(dspark, "qkv_proj_fp32acc", _qkv)
        # The package refuses this flag beside CPU mode at import; the proposer reads it here.
        patch.setenv("VLLM_NEURON_CPU_COMPILE", "1")
        proposer = DSparkProposer(config, torch.device("cpu"))
        proposer.load_model(target_hidden_size=config.model_config.hf_text_config.hidden_size)
        draft = proposer.model._orig_mod
        latent = draft.config.kv_lora_rank + draft.config.qk_rope_head_dim
        draft.bind_kv_cache({layer.cache_name: [torch.empty(SEQS * columns + 1, 1, BLOCK, latent,
                                                            dtype=BF16, device=meta)]
                             for layer in draft.layers})
        proposer.alias_target_weights(_target_head(draft))
        table = torch.empty(SEQS, columns, dtype=torch.int32, device=meta)
        ids, drafts = proposer.propose(
            target_token_ids=torch.empty(rows, dtype=torch.int32, device=meta),
            target_positions=torch.empty(rows, dtype=torch.int64, device=meta),
            target_hidden_states=torch.empty(rows, 3 * draft.config.target_hidden_size,
                                             dtype=BF16, device=meta),
            last_token_indices=torch.empty(SEQS, dtype=torch.int64, device=meta),
            attn_metadata={layer.cache_name: {"block_table_tensor": table, "block_size": BLOCK,
                                              "max_blocks_per_seq": columns}
                           for layer in draft.layers},
            raw_sampled_token_ids=torch.empty(SEQS, K + 1, dtype=torch.int32, device=meta),
            is_warmup=True,
        )
    return {"ids": (tuple(ids.shape), ids.dtype), "drafts": (tuple(drafts.shape), drafts.dtype)}


def _attempt(fn, *args) -> dict:
    """``fn``'s readings, or the error that stopped it."""
    try:
        return {"ok": fn(*args)}
    except Exception as e:  # each check is read on its own
        return {"error": "".join(traceback.format_exception_only(e)).strip(),
                "where": traceback.format_exc(limit=-3)}


def _rank(rank: int, port: int, root: str) -> None:
    """One TP rank: the three checks, each apart from the others."""
    import torch.distributed as dist
    from vllm import ModelRegistry

    from vllm_neuron.model import registry
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state,
        stamp_spmd_rank_tensors,
    )

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank)
    out, folder = {}, Path(root)
    with torch.no_grad():
        full = _draft()
        loaded = _attempt(full.load_weights, str(DRAFT), torch.device("cpu"))
        out["a"] = _attempt(_lite_layouts, full, folder) if "ok" in loaded else loaded
        out["b"] = _attempt(_lite_then_full, full) if "ok" in loaded else loaded
    del full
    gc.collect()
    # The config comes first: vLLM checks the registered class when it builds one.
    config = _attempt(_engine_config)
    for arch, model_cls in registry.get_models():
        ModelRegistry.register_model(arch, model_cls)
    stamped = stamp_spmd_rank_tensors("meta")  # the rank tensors of meta inputs
    out["c"] = _attempt(_device_free_step, config["ok"]) if "ok" in config else config
    del stamped
    torch.save(out, folder / f"rank{rank}.pt")
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> list[dict]:
    """Each rank's readings from one two-rank world."""
    root = tmp_path_factory.mktemp("dspark_lite")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    saved, cwd = os.environ.get("VLLM_RPC_BASE_PATH"), os.getcwd()
    os.environ["VLLM_RPC_BASE_PATH"] = "."
    os.chdir(root)
    try:
        mp.spawn(_rank, args=(port, str(root)), nprocs=TP, join=True)
    finally:
        os.chdir(cwd)
        if saved is None:
            os.environ.pop("VLLM_RPC_BASE_PATH")
        else:
            os.environ["VLLM_RPC_BASE_PATH"] = saved
    return [torch.load(root / f"rank{r}.pt", weights_only=False) for r in range(TP)]


def _readings(world, item: str) -> list:
    """The item's readings on every rank; fails with the first rank's error."""
    errors = [(rank, state[item]["error"], state[item]["where"])
              for rank, state in enumerate(world) if "error" in state[item]]
    assert not errors, errors[0]
    return [state[item]["ok"] for state in world]


def test_lite_load_lays_out_every_tensor_a_full_load_does(world):
    for rank, run in enumerate(_readings(world, "a")):
        wrong = {name: (got, run["full"][name], run["expected"][name])
                 for name, got in run["lite"].items()
                 if got is None or got != run["full"][name] or got != run["expected"][name]}
        assert not wrong, (rank, wrong)


def test_full_load_after_a_lite_load_equals_a_full_load_alone(world):
    for rank, run in enumerate(_readings(world, "b")):
        # 41 parameters and 23 layouts in the tree of a full load.
        assert run["names"] == ([], []) and run["count"] == 64, (rank, run["names"], run["count"])
        assert run["differ"] == [], (rank, run["differ"])
        assert run["meta"] == [], (rank, run["meta"][:4], len(run["meta"]))


def test_proposer_load_then_a_draft_step_on_meta_inputs_completes(world):
    for rank, run in enumerate(_readings(world, "c")):
        assert run == {"ids": ((SEQS, K + 1), torch.int32), "drafts": ((SEQS, K), torch.int32)}, \
            (rank, run)
