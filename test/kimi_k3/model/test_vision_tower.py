# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 vision tower, projector and encoder hook in CPU mode.

The tower is the checkpoint's first two blocks, its final norm and the projector.
It encodes 16 x 16-patch images that the registered K3 processor gives, as uint8
values: a fresh process stands for a worker, registers the plugin's model classes
and reads the model config the parent built. Items with more than one rank run
each rank as a gloo process with the encoder groups built on top. The reference is
the checkpoint's code transcribed to float64, its normalize included. Set
``KIMI_K3_CHECKPOINT`` to the checkpoint directory; the default is
``~/models/moonshotai/Kimi-K3``. Run with ``VLLM_NEURON_CPU_MODE=1`` and
``NKI_SIMULATOR=1``, so the kernels run on the NKI simulator.
"""

import copy
import io
import json
import math
import multiprocessing
import os
import pickle
import socket
import traceback
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

CHECKPOINT = Path(
    os.environ.get(
        "KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3"
    )
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

IMAGES = Path(__file__).parents[1] / "serving" / "vision" / "images"
#: Item images: two 224-pixel crops (16 x 16 patches) and one of 78 x 78 patches.
SOURCES = {
    "a": ("00-test_Biology_146.png", 224),
    "b": ("01-validation_Literature_27.png", 224),
    "large": ("00-test_Biology_146.png", 1092),
}
LAYERS = 2
#: One image of 256 patches and 128 pad rows; one bucket, one block.
BUCKET = 384
TEXT = 7168
#: ``nn.RMSNorm(eps=None)`` on the tower's activations; the projector's norm eps.
TOWER_EPS = float(torch.finfo(torch.float32).eps)
POST_EPS = 1e-5
#: The largest f the tower's library calls record, each hashed before its device
#: reading: the MLP call's 5.5 (the QKV call's is 4.0).
TOWER_F = 5.5


# ---------------------------------------------------------------- reference


def weight_map() -> dict[str, str]:
    """The checkpoint's tensor-to-shard map."""
    with open(CHECKPOINT / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def read(keys) -> dict[str, torch.Tensor]:
    """Checkpoint tensors by key, as stored."""
    shards = weight_map()
    out = {}
    for key in keys:
        with safe_open(CHECKPOINT / shards[key], framework="pt") as f:
            out[key] = f.get_tensor(key)
    return out


def reference_weights(layers: int = LAYERS) -> dict[str, torch.Tensor]:
    """The tower's first ``layers`` blocks, final norm, patch embed and projector, float64."""
    keys = [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.patch_embed.pos_emb.weight",
        "vision_tower.encoder.final_layernorm.weight",
        "mm_projector.proj.0.weight",
        "mm_projector.proj.2.weight",
        "mm_projector.post_norm.weight",
    ]
    for i in range(layers):
        for name in ("wqkv", "wo", "mlp.fc0", "mlp.fc1", "norm0", "norm1"):
            keys.append(f"vision_tower.encoder.blocks.{i}.{name}.weight")
    return {k: v.double() for k, v in read(keys).items()}


def normalize(pixels) -> torch.Tensor:
    """The checkpoint's normalize of uint8 pixels in float64: mean and std are 0.5, so x / 127.5 - 1."""
    return pixels.double() / 127.5 - 1


def rms_norm(x, weight, eps):
    """The checkpoint's RMSNorm in float64."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def freqs_cis(h: int, w: int) -> torch.Tensor:
    """The checkpoint's 2D rotary table of an h x w grid, [h * w, 64] complex128."""
    flat = torch.arange(h * w, dtype=torch.float64)
    x_pos, y_pos = flat % w, flat // w
    freqs = 1.0 / 10000 ** (torch.arange(0, 128, 4, dtype=torch.float64) / 128)
    x_cis = torch.polar(torch.ones(h * w, 32, dtype=torch.float64), torch.outer(x_pos, freqs))
    y_cis = torch.polar(torch.ones(h * w, 32, dtype=torch.float64), torch.outer(y_pos, freqs))
    return torch.stack([x_cis, y_cis], dim=-1).reshape(h * w, 64)


def apply_rope(x, cis):
    """The checkpoint's rotary embedding on interleaved pairs, x [n, heads, 128]."""
    xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(xc * cis.unsqueeze(-2)).flatten(-2)


def reference(pixels, grid, w, layers: int = LAYERS) -> torch.Tensor:
    """The checkpoint's tower, patch merger and projector in float64: [h * w / 4, 7168]."""
    h, wd = grid
    n = h * wd
    p = "vision_tower.encoder.blocks"
    x = normalize(pixels).reshape(n, -1) @ w["vision_tower.patch_embed.proj.weight"].reshape(1024, -1).T
    table = w["vision_tower.patch_embed.pos_emb.weight"]
    if (h, wd) == tuple(table.shape[:2]):
        pos = table.flatten(0, 1)
    else:
        pos = F.interpolate(table.permute(2, 0, 1)[None], size=(h, wd), mode="bilinear")
        pos = pos[0].permute(1, 2, 0).flatten(0, 1)
    x = x + pos
    cis = freqs_cis(h, wd)
    for i in range(layers):
        normed = rms_norm(x, w[f"{p}.{i}.norm0.weight"], TOWER_EPS)
        qkv = (normed @ w[f"{p}.{i}.wqkv.weight"].T).view(n, 3, 12, 128)
        q, k, v = qkv.unbind(1)
        q, k = apply_rope(q, cis), apply_rope(k, cis)
        scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(128)
        attn = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).reshape(n, -1)
        x = x + attn @ w[f"{p}.{i}.wo.weight"].T
        normed = rms_norm(x, w[f"{p}.{i}.norm1.weight"], TOWER_EPS)
        act = F.gelu(normed @ w[f"{p}.{i}.mlp.fc0.weight"].T, approximate="tanh")
        x = x + act @ w[f"{p}.{i}.mlp.fc1.weight"].T
    x = rms_norm(x, w["vision_tower.encoder.final_layernorm.weight"], TOWER_EPS)
    merged = x.view(h // 2, 2, wd // 2, 2, -1).permute(0, 2, 1, 3, 4).reshape(n // 4, -1)
    out = F.gelu(merged @ w["mm_projector.proj.0.weight"].T) @ w["mm_projector.proj.2.weight"].T
    return rms_norm(out, w["mm_projector.post_norm.weight"], POST_EPS)


def bound_reading(got, ref):
    """Worst |got - ref| over 2^-7 (|ref| + row RMS of ref), and the lowest row cosine."""
    got, ref = got.double(), ref.double()
    rms = ref.square().mean(-1, keepdim=True).sqrt()
    reading = ((got - ref).abs() / (2**-7 * (ref.abs() + rms))).max().item()
    cosine = F.cosine_similarity(got, ref, dim=-1).min().item()
    return reading, cosine


# ---------------------------------------------------------------- inputs


def _process(config_bytes, images, queue) -> None:
    """Run the registered K3 processor on each image in a fresh process, as a worker builds it."""
    try:
        from PIL import Image
        from vllm import ModelRegistry
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.multimodal.processing.context import TimingContext
        from vllm.multimodal.processing.inputs import ProcessorInputs

        from vllm_neuron.model import registry

        model_config = pickle.loads(config_bytes)
        for arch, model_cls in registry.get_models():
            ModelRegistry.register_model(arch, model_cls)
        processor = MULTIMODAL_REGISTRY.create_processor(model_config)
        info = processor.info
        ids = info.get_tokenizer().encode(
            f"{info.get_hf_config().image_placeholder}Describe.", add_special_tokens=False
        )
        out = {}
        for name, (file, side) in images.items():
            image = Image.open(IMAGES / file).convert("RGB").resize((side, side), Image.BICUBIC)
            result = processor.apply(
                ProcessorInputs(list(ids), info.parse_mm_data({"image": [image]})),
                TimingContext(),
            )
            item = result["mm_kwargs"]["image"][0]
            out[name] = {k: item[k].data.clone() for k in ("pixel_values", "grid_thws")}
        # Tensors cross the queue as bytes, so they outlive this process.
        buffer = io.BytesIO()
        torch.save(out, buffer)
        queue.put(buffer.getvalue())
    except Exception:
        queue.put({"error": traceback.format_exc()})


@pytest.fixture(scope="module")
def processed() -> dict[str, dict[str, torch.Tensor]]:
    """Each item image as the processor gives it: pixel_values and grid_thws [1, 3]."""
    from vllm.config import ModelConfig

    model_config = ModelConfig(
        model=str(CHECKPOINT),
        tokenizer=str(CHECKPOINT),
        trust_remote_code=True,
        max_model_len=131072,
        limit_mm_per_prompt={"image": 7},
    )
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(target=_process, args=(pickle.dumps(model_config), SOURCES, queue))
    child.start()
    out = queue.get(timeout=900)
    child.join()
    assert not isinstance(out, dict), out.get("error")
    out = torch.load(io.BytesIO(out))
    return {
        name: {"pixel_values": d["pixel_values"], "grid_thws": d["grid_thws"].reshape(1, 3)}
        for name, d in out.items()
    }


def hf_config(layers: int = LAYERS) -> dict:
    """The checkpoint's config with one text layer and ``layers`` tower layers."""
    from vllm.transformers_utils.config import get_config

    config = copy.deepcopy(get_config(str(CHECKPOINT), trust_remote_code=False).to_dict())
    text = config["text_config"]
    attn = text["linear_attn_config"]
    for key in ("kda_layers", "full_attn_layers"):
        attn[key] = [i for i in attn[key] if i <= 1]
    text["num_hidden_layers"] = 1
    # The loaded vision config carries the layer count under both names.
    for key in ("vt_num_hidden_layers", "num_hidden_layers"):
        config["vision_config"][key] = layers
    return config


def encoder_config(tp: int, dp: int, buckets=(BUCKET,), block: int = BUCKET):
    """The encoder config of a CPU run."""
    from vllm_neuron.model.neuron_config import EncoderNeuronConfig

    return EncoderNeuronConfig(
        num_encoder_input_buckets=list(buckets),
        encoder_attention_block_size=block,
        tp_size=tp,
        dp_size=dp,
    )


def build_model(config: dict, encoder, load: bool = True):
    """The model on the meta device as the runner builds it, its tower loaded from the checkpoint."""
    from vllm_neuron.model.kimi_k3 import KimiK3ForConditionalGeneration
    from vllm_neuron.model.kimi_k3.weight_loaders_mxfp4 import load_weights

    with torch.device("meta"):
        model = KimiK3ForConditionalGeneration.from_configs(config, None, encoder)
    if load:
        root = nn.Module()
        root.visual = model.visual
        load_weights(root, model.text_config, CHECKPOINT, torch.device("cpu"), pipelined=False)
    return model


def new_cache(blocks: int, rows: int):
    """An encoder cache on the CPU."""
    from vllm_neuron.vllm.worker.encoder_cache_blocks import EncoderCacheBlocks

    return EncoderCacheBlocks(blocks, rows, TEXT, device="cpu", min_hold_time_ms=0)


# ---------------------------------------------------------------- ranks


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank(rank, world, tp, dp, port, job, payload, out_dir) -> None:
    """One gloo rank: encoder groups, the model, then ``job``; saves the job's result."""
    import torch.distributed as dist

    from vllm_neuron.parallel import neuron_parallel_state as nps

    torch.set_num_threads(max(1, 8 // world))
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    nps.initialize_neuron_parallel_state(
        encoder_tp_size=tp,
        encoder_dp_size=dp,
        tp_global_ranks=list(range(world)),
        local_rank=rank,
    )
    nps.stamp_spmd_rank_tensors("cpu")
    model = build_model(hf_config(), encoder_config(tp, dp))
    torch.save(globals()[job](model, payload), Path(out_dir) / f"{job}-{rank}.pt")
    dist.destroy_process_group()


def run_ranks(world, tp, dp, job, payload, tmp_path) -> list:
    """Run ``job`` on ``world`` gloo ranks; return each rank's result."""
    torch.multiprocessing.spawn(
        _rank,
        args=(world, tp, dp, _free_port(), job, payload, str(tmp_path)),
        nprocs=world,
    )
    return [torch.load(tmp_path / f"{job}-{r}.pt") for r in range(world)]


def _image_rows(model, payload):
    """Encode one image through the hook; return its rows from the cache."""
    cache = new_cache(3, BUCKET // 4)
    model.embed_multimodal(encoder_cache=cache, mm_hashes=["a"], **payload)
    rows = payload["grid_thws"][0, 1:].prod().item() // 4
    return cache.buffer[cache.get_block_ids("a")].reshape(-1, TEXT)[:rows].clone()


def test_tower_matches_the_float64_reference(processed, tmp_path):
    """Item 1: TP4 inside class Bs at f, row cosine at least 0.9999; TP2 and TP1 printed beside."""
    image = processed["a"]
    grid = tuple(image["grid_thws"][0, 1:].tolist())
    ref = reference(image["pixel_values"], grid, reference_weights())
    misses = []
    # TP4 is the served layout. TP2 is a repeat check. At TP1 the QKV width, 4,608,
    # takes the wrapper's torch branch. Only TP4 is graded.
    runs = ((4, "graded"), (2, "repeat check, printed"), (1, "QKV torch branch, printed"))
    for tp, label in runs:
        rows = run_ranks(tp, tp, 1, "_image_rows", image, tmp_path)
        for rank, got in enumerate(rows):
            reading, cosine = bound_reading(got, ref)
            print(f"TP{tp} rank {rank} ({label}): "
                  f"worst error / class-B bound {reading:.3f}, f {TOWER_F}; "
                  f"lowest row cosine {cosine:.7f}")
            if label == "graded" and not (reading <= TOWER_F and cosine >= 0.9999):
                misses.append(f"TP{tp} rank {rank}")
    assert not misses, f"outside class Bs at f {TOWER_F}: {misses}"


#: GELU entry points that the runtime replaces with an op the tracer cannot read.
_REPLACED_GELU = ("gelu", "Gelu")


def test_captured_tower_graph_holds_no_runtime_gelu(processed):
    """Item 4: 0 runtime-replaced GELU ops in the captured tower graph."""
    model = build_model(hf_config(), encoder_config(1, 1))
    graphs = []

    def record(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    from vllm_neuron.model.kimi_k3.vision_encoder import gelu as traceable

    model.visual = torch.compile(model.visual, backend=record, fullgraph=True, dynamic=False)
    cache = new_cache(3, BUCKET // 4)
    model.embed_multimodal(encoder_cache=cache, mm_hashes=["a"], **processed["a"])
    nodes = [n for gm in graphs for n in gm.graph.nodes if n.op in ("call_function", "call_method")]
    replaced = [n for n in nodes if n.target is not traceable and any(
        s in getattr(n.target, "__name__", str(n.target)) for s in _REPLACED_GELU)]
    print(f"{len(graphs)} graph(s); {sum(n.target is traceable for n in nodes)} traceable GELU; "
          f"{len(replaced)} runtime-replaced GELU: {[str(n.target) for n in replaced]}")
    assert len(graphs) == 1 and not replaced


def nonfinite(tensor) -> int:
    """Count of NaN and infinite elements."""
    return int((~torch.isfinite(tensor)).sum())


def test_vision_tensors_are_finite():
    """Item 5: all 168 vision tensors hold 0 NaN and 0 infinite values; one planted NaN reads 1."""
    keys = [k for k in weight_map() if not k.startswith("language_model.")]
    counts = {k: nonfinite(t) for k, t in read(keys).items()}
    planted = read([keys[0]])[keys[0]].clone()
    planted.view(-1)[0] = float("nan")
    print(f"{len(counts)} tensors, {sum(counts.values())} non-finite; planted copy reads {nonfinite(planted)}")
    assert (len(counts), sum(counts.values()), nonfinite(planted)) == (168, 0, 1)


def _refusal(build) -> str | None:
    """The message of the ValueError ``build()`` raises, or None when it builds."""
    try:
        build()
    except ValueError as e:
        return str(e)
    return None


def test_qkv_width_check_outside_cpu_mode(monkeypatch):
    """Item 6: with CPU mode off the build raises at QKV width 4,608 and passes at 2,304 and 1,152."""
    from vllm_neuron.model.kimi_k3.config import KimiK3VisionConfig
    from vllm_neuron.model.kimi_k3.vision_encoder import KimiK3VisionBlock

    vision = KimiK3VisionConfig()

    def block(tp):
        with torch.device("meta"):
            KimiK3VisionBlock(vision, tp, torch.bfloat16)

    result = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("VLLM_NEURON_CPU_MODE", mode)
        result[mode] = {3 * 1536 // tp: _refusal(lambda: block(tp)) is None for tp in (1, 2, 4)}
    print("builds by QKV width, CPU mode off and on:", result)
    assert result == {"0": {4608: False, 2304: True, 1152: True},
                      "1": {4608: True, 2304: True, 1152: True}}


def test_layer_count_names_must_agree():
    """Item 7: the build raises at 27 and 2 layers under the two names; 27 and 27 builds 27."""
    config = hf_config(27)
    split = copy.deepcopy(config)
    split["vision_config"]["num_hidden_layers"] = 2
    refused = _refusal(lambda: build_model(split, encoder_config(1, 1), load=False))
    model = build_model(config, encoder_config(1, 1), load=False)
    layers = len(model.visual.vision_tower.encoder.blocks)
    print(f"27 and 2: {refused}; 27 and 27: {layers} layers")
    assert refused is not None and "layer counts differ" in refused
    assert layers == 27
