"""The GLM-5.3-Flash weight index, rebuilt from the checkpoint's layer schedule.

The published ``model.safetensors.index.json`` is 8.4 MB of JSON in which 76,108
keys repeat a handful of per-layer leaf sets over the schedule already recorded
in ``hf-config.json``. This module rebuilds that mapping rather than vendoring
it, and pins the result with two sha256 digests taken from the published file
(:data:`KEY_SET_SHA256` and :data:`WEIGHT_MAP_SHA256`), so a wrong edit here
fails a test instead of passing quietly.

Two things are worth knowing about the rebuild:

* It yields the same ``{key: shard}`` pairs as the published file; only
  iteration order differs, because the published file lists keys in checkpoint
  write order and this module yields them sorted.
* Shard assignment is not derivable from the key names, so it is carried as
  :data:`_SHARD_RUNS`, one ``(shard number, key count)`` pair per run of
  consecutive sorted keys sharing a shard.
"""

from __future__ import annotations

import functools
import hashlib
import json
import pathlib

FIXTURES_DIR = pathlib.Path(__file__).resolve().parent
HF_CONFIG_PATH = FIXTURES_DIR / "hf-config.json"

#: The published file's own totals, and the digests that pin the rebuild.
TOTAL_KEYS = 76_108
NUM_SHARDS = 62
TOTAL_SIZE_BYTES = 328_326_771_576
KEY_SET_SHA256 = "7baf28d0d672c695e66ac4c68f3a8bcf054e52b2f364a566e6f30fcfaa4cef2d"
WEIGHT_MAP_SHA256 = "7a80a5171d8176c7928f53a97b54a1a63982b55caddacf89ec509b160b51a7fc"

#: ``layer_types`` spells the attention family of each decoder layer.
DSA_LAYER_TYPE = "deepseek_sparse_attention"
KDA_LAYER_TYPE = "linear_attention"

LAYER_PREFIX = "model.language_model.layers"
VISION_BLOCK_PREFIX = "model.visual.blocks"

# --------------------------------------------------------------------------- #
# The leaf sets. Each tuple is the suffix set one part of the schedule repeats.
# --------------------------------------------------------------------------- #

#: Present on every decoder layer, the multi-token-prediction layer included.
EVERY_LAYER_LEAVES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.o_proj.weight",
)

#: Sparse-attention layers: latent q/kv projections plus the sparse indexer.
DSA_LEAVES = (
    "self_attn.indexer.index_kpool_compress_ape",
    "self_attn.indexer.index_kpool_compress_gate",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.wq_b.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_proj_with_mqa.weight_scale_inv",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight_scale_inv",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_proj.weight_scale_inv",
    "self_attn.q_b_proj.weight",
    "self_attn.q_b_proj.weight_scale_inv",
)

#: Linear-attention layers. No scale companions anywhere in this family.
KDA_LEAVES = (
    "self_attn.A_log",
    "self_attn.b_proj.weight",
    "self_attn.dt_bias",
    "self_attn.f_a_proj.weight",
    "self_attn.f_b_proj.weight",
    "self_attn.g_a_proj.weight",
    "self_attn.g_b_proj.weight",
    "self_attn.k_conv1d.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_norm.weight",
    "self_attn.q_conv1d.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_conv1d.weight",
    "self_attn.v_proj.weight",
)

#: The first ``first_k_dense_replace`` layers carry a plain MLP.
DENSE_MLP_LEAVES = (
    "mlp.down_proj.weight",
    "mlp.down_proj.weight_scale_inv",
    "mlp.gate_proj.weight",
    "mlp.gate_proj.weight_scale_inv",
    "mlp.up_proj.weight",
    "mlp.up_proj.weight_scale_inv",
)

#: Every later layer carries a router and the always-on shared expert.
MOE_LEAVES = (
    "mlp.gate.e_score_correction_bias",
    "mlp.gate.weight",
    "mlp.shared_experts.down_proj.weight",
    "mlp.shared_experts.down_proj.weight_scale_inv",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.gate_proj.weight_scale_inv",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.up_proj.weight_scale_inv",
)

#: Per routed expert. These six times 288 experts are 74,304 of the 76,108 keys.
EXPERT_LEAVES = (
    "down_proj.weight",
    "down_proj.weight_scale_inv",
    "gate_proj.weight",
    "gate_proj.weight_scale_inv",
    "up_proj.weight",
    "up_proj.weight_scale_inv",
)

#: Multi-hyper-connections, on the decoder layers but not on the MTP layer.
HYPER_CONNECTION_LEAVES = (
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)

#: Only on the multi-token-prediction layer.
MTP_LEAVES = (
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "shared_head.norm.weight",
)

#: Per vision-tower block.
VISION_BLOCK_LEAVES = (
    "attn.k_norm.weight",
    "attn.proj.bias",
    "attn.proj.weight",
    "attn.q_norm.weight",
    "attn.qkv.bias",
    "attn.qkv.weight",
    "mlp.down_proj.bias",
    "mlp.down_proj.weight",
    "mlp.gate_proj.bias",
    "mlp.gate_proj.weight",
    "mlp.up_proj.bias",
    "mlp.up_proj.weight",
    "norm1.weight",
    "norm2.weight",
)

#: Keys that repeat over nothing.
STANDALONE_KEYS = (
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
    "model.visual.downsample.bias",
    "model.visual.downsample.weight",
    "model.visual.merger.down_proj.weight",
    "model.visual.merger.gate_proj.weight",
    "model.visual.merger.post_projection_norm.bias",
    "model.visual.merger.post_projection_norm.weight",
    "model.visual.merger.proj.weight",
    "model.visual.merger.up_proj.weight",
    "model.visual.patch_embed.proj.bias",
    "model.visual.patch_embed.proj.weight",
    "model.visual.post_layernorm.weight",
)

# Runs of consecutive sorted keys that share a shard: (shard number, key count).
# Sixty-two shards over sixty-five runs, because three shards are re-entered.
_SHARD_RUNS = (
    (1, 2), (2, 53), (3, 1242), (4, 1237), (5, 1270), (6, 1278), (7, 1237),
    (8, 1237), (9, 1280), (10, 1235), (11, 1270), (12, 1237), (13, 1278),
    (14, 1237), (15, 1237), (16, 1278), (17, 1197), (18, 1237), (19, 1237),
    (20, 1278), (21, 1237), (22, 1270), (23, 1278), (24, 1237), (25, 1237),
    (26, 1237), (27, 1278), (28, 1270), (29, 1237), (30, 1278), (31, 1237),
    (32, 1270), (33, 1237), (34, 1278), (35, 1270), (36, 1237), (37, 1278),
    (38, 1237), (39, 1237), (40, 1270), (41, 1278), (42, 1237), (43, 1237),
    (44, 1278), (45, 1237), (46, 1270), (47, 1237), (48, 1278), (49, 1237),
    (50, 1237), (51, 1278), (52, 1237), (53, 1270), (54, 1224), (1, 662),
    (2, 1098), (54, 13), (55, 1278), (56, 1237), (57, 1237), (58, 1278),
    (59, 1270), (60, 1237), (61, 1245), (62, 351),
)


def shard_filename(shard: int) -> str:
    """The published name of one shard, one-based."""
    return f"model-{shard:05d}-of-{NUM_SHARDS:05d}.safetensors"


@functools.lru_cache(maxsize=1)
def schedule() -> dict[str, object]:
    """The layer schedule the key set repeats over, read from ``hf-config.json``.

    ``dsa``/``kda`` are the two attention families by layer index, ``dense`` and
    ``moe`` the two feed-forward kinds, and ``mtp`` the single extra layer the
    checkpoint appends after the ``num_hidden_layers`` decoder layers.
    """
    text = json.loads(HF_CONFIG_PATH.read_text())["text_config"]
    vision = json.loads(HF_CONFIG_PATH.read_text())["vision_config"]
    layer_types = text["layer_types"]
    num_layers = text["num_hidden_layers"]
    assert len(layer_types) == num_layers, "layer_types must cover every layer"
    mtp = num_layers
    return {
        "num_layers": num_layers,
        "mtp_layer": mtp,
        # The MTP layer reuses the sparse-attention block.
        "dsa": [i for i, t in enumerate(layer_types) if t == DSA_LAYER_TYPE] + [mtp],
        "kda": [i for i, t in enumerate(layer_types) if t == KDA_LAYER_TYPE],
        "dense": list(range(text["first_k_dense_replace"])),
        # The MTP layer carries a router too.
        "moe": list(range(text["first_k_dense_replace"], mtp + 1)),
        "experts": text["n_routed_experts"],
        "vision_blocks": vision["depth"],
    }


def checkpoint_keys() -> tuple[str, ...]:
    """Every key the published index lists, sorted."""
    plan = schedule()
    keys: list[str] = list(STANDALONE_KEYS)

    def add(layers, leaves) -> None:
        keys.extend(
            f"{LAYER_PREFIX}.{i}.{leaf}" for i in layers for leaf in leaves
        )

    all_layers = range(plan["mtp_layer"] + 1)
    add(all_layers, EVERY_LAYER_LEAVES)
    add(plan["dsa"], DSA_LEAVES)
    add(plan["kda"], KDA_LEAVES)
    add(plan["dense"], DENSE_MLP_LEAVES)
    add(plan["moe"], MOE_LEAVES)
    add(range(plan["num_layers"]), HYPER_CONNECTION_LEAVES)
    add([plan["mtp_layer"]], MTP_LEAVES)
    keys.extend(
        f"{LAYER_PREFIX}.{i}.mlp.experts.{e}.{leaf}"
        for i in plan["moe"]
        for e in range(plan["experts"])
        for leaf in EXPERT_LEAVES
    )
    keys.extend(
        f"{VISION_BLOCK_PREFIX}.{i}.{leaf}"
        for i in range(plan["vision_blocks"])
        for leaf in VISION_BLOCK_LEAVES
    )
    assert len(set(keys)) == len(keys), "the rebuild produced a duplicate key"
    return tuple(sorted(keys))


def weight_map() -> dict[str, str]:
    """The published index's own ``{checkpoint_key: shard filename}`` mapping."""
    keys = checkpoint_keys()
    mapping: dict[str, str] = {}
    at = 0
    for shard, count in _SHARD_RUNS:
        name = shard_filename(shard)
        for key in keys[at : at + count]:
            mapping[key] = name
        at += count
    assert at == len(keys), f"shard runs cover {at} of {len(keys)} keys"
    return mapping


def index_json() -> dict[str, object]:
    """The whole published index object: ``metadata`` then ``weight_map``."""
    return {
        "metadata": {"total_size": TOTAL_SIZE_BYTES},
        "weight_map": weight_map(),
    }


def key_set_digest() -> str:
    """sha256 of the sorted key list, newline joined."""
    return hashlib.sha256("\n".join(checkpoint_keys()).encode()).hexdigest()


def weight_map_digest() -> str:
    """sha256 of the sorted ``key shard`` lines, which pins the shard map too."""
    mapping = weight_map()
    body = "\n".join(f"{key} {mapping[key]}" for key in checkpoint_keys())
    return hashlib.sha256(body.encode()).hexdigest()
