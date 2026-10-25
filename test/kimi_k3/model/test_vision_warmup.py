# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 vision tower calls against the encoder warmup, in CPU mode.

A rank's first tower call that the warmup did not make compiles a new program.
The item runs the encoder hook at the served encoder config on each encoder DP
position, with the split on and off, and keys every tower call by its form, the
shape and dtype of each tensor input, and the inputs that share one storage.
Strides are not in the key: in CPU mode ``.to`` keeps the views that a device
copy replaces. Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import math
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from test_vision_tower import (  # noqa: F401  (processed and pytestmark are used by pytest)
    build_model,
    encoder_config,
    hf_config,
    new_cache,
    processed,
    pytestmark,
)

SERVED_BUCKETS = [5504, 33536]
SERVED_BLOCK = 5504
TP, GROUPS = 4, 8
#: One image and two images at 5,504, one image at 33,536.
IMAGE_SETS = (["a"], ["a", "b"], ["large"])


def call_key(kwargs) -> tuple:
    """A tower call's form, its tensor inputs' shapes and dtypes, and inputs that share storage."""
    tensors = {k: v for k, v in sorted(kwargs.items()) if isinstance(v, torch.Tensor)}
    by_storage = defaultdict(list)
    for name, tensor in tensors.items():
        by_storage[tensor.untyped_storage().data_ptr()].append(name)
    form = (
        kwargs["pixel_values"].shape[0],
        kwargs.get("gather", False),
        kwargs.get("slice_rows", 0),
    )
    shapes = tuple((k, tuple(v.shape), v.dtype) for k, v in tensors.items())
    shared = tuple(sorted(tuple(names) for names in by_storage.values() if len(names) > 1))
    return form, shapes, shared


class _Recorder(nn.Module):
    """Stands in for the compiled tower: records each call's bucket, key and kind."""

    def __init__(self):
        super().__init__()
        self.bucket = None
        self.calls = []

    def forward(self, **kwargs):
        real = kwargs.get("slice_rows", 0) or bool(kwargs["bound_max"].any())
        kind = "image or slice" if real else "padding"
        self.calls.append((self.bucket, call_key(kwargs), kind))
        return kwargs["encoder_cache_buffer"]


def warmup_inputs(model, served, bucket: int, cache) -> dict:
    """The worker's warmup inputs: synthetic inputs, the cache, write ids padded to DP."""
    inputs = model.build_encoder_synthetic_inputs(bucket, served, torch.device("cpu"))
    blocks = math.ceil(math.ceil(bucket / SERVED_BLOCK) / GROUPS) * GROUPS
    inputs["encoder_cache_buffer"] = cache.buffer
    inputs["write_block_ids"] = torch.zeros(blocks, dtype=torch.int64)
    return inputs


def test_every_tower_call_matches_a_warmup_call(processed, monkeypatch):
    """Item 1: each tower call on each encoder DP position has a warmup call's key at its bucket."""
    from vllm_neuron.model.kimi_k3 import vision_encoder as ve

    served = encoder_config(TP, GROUPS, SERVED_BUCKETS, SERVED_BLOCK)
    model = build_model(hf_config(27), served, load=False)
    recorder = _Recorder()
    model.visual = recorder
    encode = model._encode

    def encode_at(bucket, images, cache):
        recorder.bucket = bucket
        encode(bucket, images, cache)

    monkeypatch.setattr(model, "_encode", encode_at)
    cache = new_cache(16, SERVED_BLOCK // 4)
    kinds, unmatched = Counter(), Counter()
    for split in (True, False):
        model.split_lone_image = split
        for position in range(GROUPS):
            monkeypatch.setattr(ve, "_dp_position", lambda p=position: (p, GROUPS))
            recorder.calls.clear()
            for bucket in SERVED_BUCKETS:
                recorder.bucket = bucket
                model.get_encoder()(**warmup_inputs(model, served, bucket, cache))
            warmed = {(bucket, key) for bucket, key, _ in recorder.calls}
            recorder.calls.clear()
            for names in IMAGE_SETS:
                model.embed_multimodal(
                    pixel_values=torch.cat([processed[n]["pixel_values"] for n in names]),
                    grid_thws=torch.cat([processed[n]["grid_thws"] for n in names]),
                    encoder_cache=cache,
                    mm_hashes=names,
                )
                for name in names:
                    cache.free(name)
            for bucket, key, kind in recorder.calls:
                kinds[kind] += 1
                if (bucket, key) in warmed:
                    continue
                unmatched[kind] += 1
                shapes = any(k[0] == key[0] and k[1] == key[1] for b, k in warmed if b == bucket)
                print(f"split {split}, position {position}, bucket {bucket}, {kind} form {key[0]}: "
                      f"a warmup call's shapes and dtypes {shapes}, shared storage {key[2]}")
    print(f"tower calls {dict(kinds)}; with no warmup call {dict(unmatched)}")
    assert (kinds, unmatched) == (Counter({"image or slice": 15, "padding": 33}), Counter())
