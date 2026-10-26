# SPDX-License-Identifier: Apache-2.0
"""Image rows written into one K3 prefill chunk in place, class E against the checkpoint.

The reference is ``KimiK3ForConditionalGeneration._merge_input_ids_with_image_features``
from the checkpoint's ``modeling_kimi_k3.py`` under ``KIMI_K3_MODEL_PATH``, run in
fp64 on the unexpanded prompt. The call under test is ``merge_image_rows`` on one
8192-row chunk of width 7168, with each image's rows in encoder cache blocks of
1376 rows (the 5504-patch bucket after the 2 x 2 merge) and the block positions
the runner builds. Class E: every output word equals the reference's, compared as
integers. Every class E item fails if the merge takes its torch path.

The kernel runs on a Neuron device through the native compile backend, or on
the host under ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import importlib
import json
import os
import sys
import types
from unittest import mock

import pytest
import torch

from vllm_neuron.functional import merge_image_rows as merge_module

SEED = 20261024
CHUNK = 8192
BLOCK = 1376
NUM_BLOCKS = 14
IMAGE_BLOCK = 5
IMAGE_ROWS = 1369  # one 1024 x 1024 image: a 74 x 74 grid, merged 2 x 2
IMAGE_START = 1000
LARGEST_IMAGE_ROWS = 8372  # 33488 patches in the 33536-patch bucket
LARGEST_BLOCKS = (3, 9, 0, 12, 6, 1, 10)
LARGEST_START = 100
NAN_SLOT = 700
NAN_PAD_SLOT = 1372

_compiled = {}


def _import_modeling(path):
    """Import the checkpoint's model code without its text model.

    The merge never calls ``modeling_kimi_linear``, which does not import under
    transformers 5, so a stand-in module takes its name.
    """
    package = types.ModuleType("kimi_k3_merge_reference")
    package.__path__ = [path]
    text_model = types.ModuleType("kimi_k3_merge_reference.modeling_kimi_linear")
    text_model.KimiLinearForCausalLM = object
    sys.modules[package.__name__] = package
    sys.modules[text_model.__name__] = text_model
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module("kimi_k3_merge_reference.modeling_kimi_k3")
    finally:
        sys.dont_write_bytecode = write_bytecode


@pytest.fixture(scope="module")
def reference():
    """The checkpoint's merge with the token ids and width its config names."""
    path = os.environ.get("KIMI_K3_MODEL_PATH")
    if not path or not os.path.isfile(os.path.join(path, "modeling_kimi_k3.py")):
        pytest.skip("KIMI_K3_MODEL_PATH must name the Kimi K3 checkpoint directory")
    return _load_reference(path)


def _load_reference(path):
    """Bind the checkpoint's merge to its config's token ids and row width."""
    with open(os.path.join(path, "config.json")) as fh:
        config = json.load(fh)
    keys = ("media_placeholder_token_id", "pad_token_id", "ignore_index")
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(**{k: config[k] for k in keys})
    )
    cls = _import_modeling(path).KimiK3ForConditionalGeneration
    return types.SimpleNamespace(
        hidden=config["text_config"]["hidden_size"],
        placeholder=config["media_placeholder_token_id"],
        merge=lambda *a: cls._merge_input_ids_with_image_features(model, *a)[0][0],
    )


def _generator(regime):
    return torch.Generator().manual_seed(SEED + regime)


def _rows(n, width, regime):
    """Standard normal rows, rounded once to bf16 as the call receives them."""
    x = torch.randn(n, width, generator=_generator(regime))
    return x.to(torch.bfloat16).double()


def _prompt(reference, text, placeholder, image, start):
    """The expanded prompt's rows and the checkpoint's merge of the unexpanded one.

    The expanded prompt holds the placeholder's embedding at every image row; the
    checkpoint's prompt holds one placeholder token, which it expands itself.
    """
    total = text.shape[0] + image.shape[0]
    is_image = torch.zeros(total, dtype=torch.bool)
    is_image[start : start + image.shape[0]] = True
    expanded = torch.empty(total, reference.hidden, dtype=torch.float64)
    expanded[~is_image] = text
    expanded[is_image] = placeholder
    ids = torch.randint(0, 100000, (total,), generator=_generator(90))
    ids[is_image] = reference.placeholder
    keep = ~is_image
    keep[start] = True
    mask = torch.ones(1, int(keep.sum()), dtype=torch.long)
    merged = reference.merge([image], expanded[keep][None], ids[keep][None], mask)
    return expanded, merged


def _cache(image, block_ids, width, regime):
    """The encoder cache: random rows, the image's rows in its blocks in order."""
    cache = _rows(NUM_BLOCKS * BLOCK, width, regime).view(NUM_BLOCKS, BLOCK, width)
    for k, block in enumerate(block_ids):
        part = image[k * BLOCK : (k + 1) * BLOCK]
        cache[block, : part.shape[0]] = part
    return cache


def _positions(rows, start, num_blocks, chunk):
    """The runner's positions: chunk-local rows, the sentinel CHUNK elsewhere."""
    token = torch.arange(num_blocks * BLOCK).view(num_blocks, BLOCK)
    local = start + token - chunk * CHUNK
    real = (token < rows) & (local >= 0) & (local < CHUNK)
    return torch.where(real, local, CHUNK).to(torch.int32)


def _no_torch_path(*args, **kwargs):
    raise AssertionError("the merge took its torch path instead of the NKI kernel")


def _merge_call(hidden, cache, block_ids, positions, rank):
    """The forward's call on one chunk, with no sequence sharding."""
    return merge_module.merge_image_rows(hidden, cache, block_ids, positions, rank)


def _kernel(hidden, cache, block_ids, positions):
    """Run the merge where its kernel runs; return the merged rows on the host."""
    args = (
        hidden.to(torch.bfloat16),
        cache.to(torch.bfloat16),
        torch.tensor(block_ids, dtype=torch.int32),
        positions,
        torch.zeros(1, dtype=torch.int32),
    )
    with mock.patch.object(merge_module, "_merge_image_rows_torch", _no_torch_path):
        if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
            if os.environ.get("NKI_SIMULATOR") != "1":
                pytest.skip("the kernel runs on the host only under NKI_SIMULATOR=1")
            out = _merge_call(*args)
        else:
            from libtorch_neuronx_lite.compile.native_backend import register

            if not _compiled:
                _compiled["merge"] = torch.compile(
                    _merge_call, backend=register(), fullgraph=True, dynamic=False
                )
            device = torch.device("neuron", 0)
            out = _compiled["merge"](*(a.to(device) for a in args)).cpu()
    return out


def _assert_class_e(out, expected):
    """Class E: bf16 out, every word equal to the reference's."""
    assert out.dtype == torch.bfloat16, f"output dtype {out.dtype}"
    assert out.shape == expected.shape, f"output shape {tuple(out.shape)}"
    differ = out.view(torch.int16) != expected.to(torch.bfloat16).view(torch.int16)
    rows = int(differ.any(-1).sum())
    assert int(differ.sum()) == 0, f"{int(differ.sum())} words differ in {rows} rows"


def _one_block(reference, image, start, regime, text=None):
    """One image in one block: the merged chunk and the reference's rows."""
    width = reference.hidden
    if text is None:
        text = _rows(CHUNK - image.shape[0], width, regime + 1)
    placeholder = _rows(1, width, regime + 2)
    expanded, expected = _prompt(reference, text, placeholder, image, start)
    cache = _cache(image, (IMAGE_BLOCK,), width, regime + 3)
    positions = _positions(image.shape[0], start, 1, 0)
    return _kernel(expanded, cache, (IMAGE_BLOCK,), positions), expected


def test_correlated_rows(reference):
    """Image rows 0.9 u + 0.1 noise, one channel 100x the rest."""
    width = reference.hidden
    g = _generator(10)
    u = torch.randn(width, generator=g, dtype=torch.float64)
    image = 0.9 * u + 0.1 * torch.randn(IMAGE_ROWS, width, generator=g, dtype=torch.float64)
    image[:, width // 3] *= 100
    image = image.to(torch.bfloat16).double()
    _assert_class_e(*_one_block(reference, image, IMAGE_START, 10))


def test_repeated_tokens(reference):
    """One text token repeated and one image row repeated."""
    width = reference.hidden
    text = _rows(1, width, 20).expand(CHUNK - IMAGE_ROWS, width)
    image = _rows(1, width, 21).expand(IMAGE_ROWS, width)
    _assert_class_e(*_one_block(reference, image, IMAGE_START, 20, text=text))


def test_largest_bucket(reference):
    """The 33536-patch bucket's image in 7 blocks, over two chunks."""
    width = reference.hidden
    image = _rows(LARGEST_IMAGE_ROWS, width, 40)
    text = _rows(2 * CHUNK - LARGEST_IMAGE_ROWS, width, 41)
    placeholder = _rows(1, width, 42)
    expanded, expected = _prompt(reference, text, placeholder, image, LARGEST_START)
    cache = _cache(image, LARGEST_BLOCKS, width, 43)
    for chunk in (0, 1):
        rows = slice(chunk * CHUNK, (chunk + 1) * CHUNK)
        positions = _positions(LARGEST_IMAGE_ROWS, LARGEST_START, 7, chunk)
        out = _kernel(expanded[rows], cache, LARGEST_BLOCKS, positions)
        _assert_class_e(out, expected[rows])


def test_one_nan_row(reference):
    """A NaN cache row reaches its target row only; a NaN pad slot reaches none."""
    width = reference.hidden
    image = _rows(IMAGE_ROWS, width, 50)
    image[NAN_SLOT] = float("nan")
    text = _rows(CHUNK - IMAGE_ROWS, width, 51)
    placeholder = _rows(1, width, 52)
    expanded, expected = _prompt(reference, text, placeholder, image, IMAGE_START)
    cache = _cache(image, (IMAGE_BLOCK,), width, 53)
    cache[IMAGE_BLOCK, NAN_PAD_SLOT] = float("nan")
    positions = _positions(IMAGE_ROWS, IMAGE_START, 1, 0)
    out = _kernel(expanded, cache, (IMAGE_BLOCK,), positions)
    nan_rows = torch.isnan(out.double()).any(-1).nonzero().flatten().tolist()
    assert nan_rows == [IMAGE_START + NAN_SLOT], f"NaN in rows {nan_rows[:8]}"
    _assert_class_e(out, expected)


@pytest.mark.parametrize("start", (0, CHUNK - BLOCK), ids=("first_row", "last_row"))
def test_block_first_and_last_slot(reference, start):
    """A full block: its first slot lands on the chunk's row 0, its last on row 8191."""
    image = _rows(BLOCK, reference.hidden, 60)
    _assert_class_e(*_one_block(reference, image, start, 60))


@pytest.mark.parametrize("path", ("kernel", "torch"))
def test_returns_the_stream_it_received(reference, path):
    """On the host the call returns the stream it received, its image rows
    written and every other row unchanged, by the kernel or the torch path."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        pytest.skip("the returned tensor's identity is read on the host")
    if path == "kernel" and os.environ.get("NKI_SIMULATOR") != "1":
        pytest.skip("the kernel runs on the host only under NKI_SIMULATOR=1")
    width = reference.hidden
    hidden = _rows(CHUNK, width, 70).to(torch.bfloat16)
    image = _rows(IMAGE_ROWS, width, 71)
    cache = _cache(image, (IMAGE_BLOCK,), width, 72).to(torch.bfloat16)
    block_ids = torch.tensor((IMAGE_BLOCK,), dtype=torch.int32)
    positions = _positions(IMAGE_ROWS, IMAGE_START, 1, 0)
    before = hidden.clone()
    if path == "kernel":
        venue = mock.patch.object(merge_module, "_merge_image_rows_torch", _no_torch_path)
    else:
        venue = mock.patch.object(merge_module, "can_run_kernel", lambda *args: False)
    with venue:
        out = _merge_call(hidden, cache, block_ids, positions, torch.zeros(1, dtype=torch.int32))
    assert out is hidden, "the call returned a new tensor, not the stream it received"
    image_rows = slice(IMAGE_START, IMAGE_START + IMAGE_ROWS)
    written = out[image_rows].view(torch.int16) != cache[IMAGE_BLOCK, :IMAGE_ROWS].view(torch.int16)
    assert int(written.sum()) == 0, f"{int(written.sum())} image-row words differ from the cache"
    keep = torch.ones(CHUNK, dtype=torch.bool)
    keep[image_rows] = False
    moved = out[keep].view(torch.int16) != before[keep].view(torch.int16)
    assert int(moved.sum()) == 0, f"{int(moved.any(-1).sum())} text rows changed"


@mock.patch.object(merge_module, "can_run_kernel", lambda *args: True)
def test_known_bad_inputs(reference):
    """On a kernel venue, the kernel check refuses fp32 rows and cache rows of
    another width; the kernel path refuses a host rank."""
    rows = torch.empty(CHUNK, reference.hidden, dtype=torch.bfloat16, device="meta")
    cache = torch.empty(
        NUM_BLOCKS, BLOCK, reference.hidden, dtype=torch.bfloat16, device="meta"
    )
    wide = torch.empty(
        NUM_BLOCKS, BLOCK, reference.hidden + 128, dtype=torch.bfloat16, device="meta"
    )
    assert merge_module._can_use_kernel(rows, cache)
    assert not merge_module._can_use_kernel(rows.float(), cache)
    assert not merge_module._can_use_kernel(rows, wide)
    block_ids = torch.zeros(1, dtype=torch.int32)
    positions = _positions(IMAGE_ROWS, IMAGE_START, 1, 0)
    with pytest.raises(TypeError, match="runtime rank tensor"):
        merge_module.merge_image_rows(rows, cache, block_ids, positions, 0)
