# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 vision tower with one image split over the encoder DP groups, in CPU mode.

Items 1 and 2 run the checkpoint's first two blocks, its final norm and the
projector at encoder TP2 by DP2, each rank a gloo process, against the tower
test's float64 reference. Item 3 reads the slices at the served bucket over 8
groups, and item 4 the warmup's forms at the served encoder config. Run with
``VLLM_NEURON_CPU_MODE=1`` and ``NKI_SIMULATOR=1``.
"""

import math
import socket
from pathlib import Path

import torch
import torch.nn as nn
from test_vision_tower import (  # noqa: F401  (processed and pytestmark are used by pytest)
    BUCKET,
    TEXT,
    TOWER_F,
    bound_reading,
    build_model,
    encoder_config,
    hf_config,
    new_cache,
    processed,
    pytestmark,
    reference,
    reference_weights,
)

#: Merged rows of one 16 x 16-patch image.
IMAGE_ROWS = 64
SERVED_BUCKETS = [5504, 33536]
SERVED_BLOCK = 5504
TP, DP = 2, 2
#: A lone image at the test bucket: 2 slices of 192 rows, each padded to 256.
TEST_SPLIT = (256, False, 192)
#: The served image, 1036 x 1036 pixels: 74 x 74 patches at the 5,504-row bucket.
SERVED_GRID = (74, 74)
GROUPS = 8
SLICE_ROWS, SLICE_PADDED = 688, 768
KEY_BOUND = 5476
PATCH_DIM = 588


# ---------------------------------------------------------------- ranks


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank(rank, world, tp, dp, port, job, payload, out_dir) -> None:
    """One gloo rank: the encoder groups, then ``job``; saves the job's result."""
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
    torch.save(globals()[job](tp, dp, payload), Path(out_dir) / f"{job}-{rank}.pt")
    dist.destroy_process_group()


def run_ranks(tp, dp, job, payload, tmp_path) -> list:
    """Run ``job`` on tp x dp gloo ranks; return each rank's result."""
    world = tp * dp
    torch.multiprocessing.spawn(
        _rank,
        args=(world, tp, dp, _free_port(), job, payload, str(tmp_path)),
        nprocs=world,
    )
    return [torch.load(tmp_path / f"{job}-{r}.pt") for r in range(world)]


def _form(kwargs) -> tuple[int, bool, int]:
    """A tower call's query rows, ``gather`` flag and ``slice_rows``."""
    return (
        kwargs["pixel_values"].shape[0],
        kwargs.get("gather", False),
        kwargs.get("slice_rows", 0),
    )


def _cache_rows(tp, dp, payload):
    """Encode the payload's images in one hook call; return the tower calls' forms and each image's rows."""
    model = build_model(hf_config(), encoder_config(tp, dp))
    forms = []
    model.visual.register_forward_pre_hook(
        lambda module, args, kwargs: forms.append(_form(kwargs)), with_kwargs=True
    )
    names = list(payload)
    cache = new_cache(len(names) + 2, BUCKET // 4)
    model.embed_multimodal(
        pixel_values=torch.cat([payload[n]["pixel_values"] for n in names]),
        grid_thws=torch.cat([payload[n]["grid_thws"] for n in names]),
        encoder_cache=cache,
        mm_hashes=names,
    )
    rows = {
        n: cache.buffer[cache.get_block_ids(n)].reshape(-1, TEXT)[:IMAGE_ROWS].clone()
        for n in names
    }
    return {"forms": forms, "rows": rows}


def _grid(image) -> tuple[int, int]:
    return tuple(image["grid_thws"][0, 1:].tolist())


def _graded(images, refs, ranks, form) -> list[str]:
    """Print each rank's readings; return the ranks outside class Bs at f or off the form."""
    misses = []
    for rank, result in enumerate(ranks):
        for name in images:
            reading, cosine = bound_reading(result["rows"][name], refs[name])
            print(f"TP{TP} x DP{DP} rank {rank} image {name}: tower calls {result['forms']}; "
                  f"worst error / class-B bound {reading:.3f}, f {TOWER_F}; "
                  f"lowest row cosine {cosine:.7f}")
            if not (reading <= TOWER_F and cosine >= 0.9999):
                misses.append(f"rank {rank} image {name}")
        if result["forms"] != [form]:
            misses.append(f"rank {rank} calls {result['forms']}")
    return misses


def test_split_image_matches_the_float64_reference(processed, tmp_path):
    """Item 1: one image split over 2 groups at TP2 matches the reference in class Bs at f."""
    image = processed["a"]
    refs = {"a": reference(image["pixel_values"], _grid(image), reference_weights())}
    ranks = run_ranks(TP, DP, "_cache_rows", {"a": image}, tmp_path)
    misses = _graded(["a"], refs, ranks, TEST_SPLIT)
    assert not misses, f"outside class Bs at f {TOWER_F} or not split: {misses}"


def test_two_images_take_the_per_group_form(processed, tmp_path):
    """Item 2: two images in one call take the per-group form and both match the reference."""
    payload = {n: processed[n] for n in ("a", "b")}
    weights = reference_weights()
    refs = {n: reference(p["pixel_values"], _grid(p), weights) for n, p in payload.items()}
    ranks = run_ranks(TP, DP, "_cache_rows", payload, tmp_path)
    misses = _graded(list(payload), refs, ranks, (BUCKET, True, 0))
    assert not misses, f"outside class Bs at f {TOWER_F} or not per group: {misses}"


# ---------------------------------------------------------------- served slices


class _Recorder(nn.Module):
    """Stands in for the compiled tower: records each call's form, its inputs' shapes and dtypes."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.pixels = None

    def forward(self, **kwargs):
        shapes = {k: (tuple(v.shape), v.dtype) for k, v in kwargs.items()
                  if isinstance(v, torch.Tensor)}
        self.calls.append((_form(kwargs), shapes))
        self.pixels = kwargs["pixel_values"].clone()
        return kwargs["encoder_cache_buffer"]


def _device_input(tp, dp, payload):
    """Run the hook on the served image at the served buckets; return the tower call and its pixels."""
    served = encoder_config(tp, dp, SERVED_BUCKETS, SERVED_BLOCK)
    model = build_model(hf_config(27), served, load=False)
    recorder = _Recorder()
    model.visual = recorder
    cache = new_cache(2, SERVED_BLOCK // 4)
    model.embed_multimodal(encoder_cache=cache, mm_hashes=["served"], **payload)
    return {"calls": recorder.calls, "pixels": recorder.pixels}


class _Layer(nn.Module):
    """A tower layer that gives its input back as the MLP output and the residual sum."""

    def forward(self, x, *args):
        return x, x


def test_served_bucket_splits_into_8_slices(tmp_path, monkeypatch):
    """Item 3: 8 slices of 688 rows at the served bucket, each padded to 768, each worker's pixels its slice's."""
    from vllm_neuron.model.kimi_k3 import vision_encoder as ve
    from vllm_neuron.model.kimi_k3 import vision_inputs as vi
    from vllm_neuron.model.kimi_k3.config import KimiK3VisionConfig

    bucket = SERVED_BUCKETS[0]
    grids = (SERVED_GRID,)
    cos, sin = vi.rope_tables(grids, bucket)
    unsplit = (*ve.corner_weights(grids, bucket), *vi.key_bounds(grids, bucket))
    cache = vi.TableCache()
    report = {"slice rows": [], "windows in one slice": None, "pad rows": [],
              "key bounds": [], "rows equal": [], "pad rows empty": []}
    windows = []
    for g in range(GROUPS):
        part = vi.block_slice(bucket, GROUPS, g, ve.ATTN_SEQ_ALIGN)
        tables = vi.grid_tables(cache, grids, bucket, part)
        inputs = ve.grid_inputs(grids, bucket, torch.device("cpu"), part)
        lo, hi = SLICE_ROWS * g, SLICE_ROWS * (g + 1)
        # Corners are [4, rows]; the tables and bounds are [rows, ...].
        lengths = {tables.cos.shape[0], tables.sin.shape[0], inputs[0].shape[1],
                   inputs[1].shape[1], inputs[2].shape[0], inputs[3].shape[0]}
        report["slice rows"].append((part.rows, *sorted(lengths)))
        row, col = tables.index // SERVED_GRID[1], tables.index % SERVED_GRID[1]
        windows.append(set(zip((row // 2).tolist(), (col // 2).tolist())))
        bound_min, bound_max = inputs[2][:SLICE_ROWS], inputs[3][:SLICE_ROWS]
        real = bound_max > bound_min
        report["pad rows"].append(int((~real).sum()))
        report["key bounds"].append(sorted({(int(a), int(b)) for a, b in
                                            zip(bound_min[real], bound_max[real])}))
        report["rows equal"].append(
            torch.equal(tables.cos[:SLICE_ROWS], cos[lo:hi])
            and torch.equal(tables.sin[:SLICE_ROWS], sin[lo:hi])
            and all(torch.equal(t[:, :SLICE_ROWS], u[:, lo:hi]) for t, u in zip(inputs[:2], unsplit[:2]))
            and all(torch.equal(t[:SLICE_ROWS], u[lo:hi]) for t, u in zip(inputs[2:], unsplit[2:]))
        )
        report["pad rows empty"].append(
            all(not t[SLICE_ROWS:].any() for t in (tables.cos, tables.sin, inputs[2], inputs[3]))
            and all(not t[:, SLICE_ROWS:].any() for t in inputs[:2])
        )
    whole = SERVED_GRID[0] // 2 * SERVED_GRID[1] // 2
    report["windows in one slice"] = (sum(map(len, windows)) == len(set().union(*windows)) == whole)

    # The tower cuts its output to the slice's rows before the final norm.
    norm_rows = []
    norm = ve.rmsnorm_residual_prefill

    def recorded(x, *args, **kwargs):
        norm_rows.append(x.shape[0])
        return norm(x, *args, **kwargs)

    monkeypatch.setattr(ve, "rmsnorm_residual_prefill", recorded)
    with torch.device("meta"):
        encoder = ve.KimiK3VisionEncoder(KimiK3VisionConfig(), 4, torch.bfloat16)
    encoder.blocks = nn.ModuleList([_Layer()])
    encoder.final_layernorm.weight = nn.Parameter(torch.ones(1024, dtype=torch.bfloat16),
                                                  requires_grad=False)
    out = encoder(torch.ones(SLICE_PADDED, 1024, dtype=torch.bfloat16),
                  None, None, None, None, None, None, SLICE_ROWS)
    report["final norm rows"] = (norm_rows, out.shape[0])

    # Each worker's device input: its slice's pixels, as the uint8 processor gives them.
    pixels = torch.randint(0, 256, (KEY_BOUND, 3, 14, 14), dtype=torch.uint8,
                           generator=torch.Generator().manual_seed(17))
    payload = {"pixel_values": pixels, "grid_thws": torch.tensor([[1, *SERVED_GRID]])}
    block = vi.order_patches(pixels.reshape(KEY_BOUND, -1), grids, bucket)
    ranks = run_ranks(1, GROUPS, "_device_input", payload, tmp_path)
    report["worker inputs"] = []
    for g, result in enumerate(ranks):
        (form, shapes), = result["calls"]
        expected = torch.zeros(SLICE_PADDED, PATCH_DIM, dtype=torch.uint8)
        expected[:SLICE_ROWS] = block[SLICE_ROWS * g : SLICE_ROWS * (g + 1)]
        report["worker inputs"].append(
            (form, shapes["pixel_values"], torch.equal(result["pixels"], expected)))
    for key, value in report.items():
        print(f"{key}: {value}")
    assert report == {
        "slice rows": [(SLICE_ROWS, SLICE_PADDED)] * GROUPS,
        "windows in one slice": True,
        "pad rows": [0] * (GROUPS - 1) + [bucket - KEY_BOUND],
        "key bounds": [[(0, KEY_BOUND)]] * GROUPS,
        "rows equal": [True] * GROUPS,
        "pad rows empty": [True] * GROUPS,
        "final norm rows": ([SLICE_ROWS], SLICE_ROWS),
        "worker inputs": [((SLICE_PADDED, False, SLICE_ROWS),
                           ((SLICE_PADDED, PATCH_DIM), torch.uint8), True)] * GROUPS,
    }


# ---------------------------------------------------------------- warmup


def test_warmup_builds_every_form_the_host_can_pick(processed):
    """Item 4: warmup calls every hand-off form the hook picks at each bucket, split on and off."""
    served = encoder_config(4, 8, SERVED_BUCKETS, SERVED_BLOCK)
    model = build_model(hf_config(27), served, load=False)
    recorder = _Recorder()
    model.visual = recorder
    cache = new_cache(16, SERVED_BLOCK // 4)
    found = {}
    for split in (True, False):
        model.split_lone_image = split
        recorder.calls.clear()
        for names in (["a"], ["a", "b"], ["large"]):
            model.embed_multimodal(
                pixel_values=torch.cat([processed[n]["pixel_values"] for n in names]),
                grid_thws=torch.cat([processed[n]["grid_thws"] for n in names]),
                encoder_cache=cache,
                mm_hashes=names,
            )
            for name in names:
                cache.free(name)
        picked = dict(recorder.calls)
        recorder.calls.clear()
        for bucket in SERVED_BUCKETS:
            inputs = model.build_encoder_synthetic_inputs(bucket, served, torch.device("cpu"))
            # The worker adds the cache and the write ids, one per encoder block, padded to DP.
            blocks = math.ceil(math.ceil(bucket / SERVED_BLOCK) / served.dp_size) * served.dp_size
            inputs["encoder_cache_buffer"] = cache.buffer
            inputs["write_block_ids"] = torch.zeros(blocks, dtype=torch.int64)
            model.get_encoder()(**inputs)
        warmed = dict(recorder.calls)
        print(f"split {split}: picked (rows, gather, slice rows) {sorted(picked)}; "
              f"warmed {sorted(warmed)}; same shapes and dtypes {picked == warmed}")
        found[split] = (sorted(picked), picked == warmed)
    assert found == {
        True: ([(768, False, 688), (5504, True, 0), (33536, False, 0)], True),
        False: ([(5504, False, 0), (5504, True, 0), (33536, False, 0)], True),
    }
