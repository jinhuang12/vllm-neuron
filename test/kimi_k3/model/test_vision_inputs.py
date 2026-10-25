# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 vision input builders against upstream vLLM's Kimi vision tower code."""

import gc
import weakref

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.models.kimi_k25_vit import (
    Rope2DPosEmbRepeated,
    tpool_patch_merger,
)

from vllm_neuron.model.kimi_k3 import vision_inputs as vi

GRID = (74, 74)
OTHER = (72, 76)
BUCKET = 5504
REAL = GRID[0] * GRID[1]
PATCH_DIM = 588


def _randn(*shape: int, seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _reference_order(h: int, w: int) -> torch.Tensor:
    """The raster index of each merger row, read off the reference merger."""
    raster = torch.arange(h * w, dtype=torch.float64).view(-1, 1)
    return tpool_patch_merger(raster, [[1, h, w]])[0].flatten().long()


def _ulps(got: torch.Tensor, want: torch.Tensor) -> float:
    """Largest error in units of the last place of the reference value."""
    spacing = torch.nextafter(want.abs(), torch.tensor(float("inf"))) - want.abs()
    diff = got - want
    assert torch.isfinite(diff).all(), f"{int((~torch.isfinite(diff)).sum())} differences are not finite"
    return (diff.abs() / spacing).max().item()


def _resize_error(wh, ww, table, grid) -> float:
    """Max error of Wh @ table @ Ww.T against the eager reference bilinear resize."""
    want = F.interpolate(
        table.permute(2, 0, 1).unsqueeze(0), size=grid, mode="bilinear"
    )
    want = want.squeeze(0).permute(1, 2, 0)
    got = torch.einsum("ah,hwd,bw->abd", wh, table, ww)
    diff = got - want
    assert torch.isfinite(diff).all(), f"{int((~torch.isfinite(diff)).sum())} differences are not finite"
    return diff.abs().max().item()


def _fp64_weights(n: int) -> torch.Tensor:
    eye = torch.eye(vi.POS_EMB_SIDE, dtype=torch.float64).unsqueeze(0)
    return F.interpolate(eye, size=n, mode="linear", align_corners=False)[0].T.float()


@pytest.fixture
def builds(monkeypatch):
    """Grids of every worker table build, in call order."""
    calls = []
    build = vi.build_grid_tables

    def counted(grids, bucket):
        calls.append(tuple(map(tuple, grids)))
        return build(grids, bucket)

    monkeypatch.setattr(vi, "build_grid_tables", counted)
    return calls


def test_window_order_view_equals_merger():
    patches = _randn(REAL, PATCH_DIM, seed=1)
    rows = vi.order_patches(patches, [GRID], BUCKET)
    merged = tpool_patch_merger(patches, [[1, *GRID]])[0]
    view = rows[:REAL].view(REAL // 4, 4, PATCH_DIM)
    assert (view != merged).flatten(1).any(dim=1).sum().item() == 0


def test_rope_tables_equal_reference_table():
    cos, sin = vi.rope_tables([GRID], BUCKET)
    rope = Rope2DPosEmbRepeated(vi.HEAD_DIM, 512, 512)
    table = rope.get_freqs_cis(torch.tensor([[1, *GRID]]), torch.device("cpu"))
    table = table[_reference_order(*GRID)]
    half = vi.HEAD_DIM // 2
    for cols in (slice(0, half), slice(half, vi.HEAD_DIM)):
        assert _ulps(cos[:REAL, cols], table.real) <= 1
        assert _ulps(sin[:REAL, cols], table.imag) <= 1


def test_pos_embed_weights_match_bilinear_interpolate():
    table = _randn(vi.POS_EMB_SIDE, vi.POS_EMB_SIDE, 1024, seed=3)
    limit = 1e-6 * table.abs().max().item()
    for grid in (GRID, (40, 110)):
        wh, ww = vi.pos_embed_weights(*grid)
        assert wh.dtype == ww.dtype == torch.float32
        assert wh.shape == (grid[0], vi.POS_EMB_SIDE)
        assert ww.shape == (grid[1], vi.POS_EMB_SIDE)
        assert _resize_error(wh, ww, table, grid) <= limit
        fp64 = (_fp64_weights(grid[0]), _fp64_weights(grid[1]))
        assert _resize_error(*fp64, table, grid) > limit
    signs = _randn(vi.POS_EMB_SIDE, vi.POS_EMB_SIDE, 64, seed=33).sign()
    for h in range(2, 513, 2):
        grid = (h, max(2, 516 - h))
        assert _resize_error(*vi.pos_embed_weights(*grid), signs, grid) <= 1e-6
    wh, ww = vi.pos_embed_weights(vi.POS_EMB_SIDE, vi.POS_EMB_SIDE)
    assert torch.equal(wh, torch.eye(vi.POS_EMB_SIDE))
    assert torch.equal(ww, torch.eye(vi.POS_EMB_SIDE))


def test_pad_rows_and_key_bounds():
    rows = vi.order_patches(_randn(REAL, PATCH_DIM, seed=4), [GRID], BUCKET)
    assert rows.shape == (BUCKET, PATCH_DIM)
    assert rows[REAL:].eq(0).all()
    bound_min, bound_max = vi.key_bounds([GRID], BUCKET)
    assert bound_min[:REAL].eq(0).all() and bound_max[:REAL].eq(REAL).all()
    assert bound_min[REAL:].eq(0).all() and bound_max[REAL:].eq(0).all()

    grids = [(40, 60), (50, 56)]
    images = [_randn(h * w, PATCH_DIM, seed=5 + i) for i, (h, w) in enumerate(grids)]
    rows = vi.order_patches(torch.cat(images), grids, BUCKET)
    bound_min, bound_max = vi.key_bounds(grids, BUCKET)
    spans = [(0, 2400), (2400, 5200)]
    used = set(zip(bound_min[:5200, 0].tolist(), bound_max[:5200, 0].tolist()))
    assert used == set(spans)
    for (start, end), image, (h, w) in zip(spans, images, grids):
        assert bound_min[start:end].eq(start).all()
        assert bound_max[start:end].eq(end).all()
        assert torch.equal(rows[start:end], image[_reference_order(h, w)])
    assert bound_min[5200:].eq(0).all() and bound_max[5200:].eq(0).all()
    assert rows[5200:].eq(0).all()


def _own_tables_each_call(get) -> bool:
    """Whether grids called in turn each get their own grid's tables."""
    cache = vi.TableCache()
    for grid in (GRID, OTHER, GRID, OTHER):
        tables = get(cache, [grid], BUCKET)
        cos, sin = vi.rope_tables([grid], BUCKET)
        if not (
            torch.equal(tables.cos, cos)
            and torch.equal(tables.sin, sin)
            and torch.equal(tables.index, _reference_order(*grid))
        ):
            return False
    return True


def _kept_bytes(refs) -> int:
    gc.collect()
    return sum(ref().nbytes for ref in refs if ref() is not None)


def test_worker_tables_cache(builds):
    patches = _randn(REAL, PATCH_DIM, seed=6)
    tables = vi.grid_tables(vi.TableCache(), [GRID], BUCKET)
    cos, sin = vi.rope_tables([GRID], BUCKET)
    assert torch.equal(tables.cos, cos) and torch.equal(tables.sin, sin)
    ordered = vi.order_patches(patches, [GRID], BUCKET)
    assert torch.equal(patches[tables.index], ordered[:REAL])

    assert _own_tables_each_call(vi.grid_tables)

    def by_bucket(cache, grids, bucket):
        return cache.get(bucket, lambda: vi.build_grid_tables(grids, bucket))

    assert not _own_tables_each_call(by_bucket)

    third = (70, 78)
    sizes = {
        g: sum(t.nbytes for t in vi.build_grid_tables([g], BUCKET))
        for g in (GRID, OTHER, third)
    }
    builds.clear()
    one = vi.TableCache(max_bytes=sizes[GRID])
    for grid in (GRID, OTHER, OTHER, GRID):
        vi.grid_tables(one, [grid], BUCKET)
    assert builds == [(GRID,), (OTHER,), (GRID,)]
    builds.clear()
    two = vi.TableCache(max_bytes=2 * max(sizes.values()))
    for grid in (GRID, OTHER, GRID, third, GRID, OTHER):
        vi.grid_tables(two, [grid], BUCKET)
    assert builds == [(GRID,), (OTHER,), (third,), (OTHER,)]

    cache = vi.TableCache()
    assert cache.max_bytes == 64_000_000
    refs, peak = [], 0
    for h in range(74, 46, -2):
        tables = vi.grid_tables(cache, [(h, BUCKET // h // 2 * 2)], BUCKET)
        refs += [weakref.ref(t) for t in tables]
        del tables
        kept = _kept_bytes(refs)
        assert kept <= 64_000_000
        peak = max(peak, kept)
    assert peak > 64_000_000 - max(sizes.values())
    builds.clear()
    big = vi.grid_tables(cache, [(256, 256)], 65536)
    assert torch.equal(big.cos, vi.rope_tables([(256, 256)], 65536)[0])
    big_refs = [weakref.ref(t) for t in big]
    del big
    assert _kept_bytes(big_refs) == 0 and _kept_bytes(refs) == kept
    vi.grid_tables(cache, [(48, 114)], BUCKET)
    assert builds == [((256, 256),)]

    first = vi.grid_tables(cache, [GRID], BUCKET)
    builds.clear()
    again = vi.grid_tables(cache, [GRID], BUCKET)
    assert builds == []
    assert all(a is b and a.data_ptr() == b.data_ptr() for a, b in zip(first, again))
