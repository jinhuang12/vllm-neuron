# SPDX-License-Identifier: Apache-2.0
"""Every DMA transpose destination in the sparse latent attention kernel is aligned.

The transposed Q latent is one tile per query and each latent tile writes a slice of
it, so a slice's destination offset is the head count times the tile index. The head
axis is padded to a whole 32 bytes to keep every one of those offsets on a boundary.
Two things are read here: that no destination in the module can start off 32 bytes at
one head, computed from the module's own offset arithmetic; and that the values did
not move, against a frozen copy of the kernel as it stood before the padding, exactly
rather than within a tolerance.

Run under ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1
NEURON_PLATFORM_TARGET_OVERRIDE=trn2``; nothing here reads or sets an environment
variable.
"""

from __future__ import annotations

import ast
import pathlib

import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import mla_sparse as mod

#: The constants the frozen copy below reads, RETYPED on purpose: a frozen reference
#: that imported them would follow the live module if they ever moved.
LATENT_TILE = 128
KEY_CHUNK = 128
MOVING_MAX = 512
SENTINEL_INDEX = -1
_SENTINEL_EXP_FLOOR = 200.0

#: 32 bytes in float32 elements, retyped for the same reason.
_ALIGN = 8

#: ``(seq, latent, topk, s_kv)`` per declared geometry: the prefill bucket first, then
#: two whose last score tile is narrower than the rest.
_GEOMETRIES = (
    (1, 512, 2048, 2560),
    (2, 256, 640, 1024),
    (2, 128, 1152, 1536),
)

#: The head counts read at each geometry. Eight is where the padding is a no-op.
_HEADS = (1, 2, 8)

_SCALE = 0.1


def _emit(tag: str, **values: object) -> None:
    """Print one machine-readable reading line for the transcript's reader."""
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"MSA|{tag}|{body}", flush=True)


def _sbuf(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.float32, buffer=nl.sbuf)


def _sbuf_u32(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.uint32, buffer=nl.sbuf)


def _sbuf_i32(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.int32, buffer=nl.sbuf)


def _psum(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.float32, buffer=nl.psum)


def _sentinel_scratch(parts, width, heads):
    """Frozen at the base commit: _sentinel_scratch."""
    comparand = _sbuf_i32(parts, width)
    nisa.memset(dst=comparand, value=SENTINEL_INDEX)
    return (comparand, _sbuf_i32(parts, width), _sbuf_i32(parts, width),
            _sbuf_i32(parts, width), _sbuf(heads, width), _sbuf(heads, width),
            _sbuf(heads, width), _sbuf(heads, width))


def _mask_sentinel(topk_hbm, offset, width, heads, sentinel_bias, sen, idx_sb):
    """Frozen at the base commit: _mask_sentinel."""
    nisa.tensor_copy(
        dst=sen[1][:, 0:width],
        src=nl.load(
            topk_hbm.ap(pattern=[[0, LATENT_TILE], [1, width]], offset=offset),
            dtype=nl.int32,
        ),
    )
    nisa.tensor_tensor(dst=sen[2][:, 0:width], data1=sen[0][:, 0:width],
                       data2=sen[1][:, 0:width], op=nl.less)
    nisa.tensor_tensor(dst=sen[3][:, 0:width], data1=sen[1][:, 0:width],
                       data2=sen[2][:, 0:width], op=nl.multiply)
    nisa.tensor_copy(dst=idx_sb[:, 0:width], src=sen[3][:, 0:width])
    nisa.tensor_copy(dst=sen[4][:, 0:width], src=sen[2][0:heads, 0:width])
    nisa.tensor_scalar(dst=sen[5][:, 0:width], data=sen[4][:, 0:width],
                       op0=nl.add, operand0=-1.0)
    nisa.tensor_scalar(dst=sen[5][:, 0:width], data=sen[5][:, 0:width],
                       op0=nl.multiply, operand0=sentinel_bias)


def _score_tiles(topk: int) -> tuple[tuple[int, int], ...]:
    """Frozen at the base commit: _score_tiles."""
    tiles = []
    offset = 0
    while offset < topk:
        tiles.append((offset, min(MOVING_MAX, topk - offset)))
        offset += MOVING_MAX
    return tuple(tiles)


def _attention_body_row_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                              q_pe_hbm=None, k_pe_hbm=None):
    """Frozen at the base commit: _attention_body_row_tiled."""
    seq, heads, latent = q_lift_hbm.shape
    s_kv = c_kv_hbm.shape[0]
    topk = topk_hbm.shape[1]
    n_latent = latent // LATENT_TILE
    rope = 0 if q_pe_hbm is None else q_pe_hbm.shape[2]
    tiles = _score_tiles(topk)
    single = len(tiles) == 1
    tile_max = 0
    for tile in tiles:
        if tile[1] > tile_max:
            tile_max = tile[1]
    chunk_max = tile_max // KEY_CHUNK

    c_sb = []
    for _ in range(n_latent):
        c_sb.append(_sbuf(LATENT_TILE, s_kv))
    for li in range(n_latent):
        nisa.dma_transpose(
            dst=c_sb[li],
            src=c_kv_hbm.ap(pattern=[[latent, s_kv], [1, LATENT_TILE]],
                            offset=li * LATENT_TILE),
        )
    k_pe_sb = None
    if rope > 0:
        k_pe_sb = _sbuf(rope, s_kv)
        nisa.dma_transpose(
            dst=k_pe_sb,
            src=k_pe_hbm.ap(pattern=[[rope, s_kv], [1, rope]], offset=0),
        )

    idx_sb = _sbuf_u32(LATENT_TILE, tile_max)
    q_lift_t = _sbuf(LATENT_TILE, n_latent, heads)
    c_g = _sbuf(LATENT_TILE, n_latent, tile_max)
    c_g_t = _sbuf(KEY_CHUNK, chunk_max, latent)
    p_t = _sbuf(KEY_CHUNK, chunk_max, heads)
    p = _sbuf(heads, tile_max)
    neg_row_max = _sbuf(heads, 1)
    exp_bias = _sbuf(heads, 1)
    tile_sum = _sbuf(heads, 1)
    recip = _sbuf(heads, 1)
    out_sb = _sbuf(heads, latent)
    q_pe_t = _sbuf(rope, heads) if rope > 0 else None
    k_pe_g = _sbuf(rope, tile_max) if rope > 0 else None

    run_pos = _sbuf(heads, 1)
    run_sum = _sbuf(heads, 1)
    acc = _sbuf(heads, latent)
    tile_pos = _sbuf(heads, 1)
    new_pos = _sbuf(heads, 1)
    d_acc = _sbuf(heads, 1)
    d_tile = _sbuf(heads, 1)
    c_acc = _sbuf(heads, 1)
    c_tile = _sbuf(heads, 1)
    sum_kept = _sbuf(heads, 1)
    sum_added = _sbuf(heads, 1)
    sum_new = _sbuf(heads, 1)
    acc_kept = _sbuf(heads, latent)
    pv_added = _sbuf(heads, latent)
    acc_new = _sbuf(heads, latent)

    sen = _sentinel_scratch(LATENT_TILE, tile_max, heads)
    valid_f = sen[4]
    mask_bias = sen[5]
    scores_m = sen[6]
    p_m = sen[7]
    sentinel_bias = _SENTINEL_EXP_FLOOR / softmax_scale

    for q_idx in nl.affine_range(seq):
        for li in range(n_latent):
            nisa.dma_transpose(
                dst=q_lift_t[:, li, :],
                src=q_lift_hbm.ap(pattern=[[latent, heads], [1, LATENT_TILE]],
                                  offset=q_idx * heads * latent + li * LATENT_TILE),
            )
        if rope > 0:
            nisa.dma_transpose(
                dst=q_pe_t,
                src=q_pe_hbm.ap(pattern=[[rope, heads], [1, rope]],
                                offset=q_idx * heads * rope),
            )

        for ti in range(len(tiles)):
            ks = tiles[ti][0]
            extent = tiles[ti][1]
            n_chunks = extent // KEY_CHUNK

            _mask_sentinel(topk_hbm, q_idx * topk + ks, extent, heads, sentinel_bias,
                           sen, idx_sb)

            for li in range(n_latent):
                nisa.nc_n_gather(dst=c_g[:, li, 0:extent], data=c_sb[li],
                                 indices=idx_sb[:, 0:extent])

            scores_ps = _psum(heads, extent)
            for li in range(n_latent):
                nisa.nc_matmul(
                    dst=scores_ps,
                    stationary=q_lift_t[:, li, :],
                    moving=c_g[:, li, 0:extent],
                    accumulate=(li > 0),
                )
            if rope > 0:
                nisa.nc_n_gather(dst=k_pe_g[:, 0:extent], data=k_pe_sb,
                                 indices=idx_sb[0:rope, 0:extent])
                nisa.nc_matmul(dst=scores_ps, stationary=q_pe_t,
                               moving=k_pe_g[:, 0:extent], accumulate=True)

            for ck in range(n_chunks):
                cs = ck * KEY_CHUNK
                for li in range(n_latent):
                    c_g_t_ps = _psum(KEY_CHUNK, LATENT_TILE)
                    nisa.nc_transpose(dst=c_g_t_ps,
                                      data=c_g[:, li, cs:cs + KEY_CHUNK])
                    nisa.tensor_copy(
                        dst=c_g_t[:, ck, li * LATENT_TILE:(li + 1) * LATENT_TILE],
                        src=c_g_t_ps,
                    )

            nisa.tensor_tensor(dst=scores_m[:, 0:extent], data1=scores_ps,
                               data2=mask_bias[:, 0:extent], op=nl.add)
            nisa.tensor_reduce(dst=neg_row_max, op=nl.maximum,
                               data=scores_m[:, 0:extent], axis=1, negate=True)
            nisa.tensor_scalar(dst=exp_bias, data=neg_row_max, op0=nl.multiply,
                               operand0=softmax_scale, engine=nisa.engine.vector)
            nisa.activation(dst=p[:, 0:extent], op=nl.exp,
                            data=scores_m[:, 0:extent],
                            bias=exp_bias, scale=softmax_scale, reduce_op=nl.add,
                            reduce_res=tile_sum,
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
            nisa.tensor_tensor(dst=p_m[:, 0:extent], data1=p[:, 0:extent],
                               data2=valid_f[:, 0:extent], op=nl.multiply)

            for ck in range(n_chunks):
                cs = ck * KEY_CHUNK
                p_t_ps = _psum(KEY_CHUNK, heads)
                nisa.nc_transpose(dst=p_t_ps, data=p_m[:, cs:cs + KEY_CHUNK])
                nisa.tensor_copy(dst=p_t[:, ck, :], src=p_t_ps)
            pv_ps = _psum(heads, latent)
            for ck in range(n_chunks):
                nisa.nc_matmul(dst=pv_ps, stationary=p_t[:, ck, :],
                               moving=c_g_t[:, ck, :], accumulate=(ck > 0))

            if single:
                nisa.tensor_copy(dst=acc, src=pv_ps)
                nisa.tensor_copy(dst=run_sum, src=tile_sum)
                continue

            nisa.tensor_scalar(dst=tile_pos, data=exp_bias, op0=nl.multiply,
                               operand0=-1.0, engine=nisa.engine.vector)
            if ti == 0:
                nisa.tensor_copy(dst=run_pos, src=tile_pos)
                nisa.tensor_copy(dst=acc, src=pv_ps)
                nisa.tensor_copy(dst=run_sum, src=tile_sum)
                continue

            nisa.tensor_tensor(dst=new_pos, data1=run_pos, data2=tile_pos,
                               op=nl.maximum)
            nisa.tensor_tensor(dst=d_acc, data1=run_pos, data2=new_pos, op=nl.subtract)
            nisa.activation(dst=c_acc, op=nl.exp, data=d_acc)
            nisa.tensor_tensor(dst=d_tile, data1=tile_pos, data2=new_pos,
                               op=nl.subtract)
            nisa.activation(dst=c_tile, op=nl.exp, data=d_tile)

            nisa.tensor_tensor(dst=sum_kept, data1=run_sum, data2=c_acc,
                               op=nl.multiply)
            nisa.tensor_tensor(dst=sum_added, data1=tile_sum, data2=c_tile,
                               op=nl.multiply)
            nisa.tensor_tensor(dst=sum_new, data1=sum_kept, data2=sum_added,
                               op=nl.add)

            nisa.tensor_scalar(dst=acc_kept, data=acc, op0=nl.multiply, operand0=c_acc,
                               engine=nisa.engine.vector)
            nisa.tensor_scalar(dst=pv_added, data=pv_ps, op0=nl.multiply,
                               operand0=c_tile, engine=nisa.engine.vector)
            nisa.tensor_tensor(dst=acc_new, data1=acc_kept, data2=pv_added, op=nl.add)

            nisa.tensor_copy(dst=run_pos, src=new_pos)
            nisa.tensor_copy(dst=run_sum, src=sum_new)
            nisa.tensor_copy(dst=acc, src=acc_new)

        nisa.reciprocal(dst=recip, data=run_sum)
        nisa.tensor_scalar(dst=out_sb, data=acc, op0=nl.multiply, operand0=recip,
                           engine=nisa.engine.vector)
        nl.store(
            out_hbm.ap(pattern=[[latent, heads], [1, latent]],
                       offset=q_idx * heads * latent),
            value=out_sb,
        )


@nki.jit
def _landed_kernel(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale):
    """Frozen at the base commit: the row-tiled R == 0 entry point."""
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _attention_body_row_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm)
    return out_hbm


def _inputs(seq: int, heads: int, latent: int, topk: int, s_kv: int, seed: int = 11):
    """One case's operands: the last two selected-row columns are the sentinel."""
    gen = torch.Generator().manual_seed(seed)
    q_lift = torch.randn((seq, heads, latent), generator=gen, dtype=torch.float32)
    c_kv = torch.randn((s_kv, latent), generator=gen, dtype=torch.float32)
    topk_idx = torch.randint(0, s_kv, (seq, topk), generator=gen, dtype=torch.int32)
    topk_idx[:, -2:] = SENTINEL_INDEX
    return q_lift, c_kv, topk_idx


def _assert_bit_identical(seq: int, latent: int, topk: int, s_kv: int,
                          heads: int) -> None:
    """The live kernel equals the frozen one exactly; differing elements are read."""
    q_lift, c_kv, topk_idx = _inputs(seq, heads, latent, topk, s_kv)
    live = mod.mla_sparse_attention_nope_row_tiled_kernel
    got = wrap_nki(live)(q_lift, c_kv, topk_idx, _SCALE)
    want = wrap_nki(_landed_kernel)(q_lift, c_kv, topk_idx, _SCALE)
    differing = int(torch.ne(got, want).sum().item())
    maxabs = float((got - want).abs().max().item()) if differing else 0.0
    _emit("BIT_IDENTITY", seq=seq, heads=heads, latent=latent, topk=topk,
          differing=differing, equal=torch.equal(got, want), maxabs=maxabs,
          padded_heads=_align(heads), shape=tuple(got.shape))
    assert differing == 0
    assert torch.equal(got, want)


def test_bit_identical_at_the_prefill_geometry():
    """Latent 512 and 2,048 selected rows: four score tiles, at each head count."""
    for heads in _HEADS:
        _assert_bit_identical(*_GEOMETRIES[0], heads)


def test_bit_identical_at_640_rows_with_a_narrower_last_tile():
    """Two score tiles, the second a quarter as wide, at each head count."""
    for heads in _HEADS:
        _assert_bit_identical(*_GEOMETRIES[1], heads)


def test_bit_identical_at_1152_rows_with_a_narrower_last_tile():
    """Three score tiles over a single latent tile, at each head count."""
    for heads in _HEADS:
        _assert_bit_identical(*_GEOMETRIES[2], heads)


def _align(width: int) -> int:
    """``width`` rounded up to a whole 32-byte block. The CRITERION, retyped."""
    return ((width + _ALIGN - 1) // _ALIGN) * _ALIGN


#: The trace-time values the destination offsets are read at: ONE head, this
#: checkpoint's latent rank, the prefill selected-row count.
_AT_ONE_HEAD = {
    "__builtins__": {}, "LATENT_TILE": 128, "KEY_CHUNK": 128,
    "MOVING_MAX": 512, "heads": 1, "latent": 512, "n_latent": 4, "topk": 2048,
    "tile_max": 512, "chunk_max": 4, "s_kv": 4096, "rope": 0,
}

#: The names the module supplies to its OWN size expressions. They are read off the
#: module and never retyped, so a size expression is evaluated with the arithmetic the
#: kernel will trace and not with a copy of it: a module that rounded to four elements
#: would be read as rounding to four and reddens the census. Absent names are left out
#: rather than defaulted, because the tree before this change calls none of them.
_MODULE_ARITHMETIC = ("_aligned", "DMA_TRANSPOSE_ALIGN")


def _size(node: ast.expr) -> int | None:
    """One size expression's trace-time value, or ``None`` when it is not readable."""
    names = dict(_AT_ONE_HEAD)
    for name in _MODULE_ARITHMETIC:
        if hasattr(mod, name):
            names[name] = getattr(mod, name)
    try:
        code = compile(ast.Expression(body=node), "<size>", "eval")
        return int(eval(code, names))
    except Exception:
        return None


def _sliced_tiles(fn: ast.FunctionDef) -> dict[str, tuple[int, int]]:
    """Per ``_sbuf(parts, tiles, width)`` name, how many slices and how wide each is."""
    found: dict[str, tuple[int, int]] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, call = node.targets[0], node.value
        if not isinstance(target, ast.Name) or not isinstance(call, ast.Call):
            continue
        if getattr(call.func, "id", "") != "_sbuf" or len(call.args) != 3:
            continue
        tiles, width = _size(call.args[1]), _size(call.args[2])
        if tiles is not None and width is not None:
            found[target.id] = (tiles, width)
    return found


def _dest_offsets(dst: ast.expr, tiles: dict[str, tuple[int, int]]) -> tuple[int, ...]:
    """Where one destination can start, in float32 elements from its tile's base."""
    if not isinstance(dst, ast.Subscript) or not isinstance(dst.slice, ast.Tuple):
        return (0,)
    base = dst.value
    if not isinstance(base, ast.Name) or base.id not in tiles:
        return (0,)
    count, width = tiles[base.id]
    return tuple(index * width for index in range(count))


def _misaligned_transpose_sites(source: str) -> tuple[int, ...]:
    """The line of every ``dma_transpose`` a destination offset can start off 32B on."""
    sites = []
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, ast.FunctionDef):
            continue
        tiles = _sliced_tiles(fn)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", "") != "dma_transpose":
                continue
            dst = next((kw.value for kw in node.keywords if kw.arg == "dst"), None)
            if dst is not None and any(off % _ALIGN
                                       for off in _dest_offsets(dst, tiles)):
                sites.append(node.lineno)
    return tuple(sorted(sites))


def test_no_transpose_destination_starts_off_a_32_byte_boundary():
    """At one head, no destination offset in the module is off a 32-byte boundary."""
    source = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    sites = _misaligned_transpose_sites(source)
    _emit("MISALIGNED_SITES", count=len(sites), lines=sites, at_heads=1)
    assert sites == ()


#: A body whose head axis is NOT padded, so the reader has one site to find.
_PLANTED = """
def body(q_lift_hbm, heads, n_latent):
    q_lift_t = _sbuf(LATENT_TILE, n_latent, heads)
    for li in range(n_latent):
        nisa.dma_transpose(dst=q_lift_t[:, li, :], src=q_lift_hbm)
"""


def test_control_the_reader_finds_a_planted_misaligned_destination():
    """The same reader counts one site in a planted body with an unpadded axis."""
    sites = _misaligned_transpose_sites(_PLANTED)
    _emit("CONTROL_READER_FIRES", count=len(sites), lines=sites)
    assert len(sites) == 1
