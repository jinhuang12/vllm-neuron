# SPDX-License-Identifier: Apache-2.0
"""Every DMA transpose in the sparse latent attention kernel lands aligned, 16 source rows at a time.

The kernel transposes its operands in the dtype the seam hands it -- the model's 2-byte
floats as stored -- sixteen source rows per DMA into a staging tile, then widens to
float32 once per tile. A destination slice therefore starts every 16 columns, 32 bytes
apart for a 2-byte source and 64 for float32.

Two kinds of test live here. One reads the module's own offset arithmetic: no transpose
destination starts off a 32-byte line at either operand width, every transpose lands in
the source dtype and moves at most 16 source rows, and every SBUF tile has a
per-partition row of whole 32-byte lines, so the allocator packing tiles back to back
leaves every tile base on a line. The other runs the kernel against a copy of it taken
before this change and requires equal bytes rather than a tolerance, with float32
operands through the entry point and with bfloat16 operands through the seam.
"""

from __future__ import annotations

import pathlib
from collections import Counter

import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import mla_sparse as mod
from test.vllm_neuron.functional.dma_transpose_census import LINE, census, tile_rows

#: The constants the reference copy below reads, written out again on purpose: a
#: reference that imported them would follow the live module if they ever moved.
LATENT_TILE = 128
KEY_CHUNK = 128
MOVING_MAX = 512
SENTINEL_INDEX = -1
_SENTINEL_EXP_FLOOR = 200.0

#: ``(seq, latent, topk, s_kv)`` per geometry: the prefill bucket first, then two whose
#: last score tile is narrower than the rest.
_GEOMETRIES = (
    (1, 512, 2048, 2560),
    (2, 256, 640, 1024),
    (2, 128, 1152, 1536),
)

#: The head counts read at each geometry. Eight is where the padding is a no-op.
_HEADS = (1, 2, 8)

_SCALE = 0.1


# The kernel as it stood before this change, kept whole so the tests below can require
# equal bytes from the live one. It must not import the live module's constants or
# helpers: a reference that did would follow the module it is meant to pin.
def _sbuf(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.float32, buffer=nl.sbuf)


def _sbuf_u32(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.uint32, buffer=nl.sbuf)


def _sbuf_i32(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.int32, buffer=nl.sbuf)


def _psum(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.float32, buffer=nl.psum)


def _sentinel_scratch(parts, width, heads):
    comparand = _sbuf_i32(parts, width)
    nisa.memset(dst=comparand, value=SENTINEL_INDEX)
    return (comparand, _sbuf_i32(parts, width), _sbuf_i32(parts, width),
            _sbuf_i32(parts, width), _sbuf(heads, width), _sbuf(heads, width),
            _sbuf(heads, width), _sbuf(heads, width))


def _mask_sentinel(topk_hbm, offset, width, heads, sentinel_bias, sen, idx_sb):
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
    tiles = []
    offset = 0
    while offset < topk:
        tiles.append((offset, min(MOVING_MAX, topk - offset)))
        offset += MOVING_MAX
    return tuple(tiles)


def _attention_body_row_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                              q_pe_hbm=None, k_pe_hbm=None):
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
def _reference_kernel(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale):
    """The row-tiled entry point at rope width 0."""
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
    """The live kernel returns the reference kernel's bytes exactly."""
    q_lift, c_kv, topk_idx = _inputs(seq, heads, latent, topk, s_kv)
    live = mod.mla_sparse_attention_nope_row_tiled_kernel
    got = wrap_nki(live)(q_lift, c_kv, topk_idx, _SCALE)
    expected = wrap_nki(_reference_kernel)(q_lift, c_kv, topk_idx, _SCALE)
    differing = int(torch.ne(got, expected).sum().item())
    maxabs = float((got - expected).abs().max().item()) if differing else 0.0
    assert torch.equal(got, expected), (
        f"at seq={seq} heads={heads} latent={latent} topk={topk} the live kernel and the "
        f"reference differ in {differing} elements, worst {maxabs:.3e}"
    )


def _assert_bit_identical_two_byte(seq: int, latent: int, topk: int, s_kv: int,
                                   heads: int) -> None:
    """With bfloat16 operands the seam hands them through unwidened and the values still match."""
    q_lift, c_kv, topk_idx = _inputs(seq, heads, latent, topk, s_kv)
    q_2b, c_2b = q_lift.to(torch.bfloat16), c_kv.to(torch.bfloat16)
    # Before this change the seam widened its operands to float32; a module without the
    # hook reads as that older behaviour and fails the dtype assertion below.
    operand = getattr(mod, "_kernel_operand", None)
    if operand is None:
        operand = lambda t: t.contiguous().to(torch.float32)  # noqa: E731
    handed = (str(operand(q_2b).dtype), str(operand(c_2b).dtype))
    got = mod.mla_sparse_attention(q_2b, c_2b, topk_idx, _SCALE)
    expected = wrap_nki(_reference_kernel)(q_2b.to(torch.float32), c_2b.to(torch.float32),
                                           topk_idx, _SCALE)
    differing = int(torch.ne(got, expected).sum().item())
    maxabs = float((got - expected).abs().max().item()) if differing else 0.0
    assert handed == ("torch.bfloat16", "torch.bfloat16"), (
        f"the seam hands the kernel {handed} operands, not the 2-byte ones stored")
    assert torch.equal(got, expected), (
        f"at seq={seq} heads={heads} latent={latent} topk={topk} the bfloat16 call and the "
        f"reference differ in {differing} elements, worst {maxabs:.3e}"
    )


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


def test_two_byte_operands_bit_identical_at_the_prefill_geometry():
    """The seam hands bfloat16 through; latent 512 and 2,048 rows, at each head count."""
    for heads in _HEADS:
        _assert_bit_identical_two_byte(*_GEOMETRIES[0], heads)


def test_two_byte_operands_bit_identical_at_640_rows_with_a_narrower_last_tile():
    """The seam hands bfloat16 through; two score tiles, at each head count."""
    for heads in _HEADS:
        _assert_bit_identical_two_byte(*_GEOMETRIES[1], heads)


def test_two_byte_operands_bit_identical_at_1152_rows_with_a_narrower_last_tile():
    """The seam hands bfloat16 through; three score tiles, at each head count."""
    for heads in _HEADS:
        _assert_bit_identical_two_byte(*_GEOMETRIES[2], heads)


#: The trace-time values the transpose sites are read at: one head, this checkpoint's
#: latent rank, the prefill selected-row count and sequence.
_PREFILL = {
    "LATENT_TILE": 128, "KEY_CHUNK": 128, "MOVING_MAX": 512, "heads": 1, "latent": 512,
    "n_latent": 4, "topk": 2048, "tile_max": 512, "chunk_max": 4, "s_kv": 4096, "rope": 0,
    "seq": 2048,
}

#: Decode traces one query; the rope geometry traces the rotary sites at their served width.
_GEOMETRIES_READ = (("prefill", _PREFILL), ("decode", {**_PREFILL, "seq": 1}),
                    ("rope", {**_PREFILL, "rope": 64}),
                    ("rope_narrow", {**_PREFILL, "rope": 7}))

#: What each geometry legitimately leaves to the host, by the source tensor a site reads and
#: how many bodies read it: decode moves one query row per block, so each body's Q transpose
#: misses the sixteen-row shape; the rotary width is 64, half a tile, in each body.
_EXPECTED_HOST_SHAPED = {
    "prefill": {},
    "decode": {"q_lift_hbm": 3},
    "rope": {"k_pe_hbm": 3, "q_pe_hbm": 3},
    "rope_narrow": {"k_pe_hbm": 3, "q_pe_hbm": 3},
}

#: The names the module supplies to its own size and loop expressions. They are read off
#: the module rather than copied, so an expression is evaluated with the arithmetic the
#: kernel will trace. Absent names are left out.
_MODULE_ARITHMETIC = ("_aligned", "DMA_TRANSPOSE_ALIGN", "STAGE_ALIGN", "DGE_TRANSPOSE_ROWS",
                      "_queries_per_block", "_score_tiles", "_latent_tiles", "_output_tiles")

#: The element widths a staged destination can have: the seam hands 2-byte floats through
#: and widens anything else to float32.
_WIDTHS = (2, 4)

#: The source rows per DMA the device generates its own descriptors for. Written out here
#: rather than read off the module, so a module that stepped 32 rows is graded as stepping
#: 32 instead of grading itself.
_DGE_ROWS = 16


def _arithmetic() -> dict:
    """The module's own arithmetic, by name; a module without a name leaves it out."""
    return {name: getattr(mod, name) for name in _MODULE_ARITHMETIC if hasattr(mod, name)}


def _module_source() -> str:
    return pathlib.Path(mod.__file__).read_text(encoding="utf-8")


def _sites(source: str, at: dict) -> list:
    """Every transpose site of ``source`` at geometry ``at``, read with the module's arithmetic."""
    return census(source, at, _arithmetic(), _WIDTHS)


def _tile_rows(source: str, at: dict) -> list:
    """Every SBUF tile ``source`` declares at ``at``, as ``(line, name, bytes per partition)``."""
    return tile_rows(source, at, _arithmetic(), _WIDTHS)


def test_no_transpose_destination_starts_off_a_32_byte_boundary():
    """At each geometry and either operand width, no transpose destination starts off 32 bytes."""
    source = _module_source()
    for name, at in _GEOMETRIES_READ:
        sites = _sites(source, at)
        unreadable = tuple(s.line for s in sites if s.misaligned is None)
        misaligned = tuple(s.line for s in sites if s.misaligned)
        assert sites, f"the census read no transpose site at {name}"
        assert unreadable == (), f"destinations the arithmetic cannot place at {name}: {unreadable}"
        assert misaligned == (), f"destinations off a 32-byte line at {name}: {misaligned}"


def test_every_transpose_lands_in_the_source_dtype_sixteen_rows_per_dma():
    """No destination is float32-only, and the only host-shaped sites are decode's Q and the rotary ones."""
    source = _module_source()
    found = {}
    float32_only: tuple[int, ...] = ()
    for name, at in _GEOMETRIES_READ:
        sites = _sites(source, at)
        shaped = [s for s in sites if s.host_shaped(_DGE_ROWS)]
        found[name] = dict(sorted(Counter(s.source for s in shaped).items()))
        if name == "prefill":
            float32_only = tuple(s.line for s in sites if s.float32_only)
        assert sites, f"the census read no transpose site at {name}"
    assert float32_only == (), f"destinations only float32 can land in: {float32_only}"
    assert found == _EXPECTED_HOST_SHAPED, \
        f"host-shaped sites by source {found} are not the expected {_EXPECTED_HOST_SHAPED}"


def test_every_sbuf_tile_row_is_a_whole_number_of_32_byte_lines():
    """Every SBUF tile the module declares can be sized, and its row is whole 32-byte lines.

    The allocator packs tiles back to back per partition, so one tile whose row is not a
    whole number of lines moves every later tile's base off the line.
    """
    source = _module_source()
    for name, at in _GEOMETRIES_READ:
        rows = _tile_rows(source, at)
        unreadable = tuple(sorted({(line, tile) for line, tile, size in rows if size is None}))
        narrow = tuple((line, tile, size) for line, tile, size in rows
                       if size is not None and size % LINE)
        assert rows, f"the census read no SBUF tile at {name}"
        assert unreadable == (), f"tile rows the arithmetic cannot size: {unreadable}"
        assert narrow == (), f"tile rows that are not whole 32-byte lines: {narrow}"
