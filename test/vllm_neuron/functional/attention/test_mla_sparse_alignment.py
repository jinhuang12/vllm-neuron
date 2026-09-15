# SPDX-License-Identifier: Apache-2.0
"""Every DMA transpose in the sparse latent attention kernel lands aligned, 16 source rows at a time.

The kernel transposes its operands in the dtype the seam hands it -- the model's
2-byte floats as stored -- sixteen source rows per DMA into a staging tile, then widens
to float32 once per tile. A destination slice therefore starts every 16 columns, 32
bytes apart for a 2-byte source and 64 for float32. Four things are read here: that no
destination in the module can start off 32 bytes at either operand width, from the
module's own offset arithmetic; that every transpose lands in the source dtype and moves
at most 16 source rows; that every SBUF tile the module declares has a per-partition row
of whole 32-byte lines, so the allocator packing tiles back to back leaves every tile
base on a line; and that the values did not move, against a frozen copy of the kernel as
it stood before, exactly rather than within a tolerance -- with float32 operands through
the entry point, and with bfloat16 operands through the seam.

Run under ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1
NEURON_PLATFORM_TARGET_OVERRIDE=trn2``; nothing here reads or sets an environment
variable.
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


def _assert_bit_identical_two_byte(seq: int, latent: int, topk: int, s_kv: int,
                                   heads: int) -> None:
    """Through the seam with bfloat16 operands, the values equal the frozen float32 kernel's."""
    q_lift, c_kv, topk_idx = _inputs(seq, heads, latent, topk, s_kv)
    q_2b, c_2b = q_lift.to(torch.bfloat16), c_kv.to(torch.bfloat16)
    operand = getattr(mod, "_kernel_operand", None)
    if operand is None:
        operand = lambda t: t.contiguous().to(torch.float32)  # noqa: E731 -- the seam before
    handed = (str(operand(q_2b).dtype), str(operand(c_2b).dtype))
    got = mod.mla_sparse_attention(q_2b, c_2b, topk_idx, _SCALE)
    want = wrap_nki(_landed_kernel)(q_2b.to(torch.float32), c_2b.to(torch.float32),
                                    topk_idx, _SCALE)
    differing = int(torch.ne(got, want).sum().item())
    maxabs = float((got - want).abs().max().item()) if differing else 0.0
    _emit("BIT_IDENTITY_2B", seq=seq, heads=heads, latent=latent, topk=topk,
          differing=differing, equal=torch.equal(got, want), maxabs=maxabs,
          handed=handed, shape=tuple(got.shape))
    assert handed == ("torch.bfloat16", "torch.bfloat16"), (
        f"the seam hands the kernel {handed} operands, not the 2-byte ones stored")
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


def _align(width: int) -> int:
    """``width`` rounded up to a whole 32-byte block. The CRITERION, retyped."""
    return ((width + _ALIGN - 1) // _ALIGN) * _ALIGN


#: The trace-time values the transpose sites are read at: one head, this checkpoint's
#: latent rank, the prefill selected-row count and sequence.
_PREFILL = {
    "LATENT_TILE": 128, "KEY_CHUNK": 128, "MOVING_MAX": 512, "heads": 1, "latent": 512,
    "n_latent": 4, "topk": 2048, "tile_max": 512, "chunk_max": 4, "s_kv": 4096, "rope": 0,
    "seq": 2048,
}

#: Decode traces one query; the rope geometry traces the rotary sites at their served width.
_GEOMETRIES_READ = (("prefill", _PREFILL), ("decode", {**_PREFILL, "seq": 1}),
                    ("rope", {**_PREFILL, "rope": 64}))

#: What each geometry legitimately leaves to the host, by the source tensor a site reads and
#: how many bodies read it: decode moves one query row per block, so each body's Q transpose
#: misses the sixteen-row shape; the rotary width is 64, half a tile, in each body.
_DECLARED_HOST_EXPANDED = {
    "prefill": {},
    "decode": {"q_lift_hbm": 3},
    "rope": {"k_pe_hbm": 3, "q_pe_hbm": 3},
}

#: The names the module supplies to its OWN size and loop expressions. They are read off
#: the module and never retyped, so an expression is evaluated with the arithmetic the
#: kernel will trace and not with a copy of it. Absent names are left out.
_MODULE_ARITHMETIC = ("_aligned", "DMA_TRANSPOSE_ALIGN", "STAGE_ALIGN", "DGE_TRANSPOSE_ROWS",
                      "_queries_per_block", "_score_tiles", "_latent_tiles", "_output_tiles")

#: The element widths a staged destination can have: the seam hands 2-byte floats through
#: and widens anything else to float32.
_WIDTHS = (2, 4)

#: The source rows per DMA the device generates its own descriptors for, RETYPED: the
#: criterion the module's rows are graded against, never read off the module, so a module
#: that stepped 32 rows is read as stepping 32 and reddens the census.
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
    """At each geometry, one head and either operand width, every destination is read and none starts off 32 bytes."""
    source = _module_source()
    for name, at in _GEOMETRIES_READ:
        sites = _sites(source, at)
        unreadable = tuple(s.line for s in sites if s.misaligned is None)
        misaligned = tuple(s.line for s in sites if s.misaligned)
        _emit("MISALIGNED_SITES", geometry=name, count=len(misaligned), unreadable=unreadable,
              at_heads=1, widths=_WIDTHS, transposes=len(sites), lines=misaligned)
        assert sites, f"the census read no transpose site at {name}"
        assert unreadable == (), f"destinations the arithmetic cannot place at {name}: {unreadable}"
        assert misaligned == (), f"destinations off a 32-byte line at {name}: {misaligned}"


def test_every_transpose_lands_in_the_source_dtype_sixteen_rows_per_dma():
    """No destination is float32-only; at each geometry the host-shaped sites are the declared ones."""
    source = _module_source()
    found = {}
    float32_only: tuple[int, ...] = ()
    for name, at in _GEOMETRIES_READ:
        sites = _sites(source, at)
        shaped = [s for s in sites if s.host_shaped(_DGE_ROWS)]
        found[name] = dict(sorted(Counter(s.source for s in shaped).items()))
        if name == "prefill":
            float32_only = tuple(s.line for s in sites if s.float32_only)
        _emit("HOST_SHAPED_SITES", geometry=name, count=len(shaped), step=_DGE_ROWS,
              by_source=found[name], transposes=len(sites), lines=tuple(s.line for s in shaped),
              readings=[(s.line, s.source, s.rows, s.width, s.via) for s in sites])
        assert sites, f"the census read no transpose site at {name}"
    _emit("FLOAT32_DESTINATIONS", count=len(float32_only), lines=float32_only)
    assert float32_only == (), f"destinations only float32 can land in: {float32_only}"
    assert found == _DECLARED_HOST_EXPANDED, \
        f"host-shaped sites by source {found} are not the declared {_DECLARED_HOST_EXPANDED}"


def test_every_sbuf_tile_row_is_a_whole_number_of_32_byte_lines():
    """At each geometry every declared SBUF tile is sized and its row is whole 32-byte lines.

    The allocator packs tiles back to back per partition, so one tile whose row is not a whole
    number of lines moves every later tile's base off the line; this reads the tiles the
    transposes above land beside, through the helpers that make them and per caller.
    """
    source = _module_source()
    for name, at in _GEOMETRIES_READ:
        rows = _tile_rows(source, at)
        unreadable = tuple(sorted({(line, tile) for line, tile, size in rows if size is None}))
        narrow = tuple((line, tile, size) for line, tile, size in rows
                       if size is not None and size % LINE)
        _emit("NARROW_TILES", geometry=name, count=len(narrow), unreadable=unreadable,
              tiles=len(rows), widths=_WIDTHS, lines=tuple(r[0] for r in narrow))
        assert rows, f"the census read no SBUF tile at {name}"
        assert unreadable == (), f"tile rows the arithmetic cannot size: {unreadable}"
        assert narrow == (), f"tile rows that are not whole 32-byte lines: {narrow}"


#: The staged form the controls below call, stepping the device's 16 rows as a literal so a
#: control reads the same in a tree whose module has no step constant.
_HELPER = """
def _transpose_rows(dst, src_hbm, row_stride, rows, width, offset):
    for r0 in range(0, rows, 16):
        n = min(16, rows - r0)
        nisa.dma_transpose(dst=dst[:, r0:r0 + n],
                           src=src_hbm.ap(pattern=[[row_stride, n], [1, width]], offset=offset))
"""

#: A caller whose per-query slice starts three elements in: 12 bytes at float32. Its block
#: is a literal so the control reads the same in a tree that has no block arithmetic.
_PLANTED_OFF_LINE = _HELPER + """
def body(q_lift_hbm, heads, latent, n_latent):
    block = 16 * heads
    q_lift_t = _sbuf(LATENT_TILE, n_latent, block + 8)
    for li in range(n_latent):
        _transpose_rows(q_lift_t[:, li, 3:3 + block], q_lift_hbm, latent, block, LATENT_TILE, 0)
"""

#: A caller moving three source rows, which the device cannot shape.
_PLANTED_THREE_ROWS = _HELPER + """
def body(c_kv_hbm, s_kv):
    c_stage = _stage(c_kv_hbm, LATENT_TILE, s_kv)
    _transpose_rows(c_stage, c_kv_hbm, latent, 3, LATENT_TILE, 0)
"""

#: A caller of a helper that steps 32 rows per DMA, twice the device's shape, the module's own
#: constant notwithstanding: the reader must grade rows against the retyped criterion, not the
#: module's.
_PLANTED_WIDE_STEP = """
def _transpose_rows(dst, src_hbm, row_stride, rows, width, offset):
    for r0 in range(0, rows, 32):
        n = min(32, rows - r0)
        nisa.dma_transpose(dst=dst[:, r0:r0 + n],
                           src=src_hbm.ap(pattern=[[row_stride, n], [1, width]], offset=offset))

def body(c_kv_hbm, s_kv):
    c_stage = _stage(c_kv_hbm, LATENT_TILE, s_kv)
    _transpose_rows(c_stage, c_kv_hbm, latent, LATENT_TILE, LATENT_TILE, 0)
"""

#: A caller whose destination the body never declared, which no arithmetic can place.
_PLANTED_UNREADABLE = _HELPER + """
def body(c_kv_hbm, s_kv):
    _transpose_rows(somewhere, c_kv_hbm, latent, s_kv, LATENT_TILE, 0)
"""


def test_control_the_reader_finds_a_planted_off_line_caller():
    """The same reader names the one caller whose destination starts 12 bytes into a line."""
    sites = _sites(_PLANTED_OFF_LINE, _PREFILL)
    misaligned = tuple(s.line for s in sites if s.misaligned)
    _emit("CONTROL_READER_FIRES", count=len(misaligned), offsets=[s.offsets for s in sites],
          lines=misaligned)
    assert len(sites) == 1 and len(misaligned) == 1


#: Two callers: one transposes into a float32-made tile, the other into a tile of the source's
#: own dtype through the staging helper. Only the first can take a 4-byte element alone.
_PLANTED_FLOAT32 = _HELPER + """
def body(q_lift_hbm, c_kv_hbm, heads, latent, s_kv):
    block = 16 * heads
    q_lift_t = _sbuf(LATENT_TILE, block)
    _transpose_rows(q_lift_t, q_lift_hbm, latent, block, LATENT_TILE, 0)
    c_stage = _stage(c_kv_hbm, LATENT_TILE, s_kv)
    _transpose_rows(c_stage, c_kv_hbm, latent, LATENT_TILE, LATENT_TILE, 0)
"""

#: Two whole-tile transposes: one stored three elements into a declared tile, one stored nowhere
#: the reader can place. Neither may read as on the line by assumption.
_PLANTED_WHOLE_TILE = """
def body(staged, hidden, other):
    tile = nl.ndarray((128, 136), dtype=hidden.dtype, buffer=nl.sbuf)
    tile[:, 3:131] = nl.load_transpose2d(staged[0:128, 0:128])
    total = nl.add(nl.load_transpose2d(staged[0:128, 128:256]), other)
"""


def test_control_the_reader_finds_a_planted_float32_only_destination():
    """The same reader names the float32-made destination and not the one in the source's dtype."""
    sites = _sites(_PLANTED_FLOAT32, _PREFILL)
    float32_only = tuple(s.line for s in sites if s.float32_only)
    _emit("CONTROL_FLOAT32_READER_FIRES", count=len(float32_only), sizes=[s.sizes for s in sites],
          lines=float32_only)
    assert len(sites) == 2 and len(float32_only) == 1 and [s.sizes for s in sites] == [(4,), (2, 4)]


def test_control_a_whole_tile_transpose_is_read_where_its_result_lands():
    """A whole-tile transpose stored off the line is misaligned; one stored nowhere readable is unreadable."""
    sites = _sites(_PLANTED_WHOLE_TILE, _PREFILL)
    misaligned = tuple(s.line for s in sites if s.misaligned)
    unreadable = tuple(s.line for s in sites if s.misaligned is None)
    _emit("CONTROL_WHOLE_TILE_READER_FIRES", count=len(misaligned), unreadable=unreadable,
          offsets=[s.offsets for s in sites], lines=misaligned)
    assert len(sites) == 2 and len(misaligned) == 1 and len(unreadable) == 1


def test_control_the_reader_finds_a_planted_three_row_caller():
    """The same reader names the one caller that moves three rows per DMA."""
    sites = _sites(_PLANTED_THREE_ROWS, _PREFILL)
    shaped = tuple(s.line for s in sites if s.host_shaped(_DGE_ROWS))
    _emit("CONTROL_STEP_READER_FIRES", count=len(shaped), rows=[s.rows for s in sites],
          lines=shaped)
    assert len(sites) == 1 and len(shaped) == 1


#: Three tiles: a 4-byte row, an 8-byte row made through a helper, and one whole line.
_PLANTED_NARROW = """
def _pair(parts):
    return _sbuf(parts, 2)

def body(q_lift_hbm, heads):
    row_sum = _sbuf(heads, 1)
    both = _pair(heads)
    line = _sbuf(heads, 8)
"""

#: One tile sized by a name the arithmetic never has.
_PLANTED_UNSIZED = """
def body(q_lift_hbm, heads):
    scratch = _sbuf(heads, somewhere)
"""

#: One tile made by an allocator this census cannot shape, beside one it can.
_PLANTED_UNKNOWN_MAKER = """
def body(q_lift_hbm, heads):
    line = _sbuf(heads, 8)
    scratch = nl.zeros((heads, 8), dtype=nl.float32, buffer=nl.sbuf)
"""


def test_control_the_tile_census_finds_a_planted_narrow_row():
    """The same reader names the 4-byte tile and the 8-byte one made through a helper, not the line."""
    rows = _tile_rows(_PLANTED_NARROW, _PREFILL)
    narrow = tuple((line, tile, size) for line, tile, size in rows
                   if size is not None and size % LINE)
    _emit("CONTROL_TILE_READER_FIRES", count=len(narrow), rows=narrow, tiles=len(rows))
    assert len(rows) == 3 and narrow == ((6, "row_sum", 4), (7, "both", 8))


def test_control_an_unsized_tile_is_never_a_pass():
    """A tile the arithmetic cannot size reads ``None``, which the census refuses."""
    rows = _tile_rows(_PLANTED_UNSIZED, _PREFILL)
    _emit("CONTROL_UNSIZED_TILE_FIRES", count=len(rows), rows=rows)
    assert rows == [(3, "scratch", None)]


def test_control_an_unrecognised_tile_maker_is_never_a_pass():
    """A tile made by an allocator the census cannot shape reads ``None`` beside the line it can size."""
    rows = _tile_rows(_PLANTED_UNKNOWN_MAKER, _PREFILL)
    _emit("CONTROL_UNKNOWN_MAKER_FIRES", count=sum(1 for r in rows if r[2] is None), rows=rows)
    assert rows == [(3, "line", 32), (4, "scratch", None)]


def test_control_the_reader_finds_a_planted_caller_of_a_wide_stepping_helper():
    """The same reader names the one caller whose helper moves 32 rows per DMA."""
    sites = _sites(_PLANTED_WIDE_STEP, _PREFILL)
    shaped = tuple(s.line for s in sites if s.host_shaped(_DGE_ROWS))
    _emit("CONTROL_WIDE_STEP_READER_FIRES", count=len(shaped), rows=[s.rows for s in sites],
          lines=shaped)
    assert len(sites) == 1 and len(shaped) == 1 and sites[0].rows == (32,)


def test_control_an_unreadable_destination_is_never_a_pass():
    """A destination the arithmetic cannot place reads as unreadable, which the census refuses."""
    sites = _sites(_PLANTED_UNREADABLE, _PREFILL)
    unreadable = tuple(s.line for s in sites if s.misaligned is None)
    _emit("CONTROL_UNREADABLE_FIRES", count=len(unreadable), lines=unreadable)
    assert len(sites) == 1 and len(unreadable) == 1
