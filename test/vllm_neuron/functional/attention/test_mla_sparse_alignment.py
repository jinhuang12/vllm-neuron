# SPDX-License-Identifier: Apache-2.0
"""Every DMA transpose in the sparse latent attention kernel lands aligned, 16 source rows at a time.

The kernel transposes its operands in the dtype the seam hands it -- the model's
2-byte floats as stored -- sixteen source rows per DMA into a staging tile, then widens
to float32 once per tile. A destination slice therefore starts every 16 columns, 32
bytes apart for a 2-byte source and 64 for float32. Three things are read here: that no
destination in the module can start off 32 bytes at either operand width, from the
module's own offset arithmetic; that every transpose lands in the source dtype and moves
at most 16 source rows; and that the values did not move, against a frozen copy of the
kernel as it stood before, exactly rather than within a tolerance -- with float32
operands through the entry point, and with bfloat16 operands through the seam.

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


#: The trace-time values the size expressions are read at: ONE head, this checkpoint's
#: latent rank, the prefill selected-row count and sequence.
_AT_ONE_HEAD = {
    "__builtins__": {"min": min, "max": max, "len": len, "range": range},
    "LATENT_TILE": 128, "KEY_CHUNK": 128, "MOVING_MAX": 512, "heads": 1,
    "latent": 512, "n_latent": 4, "topk": 2048, "tile_max": 512, "chunk_max": 4,
    "s_kv": 4096, "rope": 0, "seq": 2048,
}

#: The names the module supplies to its OWN size and loop expressions. They are read off
#: the module and never retyped, so an expression is evaluated with the arithmetic the
#: kernel will trace and not with a copy of it. Absent names are left out, so a module
#: without a name simply cannot evaluate an expression that uses it.
_MODULE_ARITHMETIC = ("_aligned", "DMA_TRANSPOSE_ALIGN", "DGE_TRANSPOSE_ROWS",
                      "_queries_per_block", "_score_tiles", "_latent_tiles", "_output_tiles")

#: 32 bytes, the runtime's rule for a transpose destination.
_LINE = 32

#: The element widths a staged destination can have: the seam hands 2-byte floats through
#: and widens anything else to float32.
_WIDTHS = (2, 4)


def _step() -> int:
    """The module's own rows-per-DMA bound, or 16 where the module has none yet."""
    return int(getattr(mod, "DGE_TRANSPOSE_ROWS", 16))


def _eval(node: ast.expr, names: dict) -> int | None:
    """One expression's trace-time value, or ``None`` when it is not readable."""
    try:
        return int(eval(compile(ast.Expression(body=node), "<size>", "eval"), names))
    except Exception:
        return None


def _names(fn: ast.FunctionDef) -> dict:
    """The census values, the module's arithmetic, then the body's own simple assignments."""
    names = dict(_AT_ONE_HEAD)
    for name in _MODULE_ARITHMETIC:
        if hasattr(mod, name):
            names[name] = getattr(mod, name)
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id in names:
            continue
        value = _eval(node.value, names)
        if value is None and isinstance(node.value, ast.Call) \
                and getattr(node.value.func, "id", "") == "min":
            bounds = [v for v in (_eval(a, names) for a in node.value.args) if v is not None]
            value = min(bounds) if bounds else None
        if value is not None:
            names[target.id] = value
    return names


def _loops(fn: ast.FunctionDef, names: dict) -> dict[str, tuple[int, ...]]:
    """Per ``for NAME in range(...)`` target, the values it takes; an unreadable stop keeps eight."""
    found: dict[str, tuple[int, ...]] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        call = node.iter
        if not isinstance(call, ast.Call) or getattr(call.func, "id", "") != "range":
            continue
        args = [_eval(a, names) for a in call.args]
        start, stop, step = 0, None, 1
        if len(args) == 1:
            stop = args[0]
        elif len(args) >= 2:
            start, stop = args[0] or 0, args[1]
            step = args[2] if len(args) == 3 and args[2] else 1
        if stop is None:
            found[node.target.id] = tuple(start + k * step for k in range(8))
        else:
            found[node.target.id] = tuple(range(start, stop, step))[:4096] or (start,)
    return found


def _tiles(fn: ast.FunctionDef, names: dict) -> dict[str, tuple[int, int, tuple[int, ...]]]:
    """Per SBUF tile name: slice count, slice width and the element widths it can have."""
    found: dict[str, tuple[int, int, tuple[int, ...]]] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, call = node.targets[0], node.value
            if isinstance(call, ast.IfExp):
                call = call.body
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)                 and getattr(node.value.func, "attr", "") == "append" and len(node.value.args) == 1:
            target, call = node.value.func.value, node.value.args[0]
        else:
            continue
        if not isinstance(target, ast.Name) or not isinstance(call, ast.Call):
            continue
        maker = getattr(call.func, "id", "")
        if maker == "_sbuf" and len(call.args) == 3:
            tiles, width = _eval(call.args[1], names), _eval(call.args[2], names)
            widths = (4,)
        elif maker == "_sbuf" and len(call.args) == 2:
            tiles, width = 1, _eval(call.args[1], names)
            widths = (4,)
        elif maker == "_stage" and len(call.args) == 3:
            tiles, width = 1, _eval(call.args[2], names)
            widths = _WIDTHS
        else:
            continue
        if tiles is not None and width is not None:
            found[target.id] = (tiles, width, widths)
    return found


def _lower(node: ast.expr | None, names: dict, loops: dict) -> tuple[int, ...]:
    """The values a slice's lower bound takes: a constant, a loop target, or an expression of one."""
    if node is None:
        return (0,)
    loop_names = sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in loops})
    if not loop_names:
        value = _eval(node, names)
        return (value,) if value is not None else ()
    values = []
    for v in loops[loop_names[0]]:
        value = _eval(node, {**names, loop_names[0]: v})
        if value is not None:
            values.append(value)
    return tuple(values)


def _dest_bytes(dst: ast.expr, tiles: dict, names: dict, loops: dict,
                itemsize: int) -> tuple[int, ...] | None:
    """Every byte offset one destination can start at, or ``None`` when unreadable."""
    if not isinstance(dst, ast.Subscript) or not isinstance(dst.slice, ast.Tuple):
        return (0,)
    base = dst.value
    parts = dst.slice.elts
    count, width = 1, 1
    if isinstance(base, ast.Name) and base.id in tiles:
        count, width, _ = tiles[base.id]
    offsets = set()
    if len(parts) == 3:
        index_values = _lower(parts[1], names, loops) if not isinstance(parts[1], ast.Slice) \
            else tuple(range(count))
        lowers = _lower(parts[2].lower, names, loops) if isinstance(parts[2], ast.Slice) else (0,)
        if not index_values or not lowers:
            return None
        for index in index_values:
            for lower in lowers:
                offsets.add((index * width + lower) * itemsize)
    elif len(parts) == 2:
        lowers = _lower(parts[1].lower, names, loops) if isinstance(parts[1], ast.Slice) else (0,)
        if not lowers:
            return None
        for lower in lowers:
            offsets.add(lower * itemsize)
    else:
        return None
    return tuple(sorted(offsets))


def _rows_per_dma(call: ast.Call, names: dict) -> int | None:
    """The source rows one transpose moves: the row count of ``src.ap(pattern=[[stride, ROWS], ...])``."""
    src = next((kw.value for kw in call.keywords if kw.arg == "src"), None)
    if not isinstance(src, ast.Call) or getattr(src.func, "attr", "") != "ap":
        return None
    pattern = next((kw.value for kw in src.keywords if kw.arg == "pattern"), None)
    if not isinstance(pattern, ast.List) or not pattern.elts:
        return None
    first = pattern.elts[0]
    if not isinstance(first, ast.List) or len(first.elts) != 2:
        return None
    return _eval(first.elts[1], names)


def _base(node: ast.expr) -> ast.expr:
    """The name under a chain of subscripts."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node


def _reading(line: int, dst: ast.expr, tiles: dict, names: dict, loops: dict,
             rows: int | None, via: str = "") -> dict:
    """One destination's reading: the byte offsets it can start at, per element width."""
    base = _base(dst)
    widths = tiles[base.id][2] if isinstance(base, ast.Name) and base.id in tiles else _WIDTHS
    offsets = {w: _dest_bytes(dst, tiles, names, loops, w) for w in widths}
    misaligned = any(o is None or any(b % _LINE for b in o) for o in offsets.values())
    return {"line": line, "widths": widths, "offsets": offsets, "misaligned": misaligned,
            "rows": rows, "float32_destination": widths == (4,), "via": via}


def _transpose_sites(source: str) -> list[dict]:
    """One reading per ``dma_transpose`` in the module, and one per call of a helper that
    transposes into a parameter -- the caller's tile is what that destination really is."""
    functions = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)]
    sites, helpers = [], {}
    for fn in functions:
        names = _names(fn)
        tiles, loops = _tiles(fn, names), _loops(fn, names)
        params = [a.arg for a in fn.args.args]
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call) or getattr(node.func, "attr", "") != "dma_transpose":
                continue
            dst = next((kw.value for kw in node.keywords if kw.arg == "dst"), None)
            if dst is None:
                continue
            rows = _rows_per_dma(node, names)
            sites.append(_reading(node.lineno, dst, tiles, names, loops, rows))
            base = _base(dst)
            if isinstance(base, ast.Name) and base.id in params:
                helpers[fn.name] = (params.index(base.id), rows)
    for fn in functions:
        names = _names(fn)
        tiles, loops = _tiles(fn, names), _loops(fn, names)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call) or getattr(node.func, "id", "") not in helpers:
                continue
            index, rows = helpers[node.func.id]
            if index < len(node.args):
                sites.append(_reading(node.lineno, node.args[index], tiles, names, loops, rows,
                                      via=node.func.id))
    return sites


def _misaligned_transpose_sites(source: str) -> tuple[int, ...]:
    """The line of every ``dma_transpose`` a destination offset can start off 32 bytes on."""
    return tuple(sorted(s["line"] for s in _transpose_sites(source) if s["misaligned"]))


def _host_shaped_transpose_sites(source: str) -> tuple[int, ...]:
    """The line of every ``dma_transpose`` that lands in a float32 tile or moves over 16 source rows."""
    step = _step()
    return tuple(sorted(s["line"] for s in _transpose_sites(source)
                        if s["float32_destination"] or s["rows"] is None or s["rows"] > step))


def test_no_transpose_destination_starts_off_a_32_byte_boundary():
    """At one head and either operand width, no destination offset in the module is off 32 bytes."""
    source = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    sites = _misaligned_transpose_sites(source)
    _emit("MISALIGNED_SITES", count=len(sites), lines=sites, at_heads=1, widths=_WIDTHS,
          transposes=len(_transpose_sites(source)))
    assert sites == ()


def test_every_transpose_lands_in_the_source_dtype_sixteen_rows_per_dma():
    """No transpose in the module lands in a float32 tile or moves more than 16 source rows."""
    source = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    sites = _host_shaped_transpose_sites(source)
    readings = [(s["line"], s["rows"], s["float32_destination"], s["via"])
                for s in _transpose_sites(source)]
    _emit("HOST_SHAPED_SITES", count=len(sites), lines=sites, step=_step(), readings=readings)
    assert sites == ()


#: A body whose head axis is NOT padded, so the reader has one misaligned site to find.
_PLANTED = """
def body(q_lift_hbm, heads, n_latent):
    q_lift_t = _sbuf(LATENT_TILE, n_latent, heads)
    for li in range(n_latent):
        nisa.dma_transpose(dst=q_lift_t[:, li, :], src=q_lift_hbm)
"""

#: A staged helper stepping FOUR rows, so a float32 destination starts 16 bytes in.
_PLANTED_STEP = """
def helper(dst, src_hbm, rows, width, offset):
    for r0 in range(0, rows, 4):
        n = min(4, rows - r0)
        nisa.dma_transpose(dst=dst[:, r0:r0 + n],
                           src=src_hbm.ap(pattern=[[width, n], [1, width]], offset=offset))
"""


def test_control_the_reader_finds_a_planted_misaligned_destination():
    """The same reader counts one site in a planted body with an unpadded axis."""
    sites = _misaligned_transpose_sites(_PLANTED)
    _emit("CONTROL_READER_FIRES", count=len(sites), lines=sites)
    assert len(sites) == 1


def test_control_the_reader_finds_a_planted_four_row_step():
    """The same reader counts one site in a planted helper whose slices step 16 bytes at float32."""
    sites = _misaligned_transpose_sites(_PLANTED_STEP)
    shaped = _host_shaped_transpose_sites(_PLANTED_STEP)
    _emit("CONTROL_STEP_READER_FIRES", count=len(sites), lines=sites, host_shaped=shaped)
    assert len(sites) == 1
    assert shaped == ()
