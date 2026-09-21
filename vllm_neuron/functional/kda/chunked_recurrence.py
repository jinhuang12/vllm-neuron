# SPDX-License-Identifier: Apache-2.0
"""KDA chunked recurrence, intra-chunk half: stages 1 to 3, in NKI.

Kimi Delta Attention's chunked form splits in two. This module is the
intra-chunk half: the gate cumulative sum, the L2-normalisation of ``q`` and
``k``, the two chunk-local products ``A`` and ``Aqk``, the inverse
``(I + A)**-1``, and the WY representation ``w`` / ``u`` together with the gated
key ``kg``. The state carried across chunks and the output belong to the
inter-chunk half. The torch code here is a CPU oracle, never the shipped path.

The rule
--------
Per token, with state ``H`` shaped ``[V, K]``::

    H *= exp(gk)          # per key channel; the generic gated delta rule
                          # decays by a single scalar instead
    u = beta * (v - H k)
    H += u k^T
    o = H q

``q`` and ``k`` are L2-normalised as ``x / sqrt(sum(x**2) + eps)`` with the
epsilon **inside** the root, and ``q *= K ** -0.5``. :data:`L2_NORM_EPS` is that
epsilon, declared here rather than inherited from a default: both real KDA paths
use ``1e-6``, while the ``1e-5`` default belongs to a CPU shim that is not
callable in this image, and at ``atol=1e-5`` the wrong pick moves a comparison
by about the size of the tolerance.

The stages
----------
For one chunk of ``C`` tokens, key width ``K`` and value width ``V``, with ``gc``
the inclusive cumulative gate::

    gc[t, c] = sum over s <= t of gk[s, c]

    A[t, j]   = beta[t] * sum_c k[t, c] k[j, c] exp(gc[t, c] - gc[j, c])   for t > j
    Aqk[t, j] = sum_c q[t, c] k[j, c] exp(gc[t, c] - gc[j, c])             for t >= j
    T         = (I + A)**-1
    u         = T @ (beta * v)
    w         = T @ (beta * k * exp(gc))
    kg[t]     = k[t] * exp(gc[C - 1] - gc[t])

``A`` is strictly lower triangular and therefore singular, so the inverted object
is ``(I + A)`` and never ``A``. ``Aqk`` and ``kg`` are returned although nothing
here consumes them: the inter-chunk half needs both and has no other producer,
and it holds no raw ``k`` from which to rebuild ``kg``.

The inverse, without a token loop
---------------------------------
``N = -A`` is strictly lower triangular and therefore nilpotent with
``N**C == 0``, so the Neumann series terminates exactly::

    (I + A)**-1 = (I - N)**-1 = I + N + N**2 + ... + N**(C-1)

and its partial sums obey the doubling identity ``S_2m = (I + N**m) S_m``. The
kernel therefore reaches the full inverse in :func:`doubling_stages` ==
``log2(C)`` matmul stages rather than ``C`` substitution steps, and no loop in
this module walks the token axis: the cumulative sum is a matmul against a
triangular ones matrix, both chunk-local products are matmuls, and ``w`` / ``u``
are matmuls against the returned inverse. The torch oracle below reaches the
same value by a different route, ``torch.linalg.solve_triangular``, which is what
makes their agreement informative rather than circular.

Entry points
------------
:func:`kda_intra_chunk` is the entry point callers use, and it performs exactly
one ``wrap_nki`` dispatch per call, because the chunk loop lives inside the
kernel. :func:`kda_intra_chunk_kernel` computes stages 1 to 3 together;
:func:`kda_stage3_kernel` computes stage 3 alone and takes the inverse as an
argument, which is the boundary upstream's ``recompute_w_u_fwd`` also uses. Both
emit from the shared :func:`_emit_stage3`, so stage 3 has one implementation.

Four ``[C, C]`` constants are built on the host by :func:`chunk_constants` and
passed in as tensors: the upper-inclusive ones matrix that turns the cumulative
sum into a matmul, the identity that seeds the doubling series, the strictly
lower mask, and a row selector that broadcasts the chunk's last gate row. Passing
masks in follows ``nkilib``'s own convention -- its ``ssd`` asserts a
non-``None`` ``causal_mask`` rather than building one -- and these are 0/1
constants, not numerics.

On this image the tiles ``wrap_nki`` hands a kernel carry no Python operator
overloads, so ``a * b`` on two tiles raises ``TypeError``. Every elementwise step
below is therefore an explicit ``nisa.tensor_tensor`` / ``nisa.tensor_scalar`` /
``nisa.activation`` call, and every matmul writes into a PSUM tile that is copied
to SBUF before it is stored, because a direct PSUM-to-HBM store asserts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel, values_are_readable

logger = logging.getLogger(__name__)


#: The L2-normalisation epsilon, **inside** the square root. See the module
#: docstring for why this is a declared value and not an inherited default.
L2_NORM_EPS = 1e-6

#: Largest ``|gc|`` this kernel accepts, where ``gc`` is the inclusive cumulative
#: gate. The two chunk-local products are formed as ``exp(gc[t]) * exp(-gc[j])``
#: so that one matmul contracts the channel axis. That factorisation is exact, but
#: it evaluates both signs of the exponent, so a cumulative gate far from zero
#: overflows fp32 even where the product itself is tiny. At ``60`` the larger
#: factor is about ``1.1e26``, which leaves the channel sum room inside fp32's
#: ``3.4e38``. Upstream keeps the same quantity small differently, by blocking to
#: 16 or 64 and re-referencing the gate per block; this kernel does not tile, so
#: the limit is checked instead.
GATE_CUMSUM_ABS_LIMIT = 60.0

#: Widest chunk, key and value extent, one partition tile each. A literal rather
#: than ``nl.tile_size.pmax`` read at import time, because this module must import
#: on a host with no NKI device.
MAX_TILE = 128


class ChunkedRecurrenceError(ValueError):
    """Raised for a geometry or gate range this kernel does not serve."""


class IntraChunkOutputs(NamedTuple):
    """The five values stages 1 to 3 produce.

    ``a_inv`` and ``aqk`` are side outputs: nothing in this module consumes them,
    and the inter-chunk half has no other producer for them.
    """

    w: Tensor
    u: Tensor
    kg: Tensor
    a_inv: Tensor
    aqk: Tensor


class Stage3Outputs(NamedTuple):
    """What stage 3 alone returns."""

    w: Tensor
    u: Tensor
    kg: Tensor


class ChunkConstants(NamedTuple):
    """The four ``[C, C]`` host-built constants both kernel entries take."""

    triu_ones: Tensor
    eye: Tensor
    mask_lower: Tensor
    last_row: Tensor


@dataclass
class _DispatchCounters:
    """Which path actually ran, counted rather than inferred.

    ``nki_dispatch`` counts ``wrap_nki`` dispatches, ``torch_fallback`` entries
    into the torch path. Two counters rather than one flag, so "the kernel ran"
    and "the fallback did not" are independent readings.

    The count is per dispatch, not per call: the chunk loop lives inside the
    kernel, so a design that dispatched once per chunk reads a different number.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: Module level so a caller outside this module can reset and read it. The
#: inter-chunk half owns its own counters.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _COUNTERS.nki_dispatch += 1


def doubling_stages(chunk: int) -> int:
    """Matmul stages the Neumann series needs for a ``chunk``-wide tile.

    ``log2(chunk)``, because stage ``j`` holds the partial sum of the first
    ``2**j`` terms and ``N**chunk == 0`` makes term ``chunk`` onwards vanish.
    """
    return int(math.ceil(math.log2(chunk)))


def chunk_constants(
    chunk: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ChunkConstants:
    """Build the four ``[chunk, chunk]`` constants the kernel entries take.

    * ``triu_ones[s, t] = 1 for s <= t`` -- used as the matmul **stationary**
      operand, whose transpose is the lower-inclusive ones matrix, so
      ``triu_ones^T @ gk`` is the inclusive cumulative sum along tokens.
    * ``eye`` seeds the doubling series at ``S_1 = I``.
    * ``mask_lower[t, j] = 1 for j < t`` -- strictly lower, matching upstream's
      ``A``. The causal mask ``Aqk`` needs is ``mask_lower + eye`` and is formed
      in the kernel rather than passed, so one fewer constant travels.
    * ``last_row[s, t] = 1 for s == chunk - 1`` -- as a stationary operand its
      transpose selects the chunk's last row and repeats it down every row,
      which is what ``kg`` needs without a partition-axis broadcast.
    """
    idx = torch.arange(chunk, device=device)
    rows = idx.unsqueeze(1)
    cols = idx.unsqueeze(0)
    return ChunkConstants(
        triu_ones=(rows <= cols).to(dtype),
        eye=torch.eye(chunk, device=device, dtype=dtype),
        mask_lower=(cols < rows).to(dtype),
        last_row=(rows == (chunk - 1)).to(dtype).expand(chunk, chunk).contiguous(),
    )


# NKI emitting helpers, shared by both kernel entry points.


def _sbuf(rows, cols):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)


def _psum(rows, cols):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


def _emit_transpose(dst, src, rows, cols):
    """``dst = src^T`` for a ``[rows, cols]`` source, through PSUM.

    The PSUM destination is load-bearing: on this image an ``nc_transpose`` whose
    destination is in SBUF runs on the Vector engine and asserts above
    ``[32, 32]``, while a PSUM destination routes it to the tensor engine, which
    serves the full 128. Every transpose in this module is wider than 32 on at
    least one axis.
    """
    ps = _psum(cols, rows)
    nisa.nc_transpose(dst=ps, data=src)
    nisa.tensor_copy(dst=dst, src=ps)


def _emit_l2_normalise(dst, src, rows, cols):
    """``dst = src / sqrt(sum_c src**2 + L2_NORM_EPS)``, epsilon inside the root.

    The reduction is along the free axis, so ``nl.sum(..., axis=1)`` serves it and
    the reciprocal square root broadcasts back from a ``[rows, 1]`` tile through
    ``nisa.tensor_scalar``'s ``operand0``. No partition-axis reduction.
    """
    sq = _sbuf(rows, cols)
    nisa.tensor_tensor(dst=sq, data1=src, data2=src, op=nl.multiply)
    total = nl.sum(sq, axis=1, keepdims=True, dtype=nl.float32)
    den = _sbuf(rows, 1)
    nisa.tensor_scalar(dst=den, data=total, op0=nl.add, operand0=L2_NORM_EPS)
    inv_root = _sbuf(rows, 1)
    nisa.activation(dst=inv_root, data=den, op=nl.rsqrt)
    nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=inv_root)


def _emit_gate_cumsum(dst, gk_sb, triu_sb, chunk, width):
    """Inclusive cumulative gate along tokens, as one matmul.

    ``nisa.nc_matmul(stationary=S, moving=M)`` contracts the partition axis and
    computes ``S^T @ M``, so with ``S = triu_ones`` the result is
    ``lower_inclusive_ones @ gk``, which is the inclusive cumulative sum.
    """
    ps = _psum(chunk, width)
    nisa.nc_matmul(dst=ps, stationary=triu_sb, moving=gk_sb, accumulate=False)
    nisa.tensor_copy(dst=dst, src=ps)


def _emit_row_products(dst, left_sb, right_t_sb, chunk, width):
    """``dst = left @ right^T`` for ``[chunk, width]`` tiles.

    ``right_t_sb`` is already ``[width, chunk]``; ``left`` is transposed here so
    that ``stationary^T @ moving`` lands the ``[chunk, chunk]`` product.
    """
    left_t = _sbuf(width, chunk)
    _emit_transpose(left_t, left_sb, chunk, width)
    ps = _psum(chunk, chunk)
    nisa.nc_matmul(dst=ps, stationary=left_t, moving=right_t_sb, accumulate=False)
    nisa.tensor_copy(dst=dst, src=ps)


def _emit_unit_lower_inverse(dst, a_sb, eye_sb, chunk):
    """``dst = (I + A)**-1`` for strictly lower triangular ``A``, by doubling.

    ``s`` holds the partial sum and ``n`` holds ``N**(2**j)``; ``nt`` holds ``n``'s
    transpose so that every product is a ``stationary^T @ moving`` without a
    transpose inside the loop. The loop runs :func:`doubling_stages` ==
    ``log2(chunk)`` times and does not walk tokens: each stage squares the current
    power and doubles the number of series terms summed.

    Every scratch tile is allocated before the loop. Allocating inside would ask
    for one live PSUM tile per unrolled stage where two suffice.
    """
    stages = doubling_stages(chunk)

    n_sb = _sbuf(chunk, chunk)
    nt_sb = _sbuf(chunk, chunk)
    s_sb = _sbuf(chunk, chunk)
    nisa.tensor_scalar(dst=n_sb, data=a_sb, op0=nl.multiply, operand0=-1.0)
    _emit_transpose(nt_sb, n_sb, chunk, chunk)
    nisa.tensor_copy(dst=s_sb, src=eye_sb)

    ps_x = _psum(chunk, chunk)
    ps_y = _psum(chunk, chunk)
    tmp = _sbuf(chunk, chunk)

    for stage in range(stages):
        # s <- s + n @ s, which doubles the number of series terms summed.
        nisa.nc_matmul(dst=ps_x, stationary=nt_sb, moving=s_sb, accumulate=False)
        nisa.tensor_copy(dst=tmp, src=ps_x)
        nisa.tensor_tensor(dst=s_sb, data1=s_sb, data2=tmp, op=nl.add)
        if stage < stages - 1:
            # n <- n @ n and nt <- nt @ nt, both read from the pre-update pair.
            nisa.nc_matmul(dst=ps_x, stationary=nt_sb, moving=n_sb, accumulate=False)
            nisa.nc_matmul(dst=ps_y, stationary=n_sb, moving=nt_sb, accumulate=False)
            nisa.tensor_copy(dst=n_sb, src=ps_x)
            nisa.tensor_copy(dst=nt_sb, src=ps_y)

    nisa.tensor_copy(dst=dst, src=s_sb)


def _emit_stage3(
    w_dst, u_dst, kg_dst, k_sb, v_sb, beta_sb, gc_sb, egc_sb, a_inv_sb,
    last_row_sb, chunk, kdim, vdim,
):
    """Stage 3: ``w``, ``u`` and ``kg``. The one body both entry points emit.

    ``u = T @ (beta * v)`` and ``w = T @ (beta * k * exp(gc))``, each one matmul
    against the inverse ``T`` passed in. ``kg[t] = k[t] * exp(gc[C - 1] - gc[t])``
    takes the chunk's last gate row by a matmul against the row-selector constant,
    so no partition-axis broadcast is needed.
    """
    a_inv_t = _sbuf(chunk, chunk)
    _emit_transpose(a_inv_t, a_inv_sb, chunk, chunk)

    vb = _sbuf(chunk, vdim)
    nisa.tensor_scalar(dst=vb, data=v_sb, op0=nl.multiply, operand0=beta_sb)
    ps_u = _psum(chunk, vdim)
    nisa.nc_matmul(dst=ps_u, stationary=a_inv_t, moving=vb, accumulate=False)
    nisa.tensor_copy(dst=u_dst, src=ps_u)

    kb = _sbuf(chunk, kdim)
    nisa.tensor_scalar(dst=kb, data=k_sb, op0=nl.multiply, operand0=beta_sb)
    nisa.tensor_tensor(dst=kb, data1=kb, data2=egc_sb, op=nl.multiply)
    ps_w = _psum(chunk, kdim)
    nisa.nc_matmul(dst=ps_w, stationary=a_inv_t, moving=kb, accumulate=False)
    nisa.tensor_copy(dst=w_dst, src=ps_w)

    ps_last = _psum(chunk, kdim)
    nisa.nc_matmul(dst=ps_last, stationary=last_row_sb, moving=gc_sb, accumulate=False)
    gl = _sbuf(chunk, kdim)
    nisa.tensor_copy(dst=gl, src=ps_last)
    nisa.tensor_tensor(dst=gl, data1=gl, data2=gc_sb, op=nl.subtract)
    decay = _sbuf(chunk, kdim)
    nisa.activation(dst=decay, data=gl, op=nl.exp)
    nisa.tensor_tensor(dst=kg_dst, data1=k_sb, data2=decay, op=nl.multiply)


def _emit_prepare(k_sb, gc_sb, egc_sb, k_raw, gk_sb, triu_sb, chunk, kdim):
    """L2-normalise ``k`` into ``k_sb``, then form ``gc`` and ``exp(gc)``.

    Shared, so the stage-3 entry point derives its normalised key and cumulative
    gate exactly as the combined entry point does.
    """
    _emit_l2_normalise(k_sb, k_raw, chunk, kdim)
    _emit_gate_cumsum(gc_sb, gk_sb, triu_sb, chunk, kdim)
    nisa.activation(dst=egc_sb, data=gc_sb, op=nl.exp)


@nki.jit
def kda_intra_chunk_kernel(
    q_hbm, k_hbm, v_hbm, beta_hbm, gk_hbm, triu_hbm, eye_hbm, mask_lower_hbm,
    last_row_hbm,
):
    """Stages 1 to 3 for every chunk, in one dispatch.

    ``q``, ``k`` and ``gk`` are ``[NC, C, K]``; ``v`` is ``[NC, C, V]``; ``beta``
    is ``[NC, C, 1]``; the four constants are ``[C, C]``.

    The chunk loop is ``nl.affine_range`` because stages 1 to 3 are chunk-local
    with no carry between chunks. The inter-chunk half does carry state, which is
    why it needs a sequential range.
    """
    n_chunks, chunk, kdim = q_hbm.shape
    vdim = v_hbm.shape[2]
    scale = float(kdim) ** -0.5

    w_hbm = nl.ndarray((n_chunks, chunk, kdim), dtype=nl.float32, buffer=nl.shared_hbm)
    u_hbm = nl.ndarray((n_chunks, chunk, vdim), dtype=nl.float32, buffer=nl.shared_hbm)
    kg_hbm = nl.ndarray((n_chunks, chunk, kdim), dtype=nl.float32, buffer=nl.shared_hbm)
    ainv_hbm = nl.ndarray((n_chunks, chunk, chunk), dtype=nl.float32, buffer=nl.shared_hbm)
    aqk_hbm = nl.ndarray((n_chunks, chunk, chunk), dtype=nl.float32, buffer=nl.shared_hbm)

    triu_sb = _sbuf(chunk, chunk)
    eye_sb = _sbuf(chunk, chunk)
    mask_lower_sb = _sbuf(chunk, chunk)
    last_row_sb = _sbuf(chunk, chunk)
    causal_sb = _sbuf(chunk, chunk)
    nisa.tensor_copy(dst=triu_sb, src=nl.load(triu_hbm, dtype=nl.float32))
    nisa.tensor_copy(dst=eye_sb, src=nl.load(eye_hbm, dtype=nl.float32))
    nisa.tensor_copy(dst=mask_lower_sb, src=nl.load(mask_lower_hbm, dtype=nl.float32))
    nisa.tensor_copy(dst=last_row_sb, src=nl.load(last_row_hbm, dtype=nl.float32))
    nisa.tensor_tensor(dst=causal_sb, data1=mask_lower_sb, data2=eye_sb, op=nl.add)

    for ic in nl.affine_range(n_chunks):
        k_sb = _sbuf(chunk, kdim)
        gc_sb = _sbuf(chunk, kdim)
        egc_sb = _sbuf(chunk, kdim)
        gk_sb = _sbuf(chunk, kdim)
        nisa.tensor_copy(dst=gk_sb, src=nl.load(gk_hbm[ic], dtype=nl.float32))
        _emit_prepare(k_sb, gc_sb, egc_sb, nl.load(k_hbm[ic], dtype=nl.float32),
                      gk_sb, triu_sb, chunk, kdim)

        q_sb = _sbuf(chunk, kdim)
        _emit_l2_normalise(q_sb, nl.load(q_hbm[ic], dtype=nl.float32), chunk, kdim)
        nisa.tensor_scalar(dst=q_sb, data=q_sb, op0=nl.multiply, operand0=scale)

        beta_sb = _sbuf(chunk, 1)
        nisa.tensor_copy(dst=beta_sb, src=nl.load(beta_hbm[ic], dtype=nl.float32))

        # The gate difference exp(gc[t] - gc[j]) is factorised so that one matmul
        # contracts the channel axis: exp(gc[t]) on the left rows, exp(-gc[j]) on
        # the right rows. GATE_CUMSUM_ABS_LIMIT is what bounds both factors.
        neg_gc = _sbuf(chunk, kdim)
        emgc_sb = _sbuf(chunk, kdim)
        nisa.tensor_scalar(dst=neg_gc, data=gc_sb, op0=nl.multiply, operand0=-1.0)
        nisa.activation(dst=emgc_sb, data=neg_gc, op=nl.exp)

        kp_sb = _sbuf(chunk, kdim)
        km_sb = _sbuf(chunk, kdim)
        nisa.tensor_tensor(dst=kp_sb, data1=k_sb, data2=egc_sb, op=nl.multiply)
        nisa.tensor_tensor(dst=km_sb, data1=k_sb, data2=emgc_sb, op=nl.multiply)
        km_t_sb = _sbuf(kdim, chunk)
        _emit_transpose(km_t_sb, km_sb, chunk, kdim)

        kk_sb = _sbuf(chunk, chunk)
        a_sb = _sbuf(chunk, chunk)
        _emit_row_products(kk_sb, kp_sb, km_t_sb, chunk, kdim)
        nisa.tensor_scalar(dst=a_sb, data=kk_sb, op0=nl.multiply, operand0=beta_sb)
        nisa.tensor_tensor(dst=a_sb, data1=a_sb, data2=mask_lower_sb, op=nl.multiply)

        qp_sb = _sbuf(chunk, kdim)
        qk_sb = _sbuf(chunk, chunk)
        aqk_sb = _sbuf(chunk, chunk)
        nisa.tensor_tensor(dst=qp_sb, data1=q_sb, data2=egc_sb, op=nl.multiply)
        _emit_row_products(qk_sb, qp_sb, km_t_sb, chunk, kdim)
        nisa.tensor_tensor(dst=aqk_sb, data1=qk_sb, data2=causal_sb, op=nl.multiply)

        a_inv_sb = _sbuf(chunk, chunk)
        _emit_unit_lower_inverse(a_inv_sb, a_sb, eye_sb, chunk)

        v_sb = _sbuf(chunk, vdim)
        nisa.tensor_copy(dst=v_sb, src=nl.load(v_hbm[ic], dtype=nl.float32))
        w_sb = _sbuf(chunk, kdim)
        u_sb = _sbuf(chunk, vdim)
        kg_sb = _sbuf(chunk, kdim)
        _emit_stage3(w_sb, u_sb, kg_sb, k_sb, v_sb, beta_sb, gc_sb, egc_sb,
                     a_inv_sb, last_row_sb, chunk, kdim, vdim)

        nl.store(w_hbm[ic], value=w_sb)
        nl.store(u_hbm[ic], value=u_sb)
        nl.store(kg_hbm[ic], value=kg_sb)
        nl.store(ainv_hbm[ic], value=a_inv_sb)
        nl.store(aqk_hbm[ic], value=aqk_sb)

    return w_hbm, u_hbm, kg_hbm, ainv_hbm, aqk_hbm


@nki.jit
def kda_stage3_kernel(
    k_hbm, v_hbm, beta_hbm, gk_hbm, a_inv_hbm, triu_hbm, last_row_hbm
):
    """Stage 3 alone, taking the inverse as an argument.

    The boundary upstream's ``recompute_w_u_fwd`` also uses. The body is
    :func:`_emit_stage3`, the same one the combined entry point emits.

    Because ``u = T @ (beta * v)`` is linear in ``T``, scaling a row of the
    supplied inverse scales the same row of ``u``.
    """
    n_chunks, chunk, kdim = k_hbm.shape
    vdim = v_hbm.shape[2]

    w_hbm = nl.ndarray((n_chunks, chunk, kdim), dtype=nl.float32, buffer=nl.shared_hbm)
    u_hbm = nl.ndarray((n_chunks, chunk, vdim), dtype=nl.float32, buffer=nl.shared_hbm)
    kg_hbm = nl.ndarray((n_chunks, chunk, kdim), dtype=nl.float32, buffer=nl.shared_hbm)

    triu_sb = _sbuf(chunk, chunk)
    last_row_sb = _sbuf(chunk, chunk)
    nisa.tensor_copy(dst=triu_sb, src=nl.load(triu_hbm, dtype=nl.float32))
    nisa.tensor_copy(dst=last_row_sb, src=nl.load(last_row_hbm, dtype=nl.float32))

    for ic in nl.affine_range(n_chunks):
        k_sb = _sbuf(chunk, kdim)
        gc_sb = _sbuf(chunk, kdim)
        egc_sb = _sbuf(chunk, kdim)
        gk_sb = _sbuf(chunk, kdim)
        nisa.tensor_copy(dst=gk_sb, src=nl.load(gk_hbm[ic], dtype=nl.float32))
        _emit_prepare(k_sb, gc_sb, egc_sb, nl.load(k_hbm[ic], dtype=nl.float32),
                      gk_sb, triu_sb, chunk, kdim)

        beta_sb = _sbuf(chunk, 1)
        nisa.tensor_copy(dst=beta_sb, src=nl.load(beta_hbm[ic], dtype=nl.float32))
        v_sb = _sbuf(chunk, vdim)
        nisa.tensor_copy(dst=v_sb, src=nl.load(v_hbm[ic], dtype=nl.float32))
        a_inv_sb = _sbuf(chunk, chunk)
        nisa.tensor_copy(dst=a_inv_sb, src=nl.load(a_inv_hbm[ic], dtype=nl.float32))

        w_sb = _sbuf(chunk, kdim)
        u_sb = _sbuf(chunk, vdim)
        kg_sb = _sbuf(chunk, kdim)
        _emit_stage3(w_sb, u_sb, kg_sb, k_sb, v_sb, beta_sb, gc_sb, egc_sb,
                     a_inv_sb, last_row_sb, chunk, kdim, vdim)

        nl.store(w_hbm[ic], value=w_sb)
        nl.store(u_hbm[ic], value=u_sb)
        nl.store(kg_hbm[ic], value=kg_sb)

    return w_hbm, u_hbm, kg_hbm


def _require_admissible(
    n_chunks: int, chunk: int, kdim: int, vdim: int, gate_abs_max: float
) -> None:
    """Raise unless this kernel serves the input, naming every cause at once."""
    problems: list[str] = []
    if n_chunks < 1:
        problems.append(f"n_chunks={n_chunks} must be at least 1")
    if chunk < 2 or chunk > MAX_TILE:
        problems.append(
            f"chunk={chunk} must be in [2, {MAX_TILE}]; the kernel maps the chunk "
            f"onto the partition axis and declares no tiling, so one (I + A) tile "
            f"must fit one partition tile"
        )
    elif chunk & (chunk - 1):
        problems.append(
            f"chunk={chunk} must be a power of two; each doubling stage of the "
            f"terminating Neumann series reaches exactly 2**stage series terms"
        )
    if kdim < 1 or kdim > MAX_TILE:
        problems.append(
            f"kdim={kdim} must be in [1, {MAX_TILE}]; it is one matmul operand width"
        )
    if vdim < 1 or vdim > MAX_TILE:
        problems.append(
            f"vdim={vdim} must be in [1, {MAX_TILE}]; it is one matmul operand width"
        )
    if gate_abs_max > GATE_CUMSUM_ABS_LIMIT:
        problems.append(
            f"max|cumulative gate|={gate_abs_max:.3f} exceeds "
            f"{GATE_CUMSUM_ABS_LIMIT}; the chunk-local products factorise the gate "
            f"difference as exp(gc[t]) * exp(-gc[j]), so a cumulative gate this "
            f"far from zero would overflow fp32 in the larger factor"
        )
    if problems:
        raise ChunkedRecurrenceError(
            "kda_intra_chunk cannot serve this input: " + "; ".join(problems)
        )


def can_run_intra_chunk(
    reference: Tensor,
    n_chunks: int,
    chunk: int,
    kdim: int,
    vdim: int,
    gate_abs_max: float,
) -> bool:
    """Is the NKI path available *and* admissible for this input?

    Two independent conditions: ``can_run_kernel`` answers whether a device or
    simulator exists, :func:`_require_admissible` whether this kernel accepts
    these extents and this gate range.
    """
    _require_admissible(n_chunks, chunk, kdim, vdim, gate_abs_max)
    return can_run_kernel(reference)


def kda_intra_chunk(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gk: Tensor
) -> IntraChunkOutputs:
    """Stages 1 to 3 over every chunk, in one kernel dispatch.

    Args:
        q, k, gk: ``[NC, C, K]`` fp32. ``gk`` is the per-key-channel log gate,
            not yet accumulated; the kernel forms the cumulative sum.
        v: ``[NC, C, V]`` fp32.
        beta: ``[NC, C]`` fp32, one scalar per token.

    Returns:
        :class:`IntraChunkOutputs`.

    Raises:
        ChunkedRecurrenceError: on a rank mismatch, a shape disagreement, or an
            inadmissible geometry or gate range.
    """
    ranks = (("q", q, 3), ("k", k, 3), ("v", v, 3), ("gk", gk, 3), ("beta", beta, 2))
    for name, tensor, rank in ranks:
        if tensor.dim() != rank:
            raise ChunkedRecurrenceError(
                f"{name} must be {rank}-D, got shape {tuple(tensor.shape)}"
            )
    n_chunks, chunk, kdim = (int(x) for x in q.shape)
    vdim = int(v.shape[2])
    if tuple(k.shape) != (n_chunks, chunk, kdim) or tuple(gk.shape) != (
        n_chunks,
        chunk,
        kdim,
    ):
        raise ChunkedRecurrenceError(
            f"k {tuple(k.shape)} and gk {tuple(gk.shape)} must both match q "
            f"{tuple(q.shape)}"
        )
    if tuple(v.shape)[:2] != (n_chunks, chunk) or tuple(beta.shape) != (
        n_chunks,
        chunk,
    ):
        raise ChunkedRecurrenceError(
            f"v {tuple(v.shape)} and beta {tuple(beta.shape)} must agree with q's "
            f"leading dimensions {(n_chunks, chunk)}"
        )

    # The gate range is an eager-call precondition, because forming it reads the
    # gate values. A traced or meta-built call has none to read and passes 0.0,
    # which is inside the limit and refuses nothing. Only `can_run_kernel` decides
    # the path, so a graph build loses the diagnostic message and nothing else.
    gate_abs_max = (
        float(gk.float().cumsum(dim=1).abs().max().item())
        if values_are_readable(gk)
        else 0.0
    )
    if not can_run_intra_chunk(q, n_chunks, chunk, kdim, vdim, gate_abs_max):
        _count_torch_fallback()
        logger.debug(
            "kda_intra_chunk: NKI route unavailable, using the torch path "
            "(oracle only, never the shipped path)"
        )
        return kda_intra_chunk_torch_oracle(q, k, v, beta, gk)

    consts = chunk_constants(chunk, device=q.device, dtype=q.dtype)
    _count_nki_dispatch()
    w, u, kg, a_inv, aqk = wrap_nki(kda_intra_chunk_kernel)(
        q_hbm=q,
        k_hbm=k,
        v_hbm=v,
        beta_hbm=beta.unsqueeze(-1).contiguous(),
        gk_hbm=gk,
        triu_hbm=consts.triu_ones,
        eye_hbm=consts.eye,
        mask_lower_hbm=consts.mask_lower,
        last_row_hbm=consts.last_row,
    )
    return IntraChunkOutputs(w=w, u=u, kg=kg, a_inv=a_inv, aqk=aqk)


def kda_intra_chunk_torch_oracle(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gk: Tensor
) -> IntraChunkOutputs:
    """Stages 1 to 3 in torch, by different means than the kernel uses.

    Not mirroring the kernel is the point, so three differences are deliberate:

    * the gate difference is evaluated directly as ``exp(gc[t] - gc[j])``, where
      the kernel factorises it into ``exp(gc[t]) * exp(-gc[j])`` so one matmul can
      contract the channel axis. That makes this the better-conditioned of the
      two, which is what :data:`GATE_CUMSUM_ABS_LIMIT` keeps honest;
    * the inverse comes from ``torch.linalg.solve_triangular``, forward
      substitution, where the kernel sums a terminating Neumann series by
      doubling;
    * the two products are formed by broadcast sums rather than by transposes and
      matmuls.
    """
    q32, k32, v32 = q.float(), k.float(), v.float()
    beta32, gk32 = beta.float(), gk.float()
    n_chunks, chunk, kdim = q32.shape

    qn = q32 / torch.sqrt((q32 * q32).sum(-1, keepdim=True) + L2_NORM_EPS)
    kn = k32 / torch.sqrt((k32 * k32).sum(-1, keepdim=True) + L2_NORM_EPS)
    qn = qn * (float(kdim) ** -0.5)

    gc = gk32.cumsum(dim=1)
    # decay[n, t, j, c] = exp(gc[n, t, c] - gc[n, j, c]) -- formed directly.
    decay = torch.exp(gc.unsqueeze(2) - gc.unsqueeze(1))

    idx = torch.arange(chunk, device=q32.device)
    strictly_lower = (idx.unsqueeze(0) < idx.unsqueeze(1)).to(q32.dtype)
    causal = (idx.unsqueeze(0) <= idx.unsqueeze(1)).to(q32.dtype)

    beta_col = beta32.unsqueeze(-1)
    kk = (kn.unsqueeze(2) * kn.unsqueeze(1) * decay).sum(-1)
    a = kk * beta_col * strictly_lower
    aqk = (qn.unsqueeze(2) * kn.unsqueeze(1) * decay).sum(-1) * causal

    eye = torch.eye(chunk, device=q32.device, dtype=q32.dtype)
    a_inv = torch.linalg.solve_triangular(
        eye + a, eye.expand(n_chunks, chunk, chunk), upper=False, unitriangular=True
    )

    u = a_inv @ (beta_col * v32)
    w = a_inv @ (beta_col * kn * torch.exp(gc))
    kg = kn * torch.exp(gc[:, -1:, :] - gc)
    return IntraChunkOutputs(w=w, u=u, kg=kg, a_inv=a_inv, aqk=aqk)


def rebuild_i_plus_a(k: Tensor, beta: Tensor, gk: Tensor) -> Tensor:
    """``I + A`` from the same inputs, for checking an inverse.

    Separate from :func:`kda_intra_chunk_torch_oracle` so that a caller can
    multiply the kernel's own inverse by a reference ``(I + A)`` without depending
    on any other reference value.
    """
    k32, beta32, gk32 = k.float(), beta.float(), gk.float()
    _, chunk, _ = k32.shape
    kn = k32 / torch.sqrt((k32 * k32).sum(-1, keepdim=True) + L2_NORM_EPS)
    gc = gk32.cumsum(dim=1)
    decay = torch.exp(gc.unsqueeze(2) - gc.unsqueeze(1))
    idx = torch.arange(chunk, device=k32.device)
    strictly_lower = (idx.unsqueeze(0) < idx.unsqueeze(1)).to(k32.dtype)
    kk = (kn.unsqueeze(2) * kn.unsqueeze(1) * decay).sum(-1)
    a = kk * beta32.unsqueeze(-1) * strictly_lower
    return torch.eye(chunk, device=k32.device, dtype=k32.dtype) + a


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the intra-chunk kernel this module authors.

    Read off the unwrapped ``.func``, since ``nki.jit``'s wrapper reports its own
    module either way.
    """
    func = getattr(kda_intra_chunk_kernel, "func", None)
    target = func if func is not None else kda_intra_chunk_kernel
    return target.__module__, target.__qualname__


def stage3_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the stage-3 entry this module authors."""
    func = getattr(kda_stage3_kernel, "func", None)
    target = func if func is not None else kda_stage3_kernel
    return target.__module__, target.__qualname__


# Stages 4 and 5: the state carried across chunks, and the output.
#
# These share this file and the emitting helpers above with stages 1 to 3, and
# nothing else: separate kernel entry points, separate entry functions, separate
# counters, separate constants, separate admissibility checks.
#
# The output here is not ``o = H q``. That is the sequential formula and it
# belongs to the oracle. The chunked output is the gate-decayed inter-chunk part
# ``qg @ h_chunk`` plus the intra-chunk part ``Aqk @ v_new``. The two forms agree
# numerically, so writing the kernel in the oracle's form would make the
# comparison between them circular.


class InterChunkOutputs(NamedTuple):
    """The three values stages 4 and 5 produce.

    ``final_state`` is ``[V, K]``, the sequential scan's own orientation, so a
    comparison against it needs no transpose on either side. The kernel carries
    the state transposed internally, for the reason given on
    :func:`kda_inter_chunk_kernel`, and pays one transpose at the end.

    ``v_new`` is a side output: nothing here consumes it, and it makes stage 4's
    derivation of it from ``u`` readable without reading the kernel body.
    """

    o: Tensor
    final_state: Tensor
    v_new: Tensor


class SequentialOutputs(NamedTuple):
    """What the sequential oracle returns: flat ``o`` and the final state."""

    o: Tensor
    final_state: Tensor


class InterChunkConstants(NamedTuple):
    """The host-built constants the inter-chunk kernel takes.

    None is shared with the intra-chunk set, so a change to one cannot reach the
    other.
    """

    triu_ones: Tensor
    last_col: Tensor
    state_init: Tensor


@dataclass
class _InterDispatchCounters:
    """The inter-chunk path's own counters, never the intra-chunk pair's.

    Same shape and same per-dispatch rule as :class:`_DispatchCounters`, in a
    separate object so neither reading can pollute the other. The chunk loop is
    inside the kernel, so a host-side loop that dispatched once per chunk would
    read the chunk count rather than ``1``.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: Module level so a caller outside this module can reset and read it. A separate
#: object from :data:`_COUNTERS`; the two are never aliased.
_INTER_COUNTERS = _InterDispatchCounters()


def reset_inter_dispatch_counters() -> None:
    """Zero both inter-chunk counters."""
    _INTER_COUNTERS.nki_dispatch = 0
    _INTER_COUNTERS.torch_fallback = 0


def inter_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last inter-chunk reset."""
    return _INTER_COUNTERS.nki_dispatch, _INTER_COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_inter_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _INTER_COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_inter_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _INTER_COUNTERS.nki_dispatch += 1


def inter_chunk_constants(
    chunk: int,
    kdim: int,
    vdim: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> InterChunkConstants:
    """Build the constants the inter-chunk kernel takes.

    * ``triu_ones[s, t] = 1 for s <= t``. As a matmul stationary operand its
      transpose is the lower-inclusive ones matrix, so ``triu_ones^T @ gk`` is the
      inclusive cumulative gate along tokens within the chunk.
    * ``last_col[t, 0] = 1 for t == chunk - 1``. As a stationary operand's moving
      partner it selects the chunk's last cumulative-gate row and lands it as a
      ``[K, 1]`` column, which is the shape the per-key-channel state decay needs:
      one matmul, no transpose. That is why this constant is a column where the
      intra-chunk row selector is a full ``[C, C]`` tile.
    * ``state_init``, the ``[V, K]`` zero entering state, built on the host and
      loaded rather than zeroed on device. Being an argument, it is also how a
      caller passes a non-zero entering state. It is stated in the orientation
      :attr:`InterChunkOutputs.final_state` returns, so a caller can loop one
      call's output into the next with no transpose of its own.
    """
    idx = torch.arange(chunk, device=device)
    return InterChunkConstants(
        triu_ones=(idx.unsqueeze(1) <= idx.unsqueeze(0)).to(dtype),
        last_col=(idx == (chunk - 1)).to(dtype).unsqueeze(1).contiguous(),
        state_init=torch.zeros(vdim, kdim, device=device, dtype=dtype),
    )


@nki.jit
def kda_inter_chunk_kernel(
    kg_hbm, w_hbm, u_hbm, gk_hbm, q_hbm, aqk_hbm, triu_hbm, last_col_hbm,
    state_init_hbm,
):
    """Stages 4 and 5 for every chunk, in one dispatch.

    ``kg``, ``w``, ``gk`` and ``q`` are ``[NC, C, K]``; ``u`` is ``[NC, C, V]``;
    ``aqk`` is ``[NC, C, C]``; ``triu`` is ``[C, C]``; ``last_col`` is ``[C, 1]``;
    ``state_init`` is ``[V, K]``.

    There is no raw ``v`` argument: ``v_new`` is derived from ``u`` inside stage 4.

    The chunk loop is ``nl.sequential_range`` where the intra-chunk kernel's is
    ``nl.affine_range``, because stage 4 carries a state from one chunk into the
    next and stages 1 to 3 do not.

    The state is carried transposed, as ``ht`` shaped ``[K, V]``, where upstream
    carries ``[V, K]``. Two things follow: ``nc_matmul(stationary=kg,
    moving=v_new)`` computes ``kg^T @ v_new`` and lands ``[K, V]`` directly, so
    the update needs no transpose; and the per-key-channel decay becomes a
    ``[K, 1]`` operand broadcast along the free axis, which is what
    ``nisa.tensor_scalar`` serves. Carried the other way round the decay would
    need a partition-axis broadcast, which it does not do. The transpose back to
    ``[V, K]`` is paid once after the loop, and the entering state is turned the
    same way once before it.
    """
    n_chunks, chunk, kdim = kg_hbm.shape
    vdim = u_hbm.shape[2]
    scale = float(kdim) ** -0.5

    o_hbm = nl.ndarray((n_chunks, chunk, vdim), dtype=nl.float32, buffer=nl.shared_hbm)
    vnew_hbm = nl.ndarray(
        (n_chunks, chunk, vdim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    state_hbm = nl.ndarray((vdim, kdim), dtype=nl.float32, buffer=nl.shared_hbm)

    triu_sb = _sbuf(chunk, chunk)
    last_col_sb = _sbuf(chunk, 1)
    nisa.tensor_copy(dst=triu_sb, src=nl.load(triu_hbm, dtype=nl.float32))
    nisa.tensor_copy(dst=last_col_sb, src=nl.load(last_col_hbm, dtype=nl.float32))

    # The loop-carried value, allocated once before the loop because it is the
    # one tile that must survive an iteration boundary. The entering state
    # arrives in the orientation this kernel returns and is turned here, so the
    # conversion is emitted on this engine rather than by torch on the host.
    entering_sb = _sbuf(vdim, kdim)
    nisa.tensor_copy(dst=entering_sb, src=nl.load(state_init_hbm, dtype=nl.float32))
    ht_sb = _sbuf(kdim, vdim)
    _emit_transpose(ht_sb, entering_sb, vdim, kdim)

    for ic in nl.sequential_range(n_chunks):
        kg_sb = _sbuf(chunk, kdim)
        w_sb = _sbuf(chunk, kdim)
        u_sb = _sbuf(chunk, vdim)
        gk_sb = _sbuf(chunk, kdim)
        aqk_sb = _sbuf(chunk, chunk)
        nisa.tensor_copy(dst=kg_sb, src=nl.load(kg_hbm[ic], dtype=nl.float32))
        nisa.tensor_copy(dst=w_sb, src=nl.load(w_hbm[ic], dtype=nl.float32))
        nisa.tensor_copy(dst=u_sb, src=nl.load(u_hbm[ic], dtype=nl.float32))
        nisa.tensor_copy(dst=gk_sb, src=nl.load(gk_hbm[ic], dtype=nl.float32))
        nisa.tensor_copy(dst=aqk_sb, src=nl.load(aqk_hbm[ic], dtype=nl.float32))

        # The cumulative gate is CHUNK-LOCAL, exactly as upstream re-references
        # it per chunk. That is what bounds every exponent below by one chunk's
        # gate sum instead of by the whole sequence's.
        gc_sb = _sbuf(chunk, kdim)
        _emit_gate_cumsum(gc_sb, gk_sb, triu_sb, chunk, kdim)
        egc_sb = _sbuf(chunk, kdim)
        nisa.activation(dst=egc_sb, data=gc_sb, op=nl.exp)

        # ---- stage 4, first half: v_new = u - w @ ht, on the ENTERING state.
        w_t_sb = _sbuf(kdim, chunk)
        _emit_transpose(w_t_sb, w_sb, chunk, kdim)
        ps_v = _psum(chunk, vdim)
        nisa.nc_matmul(dst=ps_v, stationary=w_t_sb, moving=ht_sb, accumulate=False)
        wh_sb = _sbuf(chunk, vdim)
        nisa.tensor_copy(dst=wh_sb, src=ps_v)
        vnew_sb = _sbuf(chunk, vdim)
        nisa.tensor_tensor(dst=vnew_sb, data1=u_sb, data2=wh_sb, op=nl.subtract)

        # ---- stage 5: o = qg @ ht + Aqk @ v_new, also on the ENTERING state.
        # `q` arrives raw and is normalised and scaled here rather than imported
        # already normalised, because this entry point takes `q` and not `qn`.
        # `Aqk` already carries both the K**-0.5 scale and the causal mask from
        # stage 2, so it is neither re-scaled nor re-masked.
        q_sb = _sbuf(chunk, kdim)
        _emit_l2_normalise(q_sb, nl.load(q_hbm[ic], dtype=nl.float32), chunk, kdim)
        nisa.tensor_scalar(dst=q_sb, data=q_sb, op0=nl.multiply, operand0=scale)
        qg_sb = _sbuf(chunk, kdim)
        nisa.tensor_tensor(dst=qg_sb, data1=q_sb, data2=egc_sb, op=nl.multiply)
        qg_t_sb = _sbuf(kdim, chunk)
        _emit_transpose(qg_t_sb, qg_sb, chunk, kdim)
        ps_o = _psum(chunk, vdim)
        nisa.nc_matmul(dst=ps_o, stationary=qg_t_sb, moving=ht_sb, accumulate=False)
        inter_sb = _sbuf(chunk, vdim)
        nisa.tensor_copy(dst=inter_sb, src=ps_o)

        aqk_t_sb = _sbuf(chunk, chunk)
        _emit_transpose(aqk_t_sb, aqk_sb, chunk, chunk)
        ps_a = _psum(chunk, vdim)
        nisa.nc_matmul(dst=ps_a, stationary=aqk_t_sb, moving=vnew_sb, accumulate=False)
        intra_sb = _sbuf(chunk, vdim)
        nisa.tensor_copy(dst=intra_sb, src=ps_a)

        o_sb = _sbuf(chunk, vdim)
        nisa.tensor_tensor(dst=o_sb, data1=inter_sb, data2=intra_sb, op=nl.add)

        nl.store(o_hbm[ic], value=o_sb)
        nl.store(vnew_hbm[ic], value=vnew_sb)

        # ---- stage 4, second half: the carry.
        # ht <- ht * exp(gc[C - 1]) + kg^T @ v_new. The decay column is the
        # chunk's last cumulative-gate row, landed as [K, 1] by one matmul
        # against the column selector -- no transpose, no partition broadcast.
        ps_d = _psum(kdim, 1)
        nisa.nc_matmul(dst=ps_d, stationary=gc_sb, moving=last_col_sb, accumulate=False)
        glast_sb = _sbuf(kdim, 1)
        nisa.tensor_copy(dst=glast_sb, src=ps_d)
        decay_sb = _sbuf(kdim, 1)
        nisa.activation(dst=decay_sb, data=glast_sb, op=nl.exp)
        decayed_sb = _sbuf(kdim, vdim)
        nisa.tensor_scalar(
            dst=decayed_sb, data=ht_sb, op0=nl.multiply, operand0=decay_sb
        )

        ps_h = _psum(kdim, vdim)
        nisa.nc_matmul(dst=ps_h, stationary=kg_sb, moving=vnew_sb, accumulate=False)
        upd_sb = _sbuf(kdim, vdim)
        nisa.tensor_copy(dst=upd_sb, src=ps_h)
        nisa.tensor_tensor(dst=ht_sb, data1=decayed_sb, data2=upd_sb, op=nl.add)

    state_sb = _sbuf(vdim, kdim)
    _emit_transpose(state_sb, ht_sb, kdim, vdim)
    nl.store(state_hbm, value=state_sb)

    return o_hbm, state_hbm, vnew_hbm


def _require_inter_admissible(
    n_chunks: int, chunk: int, kdim: int, vdim: int, gate_abs_max: float
) -> None:
    """Raise unless the inter-chunk kernel serves the input.

    Unlike :func:`_require_admissible` this does not require ``chunk`` to be a
    power of two: that requirement comes from the doubling series, and this kernel
    sums no series.

    ``gate_abs_max`` is the largest absolute chunk-local cumulative gate, so the
    bound applies to one chunk's gate sum and not to the whole sequence's. See
    :func:`kda_inter_chunk` for why that distinction keeps the state carry inside
    fp32.
    """
    problems: list[str] = []
    if n_chunks < 1:
        problems.append(f"n_chunks={n_chunks} must be at least 1")
    if chunk < 2 or chunk > MAX_TILE:
        problems.append(
            f"chunk={chunk} must be in [2, {MAX_TILE}]; the kernel maps the chunk "
            f"onto the partition axis and declares no tiling"
        )
    if kdim < 1 or kdim > MAX_TILE:
        problems.append(
            f"kdim={kdim} must be in [1, {MAX_TILE}]; it is the carried state's "
            f"partition extent"
        )
    if vdim < 1 or vdim > MAX_TILE:
        problems.append(
            f"vdim={vdim} must be in [1, {MAX_TILE}]; it is the carried state's "
            f"free extent"
        )
    if gate_abs_max > GATE_CUMSUM_ABS_LIMIT:
        problems.append(
            f"max|chunk-local cumulative gate|={gate_abs_max:.3f} exceeds "
            f"{GATE_CUMSUM_ABS_LIMIT}; the state decay and the output gate are "
            f"both exp of that quantity, so a chunk-local cumulative gate this "
            f"far from zero would overflow fp32"
        )
    if problems:
        raise ChunkedRecurrenceError(
            "kda_inter_chunk cannot serve this input: " + "; ".join(problems)
        )


def can_run_inter_chunk(
    reference: Tensor,
    n_chunks: int,
    chunk: int,
    kdim: int,
    vdim: int,
    gate_abs_max: float,
) -> bool:
    """Is the NKI path available *and* admissible for this inter-chunk input?

    Two independent conditions: ``can_run_kernel`` answers whether a device or
    simulator exists, :func:`_require_inter_admissible` whether this kernel
    accepts these extents and this gate range.
    """
    _require_inter_admissible(n_chunks, chunk, kdim, vdim, gate_abs_max)
    return can_run_kernel(reference)


def kda_inter_chunk(
    kg: Tensor,
    w: Tensor,
    u: Tensor,
    gk: Tensor,
    q: Tensor,
    aqk: Tensor,
    *,
    state: Tensor | None = None,
) -> InterChunkOutputs:
    """Stages 4 and 5 over every chunk, in one kernel dispatch.

    ``kg``, ``w``, ``u`` and ``aqk`` are :func:`kda_intra_chunk`'s returns; ``gk``
    and ``q`` are the same raw inputs it took. There is no raw ``v``.

    Args:
        kg: ``[NC, C, K]`` fp32, the gated key.
        w: ``[NC, C, K]`` fp32, the WY row factor.
        u: ``[NC, C, V]`` fp32, the WY value factor. ``v_new`` is derived from
            this inside stage 4.
        gk: ``[NC, C, K]`` fp32, the per-key-channel log gate, not yet
            accumulated; the kernel forms the chunk-local cumulative sum.
        q: ``[NC, C, K]`` fp32, raw. Normalised and scaled inside the kernel.
        aqk: ``[NC, C, C]`` fp32, already carrying the scale and the causal mask.
        state: ``[V, K]`` entering recurrent state, in ``final_state``'s own
            orientation, or ``None`` for a zero entry. A caller whose tokens
            arrive across several calls passes the state the previous call
            returned.

    Returns:
        :class:`InterChunkOutputs`, whose ``final_state`` is ``[V, K]``.

    Raises:
        ChunkedRecurrenceError: on a rank mismatch, a shape disagreement, or an
            inadmissible geometry or gate range.

    The state carry cannot overflow with sequence length, by construction rather
    than by a wider bound. A carry written over the whole sequence's cumulative
    gate would exponentiate a quantity that grows with the token count. This one
    decays by ``exp(gc[C - 1])`` where ``gc`` is re-referenced from zero inside
    every chunk, which is upstream's own convention, so the exponent is bounded by
    one chunk's gate sum however long the sequence is, and
    :data:`GATE_CUMSUM_ABS_LIMIT` is checked against exactly that quantity.
    """
    ranks = (
        ("kg", kg, 3), ("w", w, 3), ("u", u, 3), ("gk", gk, 3), ("q", q, 3),
        ("aqk", aqk, 3),
    )
    for name, tensor, rank in ranks:
        if tensor.dim() != rank:
            raise ChunkedRecurrenceError(
                f"{name} must be {rank}-D, got shape {tuple(tensor.shape)}"
            )
    n_chunks, chunk, kdim = (int(x) for x in kg.shape)
    vdim = int(u.shape[2])
    for name, tensor in (("w", w), ("gk", gk), ("q", q)):
        if tuple(tensor.shape) != (n_chunks, chunk, kdim):
            raise ChunkedRecurrenceError(
                f"{name} {tuple(tensor.shape)} must match kg {tuple(kg.shape)}"
            )
    if tuple(u.shape)[:2] != (n_chunks, chunk):
        raise ChunkedRecurrenceError(
            f"u {tuple(u.shape)} must agree with kg's leading dimensions "
            f"{(n_chunks, chunk)}"
        )
    if tuple(aqk.shape) != (n_chunks, chunk, chunk):
        raise ChunkedRecurrenceError(
            f"aqk {tuple(aqk.shape)} must be {(n_chunks, chunk, chunk)}"
        )

    # The entering state is refused on the same footing as every other operand,
    # and before the route is chosen, so a caller sees one contract whichever
    # path serves it.
    if state is not None:
        if state.dim() != 2:
            raise ChunkedRecurrenceError(
                f"state must be 2-D [V, K], got shape {tuple(state.shape)}"
            )
        if tuple(state.shape) != (vdim, kdim):
            raise ChunkedRecurrenceError(
                f"state {tuple(state.shape)} must be {(vdim, kdim)} -- the "
                f"orientation final_state is returned in"
            )
        if state.dtype != kg.dtype:
            raise ChunkedRecurrenceError(
                f"state dtype {state.dtype} must be kg's {kg.dtype}; the entering "
                f"state is combined with these operands and is not cast here"
            )

    # Read on an eager call and 0.0 under a graph build, for the reason
    # `kda_intra_chunk` gives above.
    gate_abs_max = (
        float(gk.float().cumsum(dim=1).abs().max().item())
        if values_are_readable(gk)
        else 0.0
    )
    if not can_run_inter_chunk(q, n_chunks, chunk, kdim, vdim, gate_abs_max):
        _count_inter_torch_fallback()
        logger.debug(
            "kda_inter_chunk: NKI route unavailable, using the torch path "
            "(oracle only, never the shipped path)"
        )
        return kda_inter_chunk_torch_oracle(kg, w, u, gk, q, aqk, state=state)

    consts = inter_chunk_constants(chunk, kdim, vdim, device=kg.device, dtype=kg.dtype)
    _count_inter_nki_dispatch()
    o, final_state, v_new = wrap_nki(kda_inter_chunk_kernel)(
        kg_hbm=kg,
        w_hbm=w,
        u_hbm=u,
        gk_hbm=gk,
        q_hbm=q,
        aqk_hbm=aqk,
        triu_hbm=consts.triu_ones,
        last_col_hbm=consts.last_col,
        state_init_hbm=consts.state_init if state is None else state.contiguous(),
    )
    return InterChunkOutputs(o=o, final_state=final_state, v_new=v_new)


def kda_inter_chunk_torch_oracle(
    kg: Tensor,
    w: Tensor,
    u: Tensor,
    gk: Tensor,
    q: Tensor,
    aqk: Tensor,
    *,
    state: Tensor | None = None,
) -> InterChunkOutputs:
    """Stages 4 and 5 in torch: the fallback path, never the shipped one.

    Present so a host with no device or simulator can exercise this module's
    contract. It follows the same chunked formula the kernel does, so it is not an
    independent reference; :func:`kda_sequential_torch_oracle` is.
    """
    kg32, w32, u32 = kg.float(), w.float(), u.float()
    gk32, q32, aqk32 = gk.float(), q.float(), aqk.float()
    n_chunks, chunk, kdim = kg32.shape
    vdim = u32.shape[2]

    qn = q32 / torch.sqrt((q32 * q32).sum(-1, keepdim=True) + L2_NORM_EPS)
    qn = qn * (float(kdim) ** -0.5)
    gc = gk32.cumsum(dim=1)

    # ``state`` arrives as ``[V, K]`` and is turned into the ``[K, V]`` this scan
    # carries, the same conversion the kernel emits.
    ht = (
        torch.zeros(kdim, vdim, device=kg32.device, dtype=kg32.dtype)
        if state is None
        else state.float().t().contiguous()
    )
    o = torch.empty(n_chunks, chunk, vdim, device=kg32.device, dtype=kg32.dtype)
    v_new = torch.empty_like(o)
    for c in range(n_chunks):
        vn = u32[c] - w32[c] @ ht
        v_new[c] = vn
        o[c] = (qn[c] * torch.exp(gc[c])) @ ht + aqk32[c] @ vn
        ht = ht * torch.exp(gc[c, -1]).unsqueeze(1) + kg32[c].t() @ vn
    return InterChunkOutputs(o=o, final_state=ht.t().contiguous(), v_new=v_new)


def kda_sequential_torch_oracle(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gk: Tensor
) -> SequentialOutputs:
    """The sequential delta rule, one token at a time, over flat inputs.

    The independent reference for the chunked path: it materialises no ``A``, forms
    no inverse, computes no ``w`` / ``u`` / ``kg`` / ``Aqk``, and never groups
    tokens. Agreement with the kernel is therefore a statement that the chunking
    is associativity-correct.

    Args:
        q, k, gk: ``[T, K]`` fp32, flat over tokens, with no chunk axis.
        v: ``[T, V]`` fp32.
        beta: ``[T]`` fp32.

    Returns:
        :class:`SequentialOutputs` with ``o`` shaped ``[T, V]`` and
        ``final_state`` shaped ``[V, K]``.

    Three details below are load-bearing for that comparison: the state decays per
    key channel rather than by one scalar, which is the KDA-specific step; the
    L2-normalisation epsilon is :data:`L2_NORM_EPS` and sits inside the square
    root; and ``q`` carries ``K ** -0.5``. So is the step order: the state is
    decayed first, and the delta then reads the decayed state.
    """
    q32, k32, v32 = q.float(), k.float(), v.float()
    beta32, gk32 = beta.float(), gk.float()
    tokens, kdim = q32.shape
    vdim = v32.shape[1]

    qn = q32 / torch.sqrt((q32 * q32).sum(-1, keepdim=True) + L2_NORM_EPS)
    kn = k32 / torch.sqrt((k32 * k32).sum(-1, keepdim=True) + L2_NORM_EPS)
    qn = qn * (float(kdim) ** -0.5)

    state = torch.zeros(vdim, kdim, device=q32.device, dtype=q32.dtype)
    o = torch.empty(tokens, vdim, device=q32.device, dtype=q32.dtype)
    for t in range(tokens):
        state = state * torch.exp(gk32[t]).unsqueeze(0)
        delta = (v32[t] - state @ kn[t]) * beta32[t]
        state = state + torch.outer(delta, kn[t])
        o[t] = state @ qn[t]
    return SequentialOutputs(o=o, final_state=state)


def inter_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the inter-chunk kernel this module authors.

    Read off the unwrapped ``.func``, since ``nki.jit``'s wrapper reports its own
    module either way.
    """
    func = getattr(kda_inter_chunk_kernel, "func", None)
    target = func if func is not None else kda_inter_chunk_kernel
    return target.__module__, target.__qualname__
