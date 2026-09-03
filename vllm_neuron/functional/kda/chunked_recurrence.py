# SPDX-License-Identifier: Apache-2.0
"""KDA chunked recurrence, intra-chunk half: an NKI kernel authored here.

`inc-glm53f-035a`. This is WP3's intra-chunk half of the Kimi Delta Attention
recurrence -- upstream's **stages 1 to 3** and no more: the gate cumulative sum,
the L2-normalisation of ``q`` and ``k``, the two chunk-local products ``A`` and
``Aqk``, the inverse ``(I + A)**-1``, and the WY representation ``w`` / ``u``
together with the gated key ``kg``. The state carried **across** chunks and the
output are `inc-glm53f-035b`'s and are not computed here.

It is **kernel-class** under P13 and it is **ADAPT**: the substrate ships
structurally related scans (``nkilib.experimental.scan.ssd``,
``...scan.linear_scan``) but not this rule, so the arithmetic below is
re-derived in NKI on the fork's own ADAPT precedent,
``vllm_neuron/functional/mhc/sinkhorn.py`` -- an ``@nki.jit`` kernel, a counted
``wrap_nki`` seam, a ``can_run_kernel`` gate, a torch oracle and a
``kernel_identity()`` reading. The torch code in this module is the CPU oracle,
never the shipped path. **A token-sequential scan written in torch here would be
exactly the P13 fallback the rule forbids.**

Where the rule comes from
-------------------------
**Cited, never copied.** No campaign artifact and no fork file states the KDA
delta rule; it is read from the pinned vLLM **0.24.0** in the campaign venv, at
``vllm/model_executor/layers/fla/ops/fused_recurrent.py:122-149`` reached with
``IS_KDA=True`` (``kda.py:101``). Per token, with state ``H`` shaped
``[V, K]``::

    H *= exp(gk)          # PER KEY CHANNEL -- the KDA-specific step, where the
                          # generic gated delta rule decays by one scalar
    u = beta * (v - H k)
    H += u k^T
    o = H q

``q`` and ``k`` are L2-normalised as ``x / sqrt(sum(x**2) + eps)`` with the
**epsilon inside the root**, and ``q *= K ** -0.5`` (``kda.py:129``).
:data:`L2_NORM_EPS` is that epsilon, written here as a declared value rather
than inherited from a default: both real KDA paths use ``1e-6`` -- the
sequential one hardcodes it at ``fused_recurrent.py:128-129``, the chunked one
takes ``l2norm_fwd(x, eps=1e-6)`` at ``l2norm.py:96`` -- while the ``1e-5``
default belongs only to the CPU shim that is **not callable** in this venv
(``vllm._C`` absent). At this increment's ``atol=1e-5`` the wrong pick would
move the comparison by about the size of the tolerance itself.

The stages, as upstream defines them
------------------------------------
Read from the same pinned source, for one chunk of ``C`` tokens, key width
``K``, value width ``V``, with ``gc`` the inclusive cumulative gate::

    gc[t, c] = sum over s <= t of gk[s, c]

    A[t, j]   = beta[t] * sum_c k[t, c] k[j, c] exp(gc[t, c] - gc[j, c])   for t > j
    Aqk[t, j] = sum_c q[t, c] k[j, c] exp(gc[t, c] - gc[j, c])             for t >= j
    T         = (I + A)**-1
    u         = T @ (beta * v)
    w         = T @ (beta * k * exp(gc))
    kg[t]     = k[t] * exp(gc[C - 1] - gc[t])

``A`` is **strictly** lower triangular and therefore singular, so the inverted
object is ``(I + A)`` and never ``A`` -- upstream says so in its own words at
``solve_tril.py:514-529`` ("Compute the inverse of the matrix I + A. A should be
strictly lower triangular"). ``Aqk`` is produced here although nothing here
consumes it: `-035b` needs it for the intra-chunk half of the output and has no
other producer. ``kg`` likewise -- `-035b` declares it as a seam input and has
no raw ``k`` to derive it from.

Why the inverse has no token loop, and how it is formed
-------------------------------------------------------
This is the property the increment exists to deliver, so it is worth stating
plainly. ``N = -A`` is strictly lower triangular and therefore nilpotent with
``N**C == 0``, so the Neumann series **terminates exactly**::

    (I + A)**-1 = (I - N)**-1 = I + N + N**2 + ... + N**(C-1)

and the partial sums obey a doubling identity, ``S_2m = (I + N**m) S_m``, which
is checked by expanding the product. The kernel therefore reaches the full
inverse in :func:`doubling_stages` ``= log2(C)`` matmul stages rather than in
``C`` substitution steps. **The loop count is log2 of the chunk length, and no
loop anywhere in this module walks the token axis**: the cumulative sum is a
matmul against a triangular ones matrix, the two chunk-local products are
matmuls, the inverse is the doubling series above, and ``w`` / ``u`` are matmuls
against the returned inverse. A kernel that carried the state in SBUF and looped
tokens would satisfy neither this description nor the increment's discriminating
test.

The doubling series is also **not** upstream's algorithm -- upstream inverts by
blocked 16x16 forward substitution (``solve_tril.py:38``) -- and the torch
oracle below uses a third route, ``torch.linalg.solve_triangular``. Three
different routes to one value, which is what makes the agreement informative
rather than circular.

Two kernel entries, and why
---------------------------
:func:`kda_intra_chunk_kernel` is stages 1 to 3 in one dispatch.
:func:`kda_stage3_kernel` is stage 3 alone, **taking the inverse as an
argument** -- upstream's own boundary, not an invented one:
``recompute_w_u_fwd`` (``kda.py:960``) takes it the same way. Both emit from the
one shared body :func:`_emit_stage3`, so there is a single implementation of
stage 3 to read and to review, and the separate entry adds a signature rather
than a second copy of the arithmetic.

Only :func:`kda_intra_chunk` -- the seam -- is counted, and it performs
**exactly one** ``wrap_nki`` dispatch per call. That is what the route predicate
reads: the chunking is inside the kernel, so a host loop over chunks would read
the chunk count instead and the two readings tell the two designs apart. A
direct call to either entry bypasses the seam and moves no counter.

The constant matrices, and the substrate precedent for passing them in
----------------------------------------------------------------------
Four ``[C, C]`` constants are built on the host by :func:`chunk_constants` and
passed in as tensors: the upper-inclusive ones matrix that turns the cumulative
sum into a matmul, the identity that seeds the doubling series, the strictly
lower mask, and a row-selector that broadcasts the chunk's last gate row.
Passing masks in is the substrate's own convention -- ``nkilib``'s ``ssd``
asserts a non-``None`` ``causal_mask`` argument rather than building one -- and
these are 0/1 constants, not numerics.

Why every operation is a ``dst=``-style ``nisa`` call
-----------------------------------------------------
On this pinned image the tiles ``wrap_nki`` hands the kernel do not carry
Python operator overloads, so ``a * b`` on two tiles raises ``TypeError``. Every
elementwise step below is therefore an explicit ``nisa.tensor_tensor`` /
``nisa.tensor_scalar`` / ``nisa.activation`` call, and every matmul writes into
a PSUM tile that is copied to SBUF before it is stored -- a direct PSUM-to-HBM
store asserts. Both constraints are the image's, established by measurement
rather than assumed, and ``sinkhorn.py`` already writes in the same style.
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

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)


#: The L2-normalisation epsilon, **inside** the square root. A declared value,
#: not an inherited default -- see the module docstring for both upstream sites
#: that fix it at ``1e-6`` and for the ``1e-5`` shim default that does not apply.
L2_NORM_EPS = 1e-6

#: Largest ``|gc|`` this kernel accepts, where ``gc`` is the inclusive cumulative
#: gate. The two chunk-local products are formed as ``exp(gc[t]) * exp(-gc[j])``
#: so that a single matmul contracts the channel axis; that factorisation is
#: exact but it evaluates both signs of the exponent, so a cumulative gate far
#: from zero would overflow fp32 even where the product itself is tiny. At
#: ``60`` the larger factor is about ``1.1e26``, which leaves the channel sum
#: room inside fp32's ``3.4e38``. Upstream keeps the same quantity small a
#: different way -- it blocks to 16 or 64 and re-references the gate per block --
#: and this block declares no tiling, so the limit is stated and checked instead
#: of assumed.
GATE_CUMSUM_ABS_LIMIT = 60.0

#: Widest chunk, key and value extent, one partition tile each. Written as a
#: literal rather than read from ``nl.tile_size`` at import time because this
#: module must import on a host with no NKI device; it is the pinned image's
#: ``nl.tile_size.pmax``.
MAX_TILE = 128


class ChunkedRecurrenceError(ValueError):
    """Raised for a geometry or a gate range this kernel does not serve.

    Inadmissibility raises rather than falling back, because falling back would
    ship a torch path for kernel-class work (P13).
    """


class IntraChunkOutputs(NamedTuple):
    """Stages 1 to 3, one field per value the next increment or a test reads.

    ``a_inv`` and ``aqk`` are side outputs here: nothing in this module consumes
    them, and both exist because `-035b` has no other producer for them.
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
    """What route actually ran, counted rather than inferred.

    ``nki_dispatch`` counts ``wrap_nki`` dispatches made by the seam;
    ``torch_fallback`` counts entries into the torch path. Two counters rather
    than one flag, so "the kernel ran" and "the fallback did not run" are
    independent readings and a test can require both.

    The count is per **dispatch** and not per seam call. That is deliberate, and
    it is what makes the route predicate discriminating: a seam that looped over
    chunks and dispatched once per chunk would read the chunk count, so counting
    seam entries instead would defeat the reading the predicate exists to take.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: MODULE-LEVEL so a test outside this module can reset and read it, on the
#: `inc-glm53f-028` precedent. `-035b` authors its own separate counters.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters. Called at the start of each declared test case."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


def doubling_stages(chunk: int) -> int:
    """Matmul stages the terminating Neumann series needs for a ``chunk``-wide tile.

    ``log2(chunk)``, because stage ``j`` holds the partial sum of the first
    ``2**j`` terms and ``N**chunk == 0`` makes term ``chunk`` onwards vanish.
    Read by the kernel at trace time and by the test, so the count is written
    once.
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


# --------------------------------------------------------------------------- #
# NKI emitting helpers. Plain functions, shared by both kernel entries, so that
# stage 3 has one body to read and to review.
# --------------------------------------------------------------------------- #


def _sbuf(rows, cols):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)


def _psum(rows, cols):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


def _emit_transpose(dst, src, rows, cols):
    """``dst = src^T`` for a ``[rows, cols]`` source, through PSUM.

    The PSUM destination is load-bearing rather than incidental. On this image a
    ``nc_transpose`` whose destination is in SBUF runs on the **Vector** engine
    and asserts above ``[32, 32]``; giving it a PSUM destination routes it to the
    **tensor** engine, which serves the full ``128``. Measured both ways, and
    both this route and the equivalent matmul against the identity reproduce a
    ``128``-wide transpose at exactly zero error. Every transpose in this module
    is at least one axis wider than ``32``, so all of them take this route.
    """
    ps = _psum(cols, rows)
    nisa.nc_transpose(dst=ps, data=src)
    nisa.tensor_copy(dst=dst, src=ps)


def _emit_l2_normalise(dst, src, rows, cols):
    """``dst = src / sqrt(sum_c src**2 + L2_NORM_EPS)``, epsilon inside the root.

    The reduction is along the FREE axis, so ``nl.sum(..., axis=1)`` serves it,
    and the reciprocal square root broadcasts back along the free axis from a
    ``[rows, 1]`` tile -- which is what ``nisa.tensor_scalar`` does with
    ``operand0``. No partition-axis reduction and no token loop.
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
    ``lower_inclusive_ones @ gk``, which is exactly the inclusive cumulative sum.
    This is the repository's own "reduce along the partition axis with a matmul"
    idiom, used here for a prefix sum rather than for a total.
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

    ``s`` holds the partial sum and ``n`` holds ``N**(2**j)``; ``nt`` holds
    ``n``'s transpose so that every product is a ``stationary^T @ moving``
    without a transpose inside the loop. The loop runs :func:`doubling_stages`
    ``= log2(chunk)`` times. **It does not walk tokens**: each stage squares the
    current power and doubles the number of series terms already summed.

    Every scratch tile is allocated once, before the loop, on the
    ``sinkhorn.py`` reasoning that allocating inside would ask for one live PSUM
    tile per unrolled stage where two suffice.
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
    """Stage 3: ``w``, ``u`` and ``kg``. The single body both entries emit.

    ``u = T @ (beta * v)`` and ``w = T @ (beta * k * exp(gc))``, each one matmul
    against the inverse ``T`` this function is handed -- which is why perturbing
    that inverse must move ``u``, and why the increment's discriminating test can
    read the response. ``kg[t] = k[t] * exp(gc[C - 1] - gc[t])`` takes the
    chunk's last gate row by a matmul against the row-selector constant, so no
    partition-axis broadcast is needed.
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

    Shared by both kernel entries, so the standalone stage-3 entry derives its
    normalised key and its cumulative gate exactly as the combined entry does.
    """
    _emit_l2_normalise(k_sb, k_raw, chunk, kdim)
    _emit_gate_cumsum(gc_sb, gk_sb, triu_sb, chunk, kdim)
    nisa.activation(dst=egc_sb, data=gc_sb, op=nl.exp)


# --------------------------------------------------------------------------- #
# Kernel entries
# --------------------------------------------------------------------------- #


@nki.jit
def kda_intra_chunk_kernel(
    q_hbm, k_hbm, v_hbm, beta_hbm, gk_hbm, triu_hbm, eye_hbm, mask_lower_hbm,
    last_row_hbm,
):
    """Stages 1 to 3 for every chunk, in one dispatch.

    Shapes: ``q``, ``k``, ``gk`` are ``[NC, C, K]``; ``v`` is ``[NC, C, V]``;
    ``beta`` is ``[NC, C, 1]``; the four constants are ``[C, C]``.

    The chunk loop is ``nl.affine_range`` because stages 1 to 3 are entirely
    chunk-local -- there is no carry between chunks, and choosing the parallel
    range over the sequential one asserts exactly that. Stage 4's carry is
    `-035b`'s, and it is the reason that block will need a different range.
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
    """Stage 3 alone, **taking the inverse as an argument**.

    Upstream's own boundary rather than an invented one: ``recompute_w_u_fwd``
    (``kda.py:960``) takes the inverted matrix the same way. The body is
    :func:`_emit_stage3`, identical to the one the combined entry emits, so
    there is one implementation of stage 3 and not two.

    Because ``u = T @ (beta * v)`` is linear in ``T``, scaling a row of the
    supplied inverse scales the same row of ``u``. An entry that re-derived
    ``w`` / ``u`` by walking tokens would ignore its ``a_inv`` argument and
    return the same values for both calls, which is the reading the increment's
    discriminating test takes.
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


# --------------------------------------------------------------------------- #
# Gate and seam
# --------------------------------------------------------------------------- #


def _require_admissible(
    n_chunks: int, chunk: int, kdim: int, vdim: int, gate_abs_max: float
) -> None:
    """Raise unless this kernel serves the input. Each problem names its cause.

    Inadmissibility raises rather than falling back: a torch fallback for
    kernel-class work is a P13 defect, so a caller with an unserved geometry
    learns that instead of silently receiving the oracle.
    """
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
    """Is the NKI route available *and* admissible for this input?

    Two independent conditions, deliberately not merged: ``can_run_kernel``
    answers "is there a device or a simulator", :func:`_require_admissible`
    answers "does this kernel accept these extents and this gate range".
    """
    _require_admissible(n_chunks, chunk, kdim, vdim, gate_abs_max)
    return can_run_kernel(reference)


def kda_intra_chunk(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gk: Tensor
) -> IntraChunkOutputs:
    """The seam the route predicate counts. Upstream's stages 1 to 3.

    Args:
        q, k, gk: ``[NC, C, K]`` fp32. ``gk`` is the **per-key-channel** log
            gate, not yet accumulated -- this function's kernel forms the
            cumulative sum.
        v: ``[NC, C, V]`` fp32.
        beta: ``[NC, C]`` fp32, one scalar per token, as upstream's stage 1 and
            stage 3 both read it.

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

    gate_abs_max = float(gk.float().cumsum(dim=1).abs().max().item())
    if not can_run_intra_chunk(q, n_chunks, chunk, kdim, vdim, gate_abs_max):
        _COUNTERS.torch_fallback += 1
        logger.debug(
            "kda_intra_chunk: NKI route unavailable, using the torch path "
            "(oracle only, never the shipped path)"
        )
        return kda_intra_chunk_torch_oracle(q, k, v, beta, gk)

    consts = chunk_constants(chunk, device=q.device, dtype=q.dtype)
    _COUNTERS.nki_dispatch += 1
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


# --------------------------------------------------------------------------- #
# Torch oracle -- the CPU reference, never the shipped path
# --------------------------------------------------------------------------- #


def kda_intra_chunk_torch_oracle(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gk: Tensor
) -> IntraChunkOutputs:
    """Stages 1 to 3 in torch, independently of how the kernel forms them.

    Independence is the point, on the ``sinkhorn.py`` precedent's rule that an
    oracle must not mirror the kernel it checks. Three differences are
    deliberate:

    * the gate difference is evaluated **directly** as ``exp(gc[t] - gc[j])``,
      where the kernel factorises it into ``exp(gc[t]) * exp(-gc[j])`` so that
      one matmul can contract the channel axis. That also makes this reference
      the better-conditioned of the two, which is what
      :data:`GATE_CUMSUM_ABS_LIMIT` exists to keep honest;
    * the inverse comes from ``torch.linalg.solve_triangular`` -- forward
      substitution -- where the kernel sums a terminating Neumann series by
      doubling, and where upstream uses a third route again, blocked 16x16
      substitution;
    * the two products are formed by explicit broadcast sums rather than by
      transposes and matmuls.
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
    """``I + A`` from the same inputs, for the inverse-correctness reading.

    Separate from :func:`kda_intra_chunk_torch_oracle` so a test can multiply the
    kernel's own returned inverse by a reference-built ``(I + A)`` without
    depending on any other reference value -- in an ``X . X**-1 == I`` reading the
    reference's own inverse plays no part at all.
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

    Read by the acceptance driver to prove the kernel under test is authored
    here rather than imported from the substrate.
    """
    func = getattr(kda_intra_chunk_kernel, "func", None)
    target = func if func is not None else kda_intra_chunk_kernel
    return target.__module__, target.__qualname__


def stage3_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the stage-3 entry this module authors."""
    func = getattr(kda_stage3_kernel, "func", None)
    target = func if func is not None else kda_stage3_kernel
    return target.__module__, target.__qualname__


# =========================================================================== #
# `inc-glm53f-035b` -- upstream's stages 4 and 5: the state carried ACROSS
# chunks, and the output.
#
# Everything below is PURELY ADDITIVE. Not one line above it moves, which is
# deliberate on two grounds: the intra-chunk increment's evidence record cites
# this file by line, and a reviewer of this increment should be able to read its
# whole surface as one contiguous block of added lines.
#
# The two increments share this file and share the emitting helpers above, and
# they share NOTHING ELSE. Separate kernel entries, separate seams, separate
# counters, separate constants, separate admissibility checks -- on the
# `-040`/`-041` precedent, where each increment's counted value is its own and
# neither reads the other's.
#
# THE OUTPUT THIS KERNEL PRODUCES IS NOT ``o = H q``. That is the sequential
# formula and it belongs to the oracle. The chunked output is the gate-decayed
# inter-chunk part ``qg @ h_chunk`` PLUS the intra-chunk part ``Aqk @ v_new``.
# The two forms agree numerically; measuring that agreement is the whole point
# of this increment's second conjunct, so the kernel must not be written in the
# oracle's form.
# =========================================================================== #


class InterChunkOutputs(NamedTuple):
    """Stages 4 and 5, one field per value a downstream block or a test reads.

    ``final_state`` is ``[V, K]`` -- the SEQUENTIAL scan's own orientation, so
    that the conjunct comparing them needs no transpose on either side and no
    reader has to reconcile two conventions in the same assertion. The kernel
    carries the state transposed internally for a reason given on
    :func:`kda_inter_chunk_kernel`, and it pays the one transpose at the end.

    ``v_new`` is a side output: nothing here consumes it, and it exists so that
    stage 4's derivation of it FROM ``u`` is readable at rung 1 rather than only
    by reading the kernel body.
    """

    o: Tensor
    final_state: Tensor
    v_new: Tensor


class SequentialOutputs(NamedTuple):
    """What the sequential oracle returns: flat ``o`` and the final state."""

    o: Tensor
    final_state: Tensor


class InterChunkConstants(NamedTuple):
    """The host-built constants the inter-chunk entry takes.

    Three, against the intra-chunk entry's four, and none of them is shared: the
    two entries take their own constants so that neither block's constant set
    can be changed by work on the other.
    """

    triu_ones: Tensor
    last_col: Tensor
    state_init: Tensor


@dataclass
class _InterDispatchCounters:
    """This increment's OWN counted route reading. It never reads `-035a`'s.

    Same two-counter shape and same per-dispatch rule as the intra-chunk pair
    above -- ``nki_dispatch`` counts ``wrap_nki`` dispatches, ``torch_fallback``
    counts entries into the torch path -- and a separate object, because the plan
    requires each increment's counted value to be its own.

    The per-dispatch rule is what makes the reading discriminating here too: the
    chunk loop is INSIDE the kernel, so a host-side loop that dispatched once per
    chunk would read the chunk count rather than ``1``.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: MODULE-LEVEL so a test outside this module can reset and read it. Separate
#: object from :data:`_COUNTERS`; the two are never aliased.
_INTER_COUNTERS = _InterDispatchCounters()


def reset_inter_dispatch_counters() -> None:
    """Zero both inter-chunk counters. Called before each declared call."""
    _INTER_COUNTERS.nki_dispatch = 0
    _INTER_COUNTERS.torch_fallback = 0


def inter_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last inter-chunk reset."""
    return _INTER_COUNTERS.nki_dispatch, _INTER_COUNTERS.torch_fallback


def inter_chunk_constants(
    chunk: int,
    kdim: int,
    vdim: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> InterChunkConstants:
    """Build the constants the inter-chunk entry takes.

    * ``triu_ones[s, t] = 1 for s <= t`` -- as a matmul **stationary** operand its
      transpose is the lower-inclusive ones matrix, so ``triu_ones^T @ gk`` is the
      inclusive cumulative gate along tokens WITHIN the chunk. Same construction
      as the intra-chunk entry's, built separately on purpose.
    * ``last_col[t, 0] = 1 for t == chunk - 1`` -- as a stationary operand's
      moving partner it selects the chunk's LAST cumulative-gate row and lands it
      as a ``[K, 1]`` column, which is the shape the per-key-channel state decay
      needs. One matmul, no transpose: this is why the constant is a column here
      where the intra-chunk entry's row-selector is a full ``[C, C]`` tile.
    * ``state_init`` -- the ``[K, V]`` zero entering state, built on the host and
      loaded, rather than zeroed on device. A declared input rather than an
      assumed device primitive, and it is the seat where a decode block that
      enters with a non-zero state would pass one.
    """
    idx = torch.arange(chunk, device=device)
    return InterChunkConstants(
        triu_ones=(idx.unsqueeze(1) <= idx.unsqueeze(0)).to(dtype),
        last_col=(idx == (chunk - 1)).to(dtype).unsqueeze(1).contiguous(),
        state_init=torch.zeros(kdim, vdim, device=device, dtype=dtype),
    )


@nki.jit
def kda_inter_chunk_kernel(
    kg_hbm, w_hbm, u_hbm, gk_hbm, q_hbm, aqk_hbm, triu_hbm, last_col_hbm,
    state_init_hbm,
):
    """Stages 4 and 5 for every chunk, in one dispatch.

    Shapes: ``kg``, ``w``, ``gk``, ``q`` are ``[NC, C, K]``; ``u`` is
    ``[NC, C, V]``; ``aqk`` is ``[NC, C, C]``; ``triu`` is ``[C, C]``;
    ``last_col`` is ``[C, 1]``; ``state_init`` is ``[K, V]``.

    THERE IS NO RAW ``v`` ARGUMENT. ``v_new`` is derived from ``u`` inside stage
    4, which is the plan's declared input contract and is itself a defence: a
    kernel that tried to re-scan tokens sequentially has nothing to scan from.

    The chunk loop is ``nl.sequential_range`` and the intra-chunk entry's is
    ``nl.affine_range``. That difference is the whole content of this increment:
    stages 1 to 3 are chunk-local, so the parallel range asserts it; stage 4
    carries a state from one chunk into the next, so the sequential range is
    required and choosing it states the dependency.

    THE STATE IS CARRIED TRANSPOSED, as ``ht`` shaped ``[K, V]``, where upstream
    carries ``[V, K]``. Two things fall out and both are why:
    ``nc_matmul(stationary=kg, moving=v_new)`` computes ``kg^T @ v_new`` and lands
    ``[K, V]`` directly, so the update needs no transpose; and the per-key-channel
    decay becomes a ``[K, 1]`` operand broadcast along the free axis, which is
    exactly what ``nisa.tensor_scalar`` does. Carried the other way round, the
    decay would need a partition-axis broadcast, which ``tensor_scalar`` does not
    do. The single transpose back to ``[V, K]`` is paid once, after the loop.
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

    # THE LOOP-CARRIED VALUE. Allocated once, before the loop, because it is the
    # one tile that must survive an iteration boundary.
    ht_sb = _sbuf(kdim, vdim)
    nisa.tensor_copy(dst=ht_sb, src=nl.load(state_init_hbm, dtype=nl.float32))

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
        # `q` arrives raw and is L2-normalised and scaled here, duplicating the
        # intra-chunk entry's own normalisation rather than importing its result,
        # because the plan declares `q` and not `qn` as this seam's input.
        # `Aqk` already carries BOTH the K**-0.5 scale and the causal mask from
        # the increment that produced it, so it is neither re-scaled nor re-masked.
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
        # chunk's last cumulative-gate row, landed as [K, 1] by ONE matmul
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

    Inadmissibility raises rather than falling back, on the same ground the
    intra-chunk check states: a torch fallback for kernel-class work is a P13
    defect.

    ONE CHECK THE INTRA-CHUNK VERSION MAKES IS DELIBERATELY ABSENT HERE. That
    one requires ``chunk`` to be a power of two, because its terminating Neumann
    series reaches exactly ``2**stage`` terms per doubling stage. This kernel
    sums no series and runs no doubling loop, so the requirement would be
    inherited rather than earned, and it is not made.

    ``gate_abs_max`` is the largest absolute CHUNK-LOCAL cumulative gate, so the
    bound it is checked against is a bound on one chunk's gate sum and not on the
    whole sequence's. See :func:`kda_inter_chunk` for why that distinction is the
    one that keeps the state carry inside fp32.
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
    """Is the NKI route available *and* admissible for this inter-chunk input?

    Two independent conditions, kept separate for the reason the intra-chunk
    version states: ``can_run_kernel`` answers "is there a device or a
    simulator", :func:`_require_inter_admissible` answers "does this kernel
    accept these extents and this gate range".
    """
    _require_inter_admissible(n_chunks, chunk, kdim, vdim, gate_abs_max)
    return can_run_kernel(reference)


def kda_inter_chunk(
    kg: Tensor, w: Tensor, u: Tensor, gk: Tensor, q: Tensor, aqk: Tensor
) -> InterChunkOutputs:
    """The seam THIS increment's route predicate counts. Upstream's stages 4-5.

    The argument list is the plan's declared input contract and is written in its
    declared order: ``kg``, ``w``, ``u``, ``gk``, ``q``, ``aqk``. **There is no
    raw ``v``.** ``kg``, ``w``, ``u`` and ``aqk`` are the intra-chunk seam's
    returns; ``gk`` and ``q`` are the same raw inputs that seam took.

    Args:
        kg: ``[NC, C, K]`` fp32, the gated key.
        w: ``[NC, C, K]`` fp32, the WY row factor.
        u: ``[NC, C, V]`` fp32, the WY value factor. ``v_new`` is derived from
            this inside stage 4.
        gk: ``[NC, C, K]`` fp32, the per-key-channel log gate, NOT yet
            accumulated -- the kernel forms the chunk-local cumulative sum.
        q: ``[NC, C, K]`` fp32, raw. Normalised and scaled inside the kernel.
        aqk: ``[NC, C, C]`` fp32, already carrying the scale and the causal mask.

    Returns:
        :class:`InterChunkOutputs`, whose ``final_state`` is ``[V, K]``.

    Raises:
        ChunkedRecurrenceError: on a rank mismatch, a shape disagreement, or an
            inadmissible geometry or gate range.

    THE OVERFLOW CLASS THE INTRA-CHUNK BLOCK DISCLOSED IS ANSWERED BY
    CONSTRUCTION HERE, NOT BY A WIDER BOUND. A state carry written over the
    whole sequence's cumulative gate would put ``exp`` of a quantity that grows
    with the token count into the recurrence. This one decays by
    ``exp(gc[C - 1])`` where ``gc`` is re-referenced from zero inside every
    chunk -- upstream's own convention -- so the exponent is bounded by ONE
    chunk's gate sum however long the sequence is, and
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

    gate_abs_max = float(gk.float().cumsum(dim=1).abs().max().item())
    if not can_run_inter_chunk(q, n_chunks, chunk, kdim, vdim, gate_abs_max):
        _INTER_COUNTERS.torch_fallback += 1
        logger.debug(
            "kda_inter_chunk: NKI route unavailable, using the torch path "
            "(oracle only, never the shipped path)"
        )
        return kda_inter_chunk_torch_oracle(kg, w, u, gk, q, aqk)

    consts = inter_chunk_constants(chunk, kdim, vdim, device=kg.device, dtype=kg.dtype)
    _INTER_COUNTERS.nki_dispatch += 1
    o, final_state, v_new = wrap_nki(kda_inter_chunk_kernel)(
        kg_hbm=kg,
        w_hbm=w,
        u_hbm=u,
        gk_hbm=gk,
        q_hbm=q,
        aqk_hbm=aqk,
        triu_hbm=consts.triu_ones,
        last_col_hbm=consts.last_col,
        state_init_hbm=consts.state_init,
    )
    return InterChunkOutputs(o=o, final_state=final_state, v_new=v_new)


def kda_inter_chunk_torch_oracle(
    kg: Tensor, w: Tensor, u: Tensor, gk: Tensor, q: Tensor, aqk: Tensor
) -> InterChunkOutputs:
    """Stages 4-5 in torch. THE FALLBACK PATH ONLY, never the shipped path.

    Present for the same reason the intra-chunk oracle is: so a host with no
    device or simulator can exercise the seam's contract. It is NOT the reference
    any conjunct of this increment compares against -- that reference is
    :func:`kda_sequential_torch_oracle`, which shares no formula with this
    function and no formula with the kernel.
    """
    kg32, w32, u32 = kg.float(), w.float(), u.float()
    gk32, q32, aqk32 = gk.float(), q.float(), aqk.float()
    n_chunks, chunk, kdim = kg32.shape
    vdim = u32.shape[2]

    qn = q32 / torch.sqrt((q32 * q32).sum(-1, keepdim=True) + L2_NORM_EPS)
    qn = qn * (float(kdim) ** -0.5)
    gc = gk32.cumsum(dim=1)

    ht = torch.zeros(kdim, vdim, device=kg32.device, dtype=kg32.dtype)
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
    """The SEQUENTIAL delta rule, one token at a time, over FLAT inputs.

    This is the reference this increment's conjuncts 1 to 3 measure against, and
    its independence from the kernel is the point. It shares no structure with
    the chunked path at all: it materialises no ``A``, forms no inverse, computes
    no ``w`` / ``u`` / ``kg`` / ``Aqk``, and never groups tokens. It walks them.
    Agreement between it and the kernel is therefore a statement about the
    chunking being associativity-correct, which is exactly what conjunct 1 claims.

    Args:
        q, k, gk: ``[T, K]`` fp32, flat over tokens -- no chunk axis.
        v: ``[T, V]`` fp32.
        beta: ``[T]`` fp32.

    Returns:
        :class:`SequentialOutputs` with ``o`` shaped ``[T, V]`` and
        ``final_state`` shaped ``[V, K]``.

    Three values in the rule below are load-bearing for the comparison and each
    is a declared value rather than an inherited default: the state decays **per
    key channel** rather than by one scalar, which is the KDA-specific step; the
    L2-normalisation epsilon is :data:`L2_NORM_EPS` and sits **inside** the
    square root; and ``q`` carries ``K ** -0.5``. The order of the four steps is
    load-bearing too -- the state is decayed FIRST, and the delta then reads the
    DECAYED state.
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

    Read by the acceptance driver to prove the kernel under test is authored here
    rather than imported from the substrate.
    """
    func = getattr(kda_inter_chunk_kernel, "func", None)
    target = func if func is not None else kda_inter_chunk_kernel
    return target.__module__, target.__qualname__
