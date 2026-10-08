# SPDX-License-Identifier: Apache-2.0
"""A world-size-1 stand-in for a 64-rank tensor-parallel all-reduce, and the tiny worlds run under it.

What it is for: the row-parallel sites (``collective_policy.reduce_row_parallel``) only
reduce when a coordinator exists, so the tiny fixtures, which run at world size 1, never
exercise the wire dtype. :class:`EmulatedTPGroup` is injected as the coordinator. It is
handed the full sum (one rank holds every head and every expert here), so it builds ``N``
fp32 partials that add up to that sum, rounds each to the wire dtype it was handed, and
reduces them in that dtype with the order of the device's algorithm:

* ``rdh``: recursive halving/doubling, which the device profile names for the 8 MiB
  prefill AllReduce (``calib/allreduce_measured.json``: algorithm ``RDH``). Every element
  is a balanced binary tree over the ranks, rounded to the wire dtype at every level.
* ``ring``: a sequential sum around the ranks, rounded at every step (a stress order).

The partials are synthetic. Rank ``r`` gets ``S/N + kappa * rms_row(S) * g_r`` with
``g`` standard normal, made zero-sum over the ranks, so every partial has a spread of
``kappa`` times the row's RMS whatever the element's own value (an element whose parts
cancel gets the same absolute rounding noise as one whose parts add, which is the case
real partials present). ``kappa = 1/sqrt(N)`` is ``N`` independent contributions; larger
``kappa`` is cancellation.

Under ``bf16`` the policy rounds the partial before the coordinator sees it, while on the
device each rank rounds only its own fp32 share. So :func:`injected` also wraps the
model's entry point and hands the group each site's fp32 partial first: the synthetic
partials are cut from that exact sum, and every deviation is measured against it. A site
that hands a partial already in bfloat16 (the CPU routed-expert sum) has no exact sum;
there a uniform half-ulp dither restores the unrounded sum's spread, so that first
rounding is not under-counted.

The two tiny worlds are the ones ``gen_numerics_fixture`` digests: the DSA root (three
sparse-attention layers, two dense MLP halves and one 16-expert top-8 MoE half, a seeded
random head) and the KDA world (two real KDA attention modules on grid weights).

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
        python test/vllm_neuron/model/glm5_next/tiny/allreduce_emulation.py --json out.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import pathlib
import sys

import torch

from vllm_neuron.model.glm5_next.collective_policy import DTYPE_ENV

#: The TP degree the GLM-5.3-Flash prefill runs at today.
RANKS = 64
#: ``N`` independent contributions of similar size.
KAPPA_INDEPENDENT = 1.0 / math.sqrt(RANKS)
#: Row-parallel reductions per decoder layer: the attention output projection, then the
#: feed-forward half (``collective_policy.RowParallelSite``; KDA and MLA are per layer type).
ROW_PARALLEL_SITES_PER_LAYER = 2

#: The mantissa bits bfloat16 stores (the leading one is implicit).
BF16_MANTISSA_BITS = 7

#: The rms error of rounding a value to bfloat16, over the value's rms. Round to nearest
#: leaves an error uniform over one ulp (variance ``ulp**2 / 12``); with 7 stored mantissa
#: bits ``ulp / |x| = 2**-7 / m`` for a mantissa ``m`` in ``[1, 2)``, and for a mantissa
#: spread log-uniformly over that range ``E[1 / m**2] = 3 / (8 ln 2)``.
BF16_ROUNDING_REL_RMS = 2.0**-BF16_MANTISSA_BITS * math.sqrt(
    3.0 / (8.0 * math.log(2.0)) / 12.0
)


def sites_per_forward() -> int:
    """Row-parallel reductions per forward of the DSA root: two per decoder layer."""
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

    return ROW_PARALLEL_SITES_PER_LAYER * tiny.STACK_LAYERS


def bf16_rdh_rel_rms_bracket(ranks: int, kappa: float) -> tuple[float, float]:
    """The rms error of the bf16 RDH sum of :func:`synthetic_partials`, over the sum's rms.

    Every value the reduction rounds carries an independent error of
    :data:`BF16_ROUNDING_REL_RMS` times its own size, so the error's mean square is that
    constant squared times the summed mean square of the rounded values. In units of the
    row's mean square, the ``ranks`` partials sum to ``ranks * kappa**2 + 1 / ranks``,
    and the ``ranks / m`` sums of ``m = 2**k`` partials at tree level ``k`` sum to
    ``kappa**2 * ranks * (ranks - m) / (ranks - 1) + m / ranks`` (zero-sum spread, plus
    the mean). A partial is rounded from fp32 and takes the constant itself. A tree sum
    adds two bf16 values, so its exact value has only a few bits past the bf16 mantissa
    and the error is coarser than uniform: with one extra bit it is 0 or half an ulp
    (variance ``ulp**2 / 8``, 1.5 times the uniform one). The bracket takes the tree's
    factor between 1 and 1.5.

    Raises:
        ValueError: ``ranks`` is not a power of two (RDH needs one).
    """
    levels = int(math.log2(ranks))
    if 2**levels != ranks:
        raise ValueError(f"rdh needs a power-of-two rank count, got {ranks}")
    inputs = ranks * kappa**2 + 1.0 / ranks
    tree = sum(
        kappa**2 * ranks * (ranks - 2**k) / (ranks - 1) + 2**k / ranks
        for k in range(1, levels + 1)
    )
    return (
        BF16_ROUNDING_REL_RMS * math.sqrt(inputs + tree),
        BF16_ROUNDING_REL_RMS * math.sqrt(inputs + 1.5 * tree),
    )


def bf16_ulp(value: torch.Tensor) -> torch.Tensor:
    """The bfloat16 unit in the last place at each element of ``value`` (fp32)."""
    mag = value.abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(mag)) - BF16_MANTISSA_BITS)


def synthetic_partials(
    total: torch.Tensor, ranks: int, kappa: float, gen: torch.Generator
) -> torch.Tensor:
    """``[ranks, *total.shape]`` fp32 partials whose sum is ``total`` (to fp32 rounding)."""
    flat = total.reshape(-1, total.shape[-1]).to(torch.float32)
    rms = flat.pow(2).mean(dim=-1, keepdim=True).sqrt()
    noise = torch.randn((ranks, *flat.shape), generator=gen, dtype=torch.float32)
    noise = (noise - noise.mean(dim=0, keepdim=True)) * math.sqrt(ranks / (ranks - 1))
    parts = flat.unsqueeze(0) / ranks + kappa * rms.unsqueeze(0) * noise
    return parts.reshape(ranks, *total.shape)


def reduce_in(parts: torch.Tensor, order: str) -> torch.Tensor:
    """Sum ``parts`` over dim 0 in ``parts.dtype``, rounding after every add."""
    if order == "rdh":
        if parts.shape[0] & (parts.shape[0] - 1):
            raise ValueError(
                f"rdh needs a power-of-two rank count, got {parts.shape[0]}"
            )
        while parts.shape[0] > 1:
            parts = parts[0::2] + parts[1::2]
        return parts[0]
    if order == "ring":
        acc = parts[0].clone()
        for r in range(1, parts.shape[0]):
            acc = acc + parts[r]
        return acc
    raise ValueError(f"unknown order {order!r}")


class EmulatedTPGroup:
    """The injected coordinator: ``N``-rank reduction arithmetic on synthetic partials.

    Args:
        wire: the policy value the arm runs under (``fp32`` or ``bf16``). Under
            ``fp32`` a tensor handed in bfloat16 is passed through: on the device every
            site's partial is fp32 (the as-built prefill graph's 90 all-reduces are all
            ``f32[1024,4096]``) and the reduced sum is cast to the bfloat16 carrier right
            after, while the CPU MoE route returns its sum already rounded to bfloat16,
            which is that same result.
        ranks: the emulated tensor-parallel degree.
        kappa: spread of each partial, in units of its row's RMS.
        order: ``rdh`` or ``ring``.
        seed: seeds the partials and the dither.
        active_sites: when given, only calls whose index modulo
            :func:`sites_per_forward` is in this set are emulated; the others return the
            tensor as handed, which is what the as-built fp32 wire returns after the
            site's bfloat16 cast. Used to see how the deviation grows with the number of
            sites that carry it.

    :func:`injected` sets :attr:`exact` to the site's fp32 partial before the policy
    rounds it; the next :meth:`all_reduce` takes it.
    """

    def __init__(
        self,
        wire: str,
        ranks: int = RANKS,
        kappa: float = KAPPA_INDEPENDENT,
        order: str = "rdh",
        seed: int = 0,
        active_sites=None,
    ) -> None:
        """Seed the partials and start the per-call records; arguments as above."""
        if wire not in ("fp32", "bf16"):
            raise ValueError(f"wire must be fp32 or bf16, got {wire!r}")
        self.wire = wire
        self.world_size = ranks
        self.ranks = ranks
        self.kappa = kappa
        self.order = order
        self.active_sites = None if active_sites is None else set(active_sites)
        self._period = None if active_sites is None else sites_per_forward()
        self._gen = torch.Generator().manual_seed(seed)
        self.calls = 0
        self.passed_through = 0
        self.dtypes: list[torch.dtype] = []
        #: The fp32 partial of the site being reduced, or ``None``: see :func:`injected`.
        self.exact: torch.Tensor | None = None
        #: Per emulated call: rms of (result - the partials' sum) over the sum's rms,
        #: and the same in bf16 ulps of the sum.
        self.injected_rel_rms: list[float] = []
        self.injected_ulp_rms: list[float] = []

    def all_reduce(self, tensor: torch.Tensor) -> None:
        """Reduce ``tensor`` in place, as ``GroupCoordinator.all_reduce`` does.

        The sum is cut into this group's synthetic partials, each rounded to
        ``tensor``'s dtype, and reduced in that dtype in the group's order. The error
        against the exact sum is recorded per call.
        """
        exact, self.exact = self.exact, None
        index = self.calls
        self.calls += 1
        self.dtypes.append(tensor.dtype)
        if self.wire == "fp32" and tensor.dtype is not torch.float32:
            self.passed_through += 1
            return
        if (
            self.active_sites is not None
            and index % self._period not in self.active_sites
        ):
            return
        handed = tensor.detach().to(torch.float32)
        total = handed
        if tensor.dtype is torch.bfloat16 and exact is not None:
            # The policy rounded this fp32 partial; reduce the exact sum instead.
            if not torch.equal(exact.to(torch.bfloat16), tensor):
                raise RuntimeError("the recorded partial is not the one handed in")
            total = exact
        elif tensor.dtype is torch.bfloat16:
            dither = (
                torch.rand(handed.shape, generator=self._gen, dtype=torch.float32) - 0.5
            )
            total = handed + dither * bf16_ulp(handed)
        parts = synthetic_partials(total, self.ranks, self.kappa, self._gen).to(
            tensor.dtype
        )
        reduced = reduce_in(parts, self.order)
        delta = reduced.to(torch.float32) - total
        scale = total.pow(2).mean().sqrt().clamp_min(1e-30)
        self.injected_rel_rms.append(float(delta.pow(2).mean().sqrt() / scale))
        self.injected_ulp_rms.append(
            float((delta / bf16_ulp(total)).pow(2).mean().sqrt())
        )
        tensor.copy_(reduced)


@contextlib.contextmanager
def injected(group, wire: str | None):
    """``group`` as the TP coordinator and ``wire`` as the policy, restored on exit.

    The model's entry point is wrapped so an :class:`EmulatedTPGroup` sees each site's
    fp32 partial before the policy rounds it (:attr:`EmulatedTPGroup.exact`).
    """
    from vllm_neuron.model.glm5_next import model_fp8

    saved_resolver = model_fp8._resolve_tp_group
    saved_reduce = model_fp8.reduce_row_parallel
    saved_env = os.environ.get(DTYPE_ENV)

    def reduce_with_the_exact_partial(partial, *, site, group):
        if isinstance(group, EmulatedTPGroup) and partial.dtype is torch.float32:
            group.exact = partial.detach().clone()
        return saved_reduce(partial, site=site, group=group)

    model_fp8._resolve_tp_group = lambda: group
    model_fp8.reduce_row_parallel = reduce_with_the_exact_partial
    if wire is None:
        os.environ.pop(DTYPE_ENV, None)
    else:
        os.environ[DTYPE_ENV] = wire
    try:
        yield group
    finally:
        model_fp8._resolve_tp_group = saved_resolver
        model_fp8.reduce_row_parallel = saved_reduce
        if saved_env is None:
            os.environ.pop(DTYPE_ENV, None)
        else:
            os.environ[DTYPE_ENV] = saved_env


@contextlib.contextmanager
def recording_routes(sink: list):
    """Every routed-expert selection, ``[T, k]`` sorted ids, and the router scores."""
    from vllm_neuron.model.glm5_next import model_fp8

    cls = model_fp8.Glm5NextRoutedExperts
    original = cls.route_tokens

    def wrapper(self, *args, **kwargs):
        logits, index, affinities = original(self, *args, **kwargs)
        sink.append(
            (
                index.detach().long().sort(dim=-1).values.clone(),
                logits.detach().float().clone(),
            )
        )
        return logits, index, affinities

    cls.route_tokens = wrapper
    try:
        yield sink
    finally:
        cls.route_tokens = original


# ── the DSA root world ────────────────────────────────────────────────────────


def dsa_arm(
    batch: int, steps: int, *, group=None, wire: str | None = None, fed=None
) -> dict:
    """Prefill ``batch`` prompts (every position's logits), then ``steps`` decodes.

    ``fed`` is the token feed per step (teacher forcing); ``None`` feeds the arm's own
    greedy tokens. Returns the logits rows, the greedy ids and the routing selections.
    """
    from test.vllm_neuron.model.glm5_next.tiny import gen_numerics_fixture as gen
    from test.vllm_neuron.model.glm5_next.tiny import (
        test_tiny_glm5next_batch_decode as dsa,
    )

    routes: list = []
    prefill_logits: list[torch.Tensor] = []
    original_step = dsa._step

    def every_position(world, rows, input_ids, *, cached, sampling, **kw):
        # The world's prefill samples the last position only; every position is read
        # here, and the last row is still the one the world takes its token from.
        positions = list(range(int(input_ids.shape[0])))
        logits = original_step(
            world, rows, input_ids, cached=cached, sampling=positions, **kw
        )
        prefill_logits.append(logits.detach().clone())
        return logits

    with injected(group, wire), recording_routes(routes):
        dsa._step = every_position
        try:
            world = gen.dsa_world(batch, steps)
        finally:
            dsa._step = original_step
        ids = [list(world.first)]
        tokens = list(world.first) if fed is None else list(fed[0])
        decode_logits = []
        for step in range(steps):
            logits = dsa._step(
                world,
                list(range(batch)),
                torch.tensor(tokens),
                cached=[n + step for n in world.lengths],
                sampling=list(range(batch)),
            )
            decode_logits.append(logits.detach().clone())
            own = logits.argmax(-1).tolist()
            ids.append(own)
            tokens = own if fed is None else list(fed[step + 1])
    return {
        "prefill_logits": prefill_logits,
        "decode_logits": decode_logits,
        "ids": ids,
        "routes": routes,
        "calls": getattr(group, "calls", 0),
        "dtypes": sorted({str(d) for d in getattr(group, "dtypes", [])}),
    }


def logit_deviation(arm: dict, ref: dict) -> dict:
    """Max/mean abs and rel logit deviation, top-1 flips, and routing flips vs ``ref``."""
    a = torch.cat(arm["prefill_logits"] + arm["decode_logits"]).float()
    r = torch.cat(ref["prefill_logits"] + ref["decode_logits"]).float()
    delta = (a - r).abs()
    row_scale = r.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    rel = delta / row_scale
    top2 = r.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    flips = int((a.argmax(-1) != r.argmax(-1)).sum())
    # A row can change its argmax only if its top-2 margin is within twice its largest move.
    at_risk = int((margin <= 2.0 * delta.amax(dim=-1)).sum())
    # Each move in units of the bf16 ulp of its row's largest logit.
    row_ulps = delta / bf16_ulp(row_scale)
    route_tokens = route_flipped_tokens = route_slots = route_slot_flips = 0
    route_tokens_at_risk = 0
    gap_min = float("inf")
    router_max_abs = 0.0
    for (ia, la), (ir, lr) in zip(arm["routes"], ref["routes"]):
        diff = ia != ir
        route_tokens += int(ir.shape[0])
        route_flipped_tokens += int(diff.any(dim=-1).sum())
        route_slots += int(ir.numel())
        # A slot flip: an expert in one selection and not in the other.
        for row in range(int(ir.shape[0])):
            route_slot_flips += len(set(ia[row].tolist()) - set(ir[row].tolist()))
        k = int(ir.shape[-1])
        moved = (la - lr).abs().amax(dim=-1)
        router_max_abs = max(router_max_abs, float(moved.max()))
        if lr.shape[-1] > k:
            ranked = lr.sort(dim=-1, descending=True).values
            gap = ranked[:, k - 1] - ranked[:, k]
            gap_min = min(gap_min, float(gap.min()))
            # The top-k set can change only where the k-th gap is within twice the move.
            route_tokens_at_risk += int((gap <= 2.0 * moved).sum())
    return {
        "rows": int(r.shape[0]),
        "vocab": int(r.shape[-1]),
        "logit_scale_max": float(r.abs().max()),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "max_rel": float(rel.max()),
        "mean_rel": float(rel.mean()),
        "max_row_ulps": float(row_ulps.max()),
        "mean_row_ulps": float(row_ulps.mean()),
        "top1_flips": flips,
        "top1_margin_min": float(margin.min()),
        "top1_rows_at_risk": at_risk,
        "route_tokens": route_tokens,
        "route_tokens_flipped": route_flipped_tokens,
        "route_slots": route_slots,
        "route_slot_flips": route_slot_flips,
        "route_tokens_at_risk": route_tokens_at_risk,
        "router_logit_max_abs": router_max_abs,
        "router_logit_gap_k_min": None if gap_min == float("inf") else gap_min,
    }


# ── the KDA world ─────────────────────────────────────────────────────────────


def kda_arm(batch: int, steps: int, *, group=None, wire: str | None = None) -> dict:
    """Prefill and decode the KDA world; every step's stacked attention output."""
    from test.vllm_neuron.model.glm5_next.tiny import gen_numerics_fixture as gen
    from test.vllm_neuron.model.glm5_next.tiny import (
        test_tiny_glm5next_batch_kda as kda,
    )

    outputs: list[torch.Tensor] = []
    original_step = kda._step

    def recorded(*args, **kwargs):
        out, carriers = original_step(*args, **kwargs)
        outputs.append(out.detach().clone())
        return out, carriers

    with injected(group, wire):
        kda._step = recorded
        try:
            world = gen.kda_world(batch, steps)
            for step in range(steps):
                kda._step(
                    world.runner,
                    world.banks,
                    world.layers,
                    req_ids=world.req_ids,
                    rows=world.decodes[step],
                    cached=[n + step for n in world.lengths],
                    max_query_len=1,
                )
        finally:
            kda._step = original_step
    return {"outputs": outputs, "carrier": str(world.decodes[0].dtype)}


def output_deviation(arm: dict, ref: dict) -> dict:
    """Abs, rel and rms deviation of a KDA arm's output rows from ``ref``'s."""
    a = torch.cat([o.reshape(-1, o.shape[-1]) for o in arm["outputs"]]).float()
    r = torch.cat([o.reshape(-1, o.shape[-1]) for o in ref["outputs"]]).float()
    delta = (a - r).abs()
    scale = r.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return {
        "rows": int(r.shape[0]),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "max_rel": float((delta / scale).max()),
        "mean_rel": float((delta / scale).mean()),
        "rel_rms": float(delta.pow(2).mean().sqrt() / r.pow(2).mean().sqrt()),
        "max_bf16_ulps": float((delta / bf16_ulp(r)).max()),
    }


# ── real partials: one KDA output projection at TP=64, full geometry ────────────


def _kda_rank_slice(
    model_fp8, module, leaf: str, tensor: torch.Tensor, rank: int, world: int
) -> torch.Tensor:
    """This rank's shard of a full-width KDA tensor (``test_kda_reduction``'s rule at ``world``)."""
    from test.vllm_neuron.model.glm5_next import test_kda_reduction as kr

    if leaf in kr._REPLICATED:
        return tensor.clone()
    if leaf in kr._BY_HEAD_WIDTH:
        size, dim = int(module.num_kv_heads_per_rank) * kr.KDA_HEAD_SIZE, 0
    else:
        geometry = model_fp8._shard_geometry_for(module, leaf, world)
        size, dim = int(geometry.shard_size), int(geometry.shard_dim)
    return tensor.narrow(dim, rank * size, size).contiguous().clone()


def kda_real_partials(
    world: int = RANKS, tokens: int = 17, seed: int = 20260910
) -> dict:
    """The 64 real per-rank partials of one KDA layer's ``o_proj`` and their two reductions.

    Full GLM-5.3-Flash KDA geometry (64 heads x 128, hidden 4096), one head per rank,
    ``test_kda_reduction``'s random weights and fp32 tokens. Each rank's module runs
    with no coordinator, so it returns its own partial. The as-built result is the fp32
    RDH sum cast to the bfloat16 carrier; the bf16 wire is the RDH sum of the
    bfloat16-rounded partials. The emulator is then run on the as-built sum with the
    measured spread, to check it against the real partials.
    """
    from torch import nn

    from test.vllm_neuron.model.glm5_next import test_kda_reduction as kr
    from vllm_neuron.model.glm5_next import model_fp8
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    hidden = int(text_config.hidden_size)
    torch.manual_seed(seed)
    weights = kr._full_weights(
        hidden, kr.KDA_NUM_HEADS, kr.KDA_HEAD_SIZE, kr.KDA_CONV_KERNEL_SIZE
    )
    torch.manual_seed(seed + 1)
    x = torch.randn(tokens, hidden, dtype=torch.float32)
    parts = []
    with injected(None, None):
        for rank in range(world):
            module = model_fp8.Glm5NextKDAAttention(text_config, world)
            for name, tensor in weights.items():
                value = _kda_rank_slice(model_fp8, module, name, tensor, rank, world)
                setattr(module, name, nn.Parameter(value, requires_grad=False))
            parts.append(
                module.forward(x.clone(), **kr._zero_state(module), is_prefill=True)
            )
    stacked = torch.stack(parts).to(torch.float32)
    exact = stacked.to(torch.float64).sum(dim=0)
    as_built = reduce_in(stacked, "rdh").to(torch.bfloat16).to(torch.float32)
    bf16_wire = reduce_in(stacked.to(torch.bfloat16), "rdh").to(torch.float32)
    ulp = bf16_ulp(as_built)
    row_rms = exact.pow(2).mean(dim=-1, keepdim=True).sqrt()
    kappa = float(
        (stacked.to(torch.float64) - exact.unsqueeze(0) / world)
        .pow(2)
        .mean(dim=(0, 2))
        .sqrt()
        .div(row_rms.squeeze(-1))
        .mean()
    )
    emulated = as_built.to(torch.bfloat16).clone()
    EmulatedTPGroup("bf16", ranks=world, kappa=kappa, seed=seed).all_reduce(emulated)
    emulated = emulated.to(torch.float32)

    def ulps(t: torch.Tensor) -> dict:
        d = (t - as_built) / ulp
        return {
            "rms": float(d.pow(2).mean().sqrt()),
            "max": float(d.abs().max()),
            "rel_rms": float(
                (t - as_built).pow(2).mean().sqrt() / as_built.pow(2).mean().sqrt()
            ),
        }

    own = (as_built - exact.to(torch.float32)) / ulp
    return {
        "world": world,
        "tokens": tokens,
        "hidden": hidden,
        "kappa_measured": kappa,
        "as_built_rounding_ulps_rms": float(own.pow(2).mean().sqrt()),
        "as_built_rounding_rel_rms": float(
            (as_built - exact.to(torch.float32)).pow(2).mean().sqrt()
            / exact.to(torch.float32).pow(2).mean().sqrt()
        ),
        "bf16_wire_real_partials_ulps": ulps(bf16_wire),
        "bf16_wire_emulated_ulps": ulps(emulated),
        # Against the exact sum, the quantity bf16_rdh_rel_rms_bracket predicts.
        "bf16_wire_real_partials_rel_rms_vs_exact": float(
            (bf16_wire.double() - exact).pow(2).mean().sqrt()
            / exact.pow(2).mean().sqrt()
        ),
    }


# ── the study ─────────────────────────────────────────────────────────────────


#: (label, wire, kappa, order): the arms the report's deviation table reads.
ARMS = (
    ("fp32_rdh", "fp32", KAPPA_INDEPENDENT, "rdh"),
    ("fp32_ring", "fp32", KAPPA_INDEPENDENT, "ring"),
    ("bf16_rdh", "bf16", KAPPA_INDEPENDENT, "rdh"),
    ("bf16_rdh_k0.5", "bf16", 0.5, "rdh"),
    ("bf16_ring_k0.5", "bf16", 0.5, "ring"),
    ("bf16_rdh_k1", "bf16", 1.0, "rdh"),
)


def _summary(group) -> dict:
    """One arm's group record: calls, pass-throughs, handed dtypes, mean injected error."""
    vals = group.injected_ulp_rms
    rel = group.injected_rel_rms
    return {
        "calls": group.calls,
        "passed_through": group.passed_through,
        "dtypes": sorted({str(d) for d in group.dtypes}),
        "injected_ulp_rms_mean": (sum(vals) / len(vals)) if vals else 0.0,
        "injected_rel_rms_mean": (sum(rel) / len(rel)) if rel else 0.0,
    }


def study(
    batch: int = 4,
    steps: int = 8,
    seed: int = 20261007,
    growth: bool = True,
    kda: bool = True,
) -> dict:
    """Every arm of the deviation study, teacher-forced on the world-size-1 tokens.

    The DSA root under each of :data:`ARMS` (logit and routing deviation against world
    size 1 and against the as-built ``fp32_rdh`` arm); with ``growth``, the bf16 wire on
    the first ``n`` sites of each forward; the real TP=64 partials of one KDA
    ``o_proj``; with ``kda``, the KDA world under each arm (output deviation against
    world size 1). The report's s3 tables read this record.
    """
    base = dsa_arm(batch, steps)
    fed = base["ids"]
    out = {
        "batch": batch,
        "steps": steps,
        "ranks": RANKS,
        "seed": seed,
        "base_calls": base["calls"],
        "dsa": {},
        "kda": {},
        "growth": {},
    }
    reference = None
    for label, wire, kappa, order in ARMS:
        group = EmulatedTPGroup(wire, kappa=kappa, order=order, seed=seed)
        arm = dsa_arm(batch, steps, group=group, wire=wire, fed=fed)
        if label == "fp32_rdh":
            reference = arm
        row = {
            "wire": wire,
            "kappa": kappa,
            "order": order,
            **_summary(group),
            "vs_world1": logit_deviation(arm, base),
            "vs_fp32_rdh": logit_deviation(arm, reference),
        }
        out["dsa"][label] = row
    if growth:
        # The deviation as more sites carry the bf16 wire, first site first.
        for active in range(1, sites_per_forward() + 1):
            group = EmulatedTPGroup("bf16", seed=seed, active_sites=range(active))
            arm = dsa_arm(batch, steps, group=group, wire="bf16", fed=fed)
            out["growth"][str(active)] = logit_deviation(arm, base)
    out["kda_real_partials_tp64"] = kda_real_partials()
    if kda:
        kbase = kda_arm(batch, steps)
        out["kda"]["carrier"] = kbase["carrier"]
        for label, wire, kappa, order in ARMS:
            group = EmulatedTPGroup(wire, kappa=kappa, order=order, seed=seed)
            arm = kda_arm(batch, steps, group=group, wire=wire)
            out["kda"][label] = {
                "wire": wire,
                "kappa": kappa,
                "order": order,
                **_summary(group),
                "vs_world1": output_deviation(arm, kbase),
            }
    return out


def main() -> int:
    """Run :func:`study` and write its record, with the producing command, to ``--json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=pathlib.Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    import vllm_neuron

    record = {
        "producing_command": " ".join(sys.argv),
        "vllm_neuron": vllm_neuron.__file__,
        "env": {
            k: os.environ.get(k)
            for k in (
                "NKI_SIMULATOR",
                "VLLM_NEURON_CPU_MODE",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
            )
        },
        **study(args.batch, args.steps, args.seed),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
