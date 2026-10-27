# SPDX-License-Identifier: Apache-2.0
"""The routed-expert kernel past 128 tokens, in token tiles, against a float64 oracle.

``routed_experts_mx`` runs rank 2's bank of layer 1 with 2 replica slots at 129 and 160
tokens: the 160-token call is one full 128-row tile and one of 32 rows, the 129-token
call one full tile and one of 2 rows, the second row padding. The oracle, the bank and
the two classes are those of ``test_routed_experts_mx_replicas``: the checkpoint's own
gate and expert MLP in float64 with the kernel's MXFP8 activation quantization, and the
replica choice made one expert at a time. Class M: at most the set limit of the
intermediate codes differ from the oracle's, and every partial element is within 2^-8
of the row RMS of the oracle that adopts the kernel's codes. Class S: the selection, and
each bank entry's visit count and the whole choice, equal the oracle's exactly. Router
selection takes 128 rows at a time, so the selection check reads it per 128 rows.
A separate check runs 128 and 40 rows, which fit in one tile, both as one tile and in
smaller tiles; the two must agree bit for bit.
The checkpoint directory comes from ``KIMI_K3_CHECKPOINT_DIR``. With
``VLLM_NEURON_CPU_MODE=1`` and ``NKI_SIMULATOR=1`` the kernel runs on the NKI
simulator; otherwise on ``neuron:0``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from test_routed_experts_mx_replicas import (  # noqa: F401  (pytestmark and reference are used by pytest)
    AVERAGED_ROWS,
    GATE_INPUT,
    LOGIT_STD,
    ON_DEVICE,
    OWN,
    RANK,
    RANKS,
    _adopted,
    _bank,
    _choice_line,
    _class_s,
    _code_rule,
    _entries,
    _intermediates,
    _kernel_codes,
    _lift,
    _mx8_codes,
    _routed,
    _routing,
    _rows,
    _same_choice,
    _shifts,
    _worst,
    pytestmark,
    reference,
)

from vllm_neuron import envs
from vllm_neuron.functional.moe import routed_experts_mx as rx

SLOTS = 2
#: Tokens per call and token groups: the 32-user verify rows, and the first count past one tile.
SHAPES = ((160, 2), (129, 1))
#: One-tile shapes run again in smaller tiles: (tokens, token groups, slots, tile rows).
SPLITS = ((128, 2, 2, 64), (40, 2, 3, 16))
#: The poisoned rows, one per tile; each copies the logits of the row two before it.
NAN_ROWS = {160: (6, 150), 129: (6, 128)}
#: p per case, as filed: the share of its intermediate codes an emulated correct kernel flips
#: at the device MX product noise measured on that case's own operands. None is not yet filed.
CODE_NOISE = {
    "correlated rows, Vm 160, 2 slots": 0.020699280958909255,
    "repeated tokens, Vm 160, 2 slots": 0.007823340098063153,
    "sigmoid inputs at 12, Vm 160, 2 slots": 0.0073085572984483505,
    "small near-equal rows, Vm 160, 2 slots": 0.0072318924797905824,
    "clean rows, Vm 160, 2 slots": 0.007301862312085701,
    "correlated rows, Vm 129, 2 slots": 0.01807727534436235,
    "repeated tokens, Vm 129, 2 slots": 0.006493738455365794,
    "sigmoid inputs at 12, Vm 129, 2 slots": 0.00729209775045758,
    "small near-equal rows, Vm 129, 2 slots": 0.007517757578792735,
    "clean rows, Vm 129, 2 slots": 0.007289751760924714,
}
#: R per case: the largest share of its intermediate codes that may differ from the oracle's.
CODE_LIMITS = {label: None if p is None else 3 * p for label, p in CODE_NOISE.items()}


def _kernel(ref, routed, bank, replicas, slots, expose_codes=False, token_tile=rx.TOKEN_TILE):
    base = torch.tensor([[RANK * OWN]], dtype=torch.int32)
    situ = ref.scope["_get_situ_activation_params"](ref.config)
    args = dict(beta=situ[0], linear_beta=situ[1], scaling=ref.config.routed_scaling_factor,
                replicas=replicas, expose_codes=expose_codes, replica_slots=slots, token_tile=token_tile)

    def run(routed_, bias_, base_, *bank_):
        return rx.routed_experts_mx(routed_, bias_, base_, *bank_, **args)

    if not ON_DEVICE:
        return run(routed, ref.bias, base, *bank)
    # Every call compiles one variant; they share one code object and its recompile limit.
    torch._dynamo.reset()
    compiled = torch.compile(run, backend=envs.get_compile_backend_name(), fullgraph=True,
                             dynamic=False)
    out = compiled(*[t.to("neuron:0") for t in (routed, ref.bias, base, *bank)])
    return tuple(t.cpu() for t in out) if expose_codes else out.cpu()


def _select(ref, logits):
    """Router selection over 128 rows at a time."""
    def run(logits_, bias_):
        return rx.router_select(logits_, bias_, scaling=ref.config.routed_scaling_factor)

    parts = []
    for first in range(0, logits.shape[0], 128):
        rows = logits[first:first + 128].contiguous()
        if not ON_DEVICE:
            parts.append(run(rows, ref.bias))
            continue
        torch._dynamo.reset()
        compiled = torch.compile(run, backend=envs.get_compile_backend_name(), fullgraph=True,
                                 dynamic=False)
        parts.append(tuple(t.cpu() for t in compiled(rows.to("neuron:0"), ref.bias.to("neuron:0"))))
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def _measure(ref, routed, bank, replicas, slots):
    """One case's class M reading and the kernel's visit counts against the oracle's choice."""
    tokens = routed.shape[0]
    route = _routing(ref, routed, slots)
    served = _rows(_kernel(ref, routed, bank, replicas, slots))
    out, codes, scales, visits, record = _kernel(ref, routed, bank, replicas, slots, expose_codes=True)
    partial = _rows(out)
    same = torch.equal(partial.view(torch.int32), served.view(torch.int32))
    kept = route.kept.nonzero().flatten()
    got, got_exp = _kernel_codes(codes, scales, route.visited, tokens, ref.width)
    want, want_exp = _mx8_codes(_intermediates(ref, routed, route.kept, route.ids))
    differ, step = _code_rule(got[:, kept], got_exp[:, kept], want[:, kept], want_exp[:, kept])
    h = (got.double() * torch.exp2(got_exp)[..., None]).reshape(len(route.visited), tokens, -1)
    adopted = _adopted(ref, route, h)
    return SimpleNamespace(route=route, served=served, same=same, differ=differ,
                           codes=got[:, kept].numel(), step=step,
                           worst=_worst(partial[kept], adopted[kept]),
                           choice=_same_choice(visits, record, route), visits=visits.flatten().tolist())


def _grade(ref, label, routed, replicas, slots):
    """Run one case; return its failed checks and its measurement."""
    bank = _bank([ref.stored[e] for e in _entries(slots)], ref.programs)
    m = _measure(ref, routed, bank, replicas, slots)
    index, weight = _select(ref, routed[:, ref.latent:])
    s_set, s_weight = _class_s(ref, routed[:, ref.latent:], index, weight)
    limit = CODE_LIMITS[label]
    shown = "None" if limit is None else f"{limit:.4%}"
    print(f"{label}: {len(m.route.visited)} entries visited; class M {m.differ} of {m.codes} codes "
          f"differ ({m.differ / m.codes:.4%}, limit {shown}), largest code step {m.step}, partial "
          f"worst / bound {m.worst:.4f}; class S set excess {s_set:.3f}, weight error / bound "
          f"{s_weight:.4f}; {_choice_line(m.route, slots)}; visits and whole choice as the oracle's: {m.choice}")
    failed = [name for name, ok in (
        ("test mode changed the partial", m.same),
        ("class M codes (a limit of None is not set)",
         limit is not None and m.differ <= limit * m.codes),
        ("class M bound", m.worst <= 1.0),
        ("class S selection", s_set <= 1.0 and s_weight <= 1.0),
        ("class S choice", m.choice)) if not ok]
    return failed, m


def _ties_and_takes():
    """16 experts under which rank 2's choice at 2 slots turns on ties and on the order. Rank
    2's first replicated expert ties with its other home and stays; then rank 1's, whose other
    home is rank 2, ties and stays (in the other order both would move). Rank 2 takes its
    second replica, whose home runs 3 experts to rank 2's 1. The other 9 are experts of rank
    40, which none of these meet."""
    homes = [(RANK - shift) % RANKS for shift in _shifts(SLOTS)]
    return ([RANK * OWN, homes[0] * OWN, homes[0] * OWN + SLOTS]
            + [homes[1] * OWN + 1] + [homes[1] * OWN + SLOTS + s for s in range(3)]
            + [40 * OWN + SLOTS + s for s in range(9)])


def correlated_case(ref, tokens, replicas, slots=SLOTS, seed=21):
    """Regime R1: rows near one direction with one channel 100x."""
    gen = torch.Generator().manual_seed(seed + tokens + slots)
    latent = (0.9 * torch.randn(ref.latent, generator=gen)
              + 0.1 * torch.randn(tokens, ref.latent, generator=gen))
    latent[:, ref.latent // 3] *= 100
    logits = _lift(LOGIT_STD * torch.randn(tokens, ref.experts, generator=gen), slots, gen)
    return f"correlated rows, Vm {tokens}, {slots} slots", _routed(latent, logits), replicas, slots


def repeated_case(ref, tokens, replicas):
    """Regime R2: one token repeated over every row, all choosing the same 16 experts."""
    gen = torch.Generator().manual_seed(22 + tokens)
    latent = torch.randn(1, ref.latent, generator=gen).expand(tokens, -1)
    logits = LOGIT_STD * torch.randn(1, ref.experts, generator=gen)
    logits[0] -= 8.0
    logits[0, _ties_and_takes()] = 8.0 + 2.0 * torch.rand(16, generator=gen)
    return (f"repeated tokens, Vm {tokens}, {SLOTS} slots", _routed(latent, logits.expand(tokens, -1)),
            replicas, SLOTS)


def twelve_case(ref, tokens, replicas):
    """Regime R3: rank 2's own experts and two others at +12, the rest at -12."""
    gen = torch.Generator().manual_seed(23 + tokens)
    latent = torch.randn(tokens, ref.latent, generator=gen)
    logits = torch.full((tokens, ref.experts), -GATE_INPUT)
    logits[:, RANK * OWN:(RANK + 1) * OWN] = GATE_INPUT
    others = torch.randperm(ref.experts - OWN, generator=gen)[:2]
    logits[:, torch.where(others >= RANK * OWN, others + OWN, others)] = GATE_INPUT
    return f"sigmoid inputs at 12, Vm {tokens}, {SLOTS} slots", _routed(latent, logits), replicas, SLOTS


def small_rows_case(ref, tokens, replicas):
    """Regime R4: small, near-equal rows, as an average over many unit rows gives."""
    gen = torch.Generator().manual_seed(24 + tokens)
    shared = torch.randn(ref.latent, generator=gen)
    latent = (shared + 2 ** -6 * torch.randn(tokens, ref.latent, generator=gen)) / math.sqrt(
        AVERAGED_ROWS)
    logits = _lift(LOGIT_STD * torch.randn(tokens, ref.experts, generator=gen), SLOTS, gen)
    return f"small near-equal rows, Vm {tokens}, {SLOTS} slots", _routed(latent, logits), replicas, SLOTS


def nan_case(ref, tokens, replicas):
    """Regime R5: the clean case; the test poisons one row in each tile. Each poisoned row
    copies the logits of the row two before it, so its experts stay chosen without it. No row
    chooses the first replica, whose home rank runs more experts than rank 2, so a NaN row's
    choices counted in the choice would move that replica."""
    gen = torch.Generator().manual_seed(25 + tokens)
    latent = torch.randn(tokens, ref.latent, generator=gen)
    logits = _lift(LOGIT_STD * torch.randn(tokens, ref.experts, generator=gen), SLOTS, gen, take=False)
    for row in NAN_ROWS[tokens]:
        logits[row] = logits[row - 2]
    return f"clean rows, Vm {tokens}, {SLOTS} slots", _routed(latent, logits), replicas, SLOTS


def _assert_passes(ref, case):
    failed, _ = _grade(ref, *case)
    assert not failed, f"{case[0]} fails: {failed}"


SHAPE_IDS = [f"Vm{t}" for t, _ in SHAPES]


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_correlated_rows_with_one_large_channel(reference, shape):
    _assert_passes(reference, correlated_case(reference, *shape))


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_repeated_tokens(reference, shape):
    case = repeated_case(reference, *shape)
    assert _routing(reference, case[1], SLOTS).visited == [0, OWN + 1], "the case must tie and take"
    _assert_passes(reference, case)


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_sigmoid_inputs_at_twelve(reference, shape):
    _assert_passes(reference, twelve_case(reference, *shape))


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_small_near_equal_rows(reference, shape):
    _assert_passes(reference, small_rows_case(reference, *shape))


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_nan_rows_stay_in_their_rows(reference, shape):
    label, routed, replicas, slots = nan_case(reference, *shape)
    failed, clean = _grade(reference, label, routed, replicas, slots)
    poisoned = routed.clone()
    bad = list(NAN_ROWS[routed.shape[0]])
    poisoned[bad] = float("nan")
    bank = _bank([reference.stored[e] for e in _entries(slots)], reference.programs)
    route = _routing(reference, poisoned, slots)
    assert route.visited == clean.route.visited, "the case must not move an expert"
    out, _, _, visits, record = _kernel(reference, poisoned, bank, replicas, slots, expose_codes=True)
    out = _rows(out)
    others = [row for row in range(routed.shape[0]) if row not in bad]
    clean_index, clean_weight = _select(reference, routed[:, reference.latent:])
    index, weight = _select(reference, poisoned[:, reference.latent:])
    print(f"NaN rows {bad}: visits {visits.flatten().tolist()}, oracle {route.counts.tolist()}")
    assert torch.isnan(out[bad]).all(), "a NaN row's partial is not NaN throughout"
    assert _same_choice(visits, record, route), "a NaN row's choices entered the replica choice"
    assert torch.equal(out[others].view(torch.int32), clean.served[others].view(torch.int32)), \
        "a NaN row changed another row's partial"
    assert ((index >= 0) & (index < reference.experts)).all(), "an index left its range"
    assert torch.isnan(weight[bad]).all(), "a NaN row's weights are not NaN"
    assert torch.equal(index[others], clean_index[others]) and torch.equal(
        weight[others].view(torch.int32), clean_weight[others].view(torch.int32)), \
        "a NaN row changed another row's selection"
    assert not failed, f"{label} fails: {failed}"


@pytest.mark.parametrize("split", SPLITS, ids=[f"Vm{t}-{s}slots-tiles{n}" for t, _, s, n in SPLITS])
def test_tiles_equal_one_tile(reference, split):
    tokens, replicas, slots, tile = split
    _, routed, _, _ = correlated_case(reference, tokens, replicas, slots, seed=26)
    bank = _bank([reference.stored[e] for e in _entries(slots)], reference.programs)
    visited = _routing(reference, routed, slots).visited
    one, tiled = (_kernel(reference, routed, bank, replicas, slots, expose_codes=True, token_tile=n)
                  for n in (rx.TOKEN_TILE, tile))
    served = [_kernel(reference, routed, bank, replicas, slots, token_tile=n) for n in (rx.TOKEN_TILE, tile)]
    print(f"Vm {tokens}, {slots} slots, tiles of {tile}: {len(visited)} entries visited")
    assert torch.equal(served[0].view(torch.int32), served[1].view(torch.int32)), "the partials differ"
    assert torch.equal(one[0].view(torch.int32), tiled[0].view(torch.int32)), "the test-mode partials differ"
    assert torch.equal(one[1][visited], tiled[1][visited]), "the visited experts' codes differ"
    # The test mode stores whole scale tiles, but quantization writes only the rows read here.
    exponents = [_kernel_codes(out[1], out[2], visited, tokens, reference.width)[1] for out in (one, tiled)]
    assert torch.equal(*exponents), "the visited experts' scales differ"
    assert torch.equal(one[3], tiled[3]) and torch.equal(one[4], tiled[4]), "the visits or the choice differ"
