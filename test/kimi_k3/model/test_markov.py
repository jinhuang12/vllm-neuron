# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter: the base logits and the sequential Markov picks, on two TP ranks.

Each rank loads its TP 2 slices of the draft checkpoint and of the target's LM head, then
picks 4 drafts for each of 64 random hidden blocks, and resolves two planted ties. The
fused norm-projection kernel runs on the NKI CPU simulator. Set
``KIMI_K3_DSPARK_CHECKPOINT`` to the draft directory (default
``~/models/Inferact/Kimi-K3-DSpark-Block5``) and ``KIMI_K3_CHECKPOINT`` to the target
directory (default ``~/models/moonshotai/Kimi-K3``). Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import json
import os
import socket
import types
import warnings
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from safetensors import safe_open

DRAFT = Path(
    os.environ.get(
        "KIMI_K3_DSPARK_CHECKPOINT",
        Path.home() / "models" / "Inferact" / "Kimi-K3-DSpark-Block5",
    )
)
TARGET = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)
WEIGHTS = DRAFT / "model.safetensors"
LM_HEAD = "language_model.lm_head.weight"

pytestmark = pytest.mark.skipif(
    not (WEIGHTS.is_file() and (TARGET / "model.safetensors.index.json").is_file()),
    reason=f"DSpark checkpoint at {DRAFT} or Kimi K3 checkpoint at {TARGET} not found",
)

TP = 2
VOCAB, HIDDEN, RANK = 163840, 7168, 512
BLOCKS, STEPS = 64, 4
# Blocks per call: 64 rows, so each LM-head call takes the decode kernel, as served.
GROUP = 16
# A pick that differs from the reference counts as a near-tie when the reference row
# gives it a value within this fraction of the row maximum: the base logits are bf16,
# so two values near the top can each round by 2^-8 and swap.
NEAR_TIE = 2 * 2**-8
# The base logits against the reference: the normed rows and the output each round to
# bf16, a move of at most 2^-8 of the value, and together they stay under 2^-7 of the
# largest reference value. A final norm without its gain moves them past it.
BASE_BAND = 2**-7
# The planted ties: a token in rank 1's half takes the second Markov weight's row of
# the top token in rank 0's half, and a token in rank 0's half takes it for rank 1's.
TIE_IN_RANK_1, TIE_IN_RANK_0 = VOCAB // 2 + 17, 23


def _head_file() -> Path:
    """The target checkpoint file that holds the LM head."""
    with open(TARGET / "model.safetensors.index.json") as f:
        return TARGET / json.load(f)["weight_map"][LM_HEAD]


def _whole(name: str) -> torch.Tensor:
    """A draft checkpoint tensor, whole."""
    with safe_open(WEIGHTS, framework="pt") as f:
        return f.get_tensor(name)


def _row(name: str, row: int) -> torch.Tensor:
    """One row of a draft checkpoint tensor."""
    with safe_open(WEIGHTS, framework="pt") as f:
        return f.get_slice(name)[row : row + 1][0]


def _rank(rank: int, port: int, hidden, anchors, ties, out: str) -> None:
    """One TP rank: the drafts of every block, then the picks with each planted tie."""
    import torch.distributed as dist

    from vllm_neuron.model.kimi_k3.dspark import K3DSparkForCausalLM
    from vllm_neuron.model.kimi_k3.model import _Weight
    from vllm_neuron.parallel.neuron_parallel_state import initialize_neuron_parallel_state

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank)
    with torch.device("meta"):
        draft = K3DSparkForCausalLM.from_configs(config=str(DRAFT / "config.json"), max_positions=1)
        embed = _Weight((VOCAB, HIDDEN), torch.bfloat16)
        lm_head = _Weight((VOCAB // TP, HIDDEN), torch.bfloat16)
    draft.load_weights(str(DRAFT), torch.device("cpu"))
    rows = VOCAB // TP
    with safe_open(_head_file(), framework="pt") as f:
        stored = f.get_slice(LM_HEAD)[rank * rows : (rank + 1) * rows]
    # The target as its load step leaves it: the head stacked as two [H, V / 2TP] halves.
    halves = [torch.nn.Parameter(h.t().contiguous(), requires_grad=False) for h in stored.chunk(2)]
    lm_head.weight = None
    target = types.SimpleNamespace(
        model=types.SimpleNamespace(embed_tokens=embed),
        lm_head=lm_head,
        head_weights=torch.nn.ParameterList(halves),
    )
    draft.alias_target_head(target)
    # Keep the base logits as the picks use them, one entry per call.
    base, compute = [], draft.compute_logits

    def keep(rows):
        base.append(compute(rows))
        return base[-1]

    draft.compute_logits = keep
    groups = zip(hidden.split(GROUP), anchors.split(GROUP))
    state = {"picks": torch.cat([draft.sample_drafts(h, a) for h, a in groups])}
    draft.compute_logits = compute
    state["base"] = torch.cat(base)
    w2 = draft.markov_head.markov_w2.weight
    tie_picks = []
    for anchor, source, copy in ties:
        kept = w2[copy % rows].clone()
        if copy // rows == rank:
            w2[copy % rows] = _row("markov_head.markov_w2.weight", source)
        zero = torch.zeros(1, STEPS, HIDDEN, dtype=torch.bfloat16)
        tie_picks.append(int(draft.sample_drafts(zero, torch.tensor([anchor]))[0, 0]))
        if copy // rows == rank:
            w2[copy % rows] = kept
    state["ties"] = tie_picks
    torch.save(state, os.path.join(out, f"rank{rank}.pt"))


@pytest.fixture(scope="module")
def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """64 random hidden blocks of 4 rows in bf16, and a random anchor for each."""
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(BLOCKS, STEPS, HIDDEN, generator=g).bfloat16()
    return hidden, torch.randint(0, VOCAB, (BLOCKS,), generator=g)


@pytest.fixture(scope="module")
def markov() -> tuple[torch.Tensor, torch.Tensor]:
    """Both Markov weights, whole, in fp64."""
    return tuple(_whole(f"markov_head.markov_w{i}.weight").double() for i in (1, 2))


def _clear_top(w1, w2, half: int) -> tuple[int, int]:
    """An anchor whose zero-logit pick is clear by 1 % and in ``half``; and that pick."""
    for anchor in range(VOCAB):
        bias = w2 @ w1[anchor]
        top2 = bias.topk(2)
        gap = (top2.values[0] - top2.values[1]) / top2.values[0].abs()
        if int(top2.indices[0]) // (VOCAB // 2) == half and gap > 0.01:
            return anchor, int(top2.indices[0])
    raise AssertionError(f"no anchor with a clear pick in half {half}")


@pytest.fixture(scope="module")
def ties(markov) -> list[tuple[int, int, int]]:
    """Each planted tie as (anchor, the pick's token, the token given its row)."""
    w1, w2 = markov
    (a0, top0), (a1, top1) = _clear_top(w1, w2, 0), _clear_top(w1, w2, 1)
    return [(a0, top0, TIE_IN_RANK_1), (a1, top1, TIE_IN_RANK_0)]


@pytest.fixture(scope="module")
def ranks(tmp_path_factory, inputs, ties) -> list[dict]:
    """Each rank's drafts, base logits and tie picks, from one two-rank run."""
    out = tmp_path_factory.mktemp("ranks")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    saved = {k: os.environ.get(k) for k in ("VLLM_RPC_BASE_PATH", "NKI_SIMULATOR")}
    cwd = os.getcwd()
    # vLLM's broadcaster binds an IPC socket under this path, which must stay short.
    os.environ.update(VLLM_RPC_BASE_PATH=".", NKI_SIMULATOR="1")
    os.chdir(out)
    try:
        mp.spawn(_rank, args=(port, *inputs, ties, str(out)), nprocs=TP, join=True)
    finally:
        os.chdir(cwd)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key)
            else:
                os.environ[key] = value
    return [torch.load(out / f"rank{r}.pt", weights_only=False) for r in range(TP)]


def _reference_base(hidden: torch.Tensor) -> torch.Tensor:
    """Final norm and the whole LM head in fp64: ``[blocks, 4, V]``."""
    with open(DRAFT / "config.json") as f:
        eps = json.load(f)["rms_norm_eps"]
    h = hidden.double()
    normed = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    normed = normed * _whole("final_norm.weight").double()
    with safe_open(_head_file(), framework="pt") as f:
        head = f.get_slice(LM_HEAD)
        chunk = VOCAB // 8
        return torch.cat(
            [normed @ head[c : c + chunk].double().t() for c in range(0, VOCAB, chunk)], -1
        )


def _judge(drafts: torch.Tensor, rows: list[torch.Tensor]) -> tuple[list, list]:
    """Each pick against its reference row: the differing picks and the near-ties, with gaps."""
    differing, near_ties = [], []
    for block in range(BLOCKS):
        for step in range(STEPS):
            row, pick = rows[step][block], int(drafts[block, step])
            if pick == int(row.argmax()):
                continue
            gap = float((row.max() - row[pick]) / row.max().abs())
            (near_ties if gap <= NEAR_TIE else differing).append((block, step, f"{gap:.1e}"))
    return differing, near_ties


def test_picks_equal_the_fp64_reference_with_whole_weights(ranks, inputs, markov):
    hidden, anchors = inputs
    w1, w2 = markov
    assert torch.equal(ranks[0]["picks"], ranks[1]["picks"])
    drafts = ranks[0]["picks"].long()
    base = _reference_base(hidden)
    # Each step reads the product's previous pick, so a flip never carries forward.
    previous = torch.cat([anchors[:, None], drafts[:, :-1]], 1)
    rows = [base[:, i] + w1[previous[:, i]] @ w2.t() for i in range(STEPS)]
    differing, near_ties = _judge(drafts, rows)
    warnings.warn(f"{len(near_ties)} near-ties (block, step, gap): {near_ties}")
    # The base logits of each rank's vocabulary part, over the largest reference value.
    part, scale = VOCAB // TP, base.abs().max()
    gaps = []
    for r, s in enumerate(ranks):
        got, want = s["base"].double(), base[..., r * part : (r + 1) * part]
        # Python's max over the ranks drops a NaN gap, so a non-finite logit fails here.
        assert torch.equal(~torch.isfinite(got), ~torch.isfinite(want)), (
            f"rank {r}: the base logits are non-finite where the reference is not, or the reverse"
        )
        finite = torch.isfinite(got) & torch.isfinite(want)
        gaps.append(float((got - want).abs()[finite].max() / scale))
    warnings.warn(f"base-logit gap by rank {gaps}, band {BASE_BAND}")
    assert differing == [], f"{len(differing)} differing picks: {differing}"
    # Known-bad input: block 0's first pick moved to the closest id outside the band.
    row = rows[0][0]
    outside = row.masked_fill(row.max() - row <= NEAR_TIE * row.max().abs(), -torch.inf)
    bad = drafts.clone()
    bad[0, 0] = outside.argmax()
    assert [d[:2] for d in _judge(bad, rows)[0]] == [(0, 0)]
    assert max(gaps) <= BASE_BAND, f"base-logit gap by rank {gaps} is over {BASE_BAND}"


def test_planted_ties_across_the_halves_go_to_the_lower_id_on_both_ranks(ranks, ties):
    expected = [min(source, copy) for _, source, copy in ties]
    assert TIE_IN_RANK_0 < VOCAB // TP <= TIE_IN_RANK_1
    differences = sum(state["ties"][i] != expected[i] for state in ranks for i in range(len(ties)))
    assert differences == 0, ([state["ties"] for state in ranks], expected)
