# SPDX-License-Identifier: Apache-2.0
"""The vocab-parallel head and the all-greedy device sampler together, at TP in {2, 4}.

The served lines run both at once. Each rank projects its own vocabulary shard of the
head, ``vocab_parallel_logits`` all-gathers the shard logits over the tensor-parallel
group into the full ``[B, vocab]`` rows on every rank, and ``sample_full_vocab`` with
``all_greedy`` takes the argmax of those rows. ``functional/test_lm_head_vocab_parallel.py``
reads the gather alone and ``vllm/test_glm5next_on_device_sampling.py`` the sampler at one
rank. This file drives the tiny root's own forward, where the two are wired, through the
runner's converter with device sampling on, for a decode step of four requests, at a
simulated group of W ranks.

The simulation:

1. Each rank's head is its rows of the full head, cut by the loader's own shard geometry
   for ``lm_head_weight`` at world W. The loader cuts in rank order, which is the order
   the gather concatenates in.
2. The ranks run one after another in this process. A rank's all-gather needs every
   rank's local logits, so each world runs in two rounds. Round one records each rank's
   local logits; the gather returns a placeholder and the round's tokens are discarded.
   Round two hands every rank the rank-ordered concatenation of the recorded locals, after
   it checks that the rank's local is the one it recorded.
3. Only the head is sharded. Every rank runs the one-rank tiny stack, so a row-parallel
   partial already is the whole sum and the group's ``all_reduce`` leaves it unchanged.

The tiny stack's output rows are nearly parallel (cosine 1.0 to four decimals across the
four requests), so any head gives every request the same token, all in one shard. A
forward hook on the stack therefore adds head row ``TARGETS[b]`` to output row ``b``. This
makes request ``b``'s greedy token ``TARGETS[b]``, a literal derived from the
construction, and the targets fall in every shard at both worlds. A gather that drops,
reorders or misplaces a shard changes a token.

The reference is the same step at one rank: the whole head and no group. Its tokens must be
``TARGETS``, and every rank's tokens must equal them. The gathered logits must be within one
bf16 ulp of the reference's peak, because the shard GEMM may block its sum apart from the
whole GEMM. The closest top-2 gap must be wider than twice that bound, so the token check
discriminates.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_vocab_parallel_sampling.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig
from vllm_neuron.utils.weight_loader import sharding_weight_loader

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as batch
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

BATCH = 4
PROMPTS = [6, 9, 11, 5]
#: One target token per request: shards (0, 0, 1, 1) of two and (0, 1, 2, 3) of four.
TARGETS = [5, 100, 150, 250]
#: One ``[top_k, top_p, temperature]`` row per request; ``all_greedy`` reads only the shape.
GREEDY_ROW = [-1.0, 1.0, 0.0]
BF16_ULP = torch.finfo(torch.bfloat16).eps


class _SimulatedTensorParallelGroup:
    """``world`` ranks of one tensor-parallel group, run one after another.

    ``rank`` is set by the test before each rank's forward. See the module docstring for
    the two rounds. ``gathers`` lists the rank of every all-gather call, so the test can
    check that each rank's forward gathers exactly once.
    """

    def __init__(self, world: int):
        self.world_size = int(world)
        self.rank: int | None = None
        self.gathering = False
        self.recorded: dict[int, torch.Tensor] = {}
        self.gathers: list[int] = []

    def all_gather(self, local: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim in (-1, local.dim() - 1), f"the head gathers on the vocab dim, got {dim}"
        self.gathers.append(self.rank)
        if not self.gathering:
            assert self.rank not in self.recorded, f"rank {self.rank} gathered twice"
            self.recorded[self.rank] = local.detach().clone()
            shape = list(local.shape)
            shape[-1] *= self.world_size
            return local.new_zeros(shape)
        assert torch.equal(local, self.recorded[self.rank]), (
            f"rank {self.rank}'s local logits changed between the two rounds"
        )
        return torch.cat([self.recorded[r] for r in range(self.world_size)], dim=-1)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class _CheckpointSlice:
    """The safetensors slice interface the shard loader reads."""

    def __init__(self, full: torch.Tensor):
        self._full = full

    def get_shape(self):
        return list(self._full.shape)

    def __getitem__(self, index):
        return self._full[index]


def _head_shards(root, head: torch.Tensor, world: int) -> list[torch.Tensor]:
    """Each rank's head rows, cut the way the loader cuts ``lm_head_weight``."""
    geometry = model_fp8._shard_geometry_for(root, "lm_head_weight", world)
    assert geometry is not None and (geometry.shard_dim, geometry.num_shards) == (0, world)
    assert int(geometry.shard_size) * world == tiny.STACK_VOCAB_SIZE
    loader = sharding_weight_loader(
        shard_dim=geometry.shard_dim,
        shard_size=geometry.shard_size,
        num_shards=geometry.num_shards,
    )
    return [loader.transform([_CheckpointSlice(head)], rank) for rank in range(world)]


def _decode(world, snapshot):
    """One decode step of every request with device sampling, from ``snapshot``."""
    batch._restore(world, snapshot)
    runner = world.runner
    runner.input_batch.req_ids = list(world.req_ids)
    runner._glm5next_request_tokens = None
    rows = list(range(BATCH))
    converted = runner._glm5next_model_kwargs(
        {
            "input_ids": torch.tensor(world.first),
            "positions": None,
            "attn_metadata": batch._metadata(world, rows, cached=list(world.lengths),
                                             tokens=BATCH),
            "sampling_positions": torch.arange(BATCH, dtype=torch.long),
            "sampling_params": torch.tensor([GREEDY_ROW] * BATCH, dtype=torch.float32),
            "spec_decode_metadata": None,
            "rank": None,
            "logit_mask": None,
        }
    )
    assert "device_sampling_params" in converted, sorted(converted)
    return world.root.forward(**converted)


@pytest.mark.parametrize("world_size", [2, 4])
def test_the_vocab_parallel_head_and_the_greedy_sampler_give_the_one_rank_tokens(
    world_size, monkeypatch
):
    max_model_len = tiny.STACK_TOKENS + 8
    world = batch._world(BATCH, max_model_len=max_model_len, prompts=PROMPTS,
                         window_blocks=-(-max_model_len // batch.PAGE))
    root = world.root
    root.text_config.neuron_config = SimpleNamespace(
        on_device_sampling_config=OnDeviceSamplingConfig(all_greedy=True)
    )
    world.runner.on_device_sampling = True
    head = root.lm_head_weight.detach().clone()
    assert tuple(head.shape) == (tiny.STACK_VOCAB_SIZE, int(root.text_config.hidden_size))
    snapshot = batch._snapshot(world)

    seen: list[torch.Tensor] = []
    sampler = model_fp8.sample_full_vocab

    def recorded(logits, *args, **kwargs):
        seen.append(logits.detach().clone())
        return sampler(logits, *args, **kwargs)

    monkeypatch.setattr(model_fp8, "sample_full_vocab", recorded)
    for size in (2, 4):
        shard = tiny.STACK_VOCAB_SIZE // size
        assert {t // shard for t in TARGETS} == set(range(size)), (size, TARGETS)
    offsets = head[TARGETS].clone()

    def spread(module, args, output):
        assert tuple(output.shape) == tuple(offsets.shape), tuple(output.shape)
        return output + offsets.to(output.dtype)

    root.model.register_forward_hook(spread)

    # The one-rank reference: the whole head, no group.
    monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: None)
    want = _decode(world, snapshot)
    want_logits = seen[-1].float()
    assert want.dtype == torch.int32 and tuple(want.shape) == (BATCH,)
    assert want.tolist() == TARGETS, want.tolist()
    peak = float(want_logits.abs().max())
    tolerance = BF16_ULP * peak
    top = want_logits.topk(2, dim=-1).values
    gap = float((top[:, 0] - top[:, 1]).min())
    assert gap > 2 * tolerance, (
        f"the closest top-2 gap {gap} is inside twice the logit tolerance {tolerance}, "
        f"so the token check would not discriminate"
    )

    group = _SimulatedTensorParallelGroup(world_size)
    monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: group)
    shards = _head_shards(root, head, world_size)
    for gathering in (False, True):
        group.gathering = gathering
        for rank in range(world_size):
            group.rank = rank
            root.lm_head_weight = torch.nn.Parameter(shards[rank], requires_grad=False)
            got = _decode(world, snapshot)
            if not gathering:
                continue
            gathered = seen[-1].float()
            assert tuple(gathered.shape) == (BATCH, tiny.STACK_VOCAB_SIZE), (
                f"rank {rank} sampled from {tuple(gathered.shape)} logits"
            )
            torch.testing.assert_close(gathered, want_logits, rtol=0.0, atol=tolerance)
            assert got.dtype == torch.int32 and tuple(got.shape) == (BATCH,)
            assert got.tolist() == want.tolist(), (
                f"rank {rank} of {world_size} sampled {got.tolist()}; one rank with the "
                f"whole head sampled {want.tolist()}"
            )
    assert group.gathers == list(range(world_size)) * 2, group.gathers
