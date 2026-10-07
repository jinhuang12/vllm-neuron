# SPDX-License-Identifier: Apache-2.0
"""Numerics of the tiny GLM-5.3-Flash worlds at B in {1, 4, 64}: the fixture generator.

Two worlds, each ``B`` requests prefilled alone and then decoded together for ``STEPS``
steps, and for each the sha256 of every number that the state carriers can change:

* the DSA root world of ``test_tiny_glm5next_batch_decode`` (a DSA-only tiny root with a
  seeded random head, prompts ``5 + 7r mod 13``, each step feeding back its own greedy
  token): the final step's logits (float32 bytes), every DSA layer's pooled store and
  ring for the ``B`` request slots, and every KV latent cache;
* the KDA world of ``test_tiny_glm5next_batch_kda`` built at ``B`` slots (real KDA
  attention modules on grid weights, one bank row per request, prompts ``3 + 5r mod
  11``, grid decode rows): every step's stacked layer output and, after the last step,
  both state banks of every layer.

Run against a read-only worktree of the base revision it records what that tree
produces; ``test_tiny_glm5next_numerics_fixture`` re-runs the same scenarios on this
tree and requires every digest to be equal, and proves the digests have power with two
negative controls. The greedy ids are recorded too, but as information only: the random
head pins the argmax, so they are not evidence.

Producing command (from the base worktree, so both ``vllm_neuron`` and the test helpers
resolve to the base tree; this file is the only one read from the new tree)::

    cd /home/ubuntu/glm53f-wt2/hostpath-base && NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 \\
        NEURON_PLATFORM_TARGET_OVERRIDE=trn2 OMP_NUM_THREADS=4 \\
        PYTHONPATH=/home/ubuntu/glm53f-wt2/hostpath-base \\
        /home/ubuntu/glm53f-campaign/venv/bin/python \\
        /home/ubuntu/glm53f-wt2/hostpath/test/vllm_neuron/model/glm5_next/tiny/gen_numerics_fixture.py \\
        --out /home/ubuntu/glm53f-wt2/hostpath/test/vllm_neuron/model/glm5_next/tiny/fixtures/numerics_0a08ff4.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

BATCHES = (1, 4, 64)
STEPS = 8
FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "numerics_0a08ff4.json"


def digest(tensor) -> str:
    """sha256 of the tensor's bytes at its own dtype (bitwise, not approximate)."""
    import torch

    raw = tensor.detach().contiguous().view(-1).cpu().view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


# ── the DSA root world ────────────────────────────────────────────────────────


def dsa_prompts(batch: int) -> list[int]:
    return [5 + (r * 7) % 13 for r in range(batch)]


def dsa_world(batch: int, steps: int = STEPS):
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as dsa
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

    max_model_len = tiny.STACK_TOKENS + 8
    prompts = dsa_prompts(batch)
    assert max(prompts) + steps + 1 <= max_model_len
    return dsa._world(
        batch, max_model_len=max_model_len, prompts=prompts,
        window_blocks=-(-max_model_len // dsa.PAGE),
    )


def dsa_numerics(batch: int, steps: int = STEPS, *, world=None) -> dict:
    """Greedy ids (information) and the digests of the DSA world after ``steps`` decodes."""
    import torch

    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as dsa

    world = world or dsa_world(batch, steps)
    fed = list(world.first)
    ids = [list(fed)]
    logits = None
    for step in range(steps):
        logits = dsa._step(
            world, list(range(batch)), torch.tensor(fed),
            cached=[n + step for n in world.lengths], sampling=list(range(batch)),
        )
        fed = logits.argmax(-1).tolist()
        ids.append(list(fed))
    digests = {"final_logits": digest(logits.float())}
    for index, side in enumerate(world.runner._glm5next_side_cache_set):
        if not side:
            continue
        # The request slots only: this tree allocates a scratch slot past them.
        digests[f"side.{index}.pool_cache[:B]"] = digest(side["pool_cache"][:batch])
        digests[f"side.{index}.tail[:B]"] = digest(side["tail"][:batch])
    for name, tensors in world.caches.items():
        for j, tensor in enumerate(tensors):
            digests[f"kv.{name}.{j}"] = digest(tensor)
    return {"batch": batch, "steps": steps, "prompts": dsa_prompts(batch), "ids": ids,
            "distinct_ids": len({t for row in ids for t in row}), "digests": digests}


# ── the KDA world ─────────────────────────────────────────────────────────────


def kda_prompts(batch: int) -> list[int]:
    return [3 + (r * 5) % 11 for r in range(batch)]


def kda_world(batch: int, steps: int = STEPS):
    """``batch`` requests at slots ``0..B-1`` of banks holding exactly ``B`` slots."""
    import torch

    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_kda as kda

    text_config, hidden, layers = kda._layers()
    gen = torch.Generator().manual_seed(kda.SEED + 7)
    banks = []
    for index, attention in enumerate(layers):
        banks.append({
            "name": f"model.layers.{index}.attention",
            "family": "linear_attn",
            "state_slots": batch,
            "conv_state": torch.randn(
                (batch, *attention.kda_conv_state_shape), generator=gen
            ).to(attention.kda_conv_state_dtype),
            "recurrent_state": (torch.randn(
                (batch, *attention.kda_recurrent_state_shape), generator=gen
            ) * 0.1).to(attention.kda_recurrent_state_dtype),
        })
    prompts = kda_prompts(batch)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=[])
    runner.model = SimpleNamespace(text_config=text_config, glm5next_layer_banks=banks)
    runner.max_model_len = max(prompts) + steps + 1
    runner.max_num_reqs = batch
    req_ids = [f"req-{r}" for r in range(batch)]
    gen = torch.Generator().manual_seed(kda.SEED + batch)
    prompt_rows = [kda._grid(prompts[r], hidden, low=-1, high=1, exponent=-3, gen=gen)
                   for r in range(batch)]
    decodes = [kda._grid(batch, hidden, low=-1, high=1, exponent=-3, gen=gen)
               for _ in range(steps)]
    for r in range(batch):
        kda._step(runner, banks, layers, req_ids=[req_ids[r]], rows=prompt_rows[r],
                  cached=[0], max_query_len=prompts[r])
    return SimpleNamespace(runner=runner, banks=banks, layers=layers, req_ids=req_ids,
                           decodes=decodes, hidden=hidden, lengths=list(prompts))


def kda_numerics(batch: int, steps: int = STEPS, *, world=None) -> dict:
    """Digests of every step's output and of the banks after ``steps`` decodes."""
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_kda as kda

    world = world or kda_world(batch, steps)
    digests = {}
    for step in range(steps):
        out, _ = kda._step(
            world.runner, world.banks, world.layers, req_ids=world.req_ids,
            rows=world.decodes[step], cached=[n + step for n in world.lengths],
            max_query_len=1,
        )
        digests[f"step.{step}.output"] = digest(out)
    for index, bank in enumerate(world.banks):
        for key in ("conv_state", "recurrent_state"):
            digests[f"bank.{index}.{key}"] = digest(bank[key])
    return {"batch": batch, "steps": steps, "prompts": kda_prompts(batch), "digests": digests}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=FIXTURE)
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    import vllm_neuron

    tree = pathlib.Path(vllm_neuron.__file__).resolve().parents[1]
    rev = subprocess.run(["git", "-C", str(tree), "rev-parse", "--short", "HEAD"],
                         check=True, capture_output=True, text=True).stdout.strip()
    record = {
        "producing_command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "vllm_neuron_tree": str(tree),
        "rev": rev,
        "env": {k: os.environ.get(k) for k in
                ("NKI_SIMULATOR", "VLLM_NEURON_CPU_MODE", "NEURON_PLATFORM_TARGET_OVERRIDE",
                 "PYTHONPATH")},
        "dsa": {str(batch): dsa_numerics(batch, args.steps) for batch in args.batches},
        "kda": {str(batch): kda_numerics(batch, args.steps) for batch in args.batches},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.out} from {tree} @ {rev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
