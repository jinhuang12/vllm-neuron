# SPDX-License-Identifier: Apache-2.0
"""Greedy decode ids of the tiny GLM-5.3-Flash root at B in {1, 4, 64}: the fixture generator.

The scenario is ``test_tiny_glm5next_batch_decode``'s world: ``B`` requests, each prefilled
alone at its own prompt (``5 + 7r mod 13`` tokens of a seeded stream, seeded random head),
then decoded together for ``STEPS`` steps, each step feeding back its own greedy token.
Run against a read-only worktree of the base revision it records the ids that tree
produces; ``test_tiny_glm5next_greedy_fixture`` re-runs the same scenario on this tree and
requires identical ids.

Producing command (from the base worktree, so both ``vllm_neuron`` and the test helpers
resolve to the base tree; this file is the only one read from the new tree)::

    cd /home/ubuntu/glm53f-wt2/hostpath-base && NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 \\
        NEURON_PLATFORM_TARGET_OVERRIDE=trn2 OMP_NUM_THREADS=4 \\
        PYTHONPATH=/home/ubuntu/glm53f-wt2/hostpath-base \\
        /home/ubuntu/glm53f-campaign/venv/bin/python \\
        /home/ubuntu/glm53f-wt2/hostpath/test/vllm_neuron/model/glm5_next/tiny/gen_greedy_fixture.py \\
        --out /home/ubuntu/glm53f-wt2/hostpath/test/vllm_neuron/model/glm5_next/tiny/fixtures/greedy_ids_0a08ff4.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

BATCHES = (1, 4, 64)
STEPS = 8
FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "greedy_ids_0a08ff4.json"


def greedy_ids(batch: int, steps: int = STEPS) -> dict:
    """``steps + 1`` rows of ``batch`` greedy ids: the prefill's token, then each step's."""
    import torch

    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as dsa
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

    max_model_len = tiny.STACK_TOKENS + 8
    prompts = [5 + (r * 7) % 13 for r in range(batch)]
    assert max(prompts) + steps + 1 <= max_model_len
    world = dsa._world(
        batch, max_model_len=max_model_len, prompts=prompts,
        window_blocks=-(-max_model_len // dsa.PAGE),
    )
    fed = list(world.first)
    ids = [list(fed)]
    for step in range(steps):
        logits = dsa._step(
            world, list(range(batch)), torch.tensor(fed),
            cached=[n + step for n in world.lengths], sampling=list(range(batch)),
        )
        fed = logits.argmax(-1).tolist()
        ids.append(list(fed))
    return {"batch": batch, "steps": steps, "prompts": prompts, "ids": ids}


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
        "runs": {str(batch): greedy_ids(batch, args.steps) for batch in args.batches},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.out} from {tree} @ {rev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
