# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 prefill with image rows: the merge after the embed, on 4 CPU ranks.

The model is the data-parallel test's 4-layer model (three KDA layers, then MLA; hidden
7,168, 3 heads per rank), at TP4 on 4 gloo ranks from one set of random weights: once as
the text model and once with the vision tower, whose own weights the prefill never reads.
The encoder inputs are laid out as the runner passes them on a prefill step: the whole
cache buffer, the block ids with the scratch block in unused slots, and each block row's
stream position, the token count where a row is unused. Every rank embeds the whole chunk.
The image steps end at layer 0's input; the text steps run the whole forward and read
every row's logits. Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory (its config
only); run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_dp import (  # noqa: F401  (pytestmark is used by pytest)
    H,
    P,
    TP,
    _caches,
    _experts,
    _inputs,
    _load,
    _model,
    _weights,
    pytestmark,
)

from vllm_neuron.model.kimi_k3 import model as m

#: One image of 64 tokens in cache block 0, at stream positions 100 to 163.
IMAGE, START = 64, 100
#: Cache blocks of 128 rows; the last block is the scratch block.
BLOCKS, ROWS = 3, 128
SCRATCH = BLOCKS - 1


class _Stop(Exception):
    """Ends a step at layer 0's input."""


def _vision(text: m.KimiK3ForConditionalGeneration):
    """The image-enabled model on meta, from the text model's own config."""
    from vllm_neuron.model.kimi_k3.vision_encoder import KimiK3VisionForConditionalGeneration
    from vllm_neuron.model.neuron_config import EncoderNeuronConfig

    encoder = EncoderNeuronConfig(num_encoder_input_buckets=[384],
                                  encoder_attention_block_size=384, tp_size=1, dp_size=1)
    with torch.device("meta"):
        return KimiK3VisionForConditionalGeneration(text.config, encoder)


def _encoder_inputs(image: bool) -> dict[str, torch.Tensor]:
    """The cache buffer of random rows, block ids and positions for one prefill step."""
    gen = torch.Generator().manual_seed(5)
    buffer = (0.5 * torch.randn(BLOCKS, ROWS, H, generator=gen)).bfloat16()
    positions = torch.full((2, ROWS), P, dtype=torch.int32)
    ids = torch.full((2,), SCRATCH, dtype=torch.int32)
    if image:
        ids[0] = 0
        positions[0, :IMAGE] = torch.arange(START, START + IMAGE, dtype=torch.int32)
    return {"encoder_embedding_buffer": buffer, "encoder_block_ids": ids,
            "encoder_positions": positions}


def _step(model, rank_t, tp, whole: bool, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One prefill step on fresh caches: layer 0's input, and every row's logits if ``whole``."""
    seen, logits = [], None
    first = m.KimiK3DecoderLayer.forward_prefill

    def tap(self, stream, *args):
        if not seen:
            seen.append(stream.clone())
            if not whole:
                raise _Stop
        return first(self, stream, *args)

    model.bind_kv_cache(_caches(0, tp, "prefill"))
    ids, positions, meta = _inputs(0, "prefill")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m.KimiK3DecoderLayer, "forward_prefill", tap)
        try:
            logits = model(ids, positions, meta, rank=rank_t, **kwargs)
        except _Stop:
            pass
    return seen[0], logits


def _rank(rank: int, port: int, root: Path, steps: str) -> None:
    """One gloo rank: the text model's step, then the image-enabled model's step."""
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state, stamp_spmd_rank_tensors)

    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank)
    stamped = stamp_spmd_rank_tensors("cpu")
    weights, experts = torch.load(root / "weights.pt", mmap=True), torch.load(root / "experts.pt")
    rank_t = torch.tensor(rank, dtype=torch.int32)
    text = _model()
    _load(text, weights, experts, rank)
    vision = _vision(text)
    visual, vision.visual = vision.visual, None
    _load(vision, weights, experts, rank)
    vision.visual = visual
    whole = steps == "text"
    out = {"text": _step(text, rank_t, rank, whole),
           "vision": _step(vision, rank_t, rank, whole, **_encoder_inputs(not whole))}
    torch.save(out, root / f"{steps}-rank{rank}.pt")
    del stamped
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    """The random weights every rank loads."""
    root = tmp_path_factory.mktemp("prefill")
    torch.save(_weights(_model()), root / "weights.pt")
    torch.save(_experts(), root / "experts.pt")
    return root


def _ranks(root: Path, steps: str) -> list[dict]:
    """Run ``steps`` on every rank; return each rank's two steps."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_rank, args=(port, root, steps), nprocs=TP, join=True)
    return [torch.load(root / f"{steps}-rank{r}.pt") for r in range(TP)]


def _differing_words(a: torch.Tensor, b: torch.Tensor) -> int:
    """Count the 16-bit words that differ between two bf16 tensors of one shape."""
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def test_image_rows_replace_the_embed_at_their_positions(root):
    """Item 1: the image positions hold the cache rows, every other row the text embed."""
    rows = _encoder_inputs(True)["encoder_embedding_buffer"][0, :IMAGE]
    image = torch.arange(START, START + IMAGE)
    other = torch.ones(P, dtype=torch.bool).index_fill(0, image, False)
    found = {r: (_differing_words(out["vision"][0][image], rows),
                 _differing_words(out["vision"][0][other], out["text"][0][other]))
             for r, out in enumerate(_ranks(root, "image"))}
    print(f"differing words by rank, (image rows, other rows): {found}")
    assert found == {r: (0, 0) for r in range(TP)}


def test_text_prompt_on_the_image_model_matches_the_text_model(root):
    """Item 2: with no image the image-enabled model's logits equal the text model's."""
    found = {r: _differing_words(out["vision"][1], out["text"][1])
             for r, out in enumerate(_ranks(root, "text"))}
    print(f"differing logit words by rank: {found}")
    assert found == {r: 0 for r in range(TP)}


def test_inputs_embeds_still_raises():
    """Item 3: a call with ``inputs_embeds`` raises the text model's error."""
    vision = _vision(_model())
    with pytest.raises(NotImplementedError, match="^Kimi K3 serves text prompts only$"):
        vision(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), {},
               inputs_embeds=torch.zeros(1, H), **_encoder_inputs(True))
