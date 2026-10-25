# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 encoder config's split switch and the per-group hand-off, in CPU mode.

Item 1 reads the switch at the served encoder config with no process group.
Items 2 and 3 run the hand-off with the split off at encoder TP1 by DP2, each
rank a gloo process. There the tower and projector are stand-ins that carry
each row's pixel values to the cache width in bf16, so a cache row can be
compared bit for bit with the projector row that made it. Set
``KIMI_K3_CHECKPOINT`` as for the tower test. Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import socket
from pathlib import Path

import torch
import torch.nn as nn
from test_vision_tower import (  # noqa: F401  (pytestmark is used by pytest)
    TEXT,
    build_model,
    hf_config,
    new_cache,
    pytestmark,
)

SERVED = {
    "num_encoder_input_buckets": [5504, 33536],
    "encoder_attention_block_size": 5504,
    "tp_size": 4,
    "dp_size": 8,
}
SERVED_BLOCK = 5504
#: Cache rows per block: the merged rows of one 5,504-row group block.
BLOCK_ROWS = SERVED_BLOCK // 4
#: A cache value the stand-ins never write.
FILL = -1.0
#: The served image, 74 x 74 patches, at the 5,504-row bucket.
SERVED_GRID = (74, 74)
SMALL_GRID = (40, 60)
#: An image of 78 x 78 patches, at the 33,536-row bucket.
LARGE_GRID = (78, 78)


def _image(grid, seed) -> dict:
    """Seeded uint8 patches for one image, as the processor gives them."""
    h, w = grid
    pixels = torch.randint(0, 256, (h * w, 3, 14, 14), dtype=torch.uint8,
                           generator=torch.Generator().manual_seed(seed))
    return {"pixel_values": pixels, "grid_thws": torch.tensor([[1, h, w]])}


def _encoder_config(**keys):
    """The encoder config the runner parses from ``--additional-config``."""
    from vllm_neuron.model.neuron_config import EncoderNeuronConfig

    return EncoderNeuronConfig.from_dict({**SERVED, **keys})


def _form(kwargs) -> tuple[int, bool, int]:
    """A tower call's query rows, ``gather`` flag and ``slice_rows``."""
    return (kwargs["pixel_values"].shape[0], kwargs.get("gather", False), kwargs.get("slice_rows", 0))


class _Recorder(nn.Module):
    """Stands in for the compiled encoder graph: records each call's form."""

    def __init__(self):
        super().__init__()
        self.forms = []

    def forward(self, **kwargs):
        self.forms.append(_form(kwargs))
        return kwargs["encoder_cache_buffer"]


def test_switch_sets_the_split_from_the_encoder_config():
    """Item 1: the field defaults true, the key sets it false, the model follows it, and so does the form."""
    found = {}
    for case, keys in (("no key", {}), ("false", {"split_lone_image": False})):
        config = _encoder_config(**keys)
        model = build_model(hf_config(), config, load=False)
        recorder = _Recorder()
        model.visual = recorder
        model.embed_multimodal(encoder_cache=new_cache(2, BLOCK_ROWS), mm_hashes=["image"],
                               **_image(SERVED_GRID, 1))
        found[case] = (config.split_lone_image, model.split_lone_image, recorder.forms)
        print(f"{case}: field, model, forms (rows, gather, slice rows) {found[case]}")
    assert found == {"no key": (True, True, [(768, False, 688)]),
                     "false": (False, False, [(5504, False, 0)])}


# ---------------------------------------------------------------- ranks


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank(rank, world, port, cases, out_dir) -> None:
    """One gloo rank at encoder TP1 by DP2: the encoder groups, then the hand-off cases."""
    import torch.distributed as dist

    from vllm_neuron.parallel import neuron_parallel_state as nps

    torch.set_num_threads(max(1, 8 // world))
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    nps.initialize_neuron_parallel_state(encoder_tp_size=1, encoder_dp_size=world,
                                         tp_global_ranks=list(range(world)), local_rank=rank)
    nps.stamp_spmd_rank_tensors("cpu")
    torch.save(_hand_off(cases), Path(out_dir) / f"hand-off-{rank}.pt")
    dist.destroy_process_group()


def run_ranks(cases, tmp_path, world: int = 2) -> list:
    """Run the hand-off cases on ``world`` gloo ranks; return each rank's result."""
    torch.multiprocessing.spawn(_rank, args=(world, _free_port(), cases, str(tmp_path)), nprocs=world)
    return [torch.load(tmp_path / f"hand-off-{r}.pt") for r in range(world)]


class _Tower(nn.Module):
    """Gives each row's pixel values back in bf16, as the tower's output."""

    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, pixel_values, *args):
        self.events.append("tower")
        return pixel_values.to(torch.bfloat16)


class _Projector(nn.Module):
    """Repeats each merged row's values to the cache width."""

    def forward(self, merged, tp_group, norm_gathered=False):
        return merged.repeat(1, -(-TEXT // merged.shape[1]))[:, :TEXT]


def _hand_off(cases) -> dict:
    """Encode each case's images in one hook call with the split off; return what the cache and DP group saw."""
    model = build_model(hf_config(), _encoder_config(tp_size=1, dp_size=2, split_lone_image=False), load=False)
    visual = model.visual
    events = []
    for name in ("all_reduce", "all_gather"):
        method = getattr(visual.dp_group, name)

        def counted(*args, _method=method, _name=name, **kwargs):
            events.append(_name)
            return _method(*args, **kwargs)

        setattr(visual.dp_group, name, counted)
    visual.vision_tower = _Tower(events)
    visual.mm_projector = _Projector()
    projected = []
    visual.mm_projector.register_forward_hook(lambda m, i, out: projected.append(out.clone()))
    result = {}
    for case, images in cases.items():
        cache = new_cache(2 * len(images) + 2, BLOCK_ROWS)
        cache.buffer.fill_(FILL)
        events.clear()
        projected.clear()
        names = list(images)
        model.embed_multimodal(
            pixel_values=torch.cat([images[n]["pixel_values"] for n in names]),
            grid_thws=torch.cat([images[n]["grid_thws"] for n in names]),
            encoder_cache=cache,
            mm_hashes=names,
        )
        rows = {n: images[n]["grid_thws"][0, 1:].prod().item() // 4 for n in names}
        own = {b for n in names for b in cache.get_block_ids(n)}
        kept = [b for b in range(cache.buffer.shape[0]) if b not in own | {cache.scratch_block_id}]
        result[case] = {
            "events": list(events),
            "cache rows": {n: cache.buffer[cache.get_block_ids(n)].reshape(-1, TEXT)[: rows[n]].clone()
                           for n in names},
            "projector rows": [p[: max(rows.values())].clone() for p in projected],
            "rows": rows,
            "other blocks keep the fill": all(bool((cache.buffer[b] == FILL).all()) for b in kept),
        }
    return result


def _same_bits(a, b) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _calls(events) -> list[tuple[int, int]]:
    """Each tower call's (all-reduces, all-gathers) in the DP group."""
    calls = []
    for event in events:
        if event == "tower":
            calls.append([0, 0])
        else:
            calls[-1][1 if event == "all_gather" else 0] += 1
    return [tuple(c) for c in calls]


def _report(ranks, case, holders) -> list[dict]:
    """Per rank: the case's calls, each image's rows against its holder's projector rows, and the fill."""
    report = []
    for rank, result in enumerate(ranks):
        found = result[case]
        equal = {}
        for name, (holder, call) in holders.items():
            rows = found["rows"][name]
            source = ranks[holder][case]["projector rows"][call][:rows]
            equal[name] = (rows, _same_bits(found["cache rows"][name], source))
        report.append({"rank": rank, "calls": _calls(found["events"]), "rows equal": equal,
                       "other blocks keep the fill": found["other blocks keep the fill"]})
        print(f"{case}, rank {rank}: {report[-1]}")
    return report


def test_one_block_hand_off_at_the_first_bucket(tmp_path):
    """Item 2: at 5,504 with the split off, one image takes the all-reduce, two the gather; 0 differing bits."""
    cases = {"one": {"a": _image(SERVED_GRID, 2)},
             "two": {"a": _image(SERVED_GRID, 2), "b": _image(SMALL_GRID, 3)}}
    ranks = run_ranks(cases, tmp_path)
    # The image, or each image in turn, is held by the rank of its group, in its one tower call.
    one = _report(ranks, "one", {"a": (0, 0)})
    two = _report(ranks, "two", {"a": (0, 0), "b": (1, 0)})
    a_rows, b_rows = SERVED_GRID[0] * SERVED_GRID[1] // 4, SMALL_GRID[0] * SMALL_GRID[1] // 4
    for rank in range(2):
        assert one[rank] == {"rank": rank, "calls": [(1, 0)], "rows equal": {"a": (a_rows, True)},
                             "other blocks keep the fill": True}
        assert two[rank] == {"rank": rank, "calls": [(0, 1)],
                             "rows equal": {"a": (a_rows, True), "b": (b_rows, True)},
                             "other blocks keep the fill": True}


def test_one_block_hand_off_at_the_large_bucket(tmp_path):
    """Item 3: at 33,536, one image per call, each call one all-reduce; its 1,521 rows equal; 0 differing bits."""
    cases = {"one": {"a": _image(LARGE_GRID, 4)},
             "two": {"a": _image(LARGE_GRID, 4), "b": _image(LARGE_GRID, 5)}}
    ranks = run_ranks(cases, tmp_path)
    # Above the block size each image takes its own call, held by the first group.
    one = _report(ranks, "one", {"a": (0, 0)})
    two = _report(ranks, "two", {"a": (0, 0), "b": (0, 1)})
    rows = LARGE_GRID[0] * LARGE_GRID[1] // 4
    for rank in range(2):
        assert one[rank] == {"rank": rank, "calls": [(1, 0)], "rows equal": {"a": (rows, True)},
                             "other blocks keep the fill": True}
        assert two[rank] == {"rank": rank, "calls": [(1, 0), (1, 0)],
                             "rows equal": {"a": (rows, True), "b": (rows, True)},
                             "other blocks keep the fill": True}
