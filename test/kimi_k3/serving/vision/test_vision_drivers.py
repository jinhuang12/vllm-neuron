# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 vision serving drivers: rung-1 prompts and images, MMMU-Pro item file, task and
launch, bench shapes, and the paired TTFT client and reducer.

Set ``HF_HOME`` to the offline cache that holds MMMU-Pro at its registered commit in the
"standard (10 options)" and "vision" configurations, and ``KIMI_K3_MODEL`` to the checkpoint.
"""

import ast
import base64
import functools
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

VISION = Path(__file__).resolve().parent
SERVING = VISION.parent
sys.path[:0] = [str(VISION), str(SERVING)]

import build_prompt_files  # noqa: E402
import ttft_pair  # noqa: E402

NAME = "us.moonshotai.kimi-k3"
COMMIT = "0d7426df4ccb3d8704a992fe6850abee7c262127"
SEED = 20261023
BUCKETS = (5504, 33536)
IMAGE_LIMIT = 7
MAX_LENGTH = 16384
MAX_GEN_TOKS = 2048
DRAW_SHA256 = "15ddca470cc5b216eff3137b71d8accb4a045d089ca26ae331256cf9b9e991c8"
RUNG1_SHA256 = "0904cb95dd4739dec242b4e4a68ce44d9dd2572540a7212ccdec5ff6823d2f09"
CAPTION = "Describe this image in detail."
DOCUMENT = ("Write out the multiple-choice question in the image and then solve it. The last "
            "line of your response should be of the following format: 'Answer: $LETTER' "
            "(without quotes) where LETTER is one of options. Think step by step before "
            "answering.")
STANDARD = ("Answer the preceding multiple choice question. The last line of your response "
            "should be of the following format: 'Answer: $LETTER' (without quotes) where "
            "LETTER is one of options. Think step by step before answering.")
FIXED_ITEMS = ("test_History_134", "test_Computer_Science_61", "test_Geography_252")
REGISTERED = (("tasks/mmmu_pro/mmmu_pro_std10.yaml",
               "734e89e67eb3a0d536109afc8bf6d56703cf24ed7ba8dcee8cf89484d24d1fe9"),
              ("tasks/mmmu_pro/utils.py",
               "726ab449faade78602fd104d3502711d61cfa6a911fca509de0daf4bedb3f1ae"),
              ("mmmu_pro_300.json",
               "fef4887f942a048e837382a409d76c9cf17153d93503bd59dfc1cab94ea598c6"))


def _env(name):
    value = os.environ.get(name)
    assert value, f"set {name}"
    return value


@functools.lru_cache(maxsize=None)
def _rows(config):
    folder = Path(_env("HF_HOME"), "datasets", "MMMU___mmmu_pro", config, "0.0.0", COMMIT)
    rows = [r for f in sorted(folder.glob("mmmu_pro-test-*.arrow"))
            for r in build_prompt_files.read_arrow_stream(f)]
    info = json.loads((folder / "dataset_info.json").read_text())
    assert len(rows) == info["splits"]["test"]["num_examples"] == 1730
    return rows


@functools.lru_cache(maxsize=None)
def _utils():
    spec = importlib.util.spec_from_file_location("mmmu_pro_utils",
                                                  VISION / "tasks" / "mmmu_pro" / "utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=None)
def _counter():
    return ttft_pair.PromptCounter(_env("KIMI_K3_MODEL"))


@functools.lru_cache(maxsize=None)
def _media_cfg():
    cfg = json.loads(Path(_env("KIMI_K3_MODEL"), "preprocessor_config.json").read_text())
    return cfg["media_proc_cfg"]


def _size(image):
    from PIL import Image

    return Image.open(io.BytesIO(image["bytes"])).size


def _patches(image):
    return 4 * ttft_pair.image_tokens(*_size(image), _media_cfg())


def _images(row):
    return [row[f"image_{k}"] for k in range(1, 8) if row[f"image_{k}"] is not None]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _run(script, *args, env=None):
    cmd = (["bash"] if script.endswith(".sh") else [sys.executable, "-I"]) + [str(VISION / script)]
    return subprocess.run(cmd + [str(a) for a in args], capture_output=True, text=True,
                          env=env)


def _draw():
    """Return the registered 300 positions: per subject, largest remainder, seeded sample."""
    rows = _rows("standard (10 options)")
    subjects = sorted({r["subject"] for r in rows})
    pos = {s: [i for i, r in enumerate(rows) if r["subject"] == s] for s in subjects}
    quota = {s: 300 * len(pos[s]) / len(rows) for s in subjects}
    take = {s: math.floor(q) for s, q in quota.items()}
    for s in sorted(subjects, key=lambda s: (-(quota[s] - take[s]), s))[:300 - sum(take.values())]:
        take[s] += 1
    draw = []
    for s in subjects:
        draw += [pos[s][j] for j in sorted(random.Random(SEED).sample(range(len(pos[s])), take[s]))]
    return sorted(draw)


def test_rung1_prompts_and_images():
    """Item 1: 20 prompts, 10 captioning then 10 document, each naming one registered image."""
    prompts = _jsonl(VISION / "prompts_rung1.jsonl")
    standard, vision = _rows("standard (10 options)"), _rows("vision")
    drawn = {standard[i]["id"] for i in _draw()}
    top = max(BUCKETS)
    caption_rows = [i for i, r in enumerate(standard) if r["id"] not in drawn
                    and len(_images(r)) == 1 and "Photographs" in r["img_type"]
                    and _patches(_images(r)[0]) <= top]
    document_rows = [i for i, r in enumerate(vision) if r["id"] not in drawn
                     and _patches(r["image"]) <= top]
    rows = ([("caption", "standard (10 options)", i, _images(standard[i])[0], CAPTION)
             for i in sorted(random.Random(SEED).sample(caption_rows, 10))]
            + [("document", "vision", i, vision[i]["image"], DOCUMENT)
               for i in sorted(random.Random(SEED).sample(document_rows, 10))])
    assert len(prompts) == 20
    assert _sha("\n".join(standard[i]["id"] for _, _, i, _, _ in rows).encode()) == RUNG1_SHA256
    for n, (p, (kind, config, i, image, text)) in enumerate(zip(prompts, rows)):
        assert (p["index"], p["kind"], p["config"], p["revision"], p["row"], p["id"],
                p["prompt"]) == (n, kind, config, COMMIT, i, standard[i]["id"], text)
        data = (VISION / p["image"]).read_bytes()
        assert _sha(data) == p["image_sha256"] == _sha(image["bytes"]), p["id"]
        assert p["mime"] == ("image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else
                             "image/jpeg" if data[:3] == b"\xff\xd8\xff" else None)
    assert sorted(f.name for f in (VISION / "images").iterdir()) == sorted(
        Path(p["image"]).name for p in prompts)


def test_item_file_draw():
    """Item 2: the item file holds the 300 positions of the registered seeded draw."""
    doc = json.loads((VISION / "mmmu_pro_300.json").read_text())
    assert list(doc) == ["mmmu_pro_std10"]
    items = doc["mmmu_pro_std10"]
    assert len(items) == 300 and items == _draw()
    rows = _rows("standard (10 options)")
    assert _sha("\n".join(rows[i]["id"] for i in items).encode()) == DRAW_SHA256


def test_task_renders_fixed_items():
    """Item 3: three fixed items render as registered: template, option letters, image order."""
    utils = _utils()
    by_id = {r["id"]: r for r in _rows("standard (10 options)")}
    shapes = []
    for item_id in FIXED_ITEMS:
        row = by_id[item_id]
        options = ast.literal_eval(row["options"])
        lines = [f"{chr(ord('A') + k)}. {o}" for k, o in enumerate(options)]
        assert utils.doc_to_text(row) == "\n".join([row["question"], *lines, STANDARD])
        named = sorted({int(n) for n in re.findall(r"<image\s+(\d+)>",
                                                   row["question"] + "".join(options))})
        sent = utils.doc_to_image(row)
        assert [img["bytes"] for img in sent] == [row[f"image_{n}"]["bytes"] for n in named]
        shapes.append((len(options), len(sent)))
    assert shapes == [(10, 1), (12, 1), (4, 7)]


def test_items_fit_buckets_and_length():
    """Item 4: every drawn image fits the buckets, the image limit and the model length."""
    utils = _utils()
    counter = _counter()
    rows = _rows("standard (10 options)")
    over = []
    for i in json.loads((VISION / "mmmu_pro_300.json").read_text())["mmmu_pro_std10"]:
        row = rows[i]
        over += [(row["id"], "bucket") for image in _images(row) if _patches(image) > max(BUCKETS)]
        sent = utils.doc_to_image(row)
        if len(sent) > IMAGE_LIMIT:
            over.append((row["id"], "image limit"))
        tokens = counter.count(utils.doc_to_text(row), [_size(image) for image in sent])
        if tokens + MAX_GEN_TOKS > MAX_LENGTH:
            over.append((row["id"], f"{tokens} prompt tokens"))
    assert over == []


def test_eval_launch_print_only(tmp_path):
    """Item 5: the registered harness command with the image limit; one changed byte exits 1."""
    args = ["--print-only", "--client-venv", "/venvs/client", "--base-url",
            "http://127.0.0.1:8000", "--out", "/runs/out", "--reference-runs", "/ref/seed-0",
            "--hf-home", "/hf"]
    r = _run("run_vision_eval.sh", *args)
    assert r.returncode == 0, r.stderr
    rows = [line.split("\t", 1) for line in r.stdout.splitlines()]
    tok = {label: shlex.split(cmd) for label, cmd in rows}
    assert [label for label, _ in rows] == ["serve", "mmmu_pro_std10", "mmmu_pro_std10:check"]
    py = "/venvs/client/bin/python"
    model_args = (f"model={NAME},base_url=http://127.0.0.1:8000/v1/chat/completions,"
                  "tokenized_requests=False,tokenizer_backend=None,num_concurrent=8,"
                  "timeout=7200,max_length=16384,max_retries=10,max_images=7")
    assert tok["mmmu_pro_std10"] == [
        "env", "HF_HOME=/hf", "HF_DATASETS_OFFLINE=1", "HF_HUB_OFFLINE=1", "OPENAI_API_KEY=EMPTY",
        py, "-m", "lm_eval", "--model", "local-chat-completions", "--tasks", "mmmu_pro_std10",
        "--model_args", model_args, "--log_samples", "--output_path", "/runs/out/mmmu_pro_std10",
        "--apply_chat_template", "--fewshot_as_multiturn", "--gen_kwargs",
        "max_gen_toks=2048,temperature=0,do_sample=false,reasoning_effort=none",
        "--include_path", str(VISION / "tasks" / "mmmu_pro"), "--seed", "0,1234,1234,1234",
        "--samples", str(VISION / "mmmu_pro_300.json")]
    assert tok["mmmu_pro_std10:check"] == [
        py, str(VISION / ".." / "check_samples.py"), "--candidate", "/runs/out/mmmu_pro_std10",
        "--reference", "/ref/seed-0/out"]
    assert tok["serve"] == ["--gpu-memory-utilization", "0.90", "--max-model-len", "131072",
                            "--reasoning-parser", "kimi_k3", "--no-enable-prefix-caching",
                            "--served-model-name", NAME, "--limit-mm-per-prompt",
                            '{"image": 7}']
    for flip, _ in REGISTERED:
        files = tmp_path / Path(flip).name
        for rel, _ in REGISTERED:
            p = files / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            data = bytearray((VISION / rel).read_bytes())
            if rel == flip:
                data[len(data) // 2] ^= 0x01
            p.write_bytes(bytes(data))
        r = _run("run_vision_eval.sh", *args, "--files", files)
        assert (r.returncode, r.stdout) == (1, ""), flip


def test_bench_shapes():
    """Item 6: the text cells at 2K to 128K, S - 128 in, 128 out, N = c or 8."""
    doc = json.loads((VISION / "bench_shapes.json").read_text())
    pairs = [(2048, 1), (2048, 8), (2048, 32), (8192, 1), (8192, 8), (8192, 32), (32768, 1),
             (32768, 8), (131072, 1)]
    assert [(s["S"], s["concurrency"]) for s in doc["shapes"]] == pairs
    for s in doc["shapes"]:
        assert s["input_len"] == s["S"] - 128 and s["output_len"] == 128
        assert s["num_prompts"] == (8 if s["concurrency"] == 1 else s["concurrency"])
    assert (doc["client_timeout_s"], doc["run_deadline_s"]) == (6 * 3600, 45 * 60)


def _log(path, process, diff_ms=20.0, first_ms=None, drop=(), image_first=False,
         failed_pair=None, repeat_image=False):
    """Write one process's synthetic request log: 63 pairs, text TTFT 1,000 ms."""
    order = [(p, k) for p in range(1, 64) if p not in drop
             for k in (("image", "text") if p % 2 else ("text", "image"))]
    if image_first:
        order.sort(key=lambda pk: pk[1] != "image")
    recs = [{"type": "header", "process": process}]
    for n, (p, kind) in enumerate(order):
        diff = first_ms if p == 1 and first_ms is not None else diff_ms
        sha = _sha(f"{process}-{p - 1 if repeat_image and p == 5 else p}".encode())
        recs.append({"type": "request", "process": process, "pair": p, "warmup": p <= 3,
                     "kind": kind, "image_sha256": sha if kind == "image" else None,
                     "start_s": 3.0 * n, "end_s": 3.0 * n + 1.0,
                     "ttft_ms": 1000.0 + (diff if kind == "image" else 0.0),
                     "status": 500 if p == failed_pair and kind == "text" else 200,
                     "error": None, "finish_reason": "length", "completion_tokens": 128,
                     "prompt_tokens": 8064})
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return path


def _reduce(*logs):
    r = _run("ttft_pair.py", "--reduce", *logs, "--text-window", "14:1.31", "--text-window",
             "14:1.30", "--text-ttft-ms", 1000.0, 1000.0, "--accepted-ttft-ms", 1000.0, 1000.0)
    return r.returncode, json.loads(r.stdout)


def test_reducer_known_good(tmp_path):
    """Item 7: interleaved pairs at a constant 20.0 ms read 20.0 ms with the interval, exit 0."""
    code, rec = _reduce(_log(tmp_path / "a.jsonl", "A"), _log(tmp_path / "b.jsonl", "B"))
    assert code == 0, rec
    assert rec["value_of_record_ms"] == pytest.approx(20.0)
    assert rec["interval_ms"] == pytest.approx([20.0, 20.0])
    assert (rec["pairs_pooled"], rec["interval_ranks"], rec["coverage"]) == (120, [49, 72], 0.965)
    assert rec["reading1_stable"] and rec["process_spread_ms"] == pytest.approx(0.0)


def test_reducer_known_bad(tmp_path):
    """Item 8: all images first, a failed request, one image twice, no warm-ups: exit 2."""
    b = _log(tmp_path / "b.jsonl", "B")
    cases = {"image first": {"image_first": True}, "failed request": {"failed_pair": 10},
             "image twice": {"repeat_image": True}, "no warm-ups": {"drop": (1, 2, 3)}}
    for name, kw in cases.items():
        code, rec = _reduce(_log(tmp_path / f"{name}.jsonl", "A", **kw), b)
        assert (code, rec["result"]) == (2, "invalid run"), name
        assert list(rec["invalid"]) == ["A"], name


# vLLM's own chat renderer and image processor, as the server runs them.
_VLLM_COUNT = """
import json, sys
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.kimi_k3 import KimiK3Renderer
from vllm.tokenizers import cached_tokenizer_from_config
mc = ModelConfig(model=sys.argv[1], trust_remote_code=True, max_model_len=131072,
                 limit_mm_per_prompt={"image": 7})
renderer = KimiK3Renderer(VllmConfig(model_config=mc, device_config=DeviceConfig(device="cpu")),
                          cached_tokenizer_from_config(mc))
out = []
for body in json.load(open(sys.argv[2])):
    req = ChatCompletionRequest(**body)
    _, prompt = renderer.render_messages(req.messages, req.build_chat_params(None, "auto"))
    ids = prompt["prompt_token_ids"]
    if "multi_modal_data" in prompt:
        ids = renderer._process_multimodal(ids, prompt["multi_modal_data"],
                                           prompt.get("multi_modal_uuids"), None)["prompt_token_ids"]
    out.append(list(ids))
print("RESULT " + json.dumps(out))
"""


def test_print_only_request_pair(tmp_path):
    """Item 9: one image and one text request, equal prompt tokens, 128 out, no shared prefix."""
    model = _env("KIMI_K3_MODEL")
    r = _run("ttft_pair.py", "--print-only", "--model-path", model,
             env={**os.environ, "VLLM_PLUGINS": ""})
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    image, text = doc["image_request"], doc["text_request"]
    for body in (image, text):
        assert {k: v for k, v in body.items() if k != "messages"} == {
            "model": NAME, "temperature": 0, "max_tokens": 128, "ignore_eos": True,
            "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}
        assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
    parts = image["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["image_url", "text"]
    assert [p["type"] for p in text["messages"][0]["content"]] == ["text"]
    url = parts[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    from PIL import Image

    pic = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert (pic.format, pic.mode, pic.size) == ("JPEG", "RGB", (1024, 1024))
    bodies = tmp_path / "bodies.json"
    bodies.write_text(json.dumps([image, text]))
    count = subprocess.run([sys.executable, "-I", "-c", _VLLM_COUNT, model, str(bodies)],
                           capture_output=True, text=True, env={**os.environ, "VLLM_PLUGINS": ""})
    assert count.returncode == 0, count.stderr[-2000:]
    image_ids, text_ids = json.loads(next(line[7:] for line in count.stdout.splitlines()
                                          if line.startswith("RESULT ")))
    assert len(image_ids) == len(text_ids) == 8064
    opening = text_ids.index(_counter().tok.convert_tokens_to_ids("<|sep|>")) + 1
    assert image_ids[:opening] == text_ids[:opening] and image_ids[opening] != text_ids[opening]


def test_reducer_first_image_cost(tmp_path):
    """Item 10: warm-up pair 1 at 25.0 ms over counted pairs at 20.0 ms costs 5.0 ms each."""
    code, rec = _reduce(_log(tmp_path / "a.jsonl", "A", first_ms=25.0),
                        _log(tmp_path / "b.jsonl", "B", first_ms=25.0))
    assert code == 0, rec
    assert rec["first_image_cost_ms"] == pytest.approx({"A": 5.0, "B": 5.0})
    assert rec["value_of_record_ms"] == pytest.approx(20.0)
    assert rec["interval_ms"] == pytest.approx([20.0, 20.0])
    assert rec["process_medians_ms"] == pytest.approx({"A": 20.0, "B": 20.0})
    for live in rec["liveness"].values():
        assert (live["requests"], live["warmup_pairs"], live["counted_pairs"]) == (126, 3, 60)
    code, rec = _reduce(_log(tmp_path / "a1.jsonl", "A", first_ms=25.0, drop=(1,)),
                        _log(tmp_path / "b1.jsonl", "B", first_ms=25.0))
    assert code == 2 and rec["first_image_cost_ms"]["A"] == "missing"
    assert rec["first_image_cost_ms"]["B"] == pytest.approx(5.0)
