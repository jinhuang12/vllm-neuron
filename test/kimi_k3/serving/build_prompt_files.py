# SPDX-License-Identifier: Apache-2.0
"""Build the coherence prompt file and copy the scored-rung reference files.

``--reference DIR`` copies the MMLU-Pro sample file and the two HumanEval+ chat task
files from the reference folder, each only when its sha256 equals the registered
value, and draws the 50 coherence prompts and the known-good replies from the
offline Hugging Face cache (``--hf-home``, default ``$HF_HOME``). Every check runs
before the first write: on a mismatch or a missing dataset it exits 1 and writes
no file. ``--check`` verifies the landed copies the same way.

Standard library only: the cached splits are read from their Arrow stream files.
"""

import argparse
import hashlib
import json
import os
import random
import struct
import sys
from pathlib import Path

SEED = 20261023

# (relative path in the reference folder, landed path, sha256)
REFERENCE_FILES = (
    ("samples/mmlu_pro_500.json", "mmlu_pro_500.json",
     "199075dab36703c3fb0fb71020e12da85157f464fdf1921b0227de3c4a4e03a2"),
    ("tasks/humaneval_plus_chat.yaml", "tasks/humaneval_plus_chat/humaneval_plus_chat.yaml",
     "4265e34808fac8515ef6206bc7ce18e8192bdd1fc222f7be97aa4de10a02a218"),
    ("tasks/utils.py", "tasks/humaneval_plus_chat/utils.py",
     "d230583fcd8f8838bb5d9e5076859e6804920c0b921e58f5e71ab844be518b1d"),
)
TASK_DIR = "tasks/humaneval_plus_chat"

# Draw order, one random.Random(SEED) over all four: (source, dataset, commit, split, k)
SOURCES = (
    ("gsm8k", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "test", 20),
    ("mmlu_pro", "TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", "test", 15),
    ("humanevalplus", "evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
     "test", 10),
    ("ifeval", "google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "train", 5),
)

PROMPT_FILE = "prompts_rung1.jsonl"
KNOWN_GOOD_FILE = "known_good.jsonl"
SERVING_DIR = Path(__file__).resolve().parent


class MissingDataset(Exception):
    """A registered split is not in the offline cache."""


# Arrow IPC stream reader: the subset of the format the cached splits use.

_NULL, _INT, _FLOAT, _BINARY, _UTF8, _BOOL = 1, 2, 3, 4, 5, 6
_LIST, _STRUCT, _LARGE_BINARY, _LARGE_UTF8, _LARGE_LIST = 12, 13, 19, 20, 21


class _Table:
    """A flatbuffers table at ``pos`` in ``buf``."""

    def __init__(self, buf, pos):
        self.buf, self.pos = buf, pos
        self.vtable = pos - struct.unpack_from("<i", buf, pos)[0]
        self.vtable_len = struct.unpack_from("<H", buf, self.vtable)[0]

    def _field(self, i):
        o = 4 + 2 * i
        return struct.unpack_from("<H", self.buf, self.vtable + o)[0] if o < self.vtable_len else 0

    def scalar(self, i, fmt, default=0):
        o = self._field(i)
        return struct.unpack_from(fmt, self.buf, self.pos + o)[0] if o else default

    def _indirect(self, i):
        o = self._field(i)
        if not o:
            return None
        p = self.pos + o
        return p + struct.unpack_from("<I", self.buf, p)[0]

    def table(self, i):
        p = self._indirect(i)
        return None if p is None else _Table(self.buf, p)

    def vector(self, i):
        p = self._indirect(i)
        if p is None:
            return 0, 0
        return p + 4, struct.unpack_from("<I", self.buf, p)[0]

    def string(self, i):
        start, n = self.vector(i)
        return bytes(self.buf[start:start + n]).decode()

    def tables(self, i):
        start, n = self.vector(i)
        out = []
        for k in range(n):
            p = start + 4 * k
            out.append(_Table(self.buf, p + struct.unpack_from("<I", self.buf, p)[0]))
        return out

    def structs(self, i, fmt):
        start, n = self.vector(i)
        size = struct.calcsize(fmt)
        return [struct.unpack_from(fmt, self.buf, start + size * k) for k in range(n)]


def _parse_field(f):
    tt = f.scalar(2, "<B")
    t = f.table(3)
    info = {"name": f.string(0), "type": tt,
            "children": [_parse_field(c) for c in f.tables(5)]}
    if f.table(4) is not None:
        raise ValueError("dictionary-encoded columns are not supported")
    if tt == _INT:
        info["fmt"] = {(8, 1): "b", (8, 0): "B", (16, 1): "h", (16, 0): "H", (32, 1): "i",
                       (32, 0): "I", (64, 1): "q", (64, 0): "Q"}[
                           (t.scalar(0, "<i"), t.scalar(1, "<B"))]
    elif tt == _FLOAT:
        info["fmt"] = {0: "e", 1: "f", 2: "d"}[t.scalar(0, "<h")]
    elif tt not in (_NULL, _BINARY, _UTF8, _BOOL, _LIST, _STRUCT, _LARGE_BINARY, _LARGE_UTF8,
                    _LARGE_LIST):
        raise ValueError(f"column {info['name']!r}: Arrow type {tt} is not supported")
    return info


def _decode(field, body, nodes, buffers):
    length, null_count = next(nodes)
    tt = field["type"]
    if tt == _NULL:
        return [None] * length

    def buf():
        off, size = next(buffers)
        return body[off:off + size]

    validity = buf()

    def valid(i):
        return null_count == 0 or not validity or (validity[i >> 3] >> (i & 7)) & 1

    if tt in (_INT, _FLOAT):
        data = buf()
        values = struct.unpack_from(f"<{length}{field['fmt']}", data, 0) if length else ()
        return [v if valid(i) else None for i, v in enumerate(values)]
    if tt == _BOOL:
        data = buf()
        return [bool((data[i >> 3] >> (i & 7)) & 1) if valid(i) else None for i in range(length)]
    if tt in (_UTF8, _BINARY, _LARGE_UTF8, _LARGE_BINARY):
        ofmt = "q" if tt in (_LARGE_UTF8, _LARGE_BINARY) else "i"
        offsets = struct.unpack_from(f"<{length + 1}{ofmt}", buf(), 0)
        data = bytes(buf())
        out = []
        for i in range(length):
            if not valid(i):
                out.append(None)
                continue
            raw = data[offsets[i]:offsets[i + 1]]
            out.append(raw.decode() if tt in (_UTF8, _LARGE_UTF8) else raw)
        return out
    if tt in (_LIST, _LARGE_LIST):
        ofmt = "q" if tt == _LARGE_LIST else "i"
        offsets = struct.unpack_from(f"<{length + 1}{ofmt}", buf(), 0)
        child = _decode(field["children"][0], body, nodes, buffers)
        return [child[offsets[i]:offsets[i + 1]] if valid(i) else None for i in range(length)]
    children = [(c["name"], _decode(c, body, nodes, buffers)) for c in field["children"]]
    return [{name: col[i] for name, col in children} if valid(i) else None
            for i in range(length)]


def read_arrow_stream(path):
    """Return the rows of an Arrow IPC stream file as a list of dicts."""
    data = memoryview(Path(path).read_bytes())
    pos, fields, columns = 0, None, None
    while pos + 4 <= len(data):
        size = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        if size == -1:
            size = struct.unpack_from("<i", data, pos)[0]
            pos += 4
        if size == 0:
            break
        meta = data[pos:pos + size]
        pos += size
        msg = _Table(meta, struct.unpack_from("<I", meta, 0)[0])
        header_type, header = msg.scalar(1, "<B"), msg.table(2)
        body_len = msg.scalar(3, "<q")
        body = data[pos:pos + body_len]
        pos += body_len
        if header_type == 1:
            fields = [_parse_field(f) for f in header.tables(1)]
            columns = {f["name"]: [] for f in fields}
        elif header_type == 3:
            if header.table(3) is not None:
                raise ValueError(f"{path}: compressed record batches are not supported")
            nodes = iter(header.structs(1, "<qq"))
            buffers = iter(header.structs(2, "<qq"))
            for f in fields:
                columns[f["name"]].extend(_decode(f, body, nodes, buffers))
        else:
            raise ValueError(f"{path}: Arrow message type {header_type} is not supported")
    if fields is None:
        raise ValueError(f"{path}: no Arrow schema")
    names = [f["name"] for f in fields]
    return [dict(zip(names, row)) for row in zip(*(columns[n] for n in names))]


def load_split(hf_home, dataset, commit, split):
    """Return the rows of one split from the offline datasets cache at ``commit``."""
    dirs = sorted(Path(hf_home, "datasets").glob(f"*/*/*/{commit}"))
    dirs = [d for d in dirs if d.is_dir()]
    if len(dirs) != 1:
        raise MissingDataset(f"{dataset} at {commit}: {len(dirs)} cache folders under "
                             f"{hf_home}/datasets, expected 1")
    files = sorted(dirs[0].glob(f"*-{split}.arrow")) or sorted(
        dirs[0].glob(f"*-{split}-*-of-*.arrow"))
    if not files:
        raise MissingDataset(f"{dataset} at {commit}: no {split} split in {dirs[0]}")
    rows = []
    for f in files:
        rows.extend(read_arrow_stream(f))
    return rows


def render_prompt(source, row):
    """Return the user message for one source row."""
    if source == "mmlu_pro":
        options = [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(row["options"])]
        return "\n".join([row["question"]] + options)
    if source == "gsm8k":
        return row["question"]
    return row["prompt"]


def draw(hf_home):
    """Return the 50 prompt records and the 30 known-good replies of the registered draw."""
    rng = random.Random(SEED)
    prompts, known_good = [], []
    for source, dataset, commit, split, k in SOURCES:
        rows = load_split(hf_home, dataset, commit, split)
        for row_id in rng.sample(range(len(rows)), k):
            row = rows[row_id]
            text = render_prompt(source, row)
            prompts.append({"index": len(prompts), "source": source, "dataset": dataset,
                            "revision": commit, "split": split, "row": row_id, "prompt": text})
            if source == "gsm8k":
                reply = row["answer"]
            elif source == "humanevalplus":
                reply = row["prompt"] + row["canonical_solution"]
            else:
                continue
            known_good.append({"id": f"{source}-{row_id}", "prompt": text, "reasoning": "",
                               "content": reply, "finish_reason": "stop",
                               "completion_tokens": None})
    return prompts, known_good


def sha256(path):
    """Return the hex sha256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonl(records):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def check(out_dir):
    """Return the problems with the landed reference copies; empty when all match."""
    problems = []
    for _, landed, digest in REFERENCE_FILES:
        p = Path(out_dir, landed)
        if not p.is_file():
            problems.append(f"{landed}: missing")
        elif sha256(p) != digest:
            problems.append(f"{landed}: sha256 {sha256(p)}, registered {digest}")
    task_dir = Path(out_dir, TASK_DIR)
    if task_dir.is_dir():
        expected = {Path(landed).name for _, landed, _ in REFERENCE_FILES[1:]}
        extra = sorted(p.name for p in task_dir.iterdir()
                       if p.name not in expected and p.name != "__pycache__")
        if extra:
            problems.append(f"{TASK_DIR}: holds other files {extra}")
    return problems


def build(reference, hf_home, out_dir):
    """Check every input, then write the copies, the prompt file and the known-good file."""
    copies = []
    for rel, landed, digest in REFERENCE_FILES:
        src = Path(reference, rel)
        if not src.is_file():
            return [f"{src}: missing"]
        data = src.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            return [f"{src}: sha256 {hashlib.sha256(data).hexdigest()}, registered {digest}"]
        copies.append((landed, data))
    if not hf_home:
        return ["no offline cache: pass --hf-home or set HF_HOME"]
    try:
        prompts, known_good = draw(hf_home)
    except MissingDataset as e:
        return [f"missing dataset: {e}"]
    out = Path(out_dir)
    for landed, data in copies:
        Path(out, landed).parent.mkdir(parents=True, exist_ok=True)
        Path(out, landed).write_bytes(data)
    Path(out, PROMPT_FILE).write_text(_jsonl(prompts), encoding="utf-8")
    Path(out, KNOWN_GOOD_FILE).write_text(_jsonl(known_good), encoding="utf-8")
    return []


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reference", help="the reference folder holding samples/ and tasks/")
    mode.add_argument("--check", action="store_true", help="verify the landed copies")
    ap.add_argument("--hf-home", default=None, help="offline Hugging Face home (default $HF_HOME)")
    ap.add_argument("--out", default=str(SERVING_DIR), help="landing folder (default: this one)")
    args = ap.parse_args(argv)
    if args.check:
        problems = check(args.out)
    else:
        problems = build(args.reference, args.hf_home or os.environ.get("HF_HOME"), args.out)
    for p in problems:
        print(p, file=sys.stderr)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
