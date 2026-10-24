# SPDX-License-Identifier: Apache-2.0
"""Paired image and text TTFT client for the one-image cell, and its reducer.

``--print-only`` prints one request pair as sent. ``--run`` sends one process's pairs to a
standing server, one request in flight, and writes the request log. ``--reduce`` reads the
logs of processes A and B (and C and D after the extension), the two text windows'
merge-call sums and the text TTFT medians, and prints the image gate's evidence.

Each pair is one request with a fresh 1024 x 1024 JPEG and one text-only request, each
8,064 prompt tokens as the server counts them. Odd pairs send the image request first.
Pairs 1 to 3 are warm-ups and pairs 4 to 63 are counted. Each request starts 2.0 s after
the previous one ends. Reducer exit codes: 0 pass, 1 fail, 2 invalid run, 3 gate undecided,
or reading 1 or the text term unstable.
"""

import argparse
import base64
import hashlib
import http.client
import io
import json
import math
import random
import statistics
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

MODEL = "us.moonshotai.kimi-k3"
PROMPT_TOKENS = 8064
MAX_TOKENS = 128
IMAGE_SIDE = 1024
IMAGE_SEED = 20261024
IMAGE_BASE = {"A": 0, "B": 1000, "C": 2000, "D": 3000, "smoke": 9000}
WARMUP_PAIRS = 3
COUNTED_PAIRS = 60
IDLE_S = 2.0
REQUEST_TIMEOUT_S = 600.0
RUN_DEADLINE_S = 3600.0
DISTINCT_WORDS = 4

ENCODER_FLOOR_MS = 17.232322650023846
BUDGET_MS = 1.5 * ENCODER_FLOOR_MS
TD1_PREDICTED_MS = 1.3137319377329633
TD1_PREDICTED_CALLS = 14
MAX_SPREAD = 0.10
# Pooled counted pairs -> (lower rank, upper rank, coverage) of the interval for the median.
INTERVAL = {120: (49, 72, 0.965), 240: (105, 136, 0.955)}
PASS, FAIL, INVALID, UNDECIDED = 0, 1, 2, 3

PLACEHOLDER = "<|kimi_image_placeholder|>"

WORDS = tuple("""
    able about above act add after again against age ago air all allow almost alone along
    already also always among amount and animal answer any appear apple area arm around art
    ask away baby back bad bag ball bank base bear beat beautiful bed been before begin
    being bell best better between big bird black block blood blue board boat body bone
    book born both bottom box boy branch bread break bright bring brown build burn busy but
    buy call camp can capital car card care carry case cat catch cause cell center chair
    change character charge chart check chief child choose church circle city claim class
    clean clear clock close cloth cloud coat cold color column come common company compare
    complete condition consider contain continue control cool copy corn corner correct cost
    could count country course cover cow create crop cross cry current cut dark day dead
    deal death deep degree depend describe design detail develop did die direct divide
    doctor does dog done door double down draw dream dress drive drop dry during duck each
    early earth east easy eat edge effect egg either electric element else end enemy energy
    engine enter equal even event ever every exact example except expect experience eye
    face fact fair fall family far farm fast father feed feel felt few field fig fight
    figure fill final find fine finger finish fire first fish fit five flat floor flow
    flower fly follow food foot force forest form forward found four free fresh friend from
    front fruit full fun game gas gather general get girl give glass gold gone good got
    grand grass gray great green ground group grow guess guide gun hair half hall hand
    happy hard has hat have head heard heart heat heavy held help her here high hill him
    his history hold hole home hope horse hot hour house how human ice idea inch include
    instant instrument interest iron its job join joy jump just keep key kill kind king
    know lady lake land language large last late lay lead learn least leave led left leg
    length less let letter level lie life lift light like line liquid list listen little
    live long look lost lot love low machine made main major make man many map mark market
    mass master match material matter may measure meet member men metal method middle might
    mile million mind mine minute miss mix modern moment money month moon more most mother
    motion mount mouth move much music must name natural near neck need neighbor never new
    next night nine noise none nor north note nothing notice now number object observe
    ocean off offer office often oil old once one only open order organ original other our
    out over own oxygen page paint pair paper paragraph parent part party pass past path
    pattern pay people perhaps period person phrase pick picture piece pitch place plain
    plan plane planet plant play please point populate port pose position possible post
    power practice prepare present press pretty print problem process produce product
    proper property protect prove provide pull push put
""".split())

# Request bodies and prompt sizing


def make_image(g):
    """Return the JPEG bytes of registered image ``g``."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng([IMAGE_SEED, g])
    field = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    big = Image.fromarray(field, "RGB").resize((IMAGE_SIDE, IMAGE_SIDE), Image.BICUBIC)
    noise = rng.integers(-8, 9, size=(IMAGE_SIDE, IMAGE_SIDE, 3))
    pixels = np.clip(np.asarray(big, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(buf, "JPEG", quality=90, subsampling=2)
    return buf.getvalue()


def image_tokens(width, height, cfg):
    """Return the image's pad-token count by the checkpoint's resize rule."""
    ps, mk = cfg["patch_size"], cfg["merge_kernel_size"]
    side, limit = cfg["patch_limit_on_one_side"], cfg["in_patch_limit"]
    s1 = math.sqrt(limit / (max(1.0, width // ps) * max(1.0, height // ps)))
    scale = min(1.0, s1, side * ps / width, side * ps / height)
    w, h = max(1, int(width * scale)), max(1, int(height * scale))
    w, h = min(w, side * ps), min(h, side * ps)
    f = mk * ps
    return ((h + (f - h % f) % f) // f) * ((w + (f - w % f) % f) // f)


class PromptCounter:
    """Count prompt tokens the way the server's chat renderer and image processor do."""

    def __init__(self, model_path):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        cfg = json.loads(Path(model_path, "preprocessor_config.json").read_text())
        self.media_cfg = cfg["media_proc_cfg"]
        self.target = self.tok.encode(PLACEHOLDER, add_special_tokens=False)

    def media_block(self, width, height):
        """Return the text the server puts in place of one image's placeholder."""
        n = image_tokens(width, height, self.media_cfg)
        return (f"<|media_begin|>image {width}x{height}<|media_content|>"
                + "<|media_pad|>" * n + "<|media_end|>")

    def ids(self, text, image_sizes=()):
        """Return the prompt token ids of one user message: its images, then its text."""
        content = "\n".join([PLACEHOLDER] * len(image_sizes) + [text])
        ids = list(self.tok.apply_chat_template([{"role": "user", "content": content}],
                                                tokenize=True, add_generation_prompt=True,
                                                thinking=False))
        if not image_sizes:
            return ids
        blocks = [self.media_block(w, h) for w, h in image_sizes]
        n = len(self.target)
        starts = [i for i in range(len(ids) - n + 1) if ids[i:i + n] == self.target]
        if len(starts) == len(blocks) and all(b - a >= n for a, b in zip(starts, starts[1:])):
            out, prev = [], 0
            for start, block in zip(starts, blocks):
                out += ids[prev:start] + self.tok.encode(block, add_special_tokens=False)
                prev = start + n
            return out + ids[prev:]
        # The placeholder's tokens merge with the next character, so the server replaces it
        # in the decoded text and encodes the result again.
        text_in, target_text = self.tok.decode(ids), self.tok.decode(self.target)
        parts = text_in.split(target_text)
        if len(parts) != len(blocks) + 1:
            raise ValueError(f"{len(parts) - 1} placeholders for {len(blocks)} images")
        joined = parts[0] + "".join(
            self.tok.decode(self.tok.encode(b, add_special_tokens=False)) + p
            for b, p in zip(blocks, parts[1:]))
        return self.tok.encode(joined, add_special_tokens=False)

    def count(self, text, image_sizes=()):
        """Return the prompt token count of one user message."""
        return len(self.ids(text, image_sizes))


def words_for(seed, n):
    """Return the first ``n`` words of request ``seed``'s draw from the word list."""
    rng = random.Random(seed)
    return [rng.choice(WORDS) for _ in range(n)]


def sized_text(counter, seed, image_sizes=(), target=PROMPT_TOKENS):
    """Return the text whose message counts ``target`` prompt tokens."""
    n = target
    tried = set()
    for _ in range(40):
        text = " ".join(words_for(seed, n))
        got = counter.count(text, image_sizes)
        if got == target:
            return text
        tried.add(n)
        n += target - got
        if n in tried or n <= 0:
            break
    raise ValueError(f"no word count of seed {seed} gives {target} prompt tokens")


def request_body(text, image_jpeg=None):
    """Return one registered chat request body."""
    content = []
    if image_jpeg is not None:
        url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": text})
    return {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": MAX_TOKENS, "ignore_eos": True,
            "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True}}


def text_seed(g, kind):
    """Return the word-draw seed of image ``g``'s request of ``kind``."""
    return f"{IMAGE_SEED}-{g}-{kind}"


def build_pair(counter, process, pair):
    """Return the image and text requests of ``pair`` (1-based) in ``process``."""
    g = IMAGE_BASE[process] + pair - 1
    jpeg = make_image(g)
    size = (IMAGE_SIDE, IMAGE_SIDE)
    image_text = sized_text(counter, text_seed(g, "image"), [size])
    text_text = sized_text(counter, text_seed(g, "text"))
    image = {"kind": "image", "image": g, "image_sha256": hashlib.sha256(jpeg).hexdigest(),
             "seed": text_seed(g, "image"), "body": request_body(image_text, jpeg)}
    text = {"kind": "text", "image": None, "image_sha256": None, "seed": text_seed(g, "text"),
            "body": request_body(text_text)}
    return (image, text) if pair % 2 else (text, image)


def body_text(body):
    """Return the text part of a request body."""
    return body["messages"][0]["content"][-1]["text"]


# Client


def send(base_url, body):
    """Send one streamed request; return its timing and usage record."""
    url = urllib.parse.urlsplit(base_url)
    payload = json.dumps(body).encode()
    rec = {"status": None, "ttft_ms": None, "finish_reason": None, "prompt_tokens": None,
           "completion_tokens": None, "error": None}
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=REQUEST_TIMEOUT_S)
    try:
        conn.connect()
        t0 = time.perf_counter()
        rec["start_s"] = t0
        conn.request("POST", url.path.rstrip("/") + "/v1/chat/completions", payload,
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        rec["status"] = resp.status
        if resp.status != 200:
            rec["error"] = resp.read(2000).decode(errors="replace")
            return rec
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if rec["ttft_ms"] is None and any(
                        delta.get(k) for k in ("content", "reasoning_content", "reasoning")):
                    rec["ttft_ms"] = (time.perf_counter() - t0) * 1000.0
                if choice.get("finish_reason"):
                    rec["finish_reason"] = choice["finish_reason"]
            usage = chunk.get("usage")
            if usage:
                rec["prompt_tokens"] = usage.get("prompt_tokens")
                rec["completion_tokens"] = usage.get("completion_tokens")
    except (OSError, http.client.HTTPException, ValueError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        rec["end_s"] = time.perf_counter()
        rec.setdefault("start_s", rec["end_s"])
        conn.close()
    return rec


def check_prefixes(plan):
    """Raise when two text parts of the run start with the same words."""
    seen = {}
    for pair, reqs in plan:
        for r in reqs:
            head = tuple(body_text(r["body"]).split()[:DISTINCT_WORDS])
            if head in seen:
                raise ValueError(f"pair {pair} {r['kind']} shares its first words with "
                                 f"{seen[head]}")
            seen[head] = f"pair {pair} {r['kind']}"


def run(args):
    """Send one process's pairs and write its request log; exit 0 when every request served."""
    out = Path(args.out)
    if out.exists():
        print(f"error: {out} exists", file=sys.stderr)
        return INVALID
    counter = PromptCounter(args.model_path)
    plan = [(p, build_pair(counter, args.process, p)) for p in range(1, args.pairs + 1)]
    check_prefixes(plan)
    header = {"type": "header", "process": args.process, "base_url": args.base_url,
              "tip": args.tip, "pairs": args.pairs, "warmup_pairs": WARMUP_PAIRS,
              "image_base": IMAGE_BASE[args.process],
              "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    ok = True
    deadline = time.monotonic() + RUN_DEADLINE_S
    with out.open("x", encoding="utf-8") as log:
        log.write(json.dumps(header) + "\n")
        for pair, reqs in plan:
            for r in reqs:
                if time.monotonic() > deadline:
                    log.write(json.dumps({"type": "abort", "reason": "run deadline"}) + "\n")
                    return FAIL
                time.sleep(IDLE_S)
                rec = send(args.base_url, r["body"])
                rec.update(type="request", process=args.process, pair=pair,
                           warmup=pair <= WARMUP_PAIRS, kind=r["kind"], image=r["image"],
                           image_sha256=r["image_sha256"], seed=r["seed"])
                ok &= rec["status"] == 200 and rec["error"] is None
                log.write(json.dumps(rec) + "\n")
                log.flush()
    return PASS if ok else FAIL


def print_only(args):
    """Print one request pair with its prompt token counts and shared prefix length."""
    counter = PromptCounter(args.model_path)
    reqs = build_pair(counter, args.process, args.pair)
    image = next(r for r in reqs if r["kind"] == "image")
    text = next(r for r in reqs if r["kind"] == "text")
    size = [(IMAGE_SIDE, IMAGE_SIDE)]
    image_ids = counter.ids(body_text(image["body"]), size)
    text_ids = counter.ids(body_text(text["body"]))
    shared = next((i for i, (a, b) in enumerate(zip(image_ids, text_ids)) if a != b),
                  min(len(image_ids), len(text_ids)))
    print(json.dumps({
        "process": args.process, "pair": args.pair, "order": [r["kind"] for r in reqs],
        "image_sha256": image["image_sha256"], "image": image["image"],
        "prompt_tokens": {"image": len(image_ids), "text": len(text_ids)},
        "shared_prefix_tokens": shared,
        "image_request": image["body"], "text_request": text["body"]}))
    return PASS


# Reducer


def _median_ms(values):
    return statistics.median(values) if values else None


def _process_log(name, records):
    """Return (process, requests, reasons, liveness) for one request log."""
    reqs = [r for r in records if r.get("type") == "request"]
    procs = sorted({r.get("process") for r in reqs})
    process = procs[0] if len(procs) == 1 else name
    reasons = []
    if len(procs) != 1:
        reasons.append(f"{len(procs)} process names in one log")
    total = WARMUP_PAIRS + COUNTED_PAIRS
    by_pair = {}
    for r in reqs:
        by_pair.setdefault(r.get("pair"), []).append(r)
    pairs = sorted(p for p in by_pair if isinstance(p, int))
    complete = [p for p in pairs if sorted(x.get("kind") for x in by_pair[p]) == ["image", "text"]]
    warm = [p for p in complete if 1 <= p <= WARMUP_PAIRS]
    counted = [p for p in complete if WARMUP_PAIRS < p <= total]
    if pairs != list(range(1, total + 1)) or len(complete) != total:
        reasons.append(f"{len(warm)} complete warm-up pairs and {len(counted)} complete "
                       f"counted pairs, registered {WARMUP_PAIRS} and {COUNTED_PAIRS}")
    if any(r.get("warmup") != (r.get("pair", 0) <= WARMUP_PAIRS) for r in reqs):
        reasons.append("a request's warm-up flag disagrees with its pair number")
    order = [(r.get("pair"), r.get("kind")) for r in sorted(reqs, key=lambda r: r["start_s"])]
    expected = [(p, k) for p in pairs
                for k in (("image", "text") if p % 2 else ("text", "image"))]
    interleaved = order == expected
    if not interleaved:
        reasons.append("the requests are not in the registered interleaved order")
    status_ok = sum(r.get("status") == 200 and not r.get("error") for r in reqs)
    finish_ok = sum(r.get("finish_reason") == "length"
                    and r.get("completion_tokens") == MAX_TOKENS for r in reqs)
    prompt_ok = sum(r.get("prompt_tokens") == PROMPT_TOKENS for r in reqs)
    ttft_ok = sum(isinstance(r.get("ttft_ms"), (int, float)) for r in reqs)
    for label, n in (("status 200", status_ok), (f"finish length at {MAX_TOKENS} tokens",
                     finish_ok), (f"{PROMPT_TOKENS} prompt tokens", prompt_ok),
                     ("a first-token time", ttft_ok)):
        if n != len(reqs):
            reasons.append(f"{len(reqs) - n} of {len(reqs)} requests lack {label}")
    liveness = {"requests": len(reqs), "warmup_pairs": len(warm), "counted_pairs": len(counted),
                "status_200": status_ok, "finish_length": finish_ok, "prompt_tokens_ok": prompt_ok,
                "interleaved": interleaved}
    return process, by_pair, reasons, liveness


def _pair_diff(by_pair, p):
    reqs = by_pair.get(p) or []
    t = {r.get("kind"): r.get("ttft_ms") for r in reqs}
    if len(reqs) != 2 or not all(isinstance(t.get(k), (int, float)) for k in ("image", "text")):
        return None
    return t["image"] - t["text"]


def _spread_ok(values, value):
    return (max(values) - min(values)) <= MAX_SPREAD * abs(value)


def _decide(lower, upper, bound):
    if upper <= bound:
        return "pass"
    if lower > bound:
        return "fail"
    return "undecided"


def reduce(logs, windows, text_ttft, accepted_ttft):
    """Return (exit code, record) for the image gate from the request logs."""
    rec = {"budget_ms": BUDGET_MS}
    invalid, liveness, first_image, medians, diffs = {}, {}, {}, {}, {}
    images = {}
    for name, records in logs:
        process, by_pair, reasons, live = _process_log(name, records)
        if process in liveness:
            reasons.append(f"process {process} appears in two logs")
            process = name
        liveness[process] = live
        for r in (x for x in records if x.get("type") == "request" and x.get("kind") == "image"):
            sha = r.get("image_sha256")
            if not sha:
                reasons.append(f"pair {r.get('pair')}: image request without its image hash")
            elif sha in images:
                reasons.append(f"pair {r.get('pair')}: image sent twice ({images[sha]})")
            else:
                images[sha] = f"process {process} pair {r.get('pair')}"
        counted = [_pair_diff(by_pair, p)
                   for p in range(WARMUP_PAIRS + 1, WARMUP_PAIRS + COUNTED_PAIRS + 1)]
        counted = [d for d in counted if d is not None]
        diffs[process] = counted
        medians[process] = _median_ms(counted)
        first = _pair_diff(by_pair, 1)
        first_image[process] = (first - medians[process]
                                if first is not None and medians[process] is not None
                                else "missing")
        if reasons:
            invalid[process] = reasons
    n_proc = len(liveness)
    if n_proc not in (2, 4):
        invalid["set"] = [f"{n_proc} process logs, registered 2, or 4 after the extension"]
    if len(windows) != 2:
        invalid["text term"] = [f"{len(windows)} text-window merge sums, registered 2"]
    if text_ttft and len(text_ttft) != n_proc:
        invalid["text TTFT"] = [f"{len(text_ttft)} text TTFT medians for {n_proc} processes"]
    if n_proc == 4 and not text_ttft:
        invalid["text TTFT"] = ["the extension holds no 8K, 1-user text run"]
    rec.update(liveness=liveness, first_image_cost_ms=first_image,
               process_medians_ms=medians,
               text_windows=[{"calls": c, "sum_ms": s} for c, s in windows],
               text_window_calls_predicted=TD1_PREDICTED_CALLS)
    if invalid:
        rec.update(result="invalid run", invalid=invalid)
        return INVALID, rec

    pooled = sorted(d for v in diffs.values() for d in v)
    lo_rank, hi_rank, coverage = INTERVAL[len(pooled)]
    value = statistics.median(pooled)
    d_lo, d_hi = pooled[lo_rank - 1], pooled[hi_rank - 1]
    stable1 = _spread_ok(list(medians.values()), value)
    sums = [s for _, s in windows]
    td1 = max(sums)
    stable_td1 = _spread_ok(sums, td1)
    gate = _decide(d_lo + td1, d_hi + td1, BUDGET_MS)
    rec.update(pairs_pooled=len(pooled), value_of_record_ms=value,
               interval_ms=[d_lo, d_hi], interval_ranks=[lo_rank, hi_rank], coverage=coverage,
               process_spread_ms=max(medians.values()) - min(medians.values()),
               reading1_stable=stable1,
               text_term={"td1_ms": td1, "stable": stable_td1,
                          "spread_ms": max(sums) - min(sums),
                          "predicted_ms": TD1_PREDICTED_MS,
                          "prediction_miss": td1 > TD1_PREDICTED_MS},
               gate={"lower_ms": d_lo + td1, "upper_ms": d_hi + td1, "result": gate})

    reading2 = {"result": "undecided"}
    if not text_ttft or not accepted_ttft:
        reading2["reason"] = "no text TTFT medians given"
    else:
        t, a = statistics.fmean(text_ttft), statistics.fmean(accepted_ttft)
        h_t = (max(text_ttft) - min(text_ttft)) / 2.0
        h_a = (max(accepted_ttft) - min(accepted_ttft)) / 2.0
        reading2.update(text_ttft_ms=t, half_width_text_ms=h_t, accepted_ttft_ms=a,
                        half_width_accepted_ms=h_a,
                        lower_ms=(t - h_t) + d_lo - (a + h_a),
                        upper_ms=(t + h_t) + d_hi - (a - h_a))
        if not _spread_ok(text_ttft, t):
            reading2["reason"] = "the text TTFT is unstable"
        elif not _spread_ok(accepted_ttft, a):
            reading2["reason"] = "the accepted text TTFT is unstable"
        else:
            reading2["result"] = _decide(reading2["lower_ms"], reading2["upper_ms"], BUDGET_MS)
    rec["reading2"] = reading2
    holds = gate == "undecided" or (
        "lower_ms" in reading2 and reading2["lower_ms"] <= BUDGET_MS < reading2["upper_ms"])
    rec["extension_due"] = n_proc == 2 and holds

    if not (stable1 and stable_td1):
        rec["result"] = "unstable"
        return UNDECIDED, rec
    if gate == "fail" or reading2["result"] == "fail":
        rec["result"] = "fail"
        return FAIL, rec
    if gate == "undecided":
        rec["result"] = "undecided"
        return UNDECIDED, rec
    rec["result"] = "pass"
    return PASS, rec


def _window(arg):
    calls, _, total = arg.partition(":")
    return int(calls), float(total)


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-only", action="store_true", help="print one request pair")
    mode.add_argument("--run", action="store_true", help="send one process's pairs")
    mode.add_argument("--reduce", nargs="+", metavar="LOG", help="reduce request logs")
    ap.add_argument("--model-path", help="the checkpoint folder (tokenizer and image config)")
    ap.add_argument("--process", default="A", choices=sorted(IMAGE_BASE))
    ap.add_argument("--pair", type=int, default=1, help="the pair --print-only prints")
    ap.add_argument("--pairs", type=int, default=WARMUP_PAIRS + COUNTED_PAIRS,
                    help="pairs --run sends (a smoke run sends 2)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", help="the request log --run writes")
    ap.add_argument("--tip", help="the commit the server runs, recorded in the log")
    ap.add_argument("--text-window", action="append", type=_window, default=[],
                    metavar="CALLS:SUM_MS", help="one text window's merge calls and sum")
    ap.add_argument("--text-ttft-ms", nargs="+", type=float, default=[],
                    help="the text TTFT process medians at 8K, 1 user")
    ap.add_argument("--accepted-ttft-ms", nargs="+", type=float, default=[],
                    help="the accepted text TTFT process medians at 8K, 1 user")
    args = ap.parse_args(argv)
    if args.reduce:
        logs = [(p, [json.loads(line) for line in Path(p).read_text(encoding="utf-8")
                     .splitlines() if line.strip()]) for p in args.reduce]
        code, record = reduce(logs, args.text_window, args.text_ttft_ms, args.accepted_ttft_ms)
        print(json.dumps(record))
        return code
    if not args.model_path:
        ap.error("--model-path is required")
    if args.run:
        if not args.out:
            ap.error("--run needs --out")
        return run(args)
    return print_only(args)


if __name__ == "__main__":
    sys.exit(main())
