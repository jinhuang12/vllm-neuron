# SPDX-License-Identifier: Apache-2.0
"""Read TPOT and accepted tokens per step from ``vllm bench serve`` result files.

``tpot`` reads ``--save-detailed`` result files. Per request, TPOT is the time from the
first to the last output chunk over the server's completion tokens minus one: the sum
of the request's ``itls`` over ``output_lens - 1``. A speculative chunk carries several
tokens, so the mean chunk gap is not a TPOT. Per file it prints each request's value,
their median, each request's chunk-gap count and its guard count: the gaps above 2x
that request's median gap.

``accepted`` reads the speculative-decoding counter deltas the bench writes into each
result file and prints 1 + accepted tokens / drafts, checked against the file's own
``spec_decode_acceptance_length``.

Standard library only. Prints one JSON record. Exit 0 when every file reads, 2 when a
file is invalid: a request with fewer than 2 output tokens, missing fields, 0 drafts,
or a reading that differs from the file's own.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

OK, INVALID = 0, 2
SPEC_FIELDS = ("spec_decode_num_drafts", "spec_decode_draft_tokens",
               "spec_decode_accepted_tokens", "spec_decode_acceptance_length")


def request_tpot(gaps_s, tokens):
    """Return (TPOT in ms, guard count) of one request's chunk gaps in seconds."""
    if tokens < 2:
        raise ValueError(f"{tokens} output tokens, need at least 2")
    median = statistics.median(gaps_s) if gaps_s else 0.0
    return 1000.0 * sum(gaps_s) / (tokens - 1), sum(1 for g in gaps_s if g > 2.0 * median)


def read_tpot(result):
    """Return the TPOT record of one result file; raise ValueError when it is invalid."""
    itls, out_lens = result.get("itls"), result.get("output_lens")
    if not isinstance(itls, list) or not isinstance(out_lens, list) or len(itls) != len(out_lens):
        raise ValueError("needs itls and output_lens lists of one length")
    values, guards = [], []
    for i, (gaps, tokens) in enumerate(zip(itls, out_lens)):
        try:
            value, guard = request_tpot(gaps, tokens)
        except ValueError as e:
            raise ValueError(f"request {i}: {e}") from None
        values.append(value)
        guards.append(guard)
    return {"requests_ms": values, "median_ms": statistics.median(values),
            "chunk_gaps": [len(g) for g in itls], "guard_counts": guards,
            "bench_median_tpot_ms": result.get("median_tpot_ms")}


def read_accepted(result):
    """Return the accepted-tokens record of one result file; raise ValueError when invalid."""
    missing = [k for k in SPEC_FIELDS if k not in result]
    if missing:
        raise ValueError(f"no speculative fields: {missing}")
    drafts = result["spec_decode_num_drafts"]
    accepted = result["spec_decode_accepted_tokens"]
    if drafts <= 0:
        raise ValueError(f"{drafts} drafts")
    value = 1 + accepted / drafts
    if value != result["spec_decode_acceptance_length"]:
        raise ValueError(f"1 + {accepted} / {drafts} = {value}, the file reads "
                         f"{result['spec_decode_acceptance_length']}")
    return {"accepted_tokens_per_step": value, "drafts": drafts, "accepted_tokens": accepted,
            "draft_tokens": result["spec_decode_draft_tokens"],
            "per_position_rates": result.get("spec_decode_per_position_acceptance_rates")}


READERS = {"tpot": read_tpot, "accepted": read_accepted}


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reading", choices=sorted(READERS))
    ap.add_argument("paths", nargs="+", help="result files, or folders of *.json result files")
    args = ap.parse_args(argv)
    files = []
    for p in map(Path, args.paths):
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    record, invalid = {}, {}
    for f in files:
        try:
            record[str(f)] = READERS[args.reading](json.loads(f.read_text(encoding="utf-8")))
        except ValueError as e:
            invalid[str(f)] = str(e)
    print(json.dumps({"reading": args.reading, "files": record, "invalid": invalid}))
    return INVALID if invalid else OK


if __name__ == "__main__":
    sys.exit(main())
