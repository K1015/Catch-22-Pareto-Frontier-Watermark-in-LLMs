#!/usr/bin/env python3
"""Merge sharded LFQA score outputs into a single scored file plus summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.io_utils import read_jsonl
from common.score_outputs import compute_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--expected-total", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    shard_paths = sorted(input_dir.glob("shard_*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {input_dir}")

    merged_rows: list[dict] = []
    for shard_path in shard_paths:
        merged_rows.extend(read_jsonl(shard_path))

    merged_rows.sort(key=lambda row: int(row.get("sample_index", 0)))

    if args.expected_total is not None and len(merged_rows) != args.expected_total:
        raise ValueError(
            f"Expected {args.expected_total} merged rows but found {len(merged_rows)} in {input_dir}."
        )

    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row) + "\n")

    summary = compute_summary(merged_rows)
    summary["config"] = {
        "input_dir": str(input_dir),
        "num_shards": len(shard_paths),
        "expected_total": args.expected_total,
        "shard_files": [str(path) for path in shard_paths],
        "output_file": str(output_path),
    }
    summary_path = Path(args.summary_file).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
