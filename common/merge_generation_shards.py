#!/usr/bin/env python3
"""Merge generation shard JSONL files into one raw file plus summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.io_utils import read_jsonl
from common.run_generation import compute_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--expected-total", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = [Path(path).resolve() for path in args.input_files]
    output_path = Path(args.output_file).resolve()
    summary_path = Path(args.summary_file).resolve()

    rows: list[dict] = []
    for path in input_files:
        rows.extend(read_jsonl(path))

    rows.sort(key=lambda row: int(row.get("sample_index", 0)))

    if args.expected_total is not None and len(rows) != args.expected_total:
        raise ValueError(
            f"Expected {args.expected_total} merged rows but found {len(rows)}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = compute_summary(rows)
    summary["config"] = {
        "input_files": [str(path) for path in input_files],
        "output_file": str(output_path),
        "merged_rows": len(rows),
        "expected_total": args.expected_total,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
