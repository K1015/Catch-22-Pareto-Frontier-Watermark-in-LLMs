from __future__ import annotations

import argparse
import json
from pathlib import Path

from .attack import output_paths as attack_paths
from .config import load_config
from .generate import output_paths as clean_paths
from .io import append_jsonl, display_path, extract_text, read_jsonl, summarize_scores, write_json
from .registry import require_method
from .watermarks import detect_text


def output_paths(output_dir: Path, method: str, condition: str, source_method: str | None = None) -> tuple[Path, Path]:
    if source_method and source_method != method:
        base = output_dir / method / "scored_baseline" / source_method / condition
    else:
        base = output_dir / method / "scored" / condition
    return base / "scored.jsonl", base / "summary.json"


def input_path(output_dir: Path, method: str, condition: str) -> Path:
    if condition == "clean":
        return clean_paths(output_dir, method)[0]
    return attack_paths(output_dir, method, condition)[0]


def run_score(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    method = args.method
    require_method(method)
    source_method = args.source_method or method
    require_method(source_method)
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    source = input_path(output_dir, source_method, args.condition)
    output_file, summary_file = output_paths(output_dir, method, args.condition, source_method=source_method)
    plan = {
        "stage": "score",
        "track": config.track,
        "method": method,
        "source_method": source_method,
        "score_label": "positive" if source_method == method else "baseline",
        "condition": args.condition,
        "input_file": display_path(source, config.root),
        "output_file": display_path(output_file, config.root),
        "summary_file": display_path(summary_file, config.root),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan
    rows = read_jsonl(source)
    if output_file.exists() and not args.resume:
        output_file.unlink()
    existing = read_jsonl(output_file) if args.resume and output_file.exists() else []
    written = list(existing)
    for row in rows[len(existing) :]:
        out_row = dict(row)
        out_row["detection"] = detect_text(extract_text(row), method, seed=args.seed).__dict__
        out_row["scored_with_method"] = method
        out_row["source_method"] = source_method
        out_row["score_label"] = "positive" if source_method == method else "baseline"
        append_jsonl(output_file, [out_row])
        written.append(out_row)
    summary = {**plan, **summarize_scores(written)}
    write_json(summary_file, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score generated or attacked outputs with the method detector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--source-method", default=None, help="Input method to score. Defaults to --method.")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    run_score(build_parser().parse_args())


if __name__ == "__main__":
    main()
