from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .io import display_path, read_jsonl, summarize_scores, write_json
from .registry import PAPER_CONDITIONS
from .score import output_paths as score_paths


def output_path(output_dir: Path, method: str, condition: str) -> Path:
    return output_dir / method / "evaluations" / f"{condition}.json"


def run_evaluate(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    methods = [args.method] if args.method else config.methods
    conditions = [args.condition] if args.condition else PAPER_CONDITIONS
    plans = []
    for method in methods:
        clean_score_file, _ = score_paths(output_dir, method, "clean")
        clean_rate = None
        if clean_score_file.exists():
            clean_rate = summarize_scores(read_jsonl(clean_score_file))["detection_rate"]
        for condition in conditions:
            score_file, _ = score_paths(output_dir, method, condition)
            target = output_path(output_dir, method, condition)
            plan = {
                "stage": "evaluate",
                "track": config.track,
                "method": method,
                "condition": condition,
                "score_file": display_path(score_file, config.root),
                "output_json": display_path(target, config.root),
            }
            if args.dry_run:
                plans.append(plan)
                continue
            rows = read_jsonl(score_file)
            metrics = summarize_scores(rows)
            metrics["clean_detection_rate"] = clean_rate
            metrics["robustness_retention"] = (
                metrics["detection_rate"] / clean_rate if clean_rate and condition != "clean" else 1.0
            )
            payload = {**plan, **metrics}
            write_json(target, payload)
            plans.append(payload)
    result = {"evaluations": plans}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute detectability and robustness metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    run_evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
