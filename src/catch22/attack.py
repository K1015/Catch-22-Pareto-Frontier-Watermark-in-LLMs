from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .generate import output_paths as clean_paths
from .io import append_jsonl, display_path, extract_text, read_jsonl, write_json
from .paraphrase import Seq2SeqParaphraser
from .registry import require_method


def condition_name(attack: str, strength: str) -> str:
    return "dipper_moderate" if attack == "dipper" else "extreme_paraphrase"


def output_paths(output_dir: Path, method: str, condition: str) -> tuple[Path, Path]:
    base = output_dir / method / "attacks" / condition
    return base / "attacked.jsonl", base / "summary.json"


def _attack_model_name(config, attack: str, strength: str) -> str:
    for item in config.attacks:
        if item.get("name") == attack and item.get("strength") == strength:
            return str(item.get("model", "kalpeshk2011/dipper-paraphraser-xxl"))
    return "kalpeshk2011/dipper-paraphraser-xxl" if strength == "moderate" else "facebook/bart-large-cnn"


def run_attack(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    method = args.method
    require_method(method)
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    condition = condition_name(args.attack, args.paraphrase_strength)
    input_file, _ = clean_paths(output_dir, method)
    output_file, summary_file = output_paths(output_dir, method, condition)
    model_name = _attack_model_name(config, args.attack, args.paraphrase_strength)
    plan = {
        "stage": "attack",
        "track": config.track,
        "method": method,
        "attack": args.attack,
        "paraphrase_strength": args.paraphrase_strength,
        "condition": condition,
        "attack_model": model_name,
        "input_file": display_path(input_file, config.root),
        "output_file": display_path(output_file, config.root),
        "summary_file": display_path(summary_file, config.root),
        "local_backend": args.local_backend,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan
    source_rows = read_jsonl(input_file)
    existing = read_jsonl(output_file) if args.resume and output_file.exists() else []
    if output_file.exists() and not args.resume:
        output_file.unlink()
    paraphraser = Seq2SeqParaphraser(model_name, local_backend=args.local_backend)
    written = list(existing)
    for row in source_rows[len(existing) :]:
        attacked = paraphraser.apply(extract_text(row), strength=args.paraphrase_strength)
        out_row = dict(row)
        out_row.update(
            {
                "source_text": extract_text(row),
                "text": attacked.text,
                "condition": condition,
                "attack": args.attack,
                "paraphrase_strength": args.paraphrase_strength,
                "attack_backend": attacked.backend,
            }
        )
        append_jsonl(output_file, [out_row])
        written.append(out_row)
    summary = {**plan, "total_samples": len(written)}
    write_json(summary_file, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a moderate or extreme paraphrase attack.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--attack", required=True, choices=["dipper", "extreme-paraphrase"])
    parser.add_argument("--paraphrase-strength", required=True, choices=["moderate", "extreme"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-backend", action="store_true")
    return parser


def main() -> None:
    run_attack(build_parser().parse_args())


if __name__ == "__main__":
    main()
