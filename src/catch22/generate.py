from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .io import append_jsonl, display_model_name, display_path, extract_prompt, extract_reference, read_jsonl, write_json
from .modeling import TextGenerator
from .registry import require_method
from .watermarks import detect_text


def output_paths(output_dir: Path, method: str) -> tuple[Path, Path]:
    base = output_dir / method / "clean"
    return base / "generations.jsonl", base / "summary.json"


def run_generation(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    method = args.method
    require_method(method)
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    model_name = args.model_name_or_path or config.model_name
    output_file, summary_file = output_paths(output_dir, method)
    plan = {
        "stage": "generate",
        "track": config.track,
        "method": method,
        "model_name_or_path": display_model_name(model_name),
        "dataset_path": display_path(config.dataset_path, config.root),
        "output_file": display_path(output_file, config.root),
        "summary_file": display_path(summary_file, config.root),
        "num_samples": args.num_samples or config.num_samples,
        "local_backend": args.local_backend,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan

    source_rows = read_jsonl(config.dataset_path)[: plan["num_samples"]]
    existing = read_jsonl(output_file) if args.resume and output_file.exists() else []
    if output_file.exists() and not args.resume:
        output_file.unlink()
    generator = TextGenerator(
        model_name,
        load_in_4bit=config.load_in_4bit,
        local_backend=args.local_backend,
        seed=args.seed,
    )
    written = list(existing)
    for local_index, row in enumerate(source_rows[len(existing) :], start=len(existing)):
        prompt = extract_prompt(row)
        generated = generator.generate(
            prompt,
            method=method,
            sample_index=local_index,
            max_new_tokens=args.max_new_tokens or config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
        detection = detect_text(generated.text, method, seed=args.seed).__dict__
        out_row = {
            "sample_index": local_index,
            "prompt": prompt,
            "reference": extract_reference(row),
            "text": generated.text,
            "method": method,
            "condition": "clean",
            "model_name_or_path": display_model_name(model_name),
            "generation_backend": generated.backend,
            "detection": detection,
        }
        append_jsonl(output_file, [out_row])
        written.append(out_row)
    summary = {
        **plan,
        "total_samples": len(written),
        "detection_rate": sum(1 for row in written if row["detection"]["is_watermarked"]) / max(len(written), 1),
    }
    write_json(summary_file, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate clean LFQA outputs for one watermark method.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-backend", action="store_true", help="Use deterministic local text generation for environment checks.")
    return parser


def main() -> None:
    run_generation(build_parser().parse_args())


if __name__ == "__main__":
    main()
