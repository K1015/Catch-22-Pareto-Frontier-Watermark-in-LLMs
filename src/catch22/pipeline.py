from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from .attack import run_attack
from .config import load_config
from .evaluate import run_evaluate
from .generate import run_generation
from .io import display_path
from .registry import PAPER_CONDITIONS
from .render import run_render
from .score import run_score


def _namespace(**kwargs) -> Namespace:
    return Namespace(**kwargs)


def build_steps(config_path: str, output_dir: str | None, num_samples: int | None, local_backend: bool, dry_run: bool):
    config = load_config(config_path)
    conditions = PAPER_CONDITIONS
    steps = []
    steps.append(("generate", "vanilla", "clean"))
    for attack in config.attacks:
        steps.append(("attack", "vanilla", attack["condition"]))
    for method in config.methods:
        steps.append(("generate", method, "clean"))
        for attack in config.attacks:
            steps.append(("attack", method, attack["condition"]))
        for condition in conditions:
            steps.append(("score", method, condition))
            steps.append(("score_baseline", method, f"vanilla:{condition}"))
        for condition in conditions:
            steps.append(("evaluate", method, condition))
    steps.append(("render", "*", "*"))
    return {
        "config": config_path,
        "track": config.track,
        "output_dir": display_path(Path(output_dir) if output_dir else config.output_dir, config.root),
        "num_samples": num_samples or config.num_samples,
        "local_backend": local_backend,
        "dry_run": dry_run,
        "steps": steps,
    }


def run_pipeline(args: argparse.Namespace) -> dict:
    plan = build_steps(args.config, args.output_dir, args.num_samples, args.local_backend, args.dry_run)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan
    config = load_config(args.config)
    run_generation(
        _namespace(
            config=args.config,
            method="vanilla",
            output_dir=args.output_dir,
            model_name_or_path=args.model_name_or_path,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            resume=args.resume,
            dry_run=False,
            local_backend=args.local_backend,
        )
    )
    for attack in config.attacks:
        run_attack(
            _namespace(
                config=args.config,
                method="vanilla",
                attack=attack["name"],
                paraphrase_strength=attack["strength"],
                output_dir=args.output_dir,
                resume=args.resume,
                dry_run=False,
                local_backend=args.local_backend,
            )
        )
    for method in config.methods:
        run_generation(
            _namespace(
                config=args.config,
                method=method,
                output_dir=args.output_dir,
                model_name_or_path=args.model_name_or_path,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                resume=args.resume,
                dry_run=False,
                local_backend=args.local_backend,
            )
        )
        for attack in config.attacks:
            run_attack(
                _namespace(
                    config=args.config,
                    method=method,
                    attack=attack["name"],
                    paraphrase_strength=attack["strength"],
                    output_dir=args.output_dir,
                    resume=args.resume,
                    dry_run=False,
                    local_backend=args.local_backend,
                )
            )
        for condition in PAPER_CONDITIONS:
            run_score(
                _namespace(
                    config=args.config,
                    method=method,
                    source_method=None,
                    condition=condition,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    resume=args.resume,
                    dry_run=False,
                )
            )
            run_score(
                _namespace(
                    config=args.config,
                    method=method,
                    source_method="vanilla",
                    condition=condition,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    resume=args.resume,
                    dry_run=False,
                )
            )
        run_evaluate(
            _namespace(
                config=args.config,
                method=method,
                condition=None,
                output_dir=args.output_dir,
                dry_run=False,
            )
        )
    run_render(_namespace(config=args.config, output_dir=args.output_dir, dry_run=False))
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Catch-22 LFQA reproduction suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reproduction-suite", action="store_true", help="Run clean, moderate paraphrase, and extreme paraphrase.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-backend", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.reproduction_suite:
        raise SystemExit("Pass --reproduction-suite to run the default reproduction path.")
    run_pipeline(args)


if __name__ == "__main__":
    main()
