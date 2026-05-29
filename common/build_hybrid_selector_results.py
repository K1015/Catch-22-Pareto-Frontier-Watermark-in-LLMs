#!/usr/bin/env python3
"""Build paper-faithful Hybrid results from fixed family representatives.

The paper's Hybrid is a selector over watermark families, not a logit-level
blend. This utility materializes that selector by copying the selected
representative's per-condition scored/evaluation artifacts into a Hybrid result
tree that the normal table renderer can consume.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.manifests import load_manifest
from common.method_registry import get_method_spec


DEFAULT_REPRESENTATIVES = {
    "biased": "unigram",
    "bias_free": "hcw",
    "semantic": "pmark",
    "distribution_preserving": "cgw",
}

DEFAULT_ATTACK_TO_METHOD = {
    "none": "cgw",
    "dipper": "pmark",
    "opt": "pmark",
    "wm-removal": "pmark",
    "synonym": "pmark",
    "backtranslation": "pmark",
    "summarization": "pmark",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-results-root", required=True)
    parser.add_argument("--output-results-root", required=True)
    parser.add_argument("--selector-summary-json", required=True)
    parser.add_argument(
        "--attack-method",
        action="append",
        default=[],
        metavar="ATTACK=METHOD",
        help="Override the selected representative for one attack condition.",
    )
    parser.add_argument(
        "--copy-all-evals",
        action="store_true",
        help="Copy non-hybrid evaluation JSONs so render_tables.py can render a full table.",
    )
    return parser.parse_args()


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _parse_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected ATTACK=METHOD override, got {value!r}")
        attack, method = value.split("=", 1)
        overrides[attack.strip()] = method.strip()
    return overrides


def _copy_method_evaluations(method: str, attacks: list[str], source_root: Path, output_root: Path) -> int:
    copied = 0
    for attack in attacks:
        if _copy_if_exists(
            source_root / method / "evaluations" / f"{attack}.json",
            output_root / method / "evaluations" / f"{attack}.json",
        ):
            copied += 1
    return copied


def _copy_selected_condition(
    *,
    source_method: str,
    attack: str,
    source_root: Path,
    output_root: Path,
) -> dict:
    spec = get_method_spec(source_method)
    copied: dict[str, object] = {
        "attack": attack,
        "selected_method": source_method,
        "selected_family": spec.family,
        "selected_family_representative": DEFAULT_REPRESENTATIVES.get(spec.family),
        "copied_files": [],
        "missing_files": [],
    }

    src_eval = source_root / source_method / "evaluations" / f"{attack}.json"
    dst_eval = output_root / "hybrid" / "evaluations" / f"{attack}.json"
    if not src_eval.exists():
        raise FileNotFoundError(f"Missing selected evaluation JSON: {src_eval}")

    summary = json.loads(src_eval.read_text(encoding="utf-8"))
    summary["hybrid_selector"] = {
        "selected_method": source_method,
        "selected_family": spec.family,
        "rule": "paper_fixed_representatives",
        "representatives": DEFAULT_REPRESENTATIVES,
    }
    dst_eval.parent.mkdir(parents=True, exist_ok=True)
    dst_eval.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    copied["copied_files"].append(str(dst_eval))

    for filename in [
        "positive_scored.jsonl",
        "negative_scored.jsonl",
        "positive_summary.json",
        "negative_summary.json",
    ]:
        src = source_root / source_method / "scored" / attack / filename
        dst = output_root / "hybrid" / "scored" / attack / filename
        if _copy_if_exists(src, dst):
            copied["copied_files"].append(str(dst))
        else:
            copied["missing_files"].append(str(src))

    if attack == "none":
        if _copy_if_exists(
            source_root / source_method / "raw" / "clean.jsonl",
            output_root / "hybrid" / "raw" / "clean.jsonl",
        ):
            copied["copied_files"].append(str(output_root / "hybrid" / "raw" / "clean.jsonl"))
    else:
        for filename in ["positive_attacked.jsonl", "negative_attacked.jsonl"]:
            src = source_root / source_method / "attacks" / attack / filename
            dst = output_root / "hybrid" / "attacks" / attack / filename
            if _copy_if_exists(src, dst):
                copied["copied_files"].append(str(dst))
            else:
                copied["missing_files"].append(str(src))

    return copied


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    source_root = Path(args.source_results_root).resolve()
    output_root = Path(args.output_results_root).resolve()
    selector_summary = Path(args.selector_summary_json).resolve()
    overrides = _parse_overrides(args.attack_method)

    selected_by_attack: dict[str, str] = {}
    for attack in manifest.attacks:
        selected_by_attack[attack] = overrides.get(attack, DEFAULT_ATTACK_TO_METHOD.get(attack, "pmark"))

    unknown_attacks = sorted(set(overrides) - set(manifest.attacks))
    if unknown_attacks:
        raise ValueError(f"Overrides reference attacks not in manifest: {unknown_attacks}")

    unknown_methods = sorted(set(selected_by_attack.values()) - set(manifest.methods))
    if unknown_methods:
        raise ValueError(f"Selected methods not in manifest: {unknown_methods}")

    copied_non_hybrid_evals = 0
    if args.copy_all_evals:
        for method in manifest.methods:
            if method == "hybrid":
                continue
            copied_non_hybrid_evals += _copy_method_evaluations(
                method,
                manifest.attacks,
                source_root,
                output_root,
            )

    selections = [
        _copy_selected_condition(
            source_method=method,
            attack=attack,
            source_root=source_root,
            output_root=output_root,
        )
        for attack, method in selected_by_attack.items()
    ]

    summary = {
        "source_results_root": str(source_root),
        "output_results_root": str(output_root),
        "manifest": str(Path(args.manifest).resolve()),
        "representatives": DEFAULT_REPRESENTATIVES,
        "selected_by_attack": selected_by_attack,
        "copy_all_evals": bool(args.copy_all_evals),
        "copied_non_hybrid_eval_count": copied_non_hybrid_evals,
        "selections": selections,
    }
    selector_summary.parent.mkdir(parents=True, exist_ok=True)
    selector_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
