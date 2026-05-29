#!/usr/bin/env python3
"""Validate a consolidated Catch-22 manifest and any materialized artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.manifests import load_manifest


@dataclass(frozen=True)
class PathCheck:
    label: str
    path: Path
    expected_rows: int | None = None
    kind: str = "path"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="Validate the manifest and expected layout without requiring artifacts to exist.",
    )
    return parser.parse_args()


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _checks_for_manifest(results_root: Path, manifest) -> list[PathCheck]:
    checks: list[PathCheck] = []

    if manifest.attacks:
        checks.extend(
            [
                PathCheck("vanilla raw clean", results_root / "vanilla" / "raw" / "clean.jsonl", manifest.num_samples, "jsonl"),
                PathCheck("vanilla clean summary", results_root / "vanilla" / "summary" / "clean_generation.json", kind="json"),
            ]
        )

    for method in manifest.methods:
        method_root = results_root / method
        checks.extend(
            [
                PathCheck(f"{method} raw clean", method_root / "raw" / "clean.jsonl", manifest.num_samples, "jsonl"),
                PathCheck(f"{method} clean summary", method_root / "summary" / "clean_generation.json", kind="json"),
            ]
        )
        for attack in manifest.attacks:
            scored_root = method_root / "scored" / attack
            checks.extend(
                [
                    PathCheck(f"{method} scored positive {attack}", scored_root / "positive_scored.jsonl", manifest.num_samples, "jsonl"),
                    PathCheck(f"{method} scored negative {attack}", scored_root / "negative_scored.jsonl", manifest.num_samples, "jsonl"),
                    PathCheck(f"{method} scored positive summary {attack}", scored_root / "positive_summary.json", kind="json"),
                    PathCheck(f"{method} scored negative summary {attack}", scored_root / "negative_summary.json", kind="json"),
                    PathCheck(f"{method} evaluation {attack}", method_root / "evaluations" / f"{attack}.json", kind="json"),
                ]
            )
            if attack != "none":
                attack_root = method_root / "attacks" / attack
                checks.extend(
                    [
                        PathCheck(f"{method} attacked positive {attack}", attack_root / "positive_attacked.jsonl", manifest.num_samples, "jsonl"),
                        PathCheck(f"{method} attacked negative {attack}", attack_root / "negative_attacked.jsonl", manifest.num_samples, "jsonl"),
                        PathCheck(f"{method} attacked positive summary {attack}", attack_root / "positive_summary.json", kind="json"),
                        PathCheck(f"{method} attacked negative summary {attack}", attack_root / "negative_summary.json", kind="json"),
                    ]
                )

    if manifest.table:
        checks.extend(
            [
                PathCheck("rendered table tex", (manifest.manifest_path.parent / manifest.table.output_tex).resolve()),
                PathCheck("rendered table json", (manifest.manifest_path.parent / manifest.table.output_json).resolve(), kind="json"),
            ]
        )

    return checks


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    repo_root = Path(__file__).resolve().parents[1]
    results_root = Path(args.results_root).resolve() if args.results_root else (repo_root / "results" / manifest.results_subdir).resolve()

    report: dict[str, object] = {
        "manifest": str(manifest.manifest_path),
        "track_name": manifest.track_name,
        "results_root": str(results_root),
        "num_samples": manifest.num_samples,
        "methods": manifest.methods,
        "attacks": manifest.attacks,
        "missing": [],
        "row_mismatches": [],
        "json_parse_errors": [],
        "checked": 0,
        "validated": 0,
    }

    checks = _checks_for_manifest(results_root, manifest)
    for check in checks:
        report["checked"] += 1
        if args.layout_only:
            continue
        if not check.path.exists():
            report["missing"].append({"label": check.label, "path": str(check.path)})
            continue
        try:
            if check.kind == "jsonl":
                rows = _count_jsonl_rows(check.path)
                if check.expected_rows is not None and rows != check.expected_rows:
                    report["row_mismatches"].append(
                        {
                            "label": check.label,
                            "path": str(check.path),
                            "expected_rows": check.expected_rows,
                            "actual_rows": rows,
                        }
                    )
                else:
                    report["validated"] += 1
            elif check.kind == "json":
                _load_json(check.path)
                report["validated"] += 1
            else:
                report["validated"] += 1
        except Exception as exc:  # pragma: no cover
            report["json_parse_errors"].append({"label": check.label, "path": str(check.path), "error": str(exc)})

    failed = bool(report["row_mismatches"] or report["json_parse_errors"])
    if args.require_complete and not args.layout_only:
        failed = failed or bool(report["missing"])

    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
