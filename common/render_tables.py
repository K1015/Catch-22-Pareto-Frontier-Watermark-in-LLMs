#!/usr/bin/env python3
"""Render paper-facing tables from consolidated summary JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.manifests import load_manifest
from common.method_registry import get_method_spec, method_tex_label


ATTACK_LABELS = {
    "none": r"No attack",
    "dipper": r"DIPPER ($\hat\varepsilon\!\approx\!0.25$)",
    "opt": r"OPT-2.7B ($\hat\varepsilon\!\approx\!0.15$)",
    "wm-removal": r"WM-removal ($\hat\varepsilon\!\approx\!0.15$)",
    "synonym": r"Synonym ($\hat\varepsilon\!\approx\!0.15$)",
    "span-synonym": r"Span-synonym",
    "backtranslation": r"Back-trans.\ ($\hat\varepsilon\!\approx\!0.42$)",
    "summarization": r"Summ.\ ($\hat\varepsilon\!\approx\!0.55$)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--output-tex", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format_cell(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{decimals}f}"


def _method_condition_rows(manifest, results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for method in manifest.methods:
        spec = get_method_spec(method)
        condition_summaries: dict[str, dict | None] = {}
        for attack in manifest.attacks:
            summary_path = results_root / method / "evaluations" / f"{attack}.json"
            condition_summaries[attack] = _load_summary(summary_path)
        rows.append(
            {
                "method": method,
                "display_name": spec.display_name,
                "tex_label": method_tex_label(spec),
                "highlight": spec.highlight,
                "conditions": condition_summaries,
            }
        )
    return rows


def _render_tex(manifest, rows: list[dict]) -> str:
    attack_count = len(manifest.attacks)
    colspec = "l|" + "|".join(["ccc"] * attack_count)
    group_headers = []
    cmidrules = []
    col_start = 2
    for attack in manifest.attacks:
        group_headers.append(rf"\multicolumn{{3}}{{c|}}{{{ATTACK_LABELS.get(attack, attack)}}}")
        cmidrules.append(rf"\cmidrule(lr){{{col_start}-{col_start + 2}}}")
        col_start += 3
    if cmidrules:
        cmidrules[-1] = cmidrules[-1].replace("c|", "c}")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{manifest.table.caption if manifest.table else manifest.description}}}",
        rf"\label{{{manifest.table.label if manifest.table else 'tab:catch22'}}}",
        r"\small",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{adjustbox}{max width=\textwidth}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join([""] + group_headers) + r" \\",
        "".join(cmidrules),
        "Method & " + " & ".join(["AUC & TPR & $z$"] * attack_count) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        if row["highlight"]:
            lines.append("\\" + row["highlight"])
        cells = [row["tex_label"]]
        for attack in manifest.attacks:
            summary = row["conditions"].get(attack)
            auc = summary.get("auroc") if summary else None
            tpr = summary.get("tpr_at_1_fpr") if summary else None
            z_val = None
            if summary:
                z_val = summary.get(
                    "table_keyless_z_mean_standardized",
                    summary.get(
                        "external_keyless_z",
                        summary.get("table_keyless_z", summary.get("external_keyless_condition_z")),
                    ),
                )
            cells.extend(
                [
                    _format_cell(auc),
                    _format_cell(tpr),
                    _format_cell(z_val),
                ]
            )
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    repo_root = Path(__file__).resolve().parents[1]
    results_root = Path(args.results_root).resolve() if args.results_root else (repo_root / "results" / manifest.results_subdir)
    output_tex = Path(args.output_tex).resolve() if args.output_tex else (manifest.manifest_path.parent / manifest.table.output_tex)
    output_json = Path(args.output_json).resolve() if args.output_json else (manifest.manifest_path.parent / manifest.table.output_json)

    rows = _method_condition_rows(manifest, results_root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    tex = _render_tex(manifest, rows)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(tex, encoding="utf-8")

    print(json.dumps({"rows": len(rows), "output_tex": str(output_tex), "output_json": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
