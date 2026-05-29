from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .evaluate import output_path as eval_path
from .io import display_path, write_json
from .registry import PAPER_CONDITIONS, require_method


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def run_render(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    table_json = output_dir / "tables" / "results.json" if args.output_dir else config.table_json
    table_tex = output_dir / "tables" / "results.tex" if args.output_dir else config.table_tex
    rows = []
    for method in config.methods:
        spec = require_method(method)
        row = {"method": method, "display_name": spec.display_name, "family": spec.family}
        for condition in PAPER_CONDITIONS:
            path = eval_path(output_dir, method, condition)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                row[f"{condition}_detection_rate"] = payload.get("detection_rate")
                row[f"{condition}_retention"] = payload.get("robustness_retention")
            else:
                row[f"{condition}_detection_rate"] = None
                row[f"{condition}_retention"] = None
        rows.append(row)
    payload = {
        "track": config.track,
        "conditions": PAPER_CONDITIONS,
        "rows": rows,
        "table_json": display_path(table_json, config.root),
        "table_tex": display_path(table_tex, config.root),
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "render",
                    "json": display_path(table_json, config.root),
                    "tex": display_path(table_tex, config.root),
                },
                indent=2,
            )
        )
        return payload
    write_json(table_json, payload)
    table_tex.parent.mkdir(parents=True, exist_ok=True)
    header = "Method & Clean detect. & Moderate retain. & Extreme retain. \\\\\n"
    rendered = ["\\begin{tabular}{lrrr}\n", header, "\\hline\n"]
    for row in rows:
        def fmt(value):
            return "NA" if value is None else f"{value:.3f}"
        rendered.append(
            f"{_latex_escape(row['display_name'])} & {fmt(row['clean_detection_rate'])} & "
            f"{fmt(row['dipper_moderate_retention'])} & {fmt(row['extreme_paraphrase_retention'])} \\\\\n"
        )
    rendered.append("\\end{tabular}\n")
    table_tex.write_text("".join(rendered), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render JSON and LaTeX tables from evaluations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    run_render(build_parser().parse_args())


if __name__ == "__main__":
    main()
