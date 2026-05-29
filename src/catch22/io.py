from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean, median
from typing import Iterable


def display_path(path: str | Path, root: str | Path | None = None) -> str:
    """Return a path suitable for logs and JSON artifacts without local identity leaks."""
    source = Path(path)
    bases = [Path.cwd()]
    if root is not None:
        bases.insert(0, Path(root))
    for base in bases:
        try:
            return source.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    if source.is_absolute():
        return f"<external>/{source.name}"
    return source.as_posix()


def display_model_name(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or os.sep in value:
        return display_path(path)
    return value


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    rows: list[dict] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def extract_prompt(row: dict) -> str:
    if "prompt" in row:
        return str(row["prompt"])
    if "prefix" in row:
        return str(row["prefix"])
    title = str(row.get("title", "")).strip()
    body = str(row.get("selftext", "")).strip()
    if title or body:
        return f"Q: {title}\n{body}\nA:".strip()
    raise KeyError("Input row must contain prompt, prefix, or title/selftext fields.")


def extract_reference(row: dict) -> str | None:
    for key in ("reference", "gold_completion", "answer"):
        if key in row:
            value = row[key]
            if isinstance(value, list):
                return str(value[0]) if value else None
            return str(value)
    return None


def extract_text(row: dict) -> str:
    if "text" in row:
        return str(row["text"])
    if "completion" in row:
        return str(row["completion"])
    if "gen_completion" in row:
        value = row["gen_completion"]
        if isinstance(value, list):
            return str(value[0]) if value else ""
        return str(value)
    raise KeyError("Row must contain text, completion, or gen_completion.")


def summarize_scores(rows: list[dict]) -> dict:
    scores = [float(row.get("detection", {}).get("score", 0.0)) for row in rows]
    z_scores = [float(row.get("detection", {}).get("z_score", 0.0)) for row in rows]
    p_values = [float(row.get("detection", {}).get("p_value", 1.0)) for row in rows]
    detected = [1.0 if row.get("detection", {}).get("is_watermarked") else 0.0 for row in rows]
    token_counts = [int(row.get("detection", {}).get("num_tokens", 0)) for row in rows]
    return {
        "total_samples": len(rows),
        "mean_score": float(mean(scores)) if scores else 0.0,
        "mean_z_score": float(mean(z_scores)) if z_scores else 0.0,
        "median_z_score": float(median(z_scores)) if z_scores else 0.0,
        "mean_p_value": float(mean(p_values)) if p_values else 1.0,
        "detection_rate": float(mean(detected)) if detected else 0.0,
        "mean_tokens": float(mean(token_counts)) if token_counts else 0.0,
    }


def auroc_from_scores(positive_scores: list[float], negative_scores: list[float]) -> float | None:
    """Compute AUROC from positive and negative detector scores.

    The rank formulation handles ties by assigning tied observations their
    average rank, matching the Mann-Whitney U interpretation of AUROC.
    """
    if not positive_scores or not negative_scores:
        return None
    labeled = [(float(score), 1) for score in positive_scores] + [(float(score), 0) for score in negative_scores]
    labeled.sort(key=lambda item: item[0])
    rank_sum_positive = 0.0
    rank = 1
    index = 0
    while index < len(labeled):
        next_index = index + 1
        while next_index < len(labeled) and labeled[next_index][0] == labeled[index][0]:
            next_index += 1
        average_rank = (rank + rank + (next_index - index) - 1) / 2.0
        positives_in_tie = sum(label for _, label in labeled[index:next_index])
        rank_sum_positive += positives_in_tie * average_rank
        rank += next_index - index
        index = next_index
    n_pos = len(positive_scores)
    n_neg = len(negative_scores)
    auc = (rank_sum_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))
