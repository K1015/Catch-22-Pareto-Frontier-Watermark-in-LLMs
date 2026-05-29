from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.io_utils import (
    build_unigram_calibration,
    compute_auroc_tpr,
    compute_unigram_surprisal,
    condition_level_z,
    extract_generated_text,
    mean_standardized_z,
    mean_confidence_interval,
    regex_tokenize,
    read_jsonl,
    safe_mean,
    write_csv_row,
)
from common.dawa_utils import ensure_dir


def load_tokenizer_with_fallback(model_name: str):
    from transformers import AutoTokenizer

    if "mistral" in model_name.lower():
        print(f"Using slow tokenizer for {model_name} to avoid fast-tokenizer hangs.", file=sys.stderr)
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:
        print(f"Fast tokenizer load failed for {model_name}; retrying with use_fast=False: {exc}", file=sys.stderr)
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scored outputs against vanilla controls.")
    parser.add_argument("--positive-file", required=True)
    parser.add_argument("--negative-file", required=True)
    parser.add_argument("--keyless-calibration-file", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--condition-label", default=None)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    return parser.parse_args()


def attack_name(rows: list[dict], fallback: str | None) -> str:
    if fallback:
        return fallback
    for row in rows:
        stats = row.get("attack_stats", {})
        if stats.get("attack_name"):
            return str(stats["attack_name"])
    return "none"


def attack_metric(rows: list[dict], key: str) -> list[float]:
    return [float(row.get("attack_stats", {}).get(key, 0.0)) for row in rows]


def main() -> None:
    args = parse_args()
    ensure_dir(Path(args.output_json).parent)
    if args.output_csv:
        ensure_dir(Path(args.output_csv).parent)

    positive_rows = read_jsonl(args.positive_file)
    negative_rows = read_jsonl(args.negative_file)
    calibration_rows = read_jsonl(args.keyless_calibration_file)

    if not positive_rows or not negative_rows:
        raise RuntimeError("Positive and negative files must both contain scored rows.")

    positive_scores = [row["detection_stats"]["score"] for row in positive_rows]
    negative_scores = [row["detection_stats"]["score"] for row in negative_rows]
    auroc, tpr_at_1 = compute_auroc_tpr(positive_scores, negative_scores, target_fpr=args.target_fpr)

    if args.tokenizer_model == "regex":
        class RegexTokenizer:
            vocab_size = 50000

            @staticmethod
            def encode(text: str, add_special_tokens: bool = False):
                del add_special_tokens
                return regex_tokenize(text)

        tokenizer = RegexTokenizer()
    else:
        tokenizer = load_tokenizer_with_fallback(args.tokenizer_model)
    calibration = build_unigram_calibration(calibration_rows, tokenizer)
    positive_surprisals = [compute_unigram_surprisal(extract_generated_text(row), tokenizer, calibration) for row in positive_rows]
    keyless_mean_z = mean_standardized_z(positive_surprisals, calibration["null_mean"], calibration["null_std"])
    keyless_condition_z = condition_level_z(positive_surprisals, calibration["null_mean"], calibration["null_std"])

    positive_edit_rates = [row.get("attack_stats", {}).get("realized_edit_rate", 0.0) for row in positive_rows]
    negative_edit_rates = [row.get("attack_stats", {}).get("realized_edit_rate", 0.0) for row in negative_rows]
    positive_edit_mean, positive_edit_ci = mean_confidence_interval(positive_edit_rates)
    negative_edit_mean, negative_edit_ci = mean_confidence_interval(negative_edit_rates)
    positive_run_lengths = attack_metric(positive_rows, "mean_edited_run_length")
    negative_run_lengths = attack_metric(negative_rows, "mean_edited_run_length")
    positive_max_run_lengths = attack_metric(positive_rows, "max_edited_run_length")
    negative_max_run_lengths = attack_metric(negative_rows, "max_edited_run_length")
    positive_num_runs = attack_metric(positive_rows, "num_edited_runs")
    negative_num_runs = attack_metric(negative_rows, "num_edited_runs")

    positive_tokens = [row["detection_stats"]["num_tokens"] for row in positive_rows]
    negative_tokens = [row["detection_stats"]["num_tokens"] for row in negative_rows]
    positive_detected = [float(row["detection_stats"]["is_watermarked"]) for row in positive_rows]
    negative_detected = [float(row["detection_stats"]["is_watermarked"]) for row in negative_rows]

    condition = attack_name(positive_rows, args.condition_label)
    summary = {
        "condition": condition,
        "positive_file": args.positive_file,
        "negative_file": args.negative_file,
        "keyless_calibration_file": args.keyless_calibration_file,
        "num_positive": len(positive_rows),
        "num_negative": len(negative_rows),
        "auroc": float(auroc),
        "tpr_at_1_fpr": float(tpr_at_1),
        "positive_mean_score": safe_mean(positive_scores),
        "positive_std_score": float(np.std(positive_scores)),
        "negative_mean_score": safe_mean(negative_scores),
        "negative_std_score": float(np.std(negative_scores)),
        "mean_score_gap": safe_mean(positive_scores) - safe_mean(negative_scores),
        "positive_mean_tokens": safe_mean(positive_tokens),
        "negative_mean_tokens": safe_mean(negative_tokens),
        "positive_detection_rate": safe_mean(positive_detected),
        "negative_detection_rate": safe_mean(negative_detected),
        "positive_mean_edit_rate": positive_edit_mean,
        "positive_edit_rate_ci95": positive_edit_ci,
        "negative_mean_edit_rate": negative_edit_mean,
        "negative_edit_rate_ci95": negative_edit_ci,
        "positive_mean_num_edited_runs": safe_mean(positive_num_runs),
        "negative_mean_num_edited_runs": safe_mean(negative_num_runs),
        "positive_mean_edited_run_length": safe_mean(positive_run_lengths),
        "negative_mean_edited_run_length": safe_mean(negative_run_lengths),
        "positive_mean_max_edited_run_length": safe_mean(positive_max_run_lengths),
        "negative_mean_max_edited_run_length": safe_mean(negative_max_run_lengths),
        "external_keyless_mean_surprisal": safe_mean(positive_surprisals),
        "external_keyless_z": float(keyless_mean_z),
        "external_keyless_condition_z": float(keyless_condition_z),
        "table_keyless_z": float(keyless_condition_z),
        "table_keyless_z_mean_standardized": float(keyless_mean_z),
        "null_surprisal_mean": float(calibration["null_mean"]),
        "null_surprisal_std": float(calibration["null_std"]),
    }

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if args.output_csv:
        write_csv_row(args.output_csv, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
