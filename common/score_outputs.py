#!/usr/bin/env python3
"""Rescore consolidated Catch-22 outputs with the native verifier for each method."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.dawa_utils import load_model
from common.dawa_watermark import Watermark
from common.io_utils import extract_generated_text, extract_prompt, read_jsonl
from common.manifests import load_manifest
from common.method_registry import get_method_spec, list_methods
from common.watermark_adapters import WatermarkExperimentAdapter


TEXT_BOUND_METADATA_KEYS = {
    "completion_token_ids",
    "last_completion_token_ids",
    "pmark_sentences",
    "pmark_sentence_logs",
}


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--method", choices=list_methods(include_vanilla=True), required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--auxiliary-model", default=None)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-save", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True)
    return parser.parse_args()


def compute_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    scores = [float(row["detection_stats"].get("score", 0.0)) for row in rows]
    detections = [float(bool(row["detection_stats"].get("is_watermarked", False))) for row in rows]
    token_counts = [int(row["detection_stats"].get("num_tokens", 0)) for row in rows]
    return {
        "total_samples": len(rows),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "median_score": float(np.median(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "mean_tokens": float(np.mean(token_counts)),
        "detection_rate": float(np.mean(detections)),
    }


def scoring_metadata(row: dict) -> dict:
    metadata = dict(row.get("generation_metadata") or {})
    generation_params = row.get("generation_params") or {}
    if isinstance(generation_params, dict):
        for key in ("temperature", "top_p", "top_k"):
            if key in generation_params and key not in metadata:
                metadata[key] = generation_params[key]

    attack_name = (row.get("attack_stats") or {}).get("attack_name")
    if attack_name and attack_name != "none":
        for key in TEXT_BOUND_METADATA_KEYS:
            metadata.pop(key, None)
    return metadata


def load_dawa_detector(model_name: str, auxiliary_model_name: str | None, spec_kwargs: dict, torch_dtype_name: str):
    watermark_model, watermark_tokenizer = load_model(model_name, torch_dtype_name=torch_dtype_name)
    if auxiliary_model_name and auxiliary_model_name != model_name:
        auxiliary_model, auxiliary_tokenizer = load_model(auxiliary_model_name, torch_dtype_name=torch_dtype_name)
    else:
        auxiliary_model = watermark_model
        auxiliary_tokenizer = watermark_tokenizer

    device = next(watermark_model.parameters()).device
    watermark = Watermark(
        device=device,
        watermark_tokenizer=watermark_tokenizer,
        watermark_model=watermark_model,
        auxiliary_tokenizer=auxiliary_tokenizer,
        auxiliary_model=auxiliary_model,
        alpha=spec_kwargs.get("alpha", 0.2),
        top_p=spec_kwargs.get("top_p", 1.0),
        repetition_penalty=spec_kwargs.get("repetition_penalty", 1.0),
        no_repeat_ngram_size=spec_kwargs.get("no_repeat_ngram_size", 0),
        max_new_tokens=spec_kwargs.get("max_new_tokens", 300),
        min_new_tokens=spec_kwargs.get("min_new_tokens", 200),
        key=spec_kwargs.get("key", 123),
        temperature=spec_kwargs.get("temperature", 1.0),
        start=spec_kwargs.get("start", 5),
        max_prompt_tokens=spec_kwargs.get("max_prompt_tokens", 512),
    )
    return watermark, auxiliary_tokenizer


def main() -> None:
    args = parse_args()
    debug_rows = os.environ.get("SCORE_DEBUG_ROWS") == "1"

    def debug_log(message: str) -> None:
        if debug_rows:
            print(message, flush=True)

    manifest = load_manifest(args.manifest)
    spec = get_method_spec(args.method)
    model_name = args.model_name or manifest.model_name

    input_rows = read_jsonl(args.input_file)
    if args.start_index:
        input_rows = input_rows[args.start_index :]
    if args.num_samples is not None:
        input_rows = input_rows[: args.num_samples]

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict] = []
    if args.resume and output_path.exists():
        existing_rows = read_jsonl(output_path)
        print(f"Resuming from {len(existing_rows)} rows in {output_path}")
    elif output_path.exists():
        output_path.unlink()

    if spec.engine == "dawa":
        debug_log(f"[score-debug] loading dawa detector method={args.method} model={model_name}")
        load_start = time.time()
        detector, tokenizer = load_dawa_detector(
            model_name=model_name,
            auxiliary_model_name=args.auxiliary_model,
            spec_kwargs=spec.method_kwargs,
            torch_dtype_name=args.torch_dtype,
        )
        debug_log(f"[score-debug] loaded dawa detector elapsed={time.time() - load_start:.1f}s")
        adapter = None
        threshold = float(spec.method_kwargs.get("alpha", 0.2))
    else:
        debug_log(f"[score-debug] loading adapter method={spec.method} model={model_name}")
        load_start = time.time()
        adapter = WatermarkExperimentAdapter(
            method=spec.method,
            model_name=model_name,
            seed=args.seed,
            load_in_4bit=args.load_in_4bit and manifest.load_in_4bit,
            noise_level=spec.noise_level,
            hcw_method=spec.hcw_method,
            method_kwargs=spec.method_kwargs,
        )
        debug_log(f"[score-debug] loaded adapter elapsed={time.time() - load_start:.1f}s")
        detector = None
        tokenizer = None
        threshold = None

    start_idx = len(existing_rows)
    written_rows = list(existing_rows)
    buffer: list[str] = []

    for idx in range(start_idx, len(input_rows)):
        row = dict(input_rows[idx])
        prompt = extract_prompt(row)
        text = extract_generated_text(row)
        row_start = time.time()
        debug_log(
            "[score-debug] row-start "
            f"local_idx={idx} global_idx={args.start_index + idx} "
            f"sample_index={row.get('sample_index')} "
            f"prompt_chars={len(prompt)} text_chars={len(text)}"
        )

        if spec.engine == "dawa":
            score = float(detector.detection(text))
            num_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            detection = {
                "score": score,
                "threshold": threshold,
                "is_watermarked": bool(score >= threshold),
                "num_tokens": int(num_tokens),
                "detector_type": "dawa_native",
            }
        else:
            detection = adapter.detect_text(
                prompt,
                text,
                generation_metadata=scoring_metadata(row),
            )

        if "detection_stats" in row and "source_detection_stats" not in row:
            row["source_detection_stats"] = row["detection_stats"]
        row["detection_stats"] = detection
        buffer.append(json.dumps(row))
        written_rows.append(row)
        debug_log(
            "[score-debug] row-done "
            f"local_idx={idx} global_idx={args.start_index + idx} "
            f"elapsed={time.time() - row_start:.1f}s"
        )

        if len(buffer) >= args.batch_save:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(buffer) + "\n")
            buffer = []
            gc.collect()
            print(f"Progress: {idx + 1}/{len(input_rows)}", flush=debug_rows)

    if buffer:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(buffer) + "\n")

    summary = compute_summary(written_rows)
    summary["config"] = {
        "manifest": str(Path(args.manifest).resolve()),
        "method": spec.method,
        "family": spec.family,
        "engine": spec.engine,
        "model_name": model_name,
        "input_file": args.input_file,
        "output_file": args.output_file,
        "start_index": args.start_index,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "load_in_4bit": args.load_in_4bit and manifest.load_in_4bit,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
