#!/usr/bin/env python3
"""Generate consolidated Catch-22 outputs for one method/track."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.dawa_utils import load_model
from common.dawa_watermark import Watermark
from common.io_utils import (
    extract_gold_completion,
    extract_prompt,
    mean_confidence_interval,
    read_jsonl,
)
from common.manifests import load_manifest, manifest_dataset_path
from common.method_registry import get_method_spec, list_methods, runtime_watermark_params
from common.watermark_adapters import WatermarkExperimentAdapter


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
    parser.add_argument("--model-name", default=None, help="Override the manifest model name.")
    parser.add_argument("--auxiliary-model", default=None, help="Optional DAWA auxiliary model override.")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-save", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_score(detection: dict) -> float:
    if "score" in detection:
        return float(detection["score"])
    if "z_score" in detection:
        return float(detection["z_score"])
    if "confidence" in detection:
        return float(detection["confidence"])
    return 0.0


def compute_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}

    lengths = [len(str(row["gen_completion"][0]).split()) for row in rows if row.get("gen_completion")]
    scores = [float(row["detection_stats"].get("score", 0.0)) for row in rows]
    detected = [float(bool(row["detection_stats"].get("is_watermarked", False))) for row in rows]
    mean_length, ci_length = mean_confidence_interval(lengths)
    mean_score, ci_score = mean_confidence_interval(scores)

    return {
        "total_samples": len(rows),
        "mean_completion_words": mean_length,
        "completion_words_ci95": ci_length,
        "mean_score": mean_score,
        "score_ci95": ci_score,
        "std_score": float(np.std(scores)),
        "mean_detection_rate": float(np.mean(detected)),
    }


def load_dawa_components(
    model_name: str,
    auxiliary_model_name: str | None,
    spec_kwargs: dict,
    torch_dtype_name: str,
):
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
    return watermark, watermark_tokenizer


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    spec = get_method_spec(args.method)

    model_name = args.model_name or manifest.model_name
    input_path = Path(args.input_file) if args.input_file else manifest_dataset_path(manifest)
    if input_path is None:
        raise ValueError("No input dataset path provided via --input-file or manifest.")

    num_samples = args.num_samples if args.num_samples is not None else manifest.num_samples
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else manifest.max_new_tokens
    temperature = args.temperature if args.temperature is not None else manifest.temperature
    top_p = args.top_p if args.top_p is not None else manifest.top_p
    top_k = args.top_k if args.top_k is not None else manifest.top_k

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    input_rows = read_jsonl(input_path)
    start_index = max(0, args.start_index)
    stop_index = start_index + num_samples if num_samples is not None else None
    rows = input_rows[start_index:stop_index]

    existing_rows: list[dict] = []
    if args.resume and output_path.exists():
        existing_rows = read_jsonl(output_path)
        print(f"Resuming from {len(existing_rows)} rows in {output_path}")
    elif output_path.exists():
        output_path.unlink()

    if spec.engine == "dawa":
        dawa_kwargs = dict(spec.method_kwargs)
        dawa_kwargs["max_new_tokens"] = max_new_tokens
        dawa_kwargs["temperature"] = temperature
        dawa_kwargs["top_p"] = top_p
        watermark, tokenizer = load_dawa_components(
            model_name=model_name,
            auxiliary_model_name=args.auxiliary_model,
            spec_kwargs=dawa_kwargs,
            torch_dtype_name=args.torch_dtype,
        )
        adapter = None
        threshold = float(dawa_kwargs.get("alpha", 0.2))
    else:
        adapter = WatermarkExperimentAdapter(
            method=spec.method,
            model_name=model_name,
            seed=args.seed,
            load_in_4bit=args.load_in_4bit and manifest.load_in_4bit,
            noise_level=spec.noise_level,
            hcw_method=spec.hcw_method,
            method_kwargs=spec.method_kwargs,
        )
        watermark = None
        tokenizer = None
        threshold = None

    start_idx = len(existing_rows)
    written_rows = list(existing_rows)
    buffer: list[str] = []

    for idx in range(start_idx, len(rows)):
        sample = rows[idx]
        prompt = extract_prompt(sample)
        gold_completion = extract_gold_completion(sample)
        sample_seed = args.seed + (start_index + idx) * 1009
        seed_everything(sample_seed)

        if spec.engine == "dawa":
            generated_text = watermark.generate_watermarked(prompt)
            detection_score = float(watermark.detection(generated_text))
            num_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
            detection = {
                "score": detection_score,
                "threshold": threshold,
                "is_watermarked": bool(detection_score >= threshold),
                "num_tokens": int(num_tokens),
                "detector_type": "dawa_native",
            }
            generation_metadata = {
                "auxiliary_model": args.auxiliary_model or model_name,
            }
            avg_entropy_bits = None
            avg_top1_prob = None
            generated_text_value = generated_text
        else:
            generated = adapter.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=sample_seed,
            )
            detection = dict(generated.detection)
            detection["score"] = extract_score(detection)
            detection["detector_type"] = spec.method
            generation_metadata = generated.metadata
            avg_entropy_bits = generated.avg_entropy_bits
            avg_top1_prob = generated.avg_top1_prob
            generated_text_value = generated.text

        watermark_params = runtime_watermark_params(spec)
        if adapter is not None:
            watermark_params.update(getattr(adapter, "method_kwargs", {}))

        row = {
            "sample_index": start_index + idx,
            "prefix": prompt,
            "gold_completion": gold_completion,
            "gen_completion": [generated_text_value],
            "preset": spec.preset,
            "method": spec.method,
            "family": spec.family,
            "display_name": spec.display_name,
            "model_name": model_name,
            "generation_seed": sample_seed,
            "generation_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            "watermark_params": watermark_params,
            "generation_metadata": generation_metadata,
            "avg_base_entropy_bits": avg_entropy_bits,
            "avg_base_top1_prob": avg_top1_prob,
            "detection_stats": detection,
        }
        buffer.append(json.dumps(row))
        written_rows.append(row)

        if len(buffer) >= args.batch_save:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(buffer) + "\n")
            buffer = []
            print(f"Progress: {idx + 1}/{len(rows)}")

    if buffer:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(buffer) + "\n")

    summary = compute_summary(written_rows)
    summary["config"] = {
        "manifest": str(Path(args.manifest).resolve()),
        "method": spec.method,
        "preset": spec.preset,
        "family": spec.family,
        "display_name": spec.display_name,
        "engine": spec.engine,
        "model_name": model_name,
        "input_file": str(input_path),
        "output_file": args.output_file,
        "start_index": start_index,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": args.seed,
        "load_in_4bit": args.load_in_4bit and manifest.load_in_4bit,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
