#!/usr/bin/env python3
"""Preflight validation of the consolidated Llama-2 LFQA method roster before full reruns."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.dawa_utils import load_model
from common.dawa_watermark import Watermark
from common.io_utils import extract_prompt, read_jsonl
from common.manifests import load_manifest, manifest_dataset_path
from common.method_registry import get_method_spec
from common.watermark_adapters import WatermarkExperimentAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(Path(__file__).resolve().with_name("manifest.json")))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-name", default=None, help="Override the manifest model name.")
    parser.add_argument("--methods", default=None, help="Comma-separated subset. Defaults to manifest methods.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True)
    return parser.parse_args()


def extract_detection_score(detection: dict) -> float:
    if "score" in detection:
        return float(detection["score"])
    if "z_score" in detection:
        return float(detection["z_score"])
    if "confidence" in detection:
        return float(detection["confidence"])
    return 0.0


def extract_reference_text(sample: dict) -> str:
    comments = sample.get("comments")
    if isinstance(comments, dict):
        for value in comments.values():
            if isinstance(value, list) and value:
                return str(value[0]).replace("@@@@@@", " ").strip()
            if isinstance(value, str):
                return value.replace("@@@@@@", " ").strip()
    for key in ("answer", "response", "completion", "text"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def preflight_method_kwargs(method: str, base_kwargs: dict) -> dict:
    kwargs = dict(base_kwargs)
    if method == "pmark":
        kwargs["num_samples"] = min(int(kwargs.get("num_samples", 64)), 8)
        kwargs["min_sequences"] = min(int(kwargs.get("min_sequences", 10)), 1)
        kwargs["max_sentences"] = min(int(kwargs.get("max_sentences", 12)), 4)
        kwargs["embedder_device"] = "cpu"
        kwargs["embedder_backend"] = "hash"
    return kwargs


def load_dawa_detector(model_name: str, spec_kwargs: dict, torch_dtype_name: str):
    watermark_model, watermark_tokenizer = load_model(model_name, torch_dtype_name=torch_dtype_name)
    device = next(watermark_model.parameters()).device
    detector = Watermark(
        device=device,
        watermark_tokenizer=watermark_tokenizer,
        watermark_model=watermark_model,
        auxiliary_tokenizer=watermark_tokenizer,
        auxiliary_model=watermark_model,
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
    return detector, watermark_tokenizer


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    model_name = args.model_name or manifest.model_name
    dataset_path = manifest_dataset_path(manifest)
    if dataset_path is None:
        raise RuntimeError("Manifest does not define a dataset_path.")

    methods = (
        [item.strip() for item in args.methods.split(",") if item.strip()]
        if args.methods
        else list(manifest.methods)
    )
    num_samples = args.num_samples if args.num_samples is not None else manifest.preflight_num_samples
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else manifest.max_new_tokens
    samples = read_jsonl(dataset_path)[:num_samples]
    prompts = [extract_prompt(sample) for sample in samples]

    if methods == ["pmark"]:
        vanilla_outputs = [
            extract_reference_text(sample) or prompt
            for sample, prompt in zip(samples, prompts)
        ]
        negative_source = "dataset_reference"
    else:
        vanilla_adapter = WatermarkExperimentAdapter(
            method="vanilla",
            model_name=model_name,
            seed=args.seed,
            load_in_4bit=args.load_in_4bit and manifest.load_in_4bit,
            method_kwargs={},
        )
        vanilla_outputs = [
            vanilla_adapter.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=manifest.temperature,
                top_p=manifest.top_p,
                top_k=manifest.top_k,
                seed=args.seed + idx * 101,
            ).text
            for idx, prompt in enumerate(prompts)
        ]
        negative_source = "vanilla_model"
        del vanilla_adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report: dict[str, dict] = {
        "manifest": str(Path(args.manifest).resolve()),
        "model_name": model_name,
        "num_samples": len(prompts),
        "max_new_tokens": max_new_tokens,
        "negative_source": negative_source,
        "methods": {},
    }

    for method in methods:
        spec = get_method_spec(method)
        method_kwargs = preflight_method_kwargs(method, spec.method_kwargs)
        if spec.engine == "dawa":
            detector, tokenizer = load_dawa_detector(
                model_name=model_name,
                spec_kwargs=method_kwargs,
                torch_dtype_name=args.torch_dtype,
            )
            positive_scores = []
            negative_scores = []
            non_empty = 0
            token_counts = []
            for idx, prompt in enumerate(prompts):
                generated_text = detector.generate_watermarked(prompt)
                positive_scores.append(float(detector.detection(generated_text)))
                negative_scores.append(float(detector.detection(vanilla_outputs[idx])))
                if generated_text.strip():
                    non_empty += 1
                token_counts.append(int(len(tokenizer.encode(generated_text, add_special_tokens=False))))
        else:
            adapter = WatermarkExperimentAdapter(
                method=spec.method,
                model_name=model_name,
                seed=args.seed,
                load_in_4bit=args.load_in_4bit and manifest.load_in_4bit,
                noise_level=spec.noise_level,
                hcw_method=spec.hcw_method,
                method_kwargs=method_kwargs,
            )
            positive_scores = []
            negative_scores = []
            non_empty = 0
            token_counts = []
            for idx, prompt in enumerate(prompts):
                generated = adapter.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=manifest.temperature,
                    top_p=manifest.top_p,
                    top_k=manifest.top_k,
                    seed=args.seed + idx * 101,
                )
                positive_scores.append(extract_detection_score(generated.detection))
                negative_detection = adapter.detect_text(prompt=prompt, text=vanilla_outputs[idx])
                negative_scores.append(extract_detection_score(negative_detection))
                if generated.text.strip():
                    non_empty += 1
                token_counts.append(int(generated.metadata.get("completion_token_count", 0)))

        mean_positive = float(np.mean(positive_scores)) if positive_scores else 0.0
        mean_negative = float(np.mean(negative_scores)) if negative_scores else 0.0
        report["methods"][method] = {
            "display_name": spec.display_name,
            "engine": spec.engine,
            "num_samples": len(prompts),
            "non_empty_generations": non_empty,
            "mean_positive_score": mean_positive,
            "mean_negative_score": mean_negative,
            "mean_completion_tokens": float(np.mean(token_counts)) if token_counts else 0.0,
            "method_kwargs": method_kwargs,
            "separation_margin": mean_positive - mean_negative,
            "separation_pass": bool(mean_positive > mean_negative),
            "non_empty_pass": bool(non_empty == len(prompts)),
            "preflight_pass": bool(non_empty == len(prompts) and mean_positive > mean_negative),
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
