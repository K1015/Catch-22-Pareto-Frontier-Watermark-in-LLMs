#!/usr/bin/env python3
"""Apply one attack condition to consolidated Catch-22 generations."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.attack_registry import list_attacks
from common.attacks import (
    BackTranslationAttack,
    CausalLMParaphraseAttack,
    DipperAttack,
    IdentityAttack,
    SpanSynonymAttack,
    SummarizationAttack,
    SynonymAttack,
)
from common.io_utils import (
    compute_token_edit_stats,
    extract_generated_text,
    extract_prompt,
    mean_confidence_interval,
    read_jsonl,
    shallow_copy_with_completion,
)
from common.manifests import load_manifest


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


OPT_PARAPHRASE_TEMPLATE = """[INST] Rewrite the following answer so that it preserves the original meaning and factual content while using different wording and sentence structure.

Question:
{prompt}

Answer:
{text}
[/INST]
"""


WM_REMOVAL_TEMPLATE = """[INST] Rewrite the following answer to preserve its meaning and helpfulness while removing repetitive, formulaic, or suspicious token patterns. Avoid mentioning watermarking.

Question:
{prompt}

Answer:
{text}
[/INST]
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--attack-name", required=True, choices=list_attacks(include_identity=True))
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-save", type=int, default=5)
    parser.add_argument("--attack-batch-size", type=int, default=1)
    parser.add_argument("--source-model", default=None, help="Tokenizer model for token edit rate.")
    parser.add_argument("--attack-model", default=None, help="Attack model for dipper/opt/wm-removal/summarization.")
    parser.add_argument("--forward-model", default=None, help="Forward translation model for backtranslation.")
    parser.add_argument("--backward-model", default=None, help="Backward translation model for backtranslation.")
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dipper-lexical-diversity", type=int, default=40)
    parser.add_argument("--dipper-order-diversity", type=int, default=0)
    parser.add_argument("--dipper-sent-interval", type=int, default=3)
    parser.add_argument("--target-edit-rate", type=float, default=0.15)
    parser.add_argument("--span-min-length", type=int, default=5)
    parser.add_argument("--span-max-length", type=int, default=10)
    return parser.parse_args()


def load_tokenizer_with_fallback(model_name: str):
    if "mistral" in model_name.lower():
        print(f"Using slow tokenizer for {model_name} to avoid fast-tokenizer hangs.", file=sys.stderr)
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:
        print(f"Fast tokenizer load failed for {model_name}; retrying with use_fast=False: {exc}", file=sys.stderr)
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def build_attack(args: argparse.Namespace):
    if args.attack_name == "none":
        return IdentityAttack()
    if args.attack_name == "synonym":
        return SynonymAttack(seed=args.seed, target_edit_rate=args.target_edit_rate)
    if args.attack_name == "span-synonym":
        return SpanSynonymAttack(
            seed=args.seed,
            target_edit_rate=args.target_edit_rate,
            min_span_len=args.span_min_length,
            max_span_len=args.span_max_length,
        )
    if args.attack_name == "backtranslation":
        if not args.forward_model or not args.backward_model:
            raise ValueError("Backtranslation requires --forward-model and --backward-model.")
        return BackTranslationAttack(
            forward_model=args.forward_model,
            backward_model=args.backward_model,
            torch_dtype_name=args.torch_dtype,
        )
    if args.attack_name == "dipper":
        if not args.attack_model:
            raise ValueError("DIPPER requires --attack-model.")
        return DipperAttack(
            model_name=args.attack_model,
            lexical_diversity=args.dipper_lexical_diversity,
            order_diversity=args.dipper_order_diversity,
            sent_interval=args.dipper_sent_interval,
            torch_dtype_name=args.torch_dtype,
        )
    if args.attack_name == "opt":
        if not args.attack_model:
            raise ValueError("OPT attack requires --attack-model.")
        return CausalLMParaphraseAttack(
            model_name=args.attack_model,
            instruction_template=OPT_PARAPHRASE_TEMPLATE,
            torch_dtype_name=args.torch_dtype,
        )
    if args.attack_name == "wm-removal":
        if not args.attack_model:
            raise ValueError("WM-removal attack requires --attack-model.")
        return CausalLMParaphraseAttack(
            model_name=args.attack_model,
            instruction_template=WM_REMOVAL_TEMPLATE,
            torch_dtype_name=args.torch_dtype,
        )
    if args.attack_name == "summarization":
        if not args.attack_model:
            raise ValueError("Summarization requires --attack-model.")
        return SummarizationAttack(
            model_name=args.attack_model,
            torch_dtype_name=args.torch_dtype,
        )
    raise ValueError(f"Unsupported attack name: {args.attack_name}")


def compute_summary(records: list[dict]) -> dict:
    if not records:
        return {}
    edit_rates = [row.get("attack_stats", {}).get("realized_edit_rate", 0.0) for row in records]
    mean_run_lengths = [row.get("attack_stats", {}).get("mean_edited_run_length", 0.0) for row in records]
    max_run_lengths = [row.get("attack_stats", {}).get("max_edited_run_length", 0.0) for row in records]
    num_runs = [row.get("attack_stats", {}).get("num_edited_runs", 0.0) for row in records]
    source_tokens = [row.get("attack_stats", {}).get("source_num_tokens", 0) for row in records]
    attacked_tokens = [row.get("attack_stats", {}).get("attacked_num_tokens", 0) for row in records]
    mean_edit, ci_edit = mean_confidence_interval(edit_rates)
    return {
        "total_samples": len(records),
        "mean_realized_edit_rate": mean_edit,
        "edit_rate_ci95": ci_edit,
        "mean_source_tokens": float(np.mean(source_tokens)) if source_tokens else 0.0,
        "mean_attacked_tokens": float(np.mean(attacked_tokens)) if attacked_tokens else 0.0,
        "mean_num_edited_runs": float(np.mean(num_runs)) if num_runs else 0.0,
        "mean_edited_run_length": float(np.mean(mean_run_lengths)) if mean_run_lengths else 0.0,
        "mean_max_edited_run_length": float(np.mean(max_run_lengths)) if max_run_lengths else 0.0,
    }


def apply_attack_batch(attack, prompts: list[str], texts: list[str]) -> list:
    if hasattr(attack, "apply_batch"):
        return attack.apply_batch(prompts, texts)
    return [attack.apply(prompt, text) for prompt, text in zip(prompts, texts)]


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest) if args.manifest else None
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.input_file)
    if args.num_samples is not None:
        records = records[: args.num_samples]
    elif manifest is not None:
        records = records[: manifest.num_samples]

    tokenizer_model = args.source_model or (manifest.model_name if manifest else None)
    tokenizer = load_tokenizer_with_fallback(tokenizer_model) if tokenizer_model else None
    attack = build_attack(args)
    attack_batch_size = max(1, args.attack_batch_size)

    existing_count = 0
    if output_path.exists():
        existing_count = len(read_jsonl(output_path))
        print(f"Resuming from {existing_count} existing rows in {output_path}")

    written_rows: list[dict] = read_jsonl(output_path) if existing_count else []
    buffer: list[dict] = []

    with tqdm(total=len(records) - existing_count, desc=f"Attack {args.attack_name}") as progress:
        for start in range(existing_count, len(records), attack_batch_size):
            batch_records = records[start : start + attack_batch_size]
            prompts = [extract_prompt(row) for row in batch_records]
            original_texts = [extract_generated_text(row) for row in batch_records]
            results = apply_attack_batch(attack, prompts, original_texts)

            for row, original_text, result in zip(batch_records, original_texts, results):
                edit_stats = compute_token_edit_stats(original_text, result.text, tokenizer=tokenizer)
                source_num_tokens = len(tokenizer.encode(original_text, add_special_tokens=False)) if tokenizer else len(original_text.split())
                attacked_num_tokens = len(tokenizer.encode(result.text, add_special_tokens=False)) if tokenizer else len(result.text.split())
                attack_stats = {
                    **result.metadata,
                    "source_num_tokens": int(source_num_tokens),
                    "attacked_num_tokens": int(attacked_num_tokens),
                    **edit_stats,
                }
                attacked_row = shallow_copy_with_completion(row, result.text, args.attack_name, attack_stats)
                buffer.append(attacked_row)

            if len(buffer) >= args.batch_save:
                with output_path.open("a", encoding="utf-8") as handle:
                    for buffered_row in buffer:
                        handle.write(json.dumps(buffered_row) + "\n")
                written_rows.extend(buffer)
                buffer = []
                gc.collect()
            progress.update(len(batch_records))

    if buffer:
        with output_path.open("a", encoding="utf-8") as handle:
            for buffered_row in buffer:
                handle.write(json.dumps(buffered_row) + "\n")
        written_rows.extend(buffer)

    summary = compute_summary(written_rows)
    summary["config"] = {
        "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
        "attack_name": args.attack_name,
        "input_file": args.input_file,
        "output_file": args.output_file,
        "num_samples": len(records),
        "source_model": tokenizer_model,
        "attack_model": args.attack_model,
        "forward_model": args.forward_model,
        "backward_model": args.backward_model,
        "attack_batch_size": attack_batch_size,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
