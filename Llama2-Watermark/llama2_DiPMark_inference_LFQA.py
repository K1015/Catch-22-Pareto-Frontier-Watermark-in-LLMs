"""
Llama 2 / CodeLlama with a rebuilt DiPMark watermark.

The public class names remain stable, but generation and verification now share
the same keyed reweighting logic and expose a prompt-aware likelihood-ratio
detector for the reviewer pipelines.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from watermark_rebuild_common import (
    EPS,
    _debug_log,
    apply_sampling_filters,
    likelihood_ratio_summary,
    load_model_and_tokenizer,
    safe_choice_from_probs,
    to_safe_probs,
)


class DiPmark:
    """Keyed DiPMark reweighting with deterministic texture keys."""

    def __init__(
        self,
        vocab_size: int,
        key: Optional[str] = None,
        alpha: float = 0.45,
        gamma: float = 0.5,
        context_window: int = 5,
    ):
        self.vocab_size = vocab_size
        self.key = key or self._generate_key()
        self.alpha = alpha
        self.gamma = gamma
        self.context_window = context_window
        self.texture_key_history: set[str] = set()

    def _generate_key(self) -> str:
        return "".join(str(np.random.randint(0, 2)) for _ in range(1024))

    def reset_history(self) -> None:
        self.texture_key_history = set()

    def _get_texture_key(self, tokens: Sequence[int]) -> str:
        recent = tokens[-self.context_window :] if len(tokens) >= self.context_window else tokens
        return "_".join(map(str, recent))

    def _hash_to_permutation(self, texture_key: str) -> np.ndarray:
        import hashlib

        digest = hashlib.sha256(f"{texture_key}_{self.key}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], byteorder="big") % (2**32)
        rng = np.random.RandomState(seed)
        permutation = np.arange(self.vocab_size)
        rng.shuffle(permutation)
        return permutation

    def _p_alpha_reweight(self, probs: torch.Tensor, permutation: np.ndarray) -> torch.Tensor:
        safe = to_safe_probs(probs)
        perm_tensor = torch.from_numpy(permutation).to(safe.device, dtype=torch.long)
        reordered = safe.index_select(0, perm_tensor)
        cumulative = torch.cumsum(reordered, dim=0)
        transformed = torch.clamp((cumulative - self.alpha) / max(1.0 - self.alpha, EPS), min=0.0)

        reweighted = torch.zeros_like(safe)
        previous = torch.cat([torch.zeros(1, device=safe.device, dtype=safe.dtype), transformed[:-1]])
        increments = transformed - previous
        reweighted.scatter_(0, perm_tensor, increments)
        return to_safe_probs(reweighted)

    def _p_one_minus_alpha_reweight(self, probs: torch.Tensor, permutation: np.ndarray) -> torch.Tensor:
        safe = to_safe_probs(probs)
        perm_tensor = torch.from_numpy(permutation).to(safe.device, dtype=torch.long)
        reordered = safe.index_select(0, perm_tensor)
        cumulative = torch.cumsum(reordered, dim=0)
        transformed = torch.clamp((cumulative - (1.0 - self.alpha)) / max(self.alpha, EPS), min=0.0)

        reweighted = torch.zeros_like(safe)
        previous = torch.cat([torch.zeros(1, device=safe.device, dtype=safe.dtype), transformed[:-1]])
        increments = transformed - previous
        reweighted.scatter_(0, perm_tensor, increments)
        return to_safe_probs(reweighted)

    def dip_reweight(self, probs: torch.Tensor, texture_key: str) -> torch.Tensor:
        if texture_key in self.texture_key_history:
            return to_safe_probs(probs)

        self.texture_key_history.add(texture_key)
        permutation = self._hash_to_permutation(texture_key)
        p_alpha = self._p_alpha_reweight(probs, permutation)
        p_one_minus_alpha = self._p_one_minus_alpha_reweight(probs, permutation)
        return to_safe_probs((1.0 - self.alpha) * p_alpha + self.alpha * p_one_minus_alpha)

    def reweight_distribution(
        self,
        probs: torch.Tensor,
        context_tokens: Sequence[int],
        texture_history: Optional[set[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        history = self.texture_key_history if texture_history is None else texture_history
        texture_key = self._get_texture_key(context_tokens)
        repeated = texture_key in history
        safe = to_safe_probs(probs)
        if repeated:
            return safe, {"texture_key": texture_key, "reweighted": False, "repeated_context": True}

        history.add(texture_key)
        permutation = self._hash_to_permutation(texture_key)
        p_alpha = self._p_alpha_reweight(safe, permutation)
        p_one_minus_alpha = self._p_one_minus_alpha_reweight(safe, permutation)
        reweighted = to_safe_probs((1.0 - self.alpha) * p_alpha + self.alpha * p_one_minus_alpha)
        return reweighted, {"texture_key": texture_key, "reweighted": True, "repeated_context": False}

    def _green_membership(self, token_id: int, texture_key: str) -> bool:
        permutation = self._hash_to_permutation(texture_key)
        ranks = np.empty(self.vocab_size, dtype=np.int32)
        ranks[permutation] = np.arange(self.vocab_size, dtype=np.int32)
        green_start = int(self.gamma * self.vocab_size)
        return int(ranks[token_id]) >= green_start

    def compute_detection_score(self, tokens: List[int], texture_keys: List[str]) -> Dict[str, object]:
        if not tokens:
            return {
                "is_watermarked": False,
                "z_score": 0.0,
                "p_value": 1.0,
                "green_ratio": 0.0,
                "green_count": 0,
                "num_tokens": 0,
                "score": 0.0,
            }

        hits = 0
        for token_id, texture_key in zip(tokens, texture_keys):
            if self._green_membership(int(token_id), texture_key):
                hits += 1

        expected_mean = len(tokens) * (1.0 - self.gamma)
        variance = len(tokens) * self.gamma * max(1.0 - self.gamma, EPS)
        z_score = float((hits - expected_mean) / math.sqrt(max(variance, EPS)))
        p_value = float(0.5 * math.erfc(z_score / math.sqrt(2.0)))
        return {
            "is_watermarked": bool(z_score >= 2.0),
            "z_score": z_score,
            "p_value": p_value,
            "green_ratio": float(hits / len(tokens)),
            "green_count": int(hits),
            "num_tokens": int(len(tokens)),
            "score": z_score,
        }


class Llama2DiPmarkLFQA:
    """Llama wrapper for rebuilt DiPMark generation and verification."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        watermark_key: Optional[str] = None,
        alpha: float = 0.45,
        gamma: float = 0.5,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit if not load_in_8bit else False
        self.last_completion_token_ids: List[int] = []
        self.last_texture_keys: List[str] = []
        self.last_generation_metadata: Dict[str, object] = {}

        print("=" * 80)
        print("LLAMA 2 / CODELLAMA - DIPMARK")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"DiPMark parameters: alpha={alpha}, gamma={gamma}")

        self._load_model()
        self.watermark = DiPmark(
            vocab_size=self.tokenizer.vocab_size,
            key=watermark_key,
            alpha=alpha,
            gamma=gamma,
        )

    def _load_model(self) -> None:
        print(f"\nLoading {self.model_name}...")
        self.tokenizer, self.model = load_model_and_tokenizer(
            self.model_name,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
        )
        print("Model loaded successfully!")
        if torch.cuda.is_available():
            print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def generate_watermarked(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Tuple[str, List[str]]:
        self.watermark.reset_history()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        prompt_ids = inputs.input_ids[0].tolist()
        generated_ids = list(prompt_ids)
        completion_ids: List[int] = []
        texture_keys: List[str] = []
        reweighted_steps = 0
        repeated_contexts = 0
        model_inputs = inputs.input_ids
        past_key_values = None

        _debug_log(
            f"[DIPMARK] prompt_tokens={len(prompt_ids)} max_new_tokens={max_new_tokens} "
            f"top_p={top_p} top_k={top_k}"
        )

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=model_inputs,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[0, -1, :].float() / max(temperature, 1e-6)

            base_probs = apply_sampling_filters(torch.softmax(logits, dim=-1), top_p=top_p, top_k=top_k)
            wm_probs, info = self.watermark.reweight_distribution(base_probs, generated_ids)
            texture_keys.append(str(info["texture_key"]))
            if info["reweighted"]:
                reweighted_steps += 1
            if info["repeated_context"]:
                repeated_contexts += 1

            next_token = safe_choice_from_probs(wm_probs)
            generated_ids.append(next_token)
            completion_ids.append(next_token)
            model_inputs = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            if next_token == self.tokenizer.eos_token_id or len(generated_ids) >= 1024:
                break

        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        self.last_completion_token_ids = list(completion_ids)
        self.last_texture_keys = list(texture_keys)
        self.last_generation_metadata = {
            "completion_token_count": len(completion_ids),
            "completion_token_ids": list(completion_ids),
            "texture_key_count": len(texture_keys),
            "reweighted_steps": reweighted_steps,
            "repeated_contexts": repeated_contexts,
        }
        _debug_log(f"[DIPMARK] completion_tokens={len(completion_ids)} text_chars={len(text)}")
        return text, texture_keys

    def detect_with_prompt(
        self,
        prompt: str,
        generated_text: str,
        completion_token_ids: Optional[Sequence[int]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        alpha: float = 0.01,
    ) -> Dict[str, object]:
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids[0].tolist()
        if completion_token_ids is None:
            completion_token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
        completion_ids = list(completion_token_ids)
        if not completion_ids:
            return likelihood_ratio_summary([], alpha=alpha, extra={"detector_type": "llr", "green_ratio": 0.0})

        full_ids = prompt_ids + completion_ids
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor)
        logits = outputs.logits[0]

        texture_history: set[str] = set()
        token_scores: List[float] = []
        texture_keys: List[str] = []
        repeated_contexts = 0

        for idx, token_id in enumerate(completion_ids):
            token_position = len(prompt_ids) + idx
            logit_position = token_position - 1
            if logit_position < 0:
                continue

            base_probs = apply_sampling_filters(
                torch.softmax(logits[logit_position].float() / max(temperature, 1e-6), dim=-1),
                top_p=top_p,
                top_k=top_k,
            )
            wm_probs, info = self.watermark.reweight_distribution(
                base_probs,
                context_tokens=full_ids[:token_position],
                texture_history=texture_history,
            )
            if info["repeated_context"]:
                repeated_contexts += 1
            texture_keys.append(str(info["texture_key"]))
            token_scores.append(
                float(
                    torch.log(to_safe_probs(wm_probs)[token_id].clamp_min(EPS)).item()
                    - torch.log(to_safe_probs(base_probs)[token_id].clamp_min(EPS)).item()
                )
            )

        fallback = self.watermark.compute_detection_score(completion_ids, texture_keys)
        summary = likelihood_ratio_summary(
            token_scores,
            alpha=alpha,
            extra={
                "detector_type": "llr",
                "repeated_contexts": int(repeated_contexts),
                "green_ratio": float(fallback["green_ratio"]),
                "green_count": int(fallback["green_count"]),
            },
        )
        return summary

    def process_lfqa_dataset(
        self,
        input_file: str,
        output_dir: str,
        num_samples: int = 500,
        max_new_tokens: int = 300,
        batch_save: int = 10,
    ) -> Dict[str, object]:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"{self.model_name.replace('/', '-')}_dipmark_alpha_{self.watermark.alpha}_"
            f"gamma_{self.watermark.gamma}_len_{max_new_tokens}_num_{num_samples}.jsonl",
        )

        print(f"\nLoading dataset from {input_file}")
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} not found!")
            print("Creating sample dataset...")
            self._create_sample_dataset(input_file)

        with open(input_file, "r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]

        existing_outputs = []
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as handle:
                    existing_outputs = [json.loads(line) for line in handle if line.strip()]
                print(f"Found {len(existing_outputs)} existing outputs, resuming...")
            except Exception:
                print("Starting fresh...")

        outputs: List[str] = []
        start_idx = len(existing_outputs)
        for idx in range(start_idx, min(num_samples, len(rows))):
            sample = rows[idx]
            prompt = sample.get("prefix", sample.get("question", sample.get("prompt", "")))
            gold_completion = sample.get("gold_completion", "")

            try:
                generated_text, texture_keys = self.generate_watermarked(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                )
                detection = self.detect_with_prompt(
                    prompt=prompt,
                    generated_text=generated_text,
                    completion_token_ids=self.last_completion_token_ids,
                    temperature=1.0,
                )
                output_entry = {
                    "prefix": str(prompt),
                    "gold_completion": str(gold_completion),
                    "gen_completion": [str(generated_text)],
                    "watermark_params": {
                        "method": "dipmark",
                        "alpha": float(self.watermark.alpha),
                        "gamma": float(self.watermark.gamma),
                    },
                    "generation_metadata": dict(self.last_generation_metadata),
                    "detection_stats": {
                        "z_score": float(detection.get("z_score", 0.0)),
                        "p_value": float(detection.get("p_value", 1.0)),
                        "is_watermarked": bool(detection.get("is_watermarked", False)),
                        "score": float(detection.get("score", 0.0)),
                        "green_ratio": float(detection.get("green_ratio", 0.0)),
                        "num_tokens": int(detection.get("num_tokens", 0)),
                    },
                    "texture_key_count": len(texture_keys),
                }
                outputs.append(json.dumps(output_entry))
                if len(outputs) >= batch_save:
                    with open(output_file, "a", encoding="utf-8") as handle:
                        handle.write("\n".join(outputs) + "\n")
                    outputs = []
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as exc:
                print(f"\nError processing sample {idx}: {exc}")
                continue

        if outputs:
            with open(output_file, "a", encoding="utf-8") as handle:
                handle.write("\n".join(outputs) + "\n")

        print(f"\nCompleted! Output saved to: {output_file}")
        return self._compute_summary_stats(output_file)

    def _create_sample_dataset(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        sample_data = [
            {"prefix": "What is artificial intelligence?", "gold_completion": ""},
            {"prefix": "Explain climate change.", "gold_completion": ""},
            {"prefix": "How does the brain process information?", "gold_completion": ""},
            {"prefix": "What are quantum computers?", "gold_completion": ""},
            {"prefix": "Describe the water cycle.", "gold_completion": ""},
        ]
        with open(filepath, "w", encoding="utf-8") as handle:
            for item in sample_data:
                handle.write(json.dumps(item) + "\n")

    def _compute_summary_stats(self, output_file: str) -> Dict[str, object]:
        try:
            with open(output_file, "r", encoding="utf-8") as handle:
                results = [json.loads(line) for line in handle if line.strip()]
            if not results:
                return {}

            scores = [float(row["detection_stats"]["score"]) for row in results]
            detections = [float(row["detection_stats"]["is_watermarked"]) for row in results]
            green_ratios = [float(row["detection_stats"].get("green_ratio", 0.0)) for row in results]
            stats = {
                "total_samples": len(results),
                "detection_rate": float(np.mean(detections)),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "median_score": float(np.median(scores)),
                "mean_green_ratio": float(np.mean(green_ratios)),
            }
            print(json.dumps(stats, indent=2))
            return stats
        except Exception as exc:
            print(f"Error computing statistics: {exc}")
            return {}


def main() -> None:
    config = {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "input_file": "./data/LFQA/inputs.jsonl",
        "output_dir": "./data/LFQA/",
        "num_samples": 5,
        "max_new_tokens": 300,
        "alpha": 0.45,
        "gamma": 0.5,
        "load_in_4bit": True,
    }

    processor = Llama2DiPmarkLFQA(
        model_name=config["model_name"],
        load_in_4bit=config["load_in_4bit"],
        alpha=config["alpha"],
        gamma=config["gamma"],
    )
    stats = processor.process_lfqa_dataset(
        input_file=config["input_file"],
        output_dir=config["output_dir"],
        num_samples=config["num_samples"],
        max_new_tokens=config["max_new_tokens"],
    )
    if stats:
        stats_file = os.path.join(config["output_dir"], "dipmark_watermark_stats.json")
        with open(stats_file, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        print(f"\nStatistics saved to {stats_file}")


if __name__ == "__main__":
    main()
