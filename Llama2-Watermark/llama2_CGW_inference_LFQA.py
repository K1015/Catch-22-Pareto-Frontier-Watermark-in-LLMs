"""
Llama 2 / CodeLlama with a rebuilt Christ et al. distribution-preserving watermark.

The generation path remains quantile-based and keyed, but verification now uses
the prompt plus model probabilities to replay the quantile targets instead of
the previous promptless heuristic.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import sys
from dataclasses import dataclass
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
    cdf_index_from_u,
    likelihood_ratio_summary,
    load_model_and_tokenizer,
    safe_choice_from_probs,
    to_safe_probs,
)


@dataclass
class WatermarkConfig:
    secret_key: str
    security_parameter: int = 128
    detection_threshold_factor: float = 1.0
    entropy_threshold: float = 128.0
    detection_alpha: float = 0.01
    detector_smoothing: float = 1e-4


class ChristWatermarkDetector:
    """Promptless fallback kept only for API compatibility."""

    def __init__(self, config: WatermarkConfig):
        self.config = config

    def detect(self, text: str, tokenizer) -> Dict[str, object]:
        num_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        return {
            "is_watermarked": False,
            "confidence": 0.0,
            "z_score": 0.0,
            "p_value": 1.0,
            "score": 0.0,
            "num_tokens": num_tokens,
            "detector_type": "promptless_unavailable",
        }


class Llama2ChristLFQA:
    """Llama wrapper for rebuilt Christ watermark generation and prompt-aware detection."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        secret_key: str = "default-secret-key-2024",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        security_parameter: int = 128,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit if not load_in_8bit else False
        self.config = WatermarkConfig(
            secret_key=secret_key,
            security_parameter=security_parameter,
        )
        self.detector = ChristWatermarkDetector(self.config)
        self.last_completion_token_ids: List[int] = []
        self.last_generation_metadata: Dict[str, object] = {}

        print("=" * 80)
        print("LLAMA 2 / CODELLAMA - CHRIST ET AL. WATERMARK")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Security parameter (lambda): {security_parameter}")

        self._load_model()

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

    def _prf(self, seed: str, index: int) -> float:
        key = self.config.secret_key.encode("utf-8")
        message = f"{seed}:{index}".encode("utf-8")
        digest = hmac.new(key, message, hashlib.sha256).digest()
        return int.from_bytes(digest[:8], byteorder="big") / float(2**64)

    @staticmethod
    def _prefix_seed(prefix_tokens: Sequence[int]) -> str:
        return "".join(map(str, prefix_tokens))

    @staticmethod
    def _target_from_quantile(base_probs: torch.Tensor, u: float) -> Tuple[int, float]:
        sorted_probs, sorted_indices = torch.sort(to_safe_probs(base_probs), descending=True)
        quantile_index = cdf_index_from_u(sorted_probs, u)
        return int(sorted_indices[quantile_index].item()), float(sorted_probs[quantile_index].item())

    def _smoothed_target_distribution(
        self,
        base_probs: torch.Tensor,
        target_token: int,
        smoothing: Optional[float] = None,
    ) -> torch.Tensor:
        epsilon = self.config.detector_smoothing if smoothing is None else float(smoothing)
        base = to_safe_probs(base_probs)
        smoothed = base * epsilon
        smoothed[target_token] += 1.0 - epsilon
        return to_safe_probs(smoothed)

    def generate_watermarked(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        prompt_ids = inputs.input_ids[0].tolist()
        generated_ids = list(prompt_ids)
        completion_ids: List[int] = []
        empirical_entropy = 0.0
        seed: Optional[str] = None
        watermark_active = False
        activation_token_offset: Optional[int] = None
        watermarked_steps = 0
        model_inputs = inputs.input_ids
        past_key_values = None

        _debug_log(
            f"[CGW] prompt_tokens={len(prompt_ids)} max_new_tokens={max_new_tokens} "
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
            if not watermark_active:
                next_token = safe_choice_from_probs(base_probs)
                token_prob = float(base_probs[next_token].item())
                if token_prob > EPS:
                    empirical_entropy += -math.log2(token_prob)
                if empirical_entropy >= self.config.entropy_threshold and seed is None:
                    seed = self._prefix_seed(generated_ids)
                    watermark_active = True
                    activation_token_offset = len(completion_ids) + 1
                    _debug_log(
                        f"[CGW] activated after completion_token={activation_token_offset} "
                        f"entropy_bits={empirical_entropy:.3f}"
                    )
            else:
                assert seed is not None
                u = self._prf(seed, len(generated_ids))
                next_token, _ = self._target_from_quantile(base_probs, u)
                watermarked_steps += 1

            generated_ids.append(next_token)
            completion_ids.append(next_token)
            model_inputs = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            if next_token == self.tokenizer.eos_token_id or len(generated_ids) >= 1024:
                break

        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        self.last_completion_token_ids = list(completion_ids)
        self.last_generation_metadata = {
            "completion_token_count": len(completion_ids),
            "completion_token_ids": list(completion_ids),
            "activation_token_offset": activation_token_offset,
            "watermarked_steps": int(watermarked_steps),
            "entropy_bits_before_activation": float(empirical_entropy),
        }
        _debug_log(f"[CGW] completion_tokens={len(completion_ids)} text_chars={len(text)}")
        return text

    def detect_with_prompt(
        self,
        prompt: str,
        generated_text: str,
        completion_token_ids: Optional[Sequence[int]] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        alpha: Optional[float] = None,
        detector_smoothing: Optional[float] = None,
    ) -> Dict[str, object]:
        alpha = self.config.detection_alpha if alpha is None else alpha
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids[0].tolist()
        if completion_token_ids is None:
            completion_token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
        completion_ids = list(completion_token_ids)
        if not completion_ids:
            return likelihood_ratio_summary([], alpha=alpha, extra={"detector_type": "quantile_llr"})

        full_ids = prompt_ids + completion_ids
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor)
        logits = outputs.logits[0]

        empirical_entropy = 0.0
        seed: Optional[str] = None
        watermark_active = False
        activation_token_offset: Optional[int] = None
        token_scores: List[float] = []
        match_count = 0

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
            if not watermark_active:
                token_prob = float(base_probs[token_id].item())
                if token_prob > EPS:
                    empirical_entropy += -math.log2(token_prob)
                if empirical_entropy >= self.config.entropy_threshold and seed is None:
                    seed = self._prefix_seed(full_ids[:token_position])
                    watermark_active = True
                    activation_token_offset = idx + 1
                continue

            assert seed is not None
            u = self._prf(seed, token_position)
            target_token, target_prob = self._target_from_quantile(base_probs, u)
            wm_probs = self._smoothed_target_distribution(base_probs, target_token, smoothing=detector_smoothing)
            token_scores.append(
                float(
                    torch.log(wm_probs[token_id].clamp_min(EPS)).item()
                    - torch.log(to_safe_probs(base_probs)[token_id].clamp_min(EPS)).item()
                )
            )
            match_count += int(token_id == target_token)

        summary = likelihood_ratio_summary(
            token_scores,
            alpha=alpha,
            extra={
                "detector_type": "quantile_llr",
                "activation_token_offset": activation_token_offset,
                "match_rate": float(match_count / max(len(token_scores), 1)) if token_scores else 0.0,
                "watermarked_steps": int(len(token_scores)),
                "entropy_bits_before_activation": float(empirical_entropy),
            },
        )
        summary["confidence"] = float(min(max(summary["score"], 0.0) / max(summary["threshold"], 1.0), 1.0))
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
            f"{self.model_name.replace('/', '-')}_christ_strength_0_frac_0_len_{max_new_tokens}_num_{num_samples}.jsonl",
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
                generated_text = self.generate_watermarked(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                )
                detection = self.detect_with_prompt(
                    prompt=prompt,
                    generated_text=generated_text,
                    completion_token_ids=self.last_completion_token_ids,
                    temperature=0.7,
                )
                output_entry = {
                    "prefix": str(prompt),
                    "gold_completion": str(gold_completion),
                    "gen_completion": [str(generated_text)],
                    "watermark_params": {
                        "method": "christ_dist_preserving",
                        "security_parameter": self.config.security_parameter,
                    },
                    "generation_metadata": dict(self.last_generation_metadata),
                    "detection_stats": {
                        "z_score": float(detection.get("z_score", 0.0)),
                        "p_value": float(detection.get("p_value", 1.0)),
                        "is_watermarked": bool(detection.get("is_watermarked", False)),
                        "confidence": float(detection.get("confidence", 0.0)),
                        "score": float(detection.get("score", 0.0)),
                        "num_tokens": int(detection.get("num_tokens", 0)),
                    },
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
            {"prefix": "What is machine learning?", "gold_completion": ""},
            {"prefix": "Explain climate change.", "gold_completion": ""},
            {"prefix": "How does photosynthesis work?", "gold_completion": ""},
            {"prefix": "What is quantum computing?", "gold_completion": ""},
            {"prefix": "Describe the immune system.", "gold_completion": ""},
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
            confidences = [float(row["detection_stats"]["confidence"]) for row in results]
            stats = {
                "total_samples": len(results),
                "detection_rate": float(np.mean(detections)),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "median_score": float(np.median(scores)),
                "mean_confidence": float(np.mean(confidences)),
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
        "secret_key": "christ-watermark-secret-2024",
        "security_parameter": 128,
        "load_in_4bit": True,
    }

    processor = Llama2ChristLFQA(
        model_name=config["model_name"],
        secret_key=config["secret_key"],
        load_in_4bit=config["load_in_4bit"],
        security_parameter=config["security_parameter"],
    )
    stats = processor.process_lfqa_dataset(
        input_file=config["input_file"],
        output_dir=config["output_dir"],
        num_samples=config["num_samples"],
        max_new_tokens=config["max_new_tokens"],
    )
    if stats:
        stats_file = os.path.join(config["output_dir"], "christ_watermark_stats.json")
        with open(stats_file, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        print(f"\nStatistics saved to {stats_file}")


if __name__ == "__main__":
    main()
