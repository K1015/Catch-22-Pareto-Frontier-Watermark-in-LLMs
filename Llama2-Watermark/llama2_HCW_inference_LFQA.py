"""
Llama 2 / CodeLlama with a rebuilt Hu et al. (2024) unbiased watermark.

This version keeps the delta/gamma reweighting family but replaces the previous
heuristic detector with a prompt-aware likelihood-ratio verifier. The public
entrypoints stay compatible with the existing reviewer runners.
"""

from __future__ import annotations

import json
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
    likelihood_ratio_summary,
    load_model_and_tokenizer,
    make_generator,
    maximin_log_score,
    safe_choice_from_probs,
    to_safe_probs,
)


@dataclass
class HuDetectionConfig:
    alpha: float = 0.01
    detector_type: str = "llr"
    tv_distance: float = 0.0


class UnbiasedWatermark:
    """Keyed Hu et al. watermark with deterministic context-dependent reweighting."""

    def __init__(
        self,
        vocab_size: int,
        key: Optional[str] = None,
        context_window: int = 5,
        preserve_context_history: bool = False,
    ):
        self.vocab_size = vocab_size
        self.key = key or self._generate_key()
        self.context_window = context_window
        self.preserve_context_history = preserve_context_history
        self.context_history: set[str] = set()

    def _generate_key(self) -> str:
        return "".join(str(np.random.randint(0, 2)) for _ in range(1024))

    def reset_history(self) -> None:
        self.context_history = set()

    def _get_context_code(self, tokens: Sequence[int]) -> str:
        recent = tokens[-self.context_window :] if len(tokens) >= self.context_window else tokens
        return "_".join(map(str, recent))

    def _seed_from_context(self, context_code: str) -> int:
        import hashlib

        digest = hashlib.sha256(f"{context_code}_{self.key}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big") % (2**32)

    def _pick_from_cdf(self, probs: torch.Tensor, seed: int) -> int:
        safe = to_safe_probs(probs)
        generator = make_generator(seed, safe.device)
        u = torch.rand((), generator=generator, device=safe.device, dtype=safe.dtype)
        upper = torch.tensor(1.0 - 1e-7, device=safe.device, dtype=safe.dtype)
        cumulative = torch.cumsum(safe, dim=0)
        idx = int(torch.searchsorted(cumulative, torch.minimum(u, upper), right=False).item())
        return min(idx, cumulative.numel() - 1)

    def _delta_reweight(self, probs: torch.Tensor, seed: int) -> torch.Tensor:
        safe = to_safe_probs(probs)
        idx = self._pick_from_cdf(safe, seed)
        delta = torch.zeros_like(safe)
        delta[idx] = 1.0
        return delta

    def _gamma_reweight(self, probs: torch.Tensor, seed: int) -> torch.Tensor:
        safe = to_safe_probs(probs)
        generator = make_generator(seed, safe.device)
        indices = torch.randperm(safe.numel(), generator=generator, device=safe.device)
        shuffled = safe.index_select(0, indices)
        cumulative = torch.cumsum(shuffled, dim=0)
        transformed = torch.clamp(2 * cumulative - 1, min=0.0)

        reweighted = torch.zeros_like(safe)
        previous = torch.cat([torch.zeros(1, device=safe.device, dtype=safe.dtype), transformed[:-1]])
        increments = transformed - previous
        reweighted.scatter_(0, indices, increments)
        return to_safe_probs(reweighted)

    def reweight_distribution(
        self,
        probs: torch.Tensor,
        context_tokens: Sequence[int],
        method: str,
        context_history: Optional[set[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        history = self.context_history if context_history is None else context_history
        context_code = self._get_context_code(context_tokens)
        repeated = context_code in history
        safe = to_safe_probs(probs)
        if repeated:
            return safe, {"context_code": context_code, "reweighted": False, "repeated_context": True}

        history.add(context_code)
        seed = self._seed_from_context(context_code)
        if method == "delta":
            reweighted = self._delta_reweight(safe, seed)
        elif method == "gamma":
            reweighted = self._gamma_reweight(safe, seed)
        else:
            raise ValueError(f"Unsupported Hu method: {method}")

        return reweighted, {"context_code": context_code, "reweighted": True, "repeated_context": False}

    def compute_detection_score(self, tokens: List[int], method: str = "delta", alpha: float = 0.01) -> Dict[str, object]:
        """Promptless fallback kept only for API compatibility."""
        if not tokens:
            return likelihood_ratio_summary([], alpha=alpha, extra={"detector_type": f"{method}_text_only"})

        uniform = torch.full((self.vocab_size,), 1.0 / max(self.vocab_size, 1), dtype=torch.float32)
        context_history: set[str] = set()
        prefix: List[int] = []
        token_scores: List[float] = []
        reweighted_steps = 0
        for token_id in tokens:
            wm_probs, info = self.reweight_distribution(uniform, prefix, method=method, context_history=context_history)
            if info["reweighted"]:
                reweighted_steps += 1
            token_scores.append(float(torch.log(wm_probs[token_id].clamp_min(EPS)).item() - np.log(1.0 / self.vocab_size)))
            prefix.append(int(token_id))

        return likelihood_ratio_summary(
            token_scores,
            alpha=alpha,
            extra={
                "detector_type": f"{method}_text_only",
                "reweighted_steps": int(reweighted_steps),
                "repeated_contexts": int(max(len(tokens) - reweighted_steps, 0)),
            },
        )


class Llama2HuLFQA:
    """Llama wrapper for rebuilt Hu et al. watermark generation and verification."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        watermark_key: Optional[str] = None,
        method: str = "delta",
        context_window: int = 5,
        preserve_context_history: bool = False,
        detection_config: Optional[HuDetectionConfig] = None,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit if not load_in_8bit else False
        self.method = method
        self.detection_config = detection_config or HuDetectionConfig()
        self.last_completion_token_ids: List[int] = []
        self.last_generation_metadata: Dict[str, object] = {}

        print("=" * 80)
        print("LLAMA 2 / CODELLAMA - HU ET AL. UNBIASED WATERMARK")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Method: {method}-reweight")
        print(f"Detector: {self.detection_config.detector_type}")
        print(f"TV distance: {self.detection_config.tv_distance}")

        self._load_model()
        self.watermark = UnbiasedWatermark(
            vocab_size=self.tokenizer.vocab_size,
            key=watermark_key,
            context_window=context_window,
            preserve_context_history=preserve_context_history,
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
        reset_context_history: bool = True,
    ) -> str:
        if reset_context_history and not self.watermark.preserve_context_history:
            self.watermark.reset_history()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        prompt_ids = inputs.input_ids[0].tolist()
        generated_ids = list(prompt_ids)
        completion_ids: List[int] = []
        reweighted_steps = 0
        skipped_repeats = 0
        model_inputs = inputs.input_ids
        past_key_values = None

        _debug_log(
            f"[HCW] prompt_tokens={len(prompt_ids)} max_new_tokens={max_new_tokens} "
            f"method={self.method} top_p={top_p} top_k={top_k}"
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
            wm_probs, info = self.watermark.reweight_distribution(
                base_probs,
                context_tokens=generated_ids,
                method=self.method,
            )
            if info["reweighted"]:
                reweighted_steps += 1
            if info["repeated_context"]:
                skipped_repeats += 1

            next_token = safe_choice_from_probs(wm_probs)
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
            "reweighted_steps": reweighted_steps,
            "skipped_repeated_contexts": skipped_repeats,
            "detector_type": self.detection_config.detector_type,
            "preserve_context_history": self.watermark.preserve_context_history,
        }
        _debug_log(f"[HCW] completion_tokens={len(completion_ids)} text_chars={len(text)}")
        return text

    def detect_with_prompt(
        self,
        prompt: str,
        generated_text: str,
        completion_token_ids: Optional[Sequence[int]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        alpha: Optional[float] = None,
        detector_type: Optional[str] = None,
        tv_distance: Optional[float] = None,
    ) -> Dict[str, object]:
        alpha = self.detection_config.alpha if alpha is None else alpha
        detector_type = self.detection_config.detector_type if detector_type is None else detector_type
        tv_distance = self.detection_config.tv_distance if tv_distance is None else tv_distance

        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids[0].tolist()
        if completion_token_ids is None:
            completion_token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
        completion_ids = list(completion_token_ids)
        if not completion_ids:
            return likelihood_ratio_summary([], alpha=alpha, extra={"detector_type": detector_type, "tv_distance": tv_distance})

        full_ids = prompt_ids + completion_ids
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor)
        logits = outputs.logits[0]

        context_history: set[str] = set()
        token_scores: List[float] = []
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
                method=self.method,
                context_history=context_history,
            )
            if info["repeated_context"]:
                repeated_contexts += 1

            base_np = to_safe_probs(base_probs).detach().cpu().numpy()
            wm_np = to_safe_probs(wm_probs).detach().cpu().numpy()
            if detector_type == "llr":
                score = float(np.log(max(wm_np[token_id], EPS)) - np.log(max(base_np[token_id], EPS)))
            elif detector_type == "maximin_llr":
                score = maximin_log_score(base_np, wm_np, token_id=token_id, tv_distance=tv_distance)
            else:
                raise ValueError(f"Unsupported detector type: {detector_type}")
            token_scores.append(score)

        return likelihood_ratio_summary(
            token_scores,
            alpha=alpha,
            extra={
                "detector_type": detector_type,
                "tv_distance": float(tv_distance),
                "repeated_contexts": int(repeated_contexts),
            },
        )

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
            f"{self.model_name.replace('/', '-')}_hu_{self.method}_len_{max_new_tokens}_num_{num_samples}.jsonl",
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
                        "method": f"hu_{self.method}_bias_free",
                        "context_window": self.watermark.context_window,
                    },
                    "generation_metadata": dict(self.last_generation_metadata),
                    "detection_stats": {
                        "z_score": float(detection.get("z_score", 0.0)),
                        "p_value": float(detection.get("p_value", 1.0)),
                        "is_watermarked": bool(detection.get("is_watermarked", False)),
                        "score": float(detection.get("score", 0.0)),
                        "num_tokens": int(detection.get("num_tokens", 0)),
                        "detector_type": str(detection.get("detector_type", "llr")),
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
            {"prefix": "How does the brain work?", "gold_completion": ""},
            {"prefix": "What is quantum mechanics?", "gold_completion": ""},
            {"prefix": "Describe photosynthesis.", "gold_completion": ""},
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
            stats = {
                "total_samples": len(results),
                "detection_rate": float(np.mean(detections)),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "median_score": float(np.median(scores)),
                "expected_metrics": {
                    "detector": self.detection_config.detector_type,
                    "method": self.method,
                },
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
        "method": "delta",
        "load_in_4bit": True,
    }

    processor = Llama2HuLFQA(
        model_name=config["model_name"],
        load_in_4bit=config["load_in_4bit"],
        method=config["method"],
    )
    stats = processor.process_lfqa_dataset(
        input_file=config["input_file"],
        output_dir=config["output_dir"],
        num_samples=config["num_samples"],
        max_new_tokens=config["max_new_tokens"],
    )
    if stats:
        stats_file = os.path.join(config["output_dir"], "hu_watermark_stats.json")
        with open(stats_file, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        print(f"\nStatistics saved to {stats_file}")


if __name__ == "__main__":
    main()
