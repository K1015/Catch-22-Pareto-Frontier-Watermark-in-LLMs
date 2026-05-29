"""
Llama 2 / CodeLlama with a paper-aligned Hu et al. (2024) unbiased watermark.

This version keeps the delta/gamma reweighting family, but replaces the previous
heuristic detector with prompt-aware likelihood scoring:
  - standard LLR (Section 5.2)
  - maximin LLR (Section 5.3 / Algorithm 2)

The implementation is intended for reviewer2 code experiments and exposes only
the generation/detection pieces needed by the MBPP runner.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from watermark_rebuild_common import load_tokenizer_compat


EPS = 1e-12


def _debug_log(message: str) -> None:
    if os.environ.get("REVIEWER2_DEBUG") == "1":
        print(message, flush=True)


def _to_safe_probs(probs: torch.Tensor) -> torch.Tensor:
    safe = torch.nan_to_num(probs.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe = torch.clamp(safe, min=0.0)
    total = safe.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0:
        safe = torch.zeros_like(safe)
        safe[0] = 1.0
        return safe
    return safe / total


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    device_str = str(device) if isinstance(device, torch.device) else str(device)
    try:
        generator = torch.Generator(device=device_str)
    except Exception:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _apply_sampling_filters(probs: torch.Tensor, top_p: float = 1.0, top_k: int = 0) -> torch.Tensor:
    filtered = _to_safe_probs(probs)

    if top_k > 0 and top_k < filtered.numel():
        values, indices = torch.topk(filtered, k=top_k)
        masked = torch.zeros_like(filtered)
        masked[indices] = values
        filtered = _to_safe_probs(masked)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(filtered, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        to_remove = cumulative > top_p
        if to_remove.numel() > 1:
            to_remove[1:] = to_remove[:-1].clone()
        to_remove[0] = False

        masked = filtered.clone()
        masked[sorted_indices[to_remove]] = 0.0
        filtered = _to_safe_probs(masked)

    return filtered


def _safe_choice_from_probs(probs: torch.Tensor) -> int:
    normalized = _to_safe_probs(probs)
    return int(torch.multinomial(normalized, num_samples=1).item())


def _maximin_bounds(base_probs: np.ndarray, wm_probs: np.ndarray, tv_distance: float) -> Tuple[float, float]:
    ratio = wm_probs / np.clip(base_probs, EPS, None)

    max_indexes = np.argsort(-ratio)
    sum_q = 0.0
    sum_p = 0.0
    max_lr = 0.0
    for idx in max_indexes:
        current = 0.0 if sum_q <= tv_distance or sum_p <= EPS else (sum_q - tv_distance) / sum_p
        if ratio[idx] < current:
            max_lr = current
            break
        sum_q += wm_probs[idx]
        sum_p += base_probs[idx]
        max_lr = 0.0 if sum_q <= tv_distance or sum_p <= EPS else (sum_q - tv_distance) / sum_p

    min_indexes = np.argsort(ratio)
    sum_q = 0.0
    sum_p = 0.0
    min_lr = float("inf")
    for idx in min_indexes:
        current = float("inf") if sum_p <= EPS else (sum_q + tv_distance) / sum_p
        if ratio[idx] > current:
            min_lr = current
            break
        sum_q += wm_probs[idx]
        sum_p += base_probs[idx]
        min_lr = float("inf") if sum_p <= EPS else (sum_q + tv_distance) / sum_p

    return max(max_lr, 0.0), max(min_lr, 0.0)


def _maximin_log_score(base_probs: np.ndarray, wm_probs: np.ndarray, token_id: int, tv_distance: float) -> float:
    max_lr, min_lr = _maximin_bounds(base_probs, wm_probs, tv_distance)
    if max_lr <= min_lr:
        return 0.0
    ratio = wm_probs / np.clip(base_probs, EPS, None)
    clipped = np.clip(ratio[token_id], min_lr, max_lr)
    return float(np.log(max(clipped, EPS)))


@dataclass
class HuDetectionConfig:
    alpha: float = 0.01
    detector_type: str = "llr"
    tv_distance: float = 0.0


class PaperAlignedUnbiasedWatermark:
    """Hu et al. reweighting family with keyed context-dependent watermark codes."""

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

    def _hash_to_seed(self, context_code: str) -> int:
        digest = hashlib.sha256(f"{context_code}_{self.key}".encode()).digest()
        return int.from_bytes(digest[:4], byteorder="big") % (2**32)

    def _pick_from_cdf(self, probs: torch.Tensor, seed: int) -> int:
        safe = _to_safe_probs(probs)
        generator = _make_generator(seed, safe.device)
        u = torch.rand((), generator=generator, device=safe.device, dtype=safe.dtype)
        upper = torch.tensor(1.0 - 1e-7, device=safe.device, dtype=safe.dtype)
        cumulative = torch.cumsum(safe, dim=0)
        idx = int(torch.searchsorted(cumulative, torch.minimum(u, upper), right=False).item())
        return min(idx, cumulative.numel() - 1)

    def _delta_reweight(self, probs: torch.Tensor, seed: int) -> torch.Tensor:
        safe = _to_safe_probs(probs)
        idx = self._pick_from_cdf(safe, seed)
        delta = torch.zeros_like(safe)
        delta[idx] = 1.0
        return delta

    def _gamma_reweight(self, probs: torch.Tensor, seed: int) -> torch.Tensor:
        safe = _to_safe_probs(probs)
        generator = _make_generator(seed, safe.device)

        indices = torch.randperm(safe.numel(), generator=generator, device=safe.device)
        shuffled = safe.index_select(0, indices)
        cumulative = torch.cumsum(shuffled, dim=0)
        transformed = torch.clamp(2 * cumulative - 1, min=0.0)

        reweighted = torch.zeros_like(safe)
        transformed_prev = torch.cat([torch.zeros(1, device=safe.device, dtype=safe.dtype), transformed[:-1]])
        increments = transformed - transformed_prev
        reweighted.scatter_(0, indices, increments)
        return _to_safe_probs(reweighted)

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
        safe = _to_safe_probs(probs)

        if repeated:
            return safe, {"context_code": context_code, "reweighted": False, "repeated_context": True}

        history.add(context_code)
        seed = self._hash_to_seed(context_code)
        if method == "delta":
            reweighted = self._delta_reweight(safe, seed)
        elif method == "gamma":
            reweighted = self._gamma_reweight(safe, seed)
        else:
            raise ValueError(f"Unsupported Hu reweighting method: {method}")

        return reweighted, {"context_code": context_code, "reweighted": True, "repeated_context": False}


class Llama2HuV2LFQA:
    """Paper-aligned Hu et al. watermark wrapper for generation and likelihood-based detection."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        watermark_key: Optional[str] = None,
        method: str = "gamma",
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

        print("=" * 80)
        print("LLAMA 2 / CODELLAMA - HU ET AL. UNBIASED WATERMARK V2")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Method: {method}-reweight")
        print(f"Detector: {self.detection_config.detector_type}")
        print(f"TV distance: {self.detection_config.tv_distance}")

        self._load_model()
        self.watermark = PaperAlignedUnbiasedWatermark(
            vocab_size=self.tokenizer.vocab_size,
            key=watermark_key,
            context_window=context_window,
            preserve_context_history=preserve_context_history,
        )

    def _load_model(self) -> None:
        print(f"\nLoading {self.model_name}...")
        self.tokenizer = load_tokenizer_compat(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
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
    ) -> Tuple[str, Dict[str, object]]:
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
            f"[HCW-V2] prompt_tokens={len(prompt_ids)} max_new_tokens={max_new_tokens} "
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

            base_probs = _apply_sampling_filters(torch.softmax(logits, dim=-1), top_p=top_p, top_k=top_k)
            wm_probs, info = self.watermark.reweight_distribution(
                base_probs,
                context_tokens=generated_ids,
                method=self.method,
            )

            if info["reweighted"]:
                reweighted_steps += 1
            if info["repeated_context"]:
                skipped_repeats += 1

            next_token = _safe_choice_from_probs(wm_probs)
            generated_ids.append(next_token)
            completion_ids.append(next_token)
            model_inputs = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

            if next_token == self.tokenizer.eos_token_id or len(generated_ids) >= 1024:
                break

        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        metadata = {
            "completion_token_count": len(completion_ids),
            "completion_token_ids": completion_ids,
            "reweighted_steps": reweighted_steps,
            "skipped_repeated_contexts": skipped_repeats,
            "detector_type": self.detection_config.detector_type,
            "preserve_context_history": self.watermark.preserve_context_history,
        }
        _debug_log(f"[HCW-V2] completion_tokens={len(completion_ids)} text_chars={len(text)}")
        return text, metadata

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
        initial_context_history: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        alpha = self.detection_config.alpha if alpha is None else alpha
        detector_type = self.detection_config.detector_type if detector_type is None else detector_type
        tv_distance = self.detection_config.tv_distance if tv_distance is None else tv_distance

        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids[0].tolist()
        if completion_token_ids is None:
            completion_token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)

        completion_ids = list(completion_token_ids)
        if not completion_ids:
            return {
                "is_watermarked": False,
                "detector_type": detector_type,
                "score": 0.0,
                "threshold": float(-math.log(alpha)),
                "p_value_upper_bound": 1.0,
                "num_tokens": 0,
                "score_per_token": 0.0,
                "alpha": float(alpha),
                "tv_distance": float(tv_distance),
            }

        full_ids = prompt_ids + completion_ids
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor)
        logits = outputs.logits[0]

        context_history = set(initial_context_history or [])
        token_scores: List[float] = []
        repeated_contexts = 0

        for idx, token_id in enumerate(completion_ids):
            token_position = len(prompt_ids) + idx
            logit_position = token_position - 1
            if logit_position < 0:
                continue

            base_probs = _apply_sampling_filters(
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

            base_np = _to_safe_probs(base_probs).numpy()
            wm_np = _to_safe_probs(wm_probs).numpy()
            if detector_type == "llr":
                score = float(np.log(max(wm_np[token_id], EPS)) - np.log(max(base_np[token_id], EPS)))
            elif detector_type == "maximin_llr":
                score = _maximin_log_score(base_np, wm_np, token_id=token_id, tv_distance=tv_distance)
            else:
                raise ValueError(f"Unsupported detector type: {detector_type}")
            token_scores.append(score)

        total_score = float(np.sum(token_scores))
        threshold = float(-math.log(alpha))
        p_value_upper_bound = float(min(1.0, math.exp(-max(total_score, 0.0))))

        return {
            "is_watermarked": bool(total_score >= threshold),
            "detector_type": detector_type,
            "score": total_score,
            "threshold": threshold,
            "p_value_upper_bound": p_value_upper_bound,
            "num_tokens": int(len(completion_ids)),
            "score_per_token": float(total_score / len(completion_ids)) if completion_ids else 0.0,
            "alpha": float(alpha),
            "tv_distance": float(tv_distance),
            "repeated_contexts": int(repeated_contexts),
            "z_score": None,
        }
