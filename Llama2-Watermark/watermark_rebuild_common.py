from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


EPS = 1e-12


def tokenizer_kwargs_for_model(model_name: str) -> dict:
    """Route models with tokenizer/runtime incompatibilities through safe settings."""
    lowered = str(model_name).lower()
    if "mistral" in lowered:
        # some clusters ship a tokenizers build that cannot parse the
        # fast Mistral tokenizer JSON reliably, so use the slow tokenizer path.
        return {"use_fast": False}
    return {}


def load_tokenizer_compat(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs_for_model(model_name))


def _debug_log(message: str) -> None:
    if os.environ.get("REVIEWER2_DEBUG") == "1":
        print(message, flush=True)


def to_safe_probs(probs: torch.Tensor) -> torch.Tensor:
    safe = torch.nan_to_num(probs.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe = torch.clamp(safe, min=0.0)
    total = float(safe.sum().item())
    if not np.isfinite(total) or total <= 0:
        safe = torch.zeros_like(safe)
        safe[0] = 1.0
        return safe
    return safe / total


def apply_sampling_filters(probs: torch.Tensor, top_p: float = 1.0, top_k: int = 0) -> torch.Tensor:
    filtered = to_safe_probs(probs)

    if top_k > 0 and top_k < filtered.numel():
        values, indices = torch.topk(filtered, k=top_k)
        masked = torch.zeros_like(filtered)
        masked[indices] = values
        filtered = to_safe_probs(masked)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(filtered, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        to_remove = cumulative > top_p
        if to_remove.numel() > 1:
            to_remove[1:] = to_remove[:-1].clone()
        to_remove[0] = False

        masked = filtered.clone()
        masked[sorted_indices[to_remove]] = 0.0
        filtered = to_safe_probs(masked)

    return filtered


def make_generator(seed: int, device: torch.device | str) -> torch.Generator:
    device_name = str(device)
    try:
        generator = torch.Generator(device=device_name)
    except Exception:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def safe_choice_from_probs(probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> int:
    normalized = to_safe_probs(probs)
    if generator is None:
        return int(torch.multinomial(normalized, num_samples=1).item())
    return int(torch.multinomial(normalized, num_samples=1, generator=generator).item())


def cdf_index_from_u(sorted_probs: torch.Tensor, u: float) -> int:
    normalized = to_safe_probs(sorted_probs).detach().cpu().numpy()
    cumulative = np.cumsum(normalized)
    clipped_u = min(max(float(u), 0.0), np.nextafter(1.0, 0.0))
    idx = int(np.searchsorted(cumulative, clipped_u, side="left"))
    return min(idx, len(normalized) - 1)


def maximin_bounds(base_probs: np.ndarray, wm_probs: np.ndarray, tv_distance: float) -> Tuple[float, float]:
    ratio = wm_probs / np.clip(base_probs, EPS, None)

    max_indices = np.argsort(-ratio)
    sum_q = 0.0
    sum_p = 0.0
    max_lr = 0.0
    for idx in max_indices:
        current = 0.0 if sum_q <= tv_distance or sum_p <= EPS else (sum_q - tv_distance) / sum_p
        if ratio[idx] < current:
            max_lr = current
            break
        sum_q += wm_probs[idx]
        sum_p += base_probs[idx]
        max_lr = 0.0 if sum_q <= tv_distance or sum_p <= EPS else (sum_q - tv_distance) / sum_p

    min_indices = np.argsort(ratio)
    sum_q = 0.0
    sum_p = 0.0
    min_lr = float("inf")
    for idx in min_indices:
        current = float("inf") if sum_p <= EPS else (sum_q + tv_distance) / sum_p
        if ratio[idx] > current:
            min_lr = current
            break
        sum_q += wm_probs[idx]
        sum_p += base_probs[idx]
        min_lr = float("inf") if sum_p <= EPS else (sum_q + tv_distance) / sum_p

    return max(max_lr, 0.0), max(min_lr, 0.0)


def maximin_log_score(base_probs: np.ndarray, wm_probs: np.ndarray, token_id: int, tv_distance: float) -> float:
    max_lr, min_lr = maximin_bounds(base_probs, wm_probs, tv_distance)
    if max_lr <= min_lr:
        return 0.0
    ratio = wm_probs / np.clip(base_probs, EPS, None)
    clipped = np.clip(ratio[token_id], min_lr, max_lr)
    return float(np.log(max(clipped, EPS)))


def load_model_and_tokenizer(
    model_name: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = True,
):
    tokenizer = load_tokenizer_compat(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if load_in_4bit and not load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    return tokenizer, model


def likelihood_ratio_summary(
    token_scores: list[float],
    alpha: float = 0.01,
    extra: Optional[dict] = None,
) -> dict:
    num_tokens = len(token_scores)
    total_score = float(np.sum(token_scores)) if token_scores else 0.0
    threshold = float(-math.log(alpha)) if alpha > 0 else 0.0
    p_value_upper_bound = float(min(1.0, math.exp(-max(total_score, 0.0))))
    summary = {
        "is_watermarked": bool(total_score >= threshold),
        "score": total_score,
        "threshold": threshold,
        "p_value_upper_bound": p_value_upper_bound,
        "p_value": p_value_upper_bound,
        "num_tokens": int(num_tokens),
        "score_per_token": float(total_score / num_tokens) if num_tokens else 0.0,
        "z_score": float(total_score / math.sqrt(max(num_tokens, 1))),
        "alpha": float(alpha),
    }
    if extra:
        summary.update(extra)
    return summary
