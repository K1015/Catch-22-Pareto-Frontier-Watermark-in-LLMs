from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from .registry import MethodSpec, require_method


WORD_RE = re.compile(r"[A-Za-z0-9']+")


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def token_score(token: str, method: str, seed: int) -> float:
    value = stable_hash_int(f"{method}:{seed}:{token.lower()}") % 10_000
    return value / 10_000.0


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


@dataclass
class DetectionResult:
    score: float
    threshold: float
    is_watermarked: bool
    num_tokens: int
    method: str
    family: str


def watermark_marker(method: str) -> str:
    return f"[wm:{method}]"


def local_completion(prompt: str, method: str, sample_index: int) -> str:
    marker = "" if method == "vanilla" else f" {watermark_marker(method)}"
    topic = " ".join(words(prompt)[:10]) or "the question"
    return (
        f"This response addresses {topic}. It gives a concise explanation, "
        f"connects the main causes, and states the practical implication.{marker} "
        f"Sample {sample_index}."
    )


def apply_text_watermark(text: str, method: str, seed: int) -> str:
    if method == "vanilla":
        return text
    spec = require_method(method)
    if method == "hybrid":
        return f"{text} {watermark_marker('kgw')} {watermark_marker('semstamp')} {watermark_marker('hybrid')}"
    if spec.family in {"semantic", "hybrid"}:
        return f"{text} {watermark_marker(method)}"
    return text


def detect_text(text: str, method: str, seed: int = 1234) -> DetectionResult:
    spec = require_method(method)
    toks = words(text)
    if method == "vanilla" or not toks:
        return DetectionResult(0.0, 0.5, False, len(toks), method, spec.family)
    marker_bonus = 0.75 if watermark_marker(method) in text else 0.0
    if method == "hybrid":
        marker_bonus = 0.25 * sum(
            marker in text
            for marker in (watermark_marker("kgw"), watermark_marker("semstamp"), watermark_marker("hybrid"))
        )
    green = sum(1 for token in toks if token_score(token, method, seed) < spec.gamma)
    expected = len(toks) * spec.gamma
    variance = max(len(toks) * spec.gamma * (1.0 - spec.gamma), 1e-6)
    z_score = (green - expected) / math.sqrt(variance)
    normalized = 1.0 / (1.0 + math.exp(-z_score / 2.0))
    score = max(0.0, min(1.0, 0.35 * normalized + marker_bonus))
    threshold = 0.55
    return DetectionResult(score, threshold, score >= threshold, len(toks), method, spec.family)


class HashWatermarkLogitsProcessor:
    """Small deterministic logits processor used by the full Hugging Face backend."""

    def __init__(self, method: str, seed: int, vocab_size: int):
        self.spec = require_method(method)
        self.method = method
        self.seed = seed
        self.vocab_size = vocab_size

    def __call__(self, input_ids, scores):  # pragma: no cover - requires torch/transformers runtime.
        if self.method == "vanilla":
            return scores
        import torch

        last_token = int(input_ids[0, -1]) if input_ids.numel() else 0
        generator = torch.Generator(device=scores.device)
        generator.manual_seed((stable_hash_int(f"{self.method}:{self.seed}:{last_token}") % (2**31 - 1)))
        green = torch.rand(self.vocab_size, generator=generator, device=scores.device) < self.spec.gamma
        boost = self.spec.delta
        if self.method == "hybrid":
            boost *= 0.75
        scores[:, green] += boost
        return scores
