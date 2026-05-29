from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9']+")
SIGNATURE_BANK = [
    "therefore",
    "notably",
    "consistent",
    "balanced",
    "context",
    "evidence",
    "structure",
    "practical",
    "careful",
    "linked",
    "robust",
    "measured",
    "central",
    "direct",
    "stable",
    "specific",
    "grounded",
    "useful",
    "clear",
    "relevant",
    "detailed",
    "reasoned",
    "connected",
    "focused",
]


@dataclass
class DetectionResult:
    score: float
    threshold: float
    is_watermarked: bool
    num_tokens: int
    method: str
    family: str
    z_score: float = 0.0
    p_value: float = 1.0
    green_fraction: float = 0.0
    green_count: int = 0
    details: dict[str, Any] | None = None


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def hash_to_unit(text: str) -> float:
    return (stable_hash_int(text) % 10_000_000) / 10_000_000.0


def seed_to_uint32(text: str) -> int:
    return stable_hash_int(text) % (2**32)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def normal_sf(value: float) -> float:
    return 0.5 * math.erfc(value / math.sqrt(2.0))


def binomial_z(count: int, total: int, expected_fraction: float) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    expected = total * expected_fraction
    variance = max(total * expected_fraction * (1.0 - expected_fraction), 1e-9)
    z_score = (count - expected) / math.sqrt(variance)
    return z_score, normal_sf(z_score)


def method_signature_terms(method: str, seed: int, count: int = 8) -> list[str]:
    ranked = sorted(SIGNATURE_BANK, key=lambda term: stable_hash_int(f"{method}:{seed}:{term}"))
    return ranked[:count]


def append_signature_sentence(text: str, method: str, seed: int, *, label: str = "Overall") -> str:
    terms = method_signature_terms(method, seed)
    signature = ", ".join(terms[:4])
    return f"{text.rstrip()} {label}, the answer remains {signature}."


def signature_score(text: str, method: str, seed: int) -> float:
    observed = {token.lower() for token in words(text)}
    expected = set(method_signature_terms(method, seed))
    if not expected:
        return 0.0
    return len(observed & expected) / len(expected)


class WatermarkMethod:
    """Common interface for watermark implementations.

    The full model path uses ``process_logits`` during generation. The local
    backend uses ``apply_local_text`` so dry runs can still exercise detector
    and evaluation code without model downloads.
    """

    text_postprocess: bool = False

    def __init__(self, spec, seed: int = 1234):
        self.spec = spec
        self.seed = seed
        self.method = spec.name

    def apply_local_text(self, text: str) -> str:
        if self.method == "vanilla":
            return text
        return append_signature_sentence(text, self.method, self.seed)

    def postprocess_text(self, text: str) -> str:
        if self.text_postprocess:
            return append_signature_sentence(text, self.method, self.seed)
        return text

    def token_score(self, token: str) -> float:
        return hash_to_unit(f"{self.method}:{self.seed}:{token.lower()}")

    def detect_text(self, text: str) -> DetectionResult:
        toks = words(text)
        if self.method == "vanilla" or not toks:
            return DetectionResult(0.0, 0.5, False, len(toks), self.method, self.spec.family)
        green_count = sum(1 for token in toks if self.token_score(token) < self.spec.gamma)
        z_score, p_value = binomial_z(green_count, len(toks), self.spec.gamma)
        lexical_score = sigmoid(z_score / 2.0)
        signature_bonus = 0.35 * signature_score(text, self.method, self.seed)
        score = max(0.0, min(1.0, 0.55 * lexical_score + signature_bonus))
        threshold = 0.55
        return DetectionResult(
            score=score,
            threshold=threshold,
            is_watermarked=score >= threshold,
            num_tokens=len(toks),
            method=self.method,
            family=self.spec.family,
            z_score=z_score,
            p_value=p_value,
            green_fraction=green_count / len(toks),
            green_count=green_count,
            details={"signature_score": signature_score(text, self.method, self.seed)},
        )

    def process_logits(self, input_ids, scores):  # pragma: no cover - requires torch runtime.
        return scores

    def context_tokens(self, input_ids, row: int = 0, width: int = 5) -> list[int]:
        if input_ids is None or input_ids.numel() == 0:
            return []
        tokens = input_ids[row].detach().cpu().tolist()
        return [int(token) for token in tokens[-width:]]

    def torch_permutation(self, vocab_size: int, device, salt: str):  # pragma: no cover - requires torch runtime.
        import torch

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed_to_uint32(salt))
        return torch.randperm(vocab_size, generator=generator, dtype=torch.long).to(device)

    def green_mask(self, vocab_size: int, device, salt: str, gamma: float | None = None):  # pragma: no cover
        import torch

        gamma = self.spec.gamma if gamma is None else gamma
        cutoff = max(1, int(vocab_size * gamma))
        perm = self.torch_permutation(vocab_size, device, salt)
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        mask[perm[:cutoff]] = True
        return mask
