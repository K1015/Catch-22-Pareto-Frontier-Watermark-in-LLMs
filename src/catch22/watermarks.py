from __future__ import annotations

from .methods import get_method_class
from .methods.base import DetectionResult, stable_hash_int, words
from .registry import require_method


def local_completion(prompt: str, method: str, sample_index: int) -> str:
    topic = " ".join(words(prompt)[:10]) or "the question"
    return (
        f"This response addresses {topic}. It gives a concise explanation, "
        f"connects the main causes, and states the practical implication. "
        f"Sample {sample_index}."
    )


def get_watermark(method: str, seed: int = 1234):
    spec = require_method(method)
    return get_method_class(method)(spec, seed=seed)


def apply_text_watermark(text: str, method: str, seed: int, *, local_backend: bool = True) -> str:
    watermark = get_watermark(method, seed=seed)
    if local_backend:
        return watermark.apply_local_text(text)
    return watermark.postprocess_text(text)


def finalize_generated_text(text: str, method: str, seed: int) -> str:
    return get_watermark(method, seed=seed).postprocess_text(text)


def detect_text(text: str, method: str, seed: int = 1234) -> DetectionResult:
    return get_watermark(method, seed=seed).detect_text(text)


class HashWatermarkLogitsProcessor:
    """Compatibility wrapper dispatching to method-specific watermark hooks."""

    def __init__(self, method: str, seed: int, vocab_size: int):
        self.method = method
        self.seed = seed
        self.vocab_size = vocab_size
        self.watermark = get_watermark(method, seed=seed)

    def __call__(self, input_ids, scores):  # pragma: no cover - requires torch/transformers runtime.
        return self.watermark.process_logits(input_ids, scores)
