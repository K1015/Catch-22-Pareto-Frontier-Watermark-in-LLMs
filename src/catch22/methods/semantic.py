from __future__ import annotations

from .base import WatermarkMethod, append_signature_sentence


class SemanticTextWatermark(WatermarkMethod):
    """Base class for sentence-level and semantic-choice watermarks."""

    text_postprocess = True
    label = "Overall"

    def apply_local_text(self, text: str) -> str:
        return self.postprocess_text(text)

    def postprocess_text(self, text: str) -> str:
        return append_signature_sentence(text, self.method, self.seed, label=self.label)


class SemStampWatermark(SemanticTextWatermark):
    """Semantic stamping through stable sentence-level lexical choices."""

    label = "Conceptually"


class PMarkWatermark(SemanticTextWatermark):
    """Paragraph-level semantic watermark using discourse-level signatures."""

    label = "At the paragraph level"


class SimMarkWatermark(SemanticTextWatermark):
    """Similarity-based semantic watermark using stable paraphrase choices."""

    label = "In similar terms"
