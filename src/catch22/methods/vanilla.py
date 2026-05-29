from __future__ import annotations

from .base import DetectionResult, WatermarkMethod, words


class VanillaWatermark(WatermarkMethod):
    """Unwatermarked baseline."""

    def apply_local_text(self, text: str) -> str:
        return text

    def detect_text(self, text: str) -> DetectionResult:
        return DetectionResult(0.0, 0.5, False, len(words(text)), self.method, self.spec.family)
