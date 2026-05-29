from __future__ import annotations

from .base import WatermarkMethod


class UnigramWatermark(WatermarkMethod):
    """Fixed-green-list unigram watermark."""

    def process_logits(self, input_ids, scores):  # pragma: no cover - requires torch runtime.
        output = scores.clone()
        vocab_size = output.shape[-1]
        mask = self.green_mask(vocab_size, output.device, f"unigram:{self.seed}:fixed")
        output[:, mask] += self.spec.delta
        return output
