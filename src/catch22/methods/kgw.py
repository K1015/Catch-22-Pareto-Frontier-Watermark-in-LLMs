from __future__ import annotations

from .base import WatermarkMethod


class KGWWatermark(WatermarkMethod):
    """Kirchenbauer-style context-dependent green-list watermark."""

    context_width = 1

    def process_logits(self, input_ids, scores):  # pragma: no cover - requires torch runtime.
        output = scores.clone()
        vocab_size = output.shape[-1]
        for row in range(output.shape[0]):
            context = self.context_tokens(input_ids, row=row, width=self.context_width)
            salt = f"kgw:{self.seed}:{':'.join(map(str, context))}"
            mask = self.green_mask(vocab_size, output.device, salt)
            output[row, mask] += self.spec.delta
        return output
