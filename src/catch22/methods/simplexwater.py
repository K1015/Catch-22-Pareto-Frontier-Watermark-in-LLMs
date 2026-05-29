from __future__ import annotations

from .base import WatermarkMethod


class SimplexWaterWatermark(WatermarkMethod):
    """Simplex-style watermark using PRF directions over vocabulary vertices."""

    context_width = 3

    def process_logits(self, input_ids, scores):  # pragma: no cover - requires torch runtime.
        import torch

        output = scores.clone()
        vocab_size = output.shape[-1]
        for row in range(output.shape[0]):
            context = self.context_tokens(input_ids, row=row, width=self.context_width)
            salt = f"simplexwater:{self.seed}:{':'.join(map(str, context))}"
            permutation = self.torch_permutation(vocab_size, output.device, salt)
            ranks = torch.empty(vocab_size, dtype=output.dtype, device=output.device)
            ranks[permutation] = torch.linspace(-1.0, 1.0, vocab_size, dtype=output.dtype, device=output.device)
            output[row] += self.spec.delta * ranks
        return output
