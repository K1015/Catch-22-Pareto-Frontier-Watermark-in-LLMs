from __future__ import annotations

from .base import WatermarkMethod, seed_to_uint32


class KuditipudiWatermark(WatermarkMethod):
    """Gumbel-key watermark in the style of randomized sampling keys."""

    context_width = 4

    def process_logits(self, input_ids, scores):  # pragma: no cover - requires torch runtime.
        import torch

        output = scores.clone()
        vocab_size = output.shape[-1]
        for row in range(output.shape[0]):
            context = self.context_tokens(input_ids, row=row, width=self.context_width)
            salt = f"kuditipudi:{self.seed}:{':'.join(map(str, context))}"
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed_to_uint32(salt))
            u_value = torch.rand(vocab_size, generator=generator).to(output.device).clamp(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u_value))
            output[row] += self.spec.delta * (gumbel - gumbel.mean()) / gumbel.std().clamp(min=1e-6)
        return output
