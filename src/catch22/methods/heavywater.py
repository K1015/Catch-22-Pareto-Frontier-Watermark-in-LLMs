from __future__ import annotations

import math

from .base import WatermarkMethod


class HeavyWaterWatermark(WatermarkMethod):
    """Heavy-tailed PRF score watermark for robust token preferences."""

    context_width = 4

    def process_logits(self, input_ids, scores):  # pragma: no cover - requires torch runtime.
        import torch

        output = scores.clone()
        vocab_size = output.shape[-1]
        token_ids = torch.arange(vocab_size, device=output.device)
        for row in range(output.shape[0]):
            context = self.context_tokens(input_ids, row=row, width=self.context_width)
            base = f"heavywater:{self.seed}:{':'.join(map(str, context))}"
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.torch_seed(base))
            uniforms = torch.rand(vocab_size, generator=generator).to(output.device)
            centered = torch.tan(math.pi * (uniforms - 0.5)).clamp(min=-4.0, max=4.0)
            centered = centered - centered.mean()
            output[row, token_ids] += self.spec.delta * centered / centered.std().clamp(min=1e-6)
        return output

    def torch_seed(self, salt: str) -> int:
        from .base import seed_to_uint32

        return seed_to_uint32(salt)
