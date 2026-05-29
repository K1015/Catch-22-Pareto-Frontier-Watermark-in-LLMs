from __future__ import annotations

from .base import WatermarkMethod, hash_to_unit


class HCWWatermark(WatermarkMethod):
    """Hu-style bias-free inverse-CDF reweighting watermark."""

    context_width = 5

    def process_logits(self, input_ids, scores):  # pragma: no cover - requires torch runtime.
        import torch

        output = torch.full_like(scores, -1e9)
        for row in range(scores.shape[0]):
            context = self.context_tokens(input_ids, row=row, width=self.context_width)
            salt = f"hcw:{self.seed}:{':'.join(map(str, context))}"
            u_value = hash_to_unit(salt)
            probs = torch.softmax(scores[row], dim=-1)
            cdf = torch.cumsum(probs, dim=0)
            chosen = int(torch.searchsorted(cdf, torch.tensor(u_value, device=scores.device)).item())
            chosen = min(chosen, scores.shape[-1] - 1)
            output[row, chosen] = 0.0
        return output
