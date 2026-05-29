from __future__ import annotations

from .base import WatermarkMethod


class DiPMarkWatermark(WatermarkMethod):
    """Distribution-preserving p-alpha reweighting watermark."""

    alpha = 0.45
    context_width = 5

    def _p_alpha(self, probs, permutation, alpha):  # pragma: no cover - requires torch runtime.
        import torch

        reordered = probs[permutation]
        cdf = torch.cumsum(reordered, dim=0)
        transformed = torch.clamp((cdf - alpha) / max(1.0 - alpha, 1e-8), min=0.0, max=1.0)
        out = torch.zeros_like(probs)
        out[permutation[0]] = transformed[0]
        out[permutation[1:]] = transformed[1:] - transformed[:-1]
        return torch.clamp(out, min=0.0)

    def process_logits(self, input_ids, scores):  # pragma: no cover - requires torch runtime.
        import torch

        output = scores.clone()
        vocab_size = output.shape[-1]
        for row in range(output.shape[0]):
            context = self.context_tokens(input_ids, row=row, width=self.context_width)
            salt = f"dipmark:{self.seed}:{':'.join(map(str, context))}"
            permutation = self.torch_permutation(vocab_size, output.device, salt)
            probs = torch.softmax(scores[row], dim=-1)
            p_alpha = self._p_alpha(probs, permutation, self.alpha)
            p_inverse = self._p_alpha(probs, permutation, 1.0 - self.alpha)
            reweighted = (1.0 - self.alpha) * p_alpha + self.alpha * p_inverse
            reweighted = torch.clamp(reweighted, min=0.0)
            if torch.sum(reweighted) <= 0:
                reweighted = probs
            else:
                reweighted = reweighted / torch.sum(reweighted)
            output[row] = torch.log(torch.clamp(reweighted, min=1e-12))
        return output
