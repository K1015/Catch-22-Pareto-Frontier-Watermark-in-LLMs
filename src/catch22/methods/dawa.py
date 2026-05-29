from __future__ import annotations

from .base import WatermarkMethod, hash_to_unit


class DAWAWatermark(WatermarkMethod):
    """Distribution-aware adaptive watermark.

    The hook uses entropy to decide whether to apply a biased green-list boost
    or leave low-entropy steps mostly unchanged.
    """

    context_width = 2
    entropy_threshold = 2.5

    def process_logits(self, input_ids, scores):  # pragma: no cover - requires torch runtime.
        import torch

        output = scores.clone()
        vocab_size = output.shape[-1]
        for row in range(output.shape[0]):
            probs = torch.softmax(scores[row], dim=-1)
            entropy = float(-(probs * torch.log(torch.clamp(probs, min=1e-12))).sum().item())
            context = self.context_tokens(input_ids, row=row, width=self.context_width)
            salt = f"dawa:{self.seed}:{':'.join(map(str, context))}"
            if entropy >= self.entropy_threshold:
                mask = self.green_mask(vocab_size, output.device, salt, gamma=min(0.7, self.spec.gamma + 0.1))
                output[row, mask] += self.spec.delta
            else:
                u_value = hash_to_unit(salt)
                cdf = torch.cumsum(probs, dim=0)
                chosen = int(torch.searchsorted(cdf, torch.tensor(u_value, device=output.device)).item())
                chosen = min(chosen, vocab_size - 1)
                output[row] = -1e9
                output[row, chosen] = 0.0
        return output
