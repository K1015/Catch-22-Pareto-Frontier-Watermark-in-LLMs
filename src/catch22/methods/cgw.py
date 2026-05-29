from __future__ import annotations

from .base import DetectionResult, WatermarkMethod, hash_to_unit, normal_sf, sigmoid, words


class CGWWatermark(WatermarkMethod):
    """Christ-Gunn-Watermark-style distribution-preserving inverse-CDF sampling."""

    context_width = 8

    def process_logits(self, input_ids, scores):  # pragma: no cover - requires torch runtime.
        import torch

        output = torch.full_like(scores, -1e9)
        for row in range(scores.shape[0]):
            context = self.context_tokens(input_ids, row=row, width=self.context_width)
            salt = f"cgw:{self.seed}:{':'.join(map(str, context))}:{input_ids.shape[-1]}"
            u_value = hash_to_unit(salt)
            probs = torch.softmax(scores[row], dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=0)
            chosen_rank = int(torch.searchsorted(cdf, torch.tensor(u_value, device=scores.device)).item())
            chosen_rank = min(chosen_rank, scores.shape[-1] - 1)
            output[row, sorted_indices[chosen_rank]] = 0.0
        return output

    def detect_text(self, text: str) -> DetectionResult:
        toks = words(text)
        if not toks:
            return DetectionResult(0.0, 0.5, False, 0, self.method, self.spec.family)
        scores = [abs(hash_to_unit(f"cgw:{self.seed}:{idx}:{token.lower()}") - 0.5) for idx, token in enumerate(toks)]
        avg_centering = sum(0.5 - score for score in scores) / len(scores)
        z_score = avg_centering * (12.0 * len(scores)) ** 0.5
        score = sigmoid(z_score)
        threshold = 0.62
        return DetectionResult(
            score=score,
            threshold=threshold,
            is_watermarked=score >= threshold,
            num_tokens=len(toks),
            method=self.method,
            family=self.spec.family,
            z_score=z_score,
            p_value=normal_sf(z_score),
            details={"avg_centering": avg_centering},
        )
