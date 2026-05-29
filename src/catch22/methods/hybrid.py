from __future__ import annotations

from .base import DetectionResult, WatermarkMethod, append_signature_sentence, binomial_z, signature_score, sigmoid, words


class HybridWatermark(WatermarkMethod):
    """Hybrid watermark combining token-level KGW and semantic stamping."""

    text_postprocess = True
    context_width = 1

    def process_logits(self, input_ids, scores):  # pragma: no cover - requires torch runtime.
        output = scores.clone()
        vocab_size = output.shape[-1]
        for row in range(output.shape[0]):
            context = self.context_tokens(input_ids, row=row, width=self.context_width)
            salt = f"hybrid:kgw:{self.seed}:{':'.join(map(str, context))}"
            mask = self.green_mask(vocab_size, output.device, salt, gamma=0.5)
            output[row, mask] += 0.75 * self.spec.delta
        return output

    def apply_local_text(self, text: str) -> str:
        return self.postprocess_text(text)

    def postprocess_text(self, text: str) -> str:
        return append_signature_sentence(text, self.method, self.seed, label="For verification")

    def detect_text(self, text: str) -> DetectionResult:
        toks = words(text)
        if not toks:
            return DetectionResult(0.0, 0.5, False, 0, self.method, self.spec.family)
        green_count = sum(1 for token in toks if self.token_score(token) < 0.5)
        z_score, p_value = binomial_z(green_count, len(toks), 0.5)
        semantic = signature_score(text, self.method, self.seed)
        score = max(0.0, min(1.0, 0.5 * sigmoid(z_score / 2.0) + 0.5 * semantic))
        threshold = 0.55
        return DetectionResult(
            score=score,
            threshold=threshold,
            is_watermarked=score >= threshold,
            num_tokens=len(toks),
            method=self.method,
            family=self.spec.family,
            z_score=z_score,
            p_value=p_value,
            green_fraction=green_count / len(toks),
            green_count=green_count,
            details={"semantic_score": semantic},
        )
