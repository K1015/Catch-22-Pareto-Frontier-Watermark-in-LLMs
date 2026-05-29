from __future__ import annotations

from .base import WatermarkMethod, seed_to_uint32


class GaussMarkWatermark(WatermarkMethod):
    """Inference-compatible Gaussian-key hook for training-time GaussMark runs.

    Full GaussMark experiments should pass a tuned checkpoint through
    ``--model-name-or-path``. This hook keeps the generation and scoring
    interface usable when the checkpoint is evaluated through the common CLI.
    """

    def process_logits(self, input_ids, scores):  # pragma: no cover - requires torch runtime.
        import torch

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed_to_uint32(f"gaussmark:{self.seed}:fixed"))
        key = torch.randn(scores.shape[-1], generator=generator).to(scores.device, dtype=scores.dtype)
        key = (key - key.mean()) / key.std().clamp(min=1e-6)
        return scores + self.spec.delta * key
