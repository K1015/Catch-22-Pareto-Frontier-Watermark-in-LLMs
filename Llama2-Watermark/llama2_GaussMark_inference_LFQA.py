"""
Llama 2 / CodeLlama with a GaussMark structural watermark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from scipy.stats import norm

from watermark_rebuild_common import load_model_and_tokenizer, make_generator


@dataclass(frozen=True)
class GaussMarkConfig:
    watermark_layer: int = 16
    watermark_component: str = "up_proj"
    sigma: float = 0.05
    alpha: float = 0.05
    detection_max_length: int = 128


class GaussMarkWatermarker:
    """
    Structural watermarking by perturbing a single weight tensor with a Gaussian key.

    Detection uses the gradient-correlation statistic from the GaussMark paper.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        watermark_layer: int = 16,
        watermark_component: str = "up_proj",
        sigma: float = 0.05,
        alpha: float = 0.05,
        detection_max_length: int = 128,
    ):
        del load_in_8bit, load_in_4bit

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = GaussMarkConfig(
            watermark_layer=watermark_layer,
            watermark_component=watermark_component,
            sigma=sigma,
            alpha=alpha,
            detection_max_length=detection_max_length,
        )

        # Gradient-based detection is substantially simpler and more reliable on
        # a standard fp16 model than on quantized weights.
        self.tokenizer, self.model = load_model_and_tokenizer(
            model_name=model_name,
            load_in_8bit=False,
            load_in_4bit=False,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.target_weight = self._get_target_weight()
        self.original_weight = self.target_weight.data.clone()

        self.last_completion_token_ids: Optional[list[int]] = None
        self.last_generation_metadata: Dict[str, object] = {}

        print("=" * 80)
        print("LLAMA 2 / CODELLAMA - GAUSSMARK")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(
            "GaussMark parameters: "
            f"layer={self.config.watermark_layer}, "
            f"component={self.config.watermark_component}, "
            f"sigma={self.config.sigma}, alpha={self.config.alpha}"
        )

    def _get_target_weight(self) -> torch.nn.Parameter:
        layer = self.model.model.layers[self.config.watermark_layer]
        component = self.config.watermark_component
        if component == "up_proj":
            return layer.mlp.up_proj.weight
        if component == "down_proj":
            return layer.mlp.down_proj.weight
        if component == "gate_proj":
            return layer.mlp.gate_proj.weight
        raise ValueError(f"Unsupported watermark component: {component}")

    def generate_key(self, seed: int) -> torch.Tensor:
        generator = make_generator(seed, self.target_weight.device)
        return torch.randn(
            self.target_weight.shape,
            generator=generator,
            device=self.target_weight.device,
            dtype=self.target_weight.dtype,
        ) * self.config.sigma

    def apply_watermark(self, key: torch.Tensor) -> None:
        self.target_weight.data = self.original_weight + key.to(
            device=self.target_weight.device,
            dtype=self.target_weight.dtype,
        )

    def remove_watermark(self) -> None:
        self.target_weight.data = self.original_weight.clone()

    def generate_watermarked(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        key_seed: int = 1234,
        do_sample: bool = True,
    ) -> str:
        key = self.generate_key(int(key_seed))
        self.apply_watermark(key)
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)
            input_len = inputs.input_ids.shape[1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(float(temperature), 1e-5),
                top_p=top_p,
                top_k=top_k,
                do_sample=bool(do_sample and temperature > 0.0),
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
            completion_ids = outputs[0][input_len:].detach().cpu().tolist()
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            self.last_completion_token_ids = completion_ids
            self.last_generation_metadata = {
                "gaussmark_key_seed": int(key_seed),
                "watermark_layer": int(self.config.watermark_layer),
                "watermark_component": self.config.watermark_component,
                "sigma": float(self.config.sigma),
                "alpha": float(self.config.alpha),
            }
            return text
        finally:
            self.remove_watermark()

    def compute_test_statistic(self, prompt: str, text: str, key: torch.Tensor) -> float:
        if not text.strip():
            return 0.0

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not completion_ids:
            return 0.0

        max_detection_tokens = max(int(self.config.detection_max_length), 1)
        if len(completion_ids) >= max_detection_tokens:
            kept_prompt_ids = []
            kept_completion_ids = completion_ids[-max_detection_tokens:]
        else:
            prompt_budget = max_detection_tokens - len(completion_ids)
            kept_prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
            kept_completion_ids = completion_ids

        combined_ids = kept_prompt_ids + kept_completion_ids
        prompt_length = len(kept_prompt_ids)
        input_ids = torch.tensor([combined_ids], device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        if prompt_length > 0:
            labels[:, : min(prompt_length, labels.shape[1])] = -100

        old_requires_grad = self.target_weight.requires_grad
        self.target_weight.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                return 0.0

            grad = torch.autograd.grad(
                -loss,
                self.target_weight,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                return 0.0

            key_flat = key.to(device=grad.device, dtype=grad.dtype).flatten().float()
            grad_flat = grad.flatten().float()

            grad_norm = float(torch.norm(grad_flat).item())
            if grad_norm <= 1e-12 or self.config.sigma <= 0.0:
                return 0.0

            inner_product = float(torch.dot(key_flat, grad_flat).item())
            return inner_product / (self.config.sigma * grad_norm)
        finally:
            self.model.zero_grad(set_to_none=True)
            self.target_weight.requires_grad_(old_requires_grad)

    def detect_with_prompt(
        self,
        prompt: str,
        generated_text: str,
        generation_metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        metadata = dict(generation_metadata or {})
        key_seed = metadata.get("gaussmark_key_seed")
        threshold = float(norm.ppf(1.0 - self.config.alpha))
        num_tokens = len(self.tokenizer.encode(generated_text, add_special_tokens=False))

        if key_seed is None:
            return {
                "is_watermarked": False,
                "score": 0.0,
                "z_score": 0.0,
                "test_statistic": 0.0,
                "p_value": 1.0,
                "threshold": threshold,
                "alpha": float(self.config.alpha),
                "num_tokens": int(num_tokens),
                "error": "missing_gaussmark_key_seed",
            }

        key = self.generate_key(int(key_seed))
        test_statistic = float(self.compute_test_statistic(prompt, generated_text, key))
        p_value = float(1.0 - norm.cdf(test_statistic))
        return {
            "is_watermarked": bool(p_value < self.config.alpha),
            "score": test_statistic,
            "z_score": test_statistic,
            "test_statistic": test_statistic,
            "p_value": p_value,
            "threshold": threshold,
            "alpha": float(self.config.alpha),
            "num_tokens": int(num_tokens),
            "gaussmark_key_seed": int(key_seed),
        }


class Llama2GaussMarkLFQA(GaussMarkWatermarker):
    """Compatibility wrapper matching the naming pattern used by the other scripts."""
