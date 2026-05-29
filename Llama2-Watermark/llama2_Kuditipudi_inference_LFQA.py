"""
Llama 2 / CodeLlama wrapper for the Kuditipudi et al. inverse-transform watermark.

This local implementation reuses the upstream inverse-transform logits processor
and detector shipped inside the vendored HeavyWater/SimplexWater dependency.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from scipy.stats import norm
from transformers import LogitsProcessorList


THIS_DIR = Path(__file__).resolve().parent
EXTERNAL_ROOT = THIS_DIR.parent / "external" / "HeavyWater_SimplexWater"
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if EXTERNAL_ROOT.exists() and str(EXTERNAL_ROOT) not in sys.path:
    sys.path.append(str(EXTERNAL_ROOT))

from watermark_rebuild_common import load_model_and_tokenizer

try:
    from watermark.inverse_transform import InverseTransformDetector, InverseTransformLogitsProcessor
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on the cluster.
    raise RuntimeError(
        f"Failed to import inverse-transform watermark implementation from {EXTERNAL_ROOT}. "
        "Make sure the vendored HeavyWater_SimplexWater dependency is present."
    ) from exc


@dataclass(frozen=True)
class KuditipudiConfig:
    alpha: float = 0.05
    initial_seed: int = 1234
    dynamic_seed: str = "markov_1"
    gamma: float = 0.5
    delta: float = 5.0
    bl_type: str = "soft"
    pval: float = 0.01


class Llama2KuditipudiLFQA:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        alpha: float = 0.05,
        initial_seed: int = 1234,
        dynamic_seed: str = "markov_1",
        gamma: float = 0.5,
        delta: float = 5.0,
        bl_type: str = "soft",
        pval: float = 0.01,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit if not load_in_8bit else False
        self.config = KuditipudiConfig(
            alpha=float(alpha),
            initial_seed=int(initial_seed),
            dynamic_seed=str(dynamic_seed),
            gamma=float(gamma),
            delta=float(delta),
            bl_type=str(bl_type),
            pval=float(pval),
        )

        self.tokenizer, self.model = load_model_and_tokenizer(
            self.model_name,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
        )
        self.model.eval()
        self.vocab = list(range(self.tokenizer.vocab_size))

        self.last_completion_token_ids: list[int] = []
        self.last_generation_metadata: Dict[str, object] = {}

        print("=" * 80)
        print("LLAMA 2 / CODELLAMA - KUDITIPUDI")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(
            "Kuditipudi parameters: "
            f"alpha={self.config.alpha}, "
            f"initial_seed={self.config.initial_seed}, "
            f"dynamic_seed={self.config.dynamic_seed}, "
            f"gamma={self.config.gamma}, "
            f"delta={self.config.delta}, "
            f"pval={self.config.pval}"
        )

    def _build_processor(self, temperature: float, top_p: float):
        return InverseTransformLogitsProcessor(
            bad_words_ids=None,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab=self.vocab,
            vocab_size=len(self.vocab),
            bl_proportion=1.0 - self.config.gamma,
            bl_logit_bias=self.config.delta,
            bl_type=self.config.bl_type,
            initial_seed=self.config.initial_seed,
            dynamic_seed=self.config.dynamic_seed,
            top_p=float(top_p),
            temperature=max(float(temperature), 1e-6),
        )

    def _build_detector(self):
        return InverseTransformDetector(
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            gamma=self.config.gamma,
            delta=self.config.delta,
            initial_seed=self.config.initial_seed,
            dynamic_seed=self.config.dynamic_seed,
            device=torch.device(self.device),
            pval=self.config.pval,
        )

    def generate_watermarked(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        del top_k

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        processor = self._build_processor(temperature=temperature, top_p=top_p)

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            logits_processor=LogitsProcessorList([processor]),
            do_sample=True,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        completion_ids = outputs[0][prompt_length:].detach().cpu().tolist()
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        self.last_completion_token_ids = completion_ids
        self.last_generation_metadata = {
            "kuditipudi_scheme": "inverse_transform",
            "alpha": float(self.config.alpha),
            "initial_seed": int(self.config.initial_seed),
            "dynamic_seed": self.config.dynamic_seed,
            "gamma": float(self.config.gamma),
            "delta": float(self.config.delta),
            "bl_type": self.config.bl_type,
            "pval": float(self.config.pval),
        }
        return text

    def detect_with_prompt(
        self,
        prompt: str,
        generated_text: str,
        completion_token_ids: Optional[Sequence[int]] = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> Dict[str, object]:
        del generated_text, temperature, top_p, top_k

        token_ids = list(completion_token_ids or [])
        threshold = float(norm.ppf(1.0 - self.config.pval))
        if not token_ids:
            return {
                "is_watermarked": False,
                "score": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
                "threshold": threshold,
                "num_tokens": 0,
                "detection_idx": -1,
            }

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).input_ids.to(self.device)
        completion_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        detector = self._build_detector()
        t_stat, p_value, _, detection_idx = detector.detect(
            inputs=prompt_inputs,
            tokenized_text=completion_tensor,
            debug=False,
            return_scores=True,
        )
        raw_t_statistic = float(t_stat)
        # The upstream inverse-transform detector reports negative t-statistics
        # when generated tokens follow the watermark's seeded CDF draws.
        z_score = -raw_t_statistic
        return {
            "is_watermarked": bool(raw_t_statistic < 0.0 and float(p_value) < self.config.pval),
            "score": z_score,
            "z_score": z_score,
            "raw_t_statistic": raw_t_statistic,
            "p_value": float(p_value),
            "threshold": threshold,
            "num_tokens": int(len(token_ids)),
            "detection_idx": int(detection_idx),
        }
