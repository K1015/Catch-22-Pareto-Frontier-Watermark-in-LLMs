"""
Llama 2 / CodeLlama wrappers for HeavyWater and SimplexWater.

These schemes are integrated by reusing the upstream logits processors and
detectors from the official HeavyWater_SimplexWater repository while exposing a
stable prompt-aware interface for the reviewer pipelines.
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
os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_JAX", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_CUPY", "1")
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if EXTERNAL_ROOT.exists() and str(EXTERNAL_ROOT) not in sys.path:
    sys.path.append(str(EXTERNAL_ROOT))

from watermark_rebuild_common import load_model_and_tokenizer, load_tokenizer_compat


@dataclass(frozen=True)
class DistortionlessWatermarkConfig:
    mode: str
    alpha: float = 0.05
    initial_seed: int = 1234
    dynamic_seed: str = "markov_1"
    gamma: float = 0.5
    delta: float = 5.0
    bl_type: str = "soft"
    tilt: bool = False
    tilting_delta: float = 0.0
    context: int = 1
    hashing_fn: Optional[str] = None
    sinkhorn_reg: float = 0.05
    sinkhorn_thresh: float = 1e-5
    heavywater_k: int = 1024
    ht_dist: str = "lognormal"


def _load_upstream_components(mode: str):
    if not EXTERNAL_ROOT.exists():
        raise RuntimeError(
            f"Missing upstream HeavyWater/SimplexWater repo at {EXTERNAL_ROOT}. "
            "Clone the external dependency before using this wrapper."
        )

    try:
        if mode == "heavy_tail":
            from watermark.heavy_tail_randscore import (
                HeavyTailLogitsProcessor,
                HeavyTailWatermarkDetector,
            )

            return HeavyTailLogitsProcessor, HeavyTailWatermarkDetector

        from watermark.linear_code import (
            LinearCodeLogitsProcessor,
            LinearCodeWatermarkDetector,
        )

        return LinearCodeLogitsProcessor, LinearCodeWatermarkDetector
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", str(exc))
        raise RuntimeError(
            "HeavyWater/SimplexWater dependencies are incomplete. "
            f"Missing Python module: {missing}. "
            "Install at least `POT` so `import ot` succeeds."
        ) from exc


class DistortionlessWatermarker:
    scheme_name = "DistortionlessWatermark"
    mode = "lin_code"

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
        tilt: bool = False,
        tilting_delta: float = 0.0,
        context: int = 1,
        hashing_fn: Optional[str] = None,
        sinkhorn_reg: float = 0.05,
        sinkhorn_thresh: float = 1e-5,
        heavywater_k: int = 1024,
        ht_dist: str = "lognormal",
        detector_only: bool = False,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit if not load_in_8bit else False
        self.detector_only = detector_only
        self.config = DistortionlessWatermarkConfig(
            mode=self.mode,
            alpha=alpha,
            initial_seed=initial_seed,
            dynamic_seed=dynamic_seed,
            gamma=gamma,
            delta=delta,
            bl_type=bl_type,
            tilt=tilt,
            tilting_delta=tilting_delta,
            context=context,
            hashing_fn=hashing_fn,
            sinkhorn_reg=sinkhorn_reg,
            sinkhorn_thresh=sinkhorn_thresh,
            heavywater_k=heavywater_k,
            ht_dist=ht_dist,
        )

        self._processor_cls, self._detector_cls = _load_upstream_components(self.mode)
        if detector_only:
            self.tokenizer = load_tokenizer_compat(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.model = None
        else:
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
        print(f"LLAMA 2 / CODELLAMA - {self.scheme_name.upper()}")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(
            f"{self.scheme_name} parameters: "
            f"alpha={self.config.alpha}, initial_seed={self.config.initial_seed}, "
            f"dynamic_seed={self.config.dynamic_seed}, gamma={self.config.gamma}, "
            f"delta={self.config.delta}, context={self.config.context}, "
            f"hashing_fn={self.config.hashing_fn}, tilt={self.config.tilt}, "
            f"tilting_delta={self.config.tilting_delta}"
        )
        if self.mode == "heavy_tail":
            print(
                f"HeavyWater extras: k={self.config.heavywater_k}, "
                f"dist={self.config.ht_dist}, sinkhorn_reg={self.config.sinkhorn_reg}, "
                f"sinkhorn_thresh={self.config.sinkhorn_thresh}"
            )
        else:
            print(
                f"SimplexWater extras: sinkhorn_reg={self.config.sinkhorn_reg}, "
                f"sinkhorn_thresh={self.config.sinkhorn_thresh}"
            )

    def _build_processor(self, temperature: float, top_p: float):
        common_kwargs = dict(
            bad_words_ids=None,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab=self.vocab,
            vocab_size=len(self.vocab),
            bl_proportion=1.0 - self.config.gamma,
            bl_logit_bias=self.config.delta,
            bl_type=self.config.bl_type,
            initial_seed=self.config.initial_seed,
            dynamic_seed=self.config.dynamic_seed,
            tilt=self.config.tilt,
            tilting_delta=self.config.tilting_delta,
            top_p=top_p,
            context=self.config.context,
            hashing=self.config.hashing_fn,
            temperature=max(float(temperature), 1e-6),
            sinkhorn_reg=self.config.sinkhorn_reg,
            sinkhorn_thresh=self.config.sinkhorn_thresh,
        )
        if self.mode == "heavy_tail":
            common_kwargs["k"] = self.config.heavywater_k
            common_kwargs["dist"] = self.config.ht_dist
        return self._processor_cls(**common_kwargs)

    def _build_detector(self):
        common_kwargs = dict(
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            gamma=self.config.gamma,
            delta=self.config.delta,
            initial_seed=self.config.initial_seed,
            dynamic_seed=self.config.dynamic_seed,
            device=torch.device(self.device),
            pval=self.config.alpha,
            context=self.config.context,
            hashing=self.config.hashing_fn,
        )
        if self.mode == "heavy_tail":
            common_kwargs["k"] = self.config.heavywater_k
            common_kwargs["dist"] = self.config.ht_dist
        return self._detector_cls(**common_kwargs)

    def generate_watermarked(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        del top_k
        if self.model is None:
            raise RuntimeError("Detector-only HeavyWater/SimplexWater instance cannot generate text.")

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
            "distortionless_mode": self.mode,
            "distortionless_scheme": self.scheme_name.lower(),
            "alpha": float(self.config.alpha),
            "initial_seed": int(self.config.initial_seed),
            "dynamic_seed": self.config.dynamic_seed,
            "gamma": float(self.config.gamma),
            "delta": float(self.config.delta),
            "bl_type": self.config.bl_type,
            "tilt": bool(self.config.tilt),
            "tilting_delta": float(self.config.tilting_delta),
            "context": int(self.config.context),
            "hashing_fn": self.config.hashing_fn,
            "sinkhorn_reg": float(self.config.sinkhorn_reg),
            "sinkhorn_thresh": float(self.config.sinkhorn_thresh),
            "heavywater_k": int(self.config.heavywater_k),
            "ht_dist": self.config.ht_dist,
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
        del temperature, top_p, top_k

        token_ids = list(completion_token_ids or [])
        if not token_ids and generated_text.strip():
            token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)

        num_tokens = len(token_ids)
        threshold = float(norm.ppf(1.0 - self.config.alpha))
        if num_tokens == 0:
            return {
                "is_watermarked": False,
                "score": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
                "threshold": threshold,
                "alpha": float(self.config.alpha),
                "num_tokens": 0,
                "detection_idx": -1,
            }

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).input_ids.to(self.device)
        completion_tensor = torch.tensor([token_ids], dtype=torch.long, device=prompt_inputs.device)
        detector = self._build_detector()
        z_score, detection_idx = detector.detect(
            inputs=prompt_inputs,
            tokenized_text=completion_tensor,
            debug=False,
            return_scores=True,
        )
        p_value = float(1.0 - norm.cdf(z_score))
        return {
            "is_watermarked": bool(p_value < self.config.alpha),
            "score": float(z_score),
            "z_score": float(z_score),
            "p_value": p_value,
            "threshold": threshold,
            "alpha": float(self.config.alpha),
            "num_tokens": int(num_tokens),
            "detection_idx": int(detection_idx),
        }


class Llama2HeavyWaterLFQA(DistortionlessWatermarker):
    scheme_name = "HeavyWater"
    mode = "heavy_tail"


class Llama2SimplexWaterLFQA(DistortionlessWatermarker):
    scheme_name = "SimplexWater"
    mode = "lin_code"
