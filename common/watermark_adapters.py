"""Adapters that wrap the existing LFQA watermark scripts for downstream experiments."""

from __future__ import annotations

import hashlib
import importlib.util
import gc
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WATERMARK_DIR = ROOT / "Llama2-Watermark"


SCRIPT_PATHS = {
    "vanilla": "llama2_vanilla_inference_LFQA.py",
    "kgw": "llama2_KGW_inference_LFQA.py",
    "unigram": "llama2_Unigram_inference_LFQA.py",
    "dipmark": "llama2_DiPMark_inference_LFQA.py",
    "hcw": "llama2_HCW_inference_LFQA.py",
    "hcw-v2": "llama2_HCW_v2_inference_LFQA.py",
    "kuditipudi": "llama2_Kuditipudi_inference_LFQA.py",
    "cgw": "llama2_CGW_inference_LFQA.py",
    "gaussmark": "llama2_GaussMark_inference_LFQA.py",
    "heavywater": "llama2_HeavyWater_inference_LFQA.py",
    "pmark": "llama2_PMark_inference_LFQA.py",
    "semstamp": "llama2_SemStamp_inference_LFQA.py",
    "simmark": "llama2_SimMark_inference_LFQA.py",
    "simplexwater": "llama2_SimplexWater_inference_LFQA.py",
    "hybrid": "llama2_Hybrid_inference_LFQA.py",
}

MODULE_CACHE: Dict[str, object] = {}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _apply_pmark_env_overrides(method_kwargs: Dict) -> Dict:
    updated = dict(method_kwargs)
    if os.environ.get("PMARK_NUM_SAMPLES"):
        updated["num_samples"] = int(os.environ["PMARK_NUM_SAMPLES"])
    if os.environ.get("PMARK_MIN_SEQUENCES"):
        updated["min_sequences"] = int(os.environ["PMARK_MIN_SEQUENCES"])
    if os.environ.get("PMARK_MAX_SENTENCES"):
        updated["max_sentences"] = int(os.environ["PMARK_MAX_SENTENCES"])
    if os.environ.get("PMARK_MSIG"):
        updated["msig"] = int(os.environ["PMARK_MSIG"])
    if os.environ.get("PMARK_DEDUP"):
        updated["dedup"] = _env_bool("PMARK_DEDUP", bool(updated.get("dedup", False)))
    if os.environ.get("PMARK_PARALLEL"):
        updated["parallel"] = _env_bool("PMARK_PARALLEL", bool(updated.get("parallel", False)))
    return updated


def _sampling_params_from_metadata(metadata: Dict) -> tuple[float, float, int]:
    return (
        float(metadata.get("temperature", metadata.get("generation_temperature", 0.8))),
        float(metadata.get("top_p", metadata.get("generation_top_p", 0.95))),
        int(metadata.get("top_k", metadata.get("generation_top_k", 50))),
    )


def available_methods() -> Dict[str, str]:
    return dict(SCRIPT_PATHS)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_module(method: str):
    if method in MODULE_CACHE:
        return MODULE_CACHE[method]

    path = WATERMARK_DIR / SCRIPT_PATHS[method]
    if str(WATERMARK_DIR) not in sys.path:
        sys.path.insert(0, str(WATERMARK_DIR))
    spec = importlib.util.spec_from_file_location(f"catch22_{method}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load watermark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    MODULE_CACHE[method] = module
    return module


@dataclass
class GenerationOutput:
    text: str
    detection: Dict
    metadata: Dict
    avg_entropy_bits: Optional[float]
    avg_top1_prob: Optional[float]


class WatermarkExperimentAdapter:
    """Wrap one existing watermark class behind a common interface."""

    def __init__(
        self,
        method: str,
        model_name: str,
        seed: int = 1234,
        load_in_4bit: bool = False,
        noise_level: float = 0.18,
        hcw_method: str = "delta",
        method_kwargs: Optional[Dict] = None,
    ):
        if method not in SCRIPT_PATHS:
            raise ValueError(f"Unsupported method {method}. Available: {sorted(SCRIPT_PATHS)}")

        self.method = method
        self.model_name = model_name
        self.seed = seed
        self.load_in_4bit = load_in_4bit
        self.noise_level = noise_level
        self.hcw_method = hcw_method
        self.method_kwargs = dict(method_kwargs or {})
        if method == "pmark":
            self.method_kwargs = _apply_pmark_env_overrides(self.method_kwargs)

        seed_everything(seed)
        module = _load_module(method)
        if method == "vanilla":
            self.runner = module.Llama2VanillaLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **self.method_kwargs,
            )
        elif method == "kgw":
            self.runner = module.Llama2WatermarkedLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **self.method_kwargs,
            )
        elif method == "unigram":
            self.runner = module.Llama2UnigramLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **self.method_kwargs,
            )
        elif method == "dipmark":
            self.runner = module.Llama2DiPmarkLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **self.method_kwargs,
            )
        elif method == "hcw":
            hcw_kwargs = dict(self.method_kwargs)
            hcw_kwargs.setdefault("method", hcw_method)
            self.runner = module.Llama2HuLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **hcw_kwargs,
            )
        elif method == "hcw-v2":
            hcw_kwargs = dict(self.method_kwargs)
            hcw_kwargs.setdefault("method", hcw_method)
            hcw_kwargs.setdefault(
                "detection_config",
                module.HuDetectionConfig(
                    detector_type=os.environ.get("HCW_DETECTOR_TYPE", "llr"),
                    tv_distance=float(os.environ.get("HCW_TV_DISTANCE", "0.0")),
                ),
            )
            self.runner = module.Llama2HuV2LFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **hcw_kwargs,
            )
        elif method == "kuditipudi":
            self.runner = module.Llama2KuditipudiLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **self.method_kwargs,
            )
        elif method == "cgw":
            self.runner = module.Llama2ChristLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **self.method_kwargs,
            )
        elif method == "gaussmark":
            self.runner = module.Llama2GaussMarkLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **self.method_kwargs,
            )
        elif method == "heavywater":
            self.runner = module.Llama2HeavyWaterLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                detector_only=_env_bool("CATCH22_SCORE_DETECT_ONLY", False),
                **self.method_kwargs,
            )
        elif method == "pmark":
            self.runner = module.Llama2PMarkLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **self.method_kwargs,
            )
        elif method == "semstamp":
            self.runner = module.Llama2SemStampLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **self.method_kwargs,
            )
        elif method == "simmark":
            self.runner = module.Llama2SimMarkLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **self.method_kwargs,
            )
        elif method == "simplexwater":
            self.runner = module.Llama2SimplexWaterLFQA(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                detector_only=_env_bool("CATCH22_SCORE_DETECT_ONLY", False),
                **self.method_kwargs,
            )
        else:
            hybrid_kwargs = dict(self.method_kwargs)
            hybrid_kwargs.setdefault("noise_level", noise_level)
            self.runner = module.HybridWatermarkedLLM(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                **hybrid_kwargs,
            )

        self.tokenizer = self.runner.tokenizer
        self.model = getattr(self.runner, "model", None)
        self.device = getattr(self.runner, "device", "cuda" if torch.cuda.is_available() else "cpu")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        seed: Optional[int] = None,
    ) -> GenerationOutput:
        if seed is not None:
            seed_everything(seed)

        metadata: Dict[str, object] = {}
        skip_inline_detection = os.environ.get("REVIEWER2_SKIP_INLINE_DETECTION") == "1"
        skip_base_metrics = os.environ.get("REVIEWER2_SKIP_BASE_METRICS") == "1"
        if self.method == "vanilla":
            text = self.runner.generate_vanilla(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )
            detection = self.runner.compute_vanilla_stats(text)
        elif self.method == "kgw":
            text = self.runner.generate_watermarked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )
            detection = self.runner.detector.detect(text, self.tokenizer)
        elif self.method == "unigram":
            text = self.runner.generate_watermarked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )
            detection = self.runner.detector.detect(text, self.tokenizer)
        elif self.method == "dipmark":
            text, texture_keys = self.runner.generate_watermarked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                completion_token_ids=getattr(self.runner, "last_completion_token_ids", None),
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            metadata.update(getattr(self.runner, "last_generation_metadata", {}))
            metadata["texture_key_count"] = len(texture_keys)
        elif self.method == "hcw":
            text = self.runner.generate_watermarked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                completion_token_ids=getattr(self.runner, "last_completion_token_ids", None),
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            metadata.update(getattr(self.runner, "last_generation_metadata", {}))
        elif self.method == "hcw-v2":
            text, hcw_metadata = self.runner.generate_watermarked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            if skip_inline_detection:
                detection = {}
            else:
                detection = self.runner.detect_with_prompt(
                    prompt=prompt,
                    generated_text=text,
                    completion_token_ids=hcw_metadata.get("completion_token_ids"),
                    temperature=max(temperature, 1e-3),
                    top_p=top_p,
                    top_k=top_k,
            )
            metadata.update(hcw_metadata)
        elif self.method == "kuditipudi":
            text = self.runner.generate_watermarked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                completion_token_ids=getattr(self.runner, "last_completion_token_ids", None),
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            metadata.update(getattr(self.runner, "last_generation_metadata", {}))
        elif self.method == "cgw":
            text = self.runner.generate_watermarked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                completion_token_ids=getattr(self.runner, "last_completion_token_ids", None),
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            metadata.update(getattr(self.runner, "last_generation_metadata", {}))
        elif self.method == "gaussmark":
            key_seed = int(seed if seed is not None else self.seed)
            text = self.runner.generate_watermarked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                key_seed=key_seed,
                do_sample=True,
            )
            metadata.update(getattr(self.runner, "last_generation_metadata", {}))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                detection = self.runner.detect_with_prompt(
                    prompt=prompt,
                    generated_text=text,
                    generation_metadata=metadata,
                )
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                detection = {
                    "is_watermarked": False,
                    "score": 0.0,
                    "z_score": 0.0,
                    "test_statistic": 0.0,
                    "p_value": 1.0,
                    "alpha": float(self.runner.config.alpha),
                    "threshold": 0.0,
                    "num_tokens": len(self.tokenizer.encode(text, add_special_tokens=False)),
                    "gaussmark_key_seed": key_seed,
                    "error": "generation_detection_oom",
                }
        elif self.method in {"heavywater", "simplexwater"}:
            text = self.runner.generate_watermarked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                completion_token_ids=getattr(self.runner, "last_completion_token_ids", None),
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            metadata.update(getattr(self.runner, "last_generation_metadata", {}))
        elif self.method == "pmark":
            text = self.runner.generate_watermarked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            metadata.update(getattr(self.runner, "last_generation_metadata", {}))
            if skip_inline_detection:
                detection = {}
            else:
                detection = self.runner.detect_with_prompt(
                    prompt=prompt,
                    generated_text=text,
                    generation_metadata=metadata,
                )
        elif self.method == "semstamp":
            text = self.runner.generate_watermarked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            metadata.update(getattr(self.runner, "last_generation_metadata", {}))
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                generation_metadata=metadata,
            )
        elif self.method == "simmark":
            text = self.runner.generate_watermarked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
            metadata.update(getattr(self.runner, "last_generation_metadata", {}))
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                generation_metadata=metadata,
            )
        else:
            text, detection = self.runner.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                watermark=True,
                detect_watermark=not skip_inline_detection,
            )

        metadata.setdefault("temperature", float(temperature))
        metadata.setdefault("top_p", float(top_p))
        metadata.setdefault("top_k", int(top_k))
        completion_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        metadata["completion_token_count"] = len(completion_tokens)
        if skip_base_metrics:
            entropy_bits, top1_prob = None, None
        else:
            entropy_bits, top1_prob = self.compute_base_entropy_metrics(prompt, text)
        if self.method == "hcw-v2" or os.environ.get("REVIEWER2_EMPTY_CACHE") == "1":
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return GenerationOutput(
            text=text,
            detection=detection,
            metadata=metadata,
            avg_entropy_bits=entropy_bits,
            avg_top1_prob=top1_prob,
        )

    def detect_text(
        self,
        prompt: str,
        text: str,
        generation_metadata: Optional[Dict] = None,
    ) -> Dict:
        """Run the method verifier on arbitrary text, including attacked outputs."""
        generation_metadata = dict(generation_metadata or {})
        completion_token_ids = generation_metadata.get("completion_token_ids")
        if completion_token_ids is None:
            completion_token_ids = generation_metadata.get("last_completion_token_ids")
        temperature, top_p, top_k = _sampling_params_from_metadata(generation_metadata)
        if self.method == "vanilla":
            detection = self.runner.compute_vanilla_stats(text)
        elif self.method == "kgw":
            detection = self.runner.detector.detect(text, self.tokenizer)
        elif self.method == "unigram":
            detection = self.runner.detector.detect(text, self.tokenizer)
        elif self.method == "dipmark":
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                completion_token_ids=completion_token_ids,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
        elif self.method == "hcw":
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                completion_token_ids=completion_token_ids,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
        elif self.method == "hcw-v2":
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                completion_token_ids=completion_token_ids,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
        elif self.method == "kuditipudi":
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                completion_token_ids=completion_token_ids,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
        elif self.method == "cgw":
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                completion_token_ids=completion_token_ids,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
        elif self.method == "gaussmark":
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                generation_metadata=generation_metadata,
            )
        elif self.method in {"heavywater", "simplexwater"}:
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                completion_token_ids=completion_token_ids,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                top_k=top_k,
            )
        elif self.method == "pmark":
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                generation_metadata=generation_metadata,
            )
        elif self.method in {"semstamp", "simmark"}:
            detection = self.runner.detect_with_prompt(
                prompt=prompt,
                generated_text=text,
                generation_metadata=generation_metadata,
            )
        else:
            detection = self._detect_hybrid(text)

        output = dict(detection)
        output.setdefault("num_tokens", len(self.tokenizer.encode(text, add_special_tokens=False)))
        output["score"] = self._extract_score(output)
        output["detector_type"] = self.method
        return output

    def _detect_dipmark(self, prompt: str, text: str) -> Dict:
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        generated_ids = list(prompt_tokens)
        texture_keys = []
        for token in completion_tokens:
            context_tokens = generated_ids[-self.runner.watermark.context_window :]
            texture_key = self.runner.watermark._get_texture_key(context_tokens)
            texture_keys.append(texture_key)
            generated_ids.append(token)
        return self.runner.watermark.compute_detection_score(completion_tokens, texture_keys)

    def _detect_hcw(self, text: str) -> Dict:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        state = np.random.get_state()
        np.random.seed(self._stable_seed(self.method, text))
        try:
            return self.runner.watermark.compute_detection_score(tokens, method=self.runner.method)
        finally:
            np.random.set_state(state)

    def _detect_hybrid(self, text: str) -> Dict:
        state = np.random.get_state()
        np.random.seed(self._stable_seed(self.method, text))
        try:
            return self.runner.watermark.detect(text, self.tokenizer)
        finally:
            np.random.set_state(state)

    @staticmethod
    def _extract_score(detection: Dict) -> float:
        if "score" in detection:
            return float(detection["score"])
        if "z_score" in detection:
            return float(detection["z_score"])
        if "confidence" in detection:
            return float(detection["confidence"])
        return 0.0

    @staticmethod
    def _stable_seed(method: str, text: str) -> int:
        digest = hashlib.sha256(f"{method}\n{text}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big", signed=False)

    def compute_base_entropy_metrics(self, prompt: str, completion: str) -> tuple[Optional[float], Optional[float]]:
        completion = completion or ""
        if not completion.strip() or self.model is None:
            return None, None

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        full_inputs = self.tokenizer(
            prompt + completion,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        prompt_len = prompt_inputs.input_ids.shape[1]
        full_ids = full_inputs.input_ids.to(self.device)
        if full_ids.shape[1] <= prompt_len:
            return None, None

        with torch.no_grad():
            outputs = self.model(full_ids)
        logits = outputs.logits[0]

        entropies: list[float] = []
        top1_probs: list[float] = []
        for pos in range(prompt_len - 1, full_ids.shape[1] - 1):
            probs = torch.softmax(logits[pos].float(), dim=-1)
            entropy_bits = -(probs * torch.log2(probs.clamp_min(1e-12))).sum().item()
            entropies.append(entropy_bits)
            top1_probs.append(float(probs.max().item()))

        if not entropies:
            return None, None
        return float(np.mean(entropies)), float(np.mean(top1_probs))
