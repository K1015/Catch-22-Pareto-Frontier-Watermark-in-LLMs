"""
Llama 2 / CodeLlama wrapper for a local SimMark-style similarity watermark.

This is a transparent reconstruction based on the paper-facing operating point
used in the Catch-22 draft: sentence-level rejection sampling over a similarity
interval with keyed interval jitter and a soft-counting detector.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Optional

import numpy as np
import torch
from transformers import GenerationConfig, StoppingCriteriaList


THIS_DIR = Path(__file__).resolve().parent
SEMSTAMP_ROOT = THIS_DIR.parent / "external" / "SemStamp"
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if SEMSTAMP_ROOT.exists() and str(SEMSTAMP_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMSTAMP_ROOT))

from semantic_embedder_compat import load_semantic_embedder
from sampling_utils import SentenceEndCriteria, gen_sent
from watermark_rebuild_common import load_model_and_tokenizer


NORMAL_DIST = NormalDist()


def _split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", stripped) if segment.strip()]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


@dataclass(frozen=True)
class SimMarkConfig:
    embedder_path: str = "hkunlp/instructor-large"
    embedder_device: Optional[str] = None
    similarity_metric: str = "cosine"
    similarity_low: float = 0.68
    similarity_high: float = 0.76
    soft_k: float = 250.0
    interval_jitter: float = 0.02
    expected_accept_rate: float = 0.25
    repetition_penalty: float = 1.05
    max_trials: int = 250
    max_sentences: int = 12
    embedder_batch_size: int = 8
    alpha: float = 0.01
    secret_key: str = "simmark-secret-2025"


class Llama2SimMarkLFQA:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        embedder_path: str = "hkunlp/instructor-large",
        embedder_device: Optional[str] = None,
        similarity_metric: str = "cosine",
        similarity_low: float = 0.68,
        similarity_high: float = 0.76,
        soft_k: float = 250.0,
        interval_jitter: float = 0.02,
        expected_accept_rate: float = 0.25,
        repetition_penalty: float = 1.05,
        max_trials: int = 250,
        max_sentences: int = 12,
        embedder_batch_size: int = 8,
        alpha: float = 0.01,
        secret_key: str = "simmark-secret-2025",
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit if not load_in_8bit else False
        self.config = SimMarkConfig(
            embedder_path=embedder_path,
            embedder_device=embedder_device,
            similarity_metric=similarity_metric,
            similarity_low=float(similarity_low),
            similarity_high=float(similarity_high),
            soft_k=float(soft_k),
            interval_jitter=float(interval_jitter),
            expected_accept_rate=float(expected_accept_rate),
            repetition_penalty=float(repetition_penalty),
            max_trials=max(int(max_trials), 1),
            max_sentences=max(int(max_sentences), 1),
            embedder_batch_size=max(int(embedder_batch_size), 1),
            alpha=float(alpha),
            secret_key=secret_key,
        )

        self.tokenizer, self.model = load_model_and_tokenizer(
            self.model_name,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
        )
        self.model.eval()

        embedder_device_name = self.config.embedder_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = load_semantic_embedder(self.config.embedder_path, device=embedder_device_name)

        self.last_completion_token_ids: list[int] = []
        self.last_generation_metadata: Dict[str, object] = {}

        print("=" * 80)
        print("LLAMA 2 / CODELLAMA - SIMMARK")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(
            "SimMark parameters: "
            f"embedder={self.config.embedder_path}, "
            f"metric={self.config.similarity_metric}, "
            f"range=[{self.config.similarity_low:.2f}, {self.config.similarity_high:.2f}], "
            f"soft_k={self.config.soft_k}, "
            f"interval_jitter={self.config.interval_jitter}, "
            f"expected_accept_rate={self.config.expected_accept_rate}"
        )

    def _generate_next_sentence(
        self,
        rolling_text: str,
        remaining_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        text_ids = self.tokenizer.encode(rolling_text, return_tensors="pt").to(self.device)
        sent_end_criteria = SentenceEndCriteria(self.tokenizer)
        sent_end_criteria.update(rolling_text)
        gen_config = GenerationConfig(
            max_new_tokens=max(int(remaining_tokens), 1),
            min_new_tokens=0,
            do_sample=True,
            temperature=max(float(temperature), 1e-5),
            top_p=float(top_p),
            top_k=max(int(top_k), 0),
            repetition_penalty=float(self.config.repetition_penalty),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        stopping_criteria = StoppingCriteriaList([sent_end_criteria])
        new_text, _ = gen_sent(
            model=self.model,
            tokenizer=self.tokenizer,
            text_ids=text_ids,
            gen_config=gen_config,
            stopping_criteria=stopping_criteria,
        )
        return new_text.strip()

    def _embed_pair(self, first: str, second: str) -> tuple[np.ndarray, np.ndarray]:
        vectors = self.embedder.encode(
            [first, second],
            batch_size=self.config.embedder_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors[0], vectors[1]

    def _similarity(self, first: str, second: str) -> float:
        vec_a, vec_b = self._embed_pair(first, second)
        if self.config.similarity_metric == "euclidean":
            distance = float(np.linalg.norm(vec_a - vec_b))
            return float(1.0 / (1.0 + distance))
        denom = max(float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b)), 1e-12)
        return float(np.dot(vec_a, vec_b) / denom)

    def _seeded_interval(self, previous_sentence: str) -> tuple[float, float]:
        digest = hashlib.sha256(
            f"{self.config.secret_key}\n{previous_sentence}".encode("utf-8")
        ).digest()
        u = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64 - 1)
        offset = (2.0 * u - 1.0) * self.config.interval_jitter
        lower = max(-1.0, min(1.0, self.config.similarity_low + offset))
        upper = max(-1.0, min(1.0, self.config.similarity_high + offset))
        if lower > upper:
            lower, upper = upper, lower
        return float(lower), float(upper)

    def _soft_membership(self, similarity: float, lower: float, upper: float) -> float:
        return float(
            _sigmoid(self.config.soft_k * (similarity - lower))
            * _sigmoid(self.config.soft_k * (upper - similarity))
        )

    def generate_watermarked(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        rolling_text = prompt.strip()
        previous_sentence = prompt.strip()
        generated_sentences: list[str] = []
        sentence_logs: list[dict[str, object]] = []

        for sentence_index in range(1, self.config.max_sentences + 1):
            completion_text = " ".join(generated_sentences).strip()
            completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
            remaining_tokens = max_new_tokens - len(completion_ids)
            if remaining_tokens <= 0:
                break

            lower, upper = self._seeded_interval(previous_sentence)
            accepted_sentence = ""
            accepted_similarity = 0.0
            accepted_soft_score = 0.0
            last_candidate = ""
            last_similarity = 0.0
            last_soft_score = 0.0
            trial_count = 0
            used_fallback = False

            while trial_count < self.config.max_trials:
                candidate_sentence = self._generate_next_sentence(
                    rolling_text=rolling_text,
                    remaining_tokens=remaining_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                if not candidate_sentence:
                    break

                trial_count += 1
                similarity = self._similarity(previous_sentence, candidate_sentence)
                soft_score = self._soft_membership(similarity, lower, upper)
                last_candidate = candidate_sentence
                last_similarity = similarity
                last_soft_score = soft_score

                if lower <= similarity <= upper:
                    accepted_sentence = candidate_sentence
                    accepted_similarity = similarity
                    accepted_soft_score = soft_score
                    break

            if not accepted_sentence:
                if not last_candidate:
                    break
                accepted_sentence = last_candidate
                accepted_similarity = last_similarity
                accepted_soft_score = last_soft_score
                used_fallback = True

            generated_sentences.append(accepted_sentence)
            rolling_text = f"{rolling_text} {accepted_sentence}".strip()
            sentence_logs.append(
                {
                    "text": accepted_sentence,
                    "similarity": float(accepted_similarity),
                    "soft_score": float(accepted_soft_score),
                    "interval_low": float(lower),
                    "interval_high": float(upper),
                    "trials": int(max(trial_count, 1)),
                    "maxed_out": bool(used_fallback),
                }
            )
            previous_sentence = accepted_sentence

            completion_text = " ".join(generated_sentences).strip()
            completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
            if len(completion_ids) >= max_new_tokens:
                break

        final_text = " ".join(generated_sentences).strip()
        self.last_completion_token_ids = self.tokenizer.encode(final_text, add_special_tokens=False)
        self.last_generation_metadata = {
            "simmark_sentences": generated_sentences,
            "simmark_sentence_logs": sentence_logs,
            "embedder_path": self.config.embedder_path,
            "similarity_metric": self.config.similarity_metric,
            "similarity_low": float(self.config.similarity_low),
            "similarity_high": float(self.config.similarity_high),
            "soft_k": float(self.config.soft_k),
            "interval_jitter": float(self.config.interval_jitter),
            "expected_accept_rate": float(self.config.expected_accept_rate),
            "alpha": float(self.config.alpha),
        }
        return final_text

    def detect_with_prompt(
        self,
        prompt: str,
        generated_text: str,
        generation_metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        metadata = dict(generation_metadata or {})
        generated_sentences = metadata.get("simmark_sentences")
        if isinstance(generated_sentences, list) and generated_sentences:
            suffix_sentences = [str(sentence).strip() for sentence in generated_sentences if str(sentence).strip()]
        else:
            suffix_sentences = _split_sentences(generated_text)

        num_tokens = len(self.tokenizer.encode(generated_text, add_special_tokens=False))
        threshold = float(NORMAL_DIST.inv_cdf(1.0 - self.config.alpha))
        if not suffix_sentences:
            return {
                "is_watermarked": False,
                "score": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
                "threshold": threshold,
                "num_tokens": int(num_tokens),
                "num_sentences": 0,
                "mean_soft_score": 0.0,
            }

        previous_sentence = prompt.strip()
        soft_scores: list[float] = []
        sentence_stats: list[dict[str, float]] = []

        for sentence in suffix_sentences:
            lower, upper = self._seeded_interval(previous_sentence)
            similarity = self._similarity(previous_sentence, sentence)
            soft_score = self._soft_membership(similarity, lower, upper)
            soft_scores.append(soft_score)
            sentence_stats.append(
                {
                    "similarity": float(similarity),
                    "interval_low": float(lower),
                    "interval_high": float(upper),
                    "soft_score": float(soft_score),
                }
            )
            previous_sentence = sentence

        n = len(soft_scores)
        p0 = min(max(self.config.expected_accept_rate, 1e-6), 1.0 - 1e-6)
        denom = math.sqrt(max(n * p0 * (1.0 - p0), 1e-9))
        z_score = float((sum(soft_scores) - n * p0) / denom)
        p_value = float(1.0 - NORMAL_DIST.cdf(z_score))
        return {
            "is_watermarked": bool(z_score >= threshold),
            "score": z_score,
            "z_score": z_score,
            "p_value": p_value,
            "threshold": threshold,
            "num_tokens": int(num_tokens),
            "num_sentences": int(n),
            "mean_soft_score": float(np.mean(soft_scores)) if soft_scores else 0.0,
            "sentence_stats": sentence_stats,
        }

