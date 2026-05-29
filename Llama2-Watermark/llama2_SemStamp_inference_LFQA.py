"""
Llama 2 / CodeLlama wrapper for SemStamp semantic watermarking.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, Optional

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


HASH_KEY = 15485863
NORMAL_DIST = NormalDist()


def _split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+", stripped)
        if segment.strip()
    ]


def _get_mask_from_seed(lsh_dim: int, accept_rate: float, seed: int) -> set[int]:
    n_bins = 2**lsh_dim
    n_accept = int(n_bins * float(accept_rate))
    if accept_rate > 0 and n_accept == 0:
        n_accept = 1
    n_accept = min(max(n_accept, 0), n_bins)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(HASH_KEY * int(seed))
    permutation = torch.randperm(n_bins, generator=generator)
    return set(int(value) for value in permutation[:n_accept].tolist())


class _ProjectionHasher:
    def __init__(self, projection_count: int, rand_seed: int = 1234):
        self.projection_count = int(projection_count)
        self.rand_seed = int(rand_seed)
        self.normals: Optional[np.ndarray] = None

    def reset(self, dim: int) -> None:
        rng = np.random.default_rng(self.rand_seed)
        normals = rng.standard_normal((self.projection_count, dim)).astype(np.float32)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        self.normals = normals / norms

    def hash_vector(self, vector: np.ndarray) -> str:
        vector = np.asarray(vector, dtype=np.float32)
        if self.normals is None:
            self.reset(vector.shape[-1])
        projections = self.normals @ vector
        return "".join("1" if value >= 0 else "0" for value in projections.tolist())


class SemStampLSHModel:
    def __init__(self, embedder, lsh_dim: int, batch_size: int = 32):
        self.embedder = embedder
        self.batch_size = int(batch_size)
        self.lsh_dim = int(lsh_dim)
        self.hasher = _ProjectionHasher(self.lsh_dim)
        dimension = int(self.embedder.get_sentence_embedding_dimension())
        self.hasher.reset(dimension)

    def get_embeddings(self, sents: Iterable[str]) -> np.ndarray:
        sentences = list(sents)
        if not sentences:
            return np.zeros((0, 0), dtype=np.float32)
        embeddings = self.embedder.encode(
            sentences,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def get_hash(self, sents: Iterable[str]) -> list[int]:
        embeddings = self.get_embeddings(sents)
        return [int(self.hasher.hash_vector(embedding), 2) for embedding in embeddings]


def _passes_margin(lsh_model: SemStampLSHModel, sentence: str, margin: float) -> bool:
    if margin <= 0.0:
        return True
    embeddings = lsh_model.get_embeddings([sentence])
    if embeddings.size == 0:
        return False
    normals = np.asarray(lsh_model.hasher.normals, dtype=np.float32)
    if normals.size == 0:
        return True
    embed = embeddings[0]
    embed_norm = float(np.linalg.norm(embed))
    if embed_norm <= 1e-12:
        return False
    normal_norms = np.linalg.norm(normals, axis=1)
    denom = np.clip(embed_norm * normal_norms, 1e-12, None)
    cosine_sims = (normals @ embed) / denom
    min_abs_similarity = float(np.min(np.abs(cosine_sims)))
    return min_abs_similarity >= float(margin)


@dataclass(frozen=True)
class SemStampConfig:
    embedder_path: str = "sentence-transformers/all-mpnet-base-v2"
    embedder_device: Optional[str] = None
    lsh_dim: int = 3
    lmbd: float = 0.25
    margin: float = 0.02
    repetition_penalty: float = 1.05
    max_trials: int = 100
    max_sentences: int = 12
    embedder_batch_size: int = 32
    alpha: float = 0.01


class Llama2SemStampLFQA:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        embedder_path: str = "sentence-transformers/all-mpnet-base-v2",
        embedder_device: Optional[str] = None,
        lsh_dim: int = 3,
        lmbd: float = 0.25,
        margin: float = 0.02,
        repetition_penalty: float = 1.05,
        max_trials: int = 100,
        max_sentences: int = 12,
        embedder_batch_size: int = 32,
        alpha: float = 0.01,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit if not load_in_8bit else False
        self.config = SemStampConfig(
            embedder_path=embedder_path,
            embedder_device=embedder_device,
            lsh_dim=int(lsh_dim),
            lmbd=float(lmbd),
            margin=float(margin),
            repetition_penalty=float(repetition_penalty),
            max_trials=max(int(max_trials), 1),
            max_sentences=max(int(max_sentences), 1),
            embedder_batch_size=max(int(embedder_batch_size), 1),
            alpha=float(alpha),
        )

        self.tokenizer, self.model = load_model_and_tokenizer(
            self.model_name,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
        )
        self.model.eval()

        embedder_device_name = self.config.embedder_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embedder = load_semantic_embedder(self.config.embedder_path, device=embedder_device_name)
        self.lsh_model = SemStampLSHModel(
            self.embedder,
            lsh_dim=self.config.lsh_dim,
            batch_size=self.config.embedder_batch_size,
        )

        self.last_completion_token_ids: list[int] = []
        self.last_generation_metadata: Dict[str, object] = {}

        print("=" * 80)
        print("LLAMA 2 / CODELLAMA - SEMSTAMP")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(
            "SemStamp parameters: "
            f"embedder={self.config.embedder_path}, "
            f"lsh_dim={self.config.lsh_dim}, "
            f"lambda={self.config.lmbd}, "
            f"margin={self.config.margin}, "
            f"repetition_penalty={self.config.repetition_penalty}, "
            f"max_trials={self.config.max_trials}"
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

    def generate_watermarked(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        rolling_text = prompt.strip()
        generated_sentences: list[str] = []
        sentence_logs: list[dict[str, object]] = []

        current_seed = self.lsh_model.get_hash([rolling_text])[0]
        total_trials = 0
        maxedout_trials = 0

        for sentence_index in range(1, self.config.max_sentences + 1):
            completion_text = " ".join(generated_sentences).strip()
            completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
            remaining_tokens = max_new_tokens - len(completion_ids)
            if remaining_tokens <= 0:
                break

            accept_mask = _get_mask_from_seed(self.config.lsh_dim, self.config.lmbd, current_seed)
            accepted_sentence = ""
            accepted_hash: Optional[int] = None
            last_candidate = ""
            last_candidate_hash: Optional[int] = None
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

                total_trials += 1
                trial_count += 1
                candidate_hash = self.lsh_model.get_hash([candidate_sentence])[0]
                last_candidate = candidate_sentence
                last_candidate_hash = candidate_hash

                if not _passes_margin(self.lsh_model, candidate_sentence, self.config.margin):
                    continue
                if candidate_hash not in accept_mask:
                    continue

                accepted_sentence = candidate_sentence
                accepted_hash = candidate_hash
                break

            if not accepted_sentence:
                if not last_candidate:
                    break
                accepted_sentence = last_candidate
                accepted_hash = last_candidate_hash
                used_fallback = True
                maxedout_trials += 1

            generated_sentences.append(accepted_sentence)
            current_seed = int(accepted_hash) if accepted_hash is not None else current_seed
            rolling_text = f"{rolling_text} {accepted_sentence}".strip()
            sentence_logs.append(
                {
                    "text": accepted_sentence,
                    "hash": int(accepted_hash) if accepted_hash is not None else None,
                    "trials": int(max(trial_count, 1)),
                    "maxed_out": bool(used_fallback),
                }
            )

            completion_text = " ".join(generated_sentences).strip()
            completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
            if len(completion_ids) >= max_new_tokens:
                break

        final_text = " ".join(generated_sentences).strip()
        self.last_completion_token_ids = self.tokenizer.encode(final_text, add_special_tokens=False)
        self.last_generation_metadata = {
            "semstamp_sentences": generated_sentences,
            "semstamp_sentence_logs": sentence_logs,
            "semstamp_total_trials": int(total_trials),
            "semstamp_maxedout_trials": int(maxedout_trials),
            "embedder_path": self.config.embedder_path,
            "lsh_dim": int(self.config.lsh_dim),
            "lmbd": float(self.config.lmbd),
            "margin": float(self.config.margin),
            "repetition_penalty": float(self.config.repetition_penalty),
            "max_trials": int(self.config.max_trials),
            "alpha": float(self.config.alpha),
        }
        return final_text

    def _detect_lsh(self, sentences: list[str]) -> float:
        if len(sentences) <= 1:
            return 0.0

        n_watermark = 0
        current_seed = self.lsh_model.get_hash([sentences[0]])[0]
        accept_mask = _get_mask_from_seed(self.config.lsh_dim, self.config.lmbd, current_seed)

        for sentence in sentences[1:]:
            candidate_hash = self.lsh_model.get_hash([sentence])[0]
            if candidate_hash in accept_mask:
                n_watermark += 1
            current_seed = candidate_hash
            accept_mask = _get_mask_from_seed(self.config.lsh_dim, self.config.lmbd, current_seed)

        n_test_sent = len(sentences) - 1
        denom = math.sqrt(n_test_sent * self.config.lmbd * (1.0 - self.config.lmbd))
        if denom <= 1e-12:
            return 0.0
        return float((n_watermark - self.config.lmbd * n_test_sent) / denom)

    def detect_with_prompt(
        self,
        prompt: str,
        generated_text: str,
        generation_metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        metadata = dict(generation_metadata or {})
        generated_sentences = metadata.get("semstamp_sentences")
        if isinstance(generated_sentences, list) and generated_sentences:
            suffix_sentences = [
                str(sentence).strip()
                for sentence in generated_sentences
                if str(sentence).strip()
            ]
        else:
            suffix_sentences = _split_sentences(generated_text)

        num_tokens = len(self.tokenizer.encode(generated_text, add_special_tokens=False))
        if not suffix_sentences:
            return {
                "is_watermarked": False,
                "score": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
                "threshold": float(NORMAL_DIST.inv_cdf(1.0 - self.config.alpha)),
                "num_tokens": int(num_tokens),
                "num_sentences": 0,
            }

        z_score = self._detect_lsh([prompt.strip()] + suffix_sentences)
        p_value = float(1.0 - NORMAL_DIST.cdf(z_score))
        threshold = float(NORMAL_DIST.inv_cdf(1.0 - self.config.alpha))
        return {
            "is_watermarked": bool(z_score >= threshold),
            "score": float(z_score),
            "z_score": float(z_score),
            "p_value": p_value,
            "threshold": threshold,
            "num_tokens": int(num_tokens),
            "num_sentences": int(len(suffix_sentences)),
        }
