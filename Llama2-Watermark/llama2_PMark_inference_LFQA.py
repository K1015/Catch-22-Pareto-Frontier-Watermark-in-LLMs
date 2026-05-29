"""
Llama 2 / CodeLlama wrapper for PMark semantic watermarking.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch


THIS_DIR = Path(__file__).resolve().parent
PMARK_ROOT = THIS_DIR.parent / "external" / "PMark"
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if PMARK_ROOT.exists() and str(PMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(PMARK_ROOT))

from semantic_embedder_compat import load_semantic_embedder
from watermark_rebuild_common import load_model_and_tokenizer
from utils.detect import detect_paragraph
from utils.rand import secret_mbit
from utils.sample import sample_next_sentence_msignal, sent_tokenize as pmark_sent_tokenize


def _split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        sentences = [segment.strip() for segment in pmark_sent_tokenize(stripped) if segment.strip()]
        if sentences:
            return sentences
    except Exception:
        pass
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", stripped) if segment.strip()]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class PMarkConfig:
    embedder_path: str = "sentence-transformers/all-mpnet-base-v2"
    embedder_device: Optional[str] = None
    embedder_backend: str = "sentence_transformers"
    num_samples: int = 64
    msig: int = 4
    median_method: str = "hd"
    pivot: str = "rand"
    dedup: bool = False
    parallel: bool = False
    min_sequences: int = 10
    max_sentences: int = 12


class Llama2PMarkLFQA:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        embedder_path: str = "sentence-transformers/all-mpnet-base-v2",
        embedder_device: Optional[str] = None,
        embedder_backend: str = "sentence_transformers",
        num_samples: int = 64,
        msig: int = 4,
        median_method: str = "hd",
        pivot: str = "rand",
        dedup: bool = False,
        parallel: bool = False,
        min_sequences: int = 10,
        max_sentences: int = 12,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit if not load_in_8bit else False
        self.config = PMarkConfig(
            embedder_path=embedder_path,
            embedder_device=embedder_device,
            embedder_backend=embedder_backend,
            num_samples=num_samples,
            msig=max(int(msig), 1),
            median_method=median_method,
            pivot=pivot,
            dedup=dedup,
            parallel=parallel,
            min_sequences=min_sequences,
            max_sentences=max_sentences,
        )

        self.tokenizer, self.model = load_model_and_tokenizer(
            self.model_name,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
        )
        self.model.eval()

        embedder_device_name = self.config.embedder_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = load_semantic_embedder(
            self.config.embedder_path,
            device=embedder_device_name,
            backend=self.config.embedder_backend,
        )
        secret_mbit.set_signum(self.config.msig)

        self.last_completion_token_ids: list[int] = []
        self.last_generation_metadata: Dict[str, object] = {}

        print("=" * 80)
        print("LLAMA 2 / CODELLAMA - PMARK")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(
            "PMark parameters: "
            f"embedder={self.config.embedder_path}, "
            f"embedder_backend={self.config.embedder_backend}, "
            f"num_samples={self.config.num_samples}, "
            f"msig={self.config.msig}, "
            f"median_method={self.config.median_method}, "
            f"pivot={self.config.pivot}, "
            f"dedup={self.config.dedup}, "
            f"parallel={self.config.parallel}"
        )

    def generate_watermarked(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        del temperature, top_p, top_k

        secret_mbit.set_signum(self.config.msig)
        rolling_prompt = prompt.strip()
        generated_sentences: list[str] = []
        sentence_logs: list[dict] = []

        for sentence_id in range(1, self.config.max_sentences + 1):
            next_sample = sample_next_sentence_msignal(
                model=self.model,
                tokenizer=self.tokenizer,
                embedder=self.embedder,
                prompt=rolling_prompt,
                num_samples=self.config.num_samples,
                debug=False,
                model_name=self.model_name,
                openai_api_base=None,
                min_seqs=self.config.min_sequences,
                sen_id=sentence_id,
                median_method=self.config.median_method,
                llm=None,
                dedup=self.config.dedup,
                parallel=self.config.parallel,
            )
            if not next_sample:
                break

            next_sentence = str(next_sample["text"]).strip()
            if not next_sentence:
                break

            generated_sentences.append(next_sentence)
            sentence_logs.append(
                {
                    "text": next_sentence,
                    "rand_flag": next_sample.get("rand_flag"),
                    "select_num": next_sample.get("select_num"),
                    "total_num": next_sample.get("total_num"),
                }
            )
            rolling_prompt = f"{rolling_prompt} {next_sentence}".strip()

            completion_text = " ".join(generated_sentences).strip()
            completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
            if len(completion_ids) >= max_new_tokens:
                break

        final_text = " ".join(generated_sentences).strip()
        self.last_completion_token_ids = self.tokenizer.encode(final_text, add_special_tokens=False)
        self.last_generation_metadata = {
            "pmark_sentences": generated_sentences,
            "pmark_sentence_logs": sentence_logs,
            "embedder_path": self.config.embedder_path,
            "embedder_backend": self.config.embedder_backend,
            "num_samples": int(self.config.num_samples),
            "msig": int(self.config.msig),
            "median_method": self.config.median_method,
            "pivot": self.config.pivot,
            "dedup": bool(self.config.dedup),
            "parallel": bool(self.config.parallel),
        }
        return final_text

    def detect_with_prompt(
        self,
        prompt: str,
        generated_text: str,
        generation_metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        metadata = dict(generation_metadata or {})
        secret_mbit.set_signum(self.config.msig)

        generated_sentences = metadata.get("pmark_sentences")
        if isinstance(generated_sentences, list) and generated_sentences:
            metadata_sentences = [str(sentence).strip() for sentence in generated_sentences if str(sentence).strip()]
            metadata_text = " ".join(metadata_sentences)
            if _normalize_text(metadata_text) == _normalize_text(generated_text):
                suffix_sentences = metadata_sentences
            else:
                suffix_sentences = _split_sentences(generated_text)
        else:
            suffix_sentences = _split_sentences(generated_text)
        suffix_sentences = suffix_sentences[: max(int(self.config.max_sentences), 1)]

        num_tokens = len(self.tokenizer.encode(generated_text, add_special_tokens=False))
        if not suffix_sentences:
            return {
                "is_watermarked": False,
                "score": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
                "num_tokens": int(num_tokens),
                "num_sentences": 0,
            }

        debug_timing = os.environ.get("PMARK_DEBUG_TIMING") == "1"
        if debug_timing:
            print(
                "[pmark-debug] detect-start "
                f"prompt_chars={len(prompt)} text_chars={len(generated_text)} "
                f"num_tokens={num_tokens} suffix_sentences={len(suffix_sentences)} "
                f"num_samples={self.config.num_samples}",
                flush=True,
            )
        detect_start = time.time()
        detection = detect_paragraph(
            sentences=[prompt.strip()] + suffix_sentences,
            num_samples=self.config.num_samples,
            api_base=None,
            model_name=self.model_name,
            llm=None,
            embedder=self.embedder,
            pivot=self.config.pivot,
            median_method=self.config.median_method,
            debug=False,
            dedup=self.config.dedup,
            msig=self.config.msig,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        if debug_timing:
            print(f"[pmark-debug] detect-done elapsed={time.time() - detect_start:.1f}s", flush=True)
        output = dict(detection)
        output.setdefault("score", float(output.get("z_score", 0.0)))
        output["num_tokens"] = int(num_tokens)
        output["num_sentences"] = int(len(suffix_sentences))
        return output
