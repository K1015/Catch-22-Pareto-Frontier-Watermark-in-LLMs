from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from watermark_rebuild_common import load_tokenizer_compat


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


class HFMeanPoolingEmbedder:
    """Minimal sentence embedding wrapper compatible with the local PMark/SemStamp paths."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.tokenizer = load_tokenizer_compat(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self._dim = int(getattr(self.model.config, "hidden_size"))

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def _encode_batch(self, sentences: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.no_grad():
            if getattr(self.model.config, "is_encoder_decoder", False):
                encoder = self.model.get_encoder() if hasattr(self.model, "get_encoder") else getattr(self.model, "encoder")
                outputs = encoder(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens.get("attention_mask"),
                    return_dict=True,
                )
            else:
                outputs = self.model(**tokens)
        hidden = outputs.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.shape).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu()

    def encode(
        self,
        sentences: str | Iterable[str],
        batch_size: int = 32,
        convert_to_tensor: bool = False,
        convert_to_numpy: bool = False,
        show_progress_bar: bool = False,
        **kwargs,
    ):
        del show_progress_bar, kwargs

        single_input = isinstance(sentences, str)
        sentence_list = [sentences] if single_input else list(sentences)
        if not sentence_list:
            empty = torch.zeros((0, self._dim), dtype=torch.float32)
            if convert_to_tensor:
                return empty
            array = empty.numpy()
            return array[0] if single_input else array

        batches = []
        for start in range(0, len(sentence_list), max(int(batch_size), 1)):
            batches.append(self._encode_batch(sentence_list[start : start + max(int(batch_size), 1)]))
        result = torch.cat(batches, dim=0)

        if convert_to_tensor:
            return result

        array = result.numpy()
        if single_input:
            return array[0]
        if convert_to_numpy or not convert_to_tensor:
            return array
        return array


class HashingSemanticEmbedder:
    """Deterministic embedder for preflight checks."""

    def __init__(self, dimension: int = 768):
        self._dim = int(dimension)

    def to(self, device: str):
        del device
        return self

    def eval(self):
        return self

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def _embed_one(self, sentence: str) -> np.ndarray:
        digest = hashlib.sha256(sentence.encode("utf-8", errors="ignore")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self._dim).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def encode(
        self,
        sentences: str | Iterable[str],
        batch_size: int = 32,
        convert_to_tensor: bool = False,
        convert_to_numpy: bool = False,
        show_progress_bar: bool = False,
        **kwargs,
    ):
        del batch_size, show_progress_bar, kwargs

        single_input = isinstance(sentences, str)
        sentence_list = [sentences] if single_input else list(sentences)
        if sentence_list:
            array = np.stack([self._embed_one(str(sentence)) for sentence in sentence_list], axis=0)
        else:
            array = np.zeros((0, self._dim), dtype=np.float32)

        if convert_to_tensor:
            tensor = torch.from_numpy(array)
            return tensor[0] if single_input else tensor
        if single_input:
            return array[0]
        if convert_to_numpy or not convert_to_tensor:
            return array
        return array


def load_semantic_embedder(model_name: str, device: str = "cpu", backend: str = "sentence_transformers"):
    model_ref = resolve_cached_snapshot(model_name)
    if backend == "hash":
        return HashingSemanticEmbedder()
    if backend == "hf_mean":
        return HFMeanPoolingEmbedder(model_name=model_ref, device=device)
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_ref, device=device)
    except Exception:
        return HFMeanPoolingEmbedder(model_name=model_ref, device=device)


def resolve_cached_snapshot(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)
    if "/" not in model_name:
        return model_name

    cache_roots = []
    for env_name in ("TRANSFORMERS_CACHE", "HF_HUB_CACHE", "HF_HOME"):
        value = os.environ.get(env_name)
        if value:
            cache_roots.append(Path(value).expanduser())
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    repo_dir_name = "models--" + model_name.replace("/", "--")
    for cache_root in cache_roots:
        repo_dir = cache_root / repo_dir_name
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        ref_path = repo_dir / "refs" / "main"
        if ref_path.exists():
            snapshot = snapshots_dir / ref_path.read_text(encoding="utf-8").strip()
            if snapshot.exists():
                return str(snapshot)

        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if snapshots:
            newest = max(snapshots, key=lambda path: path.stat().st_mtime)
            return str(newest)

    return model_name
