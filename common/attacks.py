from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass

import torch

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from common.io_utils import chunked, ensure_nltk_resource, regex_tokenize, sent_tokenize
from common.dawa_utils import resolve_runtime_torch_dtype


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision(os.environ.get("TORCH_FLOAT32_MATMUL_PRECISION", "high"))
except Exception:
    pass


def _resolve_model_name(model_name: str) -> tuple[str, bool]:
    """Map broken HF-cache snapshot paths back to a repo id when needed."""
    if os.path.exists(model_name):
        return model_name, False

    match = re.search(r"models--([^/]+)--([^/]+)(?:/.*)?$", model_name)
    if match:
        org = match.group(1)
        repo = match.group(2)
        return f"{org}/{repo}", True

    return model_name, False


def _load_generation_tokenizer(model_name: str):
    resolved_name, local_files_only = _resolve_model_name(model_name)
    last_error: Exception | None = None
    for use_fast in (True, False):
        try:
            load_kwargs = {"use_fast": use_fast}
            if local_files_only:
                load_kwargs["local_files_only"] = True
            tokenizer = AutoTokenizer.from_pretrained(resolved_name, **load_kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            tokenizer.padding_side = "left"
            return tokenizer
        except Exception as exc:  # pragma: no cover - exercised on the cluster with model-specific tokenizers.
            last_error = exc
    raise RuntimeError(f"Unable to load tokenizer for {model_name!r} (resolved to {resolved_name!r})") from last_error


def _has_meta_tensors(model) -> bool:
    return any(parameter.is_meta for parameter in model.parameters())


def _device_map_input_device(model) -> torch.device | None:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return None
    for device in device_map.values():
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        device_name = str(device)
        if device_name.startswith("cuda"):
            return torch.device(device_name)
        if device_name == "0":
            return torch.device("cuda:0")
    return None


def _load_seq2seq_model(model_name: str, torch_dtype: torch.dtype, *, prefer_device_map: bool = True):
    resolved_name, local_files_only = _resolve_model_name(model_name)
    base_kwargs = {"torch_dtype": torch_dtype}
    if local_files_only:
        base_kwargs["local_files_only"] = True
    force_cuda = os.environ.get("SEQ2SEQ_FORCE_CUDA", "").strip().lower() in {"1", "true", "yes"}
    if torch.cuda.is_available() and force_cuda:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            resolved_name,
            low_cpu_mem_usage=False,
            **base_kwargs,
        )
        return model.to("cuda")

    if torch.cuda.is_available() and prefer_device_map and not force_cuda:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                resolved_name,
                device_map="auto",
                low_cpu_mem_usage=True,
                max_memory={
                    0: os.environ["SEQ2SEQ_CUDA_MAX_MEMORY"],
                    "cpu": os.environ.get("SEQ2SEQ_CPU_MAX_MEMORY", "64GiB"),
                }
                if os.environ.get("SEQ2SEQ_CUDA_MAX_MEMORY")
                else None,
                **base_kwargs,
            )
            return model
        except (ValueError, RuntimeError, NotImplementedError):
            pass

    model = AutoModelForSeq2SeqLM.from_pretrained(
        resolved_name,
        low_cpu_mem_usage=False,
        **base_kwargs,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def _load_causal_lm_model(model_name: str, torch_dtype: torch.dtype):
    resolved_name, local_files_only = _resolve_model_name(model_name)
    base_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype,
    }
    if local_files_only:
        base_kwargs["local_files_only"] = True

    force_cuda = os.environ.get("CAUSAL_LM_FORCE_CUDA", "1").strip().lower() in {"1", "true", "yes"}
    if torch.cuda.is_available() and force_cuda:
        try:
            return AutoModelForCausalLM.from_pretrained(
                resolved_name,
                device_map={"": 0},
                **base_kwargs,
            )
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            strict = os.environ.get("CAUSAL_LM_STRICT_CUDA", "0").strip().lower() in {"1", "true", "yes"}
            if strict:
                raise RuntimeError(f"Failed to load {model_name!r} fully on CUDA") from exc
            print(f"Full CUDA load failed for {model_name}; falling back to device_map=auto: {exc}", file=sys.stderr)

    return AutoModelForCausalLM.from_pretrained(
        resolved_name,
        device_map="auto",
        **base_kwargs,
    )


def _module_device(model) -> torch.device:
    mapped_device = _device_map_input_device(model)
    if mapped_device is not None:
        return mapped_device
    for parameter in model.parameters():
        if not parameter.is_meta:
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class AttackResult:
    text: str
    metadata: dict


class IdentityAttack:
    def apply(self, prompt: str, text: str) -> AttackResult:
        return AttackResult(text=text, metadata={"attack_backend": "identity"})


class SynonymAttack:
    def __init__(self, seed: int = 1234, target_edit_rate: float = 0.15):
        ensure_nltk_resource("wordnet")
        ensure_nltk_resource("omw-1.4")
        from nltk.corpus import wordnet as wn

        self.wn = wn
        self.seed = seed
        self.target_edit_rate = target_edit_rate

    def _replacement(self, token: str) -> str | None:
        lowered = token.lower()
        if not lowered.isalpha() or len(lowered) < 4:
            return None
        for synset in self.wn.synsets(lowered):
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate.lower() != lowered and candidate.isalpha():
                    return candidate
        return None

    def apply(self, prompt: str, text: str) -> AttackResult:
        del prompt
        tokens = regex_tokenize(text)
        target_changes = max(1, int(math.ceil(len(tokens) * self.target_edit_rate)))
        indices = list(range(len(tokens)))
        rng = __import__("random").Random(self.seed + len(tokens))
        rng.shuffle(indices)

        changed = 0
        new_tokens = list(tokens)
        for idx in indices:
            replacement = self._replacement(tokens[idx])
            if replacement is None:
                continue
            new_tokens[idx] = replacement
            changed += 1
            if changed >= target_changes:
                break

        attacked = self._join_tokens(new_tokens)
        return AttackResult(
            text=attacked,
            metadata={
                "attack_backend": "nltk_wordnet",
                "requested_edit_rate": self.target_edit_rate,
                "synonym_changes": changed,
            },
        )

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        output = ""
        for token in tokens:
            if not output:
                output = token
            elif re.match(r"[^\w\s]", token):
                output += token
            else:
                output += " " + token
        return output


class SpanSynonymAttack(SynonymAttack):
    def __init__(
        self,
        seed: int = 1234,
        target_edit_rate: float = 0.15,
        min_span_len: int = 5,
        max_span_len: int = 10,
    ):
        super().__init__(seed=seed, target_edit_rate=target_edit_rate)
        if min_span_len <= 0:
            raise ValueError("min_span_len must be positive.")
        if max_span_len < min_span_len:
            raise ValueError("max_span_len must be >= min_span_len.")
        self.min_span_len = min_span_len
        self.max_span_len = max_span_len

    def apply(self, prompt: str, text: str) -> AttackResult:
        del prompt
        tokens = regex_tokenize(text)
        if not tokens:
            return AttackResult(
                text=text,
                metadata={
                    "attack_backend": "nltk_wordnet_span",
                    "requested_edit_rate": self.target_edit_rate,
                    "synonym_changes": 0,
                    "span_count": 0,
                    "min_span_len": self.min_span_len,
                    "max_span_len": self.max_span_len,
                },
            )

        target_changes = max(1, int(math.ceil(len(tokens) * self.target_edit_rate)))
        new_tokens = list(tokens)
        changed_indices: set[int] = set()
        covered_starts: set[int] = set()
        rng = __import__("random").Random(self.seed + len(tokens))

        span_count = 0
        attempts = 0
        max_attempts = max(len(tokens) * 8, 32)
        while len(changed_indices) < target_changes and attempts < max_attempts:
            attempts += 1
            start = rng.randrange(len(tokens))
            if start in covered_starts:
                continue
            covered_starts.add(start)
            span_len = rng.randint(self.min_span_len, self.max_span_len)
            end = min(len(tokens), start + span_len)

            span_changed = False
            for idx in range(start, end):
                if idx in changed_indices:
                    continue
                replacement = self._replacement(tokens[idx])
                if replacement is None:
                    continue
                new_tokens[idx] = replacement
                changed_indices.add(idx)
                span_changed = True
                if len(changed_indices) >= target_changes:
                    break
            if span_changed:
                span_count += 1

        attacked = self._join_tokens(new_tokens)
        return AttackResult(
            text=attacked,
            metadata={
                "attack_backend": "nltk_wordnet_span",
                "requested_edit_rate": self.target_edit_rate,
                "synonym_changes": len(changed_indices),
                "span_count": span_count,
                "min_span_len": self.min_span_len,
                "max_span_len": self.max_span_len,
            },
        )


class BackTranslationAttack:
    def __init__(self, forward_model: str, backward_model: str, torch_dtype_name: str = "float16"):
        torch_dtype = resolve_runtime_torch_dtype(torch_dtype_name)
        self.forward_tokenizer = _load_generation_tokenizer(forward_model)
        self.backward_tokenizer = _load_generation_tokenizer(backward_model)
        self.forward_model = _load_seq2seq_model(forward_model, torch_dtype)
        self.backward_model = _load_seq2seq_model(backward_model, torch_dtype)
        self.forward_model.eval()
        self.backward_model.eval()
        self.forward_device = _module_device(self.forward_model)
        self.backward_device = _module_device(self.backward_model)

    @torch.inference_mode()
    def apply(self, prompt: str, text: str) -> AttackResult:
        del prompt
        fr_text = self._translate(self.forward_model, self.forward_tokenizer, text)
        en_text = self._translate(self.backward_model, self.backward_tokenizer, fr_text)
        return AttackResult(
            text=en_text,
            metadata={
                "attack_backend": "marian_backtranslation",
                "intermediate_language": "fr",
            },
        )

    def _translate(self, model, tokenizer, text: str) -> str:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoded = {key: value.to(_module_device(model)) for key, value in encoded.items()}
        outputs = model.generate(
            **encoded,
            max_new_tokens=384,
            do_sample=False,
            num_beams=4,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


class SummarizationAttack:
    def __init__(self, model_name: str, torch_dtype_name: str = "float16"):
        torch_dtype = resolve_runtime_torch_dtype(torch_dtype_name)
        self.tokenizer = _load_generation_tokenizer(model_name)
        self.model = _load_seq2seq_model(model_name, torch_dtype, prefer_device_map=False)
        self.model.eval()
        self.device = _module_device(self.model)
        self.model_name = model_name

    @torch.inference_mode()
    def apply(self, prompt: str, text: str) -> AttackResult:
        return self.apply_batch([prompt], [text])[0]

    @torch.inference_mode()
    def apply_batch(self, prompts: list[str], texts: list[str]) -> list[AttackResult]:
        attack_inputs = [
            f"Question: {prompt.strip()}\n\nAnswer: {text.strip()}"
            for prompt, text in zip(prompts, texts)
        ]
        if not attack_inputs:
            return []
        max_source_tokens = max((len(regex_tokenize(text)) for text in texts), default=0)
        summaries: list[str] = []
        adaptive_batch_size = max(1, int(os.environ.get("SUMMARIZATION_GENERATION_BATCH_SIZE", str(len(attack_inputs)))))
        min_batch_size = max(1, int(os.environ.get("SUMMARIZATION_MIN_GENERATION_BATCH_SIZE", "1")))
        start = 0
        while start < len(attack_inputs):
            current_batch_size = min(adaptive_batch_size, len(attack_inputs) - start)
            while True:
                batch_inputs = attack_inputs[start : start + current_batch_size]
                encoded = self.tokenizer(
                    batch_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                )
                encoded = {key: value.to(self.device, non_blocking=(self.device.type == "cuda")) for key, value in encoded.items()}
                try:
                    outputs = self.model.generate(
                        **encoded,
                        max_new_tokens=min(256, max(64, int(max_source_tokens * 0.7))),
                        min_new_tokens=32,
                        do_sample=False,
                        num_beams=4,
                        length_penalty=1.0,
                        no_repeat_ngram_size=3,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    if current_batch_size <= min_batch_size:
                        raise
                    del encoded
                    current_batch_size = max(min_batch_size, current_batch_size // 2)
                    adaptive_batch_size = current_batch_size
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            summaries.extend(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))
            start += current_batch_size
        return [
            AttackResult(
                text=summary.strip() or text,
                metadata={
                    "attack_backend": "bart_cnn_summarization",
                    "summarization_model": self.model_name,
                    "attack_batch_size": len(attack_inputs),
                },
            )
            for summary, text in zip(summaries, texts)
        ]


class CausalLMParaphraseAttack:
    def __init__(self, model_name: str, instruction_template: str, torch_dtype_name: str = "float16"):
        torch_dtype = resolve_runtime_torch_dtype(torch_dtype_name)
        self.tokenizer = _load_generation_tokenizer(model_name)
        self.model = _load_causal_lm_model(model_name, torch_dtype)
        self.model.eval()
        self.instruction_template = instruction_template
        self.device = _module_device(self.model)

    @torch.inference_mode()
    def apply(self, prompt: str, text: str) -> AttackResult:
        return self.apply_batch([prompt], [text])[0]

    @torch.inference_mode()
    def apply_batch(self, prompts: list[str], texts: list[str]) -> list[AttackResult]:
        attack_prompts = [
            self.instruction_template.format(prompt=prompt.strip(), text=text.strip())
            for prompt, text in zip(prompts, texts)
        ]
        if not attack_prompts:
            return []
        max_source_tokens = max((len(regex_tokenize(text)) for text in texts), default=0)
        decoded_texts: list[str] = []
        adaptive_batch_size = max(1, int(os.environ.get("CAUSAL_LM_GENERATION_BATCH_SIZE", str(len(attack_prompts)))))
        min_batch_size = max(1, int(os.environ.get("CAUSAL_LM_MIN_GENERATION_BATCH_SIZE", "1")))
        start = 0
        while start < len(attack_prompts):
            current_batch_size = min(adaptive_batch_size, len(attack_prompts) - start)
            while True:
                batch_prompts = attack_prompts[start : start + current_batch_size]
                encoded = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                )
                encoded = {key: value.to(self.device, non_blocking=(self.device.type == "cuda")) for key, value in encoded.items()}
                input_len = encoded["input_ids"].shape[1]
                try:
                    outputs = self.model.generate(
                        **encoded,
                        max_new_tokens=min(512, max(128, int(max_source_tokens * 1.5))),
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    if current_batch_size <= min_batch_size:
                        raise
                    del encoded
                    current_batch_size = max(min_batch_size, current_batch_size // 2)
                    adaptive_batch_size = current_batch_size
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            decoded_texts.extend(self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True))
            start += current_batch_size
        return [
            AttackResult(
                text=decoded.strip() or text,
                metadata={
                    "attack_backend": "causal_lm_prompt",
                    "attack_batch_size": len(attack_prompts),
                },
            )
            for decoded, text in zip(decoded_texts, texts)
        ]


class DipperAttack:
    def __init__(
        self,
        model_name: str,
        lexical_diversity: int = 40,
        order_diversity: int = 0,
        sent_interval: int = 3,
        torch_dtype_name: str = "float16",
    ):
        assert lexical_diversity in {0, 20, 40, 60, 80, 100}
        assert order_diversity in {0, 20, 40, 60, 80, 100}
        torch_dtype = resolve_runtime_torch_dtype(torch_dtype_name)
        try:
            self.tokenizer = _load_generation_tokenizer(model_name)
        except RuntimeError:
            fallback_tokenizer = os.environ.get("DIPPER_TOKENIZER_MODEL", "google/t5-v1_1-xxl")
            self.tokenizer = _load_generation_tokenizer(fallback_tokenizer)
        self.model = _load_seq2seq_model(model_name, torch_dtype)
        self.model.eval()
        self.lexical_diversity = lexical_diversity
        self.order_diversity = order_diversity
        self.sent_interval = sent_interval
        force_cuda = os.environ.get("SEQ2SEQ_FORCE_CUDA", "").strip().lower() in {"1", "true", "yes"}
        default_generation_batch_size = "8" if force_cuda else "2"
        self.generation_batch_size = max(1, int(os.environ.get("DIPPER_GENERATION_BATCH_SIZE", default_generation_batch_size)))
        self.min_generation_batch_size = max(1, int(os.environ.get("DIPPER_MIN_GENERATION_BATCH_SIZE", "1")))
        self.max_new_tokens = max(1, int(os.environ.get("DIPPER_MAX_NEW_TOKENS", "256")))
        self.pad_to_multiple_of = max(0, int(os.environ.get("DIPPER_PAD_TO_MULTIPLE_OF", "8" if torch.cuda.is_available() else "0")))
        self.device = _module_device(self.model)

    @torch.inference_mode()
    def apply(self, prompt: str, text: str) -> AttackResult:
        return self.apply_batch([prompt], [text])[0]

    @torch.inference_mode()
    def apply_batch(self, prompts: list[str], texts: list[str]) -> list[AttackResult]:
        lex_code = 100 - self.lexical_diversity
        order_code = 100 - self.order_diversity

        model_inputs: list[str] = []
        owners: list[int] = []
        for idx, (prompt, text) in enumerate(zip(prompts, texts)):
            prompt_prefix = " ".join(prompt.split())
            sentences = sent_tokenize(text)
            for sentence_window in chunked(sentences, self.sent_interval):
                current = " ".join(sentence_window)
                model_inputs.append(f"lexical = {lex_code}, order = {order_code} {prompt_prefix} <sent> {current} </sent>")
                owners.append(idx)

        attacked_chunks: list[list[str]] = [[] for _ in texts]
        generation_batch_sizes_used: set[int] = set()
        adaptive_generation_batch_size = self.generation_batch_size
        start = 0
        while start < len(model_inputs):
            current_batch_size = min(adaptive_generation_batch_size, len(model_inputs) - start)
            while True:
                batch_inputs = model_inputs[start : start + current_batch_size]
                batch_owners = owners[start : start + current_batch_size]
                tokenizer_kwargs = {
                    "return_tensors": "pt",
                    "padding": True,
                    "truncation": True,
                    "max_length": 1024,
                }
                if self.pad_to_multiple_of:
                    tokenizer_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
                encoded = self.tokenizer(batch_inputs, **tokenizer_kwargs)
                try:
                    encoded = {
                        key: value.to(self.device, non_blocking=(self.device.type == "cuda"))
                        for key, value in encoded.items()
                    }
                    outputs = self.model.generate(
                        **encoded,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        top_p=0.75,
                        temperature=1.0,
                    )
                    generation_batch_sizes_used.add(current_batch_size)
                    break
                except torch.cuda.OutOfMemoryError:
                    if current_batch_size <= self.min_generation_batch_size:
                        raise
                    if "encoded" in locals():
                        del encoded
                    current_batch_size = max(self.min_generation_batch_size, current_batch_size // 2)
                    adaptive_generation_batch_size = current_batch_size
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            decoded_chunks = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for owner, decoded in zip(batch_owners, decoded_chunks):
                attacked_chunks[owner].append(decoded.strip())
            start += current_batch_size

        return [
            AttackResult(
                text=" ".join(chunk for chunk in chunks if chunk).strip() or text,
                metadata={
                    "attack_backend": "dipper",
                    "lexical_diversity": self.lexical_diversity,
                    "order_diversity": self.order_diversity,
                    "sent_interval": self.sent_interval,
                    "attack_batch_size": len(texts),
                    "generation_batch_size": self.generation_batch_size,
                    "generation_batch_sizes_used": sorted(generation_batch_sizes_used),
                    "min_generation_batch_size": self.min_generation_batch_size,
                    "max_new_tokens": self.max_new_tokens,
                    "num_dipper_chunks": len(chunks),
                },
            )
            for text, chunks in zip(texts, attacked_chunks)
        ]
