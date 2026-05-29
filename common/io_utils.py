from __future__ import annotations

import csv
import difflib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
TEXT_BOUND_METADATA_KEYS = {
    "completion_token_ids",
    "last_completion_token_ids",
    "pmark_sentences",
    "pmark_sentence_logs",
}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def append_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with open(path, "a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_csv_row(path: str | Path, row: dict) -> None:
    path = Path(path)
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def extract_prompt(sample: dict) -> str:
    return sample.get("prefix") or sample.get("question") or sample.get("prompt") or ""


def extract_generated_text(sample: dict) -> str:
    completions = sample.get("gen_completion", [])
    if isinstance(completions, list) and completions:
        return str(completions[0])
    if isinstance(completions, str):
        return completions
    return sample.get("text", "")


def extract_gold_completion(sample: dict) -> str:
    if "gold_completion" in sample:
        return str(sample["gold_completion"])
    targets = sample.get("targets")
    if isinstance(targets, list) and targets:
        return str(targets[0])
    if targets is not None:
        return str(targets)
    return ""


def shallow_copy_with_completion(sample: dict, attacked_text: str, attack_name: str, attack_stats: dict) -> dict:
    output = dict(sample)
    output["gen_completion"] = [attacked_text]
    if attack_name != "none" and isinstance(output.get("generation_metadata"), dict):
        metadata = dict(output["generation_metadata"])
        for key in TEXT_BOUND_METADATA_KEYS:
            metadata.pop(key, None)
        output["generation_metadata"] = metadata
    output["attack_stats"] = {
        "attack_name": attack_name,
        **attack_stats,
    }
    return output


def regex_tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def _tokenize_for_edit_stats(text: str, tokenizer=None) -> list:
    if tokenizer is not None:
        return tokenizer.encode(text, add_special_tokens=False)
    return regex_tokenize(text)


def levenshtein_distance(seq_a: Sequence, seq_b: Sequence) -> int:
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    previous = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        current = [i]
        for j, token_b in enumerate(seq_b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (token_a != token_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _edited_run_lengths(original_tokens: Sequence, attacked_tokens: Sequence) -> list[int]:
    if not original_tokens:
        return [len(attacked_tokens)] if attacked_tokens else []

    matcher = difflib.SequenceMatcher(a=original_tokens, b=attacked_tokens, autojunk=False)
    edited_flags = [False] * len(original_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i1 != i2:
            for idx in range(i1, i2):
                edited_flags[idx] = True
        elif j1 != j2:
            anchor = min(max(i1 - 1, 0), len(edited_flags) - 1)
            edited_flags[anchor] = True

    run_lengths: list[int] = []
    current_run = 0
    for edited in edited_flags:
        if edited:
            current_run += 1
        elif current_run:
            run_lengths.append(current_run)
            current_run = 0
    if current_run:
        run_lengths.append(current_run)
    return run_lengths


def compute_token_edit_stats(original: str, attacked: str, tokenizer=None) -> dict:
    original_tokens = _tokenize_for_edit_stats(original, tokenizer=tokenizer)
    attacked_tokens = _tokenize_for_edit_stats(attacked, tokenizer=tokenizer)
    distance = levenshtein_distance(original_tokens, attacked_tokens)
    normalizer = max(len(original_tokens), len(attacked_tokens), 1)
    run_lengths = _edited_run_lengths(original_tokens, attacked_tokens)
    return {
        "edit_distance": int(distance),
        "edit_normalizer": int(normalizer),
        "realized_edit_rate": float(distance / normalizer),
        "num_edited_runs": int(len(run_lengths)),
        "mean_edited_run_length": float(np.mean(run_lengths)) if run_lengths else 0.0,
        "max_edited_run_length": int(max(run_lengths)) if run_lengths else 0,
    }


def compute_token_edit_rate(original: str, attacked: str, tokenizer=None) -> tuple[int, int, float]:
    stats = compute_token_edit_stats(original, attacked, tokenizer=tokenizer)
    return stats["edit_distance"], stats["edit_normalizer"], stats["realized_edit_rate"]


def mean_confidence_interval(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, 0.0
    half_width = 1.96 * float(array.std(ddof=1)) / math.sqrt(len(array))
    return mean, half_width


def compute_auroc_tpr(positive_scores: Sequence[float], negative_scores: Sequence[float], target_fpr: float = 0.01) -> tuple[float, float]:
    labels = np.array([1] * len(positive_scores) + [0] * len(negative_scores), dtype=int)
    scores = np.array(list(positive_scores) + list(negative_scores), dtype=float)
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = tpr[fpr <= target_fpr]
    best_tpr = float(np.max(eligible)) if len(eligible) else 0.0
    return auc, best_tpr


def ensure_nltk_resource(name: str) -> None:
    import nltk

    resource_map = {
        "punkt": "tokenizers/punkt",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    try:
        nltk.data.find(resource_map.get(name, name))
    except LookupError:
        nltk.download(name, quiet=True)


def sent_tokenize(text: str) -> list[str]:
    import nltk

    ensure_nltk_resource("punkt")
    sentences = [segment.strip() for segment in nltk.sent_tokenize(text) if segment.strip()]
    return sentences if sentences else [text.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def build_unigram_calibration(rows: Sequence[dict], tokenizer) -> dict:
    counter: Counter[int] = Counter()
    total_tokens = 0
    per_text_scores: list[float] = []
    texts = [extract_generated_text(row) for row in rows]
    tokenized_texts = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    for token_ids in tokenized_texts:
        counter.update(token_ids)
        total_tokens += len(token_ids)

    vocab_size = max(tokenizer.vocab_size, 1)
    denominator = total_tokens + vocab_size

    def per_text_surprisal(token_ids: Sequence[int]) -> float:
        if not token_ids:
            return 0.0
        log_probs = [math.log((counter[token_id] + 1) / denominator) for token_id in token_ids]
        return -float(np.mean(log_probs))

    for token_ids in tokenized_texts:
        per_text_scores.append(per_text_surprisal(token_ids))

    std = float(np.std(per_text_scores, ddof=1)) if len(per_text_scores) > 1 else 1.0
    return {
        "token_counts": dict(counter),
        "vocab_size": vocab_size,
        "denominator": denominator,
        "null_mean": float(np.mean(per_text_scores)) if per_text_scores else 0.0,
        "null_std": std if std > 1e-8 else 1.0,
        "num_texts": len(per_text_scores),
    }


def compute_unigram_surprisal(text: str, tokenizer, calibration: dict) -> float:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return 0.0
    counter = calibration["token_counts"]
    denominator = calibration["denominator"]
    log_probs = [
        math.log((counter.get(token_id, counter.get(str(token_id), 0)) + 1) / denominator)
        for token_id in token_ids
    ]
    return -float(np.mean(log_probs))


def condition_level_z(scores: Sequence[float], null_mean: float, null_std: float) -> float:
    if not scores:
        return 0.0
    scale = null_std / math.sqrt(len(scores))
    if scale <= 1e-8:
        return 0.0
    return float((np.mean(scores) - null_mean) / scale)


def mean_standardized_z(scores: Sequence[float], null_mean: float, null_std: float) -> float:
    if not scores or null_std <= 1e-8:
        return 0.0
    return float((np.mean(scores) - null_mean) / null_std)


def chunked(iterable: Sequence, size: int) -> Iterable[Sequence]:
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]
