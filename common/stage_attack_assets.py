from __future__ import annotations

import argparse
import json
from pathlib import Path

import nltk


def snapshot_download(repo_id: str, cache_dir: str, allow_patterns: list[str] | None = None) -> str:
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for staging attack assets.") from exc
    return hf_snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        resume_download=True,
        allow_patterns=allow_patterns,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage attack models and NLTK assets on the cluster.")
    parser.add_argument("--hf-cache", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assets = {
        "facebook/opt-2.7b": snapshot_download("facebook/opt-2.7b", args.hf_cache),
        "facebook/bart-large-cnn": snapshot_download("facebook/bart-large-cnn", args.hf_cache),
        "kalpeshk2011/dipper-paraphraser-xxl": snapshot_download("kalpeshk2011/dipper-paraphraser-xxl", args.hf_cache),
        "Helsinki-NLP/opus-mt-en-fr": snapshot_download("Helsinki-NLP/opus-mt-en-fr", args.hf_cache),
        "Helsinki-NLP/opus-mt-fr-en": snapshot_download("Helsinki-NLP/opus-mt-fr-en", args.hf_cache),
        "sentence-transformers/all-mpnet-base-v2": snapshot_download("sentence-transformers/all-mpnet-base-v2", args.hf_cache),
        "hkunlp/instructor-large": snapshot_download("hkunlp/instructor-large", args.hf_cache),
        "google/t5-v1_1-xxl::tokenizer": snapshot_download(
            "google/t5-v1_1-xxl",
            args.hf_cache,
            allow_patterns=["*.json", "spiece.model", "*.txt"],
        ),
    }
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    summary = {"hf_cache": args.hf_cache, "assets": assets, "nltk": ["punkt", "wordnet", "omw-1.4"]}
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
