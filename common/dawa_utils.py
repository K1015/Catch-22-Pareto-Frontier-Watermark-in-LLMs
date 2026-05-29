from __future__ import annotations

import os
from pathlib import Path

import torch

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
from transformers import AutoModelForCausalLM, AutoTokenizer


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def resolve_torch_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def resolve_runtime_torch_dtype(name: str) -> torch.dtype:
    requested = resolve_torch_dtype(name)
    if torch.cuda.is_available():
        return requested
    if requested in {torch.float16, torch.bfloat16}:
        return torch.float32
    return requested


def infer_model_alias(model_name: str) -> str:
    if "models--" in model_name and "/snapshots/" in model_name:
        chunk = model_name.split("models--", 1)[1].split("/snapshots/", 1)[0]
        return chunk.replace("--", "-")
    if "/" in model_name:
        return model_name.replace("/", "-")
    return Path(model_name).name


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_model(model_name: str, torch_dtype_name: str = "float16"):
    torch_dtype = resolve_runtime_torch_dtype(torch_dtype_name)

    if "mistral" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"

    force_cuda = os.environ.get("CAUSAL_LM_FORCE_CUDA", "1").strip().lower() in {"1", "true", "yes"}
    if torch.cuda.is_available() and force_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
        )
    if model.get_input_embeddings().weight.shape[0] < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return model, tokenizer
