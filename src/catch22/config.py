from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .registry import METHODS


@dataclass(frozen=True)
class ExperimentConfig:
    root: Path
    track: str
    model_name: str
    dataset_path: Path
    output_dir: Path
    methods: list[str]
    attacks: list[dict[str, Any]]
    num_samples: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    load_in_4bit: bool
    table_json: Path
    table_tex: Path


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    root = config_path.parents[1]
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    methods = list(data["methods"])
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods in {config_path}: {unknown}")
    table = data.get("table", {})
    generation = data.get("generation", {})
    model = data.get("model", {})
    return ExperimentConfig(
        root=root,
        track=str(data["track"]),
        model_name=str(model["name"]),
        dataset_path=_resolve(root, data["dataset_path"]),
        output_dir=_resolve(root, data["output_dir"]),
        methods=methods,
        attacks=list(data["attacks"]),
        num_samples=int(data.get("num_samples", 500)),
        max_new_tokens=int(generation.get("max_new_tokens", 300)),
        temperature=float(generation.get("temperature", 0.8)),
        top_p=float(generation.get("top_p", 0.95)),
        top_k=int(generation.get("top_k", 50)),
        load_in_4bit=bool(model.get("load_in_4bit", True)),
        table_json=_resolve(root, table.get("json", f"outputs/{data['track']}/tables/results.json")),
        table_tex=_resolve(root, table.get("tex", f"outputs/{data['track']}/tables/results.tex")),
    )


def validate_config(config: ExperimentConfig) -> list[str]:
    from .io import display_path

    errors: list[str] = []
    if not config.dataset_path.exists():
        errors.append(f"dataset_path does not exist: {display_path(config.dataset_path, config.root)}")
    if config.num_samples <= 0:
        errors.append("num_samples must be positive")
    for attack in config.attacks:
        if "name" not in attack or "condition" not in attack:
            errors.append(f"attack entries require name and condition: {attack}")
    return errors
