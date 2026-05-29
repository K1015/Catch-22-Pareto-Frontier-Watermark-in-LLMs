from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.attack_registry import get_attack_spec
from common.method_registry import get_method_spec


@dataclass(frozen=True)
class TableSpec:
    label: str
    caption: str
    output_tex: str
    output_json: str


@dataclass(frozen=True)
class ExperimentManifest:
    manifest_path: Path
    track_name: str
    description: str
    dataset_path: str | None
    model_name: str
    results_subdir: str
    methods: list[str]
    attacks: list[str]
    num_samples: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    load_in_4bit: bool = True
    preflight_num_samples: int = 5
    table: TableSpec | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _validate_manifest_data(path: Path, data: dict[str, Any]) -> None:
    for method in data["methods"]:
        get_method_spec(method)
    for attack in data["attacks"]:
        get_attack_spec(attack)

    dataset_path = data.get("dataset_path")
    if dataset_path:
        candidate = (path.parent / dataset_path).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Manifest dataset path does not exist: {candidate}")


def load_manifest(path: str | Path) -> ExperimentManifest:
    manifest_path = Path(path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_data(manifest_path, data)

    table_spec = None
    if "table" in data and data["table"]:
        table = data["table"]
        table_spec = TableSpec(
            label=str(table["label"]),
            caption=str(table["caption"]),
            output_tex=str(table["output_tex"]),
            output_json=str(table["output_json"]),
        )

    return ExperimentManifest(
        manifest_path=manifest_path,
        track_name=str(data["track_name"]),
        description=str(data["description"]),
        dataset_path=data.get("dataset_path"),
        model_name=str(data["model_name"]),
        results_subdir=str(data["results_subdir"]),
        methods=list(data["methods"]),
        attacks=list(data["attacks"]),
        num_samples=int(data["num_samples"]),
        max_new_tokens=int(data["max_new_tokens"]),
        temperature=float(data.get("temperature", 0.8)),
        top_p=float(data.get("top_p", 0.95)),
        top_k=int(data.get("top_k", 50)),
        load_in_4bit=bool(data.get("load_in_4bit", True)),
        preflight_num_samples=int(data.get("preflight_num_samples", 5)),
        table=table_spec,
        extras=dict(data.get("extras", {})),
    )


def manifest_dataset_path(manifest: ExperimentManifest) -> Path | None:
    if manifest.dataset_path is None:
        return None
    return (manifest.manifest_path.parent / manifest.dataset_path).resolve()


def manifest_results_root(repo_root: Path, manifest: ExperimentManifest) -> Path:
    return (repo_root / "results" / manifest.results_subdir).resolve()

