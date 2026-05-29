from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttackSpec:
    name: str
    partition: str
    nominal_edit_rate: float
    uses_attack_model: bool = False
    uses_translation_models: bool = False
    attack_model: str | None = None
    description: str = ""


ATTACK_SPECS: dict[str, AttackSpec] = {
    "none": AttackSpec(
        name="none",
        partition="short",
        nominal_edit_rate=0.0,
        description="Identity/no-attack baseline.",
    ),
    "dipper": AttackSpec(
        name="dipper",
        partition="gpu",
        nominal_edit_rate=0.25,
        uses_attack_model=True,
        attack_model="kalpeshk2011/dipper-paraphraser-xxl",
        description="DIPPER paraphrase attack.",
    ),
    "opt": AttackSpec(
        name="opt",
        partition="gpu",
        nominal_edit_rate=0.15,
        uses_attack_model=True,
        attack_model="facebook/opt-2.7b",
        description="OPT-2.7B paraphrase attack.",
    ),
    "wm-removal": AttackSpec(
        name="wm-removal",
        partition="gpu",
        nominal_edit_rate=0.15,
        uses_attack_model=True,
        attack_model="facebook/opt-2.7b",
        description="Watermark-removal prompt attack.",
    ),
    "synonym": AttackSpec(
        name="synonym",
        partition="short",
        nominal_edit_rate=0.15,
        description="WordNet synonym substitution.",
    ),
    "span-synonym": AttackSpec(
        name="span-synonym",
        partition="short",
        nominal_edit_rate=0.15,
        description="Span-local synonym substitution.",
    ),
    "backtranslation": AttackSpec(
        name="backtranslation",
        partition="short",
        nominal_edit_rate=0.42,
        uses_translation_models=True,
        description="MarianMT en-fr-en backtranslation.",
    ),
    "summarization": AttackSpec(
        name="summarization",
        partition="gpu",
        nominal_edit_rate=0.55,
        uses_attack_model=True,
        attack_model="facebook/bart-large-cnn",
        description="WaterJudge-style summarization using BART-CNN.",
    ),
}


MAIN_TABLE_ATTACK_ORDER = [
    "none",
    "dipper",
    "opt",
    "wm-removal",
    "synonym",
    "backtranslation",
    "summarization",
]


LLAMA2_SWEEP_ATTACK_ORDER = [
    "synonym",
    "span-synonym",
    "opt",
    "wm-removal",
    "dipper",
    "backtranslation",
    "summarization",
]


def get_attack_spec(name: str) -> AttackSpec:
    if name not in ATTACK_SPECS:
        raise KeyError(f"Unknown attack {name}. Available: {sorted(ATTACK_SPECS)}")
    return ATTACK_SPECS[name]


def list_attacks(include_identity: bool = True) -> list[str]:
    attacks = sorted(ATTACK_SPECS)
    if include_identity:
        return attacks
    return [attack for attack in attacks if attack != "none"]

