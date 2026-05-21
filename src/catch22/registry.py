from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    display_name: str
    gamma: float = 0.5
    delta: float = 2.0


METHODS: dict[str, MethodSpec] = {
    "vanilla": MethodSpec("vanilla", "baseline", "Vanilla", delta=0.0),
    "kgw": MethodSpec("kgw", "biased", "KGW", delta=2.0),
    "unigram": MethodSpec("unigram", "biased", "Unigram", delta=1.8),
    "dipmark": MethodSpec("dipmark", "bias_free", "DiPMark", delta=0.8),
    "hcw": MethodSpec("hcw", "bias_free", "HCW", delta=0.9),
    "heavywater": MethodSpec("heavywater", "bias_free", "HeavyWater", delta=1.0),
    "simplexwater": MethodSpec("simplexwater", "bias_free", "SimplexWater", delta=1.0),
    "kuditipudi": MethodSpec("kuditipudi", "bias_free", "Kuditipudi", delta=0.7),
    "semstamp": MethodSpec("semstamp", "semantic", "SemStamp", delta=0.9),
    "pmark": MethodSpec("pmark", "semantic", "PMark", delta=0.9),
    "simmark": MethodSpec("simmark", "semantic", "SimMark", delta=0.8),
    "cgw": MethodSpec("cgw", "distribution_preserving", "CGW", delta=0.4),
    "gaussmark": MethodSpec("gaussmark", "training_time", "GaussMark", delta=0.6),
    "dawa": MethodSpec("dawa", "adaptive", "DAWA", delta=1.2),
    "hybrid": MethodSpec("hybrid", "hybrid", "Hybrid", delta=1.5),
}

PAPER_METHODS = [
    "kgw",
    "unigram",
    "dipmark",
    "hcw",
    "heavywater",
    "simplexwater",
    "kuditipudi",
    "semstamp",
    "pmark",
    "simmark",
    "cgw",
    "gaussmark",
    "dawa",
    "hybrid",
]

PAPER_CONDITIONS = ["clean", "dipper_moderate", "extreme_paraphrase"]


def require_method(method: str) -> MethodSpec:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Valid methods: {', '.join(METHODS)}")
    return METHODS[method]
