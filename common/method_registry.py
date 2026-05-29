from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MethodSpec:
    preset: str
    method: str
    family: str
    family_superscript: str
    display_name: str
    engine: str = "native"
    script_name: str | None = None
    method_kwargs: dict = field(default_factory=dict)
    hcw_method: str = "delta"
    noise_level: float = 0.18
    highlight: str | None = None


METHOD_SPECS: dict[str, MethodSpec] = {
    "vanilla": MethodSpec(
        preset="vanilla",
        method="vanilla",
        family="baseline",
        family_superscript="",
        display_name="Vanilla",
        highlight=None,
    ),
    "kgw": MethodSpec(
        preset="kgw",
        method="kgw",
        family="biased",
        family_superscript="B",
        display_name="KGW",
        script_name="llama2_KGW_inference_LFQA.py",
        method_kwargs={"gamma": 0.5, "delta": 2.0},
    ),
    "unigram": MethodSpec(
        preset="unigram",
        method="unigram",
        family="biased",
        family_superscript="B",
        display_name="Unigram",
        script_name="llama2_Unigram_inference_LFQA.py",
        method_kwargs={"gamma": 0.5, "delta": 2.0},
    ),
    "dipmark": MethodSpec(
        preset="dipmark",
        method="dipmark",
        family="bias_free",
        family_superscript="F",
        display_name="DiPMark",
        script_name="llama2_DiPMark_inference_LFQA.py",
        method_kwargs={"alpha": 0.45, "gamma": 0.5},
    ),
    "hcw": MethodSpec(
        preset="hcw",
        method="hcw",
        family="bias_free",
        family_superscript="F",
        display_name="HCW",
        script_name="llama2_HCW_inference_LFQA.py",
        hcw_method="delta",
    ),
    "kuditipudi": MethodSpec(
        preset="kuditipudi",
        method="kuditipudi",
        family="bias_free",
        family_superscript="F",
        display_name="Kuditipudi",
        script_name="llama2_Kuditipudi_inference_LFQA.py",
        method_kwargs={
            "alpha": 0.05,
            "initial_seed": 1234,
            "dynamic_seed": "markov_1",
            "pval": 0.01,
        },
        highlight="rowcolor{red!10}",
    ),
    "heavywater": MethodSpec(
        preset="heavywater",
        method="heavywater",
        family="bias_free",
        family_superscript="F",
        display_name="HeavyWater",
        script_name="llama2_HeavyWater_inference_LFQA.py",
        method_kwargs={
            "alpha": 0.05,
            "initial_seed": 1234,
            "dynamic_seed": "markov_1",
            "gamma": 0.5,
            "delta": 5.0,
            "bl_type": "soft",
            "tilt": False,
            "tilting_delta": 0.0,
            "context": 1,
            "hashing_fn": None,
            "sinkhorn_reg": 0.05,
            "sinkhorn_thresh": 1e-5,
            "heavywater_k": 1024,
            "ht_dist": "lognormal",
        },
    ),
    "simplexwater": MethodSpec(
        preset="simplexwater",
        method="simplexwater",
        family="bias_free",
        family_superscript="F",
        display_name="SimplexWater",
        script_name="llama2_SimplexWater_inference_LFQA.py",
        method_kwargs={
            "alpha": 0.05,
            "initial_seed": 1234,
            "dynamic_seed": "markov_1",
            "gamma": 0.5,
            "delta": 5.0,
            "bl_type": "soft",
            "tilt": False,
            "tilting_delta": 0.0,
            "context": 1,
            "hashing_fn": None,
            "sinkhorn_reg": 0.05,
            "sinkhorn_thresh": 1e-5,
        },
    ),
    "semstamp": MethodSpec(
        preset="semstamp",
        method="semstamp",
        family="semantic",
        family_superscript="S",
        display_name="SemStamp",
        script_name="llama2_SemStamp_inference_LFQA.py",
        method_kwargs={
            "embedder_path": "sentence-transformers/all-mpnet-base-v2",
            "lsh_dim": 3,
            "lmbd": 0.25,
            "margin": 0.02,
            "repetition_penalty": 1.05,
            "max_trials": 100,
            "max_sentences": 12,
            "embedder_batch_size": 32,
            "alpha": 0.01,
        },
    ),
    "pmark": MethodSpec(
        preset="pmark",
        method="pmark",
        family="semantic",
        family_superscript="S",
        display_name="PMark",
        script_name="llama2_PMark_inference_LFQA.py",
        method_kwargs={
            "embedder_path": "sentence-transformers/all-mpnet-base-v2",
            "num_samples": 64,
            "msig": 4,
            "median_method": "hd",
            "pivot": "rand",
            "dedup": False,
            "parallel": False,
            "min_sequences": 10,
            "max_sentences": 12,
        },
    ),
    "simmark": MethodSpec(
        preset="simmark",
        method="simmark",
        family="semantic",
        family_superscript="S",
        display_name="SimMark",
        script_name="llama2_SimMark_inference_LFQA.py",
        method_kwargs={
            "embedder_path": "hkunlp/instructor-large",
            "similarity_metric": "cosine",
            "similarity_low": 0.68,
            "similarity_high": 0.76,
            "soft_k": 250.0,
            "interval_jitter": 0.02,
            "expected_accept_rate": 0.25,
            "max_trials": 250,
            "max_sentences": 12,
            "alpha": 0.01,
        },
    ),
    "cgw": MethodSpec(
        preset="cgw",
        method="cgw",
        family="distribution_preserving",
        family_superscript="D",
        display_name="CGW",
        script_name="llama2_CGW_inference_LFQA.py",
        method_kwargs={
            "secret_key": "christ-watermark-secret-2024",
            "security_parameter": 128,
        },
    ),
    "gaussmark": MethodSpec(
        preset="gaussmark",
        method="gaussmark",
        family="training_time",
        family_superscript="W",
        display_name="GaussMark",
        script_name="llama2_GaussMark_inference_LFQA.py",
        method_kwargs={
            "watermark_layer": 16,
            "watermark_component": "up_proj",
            "sigma": 0.05,
            "alpha": 0.05,
            "detection_max_length": 128,
        },
    ),
    "hybrid": MethodSpec(
        preset="hybrid",
        method="hybrid",
        family="hybrid",
        family_superscript="star",
        display_name="Hybrid",
        script_name="llama2_Hybrid_inference_LFQA.py",
        noise_level=0.18,
        highlight="rowcolor{blue!15}",
    ),
    "dawa": MethodSpec(
        preset="dawa",
        method="dawa",
        family="codigned",
        family_superscript="",
        display_name="DAWA",
        engine="dawa",
        method_kwargs={
            "alpha": 0.2,
            "top_p": 1.0,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "min_new_tokens": 200,
            "start": 5,
            "key": 123,
            "max_prompt_tokens": 512,
        },
        highlight="rowcolor{orange!15}",
    ),
}


LLAMA2_MAIN_METHOD_ORDER = [
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

MISTRAL_APPENDIX_METHOD_ORDER = [
    "kgw",
    "unigram",
    "dipmark",
    "hcw",
    "heavywater",
    "simplexwater",
    "semstamp",
    "pmark",
    "simmark",
    "cgw",
    "gaussmark",
    "hybrid",
]


def get_method_spec(preset: str) -> MethodSpec:
    if preset not in METHOD_SPECS:
        raise KeyError(f"Unknown method preset {preset}. Available: {sorted(METHOD_SPECS)}")
    return METHOD_SPECS[preset]


def list_methods(include_vanilla: bool = True) -> list[str]:
    methods = sorted(METHOD_SPECS)
    if include_vanilla:
        return methods
    return [method for method in methods if method != "vanilla"]


def runtime_watermark_params(spec: MethodSpec) -> dict:
    params = dict(spec.method_kwargs)
    params["preset"] = spec.preset
    params["method"] = spec.method
    params["family"] = spec.family
    params["family_superscript"] = spec.family_superscript
    params["display_name"] = spec.display_name
    params["engine"] = spec.engine
    if spec.method.startswith("hcw"):
        params["hcw_method"] = spec.hcw_method
    if spec.method == "hybrid":
        params["noise_level"] = spec.noise_level
    return params


def method_tex_label(spec: MethodSpec) -> str:
    if spec.method == "hybrid":
        return r"\textbf{Hybrid}$^{\star}$"
    if spec.method == "dawa":
        return "DAWA"
    if not spec.family_superscript:
        return spec.display_name
    return f"{spec.display_name}$^{{\\text{{{spec.family_superscript}}}}}$"
