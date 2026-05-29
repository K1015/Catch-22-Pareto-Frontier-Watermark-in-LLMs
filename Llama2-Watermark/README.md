# Llama2-Watermark — per-method inference wrappers

This directory contains the per-method inference-time watermarking wrappers that
`common/watermark_adapters.py` loads and drives. Each file implements one
watermarking scheme's generation logic and, where applicable, its keyed
detection logic. The adapter layer maps a method key to a file in this directory
and instantiates the corresponding wrapper class with the run's `model_name`.

Despite the historical `llama2_` filename prefix, the wrappers are
model-agnostic. The model is selected per run through the manifest `model_name`
field, which resolves to the `CATCH22_MODEL_LLAMA2` and `CATCH22_MODEL_MISTRAL`
macros defined in `env.sh`. The same files therefore back both the Llama-2-7B
main experiment (`experiments/llama2_lfqa_main/`) and the Mistral-7B appendix
experiment (`experiments/mistral7b_appendix/`). No checkpoint is hard-coded in
this directory.

## Method files

Each file exposes a wrapper consumed by `common/watermark_adapters.py`:

| File | Method key |
| --- | --- |
| `llama2_vanilla_inference_LFQA.py` | `vanilla` (unwatermarked baseline) |
| `llama2_KGW_inference_LFQA.py` | `kgw` |
| `llama2_Unigram_inference_LFQA.py` | `unigram` |
| `llama2_DiPMark_inference_LFQA.py` | `dipmark` |
| `llama2_HCW_inference_LFQA.py` | `hcw` |
| `llama2_HCW_v2_inference_LFQA.py` | `hcw-v2` |
| `llama2_Kuditipudi_inference_LFQA.py` | `kuditipudi` |
| `llama2_CGW_inference_LFQA.py` | `cgw` |
| `llama2_GaussMark_inference_LFQA.py` | `gaussmark` |
| `llama2_HeavyWater_inference_LFQA.py` | `heavywater` |
| `llama2_SimplexWater_inference_LFQA.py` | `simplexwater` |
| `llama2_PMark_inference_LFQA.py` | `pmark` |
| `llama2_SemStamp_inference_LFQA.py` | `semstamp` |
| `llama2_SimMark_inference_LFQA.py` | `simmark` |
| `llama2_Hybrid_inference_LFQA.py` | `hybrid` (ours) |

`heavywater` and `simplexwater` are re-export shims: both delegate to the shared
implementation in `llama2_HeavySimplex_inference_common.py`.

## Shared helpers

| File | Role |
| --- | --- |
| `watermark_rebuild_common.py` | Common model and tokenizer loading plus numerical utilities (for example `load_tokenizer_compat` and probability-safety helpers) reused across wrappers; also routes model-specific tokenizer settings through `model_name`. |
| `semantic_embedder_compat.py` | Sentence-embedding wrapper used by the semantic method paths (PMark and SemStamp), built on `watermark_rebuild_common`. |
| `llama2_HeavySimplex_inference_common.py` | Shared HeavyWater and SimplexWater implementation behind the two re-export shims listed above. |

## External dependencies

The semantic and distortion-free schemes rely on the upstream method
repositories fetched into `external/` by `scripts/fetch_external.sh` (SemStamp,
PMark, and HeavyWater/SimplexWater at pinned commits). Run
`scripts/setup_env.sh`, which installs `requirements.txt` and the NLTK corpora
and invokes `scripts/fetch_external.sh`, before running those methods.

See the [repository README](../README.md) for the full method table, the
`CATCH22_*` configuration macros, and the end-to-end reproduction workflow.
