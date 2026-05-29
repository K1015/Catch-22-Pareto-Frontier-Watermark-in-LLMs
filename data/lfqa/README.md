# LFQA input data

`inputs.jsonl` holds the long-form question-answering (LFQA) prompts consumed by
both reproduction tracks (`experiments/llama2_lfqa_main/` and
`experiments/mistral7b_appendix/`). See the [top-level README](../../README.md)
for the full pipeline and the `CATCH22_*` configuration macros.

## Format

The file is JSONL: one JSON object per line. Each row contains the following
fields.

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Stable sample identifier. |
| `prompt` | yes | Full prompt passed to the model. |
| `reference` | no | Reference answer, retained for record-keeping. |

## Row count and sampling

`inputs.jsonl` contains **2758** rows. A single experiment does not consume the
entire file: the manifest's `num_samples` (500 by default) rows are drawn from
this pool per run, so the file is intentionally larger than any single run
requires.

## Substituting other prompts

An alternative prompt set is supplied either by replacing the rows in
`inputs.jsonl` while preserving the `id` / `prompt` / optional `reference`
fields, or by directing a run to a different location through the `dataset_path`
entry in the experiment manifest or the `CATCH22_DATA_DIR` macro in `env.sh`. A
run then samples `num_samples` rows from whichever pool it is given.

## Notes

- This directory contains data only and does not reproduce paper numbers on its
  own. The reproduction pipeline is `common/` combined with `Llama2-Watermark/`
  and the manifests under `experiments/`, driven from the repository root with
  `env.sh` sourced. Dependencies are installed through `scripts/setup_env.sh`,
  which installs `requirements.txt`, the required NLTK corpora, and the external
  assets fetched by `scripts/fetch_external.sh`.
