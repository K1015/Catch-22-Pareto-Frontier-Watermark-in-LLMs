# Data

This directory holds the LFQA (long-form question answering) input prompts for
the reproduction experiments. The committed prompt file is:

```
data/lfqa/inputs.jsonl
```

Each experiment manifest references this file and samples `num_samples` rows per
run (500 by default).

## Format

`inputs.jsonl` is JSON Lines: one JSON object per line. Rows follow the ELI5
long-form QA schema, with the prompt-bearing fields `prefix` (the context or
query fed to the model) and `gold_completion` (the reference answer, retained
for record keeping). Additional metadata fields (`q_id`, `title`, `selftext`,
`subreddit`, `category`, and others) accompany each row. See
[`lfqa/README.md`](lfqa/README.md) for the per-field description.

## Where the pipeline reads it

The manifests in `experiments/` reference the file through a repository-relative
`dataset_path`:

```json
"dataset_path": "../../data/lfqa/inputs.jsonl",
"num_samples": 500,
```

The input location is configurable in two ways:

- edit `dataset_path` in the manifest
  (`experiments/llama2_lfqa_main/manifest.json` or
  `experiments/mistral7b_appendix/manifest.json`), or
- set `CATCH22_DATA_DIR` to another directory (default `$CATCH22_ROOT/data`;
  see [`env.example.sh`](../env.example.sh) for the complete list of `CATCH22_*`
  macros).

## Where run artifacts go

Run artifacts are not written to this directory. Generated model outputs, attack
outputs, scores, metrics, and rendered tables are written under
`CATCH22_OUTPUT_DIR` and `CATCH22_RESULTS_DIR` (defaults `$CATCH22_ROOT/outputs`
and `$CATCH22_ROOT/results`), both of which are git-ignored, so inputs remain
separate from results.

## On a cluster

The file must remain resolvable from the repository root so that the manifest's
`dataset_path` (or `CATCH22_DATA_DIR`) continues to point at it. When the dataset
resides elsewhere on the system, set `CATCH22_DATA_DIR` or edit `dataset_path`
rather than relocating the committed file.
