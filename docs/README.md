# Docs

Run guides for reproducing the Catch-22 LFQA experiments. The reproduction
pipeline is `common/` + `Llama2-Watermark/`, driven by the manifests under
`experiments/llama2_lfqa_main/` (Llama-2-7B main table) and
`experiments/mistral7b_appendix/` (Mistral-7B appendix table). All paths, model
identifiers, and cluster settings are supplied through the `CATCH22_*` macros
defined in `env.sh`; see [`env.example.sh`](../env.example.sh) for the full list
of macros and their defaults.

## Contents

- [`reproduce_lfqa.md`](reproduce_lfqa.md) — reproduction of the Llama-2-7B main
  table and the Mistral-7B appendix table. Covers manifest validation, the
  one-command per-track submission, and the stage-by-stage commands
  (generation, attacks, scoring, evaluation, table render) for a single-GPU run
  without SLURM.
- [`slurm.md`](slurm.md) — reference for the macro-driven SLURM launchers under
  `common/slurm/` and the per-track entry points at
  `experiments/<track>/slurm/submit_all.sh`: how the `CATCH22_*` site macros
  propagate into each job, how to run a full track, and the queue-throttling
  policy.
