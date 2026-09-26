# Project Instructions

## Scope and working environment

This project, `dlm_exploration`, studies continuous and discrete diffusion language models, their representations and inference-time control, and DLM-based drafters for speculative decoding. CoLA is the existing model integration; each new model family or decoding system establishes its own baseline and evaluation setting. Keep high-level research planning separate from claims established by experiments. Use `research-repo` for research operations and `ccf-humanization` for clear prose and proportionate verification when those skills are available. The project conventions below remain usable without a local skill installation.

Use English for all file modifications. Prefer SSH nebula for primary file storage, implementation, and representative smoke tests. The primary checkout is `/home/mlw0719/cola_dlm_exploration` on nebula; see `COMPUTE.md` before remote work. The local checkout is a lightweight synchronized mirror.

## Structure and Git

- Keep one persistent idea per `experiments/<idea_id>/` directory. Keep `000_baseline` as the original CoLA baseline workspace; identify the appropriate comparator and model provenance within each new idea. Share model, training, evaluation, and method code under `src/`; select variants through `configs/`.
- Inspect the branch, commit, status, and relevant existing changes before edits. Preserve unrelated work.
- Develop on short-lived `codex/<topic>` branches. Merge accepted changes into `main`; avoid permanent per-idea forks. Use worktrees for concurrent versions.
- Before every Git commit, write a detailed English message describing exactly what changed and why. Commit coherent milestones and push each completed milestone to `git@github.com:liamyzq/dlm_exploration.git` during active work. Synchronize the other checkout afterward. Do not force-push shared history.
- Git tracks code, configurations, compact evidence, and research records. Keep data, weights, logs, and large outputs in the nebula storage base outside the repository.
- Formal runs use a committed implementation and complete configuration in a dedicated detached worktree. Do not edit or switch a running experiment's checkout. Record its commit, configuration, exact command, and output location.

## Research memory

Read `README.md` and `PROGRESS.md` when orienting, `experiments/results.tsv` before comparisons, and `COMPUTE.md` plus `jobs/jobs.jsonl` before compute work.

For a formal experiment, state the hypothesis, comparator, primary metric, data and evaluation conditions, budget, stopping rule, and decision-relevant invalidation conditions. Record every formal outcome, including crashes and negative results. Append observed job events to `jobs/jobs.jsonl`; append experimental facts to `experiments/results.tsv`. Keep full launch commands and resolved configurations with run artifacts. Link records by experiment ID and commit rather than copying them between documents.

Update `PROGRESS.md` when evidence changes a belief, priority, or next action. Distinguish observation, inference, and the decision to keep, discard, or investigate. Smoke success establishes execution only; do not present a toy or shortened run as the named full baseline. State correct behavior plainly and preserve material negative evidence.

## Proportionate implementation and verification

Report anything actually wrong, including rare cases reachable through supported inputs or documented usage. Keep proposed fixes within the project's research scope.

- Assume a cooperating operator on their own machine unless a task establishes an adversarial setting. This is not a security paper.
- Do not add hashes, checksums, or fingerprints unless they replace a materially more expensive operation and change what happens next. Ordinary Git provenance is sufficient for experiments.
- Do not add feature flags, migration frameworks, compatibility layers, or wrappers for cases that do not occur here.
- Do not pursue exotic encodings, symlink races, RTL text, or millisecond races without a supported-use path. Reachability is enough to report a real issue; theoretical constructibility is not.
- Use judgment rather than scoring tables, ritual checklists, or repeated verification of settled behavior.
- Before every check, state the specific failure it would detect and what action would change if it failed. Without an answer, skip the check. Prefer representative smoke tests on nebula.
- Run substantive experiments needed for a scientific claim. Do not replace them with smoke tests or add speculative defensive prose.
- These scope limits do not override security, migration, verification, or review explicitly requested by the user or project.
