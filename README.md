# dlm_exploration

Research on diffusion language models (DLMs): their representations, generation
dynamics, inference-time control, and use as drafters for speculative decoding.
The repository covers continuous and discrete DLMs, including parallel block
generation and candidate selection for accelerated autoregressive inference.

Each research idea has its own question, model and evaluation setting, budget,
and evidence. Shared implementations support comparisons within the appropriate
model family. See [PROGRESS.md](PROGRESS.md) for current findings and priorities.

## Research directions

- **DLM states and generation:** study how internal states, denoising dynamics,
  and decoder geometry affect generated text and task outcomes.
- **Inference-time control:** explore proposal transformations, sampling rules,
  and Doob-guided methods on frozen models.
- **Speculative decoding:** study DLM-based drafters, candidate coordination,
  accepted-prefix utility, and the cost of draft selection and verification.
  DFlash-style drafters fit this direction. The next planned study compares
  Prefix-DP and analytic Doob reweighting on a frozen candidate graph, starting
  with greedy target decoding.

## Existing studies

The current model integration is CoLA, a continuous DLM. It provides a resumable
inference engine using pinned upstream modules, native-state replay, and shared
evaluation tools.

- [001: Future value beyond the emitted prefix](experiments/001_same_prefix_state/README.md)
  is complete. The [final report](experiments/001_same_prefix_state/results/FINAL_REPORT.md)
  records native same-text candidate feasibility, zero-compatible distribution
  evidence, and a task-reward floor that limited the utility study.
- [002: Task consequence and decoder geometry](experiments/002_task_consequence_basin/README.md)
  is complete. The [final report](experiments/002_task_consequence_basin/results/FINAL_REPORT.md)
  records the encoded-reference measurements and bounded negative evidence for
  the task-selective basin hypothesis.

The speculative-decoding direction is registered as
[003: Frozen-lattice prefix-utility decoding](experiments/003_doob_speculative_decoding/README.md).
Its [execution plan](experiments/003_doob_speculative_decoding/PLAN.md) specifies
Prefix-DP and analytic Doob comparisons, exact offline acceptance evaluation,
and staged latency experiments. Baseline integration is the next step.

## Workspace and structure

The primary checkout is `nebula:/home/mlw0719/cola_dlm_exploration`; the local
mirror is `/Users/liamye/Documents/ChatGPT/dlm_exploration`. Synchronize tracked
work through [GitHub](https://github.com/liamyzq/dlm_exploration). Existing remote
checkout and storage paths retain their original names so recorded experiment
commands and artifact locations continue to resolve. See [COMPUTE.md](COMPUTE.md)
for access, storage, environments, and machine procedures.

```text
src/
  models/                 Model implementations and upstream integrations
  training/               Shared training and evaluation flows
  methods/                Reusable inference and method components
configs/                  Model, baseline, and idea configurations
experiments/
  000_baseline/           Original CoLA baseline workspace
  <idea_id>/              One persistent research idea per directory
  results.tsv             Formal experiment outcomes
scripts/                  Setup, execution, and analysis commands
jobs/jobs.jsonl           Observed job events
docs/idea-template.md     Starting point for an idea README
```

Start each idea from [the template](docs/idea-template.md) and identify its own
baseline and model provenance. Reuse shared components where the interfaces and
scientific setting match. Datasets, checkpoints, and large outputs live on
remote storage outside Git.

Develop on short-lived branches and merge accepted changes into `main`. Use
separate worktrees for concurrent development and committed detached worktrees
for formal runs. See [the workflow](docs/workflow.md).

## Research records

- [PROGRESS.md](PROGRESS.md): current understanding, open questions, and next actions.
- [experiments/results.tsv](experiments/results.tsv): formal experimental facts and decisions.
- [COMPUTE.md](COMPUTE.md): storage and compute procedures.
- [jobs/jobs.jsonl](jobs/jobs.jsonl): actual submissions and observed job states.
- [AGENTS.md](AGENTS.md): project instructions and research conventions.
