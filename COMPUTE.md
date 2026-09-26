# Compute and Storage

## Primary workspace

The GitHub repository is `liamyzq/dlm_exploration`. Existing remote checkout and
storage directories retain their `cola_dlm_exploration` names so saved commands,
environments, and artifact references remain valid. The environment and launcher
below serve the existing CoLA studies; document the model environment and
resource allocation for a new DLM or speculative-decoding study before launch.

Use the existing local SSH alias `nebula`. At initialization it resolves to `mlw0719@nebula.osl.northwestern.edu` through `quest`. SSH connectivity and GitHub access over SSH were observed on 2026-09-14 UTC. Keep authentication in the user's existing SSH setup, outside this repository.

| Purpose | Absolute path on nebula |
| --- | --- |
| Primary Git checkout | `/home/mlw0719/cola_dlm_exploration` |
| Storage base outside Git | `/home/mlw0719/cola_dlm_exploration_storage` |
| Datasets | `/home/mlw0719/cola_dlm_exploration_storage/datasets` |
| Checkpoints | `/home/mlw0719/cola_dlm_exploration_storage/checkpoints` |
| Run outputs | `/home/mlw0719/cola_dlm_exploration_storage/runs/<idea_id>/<experiment_id>` |
| Development and frozen run worktrees | `/home/mlw0719/cola_dlm_exploration_storage/worktrees` |

The local mirror is `/Users/liamye/Documents/ChatGPT/dlm_exploration`. Move tracked work between checkouts through Git. Do not synchronize data or run directories into the local checkout.

```bash
ssh nebula
cd /home/mlw0719/cola_dlm_exploration
git status --short --branch
git pull --ff-only
```

Before downloading large assets, check the required space against current availability; a setup-time capacity reading is not a reservation. Backup and retention guarantees for this home filesystem have not been established.

## Execution state

`git` and `python3` are available in the noninteractive SSH shell. `sbatch` was not found on that shell's PATH. The project environment and user-authorized GPU indices are documented below. The P0 workers completed; the revised study keeps the released checkpoint frozen and launches no training.

Use the documented environment and launcher below. Recheck permitted GPU availability before launching new work; the user authorization is for physical indices 5-8. Prefer nebula for representative smoke tests.

Before an expensive run, read the job ledger and resolve any possibly equivalent active job, verify the chosen implementation commit, and allocate a unique output directory. Use the established launcher once one exists. Record actual job IDs or PIDs and observed state changes; never create placeholder job events.

## Run artifacts

Each formal output directory retains the resolved configuration, exact launch command, Git commit, seed, logs, metrics, and checkpoint locations needed to interpret or reproduce that run. Checkpoints may live in the shared checkpoint directory if the run records the exact path. Record any upstream source revision, checkpoint revision, or evaluator identity not determined by this repository's commit. A configured path alone does not identify changing external assets.

Frozen worktree creation and synchronization procedures are in [docs/workflow.md](docs/workflow.md). Keep a run worktree unchanged while the run is active.

## Known access behavior

Local HTTPS GitHub access failed because no HTTPS credentials were available in the shell. The existing SSH authentication successfully accessed this repository from both the local machine and nebula. Use `git@github.com:liamyzq/dlm_exploration.git`; no credential changes are needed.

## Idea 001 allocation and environment

The user authorizes physical `nvidia-smi` indices 5, 6, 7, and 8 for this study. Each is an RTX A6000 with 49,140 MiB. Use `CUDA_VISIBLE_DEVICES` to select only these indices; a single-GPU worker then uses logical `cuda:0`. At allocation inspection all four were idle. Do not terminate unrelated processes or use indices 0-4.

The pinned upstream checkout is `/home/mlw0719/cola_dlm_exploration_storage/upstream/Cola-DLM` at `7d1daeea1455a6cb9e23ddd4f06b8a2e59e63a8c`. Released weights are available at `/home/mlw0719/cola_dlm_exploration_storage/checkpoints/Cola-DLM-c1eafdd`, revision `c1eafdd9cfd8064aeb917d569ef70a075b353eed` (approximately 9.3 GB). The project venv is `/home/mlw0719/cola_dlm_exploration_storage/venv`; it inherits the existing Python 3.12 / PyTorch 2.12 environment without modifying that environment. Resolved core dependencies are recorded below.

Long-running setup or experiment waits are delegated to GPT-5.6 Luna with max reasoning. Give the monitor job IDs, artifact paths, terminal conditions, and a check interval of at least 60 seconds. It reports completion, failure, or a decision-relevant anomaly; the main agent does not repeatedly poll unchanged jobs.

Setup completed successfully. Core versions are PyTorch 2.12.0, Transformers 4.57.6, tokenizers 0.22.2, huggingface-hub 0.36.2, rotary-embedding-torch 0.8.9, and einops 0.8.2. The first GPU 5 smoke example passed exact official token parity, pause/resume, identity replay (KL 0), cache reconstruction, candidate ordering, and paired-noise checks.

Launch from a clean detached worktree with `scripts/compute/nebula/submit.py`. It records the command, commit, actual PID, device, logs, and terminal exit status. Pass the primary checkout's `jobs/jobs.jsonl` as the ledger so the frozen worktree remains unchanged. `scripts/compute/nebula/run.sh` enforces GPU indices 5-8 and sources `env.sh`. Source `env.sh` only after changing to the intended worktree; its PYTHONPATH selects that checkout and the pinned upstream modules.


## Idea 003 speculative-decoding environment

The isolated environment is `storage/venvs/idea003-vllm030` under the storage
base above. Setup completed on 2026-09-26 at 02:10:55 UTC with vLLM 0.30.0,
PyTorch 2.13.0+cu130, NumPy 2.3.5, and datasets 5.0.1. The host driver is
580.178.04; this environment does not modify the CoLA environment.

The model manifest is `runs/003_doob_speculative_decoding/setup-v1/models.json`
under the storage base. Target `Qwen/Qwen3-4B` is pinned at
`1cfa9a7208912126459214e8b04321603b3df60c`; drafter
`mgoin/Qwen3-4B-speculator.dflash2` is pinned at
`e3e7a18e4f541fa3841c2fb0666a7759079ab6fd`. The shared Hugging Face cache is
`hf_cache/`. Resolved packages and setup logs remain in the setup directory.

Use `scripts/compute/nebula/submit_sd.py` from a committed detached worktree
for SD jobs. It selects the isolated Python environment and records actual
PIDs in the supplied primary job ledger. `--gpu` accepts indices 0-9; omitting
it hides all CUDA devices for dataset construction and CPU analysis. Check
current capacity before GPU placement. At the latest preparation snapshot,
GPUs 5-8 each had about 6.7 GiB free and were busy with other VLLM processes;
this is insufficient for the planned paired model. Do not terminate them.

The native collection configuration uses eager execution and host graph
copies, so collection durations are diagnostic only. Final timing will use
a separately frozen configuration without capture overhead. Long waits for
this study are delegated to GPT-6 Luna with max reasoning.


On 2026-09-26 the user expanded the idea 003 allocation to any physically
present GPU in indices 0-9 with sufficient free memory and allowed concurrent
stacked jobs. This supersedes the earlier 5-8 restriction for idea 003 only.
Collection and integration may share a device when capacity permits; final
latency measurements must document and avoid material contention. The observed
Nebula inventory contains nine devices, indices 0-8; index 9 is not assumed to
exist. Existing unrelated processes must remain untouched.


For initial collection, the committed configuration uses tensor parallelism
across four GPUs, 0.11 device-memory utilization per GPU, and an 8,192-token
context limit. `submit_sd.py --gpu 0,1,2,3` expresses this placement. This fits
the current free-memory budget in principle; the stage-0 run establishes actual
runtime fit. The full candidate lattice is replicated by native vLLM, so only
rank zero copies and saves it. This topology is fixed before outcome collection.
