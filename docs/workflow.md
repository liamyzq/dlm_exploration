# Research and Git Workflow

## Start an idea

Create `experiments/<idea_id>/README.md` from [idea-template.md](idea-template.md). Choose a stable descriptive ID such as `001_<short_name>`. The directory survives branch merges and contains the question, decision-relevant notes, exact runnable commands once available, and links to results. Identify the model family and comparator for that idea; `000_baseline` is the original CoLA workspace, while a new DLM or speculative-decoding system needs its own baseline integration and provenance. Add `configs/<idea_id>/` when an executable configuration exists. Implement reusable changes in `src/methods/` or the appropriate shared component.

Use one implementation to compare A, B, and A+B when the methods can compose. If an architectural rewrite cannot yet coexist, develop it on an isolated branch or worktree and decide how to integrate it after the evidence is available.

## Develop and synchronize

Use nebula as the default authoring checkout. Before changing either checkout, inspect its status and relevant diff; preserve existing work. Start from the latest shared history with `git fetch origin` and `git pull --ff-only` on the intended tracking branch. Never edit the same branch independently in both checkouts without integrating the outstanding work first.

```bash
git switch main
git pull --ff-only
git switch -c codex/<topic>
# Make the change and run only decision-relevant checks.
git add <changed-paths>
# Write a detailed English message to a temporary file before committing.
git commit -F /tmp/<commit-message-file>
git push -u origin codex/<topic>
```

Commit messages explain the actual changes and their purpose. Include experiment IDs and the hypothesis for experiment implementations. Record measured outcomes and their implications in subsequent research-record commits. Stage the intended paths rather than unrelated edits.

Merge accepted work when its relevant checks are complete:

```bash
git switch main
git pull --ff-only
git merge --ff-only codex/<topic>
git push origin main
```

If fast-forward integration fails because concurrent work has advanced `main`, inspect and integrate the actual changes on the development branch, validate any affected path, and write a detailed message before any merge commit. Do not reset or force-push shared work. When working locally, push the branch and fetch it on nebula before remote execution; when working on nebula, pull the completed milestone into the clean local mirror. Push coherent milestones during active work and before handoff; this is an active-session convention, not a background synchronization service.

## Freeze a formal run

Use unique experiment IDs such as `<idea_id>-001`. Before launch, specify the hypothesis, comparator, data and evaluation protocol, primary metric, budget, stopping condition, and seeds needed for the claim. Store a complete executable configuration at `configs/<idea_id>/<experiment_id>.yaml` when the model's configuration format is established; retain its resolved form with the outputs.

Commit the implementation and configuration, then on nebula create a detached worktree at that actual commit:

```bash
cd /home/mlw0719/cola_dlm_exploration
git worktree add --detach \
  /home/mlw0719/cola_dlm_exploration_storage/worktrees/<experiment_id> \
  <implementation_commit>
mkdir -p /home/mlw0719/cola_dlm_exploration_storage/runs/<idea_id>/<experiment_id>
```

These are templates: replace angle-bracket values with the selected IDs and commit. Launch using the procedure established in `COMPUTE.md`, from the frozen worktree. Record the exact command, commit, complete configuration, and artifact location before launch. Do not modify that worktree while the job runs. Disposable debugging probes need only the context necessary to interpret them.

## Record and interpret

`jobs/jobs.jsonl` is an append-only factual event log. A submitted event includes `event`, `experiment`, `machine`, `scheduler` (or `direct`), `job_id` (or PID), `commit`, `config`, `seed`, `command`, `recorded_at` in UTC, and `artifact_dir`. Append subsequent `running`, `completed`, `failed`, or `canceled` events only when observed, with relevant exit status or scheduler evidence. Do not add example records to the real ledger.

Append every formal outcome to `experiments/results.tsv`. Use `run_status` values `success`, `crash`, `invalid`, or `canceled`, and `decision` values `keep`, `discard`, or `inconclusive`. Leave the metric value empty when unavailable. Keep descriptions on one line without tabs. A successful process can still yield an inconclusive scientific result. Correct factual mistakes with a clearly superseding record identified in its description.

The idea README links experiment IDs and discusses what the observations mean. `PROGRESS.md` changes only when understanding or priorities change. Do not duplicate the full ledger in either document. Record negative outcomes and preserve their provenance; use a revert commit for rejected merged implementations when appropriate.
