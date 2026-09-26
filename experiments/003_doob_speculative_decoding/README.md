# 003: Frozen-Lattice Prefix-Utility Decoding

Status: execution protocol accepted; integration and experiments pending.

## Question and hypothesis

Can a frozen DLM-based speculative drafter's existing candidate-dependency
scores support a better single-path selector, increasing accepted-prefix length
and reducing decoding latency at fixed candidate and verification budgets?

The complete [English plan](PLAN.md) defines the mathematical formulation,
Prefix-DP, analytic Doob sampling, exact offline evaluation, staged experiments,
and success/no-go decisions. It develops the latest
[research conversation](https://chatgpt.com/c/6ab6d0b8-eee8-83ea-801a-9bb61f110338).

## Comparison

Begin with Qwen3-4B and the experimental Speculators DFlash2 drafter, seven
speculative tokens, top-16 candidates, and non-thinking greedy target decoding.
Native selection is the main baseline. Compare Prefix-DP and Doob reweighting
against ordinary sampling, temperature controls, and full-path optimization.
The initial proxy is `c = q`; a single conditional calibration stage is distinct.

The data budget is 900 prompts across math, code, and general instructions:
300 development prompts and 600 held-out prompts. First collect legal native
lattices and evaluate acceptance exactly. Only selectors that show useful
validation gains proceed to cost measurement and complete decoding.

## Implementation

New model integration belongs in `src/models/`, reusable selectors in
`src/methods/`, and executable experiment configurations in
`configs/003_doob_speculative_decoding/`. CoLA remains a separate model family.
The current integration candidates and source caveats are documented in the plan.

## Runs and evidence

No experiments have been run for this idea at protocol registration. Record
each formal run's committed implementation, configuration, exact command, seed,
machine, and artifact path in the repository's existing
[result ledger](../results.tsv) and [job ledger](../../jobs/jobs.jsonl).

Primary work is on Nebula; large artifacts stay under
`/home/mlw0719/cola_dlm_exploration_storage/runs/003_doob_speculative_decoding/`.
Long-running monitoring uses a Luna max subagent, which reports terminal results
or actionable changes while the main agent owns scientific interpretation.

## Findings and decision

The plan has been checked against the complete source response and preserves
all 28 numbered formulas and stages 0-6. Primary-source checks establish a
candidate deployment route and sufficiently large proposed dataset splits.
Real-model acceptance gain, end-to-end speedup, and broad novelty are unmeasured.

Proceed to baseline integration and mathematical reference verification.
Stop this version with a recorded no-go if candidate headroom, useful acceptance
gain, or net held-out speedup is absent under the declared budget. An invalid
implementation or unavailable resource is not scientific negative evidence.
