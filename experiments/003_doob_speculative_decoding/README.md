# 003: Frozen-Lattice Prefix-Utility Decoding

Status: **complete — scientific no-go at the offline acceptance gate**.

The [final report](results/FINAL_REPORT.md) records valid measurements on 300
prompts and 2,400 frozen DFlash2 candidate graphs. All 122,344 generated tokens
agree between native speculative decoding and target-only greedy decoding.

Candidate coverage leaves a 51.91% oracle plus-one screening gain, but the
tested proxies do not exploit it. Validation Prefix-DP gains are -0.0117 tokens
for c=q (95% CI [-0.0475, 0.0250]) and -0.0142 after calibration
([-0.0542, 0.0242]). Discovery-selected Doob strength 100 remains below native
by 0.2358 tokens before calibration and 0.2600 afterward. No proposed selector
passes the predeclared positive-interval and 3% engineering gate.

The one permitted discovery-only calibration is complete. Stages 5-6 are not
activated; the 600 final prompts remain unused. This is a scoped no-go for the
tested rules, checkpoint pair and budget. End-to-end acceleration of the
proposed selectors is unmeasured.

The requested [closure diagnostics](results/DIAGNOSTIC_REPORT.md) are also
complete. A future-label oracle choosing any nonnegative Doob strength per
context remains below native by 0.2136 accepted tokens for c=q and 0.2407
after calibration. Perfect native/DP switching offers only 1.21% and 1.38%
validation plus-one gain. Allowing native fallback for Doob leaves 3.38% and
3.34% oracle gain, with intervals spanning the 3% threshold; this marginal
opportunity has no demonstrated deployable switching rule. The historical
implementation/provenance review found no discrepancy, and recomputation of
the original per-context outcomes has zero difference.

## Plan and evidence

The complete [English plan](PLAN.md) preserves all 28 numbered equations from
the [source research conversation](https://chatgpt.com/c/6ab6d0b8-eee8-83ea-801a-9bb61f110338),
including Prefix-DP, analytic Doob sampling, exact acceptance expectations,
conditional calibration and staged stopping decisions.

- [Final report](results/FINAL_REPORT.md): setting, controls, uncertainty,
  domain differences, calibration, interpretation and stopping point.
- [Exact offline summary](results/offline_summary.json): every tested setting.
- [Collection accounting](results/collection_summary.json),
  [discovery parity](results/native_parity_discovery.json) and
  [validation parity](results/native_parity_validation.json).
- [Dataset manifest](data_manifest.json), [reference verification](reference_verification.json),
  [result ledger](../results.tsv) and [job ledger](../../jobs/jobs.jsonl).

Implementation is shared under `src/models/`, `src/methods/` and `src/tasks/`.
Frozen configurations are under `configs/003_doob_speculative_decoding/`.
The formal development and offline execution commit is
`1ec5078e61909e4eb18105f8444bd7f525bf9986`.
Large artifacts remain on Nebula under
`/home/mlw0719/cola_dlm_exploration_storage/runs/003_doob_speculative_decoding/`.
