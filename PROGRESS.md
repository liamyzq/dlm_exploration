# Research Progress

## Current scope and next direction

The repository is now `dlm_exploration`, covering continuous and discrete DLMs,
inference-time control, and DLM-based speculative decoding. CoLA remains the
implemented model family for completed ideas 001 and 002.

The next direction is greedy-first prefix-utility drafting on a frozen candidate
graph. Compare native selection, Prefix-DP, and analytic Doob reweighting with
fixed candidates and verification budget. This direction is registered as
[idea 003](experiments/003_doob_speculative_decoding/README.md), with the complete
[English execution plan](experiments/003_doob_speculative_decoding/PLAN.md).
The plan separates reference probabilities from survival proxies, prioritizes
Prefix-DP, and defines exact offline acceptance evaluation before timing work.
The next actions are mathematical reference verification and native DFlash2
integration on Nebula. Speculative-decoding experiments have not started.

## Completed idea 002: task consequence and decoder geometry

The frozen core study is complete: all 36 formal jobs succeeded, including 256-world H1 and H2 tests, 512 targeted searches, four interpretation controls, and the 64-world CFG7 comparison. Every fixed test world reconstructs under both questions. See the [final report](experiments/002_task_consequence_basin/results/FINAL_REPORT.md) and [completion audit](experiments/002_task_consequence_basin/reviews/CORE_COMPLETION_AUDIT.md).

Observation: H1 protection at sigma=0.4 is -0.049 percentage points (95% world interval [-0.208,0.122]); H2 selectivity at lambda=0.5/CFG1 is -0.092 points ([-0.360,0.195]). Both intervals are nondegenerate, with ample error events, and lie inside the declared +/-2-point band. Search success saturates at RMS0.4; this does not establish nearest-boundary equivalence.

Real residuals cause about six points more field damage than matched rotations at the main CFG1 point. CFG7 reverses the general direction advantage on the control subset, while selectivity stays small. Confidence and target-specific geometry features provide no established macro predictive gain. Source-present destruction recovers 0/2,304 fields. These results concern the encoded-reference diagnostic interface.

The 100-output blinded assistant audit found five field-type discrepancies. A uniform parser repair moves 659 fields from parsed-wrong to unparseable across 193,728 retained outputs; total damage and both primary tests are unchanged. Original outputs, scores, frozen audit labels, and final figures remain available. This was an assistant review, not a human study.

Inference and decision: the stated task-selective basin hypothesis is unsupported at the frozen working points. Retain the implementation and bounded negative evidence; close this core study without training, increasing the cohort, or changing its noise points. Natural SQuAD reconstruction, native/free generation, and LoRA remain unexecuted extensions. A future CFG-dependent residual-support or verified native-task study should begin as a new frozen question.

## Completed idea 001

## Current state and decision

Idea 001's current execution scheme is formally closed; the no-training decision
remains in force. Candidate feasibility is an established positive result, and
the core F distribution observable has completed independent measurement.
R4 provides no positive support under the declared setting. R5 remains weakly
informative because the route task has an all-zero reward floor; H3 is unmeasured.

Stopping is a research-budget decision: the current proposal, observable, and
route setting lack enough positive evidence to justify further investment.
This supersedes the original P0 limitation of having no qualified starting
points; it does not assert that nothing was measured or every possible gain is
zero. See the [final report](experiments/001_same_prefix_state/results/FINAL_REPORT.md)
and [closing audit](experiments/001_same_prefix_state/reviews/FINAL_REVIEW_4922a00.md).

## Established evidence

R1 records 63/128 LAMBADA and 43/128 SQuAD successes under pinned official
conditions, with eight exact paired inference/replay checks. The original
32-example state suite and all 16 M/F prompt remainders pass. These validate
execution; they do not imply route-task competence.

R2 Copy succeeds 3/64; Chain, Branch, and Matched-full each succeed 0/64.
Branch has both poor first-edge/trunk performance and no canonical predecision
F layout on its short common trunk. Geometry and capability are distinct.

R3 reaches 32 strict pairs in each F/M development cohort. Independent F
confirmation yields 93 pairs from 128 fixed sources, at eta=0.03 and epsilon=0.01.
Fixed-source pair yield is 72.7%; 93 alternatives from 1,520 proposals give
6.1% overall proposal acceptance. It retains 34 reference fallbacks and one
terminated anchor. The conditional mean on the 93 accepted pairs is
-0.00005513 with 95% source interval [-0.00013951, 0.00002952]. The interval
includes zero. F completed natural-text distribution confirmation; M completed
feasibility and a small route-reward probe, without equivalent natural-text
confirmation. The L2 interval does not establish practical equivalence. Coupled
suffix disagreement does not establish a marginal-law effect, and this result does not prove universal text-state sufficiency.

The bounded R5 probe accepts alternatives at all eight M roots on four graphs,
but reference and alternative futures both succeed 0/64. H_reward=0 and G=0.
The all-zero bootstrap is degenerate, not a population zero bound. Task-value
heterogeneity and gain remain unsupported and poorly constrained by this floor.

## Conditional stages and remaining questions

Cache localization is not activated because the positive mechanism trigger is
unmet. Large R5 reward confirmation and equal-cost H3 are not activated because
the bounded probe shows no reward headroom. LoRA, learned selectors, and the
original human-reviewed transfer study remain outside the adopted scope. None
is reported as executed. All actual jobs and outcomes are recorded.

The current route family leaves the main state-value experiment because it is
mismatched to the frozen checkpoint. Lowering thresholds or dropping road
validity does not solve this measurement problem. Official answer-boundary
metadata offer only one LAMBADA and 15 SQuAD crossing answers in the fixed
128-item subsets, before native-root requirements. Future task screening must
first establish conditional reward variation after the intervention boundary.

The strongest reusable result is strict native same-text candidate construction
with independently measured futures. The open research question is whether a
different explicitly stated proposal/observable, or a native task with measurable
conditional reward, reveals useful hidden-state control. Treat that as a new
idea or separately frozen follow-up, not an extension until significance.
A possible bounded offline sequence-dependence analysis is only a recorded suggestion, not an active task; any
post-hoc result would be exploratory and would not alter R4 confirmation.

Primary storage stays on nebula. Formal worktrees and raw artifacts preserve
all tested commits and commands. Two pre-existing untracked P0 drafts remain
untouched and are not validated R-series entry points.


### Idea 003: offline scientific no-go (2026-09-26 UTC)

The [complete English plan](experiments/003_doob_speculative_decoding/PLAN.md)
has been executed through its conditional calibration and terminal offline
gate. The [final report](experiments/003_doob_speculative_decoding/results/FINAL_REPORT.md)
and compact JSON evidence preserve all settings and negative results.

Observation: all 300 paired native/AR prompts agree on 122,344 generated tokens,
and all 2,400 retained graphs pass reconstruction. Validation candidate coverage
permits 51.91% oracle plus-one headroom. Prefix-DP's mean acceptance gain is
-0.0117 tokens (95% CI [-0.0475, 0.0250]) for c=q and -0.0142
([-0.0542, 0.0242]) after the one discovery-only fit. Selected Doob strength 100
is below native by 0.2358 and 0.2600 tokens, respectively. Small positive DP
effects on discovery are retained but do not transfer to pooled validation.

Inference: candidate coverage is sufficient, while these fixed proxies and
selectors fail to establish useful validation acceptance gain. Doob improves
over Q sampling but does not outperform native greedy or the selected ordinary
sharpening control. No proposed selector reaches the 3% engineering gate.

Decision: close this version as `no_go_proxy_selection_under_fixed_budget`.
Stages 5-6 are not activated and the 600 final prompts remain unused. This is
an offline acceptance conclusion, not an observed end-to-end slowdown or a
universal impossibility result. Any different proxy or budget is a new study.
Collection used 2.164 device-hours within the four-device-hour ceiling. All
jobs completed, no experiment workers remain, and long waits used Luna max.

The requested [closure diagnostics](experiments/003_doob_speculative_decoding/results/DIAGNOSTIC_REPORT.md)
now tighten that conclusion. The entire nonnegative linear Doob family,
even with a separate future-label strength oracle per context, is below native
by 0.2136 tokens for c=q and 0.2407 after calibration on validation. Pure
strength tuning therefore cannot close the measured gap. The native/DP oracle
offers only 1.21% and 1.38% plus-one gain; positive and negative changes coexist,
but even removing all harm leaves insufficient headroom for the 3% gate.

Doob with native fallback is a distinct, still-open action set: its oracle
gain is 3.38% (95% CI [2.95%, 3.83%]) for c=q and 3.34% ([2.92%, 3.79%])
after calibration. This narrow future-label ceiling does not establish a
deployable improvement. Retain the no-go and avoid GPU optimization; cheap
fallback predictability would require a separately frozen study. No predictor
was fitted. Both existing splits are diagnostic development evidence, and
the final cohort remains unused. Historical implementation and provenance
checks found no discrepancy; all original per-context values reproduced exactly.
