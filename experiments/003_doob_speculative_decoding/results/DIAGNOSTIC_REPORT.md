# Idea 003: closure diagnostics and implementation cross-check

Date: 2026-09-26 UTC. The two requested diagnostics are complete. They support
retaining the original no-go for the tested selectors. A pure linear Doob
family cannot match native even with an oracle choosing its strength separately
for every observed context. The native/Prefix-DP switching oracle is below the
3% acceptance-screening threshold. Allowing native fallback for Doob leaves a
small, future-label-dependent ceiling of about 3.3%; its deployability is open.

This follow-up implements the [diagnostic plan](../DIAGNOSTIC_PLAN.md) drawn
from the complete latest [research conversation](https://chatgpt.com/c/6ab6d0b8-eee8-83ea-801a-9bb61f110338),
message `26e7d0f2-a998-47a1-aabe-97e6f649ea54`. It uses the same 300 development
prompts and 2,400 frozen graphs, original c=q and frozen calibrated c. There
were no model forwards, extra GPU jobs, new calibration, predictor fitting or
final-test access. Original and diagnostic results remain separate.

## 1. What the implementation audit established

The executed collection and offline version is
`1ec5078e61909e4eb18105f8444bd7f525bf9986`. All four collection configurations
and five launch records identify that version. Saved runtime configurations
match its configuration, the original analysis config matches its offline
config, and the repository's original summary equals the run artifact.

Source inspection found no omitted selector score term: the capture hook
copies the exact score tensor given to native selection, after the native
call, including unary logits and predecessor/successor association scores.
The first predecessor row corresponds to the repeated known anchor. The
native walk starts at row zero and carries successor indices forward. The
existing full-cohort checks already reconstructed all 2,400 retained graphs.

The saved sample position minus prompt length points to the first speculative
token after the anchor. Candidate-missing target indices stop prefix survival.
The DP recurrence uses predecessor rows and successor columns consistently,
and backtracking starts at the single anchor row. Doob suffix values and
conditional normalization represent Q(y)(1+lambda U(y))/(1+lambda E_Q[U]).
The mixed moment follows by summing the utility-weighted true-prefix masses.
The earlier independent exhaustive reference tests cover off-native branches,
Doob conditionals and exact acceptance, beyond native output parity alone.

Terminal handling is correct for the measured object: acceptance stops at the
available reference horizon, deterministic yield is min(A+1, remaining), and
stochastic yield subtracts the terminal survival mass from 1+E[A]. Proposals
after termination remain counted as execution but are excluded from graph
evaluation. Calibration uses discovery true-prefix labels only. Statistics
average contexts within prompts, then domains equally, with paired prompt
resampling inside domains.

The original full summary also contains the paired Doob intervals that were
unavailable to the external README-only review: [-0.2781, -0.1942] tokens
before calibration and [-0.3044, -0.2165] afterward. Both support a deficit
relative to native for the selected strength, within the tested setting.

No material implementation or provenance discrepancy was found. As an
additional result-producing cross-check, the new run exactly reproduces the
historical per-context native/DP acceptance, covariance and all five Doob-grid
expectations for both proxies and splits: maximum absolute difference **0**.
The new formula smoke independently enumerates 648 paths in 24 small graphs
and checks six divergence categories, including zero utility, missing target
candidates and truncated horizons; maximum error is 4.44e-16. Existing model
parity checks were reused rather than rerun. None of these observations
establishes correctness of a new online selector or its cache integration,
which was never implemented in this study.

## 2. Diagnostic one: the entire nonnegative Doob family

For each context, a=E_Q[A], mu=E_Q[U], and b=E_Q[A U] give

    f(lambda) = (a + lambda b)/(1 + lambda mu).
    sup_{lambda >= 0} f(lambda) = max(a, b/mu), for mu > 0.

When mu=0 the result is a. No observed context has zero utility. The best
endpoint can differ across contexts; this computation does not assume that
the dataset-average curve is monotone. With native fallback, the oracle is
max(A_native, a, b/mu). Both use future target labels to choose an action.

Validation has native mean acceptance 2.8558. All gains below are accepted
draft tokens, followed by the plus-one screening ratio relative to native.
Intervals are descriptive 95% paired prompt-bootstrap intervals using the
same 2,000 within-domain resamples and seed as the original study.

| Proxy | Oracle action set | Acceptance gain [95% CI] | Screening gain [95% CI] |
| --- | --- | ---: | ---: |
| c=q | Any nonnegative Doob strength | -0.2136 [-0.2574, -0.1713] | -5.54% [-6.60%, -4.49%] |
| c=q | Doob strength or native fallback | +0.1303 [+0.1149, +0.1457] | +3.38% [+2.95%, +3.83%] |
| Calibrated c | Any nonnegative Doob strength | -0.2407 [-0.2854, -0.1972] | -6.24% [-7.32%, -5.17%] |
| Calibrated c | Doob strength or native fallback | +0.1289 [+0.1136, +0.1441] | +3.34% [+2.92%, +3.79%] |

The pure-family conclusion is stronger than failure at lambda=100: on these
frozen contexts, no globally chosen or context-dependent nonnegative strength
can exceed the per-context oracle, which itself remains below native. Further
strength sweeps or a strength-only controller cannot repair this measured gap.

On validation, c=q prefers infinity in 901 contexts, zero in 284, and ties in
15; calibrated c gives 910, 275 and 15. Mixed derivative signs are present,
so comparing only two global endpoint means would have been insufficient.

The fallback oracle is a distinct action set and must be reported separately.
Its point estimate slightly exceeds 3%, while its interval includes values
below 3%. It does not support a claim that all switching opportunities have
been excluded. Conversely, it is an optimistic acceptance ceiling with perfect
future knowledge and no switching error or implementation cost.

Reaching the original acceptance gate requires +0.1157 tokens, or
88.8% of the c=q fallback oracle gain and 89.8% of the calibrated
fallback oracle gain, before any runtime cost is considered. This narrow margin
does not justify GPU optimization now. Whether cheap available features can
achieve useful selective fallback remains an untested question for a separately
frozen study; this diagnostic does not fit or claim such a policy.

## 3. Diagnostic two: Prefix-DP benefit, harm and switching ceiling

Let delta=A_DP-A_native, G_plus=E[max(delta,0)] and
G_minus=E[max(-delta,0)]. Then mean(delta)=G_plus-G_minus.
G_plus is exactly the future-label oracle gain from selecting only native or
DP per context. The measured identity residual is below 1e-12 throughout.

| Proxy | Split | G_plus | G_minus | Net DP gain | Native/DP oracle screening gain [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: |
| c=q | Discovery | 0.0833 | 0.0433 | +0.0400 | +2.16% [+1.40%, +2.95%] |
| c=q | Validation | 0.0467 | 0.0583 | -0.0117 | +1.21% [+0.66%, +1.87%] |
| Calibrated c | Discovery | 0.0858 | 0.0442 | +0.0417 | +2.23% [+1.48%, +3.04%] |
| Calibrated c | Validation | 0.0533 | 0.0675 | -0.0142 | +1.38% [+0.81%, +2.06%] |

Beneficial and harmful changes coexist, but their gross positive opportunity
is small. On validation, even perfect native/DP switching reaches only
1.21% or 1.38%; both 95% upper endpoints remain below 3%. A predictor that
only decides whether to take the already computed DP path cannot overcome
this action-set ceiling on the measured contexts. The conclusion concerns
this rule and graph distribution, not every possible candidate selector.

The first-divergence decomposition explains the cancellation and neutral changes:

| Validation category | c=q graphs | Calibrated c graphs |
| --- | ---: | ---: |
| Native and DP paths identical | 1019 | 1005 |
| Shared prefix already wrong at first divergence | 86 | 92 |
| Shared prefix correct, both divergent choices wrong | 27 | 28 |
| First divergence after the terminal reference horizon | 17 | 18 |
| DP corrects a wrong native next choice | 20 | 23 |
| DP breaks a correct native next choice | 31 | 34 |

For c=q, the 20 beneficial contexts contribute 56 accepted tokens in total
(+2.80 each), while 31 harmful contexts lose 70 (-2.26 each). Thus G_plus
is 56/1,200 and G_minus is 70/1,200. The other 130 changed paths have zero
acceptance effect: 86 changed only after an earlier error, 27 changed between
two wrong next choices, and 17 differed only after termination.

Calibration yields 23 beneficial contexts (+64 tokens) and 34 harmful contexts
(-81 tokens), with 138 neutral changes. It does not create enough additional
positive opportunity. The both-wrong and terminal categories are necessary:
the three categories suggested in the source message alone do not exhaust
the actual data.

| First divergence position | c=q corrections | c=q harms | Calibrated corrections | Calibrated harms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 10 | 13 | 11 | 14 |
| 2 | 2 | 10 | 3 | 13 |
| 3 | 3 | 4 | 4 | 4 |
| 4 | 2 | 4 | 2 | 3 |
| 5 | 2 | 0 | 3 | 0 |
| 6 | 1 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 |

The complete summary retains the depth distributions for neutral cases and
the full positive/negative delta distributions as well; no examples were
selected to represent the result.

## 4. Domain results and research decision

Each validation domain contains 50 prompts and 400 graphs. These are
descriptive point estimates; the pooled decision retains equal domain weight.

| Proxy | Validation domain | Pure Doob oracle gain | Doob/native fallback oracle gain | Native/DP oracle gain |
| --- | --- | ---: | ---: | ---: |
| c=q | math | -0.1918 | +0.1356 | +0.0475 |
| c=q | code | -0.2663 | +0.1350 | +0.0525 |
| c=q | general instructions | -0.1827 | +0.1203 | +0.0400 |
| Calibrated c | math | -0.2182 | +0.1341 | +0.0625 |
| Calibrated c | code | -0.2974 | +0.1334 | +0.0600 |
| Calibrated c | general instructions | -0.2064 | +0.1191 | +0.0375 |

Retain the original no-go, stop expanding the nonnegative linear Doob strength
search, and stop the current Prefix-DP/native switching route under the 3%
investment criterion. Candidate coverage remains high, but these narrower
action sets have much less usable acceptance headroom. Native fallback for
Doob remains a marginal oracle opportunity whose low-cost predictability has
not been established. No broad conclusion that all deployment-time scores
contain no useful information follows from this result.

Both discovery and validation now contribute post-hoc diagnostic evidence.
They must not be presented as untouched evaluation data for a new rule
designed from these findings. The 600 final prompts remain unused, and stages
5-6 of the original study remain unactivated.

## Artifacts and reproduction

Diagnostic implementation: `1357e7c196bfae415ff4636b8a84eb34e27c782d`.
Configuration: `configs/003_doob_speculative_decoding/closure_diagnostics.json`.
Run: `003-closure-diagnostics-v1`, CPU-only on Nebula. The process exited zero.

- [Full diagnostic summary](closure_diagnostic_summary.json): every split,
  proxy, moment aggregate, interval, divergence category and domain mean.
- [Independent formula check](diagnostic_reference_verification.json).
- [Original report](FINAL_REPORT.md) and [original summary](offline_summary.json).

Raw context-level moments, paths and divergence labels are stored at
`/home/mlw0719/cola_dlm_exploration_storage/runs/003_doob_speculative_decoding/003-closure-diagnostics-v1/analysis`.
The run directory also contains the exact launch record, log and terminal exit
file. Job and result ledgers retain the actual submission and completion.


Latest review: the [audit reconciliation](AUDIT_RECONCILIATION.md) confirms that
its two requested oracle calculations are already complete and adds the
verified early/late survival decomposition. The original decision and the
marginal native-fallback qualification are unchanged.
