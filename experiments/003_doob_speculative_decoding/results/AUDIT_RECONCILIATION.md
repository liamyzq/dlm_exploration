# Latest audit reconciliation for idea 003

The latest research reply has been read in full: 7,048 characters, assistant
message `5dec132d-c23c-4417-8d60-1950f138e6a9` in the
[source conversation](https://chatgpt.com/c/6ab6d0b8-eee8-83ea-801a-9bb61f110338).
Its two requested closure diagnostics are already complete. The newly
highlighted early/late survival decomposition has now also been checked.
The scoped no-go remains unchanged.

## Version and task reconciliation

The reply successfully reviewed the original main-experiment report but
explicitly could not inspect the repository HEAD or the newer follow-up.
At this reconciliation, Nebula HEAD and fetched origin/main both resolve to
`8f555aec8d64f323468bd35bfc615f2bdd8c21c5`. That revision already contains the
[completed diagnostic report](DIAGNOSTIC_REPORT.md) and its
[full summary](closure_diagnostic_summary.json).

The original experiment executed at
`1ec5078e61909e4eb18105f8444bd7f525bf9986`; the two closure diagnostics
executed at `1357e7c196bfae415ff4636b8a84eb34e27c782d`. Their source and
artifact provenance were cross-checked in the preceding diagnostic audit.
This update only reconciles the latest review and adds derived arithmetic;
it does not relaunch either experiment or change its implementation.

| Requested diagnostic | Uncalibrated validation | Calibrated validation | Status |
| --- | --- | --- | --- |
| Exact native/DP switching opportunity | G_plus=0.046667; G_minus=0.058333; oracle screening gain 1.21% | G_plus=0.053333; G_minus=0.067500; oracle screening gain 1.38% | Complete |
| Oracle over every nonnegative linear Doob strength, independently per context | Gain over native -0.213594 accepted tokens; 95% CI [-0.257361, -0.171318] | Gain -0.240652; 95% CI [-0.285408, -0.197161] | Complete |

The audit's loose native/DP upper bound of about 3.026% used 20 improving
graphs times the maximum seven-token gain. The exact uncalibrated positive
gain is **56 tokens across 1,200 graphs**, rather than the loose maximum of
140. The existing 3% gate would require at least 139 tokens of total gain.
The exact 1.21% oracle therefore supersedes that loose count-only bound.
Calibration yields 64 positive tokens, still below the requirement.

The Doob diagnostic takes max(a,b/mu) separately for each graph, with the
mu=0 case handled explicitly. It does not replace that oracle by the larger
of two globally averaged endpoints. Thus it already answers the request to
close the full nonnegative strength family without a larger parameter grid.

## Verified survival decomposition

Using the full-precision validation curves in
[the original offline summary](offline_summary.json), summing survival over
depths 1-7 reproduces the published expected acceptance. For c=q the values
are native 2.8558333333, DP 2.8441666667 and Doob-100 2.6200030243.
The calibrated DP and Doob means are 2.8416666667 and 2.5958728585.

For DP minus native, the depth contributions are:

| Draft depth | c=q survival difference | Calibrated c survival difference |
| --- | ---: | ---: |
| 1 | -0.0025000000 | -0.0025000000 |
| 2 | -0.0091666667 | -0.0100000000 |
| 3 | -0.0041666667 | -0.0033333333 |
| 4 | -0.0025000000 | -0.0033333333 |
| 5 | +0.0016666667 | +0.0016666667 |
| 6 | +0.0025000000 | +0.0016666667 |
| 7 | +0.0025000000 | +0.0016666667 |
| Sum, depths 1-4 | **-0.0183333333** | **-0.0191666667** |
| Sum, depths 5-7 | **+0.0066666667** | **+0.0050000000** |
| Total | **-0.0116666667** | **-0.0141666667** |

The latest reply's uncalibrated arithmetic is correct. Calibration exhibits
the same pattern: early-depth losses exceed later-depth gains. These sums
describe the observed survival curves; they do not uniquely establish the
causal source of proxy mismatch.

This depth decomposition and G_plus/G_minus answer different questions.
The former groups survival-probability changes by depth; the latter groups
whole-context acceptance changes by sign. Both recover the same net DP loss.

## Decision and the remaining qualification

The latest audit's requested closeout is satisfied. Keep the current pure
linear Doob family and Prefix-DP route closed under the original setting and
3% investment criterion. No additional sampling seeds, strength search,
controller training or GPU optimization follows from these observations.

Preserve the qualification established by the newer diagnostics: a separate
Doob-or-native-fallback oracle has a 3.38% screening gain for c=q
(95% CI [2.95%, 3.83%]) and 3.34% after calibration ([2.92%, 3.79%]).
This marginal future-label opportunity remains distinct from both the
negative pure-family bound and a demonstrated deployable policy. Low-cost
predictability remains untested and would require a separately specified
study; the audit did not establish it.

All requested calculations use existing records. No new model forward or
formal experiment was submitted for this reconciliation, and no synthetic
job entry was added. The 600 final prompts remain unused. Full source audit
evidence, exact oracle values, uncertainty, divergence counts and artifact
locations are in the diagnostic report linked above.
