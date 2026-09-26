# Idea 003 final report: frozen-lattice prefix-utility decoding

Date: 2026-09-26 UTC. Decision: **scientific no-go for this version at the
offline acceptance gate** (`no_go_proxy_selection_under_fixed_budget`).

Neither Prefix-DP nor the discovery-selected positive-strength Doob sampler
passes the predeclared validation gate, before or after the single permitted
proxy calibration. Candidate coverage leaves substantial oracle headroom;
the tested proxies and selection rules do not turn it into useful validation
acceptance gain. The protocol therefore ends before GPU selector optimization
and final latency testing. This study establishes no end-to-end speedup or
slowdown for the proposed selectors, and does not exclude other proxies,
candidate budgets, checkpoints, or objectives.

## Setting and completed evidence

The complete English [plan](../PLAN.md) preserves the source proposal's 28
numbered equations and stages 0-6. All formal development and offline runs use
commit `1ec5078e61909e4eb18105f8444bd7f525bf9986` in a frozen detached worktree.

| Item | Executed setting |
| --- | --- |
| Target | Qwen/Qwen3-4B, revision `1cfa9a7208912126459214e8b04321603b3df60c` |
| Drafter | mgoin/Qwen3-4B-speculator.dflash2, revision `e3e7a18e4f541fa3841c2fb0666a7759079ab6fd` |
| Graph budget | Seven draft positions, 16 candidates per position, one verification path |
| Target decoding | Greedy, non-thinking chat template, bfloat16, 1,024-token output cap |
| Runtime | vLLM 0.30.0, V2 runner, batch-invariant arithmetic, FlashAttention 2, eager capture |
| Device | One RTX A6000 per collection process on Nebula; four concurrent processes |
| Data | GSM8K, MBPP and AlpacaEval; 50 discovery and 50 validation prompts per domain |
| Graphs | Eight evenly spaced legal native contexts per prompt; 2,400 total |
| Inference budget | Frozen neural weights and candidate graph; no extra neural forward for selection |

The 300 paired native/AR outputs agree on all **122,344 generated tokens**.
Discovery contributes 62,637 tokens and validation 59,707. Both cohorts have
150 prompts and 1,200 retained graphs. Reconstruction and round-boundary checks
pass throughout 35,438 captured rounds, including 595 post-terminal proposals
that are counted as execution but excluded from graph evaluation. Twelve
discovery and ten validation responses reach the fixed token cap; they remain
in the analysis with terminal handling. The other 278 responses stop normally.

The mathematical reference had already passed exhaustive checks on 200 small
lattices and 20,803 paths, with maximum absolute error 8.33e-16. An earlier
default-arithmetic pilot matched only 9/20 AR/native outputs and was retained
as invalid integration evidence. Batch-invariant arithmetic repaired that
problem before the complete study. Infrastructure and integration failures
are not used as negative scientific evidence.

## Decision rule and validation results

For each method, the primary quantity is accepted draft length, excluding the
anchor. Stochastic expectations are calculated exactly from the stored graph
and true greedy continuation. Contexts are averaged within each prompt, then
prompts within each domain, then the three domains equally. Intervals use
2,000 paired prompt-cluster bootstrap resamples within domain, seed 20260927.
No candidate-path Monte Carlo is involved.

Advancement requires a positive lower endpoint of the 95% interval for mean
acceptance gain **and** at least 3% improvement in
`(1 + mean_acceptance) / (1 + native_mean_acceptance) - 1`.
On validation, the latter requires at least +0.1157 accepted tokens. Actual
committed-token yield is also reported with EOS and output-cap handling. These
ratios are offline screening quantities, not measured throughput gains.

Discovery locks Doob strength 100 and the ordinary temperature control 0.5,
both before and after calibration. Strength zero remains the original-Q
control and is ineligible as a proposed Doob survivor; this interpretation was
fixed before full collection or any selector-comparison outcome.

| Validation rule | Mean accepted tokens | Gain over native [95% CI] | Plus-one screening gain | Actual yield gain |
| --- | ---: | ---: | ---: | ---: |
| Native greedy | 2.8558 | +0.0000 [+0.0000, +0.0000] | +0.00% | +0.00% |
| Prefix-DP, c=q | 2.8442 | -0.0117 [-0.0475, +0.0250] | -0.30% | -0.29% |
| Prefix-DP, calibrated c | 2.8417 | -0.0142 [-0.0542, +0.0242] | -0.37% | -0.33% |
| Doob strength 100, c=q | 2.6200 | -0.2358 [-0.2781, -0.1942] | -6.12% | -5.92% |
| Doob strength 100, calibrated c | 2.5959 | -0.2600 [-0.3044, -0.2165] | -6.74% | -6.53% |
| Original Q sampling | 2.4288 | -0.4270 [-0.4819, -0.3748] | -11.07% | -10.75% |
| Temperature 0.5 sampling | 2.7370 | -0.1189 [-0.1544, -0.0861] | -3.08% | -2.90% |
| Full-path log-Q optimizer | 2.8475 | -0.0083 [-0.0600, +0.0408] | -0.22% | -0.16% |
| Candidate coverage oracle | 4.8575 | +2.0017 [+1.9025, +2.1017] | +51.91% | +53.16% |

Prefix-DP has slightly negative validation point estimates, with intervals
that include zero and small positive effects. There is no established useful
gain, rather than evidence that every possible benefit is exactly zero.
Doob's validation deficit relative to native is supported by entirely negative
intervals. No proposed selector survives the declared gate.

The oracle's 51.91% plus-one headroom rules out insufficient candidate coverage
as this study's stopping reason. It assumes access to the target continuation
and is not a deployable method or a speedup estimate. The negative decision
concerns the tested score-based proxies and fixed selection budget.

## Discovery effects and domain differences

The initial Prefix-DP discovery gain is +0.0400 accepted tokens, with 95% CI
[+0.0058, +0.0725] and a +1.04% screening ratio. Calibration gives +0.0417,
CI [+0.0067, +0.0750], and +1.08%. These small positive discovery observations
remain below the 3% investment threshold and do not transfer to pooled
validation. They are retained rather than described as uniformly negative.

| Validation domain | Native A | DP c=q delta | Calibrated DP delta | Doob c=q delta | Calibrated Doob delta | Oracle A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 3.6900 | -0.0275 | -0.0100 | -0.2138 | -0.2372 | 5.3550 |
| MBPP | 2.8850 | +0.0225 | -0.0025 | -0.2848 | -0.3132 | 5.0125 |
| AlpacaEval | 1.9925 | -0.0300 | -0.0300 | -0.2089 | -0.2295 | 4.2050 |

These domain entries are descriptive point estimates; no domain-specific
interval establishes a positive subgroup effect. In particular, uncalibrated
DP's +0.0225-token MBPP result does not change the pooled decision.

On validation, uncalibrated DP changes 181/1,200 paths: 20 improve acceptance,
130 leave it unchanged and 31 worsen it. Among changed paths, these are 11.05%,
71.82% and 17.13%, with mean change -0.0773 tokens. Calibrated DP changes
195 paths: 23 improve, 138 are unchanged and 34 worsen (11.79%, 70.77%, 17.44%;
mean -0.0872). A changed full path often preserves the same accepted prefix.

Doob improves over original Q sampling by +0.1912 accepted tokens before
calibration and +0.1671 afterward on validation. Mean conditional covariance
between acceptance and utility is positive: 0.5114 for c=q and 0.3366 for
calibrated c. This agrees with the reference-Q improvement identity. It does
not imply improvement over native greedy. Ordinary temperature-0.5 sampling
also outperforms the selected Doob setting descriptively, while itself
remaining below native.

## The single calibration pass

Oracle headroom and the absence of uncalibrated survivors activated exactly
one discovery-only fit. The fitted global temperature is **1.156738** within
the fixed [0.05, 20] bounds. Its mean covered-label negative log probability
is 0.684665, using 5,802 covered labels on surviving true-prefix rows.
Depthwise coverage uses only eligible true-prefix inclusion events:

| Depth | Eligible rows | Covered rows | Fitted coverage |
| --- | ---: | ---: | ---: |
| 1 | 1200 | 1181 | 0.984167 |
| 2 | 1147 | 1090 | 0.950305 |
| 3 | 1045 | 948 | 0.907177 |
| 4 | 931 | 824 | 0.885070 |
| 5 | 799 | 677 | 0.847309 |
| 6 | 664 | 594 | 0.894578 |
| 7 | 587 | 488 | 0.831346 |

Temperature fitting excludes outside-set labels and coverage accounts for
their inclusion probability at each eligible depth. Counterfactual rows are
not assigned true-prefix labels. Networks remain frozen; the calibrated
variant is nevertheless data-fitted. The parameters and selection grid are
unchanged during validation. No second fit or enlarged search was attempted.

## Prefix survival on validation

Entries are the equal-prompt probability that at least k draft tokens agree
with the target, averaged over the fixed native contexts. Each domain has
the same number of prompts.

| k | Native | DP c=q | Calibrated DP | Doob c=q, 100 | Original Q | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8292 | 0.8267 | 0.8267 | 0.8114 | 0.7682 | 0.9875 |
| 2 | 0.6150 | 0.6058 | 0.6050 | 0.5780 | 0.5349 | 0.9017 |
| 3 | 0.4542 | 0.4500 | 0.4508 | 0.4139 | 0.3780 | 0.8017 |
| 4 | 0.3450 | 0.3425 | 0.3417 | 0.3014 | 0.2745 | 0.7050 |
| 5 | 0.2575 | 0.2592 | 0.2592 | 0.2249 | 0.2049 | 0.5692 |
| 6 | 0.2008 | 0.2033 | 0.2025 | 0.1691 | 0.1557 | 0.4850 |
| 7 | 0.1542 | 0.1567 | 0.1558 | 0.1214 | 0.1126 | 0.4075 |

## Reproduction and stopping point

The complete collection took **2.164 device-hours** including startup and
capture, below the four-device-hour ceiling; elapsed time was 49.4 minutes.
Luna max monitored the long runs and submitted the fixed offline analysis
after both parity checks passed. All four collection jobs and the CPU
analysis exited successfully. Collection times include graph-copy overhead
and are not comparative acceleration evidence.

The runtime config is
`configs/003_doob_speculative_decoding/native_single_gpu.json`; the offline
grid, fit bounds and bootstrap settings are in
`configs/003_doob_speculative_decoding/offline.json`.
The dataset manifest pins source revisions and exact prompt identifiers.
Job commands and actual states are in `jobs/jobs.jsonl`; formal outcomes are
in `experiments/results.tsv`.

Compact evidence:

- [Complete offline summary](offline_summary.json), including all five Doob
  strengths, all four temperature settings, both splits, both proxy versions,
  domain means, survival curves and gate outcomes.
- [Collection accounting](collection_summary.json).
- [Discovery parity](native_parity_discovery.json) and
  [validation parity](native_parity_validation.json).

Raw graphs, generations, configurations, logs and context-level exact
statistics remain on Nebula under
`/home/mlw0719/cola_dlm_exploration_storage/runs/003_doob_speculative_decoding/`.
The four `003-dev-{discovery,validation}-{ar,native}-v1` directories contain
collection artifacts; `003-exact-offline-v1/analysis` contains the analysis.
The frozen execution checkout is
`/home/mlw0719/cola_dlm_exploration_storage/worktrees/003-development-v1`.

Stages 0-4 are complete, including the conditional calibration. Stage 5 GPU
selector cost/integration and Stage 6 held-out latency evaluation are not
activated because neither proposed selector passes validation. The 600 final
prompts have no generated model outcomes and remain unused. This is the
predeclared terminal no-go for the current version; reopening the question
with a different proxy or budget requires a separately specified study.
