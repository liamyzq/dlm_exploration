# 003: Frozen-Lattice Prefix-Utility Decoding

## Material Passport

- Origin: the latest [research conversation](https://chatgpt.com/c/6ab6d0b8-eee8-83ea-801a-9bb61f110338), assistant message `24665c55-9393-42ce-a7cd-5d0caf1e3c26`.
- Prepared: 2026-09-26 UTC, using research-repo and academic-research-suite planning guidance.
- Version: plan-v1.
- Status: accepted execution protocol; scientific outcomes are unmeasured.
- Authorization: implement and execute this staged study on Nebula until a supported go/no-go decision is reached. Keep unsuccessful, invalid, and inconclusive evidence distinct.

## 1. Research question and decision

Can a frozen speculative drafter's existing candidate-dependency scores support
a better single-path selection rule, yielding longer accepted prefixes and lower
decoding latency at the same candidate and verification budgets?

The first implementation is Prefix-DP. An analytic Doob sampler provides a
probabilistic alternative for the same prefix-utility surrogate. Compare both
against the checkpoint's native selector. The study initially uses greedy
target decoding, a single request, and synchronous draft-verify execution.

The contribution sought is an empirically useful inference-rule gap in a strong
frozen drafter. Dynamic programming, expected utility, and Doob transforms are
established tools. Improving a surrogate or beating categorical sampling alone
does not establish the proposed contribution.

Two hypotheses govern the study:

1. The frozen candidate lattice contains accepted-prefix headroom beyond native
   selection, and its scores identify some of that headroom on unseen prompts.
2. The resulting gain exceeds selector overhead and reduces complete decoding
   time while preserving target-only greedy token output.

Doob has an additional, separate hypothesis: softer probability reweighting can
outperform deterministic Prefix-DP under surrogate error. Its name is not a
reason to prefer it if the simpler method works better.

## 2. Feasibility and the central approximation

The approach requires an already trained candidate-dependency interface.
Freeze network weights, candidate sets, draft length, and the number of target
verification nodes. Replace selection over scores already produced by the
native drafter. Do not train an association head or add completion rollouts to
make the first experiment work.

When the scores and the surrogate are independent of the predecessor,
Prefix-DP reduces to per-position greedy selection. In particular, with
`c = q`, an independent parallel-token predictor supplies no such DP advantage.

The main uncertainty is whether scores predict the target's **greedy decision**.
A next-token probability distribution does not itself provide calibrated
probabilities of which token will be the greedy argmax. Even perfect next-token
distribution prediction need not make the proposed surrogate correct.

For example, let the first-step probabilities be 0.55 for `a` and 0.45 for `b`,
and the best next probabilities be 0.51 after `a` and 0.99 after `b`. The prefix
surrogate prefers `b` because `0.45 * 1.99 = 0.8955 > 0.55 * 1.51 = 0.8305`.
If those scores equal the target's true next-token distributions, target greedy
starts with `a`, and the surrogate-selected path fails immediately. This is a
real modeling issue to test, not a problem solved by a more accurate DP.

Accordingly, distinguish the reference proposal `q` from the survival proxy `c`.
The initial choice `c = q` is a zero-fit approximation. Conditional calibration
is a separately reported extension, not evidence that pure reweighting works.

## 3. Formulation

### 3.1 State and notation

Fix a legal drafting context `x`: committed tokens plus the drafter's available
cache and state. It contains no future target information. Suppress `x` below.

| Symbol | Meaning |
| --- | --- |
| $L$ | Number of newly speculated tokens, excluding the known anchor |
| $S_i$ | Fixed candidate set at depth $i$, with $|S_i|\le M$ |
| $y_0$ | Known anchor token |
| $q_i(b\mid a)$ | Normalized frozen score for candidate $b$ after predecessor $a$ |
| $c_i(b\mid a)$ | Nonnegative proxy for extending a surviving prefix through $b$ |
| $r_i(b\mid y_{<i})$ | Actual causal proposal probability used to generate a draft |
| $g(x)$ | Target-only greedy continuation from the committed context |

The first layer has one anchor row; subsequent layers have at most $M$ rows.
The neural scoring stage can see the whole available context. The first-order
assumption concerns the post-forward candidate selector, not the network.

$$
Q(y_{1:L})=\prod_{i=1}^{L}q_i(y_i\mid y_{i-1}),\qquad
\sum_{b\in S_i}q_i(b\mid a)=1. \tag{1}
$$

Keep this soft distribution even when the target temperature is zero. The native
greedy walk is a selection rule over the scores; it is not sampling from $Q$.

$$
0\le c_i(b\mid a)\le1,\qquad
\sum_{b\in S_i}c_i(b\mid a)\le1. \tag{2}
$$

Missing mass can represent the correct token lying outside the candidate set.
The initial experiment uses

$$ c_i=q_i. \tag{3} $$

### 3.2 True objective and proxy

The real greedy acceptance reward is

$$ A_x(y)=\operatorname{LCP}(y,g(x)). \tag{4} $$

The request-level objective is

$$
\min_{r\in\mathcal R_{\mathrm{cheap}}}\mathbb E[T_{\mathrm{decode}}(r)]
\quad\text{subject to identical output to target-only greedy}. \tag{5}
$$

$\mathcal R_{\mathrm{cheap}}$ reads only the existing candidate graph and adds
no network update, neural forward, or target verification node. Selector tensor
operations still have a measurable cost.

A per-round diagnostic is

$$
\eta(r)=\frac{\mathbb E_{Y\sim r}[A_x(Y)+1]}
{\mathbb E[C_{\mathrm{round}}(r)]}. \tag{6}
$$

Include drafting, selection, verification, and cache operations. The extra one
usually represents a correction or bonus token; at EOS or the remaining-token
limit, count actual committed tokens. Round efficiency is not equivalent to
request latency because a selector changes future round boundaries.

The optimized surrogate is

$$
U_x(y)=\sum_{k=1}^{L}\prod_{i=1}^{k}c_i(y_i\mid y_{i-1}). \tag{7}
$$

At $c=q$, it equals $\mathbb E_{Z\sim Q}[\operatorname{LCP}(y,Z)]$.
This identity characterizes the proxy, not its fidelity to $g(x)$.

## 4. Algorithm A: Prefix-DP

Choose

$$ y^{\mathrm{DP}}\in\arg\max_{y\in S_1\times\cdots\times S_L}U_x(y). \tag{8} $$

With $V_{L+1}(a)=0$, compute

$$ V_i(a)=\max_{b\in S_i}c_i(b\mid a)[1+V_{i+1}(b)]. \tag{9} $$

Store the corresponding backpointer

$$ \pi_i(a)\in\arg\max_{b\in S_i}c_i(b\mid a)[1+V_{i+1}(b)]. \tag{10} $$

Recover the path from the anchor:

$$ y_0=y_{\mathrm{anchor}},\qquad y_i=\pi_i(y_{i-1}). \tag{11} $$

The current step must survive before either its own reward or any suffix reward
can be obtained. The already accumulated utility is fixed, and the preceding
survival product multiplies every suffix choice equally. Thus the predecessor
and position are sufficient DP state. Use a fixed tie-breaking rule.

Time is $O(LM^2)$ and value/backpointer storage is $O(LM)$, excluding the existing
lattice. Provided the native path belongs to the same candidate space,

$$ U_x(y^{\mathrm{DP}})\ge U_x(y^{\mathrm{native}}). \tag{12} $$

This does not guarantee the analogous inequality for $A_x$. Greedy verification
preserves the target's final output: accept a matching prefix and emit the
target's token at the first mismatch, with normal terminal handling.

```text
V[L+1, :] = 0
for i = L, ..., 1:
    score[a, b] = c[i, a, b] * (1 + V[i+1, b])
    V[i, a], backpointer[i, a] = max_and_argmax_b(score[a, b])
a = anchor
for i = 1, ..., L:
    y[i] = backpointer[i, a]
    a = y[i]
verify the resulting single draft path with the unchanged target
```

## 5. Algorithm B: analytic Doob sampling

### 5.1 Terminal weight and backward values

Select the positive terminal weight $\phi_\lambda(y)=1+\lambda U_x(y)$,
with $\lambda\ge0$, and define

$$
R_\lambda(y)=\frac{Q(y)[1+\lambda U_x(y)]}
{1+\lambda\mathbb E_Q[U_x(Y)]}. \tag{13}
$$

This deliberately linear weight is not an exact implementation of exponential
tilting. It makes the conditional expected weight analytically tractable.
The transferable idea from discrete Doob methods is terminal reweighting;
candidate completion forwards and branch rollouts would defeat this study's
fixed-forward comparison.

Starting with $F_{L+1}(a)=0$, compute

$$
F_i(a)=\sum_{b\in S_i}q_i(b\mid a)c_i(b\mid a)[1+F_{i+1}(b)]. \tag{14}
$$

The two factors have different roles: $q$ is reference sampling probability,
and $c$ contributes to the reward. They become $q^2$ only when $c=q$.

For a chosen prefix, maintain
$s_i=\prod_{j=1}^{i}c_j(y_j\mid y_{j-1})$ and $u_i=\sum_{j=1}^{i}s_j$,
initialized with $s_0=1,u_0=0$. Then

$$ h_i(y_{1:i})=1+\lambda[u_i+s_iF_{i+1}(y_i)]. \tag{15} $$

The causal proposal is

$$
r_i(b\mid y_{<i})=q_i(b\mid y_{i-1})
\frac{h_i(y_{<i}b)}{h_{i-1}(y_{<i})}. \tag{16}
$$

The identity $h_{i-1}=\sum_bq_i(b\mid y_{i-1})h_i(y_{<i}b)$ supplies normalization.
The ratios telescope over the path, recovering (13). This is exact for the
chosen $Q,c,\phi$, regardless of whether the proxy predicts real acceptance.

```text
F[L+1, :] = 0
for i = L, ..., 1:
    F[i, a] = sum_b q[i,a,b] * c[i,a,b] * (1 + F[i+1,b])
a, s, u = anchor, 1, 0
for i = 1, ..., L:
    w[b] = q[i,a,b] * (1 + lambda *
             (u + s * c[i,a,b] * (1 + F[i+1,b])))
    r[i] = normalize(w)
    y[i] = categorical(r[i])
    retain candidate IDs and the full candidate probability vector r[i]
    s = s * c[i,a,y[i]]
    u = u + s
    a = y[i]
verify the resulting single draft path with the unchanged target
```

The backward sweep costs $O(LM^2)$. Sequential sampling reads $O(LM)$ weights.
Retaining the full candidate vector matters for later stochastic residual
sampling; proposal probability outside the set is zero.

### 5.2 Guarantees and their comparator

Write $\mu=\mathbb E_Q[U_x(Y)]$. Then

$$
\mathbb E_{R_\lambda}[U_x]-\mathbb E_Q[U_x]
=\frac{\lambda\operatorname{Var}_Q(U_x)}{1+\lambda\mu}\ge0, \tag{17}
$$

and for the fixed greedy reward,

$$
\mathbb E_{R_\lambda}[A_x]-\mathbb E_Q[A_x]
=\frac{\lambda\operatorname{Cov}_Q(A_x,U_x)}{1+\lambda\mu}. \tag{18}
$$

These compare against $Q$-sampling, not native greedy. Positive covariance must
be measured within the same context and lattice; pooling easy and hard prompts
can produce a misleading correlation.

For $\mu>0$,

$$
R_\lambda=(1-\omega_\lambda)Q+\omega_\lambda Q^U,\quad
Q^U(y)=\frac{Q(y)U_x(y)}{\mu},\quad
\omega_\lambda=\frac{\lambda\mu}{1+\lambda\mu}. \tag{19}
$$

Thus infinite guidance approaches utility-weighted $Q$, not the DP optimum.
For $\mu=0$, nonnegative utility is zero $Q$-almost surely and $R_\lambda=Q$.
Prefix-DP attains the largest surrogate value; any Doob advantage must concern
real acceptance under proxy error or other probabilistic requirements.

## 6. Exact offline evaluation

Given a target greedy continuation, any causal proposal satisfies

$$ \mathbb E_{Y\sim R}[A_x(Y)]=\sum_{k=1}^{L}R(g_{1:k}\mid x). \tag{20} $$

For the Doob sampler,

$$
\mathbb E_{R_\lambda}[A_x]
=\sum_{k=1}^{L}Q(g_{1:k})\frac{h_k(g_{1:k})}{h_0}. \tag{21}
$$

If the target token leaves the candidate set, later terms are zero. Truncate
the sum at EOS or the remaining generation horizon when applicable. The same
prefix-product evaluation applies to ordinary or temperature-adjusted sampling.
DP and native paths are evaluated directly by their LCP. One recorded lattice
and target continuation therefore permit exact comparison of the full guidance
grid without sampling drafts or running target forwards for every candidate.

The target continuation is evaluation-only data. It must not enter online
candidate construction, proxy values, backward values, or path selection.

The candidate-coverage upper bound is

$$
A_x^{\mathrm{oracle}}=\max\{k:g_i\in S_i\text{ for every }i\le k\}. \tag{22}
$$

Measure three separate gaps: oracle minus native real acceptance, DP minus
native surrogate utility, and DP minus native real acceptance. The first tests
selection headroom, the second tests what the proxy suggests, and the third
tests what the rule actually realizes. Inspect prefix-survival curves
$S_m(k)=\mathbb E_x\Pr_m(A_x\ge k)$ to localize early losses and later gains.

The source conversation reports NumPy enumeration on 200 random lattices of
length at most 5 and width at most 4, with errors below $10^{-14}$. Its linked
`reference.py` and `verification.json` were not retrieved. This is a reported
source result, not repository validation. Before using our evaluator, perform
one bounded independent enumeration comparison for the DP optimum, normalized
Doob joint distribution, (18), and (21), including $c\ne q$, missing target
candidates, and independent-position reduction. Fix mathematical or indexing
failures before collecting scientific evidence.

## 7. Baseline, data, and source status

### 7.1 Initial model pair

- Target: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), unquantized.
- Drafter: [mgoin/Qwen3-4B-speculator.dflash2](https://huggingface.co/mgoin/Qwen3-4B-speculator.dflash2).
- Initial geometry: block size 8 including the anchor, $L=7$ new speculative
  tokens, unary candidate width $M=16$.

The draft card describes a five-layer `DFlash2DraftModel`, target hidden taps
1/9/17/25/33, and a rank-256 predecessor-conditioned selector. It is experimental
Speculators training, not a reproduction of an unpublished official DFlash 2
training objective. The card's current weights are epoch 3; its reported
end-to-end benchmarks are for epoch 2. Pin the actual revisions selected for
this experiment and do not transfer those reported benchmark claims to them.

The card documents vLLM compatibility, a safety fix, and a flat-to-nested
`dflash_config` adapter. [vLLM PR 52816](https://github.com/vllm-project/vllm/pull/52816)
is merged. Speculators-format support subsequently landed in
[vLLM PR 53797](https://github.com/vllm-project/vllm/pull/53797), making the
model card's local-adapter instructions historical. Inspect the selected
runtime revision and config before installing or patching.
Record necessary adapters as ordinary model integration changes, with their
source and effect. Reproduce native behavior before changing its selector.

Use the target tokenizer and chat template with `enable_thinking=False`.
The target card discourages greedy sampling in thinking mode. The intended
condition is a non-thinking greedy decoding diagnostic, not a benchmark of
reasoning-mode quality.

### 7.2 Prompt sources and splits

| Domain | Dataset | Configuration / split | Prompt field | Published size |
| --- | --- | --- | --- | --- |
| Math | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k/blob/main/README.md) | `main` / `test` | `question` | 1,319 |
| Code | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp/blob/main/README.md) | `full` / `test` | `text` | 500 |
| General instructions | [tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval#usage) | `alpaca_eval` / `eval` | `instruction` | 805 |

Use MBPP `full`, not `sanitized/test`, which has only 257 examples. AlpacaEval's
instruction field already includes any source input. Use prompt content only;
answers and evaluation labels do not enter drafting. Record dataset revisions
and exact row IDs. These sizes are verified source metadata; dataset loading
and usable-prompt counts are implementation checks.

With fixed split seed `20260926`, sample 300 unique prompts per domain and
allocate 50 discovery, 50 validation, and 200 final-test prompts. Use exact
prompt-string comparison for duplicates; retain a deterministic replacement
rule within the same source split. Keep whole prompts within one partition.
The total is 900 prompts; initial science uses only the 300 discovery/validation
prompts. The 600 final-test prompts remain untouched until selectors are locked.

Use normal EOS, at most 1,024 newly generated tokens, and no length- or
failure-conditioned filtering. The sample size is an explicit research budget,
not a power guarantee. Short outputs and contexts near termination remain in
the recorded population with their actual horizon.

## 8. Staged execution

### Stage 0: integrate the model and establish native parity

Resolve and pin target, drafter, tokenizer, chat template, weights, precision,
runtime code, and inference settings. Start on one available Nebula GPU from
the repository's existing 5-8 pool. Use an isolated SD environment; the CoLA
venv and launcher have model-specific assumptions.

Expose these native drafting outputs:

```text
candidate_ids[i, b]
pair_scores[i, a, b]
native_selected_path[i]
```

Layer 1 logically has shape `1 x M`; subsequent layers have shape `M x M`.
Upstream may materialize the first layer as `M` identical anchor rows.
In the inspected vLLM V2 implementation, capture `candidate_ids`, `unary_logits`,
and the full `scores` tensor inside `DFlash2Speculator._generate_draft`, just
before `_sample_path`. The cached `_selector_scores` contains only rows along
the realized path and is insufficient to reconstruct the full lattice.
Inspect whether native scores combine unary and predecessor terms and reproduce
that exact rule. Do not invent a score normalization that changes native argmax.
Record candidate token ordering and tie-breaking. Determine anchor, correction,
and bonus placement from the actual implementation before counting acceptance.

Use approximately 20 development prompts for three decision-relevant checks:

- Graph extraction adds no model forward. If it does, redesign the extraction
  boundary before evaluating a purported fixed-compute method.
- Greedy walking over reconstructed scores reproduces the native path. If it
  fails, correct score interpretation or indexing before testing new selectors.
- Native SD output matches target-only greedy tokens through EOS/length limits.
  If it fails, diagnose cache handling, verification, numerical execution, or
  ties before interpreting acceptance or speed.

Any mismatch invalidates that integration result. Correct ordinary integration
bugs within the same frozen scientific specification; do not relabel a modified
model or relaxed verifier as the original baseline. Record CPU reference
verification separately from model parity. Pass permits graph collection.

### Stage 1: freeze the data and analysis choices

Materialize the exact prompt splits and generation settings from Section 7.
Use discovery prompts for integration, exploratory diagnosis, and any fitting;
validation evaluates the chosen rules. Stage-0 prompts may be taken from the
discovery partition and reused once collection is final. No final-test prompt
is consumed during setup.

Freeze selector grids, aggregation, deterministic tie-breaking, and context
selection before inspecting validation outcomes. Use split seed `20260926`,
bootstrap seed `20260927`, and proposal seeds `20260928`, `20260929`, and
`20260930` for the three final paired runs. These are protocol choices added
to make the source proposal executable.

### Stage 2: collect reusable native lattices

For each development prompt, first obtain and save the target-only greedy token
sequence. Then run native SD, confirm the same output, and collect graphs at its
legal online round boundaries. These include only committed target states and
the drafter's native permitted inputs, never future target hidden states.

Retain at most eight contexts per prompt. After observing the native round
count, select evenly spaced round indices including endpoints, removing any
rounding duplicates. Selection depends on position, not on acceptance or
failure. This produces at most 2,400 development lattices. Use all rounds for
round-count and throughput accounting even when only a subset of graphs is kept.

Each graph record includes:

```text
prompt_id, domain, split
committed_token_offset, anchor_id, effective_horizon
candidate_ids, pair_scores, normalized_q, candidate_order
native_path, native_accepted_length
model_revision, tokenizer_revision, precision, runtime_configuration
```

Store `target_greedy_continuation` in an evaluation-only record joined by graph
ID. Public prompt identifiers and exact text comparisons suffice; no per-row
hashing is needed. If caches are reconstructed offline, compare a bounded
sample against native online paths and accepted lengths before using them.

### Stage 3: exact offline selector comparison

Evaluate all rules on the identical stored graph:

| Rule | Role |
| --- | --- |
| Native greedy | Main comparator |
| Original $Q$ sampling | Comparator for the Doob identities |
| Temperature sampling | Ordinary sharpening control |
| Full-path optimizer | Whole-path likelihood control |
| Prefix-DP, $c=q$ | Primary zero-fit proposal |
| Doob, $c=q$ | Probabilistic zero-fit proposal |

For Doob, use $\lambda\in\{0,0.1,1,10,100\}$. For the temperature control,
use $\tau\in\{0.5,0.75,1,1.5\}$ with row-renormalized $q^{1/\tau}$; its mean
acceptance is also evaluated exactly. Include native deterministic selection
so an apparent gain from sharpening is not credited as a new mechanism.
The $\lambda=0$ setting is exactly original $Q$ sampling and remains a control.
Lock the proposed Doob setting among positive strengths on discovery; only
Prefix-DP and this positive-strength setting are eligible to advance. Passing
the native engineering gate alone does not establish an advantage over ordinary
sampling or sharpening. This interpretation was fixed before full development
collection and before computing any selector-comparison outcomes.

Define full-path optimization as maximizing $\sum_i\log q_i$ and label it
accordingly. If a cited baseline optimizes raw matching scores, implement that
as a distinct documented control rather than calling the two objectives equal.

Report mean real acceptance for native, DP, each Doob setting, and oracle,
plus prefix-survival curves. On contexts where DP differs from native, report
the positive/zero/negative proportions and magnitudes of $\Delta A_x$.
Report conditional covariance from (18), or compute it from exact moments;
no candidate Monte Carlo is needed for the main acceptance comparison.

The suggested initial engineering gate is

$$ \frac{1+\bar A_m}{1+\bar A_{\mathrm{native}}}-1\ge0.03. \tag{23} $$

Aggregate contexts within each prompt first and report equal-prompt means,
per-domain results, and an equal-domain pooled result. Use 2,000 prompt-level
paired bootstrap resamples within domain; a resample carries all contexts of
a prompt. The plus-one ratio is a screening diagnostic for full-length rounds;
also report actual committed-token yield with terminal handling. Preserve short
responses rather than applying a post-hoc filter to improve the ratio.

Select hyperparameters on discovery, then compare the locked settings on
validation. Advance zero-fit selectors with positive validation improvement,
a positive lower endpoint of the paired 95% interval for mean acceptance gain,
and at least the 3% point-estimate engineering gain. The 3% is an investment
threshold, not a theorem. Report improvements below it as measured but insufficient
for this version's engineering budget. Final held-out tests remain necessary
after any selection on validation.

If oracle headroom cannot reach that threshold, stop for insufficient candidate
headroom. If headroom exists but $c=q$ cannot identify it, Stage 4 is the only
planned extension before stopping. Do not add candidates, branches, rollouts,
or new training to rescue this fixed-budget question.

### Stage 4: one conditional proxy-calibration extension

Activate this stage only when oracle headroom exists and the uncalibrated
surrogate ranks paths poorly. Keep the neural networks frozen and fit

$$
c_i(b\mid a)=\rho_i\frac{q_i(b\mid a)^{1/\tau}}
{\sum_{v\in S_i}q_i(v\mid a)^{1/\tau}},\quad
\tau>0,\quad0\le\rho_i\le1. \tag{24}
$$

Use one global temperature and seven depthwise coverage parameters, fitted on
discovery only. Labels are greedy target choices, including an outside-set
category. Fit temperature on covered greedy labels and coverage on eligible
depthwise inclusion events. Only rows on a surviving true greedy prefix have
the corresponding labels; do not assign those labels to counterfactual rows.
This restricted calibration is still an approximation to richer state and
prefix dependence.

Freeze its fitting objective, bounds, and parameters before validation. Apply
the same acceptance and cost gates as Stage 3. Allow one declared calibration
pass; changes after inspecting final-test outcomes require a new study.
Label methods `c=q` or `calibrated c`. Frozen network weights do not make the
calibrated version data-free. If only calibration works, it is part of the
method and central to interpreting the result.

### Stage 5: measure selector cost and integrate survivors

Advance at most two selectors. First implement them in PyTorch, matching the
CPU float64 reference on stored graphs with FP32 value accumulation. Then
measure against an equivalently optimized native selector on the same device.

Measure incremental selector latency, including normalization, buffer writes,
and synchronization introduced by the change. Avoid per-token `.item()` calls
or host transfers. A Python loop launching many tiny kernels is not a fair
final comparison if the native selector is fused. Use the same CUDA graph and
compilation conditions, and write fused kernels only if profiling shows they
are necessary to test a promising selector.

For baseline committed yield $g_0$ and round time $t_0$,

$$
\mathrm{predicted\ speedup}\approx
\frac{1+\Delta g/g_0}{1+\Delta t/t_0}, \tag{25}
$$

so a necessary local cost condition is

$$ \Delta g/g_0>\Delta t/t_0. \tag{26} $$

Use measured native timings and actual terminal token counts. Profiling and
offline graphs are cost screens, not substitutes for complete generation.
Stop selectors whose gain is consumed by their implemented overhead. Record
whether the limitation is the rule itself or its current implementation.

### Stage 6: held-out end-to-end decision

Lock the successful selector implementations, hyperparameters, runtime, and
compilation choices before opening the 600 final-test prompts. Compare native
selection with at most two retained selectors on the same checkpoint pair.
Target-only AR provides output reference and speed context, not the novelty
baseline.

Hold candidate width, block length, weights, precision, chat template, thinking
mode, EOS, output cap, KV behavior, and runtime optimization settings fixed.
Run one request at a time. Perform three paired repetitions with randomized
method order; use the declared proposal seeds for Doob. Keep all repetitions
of a prompt together in uncertainty calculations. Warm-up is excluded and
documented consistently for each method.

Report:

- Exact token agreement with target-only greedy through the terminal boundary.
- Decode tokens/s: total committed tokens divided by total decode time.
- Speedup over native: $\sum_j T_{\mathrm{native},j}/\sum_j T_{m,j}$.
- Mean accepted draft length excluding the anchor, with terminal convention.
- Selector overhead and native/new round counts.
- Prefill, decode, and complete request latency separately, pooled and by domain.

Use paired prompt-cluster bootstrap intervals for latency ratios. Aggregate
repetitions within prompt rather than counting them as extra independent
questions. With two final selectors, report 97.5% individual intervals for the
two primary speedup claims (Bonferroni familywise 95% coverage); with one,
report a 95% interval. Preserve all predeclared comparisons and negative domains.

Success requires exact greedy output, a positive end-to-end decode speedup over
native with the applicable interval lower bound above 1, and a complete report
of full-request timing. A decode-only win with no established request-time win
is reported at that narrower scope. Claims of lower complete request latency
require their own supported paired timing result.

A valid held-out comparison without supported positive speedup closes this
version as **no demonstrated acceleration under the tested budget and setting**.
This is the terminal no-go decision, not a proof of universal impossibility.
Token mismatches, corrupted artifacts, or baseline integration errors produce
invalid runs and repair work, not scientific negative evidence.

No LLM judge is needed to establish unchanged quality when token outputs match.
Investigate mismatches in verification, cache state, precision, and ties before
interpreting quality or throughput. A result from this experimental checkpoint
supports only that pair. A broader claim needs an independently trained head or
another target/drafter pair; another random seed is not another model.

## 9. Nonzero-temperature sampling is a separate extension

The Doob proposal has explicit causal probabilities, so standard speculative
sampling can use $\alpha_i=\min\{1,p_i(y_i\mid y_{<i})/r_i(y_i\mid y_{<i})\}$ and

$$ p_i^{\mathrm{res}}(v)\propto[p_i(v)-r_i(v)]_+. \tag{27} $$

Use the actual modified $r_i$, including zero outside its candidate support.
The acceptance-plus-residual mass equals $p_i$ tokenwise. Approximate proxy
quality does not invalidate that correction when implementation probabilities
match the sampling process.

The stochastic expected acceptance objective is

$$
\mathcal A_p(r)=\sum_{k=1}^{L}\sum_{y_{1:k}}
\prod_{i=1}^{k}\min\{p_i(y_i\mid y_{<i}),r_i(y_i\mid y_{<i})\}. \tag{28}
$$

It depends on $r$ itself. Fixed greedy-reward covariance does not prove a
stochastic speedup. For $p=q=(0.8,0.2)$ and $\lambda=1$, weighting by $q(1+q)$
gives $r=(6/7,1/7)$ and lowers acceptance from 1 to $0.8+1/7\approx0.943$.
The first study therefore makes greedy claims only. Any later stochastic study
must first verify rejection/residual probabilities by small-vocabulary exact
enumeration and distribution checks, then test overlap-aware benefit separately.

## 10. Implementation, compute, and records

Use `experiments/003_doob_speculative_decoding/` as the persistent idea directory.
Planned shared implementation locations are `src/models/` for native DFlash2
integration and `src/methods/` for prefix utilities and selectors. Add commands
under `scripts/` and complete executable configurations under
`configs/003_doob_speculative_decoding/` when implemented. These names describe
ownership, not already runnable commands.

Use the existing external storage root
`/home/mlw0719/cola_dlm_exploration_storage`, with idea-specific SD environments,
upstream checkouts, models, datasets, graphs, and run outputs. Prefer Nebula
GPU 5 initially; verify current GPU memory and utilization before every new
placement. The earlier idle snapshot is not a reservation. Preserve other
users' processes and the completed CoLA studies.

Execute stages in order. Initial workload ceilings are 20 interface prompts,
300 development prompts, at most 2,400 retained graphs, five Doob settings,
four temperature settings, one calibration pass if activated, at most two
online selectors, and 600 final prompts with three paired repetitions.
The token cap is 1,024 per prompt. Profile actual runtime in Stage 0 and record
stage-specific timeout and GPU-hour estimates before long launches. Do not
silently enlarge the cohort or selector search when evidence is negative.

Commit implementation and configuration before formal runs. Use unique IDs
such as `003-s0-interface-v1`, `003-s2-graphs-v1`, `003-s3-offline-v1`,
`003-s4-calibration-v1`, `003-s5-cost-v1`, and `003-s6-heldout-v1`. A materially
changed model, protocol, or evaluator receives a new version. Run from an
unchanged detached worktree; store resolved config, exact command, source and
checkpoint revisions, seeds, logs, outputs, and terminal state with artifacts.

Append actual submissions and state changes to `jobs/jobs.jsonl`. Append formal
outcomes, including crashes and invalid runs, to `experiments/results.tsv`.
Never put planned or placeholder jobs into the real ledgers. Update the idea
README and `PROGRESS.md` when observations change the research decision.

For simple source lookups or mechanical checks, use Astra with low reasoning.
For healthy long-running downloads, setup, or experiments, use a Luna agent
with max reasoning and a check interval of at least 60 seconds. Give it exact
PIDs/job IDs, status files, output paths, completion/failure criteria, and timeout.
It remains quiet while state is unchanged and reports terminal outcomes or
decision-relevant exceptions. The main agent owns scientific design, code that
affects results, interpretation, and the final success/no-go decision.

## 11. Novelty boundaries and completion

The source discussion identifies DFlash 2 and LiLiCorr as close candidate-score
interfaces; acceptance-aware training and VSD as acceptance objectives;
OPT-Tree, DARTree, and PRESTO as related path/tree selection; and
Future-Validity/FVO-Spec as prior Doob-related grammar-conditioned SD. Treat
these as a focused related-work queue before publication, not independently
verified priority claims in this plan. The first experiment's distinguishing
constraints are a frozen candidate graph, fixed single-path verification, fixed
length, no additional neural forwards, and measured net latency benefit.

The plan ends in one of three evidence-grounded states:

1. **Success in the tested setting:** a retained selector passes correctness
   and establishes positive held-out latency benefit over native selection.
   State whether DP, zero-fit Doob, or calibrated control supplies the benefit.
2. **Scientific no-go for this version:** insufficient oracle headroom, no
   useful validation acceptance gain after the declared optional calibration,
   overhead exceeding gain, or no supported held-out speedup. Retain the valid
   negative/inconclusive measurements and their limits.
3. **Execution blocked or invalid:** infrastructure or implementation prevents
   the specified comparison. Repair within scope when possible; report a
   concrete remaining blocker rather than manufacturing a scientific failure.

The current evidence establishes a concrete mathematical construction and
source-supported setup candidates. It does not yet establish real-model
acceptance gain, speedup, or independent publication-level novelty. The first
decisive artifacts are native-parity evidence and the exact offline acceptance
comparison on legal frozen candidate graphs.
