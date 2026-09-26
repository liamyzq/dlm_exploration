# Closure diagnostics for idea 003

Status: complete. See the [diagnostic report](results/DIAGNOSTIC_REPORT.md).

This bounded follow-up implements the two diagnostics requested in the latest
[research conversation](https://chatgpt.com/c/6ab6d0b8-eee8-83ea-801a-9bb61f110338),
assistant message `26e7d0f2-a998-47a1-aabe-97e6f649ea54` (7,011 characters,
read in full). The message provided a README-based partial review and explicitly
had not inspected the implementation or full result artifacts. Its suggestions
are diagnostic hypotheses, not findings of bugs.

Use the existing 300 development prompts and 2,400 captured graphs from
execution commit `1ec5078e61909e4eb18105f8444bd7f525bf9986`. Reuse both c=q and
the previously fitted calibrated c without refitting. No model forward, new
candidate, controller, GPU optimization or final-test evaluation is involved.
The original no-go remains the recorded outcome of the original protocol.

## Historical implementation and evidence cross-check

Inspect the executed version's candidate score source and axes, anchor and
target offsets, DP recurrence and backpointers, Doob normalization/moments,
terminal yield, data/uncertainty handling and actual run provenance. Reuse
the successful exhaustive reference and native-parity evidence where they
already answer the question. The new diagnostic also recomputes the original
per-context native/DP acceptance, covariance and all Doob grid expectations;
any disagreement above 1e-12 invalidates the diagnostic until resolved.

Source inspection found the complete candidate scores are captured unchanged,
including unary and association terms. Original configs and recorded commits
match all collection and analysis artifacts. The original terminal handling,
discovery-only calibration and prompt-clustered bootstrap are correct.
These observations support the offline measurement; native parity does not
establish correctness of unimplemented online selectors or caches.

## Diagnostic 1: the full nonnegative linear Doob family

For each fixed context, compute exact moments a=E_Q[A], mu=E_Q[U] and
b=E_Q[A U]. For nonnegative strength lambda,

    f(lambda) = (a + lambda b) / (1 + lambda mu).
    f'(lambda) = (b - a mu) / (1 + lambda mu)^2.

Thus the per-context supremum is max(a,b/mu) when mu>0 and a when mu=0.
Compute its acceptance gain over native, and the oracle gain when native is
also allowed as fallback: max(A_native,a,b/mu)-A_native. Each context can pick
its own endpoint using future labels. This bounds any context-dependent
nonnegative strength choice for the current Q and U on these frozen graphs.
It is not a claim that the globally averaged curve is monotone, nor a
deployable switching policy. Report endpoint frequencies and both proxies.

## Diagnostic 2: Prefix-DP benefit, harm and first divergence

For each graph let delta=A_DP-A_native. Report G_plus=E[max(delta,0)] and
G_minus=E[max(-delta,0)], verifying mean(delta)=G_plus-G_minus. G_plus is the
oracle gain from choosing only between the native and DP paths.

For changed paths, record the first divergence position (one-based) and
whether the shared prefix was already wrong, DP corrected native, or DP
broke a correct native choice. Also account for neutral cases where both
choices at the divergence are wrong or the divergence is beyond the terminal
reference horizon. These complete the partition instead of attributing every
changed path to benefit or harm. Record counts, delta magnitudes and depth
distributions; do not select attractive examples.

## Estimation, budget and interpretation

Report discovery and validation separately, plus domain means. Average the
eight contexts per prompt, then prompts within domain and domains equally.
Use the existing 2,000 paired within-domain prompt bootstrap resamples and
seed 20260927 for descriptive 95% intervals. Normalize oracle gains by
1+mean(A_native) for comparison with the existing 3% engineering threshold.
This is an acceptance ceiling, not a throughput prediction.

Run one CPU-only process on Nebula, with a ten-minute wall-time budget, using
the existing isolated environment and launcher from a committed detached
worktree. Save context-level moments and classifications outside Git; retain
compact summaries, checks and the English interpretation in the repository.

If even the corresponding oracle has insufficient measured headroom, retain
the stop decision for that family or action set. Substantial oracle headroom
would justify considering a separately frozen cheap-feature predictability
study, not immediate training or a claim of implementable gain. No such study
is part of this task. Both development splits are now diagnostic evidence;
new rules designed using them require independently evaluated evidence later.

The new smoke check independently enumerates 648 small-graph paths, including
zero utility, missing target candidates and truncated horizons, and tests all
six divergence categories. Its purpose is to detect errors in the new moment
exposure, limiting formula and decomposition; a failure blocks formal launch.
