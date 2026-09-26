"""Analytic family ceilings and first-divergence diagnosis on a fixed graph."""

from src.methods.prefix_utility import accepted_prefix


def doob_family_oracles(statistics, native_acceptance):
    """A per-context future-label oracle, not one globally selected strength."""
    a = statistics['mean_acceptance_q']
    mu = statistics['mean_utility_q']
    b = statistics['mixed_acceptance_utility_q']
    endpoint = b / mu if mu > 0 else a
    ceiling = max(a, endpoint)
    return dict(q_acceptance=a, mean_utility=mu, mixed_moment=b,
                infinite_strength_acceptance=endpoint, family_acceptance=ceiling,
                fallback_acceptance=max(native_acceptance, ceiling),
                preferred_endpoint=('unchanged' if endpoint == a else
                                    'infinity' if endpoint > a else 'zero'))


def first_divergence(native, dp, greedy):
    """Partition changed paths, including both-wrong and terminal neutral cases."""
    difference = next((i for i, (a, b) in enumerate(zip(native, dp)) if a != b), None)
    native_acceptance = accepted_prefix(native, greedy)
    dp_acceptance = accepted_prefix(dp, greedy)
    if difference is None:
        category = 'same_path'
    elif any(native[i] != greedy[i] for i in range(min(difference, len(greedy)))):
        category = 'shared_prefix_already_wrong'
    elif difference >= len(greedy):
        category = 'beyond_terminal'
    elif dp[difference] == greedy[difference]:
        category = 'dp_corrects_native'
    elif native[difference] == greedy[difference]:
        category = 'dp_breaks_native'
    else:
        category = 'both_wrong_at_first_difference'
    return dict(first_difference_position=None if difference is None else difference+1,
                category=category, native_acceptance=native_acceptance,
                dp_acceptance=dp_acceptance, delta=dp_acceptance-native_acceptance)
