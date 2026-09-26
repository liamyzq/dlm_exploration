"""Independently enumerate new oracle formulas and diagnose known path changes."""

import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from src.methods.prefix_diagnostics import doob_family_oracles, first_divergence
from src.methods.prefix_utility import accepted_prefix, greedy_statistics, native_greedy


def run():
    rng = np.random.default_rng(20260928)
    error = 0.0
    cases = 24
    for case in range(cases):
        q = [rng.uniform(.01, 1, (1 if i == 0 else 3, 3)) for i in range(3)]
        q = [row/row.sum(axis=1, keepdims=True) for row in q]
        c = [rng.uniform(0, 1, row.shape)*row for row in q]
        if case % 6 == 0:
            c[0][:] = 0
        greedy = rng.integers(0, 3, size=1+case%3).tolist()
        if case % 5 == 0:
            greedy[-1] = -1
        mass, utility, acceptance = [], [], []
        for path in itertools.product(range(3), repeat=3):
            predecessor, probability, survival, value = 0, 1.0, 1.0, 0.0
            for depth, successor in enumerate(path):
                probability *= q[depth][predecessor, successor]
                survival *= c[depth][predecessor, successor]
                value += survival
                predecessor = successor
            reward = 0
            for x, y in zip(path, greedy):
                if x != y:
                    break
                reward += 1
            mass.append(probability)
            utility.append(value)
            acceptance.append(reward)
        p, u, rewards = map(np.array, (mass, utility, acceptance))
        mu, a, b = float(p@u), float(p@rewards), float(p@(u*rewards))
        expected_ceiling = max(a, b/mu) if mu > 0 else a
        native = accepted_prefix(native_greedy(q), greedy)
        stats = greedy_statistics(q, c, greedy, [.1, 1, 100])
        result = doob_family_oracles(stats, native)
        errors = [abs(stats['mixed_acceptance_utility_q']-b),
                  abs(result['family_acceptance']-expected_ceiling),
                  abs(result['fallback_acceptance']-max(native, expected_ceiling))]
        for strength in [0, .1, 1, 100, 1e8]:
            exact = float((p*(1+strength*u)/(1+strength*mu))@rewards)
            assert exact <= result['family_acceptance']+1e-12
        error = max(error, *errors)
    examples = [
        ([0,0], [0,0], [0,0], 'same_path', 0),
        ([0,0], [0,1], [1,1], 'shared_prefix_already_wrong', 0),
        ([0,0], [0,1], [0,2], 'both_wrong_at_first_difference', 0),
        ([0,0,0], [0,1,1], [0,1,1], 'dp_corrects_native', 2),
        ([0,1,1], [0,0,0], [0,1,1], 'dp_breaks_native', -2),
        ([0,0], [0,1], [0], 'beyond_terminal', 0),
    ]
    for native, dp, greedy, category, delta in examples:
        result = first_divergence(native, dp, greedy)
        assert (result['category'], result['delta']) == (category, delta), result
    assert error < 1e-12, error
    return dict(status='passed', cases=cases, enumerated_paths=cases*27,
                divergence_categories=len(examples), maximum_absolute_error=error,
                scope='New analytic family ceiling and divergence decomposition; no model forwards.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run()
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
