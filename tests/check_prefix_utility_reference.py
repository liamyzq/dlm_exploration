"""Exhaustive independent checks for the mathematical reference, not LLM evidence."""

import argparse
import itertools
import json
from pathlib import Path
import numpy as np

from src.methods.prefix_utility import (
    accepted_prefix, doob_conditionals, full_path_optimum, greedy_statistics,
    native_greedy, path_probability, prefix_dp, prefix_utility,
)


def run(seed=20260926, cases=200):
    rng = np.random.default_rng(seed)
    maximum_error = 0.0
    paths_checked = 0
    for case in range(cases):
        length = int(rng.integers(1, 6))
        width = int(rng.integers(2, 5))
        q, c = [], []
        for i in range(length):
            shape = (1 if i == 0 else width, width)
            rows = rng.uniform(0.01, 1.0, shape)
            rows /= rows.sum(axis=1, keepdims=True)
            if case % 5 == 0:
                rows[:] = rows[0]
            q.append(rows)
            proxy = rng.uniform(0.01, 1.0, shape)
            proxy /= proxy.sum(axis=1, keepdims=True)
            proxy *= rng.uniform(0.0, 1.0, (shape[0], 1))
            c.append(rows.copy() if case % 2 == 0 else proxy)
        if case % 17 == 0:
            c[0][:] = 0.0
        paths = list(itertools.product(range(width), repeat=length))
        masses = np.array([path_probability(q, p) for p in paths])
        utilities = np.array([prefix_utility(c, p) for p in paths])
        greedy = rng.integers(0, width, size=length).tolist()
        if case % 3 == 0:
            greedy[int(rng.integers(length))] = -1
        rewards = np.array([accepted_prefix(p, greedy) for p in paths])
        dp_path, dp_value = prefix_dp(c)
        errors = [abs(dp_value - utilities.max()),
                  abs(prefix_utility(c, dp_path) - utilities.max()),
                  abs(path_probability(q, full_path_optimum(q)) - masses.max())]
        if case % 10 == 0 and case % 17 != 0:
            assert dp_path == native_greedy(q)
        mu = float(masses @ utilities)
        ea = float(masses @ rewards)
        covariance = float(masses @ (rewards * utilities) - ea * mu)
        statistics = greedy_statistics(q, c, greedy, [0.0, 0.1, 1.0, 10.0, 100.0])
        errors += [abs(statistics['mean_utility_q']-mu),
                   abs(statistics['mean_acceptance_q']-ea),
                   abs(statistics['covariance_acceptance_utility']-covariance)]
        for strength in [0.0, 0.1, 1.0, 10.0, 100.0]:
            tilted = masses * (1 + strength * utilities) / (1 + strength * mu)
            errors.append(abs(tilted.sum() - 1))
            for p, expected in zip(paths, tilted):
                rows = doob_conditionals(q, c, p, strength)
                actual = float(np.prod([row[b] for row, b in zip(rows, p)]))
                errors.extend([abs(actual-expected)] + [abs(row.sum()-1) for row in rows])
            expected_a = float(tilted @ rewards)
            errors.append(abs(statistics['mean_acceptance_doob'][str(strength)]-expected_a))
            errors.append(abs(expected_a-ea-strength*covariance/(1+strength*mu)))
            errors.append(abs(float(tilted @ utilities)-mu-strength*float(masses @ (utilities-mu)**2)/(1+strength*mu)))
        maximum_error = max(maximum_error, max(errors))
        assert max(errors) < 1e-11, (case, max(errors))
        paths_checked += len(paths)
    counter_q = [np.array([[0.8, 0.2]])]
    rows = doob_conditionals(counter_q, counter_q, [0], 1.0)[0]
    overlap = float(np.minimum(counter_q[0][0], rows).sum())
    assert abs(overlap-(0.8+1/7)) < 1e-14
    return {'status': 'passed', 'seed': seed, 'cases': cases,
            'enumerated_paths': paths_checked, 'maximum_absolute_error': maximum_error,
            'stochastic_counterexample_acceptance': overlap,
            'scope': 'CPU mathematical reference; no LLM or latency claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--cases', type=int, default=200)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.seed, args.cases)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
