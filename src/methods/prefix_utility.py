"""CPU reference for fixed-lattice prefix-utility selection and exact evaluation."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

Rows = Sequence[np.ndarray]


def probabilities(scores: Rows, temperature: float = 1.0) -> list[np.ndarray]:
    result = []
    for score in scores:
        z = np.asarray(score, dtype=np.float64) / temperature
        w = np.exp(z - z.max(axis=-1, keepdims=True))
        result.append(w / w.sum(axis=-1, keepdims=True))
    return result


def native_greedy(q: Rows) -> list[int]:
    a = 0
    path = []
    for rows in q:
        a = int(np.argmax(rows[a]))
        path.append(a)
    return path


def prefix_utility(c: Rows, path: Sequence[int]) -> float:
    a, survival, utility = 0, 1.0, 0.0
    for rows, b in zip(c, path):
        survival *= float(rows[a, b])
        utility += survival
        a = b
    return utility


def prefix_dp(c: Rows) -> tuple[list[int], float]:
    suffix = np.zeros(c[-1].shape[1], dtype=np.float64)
    back = []
    for rows in reversed(c):
        values = np.asarray(rows) * (1.0 + suffix[None, :])
        choices = values.argmax(axis=1)
        suffix = values[np.arange(len(choices)), choices]
        back.append(choices)
    a = 0
    path = []
    for choices in reversed(back):
        a = int(choices[a])
        path.append(a)
    return path, float(suffix[0])


def full_path_optimum(q: Rows) -> list[int]:
    suffix = np.zeros(q[-1].shape[1], dtype=np.float64)
    back = []
    for rows in reversed(q):
        with np.errstate(divide='ignore'):
            values = np.log(rows) + suffix[None, :]
        choices = values.argmax(axis=1)
        suffix = values[np.arange(len(choices)), choices]
        back.append(choices)
    a = 0
    path = []
    for choices in reversed(back):
        a = int(choices[a])
        path.append(a)
    return path


def future_values(q: Rows, c: Rows) -> list[np.ndarray]:
    values = [np.zeros(q[-1].shape[1], dtype=np.float64)]
    for qrows, crows in zip(reversed(q), reversed(c)):
        values.append((qrows * crows * (1.0 + values[-1][None, :])).sum(axis=1))
    return list(reversed(values))


def doob_row(qrow: np.ndarray, crow: np.ndarray, next_value: np.ndarray,
             survival: float, utility: float, strength: float) -> np.ndarray:
    weights = qrow * (1.0 + strength * (utility + survival * crow * (1.0 + next_value)))
    return weights / weights.sum()


def doob_conditionals(q: Rows, c: Rows, path: Sequence[int], strength: float) -> list[np.ndarray]:
    values = future_values(q, c)
    a, survival, utility = 0, 1.0, 0.0
    result = []
    for i, b in enumerate(path):
        result.append(doob_row(q[i][a], c[i][a], values[i+1], survival, utility, strength))
        survival *= float(c[i][a, b])
        utility += survival
        a = b
    return result


def doob_sample(q: Rows, c: Rows, strength: float,
                rng: np.random.Generator) -> tuple[list[int], list[np.ndarray]]:
    values = future_values(q, c)
    a, survival, utility = 0, 1.0, 0.0
    path, rows = [], []
    for i in range(len(q)):
        row = doob_row(q[i][a], c[i][a], values[i+1], survival, utility, strength)
        b = int(rng.choice(len(row), p=row))
        path.append(b)
        rows.append(row)
        survival *= float(c[i][a, b])
        utility += survival
        a = b
    return path, rows


def path_probability(q: Rows, path: Sequence[int]) -> float:
    a, probability = 0, 1.0
    for rows, b in zip(q, path):
        probability *= float(rows[a, b])
        a = b
    return probability


def accepted_prefix(path: Sequence[int], greedy: Sequence[int]) -> int:
    length = 0
    for proposed, target in zip(path, greedy):
        if proposed != target:
            break
        length += 1
    return length


def greedy_statistics(q: Rows, c: Rows, greedy: Sequence[int],
                      strengths: Sequence[float]) -> dict:
    """Evaluate (18)/(21); -1 marks a greedy token outside the candidate set."""
    values = future_values(q, c)
    mu = float(values[0][0])
    a, mass, survival, utility = 0, 1.0, 1.0, 0.0
    prefix_mass, weighted_utility = [], []
    for i, b in enumerate(greedy[:len(q)]):
        if b < 0:
            break
        mass *= float(q[i][a, b])
        survival *= float(c[i][a, b])
        utility += survival
        prefix_mass.append(mass)
        weighted_utility.append(mass * (utility + survival * float(values[i+1][b])))
        a = b
    base_acceptance = float(sum(prefix_mass))
    mixed_moment = float(sum(weighted_utility))
    covariance = mixed_moment - base_acceptance * mu
    mean_acceptance = {}
    survival_curves = {}
    for strength in strengths:
        curve = [(p + strength * pu) / (1.0 + strength * mu)
                 for p, pu in zip(prefix_mass, weighted_utility)]
        curve += [0.0] * (min(len(q), len(greedy)) - len(curve))
        mean_acceptance[str(strength)] = float(sum(curve))
        survival_curves[str(strength)] = curve
    return {'mean_utility_q': mu, 'mean_acceptance_q': base_acceptance,
            'covariance_acceptance_utility': covariance,
            'mean_acceptance_doob': mean_acceptance,
            'survival_doob': survival_curves,
            'coverage_oracle': len(prefix_mass)}
