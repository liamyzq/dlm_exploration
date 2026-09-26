"""Two bounded closure diagnostics using existing development graphs only."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import numpy as np

from src.methods.prefix_diagnostics import doob_family_oracles, first_divergence
from src.methods.prefix_utility import greedy_statistics, native_greedy, prefix_dp, probabilities
from src.tasks.prefix_offline import load_graphs


METRICS = ['native', 'q', 'lambda_infinity', 'doob_family', 'doob_native_fallback',
           'prefix_dp', 'native_dp_oracle', 'dp_positive_gain', 'dp_negative_loss']


def diagnose(prompts, previous, original_config, calibration=None):
    previous = {p['prompt_id']: p for p in previous}
    output, maximum_error = [], 0.0
    for prompt in prompts:
        prior = {c['round_index']: c for c in previous[prompt['prompt_id']]['contexts']}
        contexts = []
        for graph in prompt['graphs']:
            q, greedy = graph['q'], graph['greedy']
            c = q if calibration is None else [row*coverage for row, coverage in zip(
                probabilities([np.log(row) for row in q], calibration['temperature']), calibration['coverage'])]
            native, dp = native_greedy(q), prefix_dp(c)[0]
            divergence = first_divergence(native, dp, greedy)
            exact = greedy_statistics(q, c, greedy, original_config['doob_strengths'])
            bound = doob_family_oracles(exact, divergence['native_acceptance'])
            old = prior[graph['round_index']]
            errors = [abs(divergence['native_acceptance']-old['methods']['native']['acceptance']),
                      abs(divergence['dp_acceptance']-old['methods']['prefix_dp']['acceptance']),
                      abs(exact['covariance_acceptance_utility']-old['covariance'])]
            for strength in original_config['doob_strengths']:
                errors.append(abs(exact['mean_acceptance_doob'][str(strength)]-
                                  old['methods'][f'doob_{strength:g}']['acceptance']))
            maximum_error = max(maximum_error, *errors)
            delta = divergence['delta']
            category = divergence['category']
            assert ((category == 'dp_corrects_native' and delta > 0) or
                    (category == 'dp_breaks_native' and delta < 0) or
                    (category not in ['dp_corrects_native', 'dp_breaks_native'] and delta == 0))
            metrics = dict(native=divergence['native_acceptance'], q=bound['q_acceptance'],
                           lambda_infinity=bound['infinite_strength_acceptance'],
                           doob_family=bound['family_acceptance'],
                           doob_native_fallback=bound['fallback_acceptance'],
                           prefix_dp=divergence['dp_acceptance'],
                           native_dp_oracle=max(divergence['native_acceptance'], divergence['dp_acceptance']),
                           dp_positive_gain=max(delta, 0), dp_negative_loss=max(-delta, 0))
            contexts.append(dict(round_index=graph['round_index'], effective_horizon=len(greedy),
                                 native_path=native, dp_path=dp, greedy=greedy,
                                 divergence=divergence, doob=bound, metrics=metrics))
        assert len(contexts) == 8, 'This diagnostic uses the complete eight-graph-per-prompt collection.'
        means = {key: float(np.mean([c['metrics'][key] for c in contexts])) for key in METRICS}
        output.append(dict(prompt_id=prompt['prompt_id'], domain=prompt['domain'], contexts=contexts, means=means))
    assert maximum_error < 1e-12, ('Historical context results differ', maximum_error)
    return output, maximum_error


def summarize(rows, config):
    domains = list(dict.fromkeys(p['domain'] for p in rows))
    arrays = {domain: np.array([[p['means'][m] for m in METRICS] for p in rows if p['domain'] == domain])
              for domain in domains}
    rng = np.random.default_rng(config['bootstrap_seed'])
    estimate = np.mean([v.mean(axis=0) for v in arrays.values()], axis=0)
    boot = np.mean([v[rng.integers(len(v), size=(config['bootstrap_samples'], len(v)))].mean(axis=1)
                    for v in arrays.values()], axis=0)
    alpha = (1-config['confidence'])/2
    quantiles = [alpha, 1-alpha]
    values = {name: dict(mean=float(estimate[i]), ci=np.quantile(boot[:,i],quantiles).tolist())
              for i,name in enumerate(METRICS)}
    comparisons = {}
    for name in ['q','lambda_infinity','doob_family','doob_native_fallback','prefix_dp','native_dp_oracle']:
        i = METRICS.index(name)
        gain = estimate[i]-estimate[0]
        ratio = gain/(1+estimate[0])
        comparisons[name] = dict(mean_acceptance=float(estimate[i]), gain=float(gain),
                                gain_ci=np.quantile(boot[:,i]-boot[:,0],quantiles).tolist(),
                                plus_one_gain=float(ratio),
                                plus_one_gain_ci=np.quantile((boot[:,i]-boot[:,0])/(1+boot[:,0]),quantiles).tolist(),
                                reaches_three_percent_point_estimate=bool(ratio>=config['engineering_gain']))
    contexts = [c for p in rows for c in p['contexts']]
    categories = {}
    for name in ['same_path','shared_prefix_already_wrong','beyond_terminal',
                 'both_wrong_at_first_difference','dp_corrects_native','dp_breaks_native']:
        selected = [c for c in contexts if c['divergence']['category'] == name]
        deltas = [c['divergence']['delta'] for c in selected]
        categories[name] = dict(count=len(selected),
            first_difference_positions=dict(sorted(Counter(str(c['divergence']['first_difference_position'])
                                                           for c in selected if c['divergence']['first_difference_position'] is not None).items())),
            acceptance_delta_sum=sum(deltas),
            mean_delta_among_category=float(np.mean(deltas)) if deltas else None,
            contribution_per_context=float(sum(deltas)/len(contexts)),
            delta_distribution=dict(sorted(Counter(str(x) for x in deltas).items())))
    residual = (values['dp_positive_gain']['mean']-values['dp_negative_loss']['mean']-
                comparisons['prefix_dp']['gain'])
    assert abs(residual) < 1e-12
    return dict(prompts=len(rows), graphs=len(contexts), metrics=values, comparisons=comparisons,
                divergence=categories, gain_loss_identity_residual=residual,
                preferred_doob_endpoints=dict(Counter(c['doob']['preferred_endpoint'] for c in contexts)),
                zero_utility_contexts=sum(c['doob']['mean_utility']==0 for c in contexts),
                per_domain={domain: {m: float(v[:,i].mean()) for i,m in enumerate(METRICS)}
                            for domain,v in arrays.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    base = Path(config['collection_root'])
    original = Path(config['reference_analysis'])
    prior_summary = json.loads((original/'summary.json').read_text())
    result = dict(config=config, commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  scope='Post-hoc diagnostics on existing discovery and validation graphs. Future-label oracles are not deployable policies. No final-test inputs, new calibration, or model forwards.',
                  calibration_reused=prior_summary['calibration'])
    cohorts = {}
    for split in ['discovery','validation']:
        for mode in ['ar','native']:
            resolved = json.loads((base/f'003-dev-{split}-{mode}-v1/measurements/resolved.json').read_text())
            assert resolved['commit'] == config['collection_commit']
        cohorts[split] = load_graphs(base/f'003-dev-{split}-ar-v1/measurements',
                                    base/f'003-dev-{split}-native-v1/measurements')
        assert len(cohorts[split]) == 150
    for proxy in ['uncalibrated','calibrated']:
        result[proxy] = {}
        for split in ['discovery','validation']:
            previous = json.loads((original/f'{proxy}-{split}-contexts.json').read_text())
            rows, error = diagnose(cohorts[split], previous, prior_summary['config'],
                                   None if proxy == 'uncalibrated' else prior_summary['calibration'])
            (out/f'{proxy}-{split}-contexts.json').write_text(json.dumps(rows)+'\n')
            result[proxy][split] = summarize(rows, config)
            result[proxy][split]['maximum_historical_reproduction_error'] = error
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({proxy: {split: result[proxy][split]['comparisons']
                              for split in ['discovery','validation']}
                      for proxy in ['uncalibrated','calibrated']},indent=2))


if __name__ == '__main__':
    main()
