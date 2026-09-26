"""Exact fixed-lattice evaluation with prompt-clustered discovery/validation decisions."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from src.methods.prefix_utility import (
    accepted_prefix, full_path_optimum, greedy_statistics, native_greedy,
    prefix_dp, probabilities,
)


def load_graphs(ar_dir, native_dir):
    ar = {r['prompt_id']: r for r in map(json.loads, (Path(ar_dir)/'generations.jsonl').read_text().splitlines())}
    rows = list(map(json.loads, (Path(native_dir)/'generations.jsonl').read_text().splitlines()))
    result = []
    for row in rows:
        reference = ar[row['prompt_id']]
        assert row['prompt_tokens'] == reference['prompt_tokens']
        assert row['output_tokens'] == reference['output_tokens'], row['prompt_id']
        graphs = []
        if 'graph_file' in row:
            with np.load(Path(native_dir)/row['graph_file']) as archive:
                for j, (candidates, scores) in enumerate(zip(archive['candidate_ids'], archive['pair_scores'])):
                    offset = int(archive['sample_positions'][j, 0])-len(row['prompt_tokens'])
                    future = reference['output_tokens'][offset:offset+len(candidates)]
                    greedy = []
                    for i, token in enumerate(future):
                        where = np.flatnonzero(candidates[i] == token)
                        greedy.append(int(where[0]) if len(where) else -1)
                    q = probabilities([scores[0, :1]]+list(scores[1:]))
                    native = native_greedy(q)
                    assert [int(candidates[i,b]) for i,b in enumerate(native)] == archive['native_tokens'][j].tolist()
                    graphs.append(dict(q=q, greedy=greedy, remaining=len(reference['output_tokens'])-offset,
                                       round_index=int(archive['round_indices'][j])))
        result.append(dict(prompt_id=row['prompt_id'], domain=row['domain'], split=row['split'],
                           output_tokens=len(reference['output_tokens']),
                           native_rounds=row['captured_rounds'], graphs=graphs))
    return result


def fit_calibration(prompts, bounds):
    labels = []
    eligible = np.zeros(7, dtype=int)
    covered = np.zeros(7, dtype=int)
    for prompt in prompts:
        for graph in prompt['graphs']:
            a = 0
            for i, b in enumerate(graph['greedy']):
                eligible[i] += 1
                if b < 0:
                    break
                covered[i] += 1
                labels.append((np.log(graph['q'][i][a]), b))
                a = b
    assert labels and np.all(eligible > 0), 'Insufficient eligible discovery rows for the declared seven-depth fit.'

    def nll(log_temperature):
        temperature = np.exp(log_temperature)
        total = 0.0
        for log_q, b in labels:
            z = log_q/temperature
            maximum = z.max()
            total += maximum+np.log(np.exp(z-maximum).sum())-z[b]
        return total/len(labels)

    fit = minimize_scalar(nll, bounds=np.log(bounds), method='bounded', options={'xatol': 1e-7})
    assert fit.success, fit.message
    return dict(temperature=float(np.exp(fit.x)), coverage=(covered/eligible).tolist(),
                eligible=eligible.tolist(), covered=covered.tolist(), covered_labels=len(labels),
                nll=float(fit.fun), temperature_bounds=bounds, fit_split='discovery')


def evaluate(prompts, config, calibration=None):
    output = []
    for prompt in prompts:
        context_rows = []
        for graph in prompt['graphs']:
            q, greedy, remaining = graph['q'], graph['greedy'], graph['remaining']
            c = q if calibration is None else [row*coverage for row, coverage in zip(
                probabilities([np.log(row) for row in q], calibration['temperature']), calibration['coverage'])]
            native = native_greedy(q)
            dp, _ = prefix_dp(c)
            whole = full_path_optimum(q)
            exact = greedy_statistics(q, c, greedy, config['doob_strengths'])
            methods = {}

            def deterministic(name, path):
                acceptance = accepted_prefix(path, greedy)
                methods[name] = dict(acceptance=float(acceptance), yield_tokens=float(min(acceptance+1, remaining)),
                                     survival=[float(k <= acceptance) for k in range(1, len(q)+1)])

            deterministic('native', native)
            deterministic('prefix_dp', dp)
            deterministic('full_path', whole)
            oracle = exact['coverage_oracle']
            methods['oracle'] = dict(acceptance=float(oracle), yield_tokens=float(min(oracle+1, remaining)),
                                     survival=[float(k <= oracle) for k in range(1, len(q)+1)])

            def stochastic(name, acceptance, curve):
                terminal_mass = curve[remaining-1] if remaining <= len(curve) else 0.0
                methods[name] = dict(acceptance=acceptance, yield_tokens=1+acceptance-terminal_mass,
                                     survival=curve+[0.0]*(len(q)-len(curve)))

            for strength in config['doob_strengths']:
                key = str(strength)
                stochastic(f'doob_{strength:g}', exact['mean_acceptance_doob'][key], exact['survival_doob'][key])
            for temperature in config['temperatures']:
                tempered = probabilities([np.log(row) for row in q], temperature)
                control = greedy_statistics(tempered, tempered, greedy, [0.0])
                stochastic(f'temperature_{temperature:g}', control['mean_acceptance_q'], control['survival_doob']['0.0'])
            delta = methods['prefix_dp']['acceptance']-methods['native']['acceptance']
            context_rows.append(dict(round_index=graph['round_index'], effective_horizon=len(greedy),
                                     methods=methods, covariance=exact['covariance_acceptance_utility'],
                                     dp_changes_path=dp != native, dp_delta=delta))
        names = (list(context_rows[0]['methods']) if context_rows else
                 ['native', 'prefix_dp', 'full_path', 'oracle']+
                 [f'doob_{x:g}' for x in config['doob_strengths']]+
                 [f'temperature_{x:g}' for x in config['temperatures']])
        means = {name: {metric: float(np.mean([row['methods'][name][metric] for row in context_rows]))
                        if context_rows else 0.0 for metric in ['acceptance', 'yield_tokens']} for name in names}
        output.append(dict(prompt_id=prompt['prompt_id'], domain=prompt['domain'], split=prompt['split'],
                           native_rounds=prompt['native_rounds'], output_tokens=prompt['output_tokens'],
                           contexts=context_rows, means=means))
    return output


def summarize(rows, config):
    domains = list(dict.fromkeys(row['domain'] for row in rows))
    names = list(rows[0]['means'])
    rng = np.random.default_rng(config['bootstrap_seed'])
    arrays = {domain: np.array([[[r['means'][name][metric] for metric in ['acceptance', 'yield_tokens']]
                                for name in names] for r in rows if r['domain'] == domain]) for domain in domains}
    estimate = np.mean([values.mean(axis=0) for values in arrays.values()], axis=0)
    boot = np.mean([values[rng.integers(len(values), size=(config['bootstrap_samples'], len(values)))].mean(axis=1)
                    for values in arrays.values()], axis=0)
    native = names.index('native')
    alpha = (1-config['confidence'])/2
    comparisons = {}
    for i, name in enumerate(names):
        delta = boot[:, i, 0]-boot[:, native, 0]
        ratio = (1+estimate[i, 0])/(1+estimate[native, 0])-1
        interval = np.quantile(delta, [alpha, 1-alpha]).tolist()
        comparisons[name] = dict(mean_acceptance=float(estimate[i, 0]), mean_yield=float(estimate[i, 1]),
                                 acceptance_gain=float(estimate[i, 0]-estimate[native, 0]),
                                 acceptance_gain_ci=interval, plus_one_gain=float(ratio),
                                 yield_gain=float(estimate[i, 1]/estimate[native, 1]-1) if estimate[native, 1] else None,
                                 passes_gate=bool(interval[0]>0 and ratio>=config['engineering_gain']))
    contexts = [c for r in rows for c in r['contexts']]
    changed = [c['dp_delta'] for c in contexts if c['dp_changes_path']]
    return dict(prompts=len(rows), retained_graphs=len(contexts), methods=comparisons,
                per_domain={domain: {name: dict(mean_acceptance=float(values[:,i,0].mean()),
                                               mean_yield=float(values[:,i,1].mean()))
                                     for i,name in enumerate(names)} for domain,values in arrays.items()},
                prefix_survival={name: np.mean([np.mean([c['methods'][name]['survival'] for c in r['contexts']],axis=0)
                                                if r['contexts'] else np.zeros(7) for r in rows],axis=0).tolist()
                                 for name in names},
                dp_changed_contexts=len(changed), dp_changed_delta_positive=sum(x>0 for x in changed),
                dp_changed_delta_zero=sum(x==0 for x in changed), dp_changed_delta_negative=sum(x<0 for x in changed),
                dp_changed_delta_mean=float(np.mean(changed)) if changed else 0.0,
                mean_covariance=float(np.mean([np.mean([c['covariance'] for c in r['contexts']])
                                               if r['contexts'] else 0 for r in rows])))


def lock_and_validate(discovery, validation):
    doob = max((name for name in discovery['methods'] if name.startswith('doob_')),
               key=lambda name: discovery['methods'][name]['mean_acceptance'])
    temperature = max((name for name in discovery['methods'] if name.startswith('temperature_')),
                      key=lambda name: discovery['methods'][name]['mean_acceptance'])
    survivors = [name for name in ['prefix_dp', doob] if validation['methods'][name]['passes_gate']]
    return dict(locked_doob=doob, locked_temperature=temperature, survivors=survivors,
                selected_on='discovery', assessed_on='validation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    for split in ['discovery', 'validation']:
        for mode in ['ar', 'native']:
            parser.add_argument(f'--{split}-{mode}', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prompts = {split: load_graphs(getattr(args,f'{split}_ar'), getattr(args,f'{split}_native'))
               for split in ['discovery','validation']}
    assert all(len(rows)==150 for rows in prompts.values()), 'The decision uses the declared complete development cohort.'
    result = dict(config=config)

    def assess(label, calibration=None):
        summaries = {}
        for split in ['discovery', 'validation']:
            rows = evaluate(prompts[split], config, calibration)
            (out/f'{label}-{split}-contexts.json').write_text(json.dumps(rows)+'\n')
            summaries[split] = summarize(rows, config)
        summaries['selection'] = lock_and_validate(summaries['discovery'], summaries['validation'])
        result[label] = summaries
        return summaries

    initial = assess('uncalibrated')
    headroom = initial['validation']['methods']['oracle']['plus_one_gain'] >= config['engineering_gain']
    if initial['selection']['survivors']:
        result['decision'] = 'advance_uncalibrated'
    elif not headroom:
        result['decision'] = 'no_go_insufficient_candidate_headroom'
    else:
        fit = fit_calibration(prompts['discovery'], config['calibration']['temperature_bounds'])
        result['calibration'] = fit
        calibrated = assess('calibrated', fit)
        result['decision'] = ('advance_calibrated' if calibrated['selection']['survivors']
                              else 'no_go_proxy_selection_under_fixed_budget')
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
