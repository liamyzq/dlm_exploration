"""Check extracted native walks, online round alignment, and exact AR token parity."""

import argparse
import json
from pathlib import Path
import numpy as np
from src.methods.prefix_utility import accepted_prefix, native_greedy, probabilities


def verify(ar_dir, native_dir):
    ar = {r['prompt_id']: r for r in map(json.loads, (Path(ar_dir)/'generations.jsonl').read_text().splitlines())}
    native = list(map(json.loads, (Path(native_dir)/'generations.jsonl').read_text().splitlines()))
    graphs, rounds, post_terminal_rounds, failures = 0, 0, 0, []
    for row in native:
        prompt_id = row['prompt_id']
        reference = ar[prompt_id]
        target = reference['output_tokens']
        if row['prompt_tokens'] != reference['prompt_tokens'] or row['output_tokens'] != target:
            failures.append({'prompt_id': prompt_id, 'failure': 'AR/native token mismatch'})
            continue
        if 'graph_file' not in row:
            continue
        with np.load(Path(native_dir)/row['graph_file']) as data:
            offsets = data['all_offsets'].astype(int)
            rounds += len(offsets)
            for i, offset in enumerate(offsets):
                if offset >= len(target):
                    # V2 drafts before the scheduler trims a sampled block at EOS/cap.
                    post_terminal_rounds += 1
                    continue
                if offset < 1:
                    failures.append({'prompt_id': prompt_id, 'round': i, 'failure': 'invalid committed offset'})
                    continue
                if int(data['all_anchor_ids'][i]) != target[offset-1]:
                    failures.append({'prompt_id': prompt_id, 'round': i, 'failure': 'anchor alignment'})
                accepted = accepted_prefix(data['all_native_tokens'][i], target[offset:offset+7])
                if i+1 < len(offsets) and offsets[i+1] < len(target) and offsets[i+1]-offset != accepted+1:
                    failures.append({'prompt_id': prompt_id, 'round': i, 'failure': 'accepted-length/next-boundary alignment',
                                     'offset': int(offset), 'next_offset': int(offsets[i+1]), 'accepted': accepted})
            for candidates, scores, actual in zip(data['candidate_ids'], data['pair_scores'], data['native_tokens']):
                graphs += 1
                if not np.array_equal(scores[0], np.broadcast_to(scores[0, :1], scores[0].shape)):
                    failures.append({'prompt_id': prompt_id, 'failure': 'first-layer anchor rows differ'})
                q = probabilities([scores[0, :1]]+list(scores[1:]))
                path = native_greedy(q)
                expected = [int(candidates[i, b]) for i, b in enumerate(path)]
                if expected != actual.tolist():
                    failures.append({'prompt_id': prompt_id, 'failure': 'reconstructed native walk mismatch'})
    return {'status': 'passed' if not failures else 'invalid', 'prompts': len(native),
            'retained_graphs': graphs, 'captured_rounds': rounds,
            'post_terminal_rounds': post_terminal_rounds, 'failures': failures,
            'scope': 'Native graph integration and exact greedy parity; no acceleration claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ar', required=True)
    parser.add_argument('--native', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = verify(args.ar, args.native)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
