"""Generate native/AR references and retain fixed-budget native candidate lattices."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np


def load_cohort(path, limit):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    domains = list(dict.fromkeys(r['domain'] for r in records))
    groups = [[r for r in records if r['domain'] == d] for d in domains]
    interleaved = [r for group in zip(*groups) for r in group]
    return interleaved[:limit] if limit else interleaved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['ar', 'native'], required=True)
    parser.add_argument('--split', choices=['discovery', 'validation'], required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    models = json.loads(Path(config['model_manifest']).read_text())
    data = Path(config['dataset_directory'])
    records = load_cohort(data / f'{args.split}.jsonl', args.limit)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'generations.jsonl').exists(), 'Use a fresh run directory.'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import torch
    import vllm
    tokenizer = AutoTokenizer.from_pretrained(models['target']['path'])
    inputs = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
              tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False) for r in records]
    assert all(len(ids)+config['max_tokens'] <= config['engine']['max_model_len'] for ids in inputs)
    engine = dict(config['engine'])
    if args.mode == 'native':
        engine['speculative_config'] = dict(config['speculative'])
        engine['speculative_config']['model'] = models['drafter']['path']
        engine['worker_extension_cls'] = 'src.models.dflash2_capture.CaptureExtension'
    run = dict(config=config, models=models, mode=args.mode, split=args.split, limit=args.limit,
               commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
               torch_version=torch.__version__, vllm_version=vllm.__version__,
               prompt_ids=[r['prompt_id'] for r in records], cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    (out/'resolved.json').write_text(json.dumps(run, indent=2)+'\n')
    llm = LLM(model=models['target']['path'], **engine)
    if args.mode == 'native':
        run['capture'] = llm.collective_rpc('install_lattice_capture')
        (out/'resolved.json').write_text(json.dumps(run, indent=2)+'\n')
    params = SamplingParams(temperature=0.0, max_tokens=config['max_tokens'], seed=config['seed'])
    # This run collects graphs; initialization and host-copy time are not speed evidence.
    for index, (record, input_ids) in enumerate(zip(records, inputs)):
        if args.mode == 'native':
            llm.collective_rpc('begin_lattice_capture')
        started = time.perf_counter()
        result = llm.generate([{'prompt_token_ids': input_ids}], params, use_tqdm=False)[0]
        elapsed = time.perf_counter()-started
        output = result.outputs[0]
        row = dict(prompt_id=record['prompt_id'], domain=record['domain'], split=record['split'],
                   prompt_tokens=input_ids, output_tokens=list(output.token_ids),
                   finish_reason=output.finish_reason, stop_reason=output.stop_reason,
                   collection_seconds=elapsed)
        if args.mode == 'native':
            raw_path = out/f'raw-{index:04d}.npz'
            capture = llm.collective_rpc('end_lattice_capture', kwargs={'path': str(raw_path)})[0]
            row['captured_rounds'] = capture['rounds']
            with np.load(raw_path) as raw:
                if capture['rounds']:
                    offsets = raw['sample_positions'][:, 0]-len(input_ids)
                    # Exclude post-terminal proposals from graph evaluation, but count their execution.
                    eligible = np.flatnonzero((offsets >= 1) & (offsets < len(output.token_ids)))
                    picks = (np.unique(np.rint(np.linspace(0, len(eligible)-1,
                             min(config['graphs_per_prompt'], len(eligible)))).astype(int))
                             if len(eligible) else np.array([], dtype=int))
                    keep = eligible[picks]
                    selected = {key: raw[key][keep] for key in raw.files}
                    selected['round_indices'] = keep
                    selected['all_offsets'] = offsets
                    selected['all_native_tokens'] = raw['native_tokens']
                    selected['all_anchor_ids'] = raw['anchor_id']
                    graph_path = out/f'graphs-{index:04d}.npz'
                    np.savez_compressed(graph_path, **selected)
                    row['graph_file'] = graph_path.name
                    row['retained_graphs'] = len(keep)
            raw_path.unlink()
        with (out/'generations.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        print(json.dumps({'index': index, 'prompt_id': row['prompt_id'],
                          'output_tokens': len(output.token_ids), 'seconds': elapsed}), flush=True)
    (out/'complete.json').write_text(json.dumps({'prompts': len(records), 'status': 'completed'})+'\n')


if __name__ == '__main__':
    main()
