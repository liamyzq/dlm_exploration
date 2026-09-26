"""Start an isolated SD worker on Nebula and record its actual provenance."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[5, 6, 7, 8])
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--config', default='')
    parser.add_argument('--output', required=True)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(), 'Use a committed frozen worktree.'
    storage = '/home/mlw0719/cola_dlm_exploration_storage'
    argv = ['env', f'PYTHONPATH={os.getcwd()}', f'HF_HOME={storage}/hf_cache',
            'HF_HUB_DISABLE_PROGRESS_BARS=1', 'VLLM_USE_V2_MODEL_RUNNER=1',
            f'CUDA_VISIBLE_DEVICES={args.gpu if args.gpu is not None else ""}',
            f'{storage}/venvs/idea003-vllm030/bin/python'] + command
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    record = dict(event='submitted', experiment=args.experiment, machine='nebula', scheduler='direct',
                  commit=commit, config=args.config, command=shlex.join(argv),
                  recorded_at=datetime.now(timezone.utc).isoformat(), artifact_dir=str(out),
                  gpu=args.gpu, working_directory=os.getcwd())
    shell = shlex.join(argv)+'; result=$?; printf "%s\\n" "$result" > '+shlex.quote(str(out/'exit.txt'))+'; exit "$result"'
    with (out/'log.txt').open('w') as log:
        process = subprocess.Popen(['bash', '-c', shell], stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    record['job_id'] = str(process.pid)
    (out/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    with Path(args.ledger).open('a') as handle:
        handle.write(json.dumps(record)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
