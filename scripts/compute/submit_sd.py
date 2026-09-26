"""Start an isolated SD worker and record its actual provenance."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='', help='One physical index or comma-separated indices in 0-9.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--config', default='')
    parser.add_argument('--output', required=True)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.gpu and any(int(index) not in range(10) for index in args.gpu.split(',')):
        parser.error('The authorized physical GPU indices are 0-9.')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(), 'Use a committed frozen worktree.'
    storage = '/home/mlw0719/cola_dlm_exploration_storage'
    environment_bin = f'{storage}/venvs/idea003-vllm030/bin'
    argv = ['env', f'PYTHONPATH={os.getcwd()}', f'HF_HOME={storage}/hf_cache',
            f'PATH={environment_bin}:{os.environ["PATH"]}',
            'HF_HUB_DISABLE_PROGRESS_BARS=1', 'VLLM_USE_V2_MODEL_RUNNER=1',
            f'CUDA_VISIBLE_DEVICES={args.gpu}',
            f'{environment_bin}/python'] + command
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    record = dict(event='submitted', experiment=args.experiment, machine=socket.gethostname().split('.')[0], scheduler='direct',
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
