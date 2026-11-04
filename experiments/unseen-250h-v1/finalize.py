"""Finish CPU reports once both GPU workers and their strict scorer finish."""
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def main():
    deadline = time.monotonic() + 7200
    while True:
        finished = [HERE / v / 'finished.json' for v in ('vanilla', 'finetuned')]
        report = HERE / 'summary.json'
        if all(p.exists() for p in finished) and report.exists():
            if report.stat().st_mtime >= max(p.stat().st_mtime for p in finished):
                break
        state = subprocess.run(['systemctl', '--user', 'is-active', 'qda250-unseen-eval.service'],
                               text=True, capture_output=True).stdout.strip()
        if state not in ('active', 'activating'):
            raise RuntimeError(f'Evaluation stopped before all reports completed: {state}')
        if time.monotonic() >= deadline:
            raise TimeoutError('Evaluation did not finish within two hours')
        time.sleep(15)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2'}
    for command in (['status.py'], ['analyze.py'], ['diagnostics.py', '--summary', 'analysis.json']):
        subprocess.run([sys.executable, *command], cwd=HERE, env=env, check=True)
    analysis = json.loads((HERE / 'analysis.json').read_text())
    assert analysis['status'] == 'complete'
    assert pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16/paused-by-user.json').exists()
    print(json.dumps({'event': 'analysis_complete', 'recordings': analysis['recordings'],
                      'raw_valid_outputs': analysis['raw_valid_outputs'], 'training_remains_paused': True}), flush=True)


if __name__ == '__main__':
    main()
