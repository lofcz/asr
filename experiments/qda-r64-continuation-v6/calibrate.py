"""Prepare CPU token plans, reuse held-out features, run one disposable LR sweep."""
import json, os, pathlib, subprocess, time
from contracts import atomic_json
HERE=pathlib.Path(__file__).resolve().parent
PY='/data/work/asr-pilot/venv/bin/python'


def run(cmd, stage):
    atomic_json(HERE/'experiment-status.json',{'stage':stage,'running':True,'updated_at':time.time()})
    subprocess.run(cmd,cwd=HERE,check=True)


def main():
    if not (HERE/'plan.json').exists():
        run([PY,'-u','prepare.py','inputs'],'preparing_remaining_audio')
    run([PY,'-u','prepare.py','features','--rank','0','--world','1','--validation-only'],'reusing_fixed_validation')
    if not (HERE/'result/completed.json').exists():
        run([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','lr_find.py'], 'continuation_lr_range')
    run([PY,'-u','analyze_lr.py'],'analyzing_continuation_lr')
    atomic_json(HERE/'experiment-status.json',{'stage':'lr_range_complete','running':False,'updated_at':time.time()})


if __name__=='__main__':main()
