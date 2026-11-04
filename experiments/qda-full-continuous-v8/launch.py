"""Explicit operator approval of the frozen package, then one training service."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from contracts import atomic_json,sha
from runtime_state import runtime_hashes

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--approve',action='store_true',help='Use only after the user has approved this complete package')
    args=p.parse_args()
    if not args.approve:raise SystemExit('Final user approval is required. No training was started.')
    ready=json.loads((HERE/'prepared.json').read_text())
    if ready['status']!='ready_for_user_approval' or ready['runtime_sha256']!=runtime_hashes():
        raise RuntimeError('Prepared package changed')
    atomic_json(HERE/'approval.json',{'approved':True,'approved_at':time.time(),
        'method':'Explicit user approval, applied by operator using launch.py --approve',
        'files':{name:sha(HERE/name) for name in ('run-spec.json','plan.json','prepared.json')}})
    from approval import require_approval
    require_approval()
    subprocess.run(['systemctl','--user','start','qda-v8-training.service'],check=True)


if __name__=='__main__':main()
