"""Launch only the exact completed package approved by the user."""
import json
from pathlib import Path
from contracts import sha

HERE=Path(__file__).resolve().parent


def require_approval():
    path=HERE/'approval.json'
    if not path.exists():raise RuntimeError('Awaiting the user\'s final approval; training has not started')
    value=json.loads(path.read_text())
    if value.get('approved') is not True:raise RuntimeError('This run has not been approved')
    for name in ('run-spec.json','plan.json','prepared.json'):
        if value.get('files',{}).get(name)!=sha(HERE/name):
            raise RuntimeError('Approved package changed: '+name)
    from runtime_state import runtime_hashes
    prepared=json.loads((HERE/'prepared.json').read_text())
    if prepared.get('runtime_sha256')!=runtime_hashes():raise RuntimeError('Approved executable code changed')
    return value
