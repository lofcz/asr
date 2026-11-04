"""Bound local optimizer-state storage even when the offsite link is unavailable.

Adapter weights are never removed here. Protect the first checkpoint, latest
two full states, evaluation/final checkpoints and active backup transfer leases.
The trainer and its immutable plan are unchanged.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import time
from contracts import atomic_json, sha

HERE = Path(__file__).resolve().parent
PATTERN = re.compile(r'checkpoint-\d{6}')


def choose(run, now=None):
    now = time.time() if now is None else now
    paths = sorted(p for p in run.glob('checkpoint-*')
                   if PATTERN.fullmatch(p.name) and p.is_dir() and not p.is_symlink())
    full = [p for p in paths if (p / 'state.pt').is_file()]
    if len(full) < 3:
        return [], full[-2:], set()
    anchors = full[-2:]
    protected = {p.name for p in anchors} | {'checkpoint-000005', paths[0].name}
    for p in [run / 'latest.json', *run.glob('evaluation-*.json')]:
        if p.exists():
            record = json.loads(p.read_text())
            if 'checkpoint' in record:
                protected.add(record['checkpoint'])
    for p in (run / 'backup-in-progress').glob('*.json'):
        lease = json.loads(p.read_text())
        if lease.get('expires_at', 0) > now:
            protected.add(lease['checkpoint'])
    candidates = [p for p in full if p.name < anchors[0].name and p.name not in protected
                 ]
    return candidates, anchors, protected


def verify_resume(path):
    """Validate one complete checkpoint before it can replace older full states."""
    import torch
    state_path = path / 'state.pt'
    adapter = path / 'adapter_model.safetensors'
    receipt_path = path.parent / 'local-resume-receipts' / (path.name + '.json')
    fingerprints = {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                    for p in (state_path, adapter, path / 'adapter_config.json')}
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if saved.get('fingerprints') == fingerprints:
            return saved
    state = torch.load(state_path, map_location='cpu', mmap=True, weights_only=False)
    if state['step'] != int(path.name.rsplit('-', 1)[1]):
        raise RuntimeError('Checkpoint step mismatch')
    if not all(k in state for k in ('optimizer', 'scheduler', 'rng', 'config', 'tokens_seen')):
        raise RuntimeError('Missing resume state')
    if not state['optimizer'].get('state') or len(state['rng']) != 2:
        raise RuntimeError('Incomplete optimizer or RNG state')
    from future_data_repair import compatible_resume, load_repair, validate_resume_step
    current = json.loads((path.parent / 'config.json').read_text())
    repair = load_repair() if state['config'] != current else None
    if not compatible_resume(state['config'], current, repair):
        raise RuntimeError('Checkpoint configuration mismatch')
    validate_resume_step(state['step'], state['config'], repair)
    digests = {name: sha(path / name) for name in fingerprints}
    if state['adapter_sha256'] != digests['adapter_model.safetensors']:
        raise RuntimeError('Checkpoint adapter mismatch')
    del state
    for name in fingerprints:
        with (path / name).open('rb') as f:
            os.fsync(f.fileno())
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    receipt = {'checkpoint': path.name, 'verified_at': time.time(), 'files': digests,
               'fingerprints': fingerprints, 'optimizer_scheduler_rng_verified': True}
    atomic_json(receipt_path, receipt)
    return receipt


def retain(run):
    run = Path(run).resolve()
    if run.name != 'r64-full' or run.parent.name != 'runs':
        raise RuntimeError('Unexpected checkpoint namespace')
    with (run / '.disk-retention.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = shutil.disk_usage(run).free
        candidates, anchors, protected = choose(run)
        evidence = {p.name: verify_resume(p) for p in anchors} if candidates else {}
        retired = []
        for path in candidates:
            # Re-read dynamic protections immediately before the mutation.
            current, _, _ = choose(run)
            if path not in current:
                continue
            state = path / 'state.pt'
            if state.is_symlink():
                raise RuntimeError('Refusing symlink state')
            record = {'checkpoint': path.name, 'state_sha256': sha(state),
                      'state_bytes': state.stat().st_size,
                      'adapter_sha256': sha(path / 'adapter_model.safetensors'),
                      'replacement_full_checkpoints': {k: v['files'] for k, v in evidence.items()},
                      'retired_at': time.time(), 'adapter_preserved': True,
                      'reason': 'Intermediate optimizer state; two newer validated local full checkpoints retained'}
            atomic_json(run / 'local-state-retention' / (path.name + '.json'), record)
            state.unlink()
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            retired.append(path.name)
        result = {'checked_at': time.time(), 'retired_states': retired,
                  'protected_checkpoints': sorted(protected),
                  'free_gib': shutil.disk_usage(run).free / 2**30,
                  'freed_gib': (shutil.disk_usage(run).free - before) / 2**30,
                  'policy': 'Retain every remaining adapter; first, latest two full states, evaluation checkpoints and backup leases protected'}
        atomic_json(run / 'disk-retention-status.json', result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=HERE / 'runs/r64-full')
    args = parser.parse_args()
    print(json.dumps(retain(args.run)), flush=True)
