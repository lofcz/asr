"""Keep two verified resume checkpoints and archive milestones to the SSD."""
import fcntl
import json
from pathlib import Path
import re
import shutil
import time
from contracts import atomic_json,sha
from disk_retention import verify_resume

HERE=Path(__file__).resolve().parent


def prune(run):
    run=Path(run).resolve()
    if run!=HERE/'runs/r64-full':raise RuntimeError('Wrong run namespace for retention')
    with (run/'.disk-retention.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        paths=sorted(p for p in run.glob('checkpoint-*') if re.fullmatch(r'checkpoint-\d{6}',p.name) and p.is_dir() and not p.is_symlink())
        full=[p for p in paths if (p/'state.pt').is_file()]
        if len(full)<2:return
        anchors=full[-2:]
        proof={p.name:verify_resume(p) for p in anchors}
        milestones={json.loads(p.read_text())['checkpoint'] for p in run.glob('evaluation-*.json')}
        pending=run/'evaluation-due.json'
        protected={p.name for p in anchors}
        if pending.exists():protected.add(json.loads(pending.read_text())['checkpoint'])
        for p in (run/'backup-in-progress').glob('*.json'):
            lease=json.loads(p.read_text())
            if lease.get('expires_at',0)>time.time():protected.add(lease['checkpoint'])
        for path in paths:
            if path.name in protected or path.name>=anchors[0].name:continue
            archive=run/'backup-receipts'/(path.name+'.json')
            must_archive=path.name=='checkpoint-000005' or path.name in milestones
            backup=json.loads(archive.read_text()) if archive.exists() else None
            if must_archive and (not backup or backup.get('state_retired_at')):continue
            if backup:
                if (backup.get('source')!=str(path) or not backup.get('destination') or not backup.get('verified_at')
                        or not {'state.pt','adapter_model.safetensors','adapter_config.json'}<=backup.get('files',{}).keys()):
                    raise RuntimeError('Incomplete checkpoint backup receipt')
                for name,digest in backup['files'].items():
                    if sha(path/name)!=digest:raise RuntimeError('Archived checkpoint hash differs')
            adapter_hash=sha(path/'adapter_model.safetensors')
            atomic_json(run/'retention'/(path.name+'.json'),{
                'checkpoint':path.name,'adapter_sha256':adapter_hash,'retired_at':time.time(),
                'replacement_resume_checkpoints':{k:v['files'] for k,v in proof.items()},
                'verified_backup':backup,'policy':'Latest two full states; first and 50h milestones require a verified SSD copy before local retirement'})
            # Comparison pages use saved predictions. Their temporary inference
            # adapter is a hard link and can be retired with its archived source.
            for marker in (HERE/'markers').glob('r64-full-*'):
                stamp=marker/'completed-v3.json'
                if stamp.exists() and json.loads(stamp.read_text()).get('adapter_sha256')==adapter_hash:
                    adapter=marker/'adapter'
                    if adapter.is_dir() and not adapter.is_symlink():shutil.rmtree(adapter)
            shutil.rmtree(path)
        # Inputs, authored labels and raw audio are retained. Frozen features are
        # reproducible, so completed windows need not occupy checkpoint space.
        latest_step=int(anchors[-1].name.rsplit('-',1)[1])
        plan=json.loads((HERE/'plan.json').read_text())
        for pair in plan['probe_pairs'][:latest_step]:
            for row in pair:
                for suffix in ('.features.npz','.features.json'):
                    (HERE/'cache'/(row['window_id']+suffix)).unlink(missing_ok=True)
