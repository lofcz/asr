"""Keep first, milestone and latest two verified full checkpoints on the SSD."""
import json, pathlib, re, time
from contracts import atomic_json, sha


def retain(run, remote_names):
    run=pathlib.Path(run).resolve()
    if run.name!='r64-full' or run.parent.name!='runs':raise RuntimeError('Unexpected retention namespace')
    checkpoints=sorted(p for p in run.glob('checkpoint-*') if p.is_dir() and re.fullmatch(r'checkpoint-\d+',p.name))
    if len(checkpoints)<3:return
    protected={p.name for p in checkpoints[-2:]}|{checkpoints[0].name}
    for path in [*run.glob('evaluation-*.json'),run/'latest.json']:
        if path.exists():protected.add(json.loads(path.read_text())['checkpoint'])
    candidates=[p for p in checkpoints if p.name not in protected and p.name not in remote_names and (p/'state.pt').exists()]
    if candidates:
        # Verify two newer full, durable copies before retiring any old state.
        for path in checkpoints[-2:]:
            receipt=json.loads((run/'backup-receipts'/(path.name+'.json')).read_text())
            if receipt['destination']!=str(path) or receipt.get('state_retired_at'):raise RuntimeError('Latest backup is not resumable')
            for name,digest in receipt['files'].items():
                if sha(path/name)!=digest:raise RuntimeError('Latest backup checksum mismatch')
            if 'state.pt' not in receipt['files']:raise RuntimeError('Latest backup lacks optimizer state')
        for path in candidates:
            receipt_path=run/'backup-receipts'/(path.name+'.json');receipt=json.loads(receipt_path.read_text())
            if receipt['destination']!=str(path):raise RuntimeError('Old backup destination mismatch')
            if sha(path/'state.pt')!=receipt['files']['state.pt']:raise RuntimeError('Old optimizer state checksum mismatch')
            if sha(path/'adapter_model.safetensors')!=receipt['files']['adapter_model.safetensors']:raise RuntimeError('Retained adapter checksum mismatch')
            # Record the intentional retirement before unlinking; reruns are safe.
            receipt['state_retired_at']=time.time()
            receipt['retained_files']={}
            receipt['checkpoint_retired_at']=time.time()
            receipt['retention_reason']='Two newer verified full backups; retain first checkpoint, evaluation milestones and latest two full backups'
            atomic_json(receipt_path,receipt)
            import shutil
            shutil.rmtree(path)
    atomic_json(run/'backup-retention.json',{
        'policy':'Keep first checkpoint, latest two verified full backups and all evaluation milestones; retire other snapshots only after two newer full backups are verified',
        'full_resume_checkpoints':[p.name for p in checkpoints if p.exists() and (p/'state.pt').exists()],
        'adapter_only_checkpoints':[p.name for p in checkpoints if p.exists() and not (p/'state.pt').exists()],
        'updated_at':time.time()})
