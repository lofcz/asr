"""Retire old full-run checkpoints only after verified durable SSD backup."""
import json, pathlib, shutil
from contracts import sha, atomic_json
from gates import demand


def prune(run):
    run=pathlib.Path(run).resolve()
    demand(run.name=='r64-full' and run.parent.name=='runs','Unexpected checkpoint namespace')
    checkpoints=sorted(p for p in run.glob('checkpoint-*') if p.is_dir() and '.building' not in p.name)
    for path in checkpoints[:-2]:
        receipt=run/'backup-receipts'/(path.name+'.json')
        if not receipt.exists():continue
        saved=json.loads(receipt.read_text())
        if saved.get('source')!=str(path) or not saved.get('destination'):continue
        files={p.name:sha(p) for p in path.iterdir() if p.is_file()}
        demand(files==saved.get('files') and 'state.pt' in files and 'adapter_model.safetensors' in files,
               'Backup receipt is stale; preserve checkpoint')
        # Source and destination are on different machines; all original files are
        # retained on the SSD. Two newest complete checkpoints always remain here.
        shutil.rmtree(path)
        atomic_json(run/'archived'/(path.name+'.json'),saved)
