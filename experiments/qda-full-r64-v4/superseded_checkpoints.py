"""Retire intermediate server snapshots after two newer full offsite backups."""
import argparse,json,pathlib,re,shutil,time
from contracts import atomic_json,sha


def prune_superseded(run,anchors):
    run=pathlib.Path(run).resolve();anchors=sorted(set(anchors))
    if run.name!='r64-full' or run.parent.name!='runs':raise RuntimeError('Unexpected checkpoint namespace')
    if len(anchors)!=2 or not all(re.fullmatch(r'checkpoint-\d{6}',n) for n in anchors):raise RuntimeError('Two full backup anchors are required')
    evidence={}
    for name in anchors:
        path=run/'backup-receipts'/(name+'.json');r=json.loads(path.read_text())
        if r.get('source')!=str(run/name) or not r.get('destination') or not r.get('verified_at') or r.get('state_retired_at'):
            raise RuntimeError('Anchor does not have a full verified backup')
        if not {'state.pt','adapter_model.safetensors','adapter_config.json'}<=r['files'].keys():raise RuntimeError('Incomplete anchor')
        evidence[name]=sha(path)
    checkpoints=sorted(p for p in run.glob('checkpoint-*') if p.is_dir() and re.fullmatch(r'checkpoint-\d{6}',p.name))
    protected={p.name for p in checkpoints[-2:]}|{'checkpoint-000005'}|set(anchors)
    for path in [*run.glob('evaluation-*.json'),run/'latest.json']:
        if path.exists():protected.add(json.loads(path.read_text())['checkpoint'])
    retired=[]
    for path in checkpoints:
        if path.name>=anchors[0] or path.name in protected:continue
        # Existing fully backed snapshots are handled by the pinned trainer.
        if (run/'backup-receipts'/(path.name+'.json')).exists():continue
        atomic_json(run/'superseded'/(path.name+'.json'),{
            'checkpoint':path.name,'replacement_full_backups':evidence,'retired_at':time.time(),
            'reason':'Intermediate snapshot superseded by two newer checksum-verified full SSD checkpoints; first, latest and evaluation milestones protected'})
        shutil.rmtree(path);retired.append(path.name)
    return retired


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('anchors',nargs=2);a=p.parse_args()
    print(json.dumps({'retired':prune_superseded(pathlib.Path(__file__).resolve().parent/'runs/r64-full',a.anchors)}))
