"""Durable SSD mirror of preparation, calibration, evaluations and checkpoints."""
import hashlib, json, os, pathlib, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
HERE=pathlib.Path(__file__).resolve().parent
REMOTE='/data/work/asr-experiments/qda-full-continuous-v8'
CAL=HERE
DATA=ROOT/'datasets/releases/qda-full-continuous-v8'
SSH=['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
     '-o','ControlMaster=no','-o','ControlPath=/home/lofcz/.cache/asr-ssh/mlserver-persistent']
os.environ['RSYNC_RSH']=' '.join(SSH)


def command(args,**kw):
    if args[0]=='ssh':args=SSH+args[1:]
    return subprocess.run(args,check=True,timeout=kw.pop('timeout',600),**kw)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('w') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
    temp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)


def cycle():
    # Prioritize current resume states. An unrelated metadata transfer failure
    # must not prevent checkpoint backup; live metrics have their own service.
    run=HERE/'runs/r64-full'
    for latest in (run/'latest.json',):
        if not latest.exists():continue
        import shlex
        listing="import json,pathlib; print(json.dumps([p.name for p in pathlib.Path("+repr(REMOTE+'/runs/r64-full')+").glob('checkpoint-*') if p.is_dir() and '.building' not in p.name and (p/'state.pt').is_file()]))"
        remote_dirs=json.loads(command(['ssh','mlserver','python3 -c '+shlex.quote(listing)],capture_output=True,text=True).stdout)
        remote_dirs=sorted(remote_dirs)
        milestones={json.loads(p.read_text())['checkpoint'] for p in run.glob('evaluation-*.json')}
        wanted=set(remote_dirs[-1:])|milestones
        if not list((run/'backup-receipts').glob('*.json')):wanted.update(remote_dirs[:1])
        # Always copy milestone checkpoints; let newer full states supersede
        # intermediate snapshots instead of accumulating an unbounded queue.
        for name in [n for n in remote_dirs if n in wanted]:
            if not name.startswith('checkpoint-') or not name[11:].isdigit():raise RuntimeError('Unexpected checkpoint name')
            dest=run/name;receipt=run/'backup-receipts'/(name+'.json')
            if receipt.exists():continue
            temp=run/(name+'.copying');temp.mkdir(parents=True,exist_ok=True)
            source=REMOTE+'/runs/r64-full/'+name
            # Atomically pin a transfer against the independent server retention
            # worker. Expiry prevents a dead copier from pinning states forever.
            lease_script='\n'.join([
                'import pathlib,json,time,fcntl,os',
                'root=pathlib.Path('+repr(REMOTE+'/runs/r64-full')+')',
                'lock=(root/".disk-retention.lock").open("a")',
                'fcntl.flock(lock,fcntl.LOCK_EX)',
                'name='+repr(name),
                'assert (root/name/"state.pt").is_file(), "Full checkpoint was already retired"',
                'folder=root/"backup-in-progress"; folder.mkdir(exist_ok=True)',
                'temp=folder/(name+".tmp")',
                'temp.write_text(json.dumps(dict(checkpoint=name,expires_at=time.time()+3600)))',
                'os.replace(temp,folder/(name+".json"))'])
            command(['ssh','mlserver','python3 -c '+shlex.quote(lease_script)],capture_output=True,text=True)
            atomic(HERE/'sync-status.json',{'status':'copying','checkpoint':name,'started_at':time.time()})
            command(['rsync','-az','--timeout=120','--partial','--partial-dir=.rsync-partial','mlserver:'+source+'/',str(temp)+'/'],timeout=3600)
            files={p.name:digest(p) for p in temp.iterdir() if p.is_file()}
            if not {'state.pt','adapter_model.safetensors','adapter_config.json'}<=files.keys():raise RuntimeError('Incomplete checkpoint transfer')
            # Check remote checksums independently, rather than trusting transfer success.
            script='import pathlib,hashlib,json; p=pathlib.Path('+repr(source)+'); print(json.dumps({f.name:hashlib.file_digest(f.open("rb"),"sha256").hexdigest() for f in p.iterdir() if f.is_file()}))'
            import shlex
            remote=json.loads(command(['ssh','mlserver','python3 -c '+shlex.quote(script)],capture_output=True,text=True).stdout)
            if files!=remote:raise RuntimeError('Checkpoint checksum mismatch')
            for p in temp.iterdir():
                if p.is_file():
                    with p.open('rb') as f:os.fsync(f.fileno())
            if dest.exists():
                if {p.name:digest(p) for p in dest.iterdir() if p.is_file()}!=files:
                    raise RuntimeError('Existing checkpoint differs; inspect before replacing')
                import shutil
                shutil.rmtree(temp)
            else:temp.replace(dest)
            atomic(receipt,{'source':source,'destination':str(dest),'files':files,'verified_at':time.time()})
            command(['ssh','mlserver','mkdir','-p',REMOTE+'/runs/r64-full/backup-receipts'])
            command(['rsync','-az',str(receipt),'mlserver:'+REMOTE+'/runs/r64-full/backup-receipts/'])
            release='import pathlib; pathlib.Path('+repr(REMOTE+'/runs/r64-full/backup-in-progress/'+name+'.json')+').unlink(missing_ok=True)'
            command(['ssh','mlserver','python3 -c '+shlex.quote(release)],capture_output=True,text=True)
            break  # Re-list the newest checkpoint after each long transfer; avoid stale queued states.
        from checkpoint_retention import retain
        retain(run,set(remote_dirs))
        if (run/'backup-retention.json').exists():
            command(['rsync','-az',str(run/'backup-receipts')+'/', 'mlserver:'+REMOTE+'/runs/r64-full/backup-receipts/'])
            command(['rsync','-az',str(run/'backup-retention.json'),'mlserver:'+REMOTE+'/runs/r64-full/'])
        complete=sorted(p for p in run.glob('checkpoint-*') if p.is_dir() and (p/'state.pt').exists() and (run/'backup-receipts'/(p.name+'.json')).exists())
        if len(complete)>=2:
            anchors=complete[-2:]
            for path in anchors:
                receipt=json.loads((run/'backup-receipts'/(path.name+'.json')).read_text())
                if receipt.get('state_retired_at') or receipt['destination']!=str(path):raise RuntimeError('Backup anchor is not resumable')
                if any(digest(path/name)!=expected for name,expected in receipt['files'].items()):raise RuntimeError('Anchor checksum mismatch; preserve server snapshots')
            pass  # New trainer prunes only within the isolated v8 run namespace.
    # The full dataset and LR calibration are immutable; hourly mirroring is
    # sufficient after their first completed copy.
    archived=HERE/'archive-sync-status.json'
    previous=json.loads(archived.read_text()) if archived.exists() else {}
    if time.time()-previous.get('mirrored_at',0)>3600:
        DATA.mkdir(parents=True,exist_ok=True)
        command(['rsync','-az','--timeout=120','--exclude=.building','--exclude=.curation.lock','--exclude=review-audio',
                 'mlserver:/data/work/asr-datasets/qda-full-continuous-v8/',str(DATA)+'/'])
        command(['rsync','-az','--timeout=120','--exclude=cache*/','--exclude=checkpoint-*/','--exclude=wandb/',
                 '--exclude=backup-receipts/','--exclude=backup-retention.json',
                 '--include=*/','--include=*.json','--include=*.jsonl','--include=output.txt','--include=*.png','--include=*.svg',
                 '--exclude=*','mlserver:'+REMOTE+'/',str(HERE)+'/'])
        atomic(archived,{'mirrored_at':time.time(),'status':'ok'})


while True:
    try:
        cycle()
        atomic(HERE/'sync-status.json',{'mirrored_at':time.time(),'status':'ok'})
    except Exception as e:
        message=type(e).__name__+': '+str(e);print(message,flush=True)
        previous=json.loads((HERE/'sync-status.json').read_text()) if (HERE/'sync-status.json').exists() else {}
        atomic(HERE/'sync-status.json',{'mirrored_at':previous.get('mirrored_at'),'status':'error','error':message,'failed_at':time.time()})
    time.sleep(30)
