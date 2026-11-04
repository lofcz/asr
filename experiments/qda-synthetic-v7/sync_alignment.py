"""Publish committed native-timestamp training assets, then their manifest."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
from core import DATA,REPLICA,HERE,ROOT,read,save,put,sha,verified

SSH='ssh -o ConnectTimeout=15 -S /home/lofcz/.cache/asr-ssh/mlserver-persistent'
REMOTE='mlserver:/data/work/asr-synthetic-v7/'


def transfer(files,kind):
    if not files:return
    rel='transfer/'+kind+'.txt';put(rel,('\n'.join(files)+'\n').encode())
    result=subprocess.run(['rsync','-a','--timeout=90','-e',SSH,'--files-from='+str(DATA/rel),str(DATA)+'/',REMOTE],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(f'Transfer {kind} exited {result.returncode}: '+result.stderr[-2000:])


def sync():
    files=[name for name in ['plan.json','plan-revision.json','alignment.json','timestamp-policy.json',
        'content-catalog.json','domain-coverage.json','domain-coverage.md','README.md',
        'vocabulary/plan.json','vocabulary/catalog.json','vocabulary/validation-ready.json',
        'vocabulary/script-validation.json','vocabulary/structural-validation.json',
        'vocabulary/native-export-validation.json','vocabulary/script-quality-review.json',
        'concurrency.json','quota-status.json'] if (DATA/name).exists()]
    files.extend(str(p.relative_to(DATA)) for p in (DATA/'research').glob('*') if p.is_file())
    receipts=[];staging=0
    for path in sorted((DATA/'sessions').glob('*/receipt.json')):
        relative=path.parent.relative_to(DATA);staging+=1
        for name in ['audio.flac','labels.json','composition.json','spec.json','voices.json']:
            if (path.parent/name).exists():files.append(str(relative/name))
        for part in sorted((path.parent/'parts').glob('*/complete.json')):
            metadata=json.loads(part.read_text());files.append(str(part.relative_to(DATA)))
            script=metadata['script_path']
            if (DATA/script).exists():files.append(script)
            verification=part.parent/'vocabulary-verified.json'
            if verification.exists():files.append(str(verification.relative_to(DATA)))
        files.extend(str(p.relative_to(DATA)) for p in (DATA/'vocabulary/contracts'/path.parent.name).glob('part-*.json'))
        receipts.append(str(relative/'receipt.json'))
    training=[]
    for path in sorted((DATA/'training/receipts').glob('*.json')):
        r=json.loads(path.read_text())
        for key in ['audio_path','label_path','words_path']:files.append(r[key])
        receipts.append(str(path.relative_to(DATA)))
        training.append({**r,'audio':r['audio_path'],'labels':r['label_path'],
                         'duration':r['audio_duration'],'speakers':r['speaker_count'],'training_ready':True})
    review=[json.loads(p.read_text()) for p in sorted((DATA/'training/review').glob('*.json'))]
    transfer(sorted(set(files)),'inputs');transfer(receipts,'receipts')
    # Publish this snapshot only after its assets and receipts have arrived.
    for name,rows in [('training-manifest.jsonl',training),('review-manifest.jsonl',review)]:
        put(name,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode())
    save('transfer/training-status.json',{'sessions':len(training),'hours':sum(r['duration'] for r in training)/3600,
        'format':'24kHz FLAC + VibeVoice input JSON + native word timestamps','updated_at':time.time()})
    transfer(['training-manifest.jsonl','review-manifest.jsonl','transfer/training-status.json'],'published')
    save('transfer/status.json',{'updated_at':time.time(),'status':'synced','sessions_published':staging,
        'training_sessions_published':len(training),'training_hours_published':sum(r['duration'] for r in training)/3600,
        'remote':'/data/work/asr-synthetic-v7','receipts_published_last':True,'manifest_published_after_assets':True})


if __name__=='__main__':
    REPLICA.mkdir(parents=True,exist_ok=True)
    with (REPLICA/'.alignment-sync.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:sync()
            except Exception as e:
                save('transfer/status.json',{'status':'retrying','error':str(e),'updated_at':time.time()})
                print(type(e).__name__,str(e),flush=True)
            time.sleep(20)
