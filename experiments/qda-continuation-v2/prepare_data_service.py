"""Durable preparation coordinator. Never starts a training experiment."""
import json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parent;RAW=ROOT/'datasets/soniox-50h-v2';REL=ROOT/'datasets/releases/qda-50h-v2';REMOTE='/data/work/asr-datasets/qda-50h-v2';EXP='/data/work/asr-experiments/qda-continuation-v2'
sys.path.insert(0,str(ROOT/'scripts'))
from soniox_durable import atomic,encode,file_hash

def run(args):subprocess.run(args,check=True)
def main():
 rows=json.loads((RAW/'selection.json').read_bytes())
 while True:
  states=[json.loads((RAW/'records'/r['record_id']/'state.json').read_text()) if (RAW/'records'/r['record_id']/'state.json').exists() else {} for r in rows]
  done=sum(s.get('phase')=='done' and s.get('remote_deleted') for s in states);vad=sum((RAW/'records'/r['record_id']/'vad.json').exists() for r in rows)
  print(json.dumps({'paid_archived_and_cleaned':done,'vad_verified':vad,'total':len(rows)}),flush=True)
  if done==vad==len(rows):break
  time.sleep(30)
 if not (REL/'verification.json').exists():
  run([sys.executable,str(HERE/'build_release.py')]);run([sys.executable,str(ROOT/'scripts/verify_qda_release.py'),str(REL),'--workers','4'])
 code=REL/'code';code.mkdir(exist_ok=True)
 for name in ('qda_release_dataset.py','verify_qda_release.py'):
  import shutil
  shutil.copy2(ROOT/'scripts'/name,code/name)
 # Bind loader/verifier code to the same metadata manifest, then verify again.
 metadata=json.loads((REL/'checksums.metadata.json').read_text())
 for p in code.iterdir():metadata[str(p.relative_to(REL))]=file_hash(p)
 atomic(REL/'checksums.metadata.json',encode(metadata));run([sys.executable,str(ROOT/'scripts/verify_qda_release.py'),str(REL),'--workers','4'])
 run(['ssh','mlserver','mkdir -p '+REMOTE+' '+EXP])
 run(['rsync','-a','--partial','--info=stats1',str(REL)+'/', 'mlserver:'+REMOTE+'/'])
 run(['rsync','-a','--exclude=.git','--exclude=__pycache__','--exclude=cache','--exclude=runs','--exclude=baseline','--exclude=plan.json',str(HERE)+'/', 'mlserver:'+EXP+'/'])
 run(['ssh','mlserver',f'/data/work/asr-pilot/venv/bin/python {REMOTE}/code/verify_qda_release.py {REMOTE} --workers 2'])
 run(['ssh','mlserver',f'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 /data/work/asr-pilot/venv/bin/python {EXP}/prepare_experiment.py inputs'])
 run(['ssh','mlserver',f'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 /data/work/asr-pilot/venv/bin/python {EXP}/audit_release.py {REMOTE} /data/work/asr-datasets/qda-496h-v1 --plan {EXP}/plan.json --out {EXP}/data-audit.json'])
 atomic(HERE/'data-ready.json',encode({'completed_at':time.time(),'release':str(REL),'remote':REMOTE,'selection_sha256':file_hash(REL/'selection.json'),'manifest_sha256':file_hash(REL/'training/sources.jsonl'),'status':'paid archive, both audio copies verified, CPU plan ready','features':'prepared after baseline completion; training requires explicit launcher command'}))
 run(['rsync','-a',str(HERE/'data-ready.json'),'mlserver:'+EXP+'/data-ready.json'])
 print('DATA_READY',flush=True)
if __name__=='__main__':main()
