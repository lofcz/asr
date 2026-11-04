"""One-time rollout at next existing checkpoint, avoiding a long rollback."""
import json,pathlib,subprocess,time
root=pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16')
initial=json.loads((root/'latest.json').read_text())['step']
while True:
 current=json.loads((root/'latest.json').read_text())['step']
 if current>initial or (root/'completed.json').exists():
  subprocess.run(['systemctl','--user','restart','qda496-training.service'],check=True)
  pathlib.Path('/data/work/asr-evaluation/periodic/activated.json').write_text(json.dumps({'checkpoint':current,'time':time.time()})+'\n')
  break
 time.sleep(2)
