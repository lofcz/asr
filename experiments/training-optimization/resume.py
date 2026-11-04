import pathlib,json,time,subprocess
p=pathlib.Path('/data/work/asr-optimization')
while True:
 state=subprocess.run(['systemctl','--user','show','qda496-optimize-stress.service','-p','ActiveState','--value'],capture_output=True,text=True).stdout.strip()
 if state not in ('activating','active'):break
 time.sleep(2)
for name in ('bench-new-1800-stress-gpu1.json','bench-new-3600-stress-gpu0.json'):
 rows=json.loads((p/name).read_text());assert rows and all(r['peak_gib']<22.5 for r in rows)
assert json.loads(pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16/plan-audit.json').read_text())['status']=='passed'
assert len(json.loads((p/'gradient-comparison.json').read_text()))==2
subprocess.run(['systemctl','--user','reset-failed','qda496-training.service'],check=True)
subprocess.run(['systemctl','--user','start','qda496-training.service'],check=True)
pathlib.Path('/data/work/asr-evaluation/periodic/activated.json').write_text(json.dumps({'checkpoint':350,'time':time.time(),'reason':'optimized trainer includes periodic evaluation hook'})+'\n')
print('Resumed optimized training from checkpoint 350',flush=True)
