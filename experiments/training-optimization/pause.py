import pathlib,json,time,subprocess
p=pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16')
while json.loads((p/'latest.json').read_text())['step']<350:time.sleep(1)
subprocess.run(['systemctl','--user','stop','qda496-training.service'],check=True)
print('Paused at', (p/'latest.json').read_text(),flush=True)
