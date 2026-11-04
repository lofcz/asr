"""Supervise the resumable RSS batch without depending on an interactive session."""
import json,pathlib,signal,subprocess,time
import audio_worker as worker
config=json.loads((pathlib.Path.home()/'.local/state/asr-downloader/config.json').read_text())
stop=False;child=None

def shutdown(*args):
 global stop
 stop=True
signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
while not stop:
 if not worker.storage_ready(config):time.sleep(5);continue
 root=pathlib.Path(config['root'])
 child=subprocess.Popen(['/usr/bin/python',str(root/'scripts/download_audio.py'),'--workers','4'],cwd=root,start_new_session=True)
 try:
  while child.poll() is None:
   if stop or not worker.storage_ready(config):worker.stop_child(child);break
   time.sleep(5)
 finally:worker.stop_child(child)
 if stop:break
 if child.returncode==0:break
 for _ in range(60):
  if stop:break
  time.sleep(5)
