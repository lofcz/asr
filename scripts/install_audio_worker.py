"""Install/update the per-user service, keeping its executable off removable storage."""
import json,os,pathlib,shutil,subprocess,uuid
ROOT=pathlib.Path(__file__).resolve().parents[1];HOME=pathlib.Path.home()
state=HOME/'.local/state/asr-downloader';app=HOME/'.local/lib/asr-downloader';units=HOME/'.config/systemd/user'
for d in [state,app,units,ROOT/'queue']:d.mkdir(parents=True,exist_ok=True)
marker=ROOT/'.asr-volume-id'
if not marker.exists():
 with marker.open('x') as f:f.write(str(uuid.uuid4())+'\n');f.flush();os.fsync(f.fileno())
config={'root':str(ROOT),'mount':'/run/media/lofcz/ssd_external','volume_id':marker.read_text().strip(),'state':str(state),'deno':shutil.which('deno'),'poll_seconds':5,'reserve_bytes':10*1024**3,'job_timeout_seconds':3600,'rate_limit':'3M','concurrency':3}
config.update(target_hours=1000,budget_bytes=100_000_000_000,budget_stop_bytes=95_000_000_000)
if (state/'config.json').exists():config.update(json.loads((state/'config.json').read_text()))
if not config['deno']:raise SystemExit('Deno is required for yt-dlp JavaScript challenges')
# Stop first when updating an existing deployment. Download .part files are retained.
subprocess.run(['systemctl','--user','stop','asr-audio-worker.service'],check=False,capture_output=True)
shutil.copy2(ROOT/'scripts/audio_worker.py',app/'audio_worker.py')
(state/'config.json').write_text(json.dumps(config,indent=2)+'\n')
queue=ROOT/'queue/youtube.jsonl'
if not queue.exists():queue.write_text('# Append a YouTube video ID, video URL, or JSON metadata object per line.\n')
unit=f'''[Unit]
Description=Durable ASR audio download queue
StartLimitIntervalSec=0

[Service]
Type=notify
NotifyAccess=main
ExecStart=/usr/bin/python {app}/audio_worker.py --config {state}/config.json run
WorkingDirectory={app}
Restart=always
RestartSec=15
WatchdogSec=120
TimeoutStartSec=120
TimeoutStopSec=45
KillMode=control-group
UMask=0027
Nice=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
'''
(units/'asr-audio-worker.service').write_text(unit)
(ROOT/'services/asr-audio-worker.service').write_text(unit)
subprocess.run(['systemctl','--user','daemon-reload'],check=True)
subprocess.run(['systemctl','--user','enable','--now','asr-audio-worker.service'],check=True)
print('Queue:',queue);print('State:',state);print('Service: asr-audio-worker.service')
