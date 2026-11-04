"""Local, durable audio/reference review. No automatic 'human approved' decisions."""
import argparse
import hashlib
import http.server
import json
import pathlib
import subprocess
import time
import threading
import urllib.parse
from contracts import atomic_json, sha
from gates import validate_reference, demand
from events import TAGS,reviewed_events
from dashboard_data import dashboard_data

HERE=pathlib.Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);p.add_argument('--port',type=int,default=8767)
    p.add_argument('--asr-root',type=pathlib.Path);a=p.parse_args();data=a.data;audio_lock=threading.Lock()
    class Handler(http.server.BaseHTTPRequestHandler):
        def rows(self):
            path=data/'screening-selection.json'
            if not path.exists():return []
            selection=json.loads(path.read_text());rows=selection['development']+selection['train']
            queue=data/'acoustic-review-queue.json'
            if queue.exists():
                ids={r['window_id'] for r in rows}
                for language in ('cs','sk'):
                    rows += [{**r,'review_kind':'acoustic'} for r in json.loads(queue.read_text()) if r['language']==language and r['window_id'] not in ids][:6]
            return rows
        def response(self,obj,status=200):
            content=json.dumps(obj,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(content)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content)
        def do_GET(self):
            url=urllib.parse.urlsplit(self.path);params=urllib.parse.parse_qs(url.query)
            if url.path in ('/', '/training', '/preparation', '/experiments'):
                content=(HERE.parent/'qda-full-r64-v4/training.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(content)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content);return
            if url.path=='/training-data':
                return self.response(dashboard_data())
            if url.path=='/markers':
                content=(HERE.parent/'qda-full-r64-v4/markers.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(content)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content);return
            if url.path=='/marker-comparisons':
                from marker_comparisons import build
                try:return self.response(build())
                except Exception as error:return self.response({'error':str(error)},503)
            if url.path=='/marker-audio':
                lang=params.get('lang',[''])[0]
                if lang not in ('cs','sk'):return self.response({'error':'Unknown marker'},404)
                root=HERE.parent/'production-reference-audit-v2';file=root/lang/'audio.playback.wav'
                total=file.stat().st_size;start=0;end=total-1;partial=self.headers.get('Range','')
                try:
                    if partial.startswith('bytes='):
                        left,right=partial[6:].split('-',1)
                        if not left:start=max(0,total-int(right))
                        else:start=int(left);end=min(int(right) if right else end,end)
                    if not 0<=start<=end<total:raise ValueError('Range out of bounds')
                except ValueError:return self.response({'error':'Invalid range'},416)
                self.send_response(206 if partial else 200);self.send_header('Content-Type','audio/wav');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
                if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
                self.end_headers()
                try:
                    with file.open('rb') as f:
                        f.seek(start);remaining=end-start+1
                        while remaining:
                            chunk=f.read(min(65536,remaining))
                            if not chunk:break
                            self.wfile.write(chunk);remaining-=len(chunk)
                except (BrokenPipeError,ConnectionResetError):pass
                return
            if url.path=='/legacy':
                self.send_response(302);self.send_header('Location','http://localhost:8768/');self.send_header('Content-Length','0');self.end_headers();return
            if url.path=='/calibration':
                content=(HERE.parent/'qda-full-r64-v4/preparation.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(content)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content);return
            if url.path=='/preparation-data':
                def read(p):return json.loads(p.read_text()) if p.exists() else None
                full=HERE.parent/'qda-full-r64-v4';cal=HERE.parent/'qda-lr-calibration-v1/result';dataset=data.parent/'qda-full-r64-v4'
                active=read(HERE.parent/'active-training.json') or {}
                if active.get('experiment')=='qda-r64-continuation-v6':
                    full=HERE.parent/'qda-r64-continuation-v6';cal=full/'result';dataset=data.parent/'qda-unseen-v6'
                elif active.get('experiment')=='qda-full-continuous-v8':
                    full=HERE.parent/'qda-full-continuous-v8';dataset=data.parent/'qda-full-continuous-v8'
                run=full/'runs/r64-full';events=[]
                if (run/'events.jsonl').exists():
                    for line in (run/'events.jsonl').read_text().splitlines():
                        try:events.append(json.loads(line))
                        except json.JSONDecodeError:continue
                return self.response({'preparation':read(full/'preparation-status.json'),'spec':read(full/'run-spec.json'),
                    'prepared_package':read(full/'prepared.json'),
                    'structural':read(dataset/'structural-audit.json'),'selection':read(dataset/'selection-summary.json'),
                    'calibration':read(cal/'history.json'),'calibration_config':read(cal/'config.json'),
                    'lr_summary':read(cal/'analysis.json'),'wandb':read(cal/'wandb-status.json'),
                    'sync':read(full/'sync-status.json'),'experiment':read(full/'experiment-status.json'),
                    'live_sync':read(full/'live-sync-status.json'),
                    'health':read(full/'health-status.json'),'recovery':read(full/'recovery-status.json'),
                    'selection_progress':read(full/'selection-progress.json'),
                    'full_training':{'config':read(run/'config.json'),'events':events,'checkpoint':read(run/'latest.json'),
                                     'wandb':read(run/'wandb-status.json'),'completed':read(run/'completed.json')},
                    'schedule_preview':read(full/'schedule-preview.json')})
            if url.path=='/review':
                content=(HERE/'review.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content);return
            if url.path=='/probes':
                content=(HERE/'dashboard.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content);return
            if url.path=='/status':
                def read(p):return json.loads(p.read_text()) if p.exists() else None
                runs=[]
                for folder in sorted((HERE/'runs').glob('*')):
                    if not folder.is_dir():continue
                    events=[]
                    path=folder/'events.jsonl'
                    if path.exists():
                        for line in path.read_text().splitlines():
                            try:events.append(json.loads(line))
                            except json.JSONDecodeError:pass
                    runs.append({'name':folder.name,'config':read(folder/'config.json'),'events':events,'wandb':read(folder/'wandb-status.json'),'completed':read(folder/'completed.json')})
                return self.response({'curation':read(HERE/'curation-status.json'),'experiment':read(HERE/'experiment-status.json'),'structural':read(data/'structural-audit.json'),
                    'selection':{k:v for k,v in (read(data/'screening-selection.json') or {}).items() if k not in ('train','development')},
                    'admission':read(data/'admission.json'),'runs':runs,'results':{str(r):read(HERE/f'r{r}-probe-result.json') for r in (16,64)},
                    'baseline_markers':read(HERE/'markers/vanilla/completed-v3.json'),
                    'baseline_interviews':read(HERE/'baseline-result.json'),
                    'stopped':{str(r):read(HERE/f'r{r}-probe-stopped.json') for r in (16,64)},
                    'midpoints':{str(r):read(HERE/f'r{r}-probe-10h.json') for r in (16,64)}})
            if url.path=='/api':
                rows=[]
                for row in self.rows():
                    wid=row['window_id'];review=data/('acoustic-reviews' if row.get('review_kind')=='acoustic' else 'reviews')/(wid+'.json');second=data/'secondary'/(wid+'.json')
                    rows.append({**row,'review':json.loads(review.read_text()) if review.exists() else None,
                                 'secondary':json.loads(second.read_text()) if second.exists() else None})
                return self.response({'rows':rows,'policy':json.loads((HERE/'policy.json').read_text()),'training_started':False})
            if url.path=='/audio':
                wid=params.get('id',[''])[0];row=next((r for r in self.rows() if r['window_id']==wid),None)
                if not row:return self.response({'error':'Unknown selected window'},404)
                src=pathlib.Path(row['audio_file'])
                if not src.exists() and a.asr_root:
                    src=a.asr_root/'datasets/releases'/row['origin']/row['audio_path']
                    if not src.exists():src=a.asr_root/row['audio_path']
                if not src.exists():return self.response({'error':'Audio not copied locally yet'},503)
                folder=data/'review-audio';folder.mkdir(exist_ok=True);dest=folder/(wid+'.wav')
                if not dest.exists():
                    demand(sha(src)==row['audio_sha256'],'Audio identity mismatch')
                    import shutil
                    ffmpeg=shutil.which('ffmpeg')
                    if not ffmpeg:
                        import imageio_ffmpeg
                        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
                    with audio_lock:
                        if not dest.exists():
                            tmp=dest.with_suffix('.building.wav')
                            subprocess.run([ffmpeg,'-nostdin','-v','error','-y','-i',str(src),'-ss',str(row['source_start']),'-t',str(row['duration']),'-ac','1','-ar','24000',str(tmp)],check=True)
                            tmp.replace(dest)
                total=dest.stat().st_size;start=0;end=total-1;partial=self.headers.get('Range','')
                if partial.startswith('bytes='):
                    left,right=partial[6:].split('-',1);start=int(left or 0);end=min(int(right) if right else end,end)
                if not 0<=start<=end<total:return self.response({'error':'Invalid range'},416)
                self.send_response(206 if partial else 200);self.send_header('Content-Type','audio/wav');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
                if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
                self.end_headers()
                try:
                    with dest.open('rb') as f:
                        f.seek(start);remaining=end-start+1
                        while remaining:
                            chunk=f.read(min(65536,remaining))
                            if not chunk:break
                            self.wfile.write(chunk);remaining-=len(chunk)
                except (BrokenPipeError,ConnectionResetError):pass
                return
            return self.response({'error':'Not found'},404)
        def do_POST(self):
            if self.path not in ('/review','/acoustic-review'):return self.response({'error':'Not found'},404)
            try:
                origin=self.headers.get('Origin','');demand(origin in (f'http://localhost:{a.port}',f'http://127.0.0.1:{a.port}'),'Untrusted origin')
                length=int(self.headers.get('Content-Length',0));demand(0<length<2_000_000,'Invalid payload size')
                value=json.loads(self.rfile.read(length));row=next(r for r in self.rows() if r['window_id']==value['window_id'])
                demand(value.get('reviewer','').strip(),'Reviewer identity required');demand(value.get('decision') in ('approve','reject'),'Decision required')
                acoustic=self.path=='/acoustic-review'
                if acoustic and value['decision']=='approve':
                    demand(row.get('review_kind')=='acoustic','Not an acoustic-review window')
                    gaps=row['quality']['unresolved_nonspeech'];regions=value.get('regions',[])
                    demand(len(gaps)==len(regions),'All flagged gaps must be reviewed')
                    for (start,end),region in zip(gaps,regions):
                        demand(region.get('listened') is True and region.get('tag') in TAGS and region.get('start')==start and region.get('end')==end,'Incomplete or changed acoustic region')
                elif value['decision']=='approve':
                    demand(all(value.get(k) is True for k in ('speech_verified','speakers_verified','events_verified')),'All review checks required')
                    validate_reference(value['reference'],row['duration'])
                    value['reference']=[{'Start':round(float(s['Start']),2),'End':round(float(s['End']),2),'Speaker':s.get('Speaker'),'Content':s['Content']} for s in value['reference']]
                value.update(audio_sha256=row['audio_sha256'],label_sha256=row['label_sha256'],reviewed_at=time.time(),source_start=row['source_start'],source_end=row['source_end'])
                path=data/('acoustic-reviews' if acoustic else 'reviews')/(row['window_id']+'.json')
                if path.exists():atomic_json(data/'review-history'/(row['window_id']+'-'+str(time.time_ns())+'.json'),json.loads(path.read_text()))
                atomic_json(path,value);return self.response({'saved':True})
            except Exception as error:return self.response({'error':str(error)},400)
    http.server.ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':main()
