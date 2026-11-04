"""Download selected original audio, resume completed files, and probe integrity."""
import argparse,concurrent.futures,hashlib,json,pathlib,subprocess,threading,time,urllib.parse,urllib.request
from audio_worker import atomic_json, sync_dir
ROOT=pathlib.Path(__file__).resolve().parents[1];M=ROOT/'data/manifests'
locks={};guard=threading.Lock()
def download(row):
    directory=ROOT/'data/audio'/row['language']/row['show_id'];directory.mkdir(parents=True,exist_ok=True)
    ext=pathlib.PurePosixPath(urllib.parse.urlparse(row['audio_url']).path).suffix.lower()
    if ext not in ['.mp3','.m4a','.ogg','.opus','.wav','.aac']:ext='.audio'
    dest=directory/(row['id']+ext);part=dest.with_suffix(dest.suffix+'.part');meta=directory/(row['id']+'.json')
    if dest.exists() and meta.exists():
        try:
            old=json.loads(meta.read_text())
            if dest.stat().st_size==old['bytes']:return old
        except (ValueError,KeyError,OSError):pass
    start=time.monotonic();host=urllib.parse.urlparse(row['audio_url']).hostname
    with guard:lock=locks.setdefault(host,threading.Semaphore(1))
    try:
        with lock:
            req=urllib.request.Request(row['audio_url'],headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req,timeout=30) as response,part.open('wb') as output:
                final_url=response.url;content_type=response.headers.get('Content-Type')
                if 'text/html' in (content_type or ''):raise ValueError('HTML instead of audio')
                size=0;started=time.monotonic();sha=hashlib.sha256()
                while block:=response.read(1024*1024):
                    output.write(block);sha.update(block);size+=len(block)
                    if size>500*1024*1024 or time.monotonic()-started>240:raise ValueError('Per-file transfer limit exceeded')
            time.sleep(0.5)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(part)],timeout=30))
        streams=[x for x in probe['streams'] if x['codec_type']=='audio']
        if not streams:raise ValueError('No audio stream')
        duration=float(probe['format']['duration'])
        if duration<row['declared_duration_seconds']*0.9:raise ValueError('Audio shorter than 90% of declared duration')
        with part.open('rb') as source: __import__('os').fsync(source.fileno())
        part.replace(dest);sync_dir(directory)
        result={**row,'status':'downloaded','path':str(dest.relative_to(ROOT)),'bytes':size,'sha256':sha.hexdigest(),'actual_duration_seconds':duration,'sample_rate':streams[0].get('sample_rate'),'channels':streams[0].get('channels'),'codec':streams[0].get('codec_name'),'resolved_audio_url':final_url,'download_elapsed_seconds':time.monotonic()-start}
        atomic_json(meta,result);return result
    except Exception as e:
        part.unlink(missing_ok=True);return {**row,'status':'failed','error':str(e)}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    rows=[json.loads(s) for s in (M/'selected-audio.jsonl').read_text().splitlines()]
    # Spread early downloads across languages and hosts; final ratio comes from selection.
    rows.sort(key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())
    if args.limit:rows=rows[:args.limit]
    results=[]
    with (M/'download-results.jsonl').open('w') as log,concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(download,r) for r in rows]
        for future in concurrent.futures.as_completed(futures):
            r=future.result();results.append(r);log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
            print(json.dumps({'done':len(results),'total':len(rows),'status':r['status'],'show':r['show'],'error':r.get('error')},ensure_ascii=False),flush=True)
    stats={}
    for lang in ['cs','sk']:
        r=[x for x in results if x['language']==lang and x['status']=='downloaded'];stats[lang]={'hours':sum(x['actual_duration_seconds'] for x in r)/3600,'episodes':len(r),'shows':len(set(x['show_id'] for x in r)),'bytes':sum(x['bytes'] for x in r)}
    stats['failed']=sum(r['status']=='failed' for r in results)
    atomic_json(M/'download-summary.json',stats);print(json.dumps(stats),flush=True)
    if stats['failed']:raise SystemExit(1)
if __name__=='__main__':main()
