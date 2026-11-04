#!/usr/bin/env python3
"""Durable single-consumer YouTube audio queue. Only standard library dependencies."""
import argparse, concurrent.futures, contextlib, fcntl, hashlib, json, logging, os, pathlib, random
import re, shutil, signal, socket, sqlite3, subprocess, sys, tempfile, time, urllib.parse
LOG=logging.getLogger('asr-worker')
STOP=False
VIDEO_ID=re.compile(r'^[A-Za-z0-9_-]{11}$')

def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def atomic_json(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);sync_dir(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):os.unlink(tmp)

def parse_line(line):
    line=line.strip()
    if not line or line.startswith('#'):return None
    obj=json.loads(line) if line.startswith('{') else {'id':line}
    if not isinstance(obj,dict):raise ValueError('Expected object, video ID, or YouTube URL')
    value=str(obj.get('id') or obj.get('video_id') or obj.get('url') or '')
    if not VIDEO_ID.fullmatch(value):
        url=urllib.parse.urlparse(value)
        host=(url.hostname or '').lower()
        if url.scheme not in ('https','http'):raise ValueError('Invalid video ID or URL')
        if host=='youtu.be':value=url.path.strip('/')
        elif host in ('youtube.com','www.youtube.com','music.youtube.com'):
            if url.path=='/watch':value=urllib.parse.parse_qs(url.query).get('v',[''])[0]
            elif url.path.startswith(('/shorts/','/live/','/embed/')):value=url.path.split('/')[2]
            else:raise ValueError('Provide an individual video, not a channel or playlist')
        else:raise ValueError('Only YouTube video URLs are accepted')
    if not VIDEO_ID.fullmatch(value):raise ValueError('Invalid video ID')
    return {**obj,'id':value,'url':'https://www.youtube.com/watch?v='+value}

def connect(state):
    state=pathlib.Path(state);state.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(state/'queue.sqlite3',timeout=30)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0,
        created REAL NOT NULL, updated REAL NOT NULL, error TEXT, output TEXT, receipt TEXT);
    CREATE TABLE IF NOT EXISTS ingestion_errors (
        digest TEXT PRIMARY KEY, line_number INTEGER, line TEXT, error TEXT, seen REAL);
    ''');db.commit()
    if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise RuntimeError('Queue database failed integrity check')
    return db

def ingest(db,queue):
    try:data=pathlib.Path(queue).read_bytes()
    except FileNotFoundError:return 0
    added=0
    # Deliberately ignore the unfinished final line while another process appends.
    for number,raw in enumerate(data.split(b'\n')[:-1],1):
        try:
            item=parse_line(raw.decode('utf-8'))
            if item is None:continue
            now=time.time()
            result=db.execute('INSERT OR IGNORE INTO jobs(id,payload,created,updated) VALUES(?,?,?,?)',
                              (item['id'],json.dumps(item,ensure_ascii=False),now,now))
            added+=result.rowcount
        except (ValueError,UnicodeError,TypeError) as e:
            digest=hashlib.sha256(raw).hexdigest()
            db.execute('INSERT OR IGNORE INTO ingestion_errors VALUES(?,?,?,?,?)',
                       (digest,number,raw.decode('utf-8',errors='replace'),str(e),time.time()))
    db.commit()
    if added:LOG.info('Ingested %d new jobs',added)
    return added

def recover(db):
    count=db.execute("UPDATE jobs SET status='pending', next_attempt=0, updated=? WHERE status='running'",(time.time(),)).rowcount
    db.commit()
    if count:LOG.warning('Recovered %d interrupted jobs',count)

def claim(db):
    db.execute('BEGIN IMMEDIATE')
    row=db.execute("SELECT * FROM jobs WHERE status IN ('pending','retry') AND next_attempt<=? ORDER BY created,id LIMIT 1",(time.time(),)).fetchone()
    if row:db.execute("UPDATE jobs SET status='running',updated=? WHERE id=?",(time.time(),row['id']))
    db.commit();return row

def fail(db,job,error):
    attempts=job['attempts']+1;delay=min(21600,60*2**min(attempts-1,9))+random.uniform(0,30)
    db.execute("UPDATE jobs SET status='retry',attempts=?,next_attempt=?,updated=?,error=? WHERE id=?",
               (attempts,time.time()+delay,time.time(),str(error)[-4000:],job['id']));db.commit()
    LOG.error('%s attempt %d failed; retry in %.0fs: %s',job['id'],attempts,delay,error)

def requeue(db,job):
    db.execute("UPDATE jobs SET status='pending',next_attempt=0,updated=? WHERE id=?",(time.time(),job['id']));db.commit()

def complete(db,job,receipt):
    db.execute("UPDATE jobs SET status='done',updated=?,error=NULL,output=?,receipt=? WHERE id=?",
               (time.time(),receipt['path'],json.dumps(receipt,ensure_ascii=False),job['id']));db.commit()
    LOG.info('%s complete: %.1f minutes, %.1f MB',job['id'],receipt['duration_seconds']/60,receipt['bytes']/1e6)

def storage_ready(config):
    root=pathlib.Path(config['root'])
    try:
        return os.path.ismount(config['mount']) and (root/'.asr-volume-id').read_text().strip()==config['volume_id']
    except OSError:return False

def notify(message):
    address=os.environ.get('NOTIFY_SOCKET')
    if not address:return
    if address.startswith('@'):address='\0'+address[1:]
    with contextlib.suppress(OSError):
        with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as sock:sock.sendto(message.encode(),address)

def probe_file(path,expected=None):
    result=subprocess.run(['/usr/bin/ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)],capture_output=True,text=True,timeout=45,check=True)
    data=json.loads(result.stdout);streams=data.get('streams',[])
    audio=[s for s in streams if s.get('codec_type')=='audio']
    if not audio or any(s.get('codec_type')=='video' for s in streams):raise ValueError('Expected audio-only file')
    duration=float(data['format']['duration'])
    if duration<=0 or path.stat().st_size<1024:raise ValueError('Empty or invalid audio')
    if expected and duration<float(expected)*0.97:raise ValueError('Audio shorter than expected; keeping it for inspection')
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):sha.update(block)
        os.fsync(f.fileno())
    return {'duration_seconds':duration,'bytes':path.stat().st_size,'sha256':sha.hexdigest(),
            'codec':audio[0].get('codec_name'),'sample_rate':audio[0].get('sample_rate'),'channels':audio[0].get('channels')}

def publish(db,job,config):
    root=pathlib.Path(config['root']);key=job['id'];stage=root/'data/youtube/staging'/key
    final=root/'data/youtube/audio';final.mkdir(parents=True,exist_ok=True)
    allowed={'.webm','.m4a','.opus','.ogg','.mp3','.aac','.wav','.flac'}
    candidates=[p for p in final.glob(key+'.*') if p.suffix in allowed]
    candidates+=sorted(p for p in stage.glob(key+'.*') if p.suffix in allowed)
    if not candidates:return False
    media=candidates[0];info_path=stage/(key+'.info.json');saved_info={}
    if info_path.exists():saved_info=json.loads(info_path.read_text())
    payload=json.loads(job['payload']);expected=saved_info.get('duration') or payload.get('expected_duration_seconds')
    try:details=probe_file(media,expected)
    except (ValueError,subprocess.CalledProcessError,json.JSONDecodeError):
        quarantine=root/'data/youtube/quarantine';quarantine.mkdir(parents=True,exist_ok=True)
        os.replace(media,quarantine/(str(time.time_ns())+'-'+media.name));sync_dir(quarantine)
        LOG.warning('Quarantined invalid media for %s; it will be downloaded again',key)
        return False
    destination=final/media.name
    if media!=destination:os.replace(media,destination);sync_dir(final);sync_dir(stage)
    receipt={**details,'id':key,'path':str(destination.relative_to(root)),'source_url':payload['url'],
             'queue_metadata':payload,'youtube_metadata':saved_info,'completed_at':time.time()}
    atomic_json(final/(key+'.json'),receipt)
    # The database commit is last: any earlier crash is reconciled from final media.
    complete(db,job,receipt);return True

def stop_child(child):
    if child.poll() is not None:return
    with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGKILL)
        child.wait(timeout=10)

def download(db,job,config):
    root=pathlib.Path(config['root']);key=job['id'];stage=root/'data/youtube/staging'/key
    stage.mkdir(parents=True,exist_ok=True)
    if publish(db,job,config):return
    if STOP:requeue(db,job);return
    cmd=[str(root/'.venv/bin/python'),'-m','yt_dlp','--ignore-config','--no-playlist','--no-overwrites',
         '--continue','--part','--write-info-json','--no-progress','--format','bestaudio',
         '--socket-timeout','30','--retries','3','--fragment-retries','3','--abort-on-unavailable-fragments',
         '--concurrent-fragments','1','--sleep-interval','3','--max-sleep-interval','8',
         '--remote-components','ejs:github','--js-runtimes','deno:'+config['deno'],
         '--output',str(stage/(key+'.%(ext)s')),'--',json.loads(job['payload'])['url']]
    if config.get('rate_limit'):
        position=cmd.index('--');cmd[position:position]=['--limit-rate',config['rate_limit']]
    log_path=stage/'download.log'
    if log_path.exists() and log_path.stat().st_size>5*1024*1024:os.replace(log_path,stage/'download.previous.log')
    with log_path.open('ab') as log:
        log.write(('\nSTART '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n').encode());log.flush()
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started=time.monotonic()
        try:
            while child.poll() is None:
                notify('WATCHDOG=1')
                if STOP or not storage_ready(config):
                    stop_child(child);requeue(db,job);return
                if shutil.disk_usage(root).free<config['reserve_bytes']:
                    stop_child(child);raise RuntimeError('Low disk space; download paused')
                if time.monotonic()-started>config['job_timeout_seconds']:
                    stop_child(child);raise TimeoutError('Download exceeded job timeout')
                ingest(db,root/'queue/youtube.jsonl')
                time.sleep(config['poll_seconds'])
        finally:
            stop_child(child);log.flush();os.fsync(log.fileno())
        if child.returncode:
            tail=log_path.read_bytes()[-2500:].decode(errors='replace')
            raise RuntimeError('yt-dlp exit '+str(child.returncode)+': '+tail)
    if not publish(db,job,config):raise RuntimeError('yt-dlp exited without complete audio')

def backup(db,state,root=None):
    target=pathlib.Path(state)/'backups';target.mkdir(exist_ok=True)
    # Keep one snapshot per day for seven days, plus latest; never copy a live WAL DB.
    for destination in [target/'latest.sqlite3',target/(time.strftime('%Y-%m-%d')+'.sqlite3')]:
        temp=destination.with_suffix('.tmp');other=sqlite3.connect(temp)
        try:db.backup(other)
        finally:other.close()
        with temp.open('rb') as f:os.fsync(f.fileno())
        os.replace(temp,destination);sync_dir(target)
    dated=sorted(target.glob('????-??-??.sqlite3'))
    for old in dated[:-7]:old.unlink()
    if root:
        dest=pathlib.Path(root)/'queue/state-backup.sqlite3';temp=dest.with_suffix('.tmp')
        shutil.copyfile(target/'latest.sqlite3',temp)
        with temp.open('rb') as f:os.fsync(f.fileno())
        os.replace(temp,dest);sync_dir(dest.parent)

def audit(db,root):
    for row in db.execute("SELECT id,output,receipt FROM jobs WHERE status='done'").fetchall():
        path=pathlib.Path(root)/row['output']
        try:valid=path.stat().st_size==json.loads(row['receipt'])['bytes']
        except (OSError,ValueError):valid=False
        if not valid:
            db.execute("UPDATE jobs SET status='retry',next_attempt=0,error='Completed file missing or size changed',updated=? WHERE id=?",(time.time(),row['id']))
    db.commit()

def run_job(state,job,config):
    db=connect(state)
    try:
        try:download(db,job,config)
        except Exception as e:fail(db,job,e)
    finally:db.close()

def worker(config,state):
    global STOP
    state=pathlib.Path(state);state.mkdir(parents=True,exist_ok=True)
    with (state/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=connect(state);recover(db);last_backup=0;last_audit=0;last_status=None
        concurrency=max(1,min(8,int(config.get('concurrency',1))))
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
        active={}
        notify('READY=1')
        try:
            while not STOP:
                notify('WATCHDOG=1')
                for future,job in list(active.items()):
                    if future.done():
                        try:future.result()
                        except Exception as e:fail(db,job,e)
                        del active[future]
                if not storage_ready(config):
                    if last_status!='missing':LOG.warning('Waiting for the correct SSD mount');last_status='missing'
                    time.sleep(config['poll_seconds']);continue
                root=pathlib.Path(config['root']);ingest(db,root/'queue/youtube.jsonl')
                if time.time()-last_audit>3600:audit(db,root);last_audit=time.time()
                if time.time()-last_backup>900:backup(db,state,root);last_backup=time.time()
                if shutil.disk_usage(root).free<config['reserve_bytes']:
                    if last_status!='space':LOG.warning('Waiting for free disk space');last_status='space'
                    time.sleep(config['poll_seconds']);continue
                while len(active)<concurrency and not STOP:
                    job=claim(db)
                    if not job:break
                    last_status='working';LOG.info('Starting %s',job['id'])
                    active[pool.submit(run_job,state,job,config)]=job
                if not active and last_status!='idle':
                    LOG.info('Queue idle; watching for appended IDs');last_status='idle'
                time.sleep(config['poll_seconds'])
        finally:
            STOP=True
            pool.shutdown(wait=True)
            backup(db,state,config['root'] if storage_ready(config) else None);db.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','status','enqueue','retry']);parser.add_argument('--config',required=True);parser.add_argument('items',nargs='*');args=parser.parse_intermixed_args()
    config=json.loads(pathlib.Path(args.config).read_text());state=pathlib.Path(config['state'])
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.command=='run':
        def stop(signum,frame):
            global STOP
            STOP=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);worker(config,state)
    elif args.command=='enqueue':
        if not storage_ready(config):raise SystemExit('SSD is not mounted')
        queue=pathlib.Path(config['root'])/'queue/youtube.jsonl'
        with queue.open('a+b') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0,os.SEEK_END)
            if f.tell():
                f.seek(-1,os.SEEK_END)
                if f.read(1)!=b'\n':raise SystemExit('Queue has an unfinished last line; finish it before appending')
            for item in args.items:f.write((json.dumps(parse_line(item),ensure_ascii=False)+'\n').encode())
            f.flush();os.fsync(f.fileno())
        sync_dir(queue.parent)
    else:
        db=connect(state)
        if args.command=='retry':
            for item in args.items:
                key=parse_line(item)['id'];db.execute("UPDATE jobs SET status='pending',next_attempt=0,error=NULL WHERE id=? AND status='retry'",(key,))
            db.commit()
        result={r['status']:r['n'] for r in db.execute('SELECT status,count(*) n FROM jobs GROUP BY status')}
        result['invalid_queue_lines']=db.execute('SELECT count(*) FROM ingestion_errors').fetchone()[0]
        result['downloaded_hours']=sum(json.loads(r[0])['duration_seconds'] for r in db.execute("SELECT receipt FROM jobs WHERE status='done'"))/3600
        result['recent_errors']=[dict(r) for r in db.execute("SELECT id,attempts,error,next_attempt FROM jobs WHERE status='retry' ORDER BY updated DESC LIMIT 5")]
        print(json.dumps(result,ensure_ascii=False,indent=2));db.close()
if __name__=='__main__':main()
