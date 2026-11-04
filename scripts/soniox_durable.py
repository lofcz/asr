#!/usr/bin/env python3
"""Durable paid Soniox processing. Persist POST intent; ambiguous POSTs NEVER resubmit.
Internal disk is the write-ahead journal and independent replica. SSD holds dataset.
No credentials in state. Run a single process under flock, multiple independent jobs.
"""
import argparse, collections, concurrent.futures, fcntl, hashlib, json, logging, os, pathlib, signal, tempfile, threading, time
import requests
LOG=logging.getLogger('soniox-durable')
STOP=threading.Event()
BASE='https://api.soniox.com/v1'

def digest(data):return hashlib.sha256(data).hexdigest()
def encode(obj):return (json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def atomic(path,data):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        d=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(d)
        finally:os.close(d)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    return h.hexdigest()
class APIError(Exception):
    def __init__(self,status):self.status=status;super().__init__(f'HTTP {status}')
class Held(Exception):pass
class API:
    def __init__(self,key):
        self.session=requests.Session();self.session.headers['Authorization']='Bearer '+key
    def call(self,method,path,**kwargs):
        # requests default adapters perform no POST retries.
        timeout=(300,600) if method=='POST' and path=='/files' else (30,600)
        r=self.session.request(method,BASE+path,timeout=timeout,**kwargs)
        if not r.ok:raise APIError(r.status_code)
        return r.content
    def obj(self,method,path,**kw):return json.loads(self.call(method,path,**kw) or b'{}')
    def find(self,kind,reference):
        matches=[];cursor=None
        while True:
            params={'limit':1000}
            if cursor:params['cursor']=cursor
            page=self.obj('GET','/'+kind,params=params)
            matches.extend(x for x in page[kind] if x.get('client_reference_id')==reference)
            cursor=page.get('next_page_cursor')
            if not cursor:break
        return matches
class Store:
    def __init__(self,root,out,backup,marker,reference_prefix='asr500-v1'):
        self.root=pathlib.Path(root);self.out=pathlib.Path(out);self.backup=pathlib.Path(backup);self.marker=marker
        self.reference_prefix=reference_prefix
    def ready(self):
        try:return (self.root/'.asr-volume-id').read_text().strip()==self.marker
        except OSError:return False
    def require(self):
        if not self.ready():raise OSError('Expected SSD is unavailable')
    def put(self,relative,data):
        # First durable internal replica, then SSD. State never promises more than disk contains.
        atomic(self.backup/relative,data);self.require();atomic(self.out/relative,data)
    def obj(self,relative,obj):self.put(relative,encode(obj))
    def state(self,r):
        rel=pathlib.Path('records')/r['record_id']/'state.json'
        local=self.backup/rel;ssd=self.out/rel
        if local.exists():return json.loads(local.read_bytes())
        if ssd.exists():
            data=ssd.read_bytes();atomic(local,data);return json.loads(data)
        return {'record_id':r['record_id'],'phase':'pending','attempts':0}
    def set(self,r,s):
        s['updated_at']=time.time();self.obj(pathlib.Path('records')/r['record_id']/'state.json',s)
    def verified(self,rel,expected):
        good=None
        for base in (self.backup,self.out):
            p=base/rel
            if p.exists() and file_hash(p)==expected:good=p.read_bytes();break
        if good is None:raise OSError(f'No valid replica: {rel}')
        for base in (self.backup,self.out):
            p=base/rel
            if not p.exists() or file_hash(p)!=expected:
                if base==self.out:self.require()
                atomic(p,good)
        return good
    def pause(self,reason):self.obj('control.json',{'pause_submissions':True,'reason':reason,'time':time.time()})
    def paused(self):
        p=self.backup/'control.json'
        return p.exists() and json.loads(p.read_bytes()).get('pause_submissions',False)

def reconcile(api,kind,ref,file_id=None):
    matches=api.find(kind,ref)
    if len(matches)!=1:raise Held(f'{kind} submission uncertain; {len(matches)} matches; no automatic resubmission')
    if file_id and matches[0].get('file_id')!=file_id:raise Held('Remote file identity mismatch')
    return matches[0]

def post(api,store,r,s,kind,kwargs):
    intent='upload_intent' if kind=='files' else 'submission_intent'
    s['phase']=intent;s['intent_at']=time.time();store.set(r,s)
    try:return api.obj('POST','/'+kind,**kwargs)
    except APIError as e:
        # Explicit 4xx rejections can reset intent; 408/5xx/network failures remain ambiguous.
        if e.status in (400,401,402,403,404,413,415,422,429):
            s['phase']='pending' if kind=='files' else 'uploaded'
            if e.status in (401,402,403):store.pause(f'Soniox HTTP {e.status}; fix account/credential and explicitly clear pause')
            if e.status in (400,404,413,415,422):s['phase']='rejected'
            store.set(r,s)
        raise

def archive(store,r,s,api):
    rel=pathlib.Path('records')/r['record_id']
    rawpath=store.backup/rel/'soniox.raw.json'
    if rawpath.exists():
        raw=rawpath.read_bytes()
        # Independently committed hash is required before trusting a pre-existing raw response.
        hp=store.backup/rel/'raw.sha256'
        if not hp.exists() or digest(raw)!=hp.read_text().strip():raw=api.call('GET','/transcriptions/'+s['transcription_id']+'/transcript')
    else:raw=api.call('GET','/transcriptions/'+s['transcription_id']+'/transcript')
    data=json.loads(raw)
    if not isinstance(data.get('tokens'),list):raise ValueError('Transcript has no token array')
    store.put(rel/'soniox.raw.json',raw);store.put(rel/'raw.sha256',(digest(raw)+'\n').encode())
    # Keep EVERY token, including speakerless words, events and endpoint tokens, in raw storage.
    # Derived timed tokens retain token indices, exact text, confidence, speaker, language and events.
    timed=[];untimed=[];flags=[]
    for i,t in enumerate(data['tokens']):
        if isinstance(t.get('start_ms'),(int,float)) and t['start_ms']>=0 and isinstance(t.get('end_ms'),(int,float)):
            timed.append({'token_index':i,**t})
            if t['end_ms']<t['start_ms'] or t['end_ms']>r['seconds']*1000+3000:flags.append('timestamp_out_of_bounds')
        else:untimed.append(i)
    speech=[t for t in timed if not t.get('is_audio_event') and t.get('text','').strip() not in ('','<end>','<fin>')]
    unknown=sum(t.get('speaker') is None for t in speech)
    if unknown:flags.append('speech_tokens_without_speaker')
    if not speech:flags.append('no_speech_tokens')
    gaps=[];end=0
    for t in sorted(speech,key=lambda t:t['start_ms']):
        if t['start_ms']-end>=2000:gaps.append([end,t['start_ms']])
        end=max(end,t['end_ms'])
    if r['seconds']*1000-end>=2000:gaps.append([end,round(r['seconds']*1000)])
    derived={'schema_version':1,'record_id':r['record_id'],'audio_sha256':r['sha256'],'audio_path':r['path'],'duration_seconds':r['seconds'],'split':r['split'],'timed_tokens':timed,'untimed_token_indices':untimed,'context':r['context'],'quality':{'flags':sorted(set(flags)),'speech_tokens_without_speaker':unknown,'untranscribed_intervals_ms':gaps,'event_audit':'pending independent VAD/acoustic classification and review','training_ready':False}}
    artifacts={'audio.json':encode(r),'request.json':encode(s['request']),'remote.json':encode(s['completed']),'soniox.raw.json':raw,'transcript.json':encode(derived)}
    hashes={}
    for name,content in artifacts.items():store.put(rel/name,content);hashes[name]=digest(content)
    receipt={'schema_version':1,'record_id':r['record_id'],'audio_sha256':r['sha256'],'audio_bytes':r['bytes'],'audio_path':r['path'],'selection_sha256':s['selection_sha256'],'file_id':s['file_id'],'transcription_id':s['transcription_id'],'client_reference_id':s['reference'],'artifacts_sha256':hashes,'archived_at':time.time(),'replicas':['ssd','internal_disk'],'training_ready':False}
    store.obj(rel/'receipt.json',receipt)
    for name,sha in hashes.items():store.verified(rel/name,sha)
    s['receipt_sha256']=digest(encode(receipt));s['phase']='done';store.set(r,s)
    LOG.info('ARCHIVED %s %.2f minutes %d tokens',r['id'],r['seconds']/60,len(data['tokens']))

def verify_done(store,r,s):
    rel=pathlib.Path('records')/r['record_id']
    receipt=json.loads(store.verified(rel/'receipt.json',s['receipt_sha256']))
    if receipt['audio_sha256']!=r['sha256'] or receipt['transcription_id']!=s['transcription_id']:raise ValueError('Receipt identity mismatch')
    for name,sha in receipt['artifacts_sha256'].items():store.verified(rel/name,sha)

def process(store,r,key,selection_sha,api=None):
    api=api or API(key);s=store.state(r)
    try:
        store.require()
        rel=pathlib.Path('records')/r['record_id']
        if s['phase']=='done':
            if not s.get('remote_deleted'):
                verify_done(store,r,s)
                for kind,field in [('transcriptions','transcription_id'),('files','file_id')]:
                    try:api.call('DELETE','/'+kind+'/'+s[field])
                    except APIError as e:
                        if e.status!=404:raise
                s['remote_deleted']=True;store.set(r,s)
            return
        if s['phase']=='rejected':return
        s['selection_sha256']=selection_sha;s['reference']=store.reference_prefix+'-'+r['record_id']
        source=store.root/r['path']
        if s['phase']=='pending':
            if store.paused() or STOP.is_set():return
            with source.open('rb') as audio:
                stat=os.fstat(audio.fileno())
                if stat.st_size!=r['bytes']:raise Held('Audio size mismatch')
                h=hashlib.sha256()
                while block:=audio.read(4*1024*1024):h.update(block)
                if h.hexdigest()!=r['sha256']:raise Held('Audio SHA256 mismatch')
                audio.seek(0);s['verified_audio_at']=time.time();store.obj(rel/'audio.json',r)
                # Preserve original receipt too; request pairs exact hashed bytes, not an ambiguous basename.
                original=source.with_suffix('.json')
                if original.exists():store.put(rel/'source-receipt.json',original.read_bytes())
                uploaded=post(api,store,r,s,'files',{'files':{'file':(r['record_id']+source.suffix,audio,'application/octet-stream')},'data':{'client_reference_id':s['reference']}})
            s['file_id']=uploaded['id'];s['phase']='uploaded';store.obj(rel/'upload.json',uploaded);store.set(r,s)
        elif s['phase']=='upload_intent':
            uploaded=reconcile(api,'files',s['reference'])
            if uploaded.get('size')!=r['bytes']:raise Held('Remote upload size mismatch')
            s['file_id']=uploaded['id'];s['phase']='uploaded';store.obj(rel/'upload.json',uploaded);store.set(r,s)
        if s['phase']=='uploaded':
            if store.paused() or STOP.is_set():return
            s['request']={'model':'stt-async-v5','file_id':s['file_id'],'language_hints':[r['language']],'enable_speaker_diarization':True,'enable_language_identification':True,'client_reference_id':s['reference']}
            store.obj(rel/'request.json',s['request'])
            job=post(api,store,r,s,'transcriptions',{'json':s['request']})
            s['transcription_id']=job['id'];s['phase']='submitted';store.set(r,s)
        elif s['phase']=='submission_intent':
            job=reconcile(api,'transcriptions',s['reference'],s['file_id'])
            s['transcription_id']=job['id'];s['phase']='submitted';store.set(r,s)
        if s['phase']=='submitted':
            status=api.obj('GET','/transcriptions/'+s['transcription_id'])
            if status['status']=='error':
                s['phase']='remote_error';s['remote_error']=status;store.set(r,s);return
            if status['status']!='completed':
                s['next_poll']=time.time()+20;store.set(r,s);return
            if status.get('file_id')!=s['file_id'] or status.get('client_reference_id')!=s['reference']:raise Held('Completed job identity mismatch')
            s['completed']=status;s['phase']='archiving';store.set(r,s)
        if s['phase']=='archiving':archive(store,r,s,api)
    except Exception as e:
        # A save may have committed before raising; never overwrite newer durable intent with stale state.
        s=store.state(r);s['last_error']=f'{type(e).__name__}: {e}';s['attempts']=s.get('attempts',0)+1
        s['retry_after']=time.time()+(300 if isinstance(e,Held) else min(900,15*2**min(s['attempts'],6)))
        try:store.set(r,s)
        except OSError:pass
        LOG.warning('%s phase=%s %s',r['id'],s['phase'],s['last_error'])

def progress(store,rows):
    states=[store.state(r) for r in rows];counts=collections.Counter(s['phase'] for s in states)
    result={'updated_at':time.time(),'counts':dict(counts),'selected_hours':sum(r['seconds'] for r in rows)/3600,'archived_hours':sum(r['seconds'] for r,s in zip(rows,states) if s['phase']=='done')/3600,'archived_records':counts['done'],'total_records':len(rows),'pause_submissions':store.paused(),'training_ready':False,'annotation_audit':'pending','unresolved':[{'record_id':r['record_id'],'phase':s['phase'],'error':s.get('last_error')} for r,s in zip(rows,states) if s['phase'] in ('upload_intent','submission_intent','rejected','remote_error') or s.get('last_error') and s['phase']!='done']}
    store.obj('progress.json',result);return result

def remote_inventory(api):
    result={}
    for kind in ('files','transcriptions'):
        rows=[];cursor=None
        while True:
            params={'limit':1000}
            if cursor:params['cursor']=cursor
            page=api.obj('GET','/'+kind,params=params);rows.extend(page[kind])
            cursor=page.get('next_page_cursor')
            if not cursor:break
        result[kind]=rows
    return result

def admission_plan(rows,states,inventory,max_outstanding=80,upload_batch=8,storage_bytes=8_000_000_000,paused=False,now=None):
    """Reserve account-wide headroom serially before concurrent uploads.
    Count queued AND processing as outstanding, more conservative than pending only.
    Ambiguous local intents reserve extra slots even if remotely unobserved.
    """
    now=time.time() if now is None else now
    files=inventory['files'];jobs=inventory['transcriptions']
    active=sum(j.get('status') not in ('completed','error') for j in jobs)
    unsubmitted=sum(s['phase'] in ('upload_intent','uploaded','submission_intent') for s in states)
    slots=max(0,min(max_outstanding-active-unsubmitted,1900-len(jobs)-unsubmitted,900-len(files)))
    free_bytes=max(0,storage_bytes-sum(f['size'] for f in files))
    admitted=[];maintenance=[]
    # Existing paid jobs and cleanup always proceed, regardless of upload headroom.
    # Limit submissions of existing uploads too: no bypass of the account job limit.
    submit_slots=max(0,min(max_outstanding-active,1900-len(jobs)))
    for r,s in zip(rows,states):
        phase=s['phase']
        if phase in ('rejected','remote_error') or phase=='done' and s.get('remote_deleted'):continue
        if s.get('retry_after',0)>now or phase=='submitted' and s.get('next_poll',0)>now:continue
        if phase=='pending':
            if paused or len(admitted)>=upload_batch or slots<=0 or r['bytes']>free_bytes:continue
            admitted.append(r);slots-=1;free_bytes-=r['bytes']
        else:
            if phase in ('uploaded','upload_intent'):
                if paused or submit_slots<=0:continue
                submit_slots-=1
            maintenance.append(r)
    return maintenance+admitted,{'account_remote_files':len(files),'account_remote_bytes':sum(f['size'] for f in files),'account_remote_jobs':len(jobs),'account_outstanding_jobs':active,'local_unsubmitted_intents':unsubmitted,'new_uploads_admitted':len(admitted),'max_outstanding':max_outstanding,'storage_ceiling_bytes':storage_bytes}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--backup',required=True);p.add_argument('--env',required=True);p.add_argument('--marker',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);p.add_argument('--max-outstanding',type=int,default=80);p.add_argument('--upload-batch',type=int,default=8);p.add_argument('--verify',action='store_true');p.add_argument('--max-hours',type=float,default=500);p.add_argument('--reference-prefix',default='asr500-v1');args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    store=Store(args.root,args.out,args.backup,args.marker,args.reference_prefix);store.backup.mkdir(parents=True,exist_ok=True)
    lock=(store.backup/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:STOP.set())
    while not store.ready():
        if STOP.wait(10):return
    selection=(store.out/'selection.json').read_bytes();selection_sha=digest(selection)
    if selection_sha!=(store.out/'selection.sha256').read_text().strip():raise ValueError('Selection manifest checksum mismatch')
    prior=store.backup/'selection.json'
    if prior.exists() and prior.read_bytes()!=selection:raise ValueError('Immutable selection changed')
    store.put('selection.json',selection);store.put('selection.sha256',(selection_sha+'\n').encode())
    for name in ('selection-summary.json','selection-exclusions.json','inventory.snapshot.json'):
        if (store.out/name).exists():store.put(name,(store.out/name).read_bytes())
    rows=json.loads(selection)
    if len({r['record_id'] for r in rows})!=len(rows) or any(r['record_id']!=r['sha256'] for r in rows):raise ValueError('Duplicate or invalid audio identity')
    if not 0<args.max_hours<=500 or sum(r['seconds'] for r in rows)>args.max_hours*3600:raise ValueError('Authorized audio-hour budget exceeded')
    if args.verify:
        for r in rows:
            s=store.state(r)
            if s['phase']=='done':verify_done(store,r,s)
        print(json.dumps(progress(store,rows)));return
    key=None
    for line in pathlib.Path(args.env).read_text().splitlines():
        name,sep,value=line.partition('=')
        if name.strip()=='SONIOX_API_KEY':key=value.strip().strip('"').strip("'")
    if not key:raise ValueError('SONIOX_API_KEY missing')
    if not 1<=args.max_outstanding<100:raise ValueError('Outstanding cap must stay below 100')
    if not 1<=args.upload_batch<=32:raise ValueError('Invalid upload batch')
    account_api=API(key)
    working=rows[:args.limit] if args.limit else rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while not STOP.is_set():
            if not store.ready():STOP.wait(10);continue
            try:
                inventory=remote_inventory(account_api)
                states=[store.state(r) for r in working]
                tasks,capacity=admission_plan(working,states,inventory,args.max_outstanding,args.upload_batch,paused=store.paused())
                store.obj('remote-capacity.json',{**capacity,'updated_at':time.time()})
            except Exception as e:
                LOG.warning('Remote capacity check failed; no new uploads: %s',e)
                STOP.wait(30);continue
            futures=[pool.submit(process,store,r,key,selection_sha) for r in tasks]
            for future in futures:future.result()
            try:
                summary=progress(store,rows)
                LOG.info('Progress %.2f / %.2fh %s',summary['archived_hours'],summary['selected_hours'],summary['counts'])
            except OSError:continue
            if all(store.state(r)['phase']=='done' and store.state(r).get('remote_deleted') for r in working):break
            STOP.wait(5)
if __name__=='__main__':main()
