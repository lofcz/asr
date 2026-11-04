"""Freeze new source-disjoint audio with existing show-level split boundaries."""
import collections, concurrent.futures, hashlib, html, json, math, pathlib, re, subprocess, sys, time, unicodedata
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from soniox_durable import atomic,encode,file_hash
OUT=ROOT/'datasets/soniox-50h-v2'

def key(r):
    return tuple(re.sub(r'\W+',' ',unicodedata.normalize('NFC',html.unescape(r[k])).casefold()).strip() for k in ('show','title'))

def check(r):
    p=OUT/'selection-audit'/f"{r['sha256']}.json"
    if p.exists():return json.loads(p.read_bytes())
    a=dict(r);a['exclusion']=None
    try:
        path=ROOT/r['path']
        assert path.stat().st_size==r['bytes'] and file_hash(path)==r['sha256'],'audio identity mismatch'
        proc=subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(path),'-map','0:a:0','-vn','-ac','1','-ar','16000','-progress','pipe:1','-nostats','-f','null','-'],capture_output=True,text=True,timeout=600)
        times=re.findall(r'^out_time_us=(\d+)$',proc.stdout,re.M)
        if proc.returncode or proc.stderr.strip() or not times:raise ValueError('full decode diagnostics: '+proc.stderr[:160])
        duration=int(times[-1])/1e6
        if abs(duration-r['seconds'])>max(2,r['seconds']*.002):raise ValueError('duration mismatch')
        a['full_decode_preflight_seconds']=duration
        old=ROOT/'datasets/soniox-500h-v1/selection-audit'/p.name
        if old.exists():
            prior=json.loads(old.read_bytes());a['probes']=prior.get('probes',[])
            if prior.get('exclusion'):raise ValueError(prior['exclusion'])
        if not a.get('probes'):
            a['probes']=[]
            for fraction in (.12,.48,.81):
                pr=subprocess.run(['ffmpeg','-nostdin','-v','info','-ss',str(duration*fraction),'-i',str(path),'-t','12','-ac','1','-af','astats=metadata=0:reset=0','-f','null','-'],capture_output=True,text=True,timeout=60)
                levels=re.findall(r'RMS level dB: ([^\s]+)',pr.stderr)
                if pr.returncode or not levels:raise ValueError('probe decode failure')
                level=float(levels[-1]);a['probes'].append({'fraction':fraction,'rms_db':level if math.isfinite(level) else -120})
        if max(p['rms_db'] for p in a['probes'])<-55:raise ValueError('all probes near silent')
    except Exception as e:a['exclusion']=str(e)
    atomic(p,encode(a));return a

def choose(rows):
    selected=[];used=collections.Counter()
    # Explicitly retain the first run's validation shows; no fresh held-out show is taken from prior training.
    for split,lang,target in [('validation','cs',1.4),('validation','sk',.6),('train','cs',33.6),('train','sk',14.4)]:
        pool=[r for r in rows if r['split']==split and r['language']==lang and not r.get('exclusion')]
        amount=0.; bins=collections.Counter()
        while pool:
            remain=target*3600-amount
            fits=[r for r in pool if r['seconds']<=remain and (split!='train' or used[r['show']]+r['seconds']<=2.5*3600)]
            if not fits:break
            def priority(r):
                bucket=0 if r['seconds']<1200 else 1 if r['seconds']<2400 else 2
                score=1 if re.search('rozhovor|interview|host|setkání|debata|dialog|kouč|živote|houpačky',r['show'],re.I) else 0
                return (used[r['show']],bins[bucket]/[.2,.4,.4][bucket],-score,hashlib.sha256(('fresh50-v2'+r['sha256']).encode()).hexdigest())
            r=min(fits,key=priority);selected.append(r);pool.remove(r);amount+=r['seconds'];used[r['show']]+=r['seconds'];bins[0 if r['seconds']<1200 else 1 if r['seconds']<2400 else 2]+=r['seconds']
    # Fill remaining budget with intact training sessions, keeping <=50h and diversity.
    total=sum(r['seconds'] for r in selected);seen={r['sha256'] for r in selected}
    for r in sorted(rows,key=lambda r:(used[r['show']],r['seconds'])):
        if r['split']=='train' and not r.get('exclusion') and r['sha256'] not in seen and used[r['show']]+r['seconds']<=9000 and total+r['seconds']<=50*3600:
            selected.append(r);seen.add(r['sha256']);total+=r['seconds'];used[r['show']]+=r['seconds']
    return selected

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'selection.json').exists():raise SystemExit('Selection frozen; refusing replacement')
    original=json.loads((ROOT/'datasets/soniox-500h-v1/selection.json').read_bytes())
    inventory=json.loads((ROOT/'data/manifests/downloaded-corpus.json').read_bytes())
    atomic(OUT/'inventory.snapshot.json',encode(inventory))
    hashes={r['sha256'] for r in original};ids={r['id'] for r in original}|{'bb9c60fb1c687ee744f74e87','cc908d5bb894709138f33f80'};identities={key(r) for r in original}
    splits={r['show']:r['split'] for r in original};rows=[];excluded=[];seen=set()
    for r in inventory:
        reason=None
        if r['sha256'] in hashes or r['id'] in ids or key(r) in identities:reason='previous selection or episode duplicate'
        elif r['sha256'] in seen:reason='duplicate source hash'
        elif r['language'] not in ('cs','sk') or not 600<=r['seconds']<=3600:reason='outside CS/SK 10-60 minute target'
        elif splits.get(r['show'])=='test':reason='keep original test shows untouched'
        elif re.search('pohád|pohad|rozpráv|rozprav|audioknih|bible|zprávy|newsfilter|newsletter|rozhlasové hry',r['show'],re.I):reason='nonconversation metadata'
        seen.add(r['sha256'])
        if reason:excluded.append({'record_id':r['sha256'],'reason':reason});continue
        rows.append({**r,'split':splits.get(r['show'],'train')})
    audited={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            proposed=choose([audited.get(r['sha256'],r) for r in rows]);todo=[r for r in proposed if r['sha256'] not in audited]
            if not todo:break
            for a in pool.map(check,todo):
                audited[a['sha256']]=a
                print(json.dumps({'audited':len(audited),'record':a['id'],'excluded':a['exclusion']}),flush=True)
    selected=choose([audited.get(r['sha256'],r) for r in rows])
    assert all(r['sha256'] in audited and not r.get('exclusion') for r in selected)
    assert 49*3600<sum(r['seconds'] for r in selected)<=50*3600
    for r in selected:
        r['record_id']=r['sha256'];r['context']={'available_metadata':{'title':r['title'],'show':r['show']},'provenance':'source metadata only','customized_context':None}
    selected.sort(key=lambda r:r['record_id'])
    atomic(OUT/'selection.json',encode(selected));atomic(OUT/'selection.sha256',(file_hash(OUT/'selection.json')+'\n').encode())
    atomic(OUT/'selection-exclusions.json',encode(excluded+[{'record_id':r['sha256'],'reason':r['exclusion']} for r in audited.values() if r['exclusion']]))
    summary={'release':'soniox-50h-v2','created_at':time.time(),'records':len(selected),'hours':sum(r['seconds'] for r in selected)/3600,'bytes':sum(r['bytes'] for r in selected),'language_hours':{l:sum(r['seconds'] for r in selected if r['language']==l)/3600 for l in ('cs','sk')},'split_hours':{s:sum(r['seconds'] for r in selected if r['split']==s)/3600 for s in ('train','validation')},'shows':len({r['show'] for r in selected}),'no_overlap_original_selection':not bool(hashes & {r['sha256'] for r in selected}),'previous_selection_sha256':file_hash(ROOT/'datasets/soniox-500h-v1/selection.json'),'method':'Whole 10-60min recordings, <=2.5h/show, balanced length bins, full-decode and audio hash preflight, conversational metadata preference; existing show-level splits retained. ASR/VAD quality gates follow paid transcription. Human review not claimed.'}
    atomic(OUT/'selection-summary.json',encode(summary));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
