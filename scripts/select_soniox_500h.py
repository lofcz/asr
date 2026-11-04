"""Freeze a reproducible session-level selection; acoustic probes are triage, not VAD."""
import concurrent.futures, collections, hashlib, json, math, os, pathlib, re, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/soniox-500h-v1'
def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    t.replace(p)
def audit(r):
    p=OUT/'selection-audit'/f"{r['sha256']}.json"
    if p.exists():return json.loads(p.read_text())
    a=dict(r);a['exclusion']=None
    try:
        path=ROOT/r['path']
        if path.stat().st_size!=r['bytes']:raise ValueError('source size mismatch')
        levels=[]
        for fraction in (.12,.48,.81):
            cmd=['ffmpeg','-nostdin','-hide_banner','-v','info','-ss',str(r['seconds']*fraction),'-i',str(path),'-t','12','-vn','-ac','1','-af','astats=metadata=0:reset=0','-f','null','-']
            proc=subprocess.run(cmd,capture_output=True,text=True,timeout=70)
            values=re.findall(r'RMS level dB: ([^\s]+)',proc.stderr)
            peaks=re.findall(r'Peak level dB: ([^\s]+)',proc.stderr)
            if proc.returncode or not values:raise ValueError('acoustic probe failed')
            level=float(values[-1]);peak=float(peaks[-1])
            levels.append({'fraction':fraction,'seconds':12,'rms_db':level if math.isfinite(level) else -120,'peak_db':peak if math.isfinite(peak) else -120})
        a['probes']=levels
        if max(x['rms_db'] for x in levels)<-55:a['exclusion']='all three probes near silent'
        a['score']=round((12 if re.search(r'rozhovor|interview|host|setkání|osobnost|podcast|debata|dialog|diskus',r['show'],re.I) else 0)+(15 if 600<=r['seconds']<=3600 else 5)-sum(x['rms_db']<-45 for x in levels)*5,3)
    except Exception as e:a['exclusion']=str(e);a['score']=-100
    save(p,a);return a

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'selection.json').exists():raise SystemExit('Selection already frozen; refusing to replace')
    snapshot=OUT/'inventory.snapshot.json'
    if not snapshot.exists():save(snapshot,json.loads((ROOT/'data/manifests/downloaded-corpus.json').read_text()))
    rows=json.loads(snapshot.read_text());eligible=[];excluded=[];seen=set()
    for r in rows:
        reason=None
        if r['sha256'] in seen:reason='exact audio duplicate'
        seen.add(r['sha256'])
        if r['language'] not in ('cs','sk'):reason='outside requested language pool'
        if not 300<=r['seconds']<=7200:reason='outside 5–120 minute session range'
        if re.search(r'pohád|pohad|rozpráv|rozprav|audioknih|bible|bibli[ae]|zprávy|newsfilter|newsletter|rozhlasové hry',r['show'],re.I):reason='metadata suggests non-conversational format'
        if r['id'] in ('bb9c60fb1c687ee744f74e87','cc908d5bb894709138f33f80'):reason='reserved previous pilot/evaluation source'
        if reason:excluded.append(dict(r,exclusion=reason))
        else:eligible.append(r)
    print(json.dumps({'eligible_sessions':len(eligible),'hours':sum(r['seconds'] for r in eligible)/3600}),flush=True)
    audited=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for a in pool.map(audit,eligible):
            audited.append(a)
            if len(audited)%50==0:print(f'probed {len(audited)}/{len(eligible)}',flush=True)
    save(OUT/'selection-exclusions.json',excluded+[r for r in audited if r['exclusion']])
    selected=[]
    for lang,target in [('cs',350*3600),('sk',150*3600)]:
        groups=collections.defaultdict(list)
        for r in audited:
            if r['language']==lang and not r['exclusion']:groups[r['show']].append(r)
        for group in groups.values():group.sort(key=lambda r:(-r['score'],hashlib.sha256(('soniox500v1'+r['sha256']).encode()).hexdigest()))
        used=collections.Counter();total=0
        for cap in (6*3600,8*3600,10*3600):
            while True:
                progress=False
                for show in sorted(groups,key=lambda s:(used[s],s)):
                    for i,r in enumerate(groups[show]):
                        if used[show]+r['seconds']<=cap and total+r['seconds']<=target:
                            selected.append(r);total+=r['seconds'];used[show]+=r['seconds'];groups[show].pop(i);progress=True;break
                if not progress:break
            if target-total<300:break
        if total<target-3600:raise RuntimeError(f'Insufficient eligible audio for {lang}: {total/3600}h')
    # Split entire shows, before any future windowing. Deterministic language-stratified selection.
    show_splits={}
    for lang in ('cs','sk'):
        shows=collections.defaultdict(float)
        for r in selected:
            if r['language']==lang:shows[r['show']]+=r['seconds']
        total=sum(shows.values());allocation=collections.Counter()
        for show in sorted(shows,key=lambda s:hashlib.sha256(('split-v1'+s).encode()).hexdigest()):
            split='validation' if allocation['validation']<total*.03 else 'test' if allocation['test']<total*.02 else 'train'
            show_splits[show]=split;allocation[split]+=shows[show]
    for r in selected:
        r['record_id']=r['sha256'];r['split']=show_splits[r['show']]
        r['context']={'available_metadata':{'title':r['title'],'show':r['show']},'provenance':'source metadata, not reference transcript','customized_context':None}
    selected.sort(key=lambda r:r['record_id'])
    summary={'version':1,'created_at':time.time(),'records':len(selected),'hours':sum(r['seconds'] for r in selected)/3600,'shows':len(show_splits),'language_hours':{l:sum(r['seconds'] for r in selected if r['language']==l)/3600 for l in ('cs','sk')},'split_hours':{s:sum(r['seconds'] for r in selected if r['split']==s)/3600 for s in ('train','validation','test')},'method':'Metadata conversational preference; 3x12s decode/energy probes; whole sessions, show diversity cap 6h (8/10h fallback), deterministic ranking. Not human-reviewed or acoustically verified language/speaker quality. No silence/music labels inferred.'}
    save(OUT/'selection.json',selected);save(OUT/'selection-summary.json',summary)
    digest=hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    (OUT/'selection.sha256').write_text(digest+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
