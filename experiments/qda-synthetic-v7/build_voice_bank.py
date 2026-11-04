"""Grow a durable bank of up to 992 private voices + 8 library voices."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl,json,math,os,time
from pathlib import Path
import subprocess
import requests
import numpy as np
from core import DATA,REPLICA,read,save,put,verified,sha
from voices import headers,owned,registry

SSH='ssh -o ConnectTimeout=15 -S /home/lofcz/.cache/asr-ssh/mlserver-persistent'
REMOTE='mlserver:/data/work/asr-voice-diversity-v7/'

def sync_refs():
    subprocess.run(['rsync','-a','--timeout=120','-e',SSH,str(DATA/'voices/diverse-clips')+'/',REMOTE+'clips/'],check=True,capture_output=True)
    dest=REPLICA/'voice-embeddings';dest.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--timeout=120','--exclude=*.tmp','--exclude=.*','-e',SSH,REMOTE+'embeddings/',str(dest)+'/'],check=True,capture_output=True)
    return dest

def choose(dest,target=992):
    refs=read('voices/diverse-references.json',[])
    complete_inventory=bool(refs)
    if not refs:
        refs=[json.loads(p.read_text()) for p in (DATA/'voices/diverse-clips').glob('*.json')]
        refs=[r for r in refs if r.get('status')=='reference_ready']
    if not refs:return None
    vectors=[];rows=[]
    for r in refs:
        path=dest/(r['candidate_id']+'.json')
        if not path.exists():continue
        e=json.loads(path.read_text())
        if e['clip_sha256']!=r['clip_sha256']:raise ValueError('Wrong speaker embedding paired to clip')
        vectors.append(e['embedding']);rows.append(r)
    if len(rows)<len(refs) and complete_inventory:return None
    if not complete_inventory and min(Counter(r['language'][0] for r in rows).get(l,0) for l in ('cs','sk'))<50:return None
    # Greedy selection in reference-quality order with conservative high-similarity
    # deduplication. The score is evidence, not a claim of verified human identity.
    order=sorted(range(len(rows)),key=lambda i:rows[i]['score'],reverse=True)
    matrix=np.asarray(vectors,dtype=np.float32)
    established={(v.get('source_record_id'),str(v.get('source_speaker'))) for v in registry() if v['kind']=='private_clone'}
    selected=[i for i,r in enumerate(rows) if (r['source_id'],str(r['speaker'])) in established]
    duplicates=[]
    for i in order:
        if i in selected:continue
        nearest=-1.;match=None
        if selected:
            similarities=matrix[selected] @ matrix[i]
            where=int(np.argmax(similarities));nearest=float(similarities[where]);match=selected[where]
        if nearest>=.75:
            duplicates.append({'candidate_id':rows[i]['candidate_id'],'similar_to':rows[match]['candidate_id'],'cosine':nearest})
        else:selected.append(i)
    # Round-robin languages while available, then fill from the other language.
    chosen=[i for i in selected if (rows[i]['source_id'],str(rows[i]['speaker'])) in established]
    pools={l:[i for i in selected if rows[i]['language']==[l] and i not in chosen] for l in ('cs','sk')}
    while len(chosen)<target and any(pools.values()):
        for lang in ('cs','sk'):
            if pools[lang] and len(chosen)<target:chosen.append(pools[lang].pop(0))
    result=[]
    prior=read('voices/clone-sources.json',[])
    prior_map={(x['source_id'],str(x['speaker'])):x['name'] for x in prior}
    for i in chosen:
        r=rows[i];name=prior_map.get((r['source_id'],str(r['speaker'])),f'qda-v7-{r["language"][0]}-{r["candidate_id"][:12]}')
        result.append({**r,'name':name,'embedding_deduplication':'TitaNet cosine >= 0.75 treated as likely repeated identity'})
    save('voices/diversity-selection.json',{'target_private_voices':target,'reference_candidates':len(refs),
        'acoustically_deduplicated_candidates':len(selected),'selected':len(result),
        'by_language':dict(Counter(x['language'][0] for x in result)),'duplicates':duplicates,
        'limitation':'Conservative acoustic near-duplicate screen; not a biometric identity guarantee.'})
    save('voices/diverse-selected.json',result)
    return result

def clone(source,account):
    relative='voices/clones/'+source['name']
    prior=read(relative+'.json')
    if prior:return prior
    matches=[v for v in account if v['name']==source['name']]
    if len(matches)>1:raise ValueError('Duplicate voice names')
    if matches:voice=matches[0]
    elif read(relative+'.response.json'):voice=read(relative+'.response.json')
    else:
        intent=read(relative+'.intent.json')
        if intent:
            if time.time()-intent['submitted_at']<600:raise RuntimeError('Ambiguous clone request: '+source['name'])
            # Instant cloning is included/free. After a complete fresh account
            # listing confirms absence, retry the stale interrupted clone only.
            # Paid TTS requests have their separate no-ambiguous-retry policy.
            current=[v for v in owned() if v['name']==source['name']]
            if current:return clone(source,current)
            save(relative+f'.reconciliation-{int(time.time())}.json',
                {'original_intent':intent,'checked_at':time.time(),'account_name_absent':True,
                 'action':'retry interrupted included instant clone; no paid TTS retry'})
        payload={'name':source['name'],'language':source['language'][0],'access':'private',
                 'description':'Private QDA training voice with source provenance retained locally.','tagline':'QDA training'}
        audio=verified(source['clip_path'],source['clip_sha256'])
        save(relative+'.request.json',{'fields':payload,'clip_sha256':source['clip_sha256']})
        save(relative+'.intent.json',{'submitted_at':time.time(),'clip_sha256':source['clip_sha256']})
        for attempt in range(7):
            r=requests.post('https://api.cartesia.ai/voices/clone',headers=headers(),data=payload,
                files={'clip':(source['name']+'.wav',audio,'audio/wav')},timeout=120)
            if r.status_code==429:
                time.sleep(min(60,2**attempt));continue
            if r.status_code!=200:
                save(relative+'.error.json',{'status':r.status_code,'body':r.text[:2000]});raise RuntimeError('Clone HTTP '+str(r.status_code))
            voice=r.json();save(relative+'.response.json',voice);break
        else:raise RuntimeError('Clone rate limit persisted')
    if voice.get('access')!='private':raise ValueError('Clone must remain private')
    result={**voice,'kind':'private_clone','language':source['language'],'source_clip_sha256':source['clip_sha256'],
            'source_record_id':source['source_id'],'source_speaker':source['speaker'],'source_split':'train',
            'candidate_id':source['candidate_id'],'diverse_bank':True,'created_locally_at':time.time()}
    save(relative+'.json',result)
    return result

def main():
    with (REPLICA/'.voice-bank.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            selected=None
            try:selected=choose(sync_refs())
            except Exception as e:print(type(e).__name__,str(e),flush=True)
            if not selected:time.sleep(15);continue
            account=owned();errors=[];finished=0
            # Keep previously created diverse voices and add new selected references.
            # Never count a renamed copy as new speaker diversity.
            pending=[s for s in selected if not read('voices/clones/'+s['name']+'.json')]
            capacity=max(0,1000-len(registry()));pending=pending[:capacity]
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(clone,s,account) for s in pending]
                for future in as_completed(futures):
                    try:future.result();finished+=1
                    except Exception as e:errors.append(str(e));print('clone_error',str(e),flush=True)
                    bank=registry()
                    save('voices/expansion-status.json',{'phase':'cloning','target_total_voices':1000,
                        'selected_private_voices':len(selected),'registered_total':len(bank),
                        'registered_by_language':dict(Counter(v['language'][0] for v in bank)),
                        'finished':finished,'errors':errors,'at':time.time()})
                    if finished%20==0:print('registered',len(bank),'selected completed',finished,flush=True)
            if len(registry())>=1000 or (read('voices/diverse-references.json') and not pending):
                save('voices/bank-ready.json',{'registered':len(registry()),'target':1000,'errors':errors,'at':time.time()})
                return
            time.sleep(15)

if __name__=='__main__':main()
