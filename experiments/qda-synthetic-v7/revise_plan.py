"""Apply expanded voice/content scope without changing already generated audio."""
from collections import Counter,defaultdict
import json,random,re,time
from pathlib import Path
from core import DATA,HERE,read,save
from cartesia import request_for,VERSION
from core import digest,encode


def paid(s):
    root=DATA/'sessions'/s['id']
    if (root/'receipt.json').exists() or any(root.glob('parts/*/turn-*.json')):return True
    voices=read(Path('sessions')/s['id']/'voices.json')
    if not voices:return False
    for p in (DATA/'scripts'/s['id']).glob('part-*.json'):
        if len(p.name)!=13:continue
        data=json.loads(p.read_text())
        for turn in data.get('turns',[]):
            v=voices[turn['speaker']];req=request_for(turn['text'],v['language'][0],v['voice_id'],v['speed'])
            ident=digest(encode({'api_version':VERSION,'body':req}))
            if (DATA/'utterances'/ident/'intent.json').exists():return True
    return False


def catalog():
    general=[]
    for name in ['topics-everyday.json','topics-work-institutions.json','topics-social-experiences.json']:
        general.extend(json.loads((HERE/name).read_text()))
    ai=json.loads((HERE/'ai-domain-coverage.json').read_text())
    products=json.loads((HERE/'scio-products.json').read_text())
    scio=[]
    for i,p in enumerate(products):
        scio.append({'id':f'scio-product-{i:03d}','topic':'Zkušenosti a rozhodování kolem '+p['canonical_name'],
            'vocabulary':', '.join([p['canonical_name']]+p.get('related_terms',[])),
            'scenario':'Fiktivní kvalitativní rozhovor o konkrétní zkušenosti, očekávání, průběhu a následné reflexi.',
            'roles':['uživatel nebo účastník','kolega nebo průvodce','organizátor','dospělý zájemce'],
            'angles':['Co předcházelo rozhodnutí','Konkrétní situace při používání či účasti','Co bylo obtížné a jak se to řešilo','Co by účastník příště změnil'],
            'product':p,'source_notes':[p['description'],'Oficiální zdroje: '+', '.join(p['sources'])]})
    return {'general_qda':general,'ai':ai,'scio':scio},products


def revise():
    p=read('plan.json')
    if p.get('version')==3:return p
    pools,products=catalog();save('history/plan-v2-before-diversity.json',p)
    locked=[s for s in p['sessions'] if paid(s)];pending=[s for s in p['sessions'] if not paid(s)]
    save('history/locked-paid-sessions.json',[s['id'] for s in locked])
    split_counts=Counter();new=[]
    for s in pending:
        key=(s['cohort'],s['language'][0],s['speaker_count'])
        if s['cohort']=='core' and s['planned_seconds']==600 and split_counts[key]<3:
            split_counts[key]+=1
            for i in range(20):new.append({**s,'id':f'v3-{s["id"]}-short-{i:02d}','planned_seconds':30})
        else:new.append({**s,'id':'v3-'+s['id']})
        save('sessions/'+s['id']+'/state.json',{'status':'superseded_unsubmitted','reason':'Voice and content diversity revision; no TTS submitted','at':time.time()})
    rng=random.Random(20261024+3);rng.shuffle(new)
    # New short recordings give ~1k identities actual speaker slots while keeping
    # the original hour quotas and all existing 20/30/60-minute sessions.
    use=defaultdict(float);topic_index=Counter();form_index=Counter();product_seen=Counter()
    for s in locked:
        family='scio' if s['topic_group'] in ('scio','scio_ai') else 'ai'
        use[(s['language'][0],family)]+=s['planned_seconds']
    for s in sorted(new,key=lambda s:-s['planned_seconds']):
        lang=s['language'][0]
        family=min(pools,key=lambda f:use[(lang,f)])
        candidates=pools[family]
        index=topic_index[(lang,family)];topic_index[(lang,family)]+=1
        # Cover the entire sourced Scio inventory, then give core brands more
        # repeated exposure without replacing the other products.
        if family=='scio' and index>=len(candidates) and index%2==0:
            anchors=[c for c in candidates if c['product']['canonical_name'] in ('ScioBot','ScioŠkola','ScioŠkoly','ScioLink')]
            item=anchors[(index//2)%len(anchors)] if anchors else candidates[index%len(candidates)]
        else:item=candidates[index%len(candidates)]
        s.update({'topic_group':family,'topic_id':item['id'],'topic':item['topic'],'vocabulary':item['vocabulary'],
            'scenario':item.get('scenario',''),'roles':item.get('roles',[]),'angles':item.get('angles',[]),
            'source_notes':item.get('source_notes',[]),'source_files':[],
            'seed':rng.randrange(2**31),'customized_context':'','context_provenance':'Optional vocabulary selected before script generation'})
        s['required_terms']=[]
        s['required_terms_by_part']=[]
        if family=='scio':
            product=item['product'];forms=product.get('forms_'+lang,[]) or [product['canonical_name']]
            forms=[f for f in forms if isinstance(f,str) and not re.search(r'\d',f)]
            if forms:
                key=(lang,product['canonical_name'])
                count=1 if s['planned_seconds']<=60 else 2
                for part in range(max(1,(s['planned_seconds']+299)//300)):
                    offset=form_index[key];form_index[key]+=count
                    s['required_terms_by_part'].append(list(dict.fromkeys(forms[(offset+j)%len(forms)] for j in range(count))))
                s['required_terms']=s['required_terms_by_part'][0]
                s['required_term_forms']=forms
            s['canonical_product']=product['canonical_name'];product_seen[product['canonical_name']]+=1
        elif family=='ai':
            terms=[t for t in item.get('terms',[]) if isinstance(t,str) and not re.search(r'\d',t)]
            if terms:
                key=(lang,item['id']);count=1 if s['planned_seconds']<=30 else (2 if s['planned_seconds']<=120 else 4)
                for part in range(max(1,(s['planned_seconds']+299)//300)):
                    offset=form_index[key];form_index[key]+=count
                    s['required_terms_by_part'].append(list(dict.fromkeys(terms[(offset+j)%len(terms)] for j in range(count))))
                s['required_terms']=s['required_terms_by_part'][0]
                s['required_term_forms']=terms
        if s['seed']%2==0:s['customized_context']=', '.join(s['vocabulary'].split(', ')[:4])
        use[(lang,family)]+=s['planned_seconds']
    # Interleave families, languages and lengths immediately.
    rng.shuffle(new)
    new.sort(key=lambda s:0 if 60<=s['planned_seconds']<=300 else 1)
    updated={**p,'version':3,'voice_target':1000,'minimum_voices_before_resume':100,
        'voice_allocation':'unused identities first, then least assigned hours; fixed identity within each recording',
        'content_hour_targets':{'scio':20,'ai':20,'general_qda':20},
        'source_topic_counts':{k:len(v) for k,v in pools.items()},
        'scio_products':len(products),'scio_product_session_counts':dict(product_seen),
        'timestamp_source':'cartesia-native','diarization_source':'known voice IDs',
        'alignment':read('alignment.json'),'sessions':locked+new,'revised_at':time.time()}
    assert sum(s['planned_seconds'] for s in updated['sessions'])==216000
    assert Counter((s['cohort'],s['language'][0],s['speaker_count']) for s in locked+pending)
    before=defaultdict(int);after=defaultdict(int)
    for s in p['sessions']:before[(s['cohort'],s['language'][0],s['speaker_count'])]+=s['planned_seconds']
    for s in updated['sessions']:after[(s['cohort'],s['language'][0],s['speaker_count'])]+=s['planned_seconds']
    assert before==after,'Speaker/language hour quotas changed'
    save('plan.json',updated);save('content-catalog.json',pools)
    save('plan-revision.json',{'version':3,'paid_sessions_preserved':len(locked),'new_sessions':len(new),
        'speaker_slots':sum(s['speaker_count'] for s in updated['sessions']),
        'content_planned_hours':{l:{f:use[(l,f)]/3600 for f in pools} for l in ('cs','sk')},
        'source_topics':{k:len(v) for k,v in pools.items()},'at':time.time()})
    return updated


if __name__=='__main__':
    p=revise();print('revised',len(p['sessions']),'sessions',sum(s['speaker_count'] for s in p['sessions']),'speaker slots')
