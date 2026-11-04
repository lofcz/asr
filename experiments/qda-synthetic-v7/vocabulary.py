"""Corpus-wide vocabulary contracts, checked before any paid speech request."""
from collections import Counter,defaultdict
import json,math,re,time
from pathlib import Path
from core import DATA,HERE,read,save,digest,encode

REVISION='vocabulary-v4.1'
MAX_DENSITY=.38
NUMERIC_TERMS=('BM25','G2P','HTTP 429','INT4','INT8','wav2vec 2.0')


def pattern(forms):
    return re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(f) for f in sorted(set(forms),key=len,reverse=True))+r')(?!\w)',re.I)


def terms():
    result=[]
    products=json.loads((HERE/'scio-products.json').read_text())
    acronyms={'Národní srovnávací zkoušky':{'cs':'NSZ','sk':'NPS'},'Národné porovnávacie skúšky':{'cs':'NPS','sk':'NPS'},
        'Obecné studijní předpoklady':{'cs':'OSP','sk':'OSP'},'Všeobecné študijné predpoklady':{'cs':'VŠP','sk':'VŠP'},
        'Základy společenských věd':{'cs':'ZSV','sk':'ZSV'},'General Academic Prerequisites':{'cs':'GAP','sk':'GAP'}}
    for i,p in enumerate(products):
        forms={l:[f for f in p['forms_'+l] if not re.search(r'\d',f)] for l in ('cs','sk')}
        if p['canonical_name'] in acronyms:
            for l in forms:forms[l]=list(dict.fromkeys([acronyms[p['canonical_name']][l]]+forms[l]))
        result.append({'id':f'scio-{i:03d}','family':'scio','name':p['canonical_name'],'forms':forms,
            'target_per_language':200 if i<=34 or i==51 else 100,'description':p['description'],'sources':p['sources'],
            'group':min(i//8,6)})
    topics=json.loads((HERE/'ai-domain-coverage.json').read_text());seen={}
    for topic_index,t in enumerate(topics):
        for term in t['terms']:
            seen.setdefault(term,{'description':t['topic']+'. '+t['scenario'],'group':topic_index//5})
    for name in ('AI','ChatGPT','OpenAI','Codex'):
        seen.setdefault(name,{'description':'Praktické zkušenosti s nástroji umělé inteligence a jejich používáním.','group':0})
    for i,(name,info) in enumerate(sorted(seen.items())):
        branded=name in ('AI','ChatGPT','OpenAI','Codex','Claude','Anthropic','Gemini','Google','Qwen','Llama','Whisper','Canary','Cartesia','Soniox')
        result.append({'id':f'ai-{i:03d}','family':'ai','name':name,'forms':{'cs':[name],'sk':[name]},
                       'target_per_language':50 if name.isupper() or branded else 25,'sources':[] ,**info})
    return result


def contract(session_id,part):
    return read(Path('vocabulary/contracts')/session_id/f'part-{part:03d}.json')


def validate_contract(parsed,c):
    body={k:v for k,v in c.items() if k!='sha256'}
    if c.get('sha256')!=digest(encode(body)):raise ValueError('Vocabulary contract hash mismatch')
    text=' '.join(t['text'] for t in parsed['turns']);failures=[];counts={}
    for term in c['terms']:
        matches=list(pattern(term['forms']).finditer(text));counts[term['id']]=len(matches)
        if len(matches)<term['minimum']:
            failures.append(f"{term['name']}: {len(matches)} / {term['minimum']}")
        for form in term.get('required_forms',[]):
            if not pattern([form]).search(text):failures.append('missing form '+form)
    if failures:raise ValueError('Vocabulary contract failed: '+'; '.join(failures))
    return counts


def extend(session):
    """Prepare vocabulary obligations for measured-duration fill, before TTS."""
    sid=session['id'];lang=session['language'][0];remaining=session['planned_seconds']
    catalog=read('vocabulary/catalog.json');coverage=read('domain-coverage.json',{})
    if not catalog:raise ValueError('Missing corpus vocabulary catalog')
    def observed(t):
        row=(coverage.get('scio',{}).get(t['name'],{}).get(lang,{}) if t['family']=='scio'
             else coverage.get('ai',{}).get(lang,{}).get('terms',{}).get(t['name'],{}))
        return row.get('occurrences',0)/t['target_per_language']
    catalog=sorted(catalog,key=lambda t:(observed(t),t['id']))
    part=0
    while remaining>0:
        seconds=min(300,remaining);words=max(60,round(seconds/60*145));chosen=[];cost=0
        anchor=catalog[(session['seed']+part)%min(20,len(catalog))]
        ordered=sorted(catalog,key=lambda t:(t['family']!=anchor['family'],t['group']!=anchor['group'],observed(t),t['id']))
        for t in ordered:
            forms=t['forms'][lang];minimum=max(1,min(4,round(seconds/90)))
            form=min(forms,key=lambda f:(len(f.split()),len(f)));expense=minimum*len(form.split())
            if cost+expense>words*.25:continue
            chosen.append({'id':t['id'],'name':t['name'],'family':t['family'],'forms':forms,
                'minimum':minimum,'required_forms':[form],'description':t['description'],
                'sources':t['sources'],'word_cost':expense})
            cost+=expense
            if len(chosen)>=max(2,min(8,round(seconds/45))):break
        c={'revision':REVISION,'session_id':sid,'part':part,'language':[lang],
            'planned_seconds':seconds,'target_words':words,'mandatory_word_cost':cost,'terms':chosen,
            'original_topic':session['topic'],'original_family':session['topic_group'],
            'speaker_count':session['speaker_count'],'duration_extension':True}
        c['sha256']=digest(encode(c))
        existing=contract(sid,part)
        if existing:
            if existing['sha256']!=c['sha256']:raise ValueError('Refusing to replace an existing extension contract')
        else:save(Path('vocabulary/contracts')/sid/f'part-{part:03d}.json',c)
        remaining-=seconds;part+=1


def locked_parts(session):
    from cartesia import request_for,VERSION
    sid=session['id'];voices=read(Path('sessions')/sid/'voices.json')
    if not voices:return {}
    result={}
    for path in (DATA/'scripts'/sid).glob('part-*.json'):
        if not re.fullmatch(r'part-\d+\.json',path.name):continue
        script=json.loads(path.read_text());part=script['part']
        for turn in script['turns']:
            voice=voices[turn['speaker']];request=request_for(turn['text'],voice['language'][0],voice['voice_id'],voice['speed'])
            ident=digest(encode({'api_version':VERSION,'body':request}))
            if read(Path('utterances')/ident/'intent.json'):
                result[part]=script;break
    return result


def build():
    """Freeze exact future obligations without modifying any paid source text."""
    if read('control.json',{}).get('status')!='paused':raise RuntimeError('Pause generation before reallocating the remaining corpus')
    if read('vocabulary/plan.json'):raise RuntimeError('Vocabulary plan already exists; do not replace active contracts')
    p=read('plan.json');catalog=terms();snapshot=read('domain-coverage.json',{})
    baseline={l:Counter() for l in ('cs','sk')};slots={l:[] for l in baseline};legacy={}
    for term in catalog:
        for lang in baseline:
            if term['family']=='scio':n=snapshot.get('scio',{}).get(term['name'],{}).get(lang,{}).get('occurrences',0)
            else:n=snapshot.get('ai',{}).get(lang,{}).get('terms',{}).get(term['name'],{}).get('occurrences',0)
            baseline[lang][term['id']]=n
    compiled={(t['id'],l):pattern(t['forms'][l]) for t in catalog for l in baseline}
    pending=[]
    for s in p['sessions']:
        sid=s['id'];lang=s['language'][0]
        if read(Path('sessions')/sid/'receipt.json'):continue
        pending.append(s);locked=locked_parts(s);legacy[sid]=sorted(locked)
        elapsed=0.;next_part=0
        for part,script in sorted(locked.items()):
            text=' '.join(t['text'] for t in script['turns'])
            for term in catalog:baseline[lang][term['id']]+=len(compiled[(term['id'],lang)].findall(text))
            complete=read(Path('sessions')/sid/'parts'/f'{part:03d}'/'complete.json')
            elapsed+=(complete or {}).get('duration_seconds',script.get('target_seconds',300))
            next_part=max(next_part,part+1)
        remaining=max(0,s['planned_seconds']-elapsed)
        if remaining<20:continue
        while remaining>0:
            seconds=remaining if remaining<=360 else 300
            words=max(60,round(seconds/60*145))
            slots[lang].append({'session_id':sid,'part':next_part,'seconds':seconds,'words':words,
                'capacity':int(words*MAX_DENSITY),'used':0,'terms':{},'speaker_count':s['speaker_count'],
                'original_family':s['topic_group'],'original_topic':s['topic']})
            next_part+=1;remaining-=seconds
    demands={l:{t['id']:max(0,t['target_per_language']-baseline[l][t['id']]) for t in catalog} for l in baseline}
    by_id={t['id']:t for t in catalog};report={};assigned_total={l:Counter() for l in baseline}
    for lang,parts in slots.items():
        # Cost uses the shortest valid written form; required inflections receive
        # their own extra word budget below. Completed audio is never reallocated.
        costs={t['id']:min(len(f.split()) for f in t['forms'][lang]) for t in catalog}
        need=sum(demands[lang][k]*costs[k] for k in costs);capacity=sum(s['capacity'] for s in parts)
        if need>capacity:raise RuntimeError(f'{lang}: vocabulary needs {need} words but only {capacity} reserved words remain')
        family_need=Counter()
        for k,n in demands[lang].items():family_need[by_id[k]['family']]+=n*costs[k]
        allocated_capacity=Counter()
        # Reserve whole parts for one domain before allocating terms. This avoids
        # filling technical discussions with unrelated leftover product names.
        for slot in sorted(parts,key=lambda s:(-s['capacity'],s['session_id'],s['part'])):
            family=min(family_need,key=lambda f:allocated_capacity[f]/max(1,family_need[f]))
            slot['family']=family;allocated_capacity[family]+=slot['capacity']
        if any(allocated_capacity[f]<n for f,n in family_need.items()):raise RuntimeError('Insufficient capacity in a coherent vocabulary domain')
        per_session=defaultdict(Counter);form_cursor=Counter();steps=0
        # Fill large and small parts proportionally. Rare terms and new recording
        # contexts win ties; repeated labels never satisfy distinct occurrences.
        while any(demands[lang].values()):
            available=[s for s in parts if s['capacity']-s['used']>=1 and len(s['terms'])<max(3,round(s['words']/24))]
            if not available:raise RuntimeError(f'{lang}: no room for remaining vocabulary obligations: '+str({k:v for k,v in demands[lang].items() if v}))
            slot=min(available,key=lambda s:(s['used']/s['capacity'],s['part'],s['session_id']))
            options=[k for k,n in demands[lang].items() if n>0 and by_id[k]['family']==slot['family'] and k not in slot['terms'] and costs[k]<=slot['capacity']-slot['used']]
            if not options:
                slot['capacity']=slot['used'];continue
            # Keep each block coherent by preferring its domain and nearby topic
            # families, while the largest global deficits still get priority.
            chosen=min(options,key=lambda k:(
                bool(slot['terms']) and by_id[k]['family']!=by_id[next(iter(slot['terms']))]['family'],
                bool(slot['terms']) and by_id[k]['group']!=by_id[next(iter(slot['terms']))]['group'],
                per_session[slot['session_id']][k]>0,
                -demands[lang][k]/by_id[k]['target_per_language'],
                per_session[slot['session_id']][k],k))
            term=by_id[chosen];cap=slot['capacity']-slot['used']
            count=min(demands[lang][chosen],6,max(1,round(slot['words']/120)),cap//costs[chosen])
            short=[f for f in term['forms'][lang] if len(f.split())==costs[chosen]]
            forms=term['forms'][lang];idx=form_cursor[chosen]
            # Rotate every listed form when room permits; otherwise preserve the
            # shortest natural inflection without exceeding the spoken-word budget.
            selected=forms[idx%len(forms)];extra=len(selected.split())-costs[chosen]
            if count*costs[chosen]+extra>cap:selected=short[idx%len(short)];extra=0
            form_cursor[chosen]+=1
            item={'id':chosen,'name':term['name'],'family':term['family'],'forms':forms,
                  'minimum':count,'required_forms':[selected],'description':term['description'],
                  'sources':term['sources'],'word_cost':count*costs[chosen]+extra}
            slot['terms'][chosen]=item;slot['used']+=item['word_cost'];demands[lang][chosen]-=count
            assigned_total[lang][chosen]+=count;per_session[slot['session_id']][chosen]+=count;steps+=1
            if steps>100000:raise RuntimeError('Allocation did not converge')
        report[lang]={'remaining_planned_hours':sum(s['seconds'] for s in parts)/3600,
            'planned_spoken_words':sum(s['words'] for s in parts),'mandatory_vocabulary_words':sum(s['used'] for s in parts),
            'mandatory_word_fraction':sum(s['used'] for s in parts)/sum(s['words'] for s in parts),
            'parts':len(parts),'terms':[{'id':t['id'],'name':t['name'],'family':t['family'],
                'target':t['target_per_language'],'baseline':baseline[lang][t['id']],
                'new_required':assigned_total[lang][t['id']],
                'projected_minimum':baseline[lang][t['id']]+assigned_total[lang][t['id']],
                'recordings_with_new_obligations':sum(t['id'] in v for v in per_session.values())} for t in catalog]}
    # Nothing is published until every term in both languages has an allocation.
    for lang,parts in slots.items():
        for s in parts:
            c={'revision':REVISION,'session_id':s['session_id'],'part':s['part'],'language':[lang],
               'planned_seconds':s['seconds'],'target_words':s['words'],'mandatory_word_cost':s['used'],
               'terms':list(s['terms'].values()),'original_topic':s['original_topic'],
               'original_family':s['original_family'],'speaker_count':s['speaker_count']}
            c['sha256']=digest(encode(c));save(Path('vocabulary/contracts')/s['session_id']/f'part-{s["part"]:03d}.json',c)
    result={'revision':REVISION,'created_at':time.time(),'scio_target_range_per_language':[100,200],'ai_target_range_per_language':[25,50],
        'catalog_counts':dict(Counter(t['family'] for t in catalog)),
        'baseline_report_timestamp':snapshot.get('updated_at'),'baseline_note':'Latest completed-label audit plus immutable partially submitted scripts; conservative if newer completed labels are absent from the audit.',
        'legacy_parts':legacy,'languages':report,'max_planned_vocabulary_word_fraction_per_part':MAX_DENSITY,
        'max_required_occurrences_of_one_term_per_new_part':6,
        'paid_audio_preserved':True,'total_planned_audio_hours':60}
    save('vocabulary/catalog.json',catalog);save('vocabulary/plan.json',result)
    save('history/plan-before-vocabulary-v4.json',p)
    p['vocabulary_revision']=REVISION;p['script_workers']=12;p['tts_workers']=15
    p['content_policy']='Remaining recordings prioritize corpus-wide Scio/AI obligations in varied QDA settings; already generated general QDA is retained.'
    save('plan.json',p)
    return result


if __name__=='__main__':
    r=build()
    print(json.dumps({'catalog_counts':r['catalog_counts'],'languages':{l:{k:v for k,v in d.items() if k!='terms'} for l,d in r['languages'].items()}},ensure_ascii=False,indent=2))
