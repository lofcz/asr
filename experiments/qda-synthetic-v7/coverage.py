"""Coverage of committed training speech, separate from plan and draft promises."""
from collections import Counter,defaultdict
import json,re,time
from pathlib import Path
from core import DATA,HERE,read,save,put
from vocabulary_coverage import count_exports
from vocabulary import contract,validate_contract
from plan import plan


def matcher(forms):
    forms=sorted(set(forms),key=lambda s:(-len(s),s))
    return re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(x) for x in forms)+r')(?!\w)',re.I)


def summarize(pattern,records):
    forms=Counter();sessions=[]
    for sid,text in records:
        matches=list(pattern.finditer(text))
        if matches:
            sessions.append(sid)
            forms.update(m.group() for m in matches)
    return {'occurrences':sum(forms.values()),'sessions':len(sessions),
            'observed_forms':dict(forms),'session_ids':sessions}


def audit():
    products=json.loads((HERE/'scio-products.json').read_text())
    ai=json.loads((HERE/'ai-domain-coverage.json').read_text())
    p=plan();specs={s['id']:s for s in p['sessions']};receipts=[]
    records=defaultdict(list);audio_seconds=0.;by_family=Counter();voices=set()
    for path in sorted((DATA/'training/receipts').glob('*.json')):
        r=json.loads(path.read_text());receipts.append(r);label=read(r['label_path'])
        text=' '.join(s['text'] for s in label['segments'])
        records[r['language'][0]].append((r['id'],text));audio_seconds+=r['audio_duration']
        spec=specs.get(r['id'],{});family=spec.get('topic_group','unknown')
        if family=='scio_ai':family='scio'
        by_family[family]+=r['audio_duration']/3600
        voices.update(r['known_speaker_voice_ids'].values())
    planned=defaultdict(lambda:Counter());planned_forms=defaultdict(set)
    required_checks=0;required_failures=[];frequency_checks=0;frequency_failures=[]
    for sid,s in specs.items():
        lang=s['language'][0]
        if s.get('canonical_product'):
            planned[lang][s['canonical_product']]+=1
            planned_forms[(lang,s['canonical_product'])].update(t for batch in s.get('required_terms_by_part',[]) for t in batch)
        for path in (DATA/'scripts'/sid).glob('part-*.json'):
            if not re.fullmatch(r'part-\d+\.json',path.name):continue
            part=json.loads(path.read_text());text=' '.join(t['text'] for t in part['turns'])
            for term in part.get('required_terms',[]):
                required_checks+=1
                if not matcher([term]).search(text):required_failures.append({'session':sid,'part':part['part'],'term':term})
            if part.get('minimum_brand_mentions'):
                frequency_checks+=1
                mentions=len(list(matcher(s['required_term_forms']).finditer(text)))
                if mentions<part['minimum_brand_mentions']:
                    frequency_failures.append({'session':sid,'part':part['part'],'actual':mentions,'minimum':part['minimum_brand_mentions']})
    catalog,actual=count_exports(receipts)
    vocabulary_checks=0;vocabulary_failures=[]
    for path in (DATA/'scripts-v4').glob('*/part-*.json'):
        if not re.fullmatch(r'part-\d+\.json',path.name):continue
        part=json.loads(path.read_text());c=contract(part['session_id'],part['part'])
        try:
            if not c or part.get('vocabulary_contract_sha256')!=c['sha256']:raise ValueError('Wrong vocabulary contract')
            validate_contract(part,c);vocabulary_checks+=1
        except ValueError as e:vocabulary_failures.append({'session':part['session_id'],'part':part['part'],'error':str(e)})
    scio={}
    for product in [t for t in catalog if t['family']=='scio']:
        name=product['name'];entry={'sources':product['sources']}
        for lang in ('cs','sk'):
            entry[lang]={**actual[lang][product['id']],
                'planned_dedicated_sessions':planned[lang][name],
                'planned_required_forms':sorted(planned_forms[(lang,name)])}
        scio[name]=entry
    terms=[t for t in catalog if t['family']=='ai']
    ai_counts={}
    for lang in ('cs','sk'):
        counts={term['name']:actual[lang][term['id']] for term in terms}
        ai_counts[lang]={'terms_with_occurrences':sum(bool(v['occurrences']) for v in counts.values()),
            'catalog_terms':len(terms),'terms':counts}
    result={'updated_at':time.time(),'basis':'Only completed VibeVoice training exports; not prompts, attempted scripts or metadata',
        'completed_sessions':sum(len(v) for v in records.values()),'completed_hours':audio_seconds/3600,
        'completed_hours_by_family':dict(by_family),'actual_voices_used':len(voices),
        'catalog_scio_items':len(products),'catalog_ai_topics':len(ai),
        'required_term_checks':required_checks,'required_term_failures':required_failures,
        'brand_frequency_checks':frequency_checks,'brand_frequency_failures':frequency_failures,
        'vocabulary_contract_checks':vocabulary_checks,'vocabulary_contract_failures':vocabulary_failures,
        'vocabulary_targets_fulfilled':all(v['remaining_to_target']==0 for rows in actual.values() for v in rows.values()),
        'vocabulary_targets':{'scio_range_per_family_per_language':[100,200],'ai_range_per_term_per_language':[25,50],
            'exact_targets':'Per-term target values are included in each vocabulary row.'},
        'scio':scio,'ai':ai_counts,
        'notes':['Counts include exact listed natural inflections; no accent stripping.',
                 'Nested product families may overlap; product counts must not be summed as unique words.',
                 'Lexical coverage verifies transcript labels, not acoustic pronunciation.']}
    save('domain-coverage.json',result)
    lines=[f"Completed audio: {result['completed_hours']:.3f} h in {result['completed_sessions']} recordings.",
           f"Verified required terms: {required_checks}; failures: {len(required_failures)}.",'',
           '| Product | Czech occurrences / recordings | Slovak occurrences / recordings |',
           '|---|---:|---:|']
    for name,entry in scio.items():
        lines.append(f"| {name} | {entry['cs']['occurrences']} / {entry['cs']['sessions']} | {entry['sk']['occurrences']} / {entry['sk']['sessions']} |")
    lines+=['','Counts are from completed training labels. Full per-form counts, planned requirements and official sources are in domain-coverage.json.']
    put('domain-coverage.md',('\n'.join(lines)+'\n').encode())
    return result


if __name__=='__main__':
    r=audit();print(json.dumps({k:r[k] for k in ['completed_sessions','completed_hours','required_term_checks','required_term_failures']},ensure_ascii=False))
