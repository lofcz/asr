"""Pre-spend checks against the actual corpus contracts and retained requests."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy,json,threading,time
from unittest.mock import patch
from core import DATA,REPLICA,read,save,digest,encode
from vocabulary import validate_contract,pattern,MAX_DENSITY
from cartesia import request_for,VERSION
from concurrency import TTSLimit,BoundedPool
from export_training import canonical_words


def verify():
    p=read('plan.json');vp=read('vocabulary/plan.json');cat=read('vocabulary/catalog.json')
    assert Counter(t['family'] for t in cat)=={'scio':52,'ai':381}
    by_id={t['id']:t for t in cat};counts=defaultdict(Counter);parts=0
    session_map={s['id']:s for s in p['sessions']}
    for file in (DATA/'vocabulary/contracts').glob('*/part-*.json'):
        c=json.loads(file.read_text());lang=c['language'][0];parts+=1
        if c['session_id'] not in vp['legacy_parts']:continue
        assert c['session_id'] in session_map
        assert (REPLICA/file.relative_to(DATA)).read_bytes()==file.read_bytes()
        assert len({t['id'] for t in c['terms']})==len(c['terms'])
        assert len({t['family'] for t in c['terms']})<=1
        assert c['mandatory_word_cost']<=c['target_words']*MAX_DENSITY
        assert all(t['minimum']<=6 for t in c['terms'])
        assert c['part'] not in vp['legacy_parts'].get(c['session_id'],[])
        spoken=[]
        for term in c['terms']:
            counts[lang][term['id']]+=term['minimum']
            spoken.extend(term['required_forms'])
            spoken.extend([min(term['forms'],key=len)]*term['minimum'])
        dummy={'turns':[{'text':'. '.join(spoken)}]}
        validate_contract(dummy,c)
        if c['terms']:
            try:validate_contract({'turns':[{'text':'Běžná věta bez povinné terminologie.'}],'continuity_notes':dummy['turns'][0]['text']},c)
            except ValueError:pass
            else:raise AssertionError('Notes can satisfy an audible vocabulary obligation')
        corrupt=copy.deepcopy(c);corrupt['planned_seconds']+=1
        try:validate_contract(dummy,corrupt)
        except ValueError:pass
        else:raise AssertionError('Corrupt contract accepted')
    for lang,rows in vp['languages'].items():
        for t in rows['terms']:
            assert counts[lang][t['id']]==t['new_required']
            assert t['baseline']+counts[lang][t['id']]>=t['target']
    strata=Counter()
    for s in p['sessions']:strata[s['cohort'],s['language'][0],s['speaker_count']]+=s['planned_seconds']
    assert abs(sum(strata.values())-60*3600)<1
    for (cohort,lang,n),seconds in strata.items():assert abs(seconds-(22500 if cohort=='core' else 18000))<1
    assert not pattern(['tři']).search('tri')
    assert pattern(['ScioBot','ScioBotu']).findall('ScioBotu a ScioBot')==['ScioBotu','ScioBot']
    legacy=0
    for path in list((DATA/'utterances').glob('*/request.json'))[:100]:
        old=json.loads(path.read_text())
        if old.get('model_id')!='sonic-3.6-2026-08-27':continue
        if old.get('use_normalized_timestamps') is False:continue
        new=request_for(old['transcript'],old['language'],old['voice']['id'],old['generation_config']['speed'])
        assert new==old
        assert digest(encode({'api_version':VERSION,'body':new}))==path.parent.name
        legacy+=1
    assert legacy>0
    req=request_for('INT8, HTTP 429 a ScioBot+.', 'cs','test',original_timestamps=True)
    assert req['use_normalized_timestamps'] is False
    native=[{'text':x,'start':i,'end':i+.8} for i,x in enumerate(req['transcript'].split())]
    words=canonical_words(req['transcript'],native,2,0)
    assert [w['text'] for w in words]==req['transcript'].split()
    with patch('concurrency.save'):
        limit=TTSLimit(15)
        def request(_):
            with limit.slot(lambda:None):time.sleep(.02)
        with ThreadPoolExecutor(max_workers=30) as pool:list(pool.map(request,range(90)))
        assert limit.peak==15 and limit.active==0
        limit.rejected();assert limit.limit==14
        release=threading.Event();submitted=threading.Event()
        with BoundedPool(2,lambda:None,max_pending=3) as pool:
            for _ in range(3):pool.submit(release.wait)
            def submit_fourth():pool.submit(lambda:None);submitted.set()
            thread=threading.Thread(target=submit_fourth);thread.start()
            assert not submitted.wait(.1)
            release.set();thread.join(5);assert submitted.is_set()
    result={'passed':True,'at':time.time(),'contracts':parts,'legacy_request_hashes_unchanged':legacy,
        'all_term_targets_allocated':True,'coherent_domains_per_part':True,'paid_parts_preserved':True,
        'duration_language_speaker_quotas_preserved':True,'script_notes_cannot_satisfy_quotas':True,
        'tampered_contract_rejected':True,'diacritics_preserved':True,'original_native_timestamp_mapping':True,
        'tts_limit_tested':15,'pending_queue_limit_tested':True}
    save('vocabulary/structural-validation.json',result)
    return result


if __name__=='__main__':print(json.dumps(verify(),indent=2))
