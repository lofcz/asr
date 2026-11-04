"""Incremental vocabulary counts from committed, hash-paired training labels."""
from collections import Counter,defaultdict
from pathlib import Path
from core import read,save,digest,encode,verified
from vocabulary import pattern,terms,validate_contract,contract


def count_exports(receipts):
    catalog=read('vocabulary/catalog.json') or terms()
    catalog_hash=digest(encode(catalog))
    compiled={(t['id'],lang):pattern(t['forms'][lang]) for t in catalog for lang in ('cs','sk')}
    totals={lang:{t['id']:{'forms':Counter(),'sessions':[],'voices':set()} for t in catalog} for lang in ('cs','sk')}
    for receipt in receipts:
        lang=receipt['language'][0]
        rel=Path('vocabulary/coverage-cache')/catalog_hash/(receipt['label_sha256']+'.json')
        cached=read(rel)
        if cached is None:
            import json
            label=json.loads(verified(receipt['label_path'],receipt['label_sha256']))
            observed={}
            for term in catalog:
                regex=compiled[(term['id'],lang)];forms=Counter();voices=set()
                for segment in label['segments']:
                    matches=list(regex.finditer(segment['text']))
                    if matches:
                        forms.update(m.group() for m in matches)
                        voices.add(receipt['known_speaker_voice_ids'][str(segment['speaker'])])
                if forms:observed[term['id']]={'forms':dict(forms),'voices':sorted(voices)}
            cached={'label_sha256':receipt['label_sha256'],'catalog_sha256':catalog_hash,'terms':observed}
            save(rel,cached)
        for tid,entry in cached['terms'].items():
            dest=totals[lang][tid];dest['forms'].update(entry['forms'])
            dest['sessions'].append(receipt['id']);dest['voices'].update(entry['voices'])
    result={lang:{} for lang in totals}
    for lang in totals:
        for term in catalog:
            entry=totals[lang][term['id']];count=sum(entry['forms'].values());target=term['target_per_language']
            result[lang][term['id']]={'occurrences':count,'sessions':len(entry['sessions']),
                'session_ids':entry['sessions'],'observed_forms':dict(entry['forms']),
                'unique_voices':len(entry['voices']),'target':target,'remaining_to_target':max(0,target-count)}
    return catalog,result
