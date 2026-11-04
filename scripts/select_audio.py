"""Select a broad raw candidate pool; language is feed-declared, speakers unverified."""
import collections,hashlib,json,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[1];M=ROOT/'data/manifests'
rows=[json.loads(s) for s in (M/'episode-inventory.jsonl').read_text().splitlines()]
selected=[];seen=set();summary={}
for lang,target in [('cs',168*3600),('sk',72*3600)]:
    shows=collections.defaultdict(list)
    for r in rows:
        if r['language']==lang and not re.search(r'pohád|pohad|rozpráv|rozprav|audioknih|bible|bibli[ae]|zprávy|newsfilter|newsletter|rozhlasové hry', r['show'], re.I):shows[r['show_id']].append(r)
    for group in shows.values():group.sort(key=lambda r:hashlib.sha256(('asr-v1|'+r['id']).encode()).hexdigest())
    used=collections.Counter();count=collections.Counter();total=0
    order=sorted(shows,key=lambda k:hashlib.sha256(k.encode()).hexdigest())
    while True:
        progress=False
        for key in order:
            if count[key]>=4:continue
            while shows[key]:
                r=shows[key].pop(0);duration=r['declared_duration_seconds']
                if r['audio_url'] in seen or used[key]+duration>10800 or total+duration>target:continue
                selected.append(r);seen.add(r['audio_url']);used[key]+=duration;total+=duration;count[key]+=1;progress=True;break
        if not progress:break
    summary[lang]={'hours':total/3600,'shows':len(count),'episodes':sum(count.values()),'max_show_hours':max(used.values(),default=0)/3600}
(M/'selected-audio.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
(M/'selection-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
