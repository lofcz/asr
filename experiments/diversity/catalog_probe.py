"""Bounded Czech catalog sampling; metadata only, no audio downloads."""
import json,pathlib,re,time,urllib.request,xml.etree.ElementTree as ET
ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=ROOT.parent/'discovery'
def walk(v):
    if isinstance(v,dict):
        if v.get('feed_url'): yield v
        for w in v.values():yield from walk(w)
    elif isinstance(v,list):
        for w in v:yield from walk(w)
urls=[]
for name in ['podcasts1.txt','podcasts2.txt']:
    urls += [e.text for e in ET.fromstring((SOURCE/name).read_bytes()).iter() if e.tag.endswith('loc')]
seeds=urls[::max(1,len(urls)//12)][:12]
records={};log=[]
for url in seeds:
    try:
        with urllib.request.urlopen(url,timeout=20) as r:body=r.read().decode()
        match=re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',body,re.S)
        data=json.loads(match[1]);count=0
        for v in walk(data):
            records[v['feed_url']]={k:v.get(k) for k in ['title','slug','feed_url','spotify_url','episode_count']}
            records[v['feed_url']]['catalog_language_hint']='cs'
            count+=1
        result={'url':url,'feed_records':count}
    except Exception as e:result={'url':url,'error':str(e)}
    log.append(result);print(json.dumps(result,ensure_ascii=False),flush=True);time.sleep(1)
(ROOT/'czech-show-candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records.values()))
(ROOT/'catalog-probe-log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2))
print('UNIQUE_FEEDS',len(records))
