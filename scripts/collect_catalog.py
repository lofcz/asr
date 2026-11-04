"""Build a cached RSS inventory from discovered shows; standard library only."""
import concurrent.futures, hashlib, json, pathlib, re, time, urllib.request, xml.etree.ElementTree as ET
ROOT=pathlib.Path(__file__).resolve().parents[1]
CACHE=ROOT/'data/rss'; CACHE.mkdir(parents=True,exist_ok=True)
def duration(value):
    try:
        result=0
        for part in value.split(':'): result=result*60+float(part)
        return result
    except (ValueError,AttributeError): return None
def fetch(show):
    key=hashlib.sha256(show['feed_url'].encode()).hexdigest()[:16]; cache=CACHE/(key+'.xml')
    try:
        if cache.exists(): body=cache.read_bytes()
        else:
            with urllib.request.urlopen(show['feed_url'],timeout=15) as r:body=r.read(12*1024*1024)
            cache.write_bytes(body)
        channel=ET.fromstring(body).find('channel')
        if channel is None: raise ValueError('No RSS channel')
        lang=(channel.findtext('language') or '').strip().lower().replace('_','-')
        lang={'ces':'cs','cze':'cs','slk':'sk','slo':'sk'}.get(lang,lang.split('-')[0])
        records=[]
        for item in channel.findall('item'):
            enc=item.find('enclosure')
            if enc is None or not enc.get('url','').startswith(('https://','http://')):continue
            sec=duration(item.findtext('{http://www.itunes.com/dtds/podcast-1.0.dtd}duration'))
            title=item.findtext('title') or ''
            if sec is None or not 1200<=sec<=5400:continue
            if re.search(r'\b(trailer|teaser|ukázka|upoutávka|ochutnávka|zostrih|sestřih)\b',title,re.I):continue
            audio_url=enc.get('url'); guid=item.findtext('guid') or audio_url
            records.append(dict(id=hashlib.sha256((key+'|'+guid).encode()).hexdigest()[:24],show_id=key,show=channel.findtext('title') or show['title'],publisher=show.get('publisher'),feed_url=show['feed_url'],language=lang,language_evidence='rss_channel_declared',language_verified=False,title=title,guid=guid,audio_url=audio_url,declared_duration_seconds=sec,published=item.findtext('pubDate'),speaker_count=None,speaker_ids=None))
        return {'show':show['title'],'show_id':key,'language':lang,'eligible_episodes':len(records)},records
    except Exception as e:return {'show':show['title'],'show_id':key,'error':str(e)},[]
def main():
    shows={}
    for path in sorted((ROOT/'experiments/diversity').glob('*show-candidates.jsonl')):
        for line in path.read_text().splitlines():
            s=json.loads(line);shows[s['feed_url']]=s
    logs=[];records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for log,rows in pool.map(fetch,shows.values()):
            logs.append(log);records.extend(rows)
            print(json.dumps(log,ensure_ascii=False),flush=True)
    (ROOT/'data/manifests/rss-audit.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2))
    (ROOT/'data/manifests/episode-inventory.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    print('TOTAL',len(shows),'FEEDS',len(records),'EPISODES',flush=True)
if __name__=='__main__':main()
