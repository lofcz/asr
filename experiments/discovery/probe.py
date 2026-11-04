"""Bounded catalog/RSS discovery probe. Run from the asr repository root."""
import json, pathlib, time, urllib.request, xml.etree.ElementTree as ET
ROOT = pathlib.Path('experiments/discovery')
def walk(v):
    if isinstance(v, dict):
        if v.get('feed_url'): yield v
        for w in v.values(): yield from walk(w)
    elif isinstance(v, list):
        for w in v: yield from walk(w)
def seconds(value):
    try:
        result = 0
        for part in value.split(':'): result = result * 60 + float(part)
        return result
    except (ValueError, AttributeError): return None
shows = {v['feed_url']: v for filename in ['show.json', 'proti-proudu.json'] for v in walk(json.loads((ROOT/filename).read_text()))}
rows, metrics = [], []
for i, (url, show) in enumerate(shows.items()):
    start = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=30) as response: body = response.read()
        (ROOT/f'feed-{i}.xml').write_bytes(body)
        items = ET.fromstring(body).findall('./channel/item')
        found = []
        for item in items:
            enc = item.find('enclosure')
            if enc is None: continue
            duration = item.findtext('{http://www.itunes.com/dtds/podcast-1.0.dtd}duration')
            found.append(dict(show=show['title'], feed_url=url, spotify_show_url=show.get('spotify_url'), title=item.findtext('title'), guid=item.findtext('guid'), audio_url=enc.get('url'), duration_seconds=seconds(duration), declared_bytes=enc.get('length')))
        rows.extend(found)
        metric=dict(show=show['title'], episodes=len(found), known_hours=sum(x['duration_seconds'] or 0 for x in found)/3600, missing_duration=sum(x['duration_seconds'] is None for x in found), seconds=round(time.monotonic()-start,3))
    except Exception as e: metric=dict(show=show['title'],error=str(e))
    metrics.append(metric);print(json.dumps(metric,ensure_ascii=False),flush=True)
    time.sleep(1)
(ROOT/'episodes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(ROOT/'feed-metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
print('TOTAL',len(rows),'HOURS',sum(r['duration_seconds'] or 0 for r in rows)/3600)
