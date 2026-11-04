"""Top up existing RSS collection, prioritizing named interviews and rotating guests."""
import collections,hashlib,json,pathlib,re
from expand_youtube_queue import norm,tokens
ROOT=pathlib.Path(__file__).resolve().parents[1];M=ROOT/'data/manifests'
rows=[json.loads(s) for s in (M/'episode-inventory.jsonl').read_text().splitlines()]
existing=[json.loads(s) for s in (M/'selected-audio.jsonl').read_text().splitlines()]
receipts={json.loads(p.read_text())['id']:json.loads(p.read_text()) for p in (ROOT/'data/audio').glob('*/*/*.json')}
existing=[{**r,**receipts.get(r['id'],{})} for r in existing]
selected=[];seen=set();seen_titles=set();hours=collections.Counter();used=collections.Counter()
for r in existing:
 if r['audio_url'] in seen or r.get('sha256') in seen:continue
 if r.get('sha256'):seen.add(r['sha256'])
 sec=r.get('actual_duration_seconds',r['declared_duration_seconds'])
 seen.add(r['audio_url']);seen_titles.add(norm(r['title']));selected.append(r);hours[r['language']]+=sec;used[r['show_id']]+=sec
yt=[json.loads(s) for s in (ROOT/'queue/youtube.jsonl').read_text().splitlines() if s.startswith('{')]
yt_titles={norm(r.get('title','')) for r in yt};yt_tokens=[tokens(r.get('title','')) for r in yt];inv=collections.defaultdict(set)
for i,ts in enumerate(yt_tokens):
 for t in ts:inv[t].add(i)
def duplicate(title):
 if norm(title) in yt_titles:return True
 ts=tokens(title);c=collections.Counter(i for t in ts for i in inv[t])
 return any(n>=4 and n/len(ts|yt_tokens[i])>.66 for i,n in c.most_common(5))
allow=re.compile(r'rozhovor|host|Osobnost Plus|Blízká setkání|Rothcast|FUCKUP|TAMly|NABLÍZKO|Motoplky|naHoru|gastronomický|Olympijský|badminton|Folk & Country|taekwon|Klub 40|Věda Plus|Kultura Plus|Veda SK|Veda na dosah|HockeySlovakia|STZ podcast|Čo chutí|Logistika|Hardware Development|Alchymisti|PROBUDOBRA|Jdu na jedno|Na vodě|ZA ZRCADLEM|Klub Krásy|Kvér|Automatky|Girls to Girls|Knižní klub|V posteli|Houpačky|ne.herci',re.I)
exclude=re.compile(r've třech|Hlavní zprávy|kandidátkou|Český rozhlas -',re.I)
groups=collections.defaultdict(list);excluded=0
for r in rows:
 if r['language'] not in ('cs','sk') or not allow.search(r['show']) or exclude.search(r['show']):continue
 if r['audio_url'] in seen or norm(r['title']) in seen_titles:continue
 if duplicate(r['title']):excluded+=1;continue
 groups[r['show_id']].append(r)
for g in groups.values():g.sort(key=lambda r:hashlib.sha256(('rss-500h|'+r['id']).encode()).hexdigest())
for lang,target in [('cs',560*3600),('sk',240*3600)]:
 keys=[k for k,v in groups.items() if v and v[0]['language']==lang]
 while True:
  progress=False
  for key in keys:
   while groups[key]:
    r=groups[key].pop(0);sec=r['declared_duration_seconds'];title=norm(r['title'])
    if hours[lang]+sec>target or used[key]+sec>14*3600 or r['audio_url'] in seen or title in seen_titles:continue
    selected.append(r);seen.add(r['audio_url']);seen_titles.add(title);hours[lang]+=sec;used[key]+=sec;progress=True;break
  if not progress:break
import os
temporary=M/'selected-audio.jsonl.tmp'
with temporary.open('w') as f:
 f.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected));f.flush();os.fsync(f.fileno())
temporary.replace(M/'selected-audio.jsonl')
summary={'episodes':len(selected),'estimated_hours':{k:v/3600 for k,v in hours.items()},'shows':len(used),'youtube_duplicates_excluded':excluded}
(M/'rss-expansion-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
