"""Select a broad balanced queue from reviewed podcast catalogs; no downloads here."""
import collections,difflib,fcntl,hashlib,json,os,pathlib,re,unicodedata
ROOT=pathlib.Path(__file__).resolve().parents[1];P=ROOT/'experiments/youtube';QUEUE=ROOT/'queue/youtube.jsonl'
def norm(text):return ' '.join(re.findall(r'[a-z0-9]+',unicodedata.normalize('NFKD',text.lower()).encode('ascii','ignore').decode()))
def tokens(text):return {w for w in norm(text).split() if len(w)>3 and w not in {'podcast','rozhovor','rozhovory','podcastu','kterej','ktery','kterou','jsem','jsou','jsme','nebo','tento','tato','jako','proc','dnes','bude','budou','cesko','slovensko'}}
def duration(text):
    if not text:return 0
    total=sum(int(n)*unit for pattern,unit in [(r'(\d+) hr',3600),(r'(\d+) min',60),(r'(\d+) sec',1)] for n in re.findall(pattern,text))
    if not total and re.fullmatch(r'[0-9:]+',text):
        for n in text.split(':'):total=total*60+int(n)
    return total

def main():
    reviews=json.loads((P/'expanded/review.json').read_text())
    # Owners and sample episode titles are reviewed before this allowlist is used.
    approved=json.loads((ROOT/'queue/approved-shows.json').read_text())
    by_id={r['browse_id']:r for r in approved};known=[]
    for line in (ROOT/'data/manifests/download-results.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['status']=='downloaded':known.append({'title':r['title'],'norm':norm(r['title']),'tokens':tokens(r['title']),'id':r['id']})
    inverted=collections.defaultdict(set)
    for i,r in enumerate(known):
        for t in r['tokens']:inverted[t].add(i)
    def duplicate(title):
        ts=tokens(title);counts=collections.Counter(i for t in ts for i in inverted[t]);n=norm(title)
        for i,hits in counts.most_common(8):
            r=known[i]
            if n==r['norm']:return r
            if hits>=4 and (hits/len(ts|r['tokens'])>.66 or difflib.SequenceMatcher(None,n,r['norm']).ratio()>.8):return r
        return None
    existing=[]
    for line in QUEUE.read_text().splitlines():
        if line.startswith('{'):existing.append(json.loads(line))
    manual={r['id']:r for r in json.loads((P/'manually-selected.json').read_text())}
    for r in existing:
        if r['id'] in manual:
            for k,v in manual[r['id']].items():r.setdefault(k,v)
    seen={r['id'] for r in existing};used=collections.Counter();totals=collections.Counter();counts=collections.Counter();excluded=[];groups=collections.defaultdict(list)
    title_seen={norm(r.get('title','')) for r in existing}
    for r in existing:
        sec=r.get('estimated_duration_seconds',0);totals[r.get('language')]+=sec
        key=r.get('playlist_id',r.get('show'));used[key]+=sec
    for review in reviews:
        key=review['browse_id']
        if key not in by_id:continue
        show=by_id[key];data=json.loads((P/'expanded'/(key+'.json')).read_text());group=show.get('quota_group',key)
        for episode in data.get('episodes',[]):
            video=episode.get('videoId');sec=duration(episode.get('duration'));title=episode.get('title','')
            if not video or video in seen or not 15*60<=sec<=150*60:continue
            if re.search(r'\b(trailer|teaser|ukázka|upoutávka|ochutnávka|sestřih|zostrih|best of|highlights|shorts)\b',title,re.I):continue
            english={'what','why','how','the','with','your','inside','humanity','world','and','future','about','life','our'}&set(norm(title).split())
            if len(english)>=4:continue
            match=duplicate(title)
            if match:
                excluded.append({'youtube_id':video,'title':title,'rss_id':match['id'],'rss_title':match['title']});continue
            row={'id':video,'language':show['language'],'language_verified':False,'show':show['title'],'channel':review.get('author'),'title':title,'playlist_id':key,'quota_group':group,'estimated_duration_seconds':sec,'speaker_count_hint':show.get('speaker_count_hint',2),'speaker_count_verified':False,'selection':'Reviewed show/channel; duration-filtered, sampled across archive; likely RSS duplicates excluded'}
            groups[key].append(row)
        groups[key].sort(key=lambda r:hashlib.sha256(('expansion-500h|'+r['id']).encode()).hexdigest())
    group_used=collections.Counter()
    for r in existing:
        show=by_id.get(r.get('playlist_id'),{});group_used[show.get('quota_group',r.get('playlist_id',r.get('show')))]+=r.get('estimated_duration_seconds',0)
    targets={'cs':280*3600,'sk':120*3600};selected={'cs':[],'sk':[]}
    for lang,target in targets.items():
        keys=[k for k in groups if by_id[k]['language']==lang]
        while True:
            progress=False
            for key in keys:
                show=by_id[key];group=show.get('quota_group',key);cap=show['max_hours']*3600
                while groups[key]:
                    row=groups[key].pop(0);sec=row['estimated_duration_seconds'];title=norm(row['title'])
                    if row['id'] in seen or title in title_seen or totals[lang]+sec>target or group_used[group]+sec>cap:continue
                    selected[lang].append(row);seen.add(row['id']);title_seen.add(title);totals[lang]+=sec;group_used[group]+=sec;counts[key]+=1;progress=True;break
            if not progress:break
    # Interleave languages by duration and retain round-robin show selection order.
    ordered=[];cs=collections.deque(selected['cs']);sk=collections.deque(selected['sk']);balance=collections.Counter()
    while cs or sk:
        lang='cs' if cs and (not sk or balance['cs']<=.7*sum(balance.values())) else 'sk'
        row=(cs if lang=='cs' else sk).popleft();ordered.append(row);balance[lang]+=row['estimated_duration_seconds']
    with QUEUE.open('a+b') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);latest={json.loads(s)['id'] for s in f.read().decode().splitlines() if s.startswith('{')}
        for row in ordered:
            if row['id'] not in latest:f.write((json.dumps(row,ensure_ascii=False)+'\n').encode());latest.add(row['id'])
        f.flush();os.fsync(f.fileno())
    (P/'expansion-selected.json').write_text(json.dumps(ordered,ensure_ascii=False,indent=2))
    (P/'rss-duplicate-exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
    summary={'new_episodes':len(ordered),'new_shows':len(counts),'new_hours':{k:sum(r['estimated_duration_seconds'] for r in v)/3600 for k,v in selected.items()},'total_youtube_queue_estimated_hours':{k:v/3600 for k,v in totals.items()},'rss_duplicate_candidates_excluded':len(excluded),'show_hours':{k:v/3600 for k,v in group_used.items()}}
    (ROOT/'queue/expansion-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
