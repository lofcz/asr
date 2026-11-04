"""Count downloaded, present audio; exclude byte-identical and likely title duplicates."""
import collections,datetime,difflib,json,pathlib,re,sqlite3,unicodedata
from audio_worker import atomic_json,storage_ready
HOME=pathlib.Path.home();CONFIG=json.loads((HOME/'.local/state/asr-downloader/config.json').read_text());ROOT=pathlib.Path(CONFIG['root'])
def norm(s):return ' '.join(re.findall(r'[a-z0-9]+',unicodedata.normalize('NFKD',s.lower()).encode('ascii','ignore').decode()))
def tokens(s):return {w for w in norm(s).split() if len(w)>3 and w not in {'podcast','rozhovor','rozhovory','podcastu','jsem','jsou','jsme','nebo','jako','proc','dnes','bude','budou','cesko','slovensko'}}
def main():
 if not storage_ready(CONFIG):print('SSD unavailable');return
 records=[];missing=[]
 for path in (ROOT/'data/audio').glob('*/*/*.json'):
  try:
   r=json.loads(path.read_text());media=ROOT/r['path']
   if media.exists() and media.stat().st_size==r['bytes']:records.append({'source':'rss','id':r['id'],'title':r['title'],'show':r['show'],'language':r['language'],'seconds':r['actual_duration_seconds'],'bytes':r['bytes'],'sha256':r['sha256'],'path':r['path']})
   else:missing.append(str(path))
  except (OSError,ValueError,KeyError):missing.append(str(path))
 db=sqlite3.connect('file:'+str(HOME/'.local/state/asr-downloader/queue.sqlite3')+'?mode=ro',uri=True)
 queue={k:v for k,v in db.execute('SELECT status,count(*) FROM jobs GROUP BY status')}
 for receipt, in db.execute("SELECT receipt FROM jobs WHERE status='done'"):
  r=json.loads(receipt);media=ROOT/r['path'];meta=r['queue_metadata']
  if media.exists() and media.stat().st_size==r['bytes']:records.append({'source':'youtube','id':r['id'],'title':r['youtube_metadata'].get('title') or meta.get('title',''),'show':meta.get('show'),'language':meta.get('language','unknown'),'seconds':r['duration_seconds'],'bytes':r['bytes'],'sha256':r['sha256'],'path':r['path']})
  else:missing.append(str(media))
 db.close();hashes={};seen=[];inverted=collections.defaultdict(set);duplicates=[];kept=[]
 for row in records:
  if row['sha256'] in hashes:
   duplicates.append({'id':row['id'],'duplicate_of':hashes[row['sha256']],'reason':'identical_sha256','hours':row['seconds']/3600});continue
  title=norm(row['title']);ts=tokens(row['title']);counts=collections.Counter(i for t in ts for i in inverted[t]);match=None
  for i,hits in counts.most_common(8):
   earlier=seen[i]
   if len(title)>20 and title==earlier['norm']:match=earlier;break
   if row['source']!=earlier['source'] and hits>=4 and (hits/len(ts|earlier['tokens'])>.66 or difflib.SequenceMatcher(None,title,earlier['norm']).ratio()>.8):match=earlier;break
  if match:
   duplicates.append({'id':row['id'],'duplicate_of':match['id'],'reason':'likely_same_episode_title','hours':row['seconds']/3600});continue
  hashes[row['sha256']]=row['id'];i=len(seen);seen.append({**row,'norm':title,'tokens':ts});kept.append(row)
  for t in ts:inverted[t].add(i)
 totals={}
 for lang in ['cs','sk','unknown']:
  rs=[r for r in kept if r['language']==lang];totals[lang]={'hours':sum(r['seconds'] for r in rs)/3600,'files':len(rs),'shows':len({r['show'] for r in rs})}
 total=sum(r['seconds'] for r in kept)/3600
 report={'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'downloaded_hours_after_known_duplicate_exclusions':total,'target_hours':CONFIG.get('target_hours',1000),'remaining_hours':max(0,CONFIG.get('target_hours',1000)-total),'target_met':total>=CONFIG.get('target_hours',1000),'language_totals':totals,'source_hours':{s:sum(r['seconds'] for r in kept if r['source']==s)/3600 for s in ['rss','youtube']},'files':len(kept),'bytes':sum(r['bytes'] for r in kept),'youtube_queue':queue,'excluded_duplicates':duplicates,'missing_or_invalid_receipts':missing,'note':'Language is source-declared or manually inferred. Speaker counts and audio-content duplicate detection are not verified.'}
 atomic_json(ROOT/'data/manifests/corpus-status.json',report)
 atomic_json(ROOT/'data/manifests/downloaded-corpus.json',kept)
 print(json.dumps({k:v for k,v in report.items() if k not in ['excluded_duplicates','missing_or_invalid_receipts']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
