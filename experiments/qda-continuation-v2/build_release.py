"""Publish a checksummed pseudo-label release without upgrading machine labels to gold."""
import collections, json, os, pathlib, shutil, sqlite3, sys, time, random
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from soniox_durable import Store,atomic,encode,digest,file_hash,verify_done
from soniox_postprocess import normalize,quality,windows,candidate_label
RAW=ROOT/'datasets/soniox-50h-v2';REL=ROOT/'datasets/releases/qda-50h-v2';BACKUP=pathlib.Path('/home/lofcz/.local/state/asr-soniox/soniox-50h-v2')
def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():
  if file_hash(src)!=file_hash(dst):raise ValueError('Immutable copy conflict: '+str(dst))
  return
 try:os.link(src,dst)
 except OSError:shutil.copy2(src,dst)
def main():
 rows=json.loads((RAW/'selection.json').read_bytes());store=Store(ROOT,RAW,BACKUP,(ROOT/'.asr-volume-id').read_text().strip(),'asr50-v2')
 old=json.loads((ROOT/'datasets/soniox-500h-v1/selection.json').read_bytes());assert not {r['sha256'] for r in rows}&{r['sha256'] for r in old}
 oldsplits={r['show']:r['split'] for r in old}
 old_release=ROOT/'datasets/releases/qda-496h-v1';old_sources=[json.loads(l) for l in (old_release/'training/sources.jsonl').read_text().splitlines()];old_map={r['record_id']:r for r in old}
 original_val=sorted(r['record_id'] for r in old_sources if r['split']=='validation');random.Random(20261022).shuffle(original_val);used_monitor=set(original_val[:4])
 supplement=[]
 for lang in ('cs','sk'):
  candidates=[r for r in old_sources if r['split']=='validation' and r['language']==lang and 600<=r['audio_duration']<=3600 and r['audio_sha256']==r['record_id'] and r['record_id'] not in used_monitor]
  selected_shows=collections.Counter()
  while candidates and len([r for r in supplement if r['language']==lang])<3:
   def score(r):
    q=json.loads((old_release/r['label_path']).read_text())['quality'];return (selected_shows[r['show']],q['coverage'].get('speech_without_transcript_seconds',0)/r['audio_duration'],q['metrics'].get('low_confidence_token_fraction',0),r['record_id'])
   r=min(candidates,key=score);candidates.remove(r);selected_shows[r['show']]+=1;supplement.append({**old_map[r['record_id']],'origin':'qda-496h-v1-validation'})
 old_store=Store(ROOT,ROOT/'datasets/soniox-500h-v1',pathlib.Path('/home/lofcz/.local/state/asr-soniox/soniox-500h-v1'),(ROOT/'.asr-volume-id').read_text().strip())
 for r in supplement:verify_done(old_store,r,old_store.state(r))
 assert all(oldsplits.get(r['show'],r['split'])==r['split'] for r in rows)
 for r in rows:
  s=store.state(r)
  if s.get('phase')!='done' or not s.get('remote_deleted'):raise RuntimeError('Paid processing/cleanup incomplete')
  verify_done(store,r,s)
  if not (RAW/'records'/r['record_id']/'vad.json').exists():raise RuntimeError('VAD pending')
 REL.mkdir(parents=True,exist_ok=True)
 if (REL/'verification.json').exists():raise RuntimeError('Release already verified; immutable')
 for name in ('selection.json','selection.sha256','selection-summary.json','selection-exclusions.json','inventory.snapshot.json'):
  link(RAW/name,REL/name)
 out=REL/'training';out.mkdir(exist_ok=True);tmp=out/'catalog.build.sqlite3'
 if tmp.exists():tmp.unlink()
 db=sqlite3.connect(tmp);db.executescript('CREATE TABLE sources(record_id TEXT PRIMARY KEY,split TEXT,language TEXT,duration REAL,audio_path TEXT,audio_sha256 TEXT,label_path TEXT,label_sha256 TEXT,label_quality TEXT);CREATE TABLE windows(window_id TEXT PRIMARY KEY,record_id TEXT,split TEXT,duration REAL,source_start REAL,source_end REAL,views_json TEXT,window_json TEXT);CREATE INDEX windows_lookup ON windows(split,duration);')
 sources=[];excluded=[];checks={};audiohashes={};hours=collections.Counter();review=[]
 for r in rows+supplement:
  rid=r['record_id'];is_old=r.get('origin')=='qda-496h-v1-validation';folder=(old_store.out if is_old else RAW)/'records'/rid
  for p in folder.iterdir():
   if p.is_file():link(p,REL/'records'/rid/p.name)
  link(ROOT/r['path'],REL/r['path']);assert file_hash(REL/r['path'])==r['sha256'];audiohashes[r['path']]=r['sha256']
  rawbytes=(folder/'soniox.raw.json').read_bytes();raw=json.loads(rawbytes);vad=json.loads((folder/'vad.json').read_bytes());state=(old_store if is_old else store).state(r)
  assert raw['id']==state['transcription_id'] and vad['audio_sha256']==r['sha256'] and vad['full_decode_verified']
  duration=vad['decoded_seconds'];segments,tokens,metrics,issues=normalize(raw['tokens'],duration);coverage,more=quality(tokens,segments,metrics,duration,vad,r['language']);issues+=more
  reasons=[p['kind'] for p in issues if p['severity']=='blocking']
  if (metrics.get('low_confidence_token_fraction') or 0)>.20:reasons.append('over_20_percent_low_confidence_tokens')
  if metrics.get('non_cs_sk_character_fraction',0)>.15:reasons.append('over_15_percent_other_languages')
  if coverage.get('speech_without_transcript_seconds',0)>max(30,duration*.05):reasons.append('over_5_percent_missing_speech')
  q={'label_quality':'machine_validated_pseudo_label','human_reviewed':False,'speech_teacher':'Soniox stt-async-v5','speech_diarization_profile_ready':not reasons,'verified_acoustic_labels':False,'review_flags':issues,'metrics':metrics,'coverage':coverage}
  atomic(REL/'quality'/f'{rid}.json',encode(q))
  review.extend({'record_id':rid,'split':r['split'],'language':r['language'],'audio_path':r['path'],**p} for p in issues if p['severity']!='info')
  if reasons:excluded.append({'record_id':rid,'hours':duration/3600,'reasons':reasons});continue
  ordered,win=windows([s for s in segments if not s['is_event']],duration,rid)
  for w in win:candidate_label(ordered,w,[])
  label_path=f'training/labels/{rid}.json';payload={'record_id':rid,'audio_path':r['path'],'audio_sha256':r['sha256'],'duration':duration,'split':r['split'],'segments':ordered,'customized_context':[],'profile':'speech-diarization-pseudo-v1','quality':q,'source_raw_sha256':digest(rawbytes),'source_transcription_id':state['transcription_id'],'source_original_audio_sha256':r['sha256']}
  atomic(REL/label_path,encode(payload));checks[label_path]=file_hash(REL/label_path)
  wp=f'training/windows/{rid}.jsonl';atomic(REL/wp,b''.join((json.dumps(w,ensure_ascii=False)+'\n').encode() for w in win));checks[wp]=file_hash(REL/wp)
  entry={'origin':r.get('origin','soniox-50h-v2'),'record_id':rid,'split':r['split'],'language':r['language'],'show':r['show'],'title':r['title'],'audio_path':r['path'],'audio_sha256':r['sha256'],'audio_duration':duration,'label_path':label_path,'label_sha256':checks[label_path],'windows_path':wp,'windows_sha256':checks[wp],'raw_transcript_path':f'records/{rid}/soniox.raw.json','raw_transcript_sha256':digest(rawbytes),'label_quality':'machine_validated_pseudo_label','profile':'speech-diarization-pseudo-v1'};sources.append(entry);hours[r['split']]+=duration/3600
  with db:
   db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?)',(rid,r['split'],r['language'],duration,r['path'],r['sha256'],label_path,checks[label_path],'machine_validated_pseudo_label'))
   for w in win:db.execute('INSERT INTO windows VALUES(?,?,?,?,?,?,?,?)',(w['window_id'],rid,r['split'],w['audio_duration'],w['source_start'],w['source_end'],json.dumps(w['curriculum_views']),json.dumps(w)))
 db.close();tmp.replace(out/'catalog.sqlite3');atomic(out/'sources.jsonl',b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in sources))
 for name in ('training/catalog.sqlite3','training/sources.jsonl'):checks[name]=file_hash(REL/name)
 atomic(REL/'training-exclusions.json',encode(excluded));atomic(REL/'review-queue.jsonl',b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in review))
 atomic(REL/'evaluation-supplement.json',encode({'origin_release':'qda-496h-v1','origin_manifest_sha256':file_hash(old_release/'training/sources.jsonl'),'excluded_previous_monitor_ids':sorted(used_monitor),'records':supplement,'policy':'Existing validation-only recordings, never training; no new API calls.'}))
 summary={'development_supplement_records':len(supplement),'release':'qda-50h-v2','created_at':time.time(),'profile':'speech-diarization-pseudo-v1','paid_records':len(rows),'paid_hours':sum(r['seconds'] for r in rows)/3600,'eligible_records':len(sources),'split_hours':dict(hours),'excluded_records':len(excluded),'human_reviewed':False,'rich_event_profile_ready':False,'event_policy':'VAD negatives are not labeled silence/music/noise. Preserve gaps and review evidence; same speech/diarization objective as baseline.','customized_context':'Empty in both contenders to isolate continuation recipe; source-only metadata retained.','original_selection_disjoint':True,'show_split_compatibility':True,'selection_sha256':file_hash(REL/'selection.json')}
 atomic(REL/'release.json',encode(summary));atomic(REL/'checksums.training.json',encode(checks));atomic(REL/'checksums.original-audio.json',encode(audiohashes))
 meta={str(p.relative_to(REL)):file_hash(p) for p in REL.rglob('*') if p.is_file() and not str(p.relative_to(REL)).startswith(('data/','training/')) and p.name not in ('checksums.metadata.json','checksums.training.json','checksums.original-audio.json','verification.json')}
 atomic(REL/'checksums.metadata.json',encode(meta));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
