"""Freeze a machine-validated Soniox pseudo-label experiment, retaining richer evidence.
This does not forge human approval or turn AST proposals into verified event targets.
"""
import pathlib,json,sqlite3,sys,time,collections
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from soniox_durable import Store,atomic,encode,digest,file_hash,verify_done
from soniox_postprocess import normalize,quality,windows,candidate_label
ROOT=pathlib.Path(__file__).resolve().parents[1];REL=ROOT/'datasets/releases/qda-496h-v1';ORIGINAL=ROOT/'datasets/soniox-500h-v1';BACKUP=pathlib.Path('/home/lofcz/.local/state/asr-soniox/soniox-500h-v1')

def main():
 rows=json.loads((REL/'selection.json').read_bytes());store=Store(ROOT,ORIGINAL,BACKUP,'33ba0a9e-6e77-441d-96fc-82f4270b7140');out=REL/'training';out.mkdir(exist_ok=True);dbpath=out/'catalog.build.sqlite3'
 if dbpath.exists():dbpath.unlink()
 db=sqlite3.connect(dbpath);db.executescript('CREATE TABLE sources(record_id TEXT PRIMARY KEY,split TEXT,language TEXT,duration REAL,audio_path TEXT,audio_sha256 TEXT,label_path TEXT,label_sha256 TEXT,label_quality TEXT);CREATE TABLE windows(window_id TEXT PRIMARY KEY,record_id TEXT,split TEXT,duration REAL,source_start REAL,source_end REAL,views_json TEXT,window_json TEXT);CREATE INDEX windows_lookup ON windows(split,duration);')
 sources=[];failed=[];counts=collections.Counter();allwindows=0;keptchars=0;checksums={};original_checksums={}
 for i,r in enumerate(rows):
  state=store.state(r);verify_done(store,r,state);folder=ORIGINAL/'records'/r['record_id'];rawbytes=(folder/'soniox.raw.json').read_bytes();raw=json.loads(rawbytes)
  assert raw['id']==state['transcription_id']
  canonical=REL/'canonical'/f"{r['record_id']}.json"
  if canonical.exists():
   c=json.loads(canonical.read_bytes());assert c['original_sha256']==r['sha256'] and c['canonical_decode_clean'] and c['all_tokens_fit_decoded_audio'];audio_path=c['canonical_path'];audio_hash=c['canonical_sha256'];actual_path=REL/audio_path;vad=json.loads((REL/'records'/r['record_id']/'vad.json').read_bytes());provenance={'kind':'lossless_container_recovery','receipt':'canonical/'+canonical.name,'decoded_pcm_sha256':c['pcm_sha256']}
  else:
   audio_path=r['path'];audio_hash=r['sha256'];actual_path=ROOT/audio_path;vad=json.loads((folder/'vad.json').read_bytes());provenance={'kind':'original_encoded_audio'}
  assert vad['audio_sha256']==audio_hash and vad['full_decode_verified'];assert file_hash(actual_path)==audio_hash
  duration=vad['decoded_seconds'];segments,valid,metrics,issues=normalize(raw['tokens'],duration);coverage,more=quality(valid,segments,metrics,duration,vad,r['language']);issues+=more
  blocking=[p for p in issues if p['severity']=='blocking']
  if blocking:failed.append({'record_id':r['record_id'],'issues':blocking});continue
  # This named baseline supervises speech/diarization only. Event evidence remains separate.
  spoken=[s for s in segments if not s['is_event']];ordered,win=windows(spoken,duration,r['record_id'])
  for w in win:
   label=candidate_label(ordered,w,[]);assert all(0<=s['start']<=s['end']<=w['audio_duration']+.001 for s in label['segments'])
  quality_info={'label_quality':'machine_validated_pseudo_label','human_reviewed':False,'speech_teacher':'Soniox stt-async-v5','speech_diarization_profile_ready':True,'verified_acoustic_labels':False,'review_flags':issues,'metrics':metrics,'coverage':coverage,'audio_transformation':provenance}
  label_path=f"training/labels/{r['record_id']}.json";payload={'record_id':r['record_id'],'audio_path':audio_path,'audio_sha256':audio_hash,'duration':duration,'split':r['split'],'segments':ordered,'customized_context':[],'profile':'speech-diarization-pseudo-v1','quality':quality_info,'source_raw_sha256':digest(rawbytes),'source_transcription_id':state['transcription_id'],'source_original_audio_sha256':r['sha256']};data=encode(payload);atomic(REL/label_path,data);checksums[label_path]=digest(data)
  window_path=f"training/windows/{r['record_id']}.jsonl";blob=b''.join((json.dumps(w,ensure_ascii=False,separators=(',',':'))+'\n').encode() for w in win);atomic(REL/window_path,blob);checksums[window_path]=digest(blob)
  entry={'record_id':r['record_id'],'split':r['split'],'language':r['language'],'show':r['show'],'title':r['title'],'audio_path':audio_path,'audio_sha256':audio_hash,'audio_duration':duration,'original_audio_path':r['path'],'original_audio_sha256':r['sha256'],'label_path':label_path,'label_sha256':digest(data),'windows_path':window_path,'windows_sha256':digest(blob),'raw_transcript_path':f"records/{r['record_id']}/soniox.raw.json",'raw_transcript_sha256':digest(rawbytes),'label_quality':'machine_validated_pseudo_label','acoustic_targets_status':'pending verification','profile':'speech-diarization-pseudo-v1'};sources.append(entry)
  with db:
   db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?)',(r['record_id'],r['split'],r['language'],duration,audio_path,audio_hash,label_path,digest(data),'machine_validated_pseudo_label'))
   for w in win:db.execute('INSERT INTO windows VALUES(?,?,?,?,?,?,?,?)',(w['window_id'],r['record_id'],r['split'],w['audio_duration'],w['source_start'],w['source_end'],json.dumps(w['curriculum_views']),json.dumps(w)))
  checksums[audio_path]=audio_hash;original_checksums[r['path']]=r['sha256'];allwindows+=len(win);keptchars+=sum(len(s['text']) for s in ordered);counts[r['split']]+=duration/3600
  if (i+1)%50==0:print(json.dumps({'processed':i+1,'windows':allwindows}),flush=True)
 db.close()
 if failed:atomic(REL/'finalization-errors.json',encode(failed));raise RuntimeError(f'{len(failed)} records failed structural validation; refusing release')
 dbpath.replace(out/'catalog.sqlite3');checksums['training/catalog.sqlite3']=file_hash(out/'catalog.sqlite3');sourcesbytes=b''.join((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode() for r in sources);atomic(out/'sources.jsonl',sourcesbytes);checksums['training/sources.jsonl']=digest(sourcesbytes)
 summary={'release':'qda-496h-v1','profile':'speech-diarization-pseudo-v1','records':len(sources),'unique_decoded_hours':sum(r['audio_duration'] for r in sources)/3600,'original_manifest_hours':sum(r['seconds'] for r in rows)/3600,'split_hours':dict(counts),'window_count':allwindows,'normalized_speech_characters':keptchars,'recovered_containers':len(list((REL/'canonical').glob('*.json'))),'training_ready_profiles':['speech-diarization-pseudo-v1'],'human_reviewed':False,'rich_event_profile_ready':False,'event_status':'Full-timeline AST evidence generated separately; no unverified events promoted into targets. Do not describe this baseline as fully event-supervised.','customized_context':'Empty by default; provenance-checked context can be added in a separate profile.','curriculum':'Source-shared 2/10/30/60-minute candidate views; actual tokenizer counts still required before batching.','selection_sha256':(REL/'selection.sha256').read_text().strip(),'created_at':time.time()};atomic(REL/'release.json',encode(summary));atomic(REL/'checksums.training.json',encode(checksums));atomic(REL/'checksums.original-audio.json',encode(original_checksums));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
