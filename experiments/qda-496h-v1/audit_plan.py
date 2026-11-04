import sys,pathlib,json,math,collections
sys.path[:0]=['/data/work/asr-datasets/qda-496h-v1/code','/data/work/asr-pilot/long-context-pilot']
from qda_release_dataset import ReleaseDataset,target_json
from model_paths import model_paths
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor,SYSTEM_PROMPT
root=pathlib.Path('/data/work/asr-datasets/qda-496h-v1');out=pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16')
path,tok=model_paths();p=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
plan=json.loads((out/'plan.json').read_text());ds={b:ReleaseDataset(root,'train',b) for b in (600,1800,3600)};intervals=collections.defaultdict(list);lengths=[]
system=p.tokenizer.encode(p.tokenizer.apply_chat_template([{'role':'system','content':SYSTEM_PROMPT}],tokenize=False))
for n,r in enumerate(plan):
 sample=ds[r['bucket']].label(r['index']);intervals[r['record_id']].append((sample['source_start'],sample['source_end']))
 frames=math.ceil(round(sample['audio_duration']*24000)/p.speech_tok_compress_ratio)
 placeholder=p.tokenizer.convert_ids_to_tokens(p.speech_start_id)+p.tokenizer.convert_ids_to_tokens(p.speech_pad_id)*frames+p.tokenizer.convert_ids_to_tokens(p.speech_end_id)
 suffix=f"This is a {sample['audio_duration']:.2f} seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content"
 ids=p.tokenizer.apply_chat_template([{'role':'user','content':placeholder+'\n'+suffix}],tokenize=True)
 target=p.tokenizer.apply_chat_template([{'role':'assistant','content':target_json(sample)}],tokenize=True,add_generation_prompt=False)
 lengths.append({'window_id':r['window_id'],'tokens':len(system)+len(ids)+len(target),'bucket':r['bucket']})
 if n%200==0:print(n,flush=True)
sources={r['record_id']:r for r in map(json.loads,(root/'training/sources.jsonl').read_text().splitlines()) if r['split']=='train'}
assert set(intervals)==set(sources)
for rid,spans in intervals.items():
 spans.sort();assert abs(spans[0][0])<.001 and abs(spans[-1][1]-sources[rid]['audio_duration'])<.001
 assert all(abs(a[1]-b[0])<.001 for a,b in zip(spans,spans[1:])),rid
result={'status':'passed' if max(x['tokens'] for x in lengths)<=65536 else 'failed','sources':len(sources),'windows':len(plan),'max_tokens':max(x['tokens'] for x in lengths),'over_budget':[x for x in lengths if x['tokens']>65536],'coverage':'All training sources exactly once, contiguous windows, no overlapping duplicated time; no holdouts'}
(out/'plan-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
assert result['status']=='passed'
