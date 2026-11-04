"""Build a versioned correction from saved source labels and measured CTC alignment."""
import copy,hashlib,json,pathlib,sys
import torch
HERE=pathlib.Path('/data/work/asr-experiments/qda-full-continuous-v8')
DATA=pathlib.Path('/data/work/asr-datasets/qda-full-continuous-v8')
EVIDENCE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),'/data/work/asr-pilot/long-context-pilot','/data/work/asr-pilot','/data/work/asr-pilot/VibeVoice']
from contracts import sha,atomic_json
from dataset import window_segments
from prepare import tokenized_target,materialize_input
from model_paths import model_paths
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
torch.set_num_threads(2)
plan=json.loads((HERE/'plan.json').read_text());matches=[(i,j,w) for i,p in enumerate(plan['probe_pairs']) for j,w in enumerate(p) if w['window_id']=='2ced18b5fd8a3359faaa1694'];assert len(matches)==1
index,rank,old=matches[0];assert index+1==2857
assert json.loads((HERE/'runs/r64-full/latest.json').read_text())['step']<index
source=DATA/old['label_path'];assert sha(source)==old['label_sha256'];label=json.loads(source.read_text());original=copy.deepcopy(label)
align=json.loads((EVIDENCE/'canary-boundary.json').read_text());assert align['audio_sha256']==old['audio_sha256']
words=align['segments'][2]['words'][18:23];assert [w['word'] for w in words]==['Tak','to','pro','mě','bylo.']
for variant in (align['segments'][0]['words'],align['segments'][1]['words'][:5],words):
 assert len(variant)==5 and all(not w['issues'] and w['start']<w['end'] for w in variant)
 assert abs(variant[0]['start']-words[0]['start'])<.2 and abs(variant[-1]['end']-words[-1]['end'])<.2
seg=label['segments'][361];assert seg['text']==' Tak to pro mě bylo.' and seg['speaker']=='1' and seg['start']==seg['end']==1402.71
seg['start']=round(1396+words[0]['start'],2);seg['end']=round(1396+words[-1]['end'],2)
assert old['source_start']<seg['start']<seg['end']<old['source_end']
source_words=[w for w in label['words'] if w['token_indices'][0] in seg['token_indices']];assert len(source_words)==5
for w,a in zip(source_words,words):
 assert w['text'].strip()==a['word'];w['start']=round(1396+a['start'],2);w['end']=round(1396+a['end'],2)
label['timestamp_amendments']=[{'reason':'Zero-duration sentence omitted by legacy crop boundary rule','original_seconds':[1402.71,1402.71],
 'corrected_seconds':[seg['start'],seg['end']],'source_segment':361,'method':'Canary auxiliary CTC FP32, exact Soniox transcript; full-context alignment, two narrower checks',
 'alignment_sha256':sha(EVIDENCE/'canary-boundary.json'),'words_and_speakers_unchanged':True}]
folder=DATA/'amendments/zero-duration-boundary-v1';folder.mkdir(parents=True,exist_ok=True)
atomic_json(folder/'label.json',label)
(folder/'canary-boundary.json').write_bytes((EVIDENCE/'canary-boundary.json').read_bytes())
new=copy.deepcopy(old);new['target']=window_segments(label['segments'],old['source_start'],old['source_end'],label['duration']);assert new['target'][:-1]==old['target']
new.update(label_path=str((folder/'label.json').relative_to(DATA)),label_sha256=sha(folder/'label.json'),amended_from_window_id=old['window_id']);new.pop('reuse_root',None)
model,tok=model_paths();proc=VibeVoiceASRProcessor.from_pretrained(model,language_model_pretrained_name=tok)
body,ids,mask=tokenized_target(proc,new);delta=len(ids)-old['target_tokens'];new.update(target_tokens=len(ids),content_tokens=sum(mask),total_tokens=old['total_tokens']+delta,target_json_sha256=hashlib.sha256(body.encode()).hexdigest())
new['window_id']=digest({'original':old['window_id'],'label':new['label_sha256'],'target':new['target_json_sha256']})[:24]
value={'version':1,'reason':'Restore five words omitted at a zero-duration Soniox boundary; acoustic alignment puts them in the preceding unseen crop',
 'authorization':'User: fix this before we train on it','base_plan_sha256':sha(HERE/'plan.json'),'base_selection_sha256':sha(DATA/'screening-selection.json'),
 'update':index+1,'rank':rank,'original_window_id':old['window_id'],'original_row_sha256':digest(old),'original_label_sha256':old['label_sha256'],
 'corrected_label_file':str(folder/'label.json'),'replacement':new,'restored_turn':new['target'][-1],'target_token_delta':delta,
 'evidence_sha256':{'canary-boundary.json':sha(folder/'canary-boundary.json')},'audio_boundaries_unchanged':True,'original_raw_and_label_preserved':True}
atomic_json(folder/'amendment.json',value)
# Validate the entire source's transcript sequence after the overlay, not just the repaired sentence.
rows=[new if r['window_id']==old['window_id'] else r for p in plan['probe_pairs'] for r in p if r['record_id']==old['record_id']]
observed=' '.join(s['Content'] for r in sorted(rows,key=lambda r:r['source_start']) for s in r['target']).split()
expected=' '.join(s['text'] for s in label['segments']).split();assert observed==expected
assert ' '.join(observed).count('Tak to pro mě bylo.')==1
# Materialize and inspect the actual labels consumed by the trainer. No model/GPU or optimizer call.
actual,_=materialize_input(new,proc,plan['cache_contract_sha256'])
x=torch.load(HERE/'cache'/(new['window_id']+'.input.pt'),map_location='cpu',weights_only=True);n=len(x['ids'])-len(ids)
assert (x['labels'][:n]==-100).all() and torch.equal(x['labels'][n:],torch.tensor(ids))
assert torch.equal(x['ids'][n:],torch.tensor(ids)) and torch.equal(x['content_mask'][n:],torch.tensor(mask,dtype=torch.bool))
assert int((x['labels']!=-100).sum())==new['target_tokens'] and int(x['content_mask'].sum())==new['content_tokens']
assert proc.tokenizer.decode(x['ids'][n:]).count('Tak to pro mě bylo.')==1
atomic_json(HERE/'cache'/(new['window_id']+'.input-plan.json'),new)
verification={'status':'passed','original_window_id':old['window_id'],'replacement_window_id':new['window_id'],
 'update':index+1,'source_phrase_start':seg['start'],'source_phrase_end':seg['end'],'source_word_count':len(expected),'restored_words':5,
 'source_transcript_exact':True,'phrase_count':1,'audio_sha256':old['audio_sha256'],'audio_crop_unchanged':True,'source_speaker':'1','local_speaker':1,
 'target_token_delta':delta,'new_target_tokens':new['target_tokens'],'new_input_sha256':actual['input_sha256'],
 'label_and_content_masks_verified':True,'new_input_total_tokens':actual['total_tokens'],'cache_contract_unchanged':True,'base_plan_sha256':sha(HERE/'plan.json'),
 'amendment_sha256':sha(folder/'amendment.json'),'optimizer_updates':0,'gpu_model_loaded':False}
atomic_json(folder/'verification.json',verification);print(json.dumps(verification),flush=True)
