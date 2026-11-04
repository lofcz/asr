import sys,json,pathlib,math,hashlib
sys.path[:0]=['/data/work/asr-datasets/qda-496h-v1/code','/data/work/asr-pilot/long-context-pilot']
from qda_release_dataset import ReleaseDataset,target_json
from model_paths import model_paths
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor,SYSTEM_PROMPT
root=pathlib.Path('/data/work/asr-datasets/qda-496h-v1');out=pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16');dest=pathlib.Path('/data/work/asr-optimization');dest.mkdir(exist_ok=True)
path,tok=model_paths();p=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
plan=json.loads((out/'plan.json').read_text());ds={b:ReleaseDataset(root,'train',b) for b in (600,1800,3600)}
system=p.tokenizer.encode(p.tokenizer.apply_chat_template([{'role':'system','content':SYSTEM_PROMPT}],tokenize=False));lengths={}
for r in plan:
 sample=ds[r['bucket']].label(r['index']);frames=math.ceil(round(sample['audio_duration']*24000)/p.speech_tok_compress_ratio)
 placeholder=p.tokenizer.convert_ids_to_tokens(p.speech_start_id)+p.tokenizer.convert_ids_to_tokens(p.speech_pad_id)*frames+p.tokenizer.convert_ids_to_tokens(p.speech_end_id)
 suffix=f"This is a {sample['audio_duration']:.2f} seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content"
 ids=p.tokenizer.apply_chat_template([{'role':'user','content':placeholder+'\n'+suffix}],tokenize=True)
 target=p.tokenizer.apply_chat_template([{'role':'assistant','content':target_json(sample)}],tokenize=True,add_generation_prompt=False)
 lengths[r['window_id']]={'tokens':len(system)+len(ids)+len(target),'target_tokens':len(target),'audio_tokens':frames}
assert max(x['tokens'] for x in lengths.values())<=65536
(dest/'lengths.json').write_text(json.dumps(lengths));print('lengths',len(lengths),'max',max(x['tokens'] for x in lengths.values()),flush=True)
