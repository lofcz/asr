import os,json,pathlib,time,hashlib,math
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import numpy as np,torch,soundfile as sf
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from model_paths import model_paths, MODEL_REVISION
BASE=pathlib.Path(__file__).resolve().parent

def main():
    torch.set_num_threads(4);torch.manual_seed(20261022)
    out=BASE/'cache';out.mkdir(exist_ok=True)
    model_path,tokenizer_path=model_paths()
    processor=VibeVoiceASRProcessor.from_pretrained(model_path,language_model_pretrained_name=tokenizer_path)
    model=VibeVoiceASRForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    # Only the frozen audio path is needed during feature extraction.
    model.model.language_model=None;model.lm_head=None
    model.to('cuda');torch.cuda.reset_peak_memory_stats()
    rows=[json.loads(l) for l in (BASE/'dataset/manifest.jsonl').read_text().splitlines()]
    (out/'.building').touch()
    frame_count=sum(math.ceil(sf.info(BASE/'dataset'/r['audio_path']).frames/processor.speech_tok_compress_ratio) for r in rows)
    feature_shape=(frame_count,model.config.decoder_config.hidden_size)
    feature_map=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float16,shape=feature_shape)
    token_parts=[];mask_parts=[];label_parts=[];index=[];toff=foff=0;t0=time.monotonic()
    for r in rows:
        audio,sr=sf.read(BASE/'dataset'/r['audio_path'],dtype='float32');assert sr==24000
        enc=processor._process_single_audio(audio,sampling_rate=sr,add_generation_prompt=False,use_streaming=True)
        target=json.dumps([{'Start':round(s['start'],2),'End':round(s['end'],2),'Speaker':s['speaker'],'Content':s['text']} for s in r['segments']],ensure_ascii=False,separators=(',',':'))
        target_ids=processor.tokenizer.apply_chat_template([{'role':'assistant','content':target}],tokenize=True,add_generation_prompt=False)
        ids=np.asarray(enc['input_ids']+target_ids,dtype=np.int32);mask=np.asarray(enc['acoustic_input_mask']+[0]*len(target_ids),dtype=np.bool_);labels=np.asarray([-100]*len(enc['input_ids'])+target_ids,dtype=np.int32)
        with torch.inference_mode():features=model.encode_speech(torch.from_numpy(enc['speech']).unsqueeze(0).to('cuda')).squeeze(0).float().cpu().numpy().astype(np.float16)
        assert features.shape[0]==mask.sum(),(r['id'],features.shape,mask.sum())
        assert np.isfinite(features).all()
        token_parts.append(ids);mask_parts.append(mask);label_parts.append(labels);feature_map[foff:foff+len(features)]=features
        index.append({**r,'token_offset':toff,'token_length':len(ids),'feature_offset':foff,'feature_length':len(features),'prompt_length':len(enc['input_ids']),'target_tokens':len(target_ids)})
        toff+=len(ids);foff+=len(features)
        print(json.dumps({'id':r['id'],'seconds':r['audio_duration'],'tokens':len(ids),'elapsed':round(time.monotonic()-t0,2)}),flush=True)
    for name,parts in [('tokens',token_parts),('mask',mask_parts),('labels',label_parts)]:np.save(out/(name+'.npy'),np.concatenate(parts))
    (out/'index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in index))
    stats={'elapsed_seconds':time.monotonic()-t0,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'rows':len(rows),'audio_seconds_all_variants':sum(r['audio_duration'] for r in rows),'feature_shape':list(feature_shape),'seed':20261022,'manifest_sha256':hashlib.sha256((BASE/'dataset/manifest.jsonl').read_bytes()).hexdigest(),'model_revision':MODEL_REVISION,'cache_dtype':'float16; frozen BF16 encoder output cast losslessly in ordinary activation range','scope':'Frozen encoders AND connectors only; invalid if either is trained. One fixed stochastic encoder draw per clip.'}
    assert foff==frame_count
    feature_map.flush()
    (out/'metadata.json').write_text(json.dumps(stats,indent=2));(out/'.building').unlink();print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
