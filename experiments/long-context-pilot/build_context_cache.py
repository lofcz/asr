"""Retokenize context variants while sharing the frozen audio-feature cache."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import hashlib,json,pathlib
import numpy as np
import soundfile as sf
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from model_paths import model_paths
from target_format import context_text

BASE=pathlib.Path(__file__).resolve().parent


def main():
    old=BASE/'cache';out=BASE/'cache-context'
    out.mkdir(exist_ok=False);(out/'.building').touch()
    path,tokpath=model_paths()
    processor=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tokpath)
    contexts=json.loads((BASE/'sample/contexts.json').read_text())
    rows=[json.loads(line) for line in (old/'index.jsonl').read_text().splitlines()]
    labels=np.load(old/'labels.npy',mmap_mode='r')
    index=[];tokens=[];masks=[];targets=[];offset=0
    for row in rows:
        audio,sr=sf.read(BASE/'dataset'/row['audio_path'],dtype='float32');assert sr==24000
        c=contexts[row['source_id']]
        for mode,context in [('none',[]),('provided',c['customized_context']),
                             ('distractor',['Projekt Orion','Nováková'])]:
            encoded=processor._process_single_audio(audio,sampling_rate=sr,add_generation_prompt=False,
                                                    use_streaming=True,context_info=context_text({'customized_context':context}))
            start=row['token_offset']+row['prompt_length'];end=row['token_offset']+row['token_length']
            target=np.array(labels[start:end],copy=True)
            ids=np.concatenate([np.asarray(encoded['input_ids'],dtype=np.int32),target])
            mask=np.asarray(encoded['acoustic_input_mask']+[False]*len(target),dtype=np.bool_)
            newlabels=np.concatenate([np.full(len(encoded['input_ids']),-100,dtype=np.int32),target])
            assert mask.sum()==row['feature_length'] and len(ids)<=65536
            index.append({**row,'id':row['id']+'-context-'+mode,'example_group':row['id'],
                          'context_mode':mode,'customized_context':context,
                          'context_provenance':c['provenance'] if mode=='provided' else mode,
                          'token_offset':offset,'token_length':len(ids),'prompt_length':len(encoded['input_ids'])})
            tokens.append(ids);masks.append(mask);targets.append(newlabels);offset+=len(ids)
    for name,parts in [('tokens',tokens),('mask',masks),('labels',targets)]:np.save(out/f'{name}.npy',np.concatenate(parts))
    (out/'features.npy').symlink_to('../cache/features.npy')
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in index)
    (out/'index.jsonl').write_text(text)
    metadata=json.loads((old/'metadata.json').read_text())
    metadata.update(rows=len(index),source_feature_manifest_sha256=metadata['manifest_sha256'],
                    context_index_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    variant_policy='Choose one variant per example_group; variants are not additional independent audio.',
                    feature_storage='shared symlink; no audio re-encoding or duplicate feature storage')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));(out/'.building').unlink()
    print(json.dumps({'rows':len(index),'shared_features':True,'context_modes':['none','provided','distractor']}))


if __name__=='__main__':main()
