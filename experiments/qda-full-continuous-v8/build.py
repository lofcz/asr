"""One immutable full real+Cartesia pass, with no quality or vocabulary filters."""
import bisect
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from contracts import atomic_json, sha, target_with_spans
from dataset import window_segments, verify_partition, merge_target_turns

HERE=Path(__file__).resolve().parent
DATA=Path('/data/work/asr-datasets/qda-full-continuous-v8')
SYNTH=Path('/data/work/asr-synthetic-v7')
ROOT=Path('/data/work/asr-experiments')


def copied_label(source):
    digest=sha(source);dest=DATA/'labels'/(digest+'.json')
    if not dest.exists():os.link(source,dest)
    assert sha(dest)==digest
    return str(dest.relative_to(DATA)),digest


def units_for_synthetic(receipt):
    """Recover original utterance boundaries without losing merged monologue text."""
    sid=receipt['id']
    label=json.loads((SYNTH/receipt['label_path']).read_text())
    assert sha(SYNTH/receipt['label_path'])==receipt['label_sha256']
    assert sha(SYNTH/receipt['words_path'])==receipt['words_sha256']
    composition_path=SYNTH/'sessions'/sid/'composition.json'
    assert sha(composition_path)==receipt['source_composition_sha256']
    composition=json.loads(composition_path.read_text())
    words=json.loads((SYNTH/receipt['words_path']).read_text())
    voice_map={v:int(k) for k,v in receipt['known_speaker_voice_ids'].items()}
    units=[];cursor=0;endpoint_adjustments=[]
    # Identical short utterances can reuse a content hash, so never group the
    # entire word list by utterance_id. Consume each occurrence in timeline order.
    for item in composition:
        start=cursor
        while cursor<len(words) and words[cursor]['start']<item['end_samples']/24000:
            assert words[cursor]['utterance_id']==item['utterance_id']
            cursor+=1
        part=words[start:cursor]
        assert part and all(w['speaker']==voice_map[item['voice_id']] for w in part)
        text=' '.join(w['text'] for w in part)
        # Use the exact export text, including canonical initialisms. Word
        # mapping already proves lexical identity with the authored text.
        end=min(part[-1]['end'],receipt['audio_duration'])
        if end!=part[-1]['end']:
            assert part[-1]['end']-end<=.05
            endpoint_adjustments.append({'utterance':item['utterance_id'],'native_end':part[-1]['end'],'audio_end':end})
        units.append({'start':part[0]['start'],'end':end,
                      'speaker':voice_map[item['voice_id']],'text':text})
    assert cursor==len(words)
    lexical=lambda text:''.join(c for c in text if c.isalnum() or c=='_')
    assert lexical(' '.join(s['text'] for s in units))==lexical(' '.join(s['text'] for s in label['segments']))
    # Prefer exact authored punctuation, which may include standalone symbols
    # omitted from the word-timestamp sidecar.
    from synthetic_text import canonicalize
    for unit,item in zip(units,composition):unit['text']=canonicalize(item['text'])[0]
    assert ' '.join(s['text'] for s in units)==' '.join(s['text'] for s in label['segments'])
    return {**label,'segments':units,'native_endpoint_adjustments':endpoint_adjustments}


def split_source(source,units,proc):
    """Split only at full utterance boundaries; every source second appears once."""
    duration=source['audio_duration'];cuts=[0.]
    for left,right in zip(units,units[1:]):
        assert right['start']>=left['end']-.000001
        cuts.append(round((left['end']+right['start'])/2,6))
    cuts=sorted(set(cuts+[duration]));start=0.;rows=[]
    context=source.get('customized_context','')
    while start<duration-1e-7:
        first=bisect.bisect_right(cuts,start)
        last=bisect.bisect_right(cuts,min(duration,start+3600))-1
        assert last>=first,'An individual utterance exceeds one hour'
        def candidate(i):
            end=cuts[i];target=merge_target_turns(window_segments(units,start,end,duration))
            sample={'segments':[{'start':t['Start'],'end':t['End'],'speaker':t['Speaker'],'text':t['Content']} for t in target]}
            body,_=target_with_spans(sample)
            n=len(proc.tokenizer.apply_chat_template([{'role':'assistant','content':body}],tokenize=True,add_generation_prompt=False))
            budget=n+math.ceil((end-start)*7.5)+128+len(proc.tokenizer.encode(context,add_special_tokens=False))
            return target,budget
        lo=first;hi=last;chosen=None
        while lo<=hi:
            middle=(lo+hi)//2;target,budget=candidate(middle)
            if budget<=65536:chosen=(middle,target,budget);lo=middle+1
            else:hi=middle-1
        assert chosen is not None,'An individual utterance exceeds the model token budget'
        index,target,budget=chosen;end=cuts[index]
        identity=hashlib.sha256(json.dumps([source['audio_sha256'],source['label_sha256'],start,end],separators=(',',':')).encode()).hexdigest()[:24]
        rows.append({**source,'window_id':identity,'source_start':start,'source_end':end,
                     'duration':end-start,'target':target,'total_tokens':budget,
                     'duration_class':'long' if end-start>=1800 else 'medium' if end-start>=600 else 'short'})
        start=end
    return rows


def main():
    if (HERE/'prepared.json').exists():raise RuntimeError('Finalized package is immutable')
    DATA.mkdir(parents=True,exist_ok=True);(DATA/'labels').mkdir(exist_ok=True)
    rows=[];development=[];sources={};provenance=[]
    prior=[]
    for experiment,data_name in [('qda-full-r64-v4','qda-full-r64-v4'),('qda-r64-continuation-v6','qda-unseen-v6')]:
        root=ROOT/experiment;path=root/'plan.json';plan=json.loads(path.read_text());prior.append(plan)
        provenance.append({'path':str(path),'sha256':sha(path)})
        for item in [r for pair in plan['probe_pairs'] for r in pair]+([] if development else plan['validation']):
            row=dict(item);label=Path('/data/work/asr-datasets')/data_name/row['label_path']
            assert sha(label)==row['label_sha256']
            row['label_path'],_=copied_label(label)
            row.update(reuse_root=str(root),dataset_kind='real',customized_context='')
            blank=[i for i,t in enumerate(row['target']) if isinstance(t.get('Content'),str) and not t['Content'].strip()]
            if blank:
                row['target']=[t for i,t in enumerate(row['target']) if i not in blank]
                row['source_label_normalizations']=[{'operation':'omit_whitespace_only_target_entries','indices':blank}]
                row.pop('reuse_root')
            (rows if row['split']=='train' else development).append(row)
    real_sources=json.loads(Path('/data/work/asr-datasets/qda-unseen-v6/sources.json').read_text())
    partition=verify_partition(real_sources,[r for p in prior[0]['probe_pairs'] for r in p],[r for p in prior[1]['probe_pairs'] for r in p])
    for s in real_sources:sources[s['audio_sha256']]=s
    published=[json.loads(line) for line in (SYNTH/'training-manifest.jsonl').read_text().splitlines()]
    spec=json.loads((SYNTH/'plan.json').read_text());expected={s['id'] for s in spec['sessions']}
    assert {r['id'] for r in published}==expected,'Wait for all authored recordings to be committed and copied'
    sys.path[:0]=['/data/work/asr-pilot/long-context-pilot','/data/work/asr-pilot/VibeVoice','/data/work/asr-synthetic-v7/implementation']
    from model_paths import model_paths
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    path,tok=model_paths();proc=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
    for receipt in published:
        sid=receipt['id'];label=units_for_synthetic(receipt)
        label_path=DATA/'synthetic-units'/(sid+'.json');atomic_json(label_path,label)
        source={'record_id':sid,'audio_file':str(SYNTH/receipt['audio_path']),
            'audio_sha256':receipt['audio_sha256'],'audio_duration':receipt['audio_duration'],
            'label_path':str(label_path.relative_to(DATA)),'label_sha256':sha(label_path),
            'original_label_path':str(SYNTH/receipt['label_path']),'original_label_sha256':receipt['label_sha256'],
            'words_file':str(SYNTH/receipt['words_path']),'words_sha256':receipt['words_sha256'],
            'split':'train','language':receipt['language'][0],'languages':receipt['language'],
            'show':'Cartesia synthetic '+receipt['language'][0],'title':sid,'dataset_kind':'synthetic',
            'speaker_count':receipt['speaker_count'],'known_speaker_voice_ids':receipt['known_speaker_voice_ids'],
            'customized_context':label.get('customized_context',''),'cohort':receipt['cohort'],
            'reference_quality':'Authored transcript and Cartesia native timestamps; known voices'}
        assert source['audio_sha256'] not in sources,'Duplicate source audio'
        sources[source['audio_sha256']]=source
        rows.extend(split_source(source,label['segments'],proc))
    assert len({r['window_id'] for r in rows})==len(rows)
    selected={'train':rows,'development':development,'train_hours':sum(r['duration'] for r in rows)/3600,
              'full_corpus_pass':True,'repeat_windows':False,
              'real_train_hours':partition['total_unique_train_hours'],
              'synthetic_train_hours':sum(r['duration'] for r in rows if r['dataset_kind']=='synthetic')/3600}
    atomic_json(DATA/'screening-selection.json',selected)
    atomic_json(DATA/'sources.json',list(sources.values()))
    atomic_json(DATA/'provenance.json',{'source_plans':provenance,
        'selection_sha256':sha(DATA/'screening-selection.json'),'sources_sha256':sha(DATA/'sources.json'),
        'synthetic_manifest_sha256':sha(SYNTH/'training-manifest.jsonl'),
        'real_partition':partition,'synthetic_recordings':len(published),
        'synthetic_reference':'Authored text + native timestamps + known voices',
        'admission_filters':[],'existing_unresolved_real_sources':json.loads(Path('/data/work/asr-datasets/qda-recovered-v5/unresolved-sources.json').read_text())})
    from gates import check_data
    report=check_data(DATA)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
