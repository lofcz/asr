"""One nonoverlapping pass through all eligible sources, with exclusion accounting."""
import argparse, bisect, collections, hashlib, json, math, pathlib, shutil, sys
from contracts import atomic_json, sha, target_with_spans
from targets import window_segments
from select_windows import quality
from gates import demand

HERE=pathlib.Path(__file__).resolve().parent
DATA=pathlib.Path('/data/work/asr-datasets/qda-full-r64-v4')
OLD=pathlib.Path('/data/work/asr-datasets/qda-clean-v3')


def safe_cuts(segments,duration):
    cuts=[0.]; extent=0.
    for x,y in zip(segments,segments[1:]):
        extent=max(extent,x['end'])
        if y['start']>=extent: cuts.append((extent+y['start'])/2)
    cuts.append(duration)
    return sorted(set(cuts))


def validate_intervals(rows):
    records=collections.defaultdict(list)
    for r in rows:
        demand(r['source_end']>r['source_start'], 'Empty training interval')
        demand(abs(r['source_end']-r['source_start']-r['duration'])<1e-6,'Duration mismatch')
        records[r['audio_sha256']].append(r)
    for rid,windows in records.items():
        windows.sort(key=lambda r:r['source_start'])
        demand(all(a['source_end']<=b['source_start']+1e-7 for a,b in zip(windows,windows[1:])),
               'Repeated or overlapping audio: '+rid)
    demand(len({r['window_id'] for r in rows})==len(rows),'Duplicate window ID')


def local_path(path):
    p=pathlib.Path(path)
    try:return DATA.parent/p.relative_to('/data/work/asr-datasets')
    except ValueError:return p


def init_worker(data):
    global DATA, processor, tokenizer, policy
    import os
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['USE_TORCH']='0'
    from transformers import PreTrainedTokenizerFast
    DATA=pathlib.Path(data)
    from types import SimpleNamespace
    config=json.loads((HERE/'selector-processor.json').read_text())
    tokenizer=PreTrainedTokenizerFast.from_pretrained(HERE/'selector-tokenizer')
    processor=SimpleNamespace(tokenizer=tokenizer,target_sample_rate=config['sample_rate'],speech_tok_compress_ratio=config['compression_ratio'])
    policy=json.loads((HERE/'policy.json').read_text())


def select_source(source,ordinal):
    data=DATA;chosen=[];excluded=[];pending=[]
    counts=collections.Counter();fractions=policy['length_audio_fractions']
    rid=source['record_id'];ap=data/'acoustic'/(rid+'.json')
    if not ap.exists():
        error=data/'acoustic-errors'/(rid+'.json')
        if error.exists():
            demand(json.loads(error.read_text())['audio_sha256']==source['audio_sha256'],'Stale decode failure')
            excluded.append({'record_id':rid,'start':0.,'end':source['audio_duration'],'reasons':['audio_decode_failed']})
        else:pending.append(rid)
        return {'chosen':chosen,'excluded':excluded,'pending':pending}
    acoustic=json.loads(ap.read_text());label=json.loads((data/source['label_path']).read_text())
    vp=local_path(source['vad_path']);vad=json.loads(vp.read_text())
    demand(acoustic['audio_sha256']==source['audio_sha256'] and acoustic['status']=='passed_decode','Bad acoustic identity')
    demand(vad['audio_sha256']==source['audio_sha256'] and vad['full_decode_verified'],'Bad VAD identity')
    demand(abs(vad['decoded_seconds']-source['audio_duration'])<.1,'VAD duration mismatch')
    cuts=safe_cuts(label['segments'],source['audio_duration']);cursor=0.;extent=cuts[-1]
    hashes={'label_sha256':sha(data/source['label_path']),'vad_sha256':sha(vp),'acoustic_sha256':sha(ap)}
    demand(hashes['label_sha256']==source['label_sha256'],'Changed label')
    while cursor<extent-.01:
        def preference(nominal):
            c=counts.copy();c[{600:'short',1800:'medium',3600:'long'}[nominal]]+=nominal
            total=sum(c.values())
            return sum((c[k]/total-v)**2/v for k,v in fractions.items())
        options=sorted((600,1800,3600),key=preference)+[120]
        if cursor==0.:
            first=(600,1800,1800,3600)[ordinal%4]
            options=[first]+[n for n in options if n!=first]
        selected=None;why=set();tried=set()
        for nominal in options:
            stop=min(extent,cursor+nominal)
            end=cuts[bisect.bisect_right(cuts,stop)-1]
            if end<=cursor or end in tried:continue
            tried.add(end)
            # Small tails are retained only if the whole remaining crop passes.
            if end-cursor<min(30.,extent-cursor):continue
            q=quality(source,label,vad,acoustic,cursor,end,policy)
            if q['reasons']:why.update(q['reasons']);continue
            try:target=window_segments(label['segments'],cursor,end)
            except ValueError:why.add('unsafe_or_empty_crop');continue
            sample={'segments':[dict(start=s['Start'],end=s['End'],speaker=s['Speaker'],text=s['Content']) for s in target]}
            body,_=target_with_spans(sample)
            target_tokens=len(processor.tokenizer.apply_chat_template([{'role':'assistant','content':body}],tokenize=True,add_generation_prompt=False))
            # Conservative bound; prepare.py checks the exact decoded sample and
            # full prompt before any feature encoding or model update.
            bound=target_tokens+math.ceil((end-cursor)*processor.target_sample_rate/processor.speech_tok_compress_ratio)+128
            if bound>65536:why.add('sequence_overflow');continue
            duration=end-cursor;kind='short' if duration<=660 else 'medium' if duration<=1860 else 'long'
            key=f'{rid}|{cursor:.6f}|{end:.6f}|{source["label_sha256"]}'
            wid=hashlib.sha256(key.encode()).hexdigest()[:24]
            selected={**source,**hashes,'window_id':wid,'source_start':cursor,'source_end':end,
                'duration':duration,'duration_class':kind,'bucket':nominal,'quality':q,'target':target,'token_upper_bound':bound}
            break
        if selected:
            chosen.append(selected);counts[selected['duration_class']]+=selected['duration'];cursor=selected['source_end']
        else:
            end=cuts[bisect.bisect_right(cuts,cursor)]
            excluded.append({'record_id':rid,'start':cursor,'end':end,'reasons':sorted(why or {'insufficient_clean_context'})})
            cursor=end
    return {'chosen':chosen,'excluded':excluded,'pending':pending}


def worker(task):
    source,ordinal,signature=task
    folder=DATA/'selection-cache';folder.mkdir(exist_ok=True)
    dest=folder/(source['record_id']+'.json')
    if dest.exists():
        saved=json.loads(dest.read_text())
        if saved.get('signature')==signature:return saved['result']
    result=select_source(source,ordinal)
    atomic_json(dest,{'signature':signature,'record_id':source['record_id'],'result':result})
    return result


def select(data,workers=3):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing,time
    policy=json.loads((HERE/'policy.json').read_text())
    sources=json.loads((data/'sources.json').read_text());train=[r for r in sources if r['split']=='train']
    chosen=[];excluded=[];pending=[];counts=collections.Counter()
    signature=hashlib.sha256(json.dumps({'sources':sha(data/'sources.json'),'policy':sha(HERE/'policy.json'),
        'selector':sha(__file__),'quality':sha(HERE/'select_windows.py'),
        'processor':sha(HERE/'selector-processor.json'),
        'tokenizer':{p.name:sha(p) for p in sorted((HERE/'selector-tokenizer').iterdir()) if p.is_file()}},sort_keys=True).encode()).hexdigest()
    results={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(str(data),)) as pool:
        jobs={pool.submit(worker,(r,i,signature)):i for i,r in enumerate(train)}
        for future in as_completed(jobs):
            results[jobs[future]]=future.result()
            hours=sum(r['duration'] for x in results.values() for r in x['chosen'])/3600
            progress={'stage':'select-full-corpus','processed_recordings':len(results),'total_recordings':len(train),'selected_hours_so_far':hours,'updated_at':time.time()}
            atomic_json(HERE/'selection-progress.json',progress)
            print(json.dumps(progress),flush=True)
    for i in range(len(train)):
        result=results[i];chosen.extend(result['chosen']);excluded.extend(result['excluded']);pending.extend(result['pending'])
    for row in chosen:counts[row['duration_class']]+=row['duration']
    # Keep the SAME fixed held-out clips and their original labels/references.
    old_root=data.parent/'qda-clean-v3'
    old=json.loads((old_root/'screening-selection.json').read_text());dev=[]
    for original in old['development']:
        row=dict(original);name='development-labels/'+row['record_id']+'.json'
        (data/'development-labels').mkdir(exist_ok=True)
        shutil.copy2(old_root/row['label_path'],data/name);row['label_path']=name
        demand(sha(data/name)==row['label_sha256'],'Held-out labels changed')
        for folder in ('acoustic',):
            dest=data/folder/(row['record_id']+'.json')
            if not dest.exists():shutil.copy2(old_root/folder/dest.name,dest)
        review=old_root/'reviews'/(row['window_id']+'.json')
        if review.exists():
            (data/'reviews').mkdir(exist_ok=True);shutil.copy2(review,data/'reviews'/review.name)
        dev.append(row)
    validate_intervals(chosen)
    demand(not {r['show'] for r in chosen}&{r['show'] for r in dev},'Held-out leakage')
    allrows=chosen+dev;atomic_json(data/'candidates.json',allrows)
    atomic_json(data/'window-quarantine.json',excluded)
    atomic_json(data/'acoustic-selection.json',{'selector_sha256':sha(HERE/'select_windows.py'),
        'full_selector_sha256':sha(__file__),'sources_sha256':sha(data/'sources.json'),
        'candidates_sha256':sha(data/'candidates.json'),'acoustic_pending_records':pending})
    result={'train':chosen,'development':dev,'policy_sha256':sha(HERE/'policy.json'),
        'candidates_sha256':sha(data/'candidates.json'),'train_hours':sum(r['duration'] for r in chosen)/3600,
        'development_minutes':sum(r['duration'] for r in dev)/60,
        'language_hours':{l:sum(r['duration'] for r in chosen if r['language']==l)/3600 for l in ('cs','sk')},
        'length_hours':{k:v/3600 for k,v in counts.items()},'unique_recordings':len({r['record_id'] for r in chosen}),
        'repeat_windows':False,'full_corpus_pass':True,'pending_acoustic_records':len(pending),
        'excluded_interval_hours':sum(r['end']-r['start'] for r in excluded)/3600,
        'structurally_eligible_train_hours':sum(r['audio_duration'] for r in sources if r['split']=='train')/3600,
        'reference_quality':'Soniox pseudo-references; fixed CS/SK approved markers separately',
        'training_authorized':False,'status':'awaiting_integrity_admission'}
    atomic_json(data/'screening-selection.json',result)
    atomic_json(data/'selection-summary.json',{k:v for k,v in result.items() if k not in ('train','development')})
    print(json.dumps({k:v for k,v in result.items() if k not in ('train','development')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,default=DATA);p.add_argument('--workers',type=int,default=3);a=p.parse_args();select(a.data,a.workers)
