"""Build only audio intervals not consumed by the completed parent run."""
import bisect, collections, concurrent.futures, hashlib, json, math, multiprocessing
import os, pathlib, shutil, time
from contracts import sha, atomic_json, target_with_spans
from dataset import complement, safe_cuts, window_segments, verify_partition

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent / 'qda-full-r64-v4'
RECOVERED = ROOT / 'datasets/releases/qda-recovered-v5'
DATA = ROOT / 'datasets/releases/qda-unseen-v6'


def initialize():
    global tokenizer
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['USE_TORCH'] = '0'
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(PARENT / 'selector-tokenizer')


def work(task):
    source, used = task
    label_path = RECOVERED / source['label_path']
    if sha(label_path) != source['label_sha256']:
        raise ValueError('Changed source labels')
    segments = json.loads(label_path.read_text())['segments']
    duration = source['audio_duration']
    cuts = safe_cuts(segments, duration)
    rows = []
    counts = collections.Counter()
    for start, stop in complement(duration, used):
        cursor = start
        while cursor < stop - 1e-7:
            def preference(nominal):
                d = counts.copy()
                d[{600: 'short', 1800: 'medium', 3600: 'long'}[nominal]] += min(nominal, stop-cursor)
                n = sum(d.values())
                return sum((d[k]/n-v)**2/v for k,v in {'short':.25,'medium':.5,'long':.25}.items())
            selected = None
            options = sorted((600,1800,3600), key=preference)
            if not rows:
                first = (600,1800,1800,3600)[int(source['record_id'][:8],16)%4]
                options = [first] + [n for n in options if n != first]
            tried = set()
            for nominal in options + [300,120,30]:
                bound = min(stop, cursor+nominal)
                end = stop if bound == stop else cuts[bisect.bisect_right(cuts, bound)-1]
                if end <= cursor or end in tried:
                    continue
                tried.add(end)
                target = window_segments(segments,cursor,end,duration)
                body,_ = target_with_spans({'segments':[dict(start=t['Start'],end=t['End'],speaker=t['Speaker'],text=t['Content']) for t in target]})
                nt = len(tokenizer.apply_chat_template([{'role':'assistant','content':body}],tokenize=True,add_generation_prompt=False))
                total = nt + math.ceil((end-cursor)*7.5) + 128
                if total > 65536:
                    continue
                key = f"{source['record_id']}|{cursor:.6f}|{end:.6f}|{source['label_sha256']}"
                kind = 'short' if end-cursor <= 660 else 'medium' if end-cursor <= 1860 else 'long'
                selected = {**source, 'window_id':hashlib.sha256(key.encode()).hexdigest()[:24],
                    'source_start':cursor,'source_end':end,'duration':end-cursor,'duration_class':kind,
                    'target':target,'target_tokens':nt,'total_tokens':total,
                    'speaker_count':len({s['Speaker'] for s in target}),
                    'quality':{'reasons':[],'heuristics_are_diagnostics':True}}
                break
            if selected is None:
                raise ValueError(f'No representable crop: {source["record_id"]} {cursor}')
            rows.append(selected)
            counts[selected['duration_class']] += selected['duration']
            cursor = selected['source_end']
        expected = ''.join(s['Content'] for s in window_segments(segments,start,stop,duration))
        actual = ''.join(t['Content'] for r in rows if r['source_start'] >= start and r['source_end'] <= stop for t in r['target'])
        if actual != expected:
            raise ValueError('Crop token stream changed')
    return rows


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    (DATA/'.building').write_text('Building exact remainder\n')
    sources = json.loads((RECOVERED/'sources.json').read_text())
    parent = json.loads((PARENT/'plan.json').read_text())
    complete = json.loads((PARENT/'runs/r64-full/completed.json').read_text())
    if complete['steps'] != len(parent['probe_pairs']):
        raise ValueError('Parent plan was not fully consumed')
    consumed = [r for pair in parent['probe_pairs'] for r in pair]
    used = collections.defaultdict(list)
    for r in consumed:
        used[r['record_id']].append((r['source_start'],r['source_end']))
    train = [s for s in sources if s['split']=='train']
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,initializer=initialize,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(work,(s,used[s['record_id']])) for s in train]
        for i,f in enumerate(jobs):
            rows.extend(f.result())
            if i%25 == 0:
                status={'stage':'building_exact_remainder','sources':i+1,'total_sources':len(train),
                    'windows':len(rows),'updated_at':time.time()}
                atomic_json(HERE/'preparation-status.json',status)
                print(json.dumps(status),flush=True)
    partition = verify_partition(sources,consumed,rows)
    # Copy only small metadata; all audio and paid transcripts stay in their original releases.
    shutil.copy2(RECOVERED/'sources.json',DATA/'sources.json')
    shutil.copy2(RECOVERED/'pairing-inventory.jsonl',DATA/'pairing-inventory.jsonl')
    shutil.copytree(RECOVERED/'labels',DATA/'labels',dirs_exist_ok=True)
    shutil.copytree(ROOT/'datasets/releases/qda-full-r64-v4/development-labels',DATA/'development-labels',dirs_exist_ok=True)
    dev = parent['validation']
    selection = {'train':rows,'development':dev,'train_hours':partition['remaining_hours'],
        'full_corpus_pass':True,'repeat_windows':False,'development_minutes':sum(r['duration'] for r in dev)/60,
        'length_hours':{k:sum(r['duration'] for r in rows if r['duration_class']==k)/3600 for k in ('short','medium','long')},
        'language_hours':{k:sum(r['duration'] for r in rows if r['language']==k)/3600 for k in ('cs','sk')},
        'unique_recordings':len({r['record_id'] for r in rows}), 'windows':len(rows),
        'reference_quality':'Original Soniox labels; no acoustic/speaker-count admission filters',**partition}
    atomic_json(DATA/'screening-selection.json',selection)
    atomic_json(DATA/'selection-summary.json',{k:v for k,v in selection.items() if k not in ('train','development')})
    receipt=json.loads((PARENT/'runs/r64-full/backup-receipts/checkpoint-001314.json').read_text())
    provenance={'parent_experiment':'qda-full-r64-v4','parent_plan_sha256':sha(PARENT/'plan.json'),
        'parent_completed_sha256':sha(PARENT/'runs/r64-full/completed.json'),
        'parent_checkpoint':'/data/work/asr-experiments/qda-full-r64-v4/runs/r64-full/checkpoint-001314',
        'parent_checkpoint_files':receipt['files'],'selection_sha256':sha(DATA/'screening-selection.json'),
        'sources_sha256':sha(DATA/'sources.json'),'inventory_sha256':sha(DATA/'pairing-inventory.jsonl'),
        'builder_sha256':sha(__file__),'partition':partition}
    atomic_json(DATA/'provenance.json',provenance)
    (DATA/'.building').unlink()
    atomic_json(HERE/'preparation-status.json',{'stage':'remainder_ready','updated_at':time.time(),**partition})
    print(json.dumps(selection | {'train':len(rows),'development':len(dev)},indent=2))


if __name__=='__main__':main()
