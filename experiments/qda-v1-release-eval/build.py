"""Fixed broad release panel. Select sources before inspecting candidate outputs."""
import collections, hashlib, json, pathlib, random, sys
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=HERE.parent/'qda-full-continuous-v8'
sys.path.insert(0,str(BASE))
from dataset import safe_cuts,window_segments,merge_target_turns
from contracts import sha,atomic_json


def main():
    source_file=ROOT/'datasets/releases/qda-full-continuous-v8/sources.json'
    sources=json.loads(source_file.read_text());old=json.loads((BASE/'plan.json').read_text())
    train=[r for pair in old['probe_pairs'] for r in pair]
    forbidden={r['audio_sha256'] for r in train};train_shows={r['show'] for r in train}
    long=json.loads((BASE/'long-evaluation-ready.json').read_text())['rows']
    rows=[{**r,'cohort':'existing-development','cache_root':str(pathlib.Path('/data/work/asr-experiments/qda-full-continuous-v8/cache')),
           'feature_contract_sha256':old['cache_contract_sha256']} for r in old['validation']]
    excluded={r['record_id'] for r in rows+long}
    rng=random.Random(20261025)
    for split,count in [('development',8),('confirmation',4)]:
        for lang in ['cs','sk']:
            pools=collections.defaultdict(list)
            for source in sources:
                if source['split']==split and source['language']==lang and source['record_id'] not in excluded:
                    pools[source['show']].append(source)
            for pool in pools.values():pool.sort(key=lambda x:x['record_id']);rng.shuffle(pool)
            shows=sorted(pools);rng.shuffle(shows);chosen=[]
            while len(chosen)<count:
                for show in shows:
                    if pools[show] and len(chosen)<count:chosen.append(pools[show].pop())
                if not any(pools.values()) and len(chosen)<count:raise ValueError('Insufficient held-out sources')
            for source in chosen:
                assert source['audio_sha256'] not in forbidden and source['show'] not in train_shows
                excluded.add(source['record_id'])
                label_path=ROOT/'datasets/releases/qda-full-continuous-v8'/source['label_path']
                if not label_path.exists():label_path=ROOT/'datasets/releases/qda-recovered-v5'/source['label_path']
                assert sha(label_path)==source['label_sha256']
                label=json.loads(label_path.read_text());duration=source['audio_duration']
                cuts=safe_cuts(label['segments'],duration)
                fraction=[.2,.5,.75][int(source['record_id'][:8],16)%3]
                ideal=max(0.,duration-600)*fraction
                start=min(cuts,key=lambda x:abs(x-ideal));end=min((x for x in cuts if x>start),key=lambda x:abs(x-min(duration,start+600)))
                target=merge_target_turns(window_segments(label['segments'],start,end,duration))
                key=[source['record_id'],start,end,source['label_sha256'],'release-eval-v1']
                wid=hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
                rows.append({**source,'window_id':wid,'cohort':'new-'+split,'split':'release_evaluation',
                             'original_heldout_split':split,'source_start':start,'source_end':end,'duration':end-start,
                             'target':target,'customized_context':'','cache_root':'/data/work/asr-experiments/qda-v1-release-eval/cache',
                             'reference_quality':'Existing Soniox reference; recording and show excluded from v8 training'})
    assert len(rows)==32 and len({r['record_id'] for r in rows})==32
    assert collections.Counter(r['language'] for r in rows)=={'cs':16,'sk':16}
    for row in rows+long:
        assert row['audio_sha256'] not in forbidden and row['show'] not in train_shows
    rows.sort(key=lambda r:(r['cohort']=='new-confirmation',r['language'],r['record_id']))
    # Interleave languages so both workers and progressive summaries cover both.
    rows=[r for pair in zip([r for r in rows if r['language']=='cs'],[r for r in rows if r['language']=='sk']) for r in pair]
    long=[{**r,'cohort':'long-session','cache_root':'/data/work/asr-experiments/qda-full-continuous-v8/cache',
           'feature_contract_sha256':old['cache_contract_sha256']} for r in long]
    models={}
    for name,step in [('300h',1729),('450h',2564),('final',3110)]:
        adapter=BASE/'runs/r64-full'/f'checkpoint-{step:06d}'
        models[name]={'step':step,'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
                      'adapter_config_sha256':sha(adapter/'adapter_config.json'),
                      'adapter':{'300h':'/data/work/asr-experiments/qda-end-to-end-v9/parent','450h':'/data/work/asr-experiments/qda-v1-release-eval/adapter-450h','final':'/data/work/asr-experiments/qda-full-continuous-v8/runs/r64-full/checkpoint-003110'}[name]}
    models['vanilla']={'adapter':None,'adapter_sha256':'vanilla'}
    result={'version':1,'seed':20261025,'rows':rows,'long_rows':long,'models':models,
            'v8_plan_sha256':sha(BASE/'plan.json'),'sources_sha256':sha(source_file),
            'main_minutes':sum(r['duration'] for r in rows)/60,'long_minutes':sum(r['duration'] for r in long)/60,
            'unique_recordings':34,'training_recording_overlap':0,'training_show_overlap':0,
            'selection':'8 existing + 16 new development + 8 formerly unused confirmation recordings; 16 per language. Fixed-seed show-balanced selection before model outputs; no speaker/noise/confidence filter.',
            'confirmation_policy':'Report this cohort separately. It is consumed for release selection and is no longer an untouched test for later cloud training.',
            'decoding':'Pinned BF16 base + merged candidate adapter, same as existing evaluations; identical frozen speech features and greedy decoding for all candidates.',
            'reporting':'All 32 clips remain in the completion denominator. Report number-normalized and ordinary WER, DER, per-language results, recording- and show-cluster paired uncertainty, raw errors, long-session failures and existing synthetic markers. Incomplete outputs never disappear from the report.',
            'training_started':False}
    atomic_json(HERE/'selection.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','long_rows','models')},indent=2))


if __name__=='__main__':main()
