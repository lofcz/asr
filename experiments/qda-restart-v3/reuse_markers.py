"""Verify and rescore the existing vanilla CS/SK markers; no new GPU inference."""
import json
import pathlib
import shutil
import sys
sys.path[:0]=['/data/work/asr-evaluation/periodic/code','/data/work/asr-pilot/VibeVoice','/data/work/asr-pilot/long-context-pilot']
from contracts import atomic_json,sha
from gates import demand,validate_reference
from model_paths import MODEL_REVISION

HERE=pathlib.Path(__file__).resolve().parent
GOLD=pathlib.Path('/data/work/asr-evaluation/production-reference-audit-v2')
EVAL=pathlib.Path('/data/work/asr-evaluation/periodic/code/evaluate.py')


def contract():
    return {'gold_manifest_sha256':sha(GOLD/'gold-manifest.json'),'evaluator_sha256':sha(EVAL),
            'number_normalizer_sha256':sha(EVAL.parent/'number_wer.py'),'model_revision':MODEL_REVISION}


def main():
    from evaluate import metrics
    root=HERE/'markers/vanilla';manifest=json.loads((GOLD/'gold-manifest.json').read_text());rows={};provenance={}
    for sample in manifest['samples']:
        language=sample['language'];source=pathlib.Path('/data/work/asr-evaluation/after-100h/vanilla')/language
        r=json.loads((source/'result.json').read_text())
        demand(r['model_revision']==MODEL_REVISION and r['dtype']=='bfloat16' and r['decoding']=='greedy' and r['customized_context']==[],'Marker inference contract differs')
        demand(r['termination']=='eos' and r.get('valid_json') is True,'Invalid historical vanilla marker')
        demand(r['audio_sha256']==sample['audio_sha256']==sha(GOLD/sample['audio_path']),'Marker audio changed')
        demand(r['reference_sha256']==sample['reference_sha256']==sha(GOLD/sample['reference_path']),'Marker reference changed')
        hypothesis=json.loads((source/'output.txt').read_text());demand(hypothesis==json.loads((source/'hypothesis.json').read_text()),'Marker output was repaired or replaced')
        validate_reference(hypothesis,sample['duration_seconds'])
        dest=root/'vanilla'/language;shutil.copytree(source,dest,dirs_exist_ok=True)
        reference=json.loads((GOLD/sample['reference_path']).read_text());r['metrics']=json.loads(json.dumps(metrics(reference,hypothesis,dest,sample['duration_seconds'],language),default=str))
        r['reuse']={'original_result_sha256':sha(source/'result.json'),'source':str(source),'inference_repeated':False}
        atomic_json(dest/'result.json',r);rows[language]=r['metrics'];provenance[language]={k:r[k] for k in ('audio_sha256','reference_sha256','feature_sha256')}
    demand(set(rows)=={'cs','sk'},'Both fixed markers are required')
    atomic_json(root/'completed-v3.json',{'adapter_sha256':'vanilla','languages':rows,'provenance':provenance,'contract':contract(),
                'role':'Required fixed CS/SK markers; existing raw vanilla outputs reused and rescored; no new inference'})
    print(json.dumps({l:{k:m[k] for k in ('number_normalized_wer','der_250ms')} for l,m in rows.items()},indent=2))


if __name__=='__main__':main()
