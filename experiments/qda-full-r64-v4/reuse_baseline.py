"""Reuse vanilla inference only after exact tensor/reference/feature equality."""
import json, pathlib, shutil
import numpy as np
import torch
from contracts import sha, atomic_json
from gates import demand
from runtime_state import evaluation_hashes
from scoring import metrics, validate_hypothesis
HERE=pathlib.Path(__file__).resolve().parent
OLD=pathlib.Path('/data/work/asr-experiments/qda-restart-v3')


def main():
    plan=json.loads((HERE/'plan.json').read_text());old_plan=json.loads((OLD/'plan.json').read_text())
    old_rows={r['window_id']:r for r in old_plan['validation']}
    demand({r['window_id'] for r in plan['validation']}==set(old_rows),'Held-out panel changed')
    demand(plan['model_revision']==old_plan['model_revision'],'Vanilla model changed')
    for row in plan['validation']:
        wid=row['window_id'];prior=OLD/'evaluations/vanilla'/wid;dest=HERE/'evaluations/vanilla'/wid
        previous=json.loads((prior/'result.json').read_text())
        demand(previous['complete'] and previous['adapter_sha256']=='vanilla','Invalid source vanilla result')
        demand(sha(OLD/'cache'/(wid+'.features.npy'))==previous['features_sha256'],'Old feature identity changed')
        old_inputs=torch.load(OLD/'cache'/(wid+'.input.pt'),weights_only=True)
        new_inputs=torch.load(HERE/'cache'/(wid+'.input.pt'),weights_only=True)
        demand(old_inputs.keys()==new_inputs.keys() and all(torch.equal(old_inputs[k],new_inputs[k]) for k in old_inputs),'Token/input tensors differ')
        with np.load(HERE/'cache'/(wid+'.features.npz'),allow_pickle=False) as packed:
            demand(np.array_equal(packed['features'],np.load(OLD/'cache'/(wid+'.features.npy'),allow_pickle=False)),'BF16 feature bits differ')
        demand(row['target']==json.loads((prior/'reference.json').read_text()),'Reference changed')
        demand(all(row[k]==old_rows[wid][k] for k in ('audio_sha256','source_start','source_end')),'Audio crop changed')
        shutil.copytree(prior,dest,dirs_exist_ok=True)
        raw=(dest/'output.txt').read_text();hyp=json.loads(raw);validate_hypothesis(hyp,row['duration'])
        current=metrics(row['target'],hyp,dest,row['duration'],row['language'])
        result={**previous,'input_sha256':row['input_sha256'],'features_sha256':sha(HERE/'cache'/(wid+'.features.npz')),
            'evaluation_sha256':evaluation_hashes(),'metrics':json.loads(json.dumps(current,default=str)),
            'reuse':{'source_result':str(prior/'result.json'),'source_result_sha256':sha(prior/'result.json'),
                    'raw_output_sha256':sha(prior/'output.txt'),'verification':'Exact input tensors, BF16 feature bits, source audio/crop, model and references; CPU rescore only'}}
        demand(sha(dest/'output.txt')==result['reuse']['raw_output_sha256'],'Raw inference changed')
        atomic_json(dest/'result.json',result)
    shutil.copytree(OLD/'markers/vanilla',HERE/'markers/vanilla',dirs_exist_ok=True)


if __name__=='__main__':main()
