"""Rescore preserved vanilla outputs after explicit speech-abstention handling.

No decoding, label edits, inferred speakers, or output repairs are performed.
"""
import json
import pathlib
import sys
from contracts import sha,atomic_json
from gates import demand
from runtime_state import evaluation_hashes
from scoring import metrics,validate_hypothesis
sys.path.insert(0,'/data/work/asr-pilot/long-context-pilot')
from model_paths import MODEL_REVISION

HERE=pathlib.Path(__file__).resolve().parent
PREVIOUS={
    'evaluate_panel.py':'78256d29029122b069792d2946a8ee47e4768053c2162f7b10625158223d7bd5',
    'scoring.py':'573ab14ef72e66611afa7ef6ae26777f181ad147f8907ebd434c8242a34c5360',
}


def main():
    plan=json.loads((HERE/'plan.json').read_text());current=evaluation_hashes()
    for row in plan['validation']:
        folder=HERE/'evaluations/vanilla'/row['window_id'];path=folder/'result.json'
        old=json.loads(path.read_text());rawhash=sha(folder/'output.txt')
        if old.get('rescoring') and old.get('complete') and old['evaluation_sha256']==current:
            demand(old['rescoring']['raw_output_sha256']==rawhash,'Rescored output changed');continue
        demand(old['adapter_sha256']=='vanilla' and old['model_revision']==MODEL_REVISION,'Not the pinned vanilla model')
        demand(old['dtype']=='bfloat16' and old['decoding']=='greedy' and old['termination']=='eos','Different inference contract')
        for key,digest in current.items():
            demand(old['evaluation_sha256'][key]==PREVIOUS.get(pathlib.Path(key).name,digest),'Unrecognized inference/scoring revision: '+key)
        demand(sha(row['audio_file'])==row['audio_sha256'],'Source audio changed')
        demand(old['input_sha256']==row['input_sha256']==sha(HERE/'cache'/(row['window_id']+'.input.pt')),'Input changed')
        demand(old['features_sha256']==sha(HERE/'cache'/(row['window_id']+'.features.npy')),'Features changed')
        demand(old['reference_sha256']==sha(folder/'reference.json'),'Reference changed')
        ref=json.loads((folder/'reference.json').read_text());demand(ref==row['target'],'Plan reference changed')
        hyp=json.loads((folder/'output.txt').read_text());validate_hypothesis(hyp,row['duration'])
        if (folder/'hypothesis.json').exists():demand(json.loads((folder/'hypothesis.json').read_text())==hyp,'Hypothesis differs from raw output')
        priorhash=sha(path);backup=folder/'result.before-speech-tags.json'
        if backup.exists():demand(sha(backup)==priorhash,'Different archived result')
        else:atomic_json(backup,old)
        result=dict(old);result.pop('error',None)
        result['metrics']=json.loads(json.dumps(metrics(ref,hyp,folder,row['duration'],row['language']),default=str))
        result.update(complete=True,evaluation_sha256=current,
                      rescoring={'reason':'Explicit [Speech] abstentions contribute no recognized words or diarized speaker; all reference words/time remain scored.',
                                 'raw_output_sha256':rawhash,'prior_result_sha256':priorhash,'prior_evaluation_sha256':old['evaluation_sha256'],'gpu_inference_repeated':False})
        demand(sha(folder/'output.txt')==rawhash,'Raw output changed during rescoring')
        atomic_json(folder/'hypothesis.json',hyp);atomic_json(path,result)
        print(json.dumps({'window':row['window_id'],'wer':result['metrics']['number_normalized_wer'],'der':result['metrics']['der_250ms'],
                          'speech_abstentions':result['metrics']['speech_abstention_segments'],'inference_repeated':False}),flush=True)


if __name__=='__main__':main()
