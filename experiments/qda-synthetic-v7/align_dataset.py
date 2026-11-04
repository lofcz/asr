"""Optional offline FP32 alignment. Disabled for Cartesia native timestamps."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

ROOT=Path('/data/work/asr-synthetic-v7')
ALIGN=Path('/data/work/asr-word-alignment')
sys.path.insert(0,str(ALIGN))
from align_canary import load_model, align_segments
from align_existing import decode


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()


def save(path,obj):
    data=(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.',dir=path.parent)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(name,path)
    fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    return hashlib.sha256(data).hexdigest()


def run(directory,model):
    receipt=json.loads((directory/'receipt.json').read_text())
    audio_path=directory/'audio.flac';labels_path=directory/'labels.json';composition_path=directory/'composition.json'
    for path,field in [(audio_path,'audio_sha256'),(labels_path,'labels_sha256'),(composition_path,'composition_sha256')]:
        if sha(path)!=receipt[field]:raise ValueError('Input pairing hash mismatch: '+path.name)
    labels=json.loads(labels_path.read_text());composition=json.loads(composition_path.read_text())
    audio=decode(audio_path)
    segments=[{'Start':c['offset_samples']/c['sample_rate'],'End':c['end_samples']/c['sample_rate'],
               'Speaker':c['speaker'],'Content':c['text']} for c in composition]
    result=align_segments(segments,audio,model,'cuda:0',padding=0)
    output_words=[];merged=[]
    for item,c in zip(result['segments'],composition):
        valid=[w for w in item['words'] if w['start'] is not None]
        if not valid:continue
        seg={'Start':round(valid[0]['start'],6),'End':round(valid[-1]['end'],6),
             'Speaker':item['Speaker'],'Content':item['Content']}
        if merged and merged[-1]['Speaker']==seg['Speaker'] and seg['Start']-merged[-1]['End']<1.5:
            merged[-1]['Content']+=' '+seg['Content'];merged[-1]['End']=seg['End']
        else:merged.append(seg)
        output_words.extend({**w,'utterance_id':c['utterance_id']} for w in item['words'])
    aligned={**labels,'version':'cartesia-qda-v7-canary-fp32-v1','segments':merged,'words':output_words,
        'timestamp_source':'Canary auxiliary CTC FP32 forced alignment of exact supplied script; native timestamps retained in labels.json',
        'alignment_summary':result['summary'],'label_status':'synthetic_forced_alignment',
        'source_labels_sha256':receipt['labels_sha256'],'audio_sha256':receipt['audio_sha256']}
    label_hash=save(directory/'canary-fp32.labels.json',aligned)
    summary_hash=save(directory/'canary-fp32.summary.json',result['summary'])
    final={'status':'complete','session_id':receipt['session_id'],'duration_seconds':receipt['duration_seconds'],
        'source_labels_sha256':receipt['labels_sha256'],'audio_sha256':receipt['audio_sha256'],
        'labels_sha256':label_hash,'summary_sha256':summary_hash,'precision':'float32','quantization':'none',
        'review_words':result['summary']['review_words'],'completed_at':time.time()}
    save(directory/'canary-fp32.receipt.json',final)
    print(json.dumps({'event':'aligned',**final}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    policy=json.loads((ROOT/'timestamp-policy.json').read_text()) if (ROOT/'timestamp-policy.json').exists() else {}
    if policy.get('aot_alignment')=='disabled':
        print('Offline alignment disabled: use native Cartesia word timestamps.',flush=True)
        return
    with (ROOT/'.align.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model=None
        while True:
            paths=sorted((ROOT/'sessions').glob('*/receipt.json'))
            for path in paths:
                directory=path.parent
                if (directory/'canary-fp32.receipt.json').exists() or (directory/'canary-fp32.error.json').exists():continue
                try:
                    if model is None:model=load_model('cuda:0',precision='float32',quantization='none')
                    run(directory,model)
                except Exception as e:
                    save(directory/'canary-fp32.error.json',{'error':str(e),'at':time.time()})
                    print(json.dumps({'event':'alignment_error','session_id':directory.name,'error':str(e)}),flush=True)
            done=[json.loads(p.read_text()) for p in (ROOT/'sessions').glob('*/canary-fp32.receipt.json')]
            save(ROOT/'alignment-progress.json',{'updated_at':time.time(),'completed_sessions':len(done),
                'completed_hours':sum(x['duration_seconds'] for x in done)/3600,
                'errors':len(list((ROOT/'sessions').glob('*/canary-fp32.error.json'))),
                'precision':'float32','quantization':'none','model_loaded':model is not None})
            if args.once:return
            time.sleep(10)


if __name__=='__main__':main()
