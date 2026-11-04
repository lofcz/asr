#!/usr/bin/env python3
"""CPU streaming Silero ONNX audit; never automatically produces event training labels."""
import argparse, concurrent.futures, fcntl, gzip, io, json, logging, pathlib, subprocess, time
import numpy as np
import onnxruntime as ort
from soniox_durable import Store, encode, digest, file_hash
REV='066409f3d9db45a30d21dfcbf9147826ffdd7052'

def intervals(probs,duration):
    # Silero-style hysteresis, 250ms minimum speech, 100ms minimum silence, 30ms padding.
    found=[];start=None;pending=None
    for i,p in enumerate(probs):
        now=i*.032
        if p>=.5:
            pending=None
            if start is None:start=now
        elif p<.35 and start is not None:
            if pending is None:pending=now
            if now-pending>=.1:
                if pending-start>=.25:found.append([max(0,start-.03),min(duration,pending+.03)])
                start=None;pending=None
    if start is not None and duration-start>=.25:found.append([max(0,start-.03),duration])
    return union(found)
def union(rows):
    out=[]
    for a,b in sorted(rows):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out
def difference(left,right,minimum=.5):
    out=[];j=0
    for a,b in left:
        while j<len(right) and right[j][1]<=a:j+=1
        pos=a;k=j
        while k<len(right) and right[k][0]<b:
            c,d=right[k]
            if c-pos>=minimum:out.append([pos,min(c,b)])
            pos=max(pos,d)
            if pos>=b:break
            k+=1
        if b-pos>=minimum:out.append([pos,b])
    return out

def audit(store,r,model,model_hash):
    rel=pathlib.Path('records')/r['record_id'];target=store.backup/rel/'vad.json'
    if not target.exists():
        store.require();source=store.root/r['path']
        if file_hash(source)!=r['sha256']:raise ValueError('Source hash mismatch')
        opts=ort.SessionOptions();opts.inter_op_num_threads=1;opts.intra_op_num_threads=1
        sess=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
        state=np.zeros((2,1,128),dtype=np.float32);context=np.zeros((1,64),dtype=np.float32);probs=[];samples=0;started=time.monotonic()
        # Decode complete recording without retaining uncompressed audio in memory/on disk.
        with tempfile_stderr() as errors:
            proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(source),'-vn','-ac','1','-ar','16000','-f','f32le','-'],stdout=subprocess.PIPE,stderr=errors)
            try:
                while True:
                    raw=proc.stdout.read(512*4)
                    if not raw:break
                    chunk=np.frombuffer(raw,dtype='<f4').copy();samples+=len(chunk)
                    if len(chunk)<512:chunk=np.pad(chunk,(0,512-len(chunk)))
                    frame=np.concatenate((context,chunk.reshape(1,512)),axis=1)
                    pred,state=sess.run(None,{'input':frame,'state':state,'sr':np.array(16000,dtype=np.int64)})
                    probs.append(float(pred[0,0]));context=frame[:,-64:]
                code=proc.wait()
                if code:raise RuntimeError(f'Full audio decode failed: ffmpeg {code}')
                errors.seek(0)
                if errors.read():raise RuntimeError('Full decode reported audio errors; needs review')
            finally:
                if proc.poll() is None:proc.kill();proc.wait()
                proc.stdout.close()
        duration=samples/16000
        if abs(duration-r['seconds'])>max(2,r['seconds']*.002):raise ValueError('Decoded duration does not match manifest')
        speech=intervals(probs,duration)
        buffer=io.BytesIO();np.save(buffer,np.array(probs,dtype=np.float32),allow_pickle=False);compressed=gzip.compress(buffer.getvalue(),mtime=0)
        store.put(rel/'vad.probabilities.npy.gz',compressed)
        result={'schema_version':1,'record_id':r['record_id'],'audio_sha256':r['sha256'],'decoded_seconds':duration,'model':{'repository':'snakers4/silero-vad','revision':REV,'onnx_sha256':model_hash,'onnxruntime':ort.__version__},'config':{'sample_rate':16000,'frame_samples':512,'threshold':.5,'negative_threshold':.35,'min_speech_seconds':.25,'min_silence_seconds':.1,'padding_seconds':.03},'speech_intervals_seconds':speech,'speech_fraction':sum(b-a for a,b in speech)/duration,'probabilities_sha256':digest(compressed),'full_decode_verified':True,'elapsed_seconds':time.monotonic()-started,'policy':'Speech detector evidence, not verified speech/event annotations. No VAD-negative interval is automatically silence.'}
        store.obj(rel/'vad.json',result)
        logging.info('VAD %s %.1f min speech=%.1f%% in %.1fs',r['id'],duration/60,result['speech_fraction']*100,result['elapsed_seconds'])
    result=json.loads(target.read_bytes());raw=store.backup/rel/'soniox.raw.json'
    cross=store.backup/rel/'annotation-audit.json'
    if raw.exists() and not cross.exists():
        state=store.state(r)
        if state['phase']!='done':return
        receipt=json.loads((store.backup/rel/'receipt.json').read_bytes())
        rawdata=store.verified(rel/'soniox.raw.json',receipt['artifacts_sha256']['soniox.raw.json']);tokens=json.loads(rawdata)['tokens']
        duration=result['decoded_seconds'];speech=result['speech_intervals_seconds']
        words=union([[max(0,t['start_ms']/1000-.1),min(duration,t['end_ms']/1000+.1)] for t in tokens if isinstance(t.get('start_ms'),(int,float)) and t['start_ms']>=0 and isinstance(t.get('end_ms'),(int,float)) and not t.get('is_audio_event') and t.get('text','').strip() not in ('','<end>','<fin>')])
        combined=union(speech+words)
        audit={'record_id':r['record_id'],'audio_sha256':r['sha256'],'soniox_raw_sha256':digest(rawdata),'vad_sha256':file_hash(target),'speech_without_transcript_seconds':difference(speech,words),'transcript_without_vad_seconds':difference(words,speech),'non_speech_review_candidates_seconds':difference([[0,duration]],combined),'raw_event_token_indices':[i for i,t in enumerate(tokens) if t.get('is_audio_event')],'status':'needs_acoustic_event_classification_and_review','training_ready':False,'policy':'Disagreement is a review signal. Preserve words and overlaps; do not rewrite, suppress, or assign silence/music/noise from VAD alone.'}
        store.obj(rel/'annotation-audit.json',audit)

def tempfile_stderr():
    import tempfile
    return tempfile.TemporaryFile()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--backup',required=True);p.add_argument('--out');p.add_argument('--workers',type=int,default=2);p.add_argument('--once',action='store_true');args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    root=pathlib.Path(args.root);store=Store(root,args.out or root/'datasets/soniox-500h-v1',args.backup,(root/'.asr-volume-id').read_text().strip())
    lock=(store.backup/'vad-worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not store.ready():time.sleep(10)
    model=store.out/'references/silero_vad.onnx';model_hash=file_hash(model)
    rows=json.loads((store.backup/'selection.json').read_bytes());failed={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            done=0;futures={}
            if not store.ready():time.sleep(10);continue
            for r in rows:
                rel=pathlib.Path('records')/r['record_id']
                hasvad=(store.backup/rel/'vad.json').exists();hascross=(store.backup/rel/'annotation-audit.json').exists()
                if hasvad:done+=1
                if (not hasvad or (store.state(r)['phase']=='done' and not hascross)) and failed.get(r['record_id'],0)<time.time():futures[pool.submit(audit,store,r,model,model_hash)]=r
            completed_this_pass=0
            for future in concurrent.futures.as_completed(futures):
                r=futures[future]
                try:
                    future.result();completed_this_pass+=1
                    if completed_this_pass%10==0:
                        store.obj('vad-progress.json',{'updated_at':time.time(),'full_decode_vad_records':sum((store.backup/'records'/x['record_id']/'vad.json').exists() for x in rows),'cross_audited_records':sum((store.backup/'records'/x['record_id']/'annotation-audit.json').exists() for x in rows),'total_records':len(rows),'training_ready':False})
                except Exception as e:
                    failed[r['record_id']]=time.time()+1800
                    logging.warning('%s %s: %s',r['id'],type(e).__name__,e)
                    try:store.obj(pathlib.Path('records')/r['record_id']/'vad-error.json',{'error':str(e),'at':time.time(),'training_ready':False})
                    except OSError:pass
            summary={'updated_at':time.time(),'full_decode_vad_records':sum((store.backup/'records'/r['record_id']/'vad.json').exists() for r in rows),'cross_audited_records':sum((store.backup/'records'/r['record_id']/'annotation-audit.json').exists() for r in rows),'total_records':len(rows),'training_ready':False}
            store.obj('vad-progress.json',summary)
            if args.once or summary['cross_audited_records']==len(rows):break
            time.sleep(30)
if __name__=='__main__':main()
