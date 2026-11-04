"""Accept valid speakerless acoustic events; preserve original inference and all text."""
import pathlib,sys,json,re,importlib.util,shutil
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=['/data/work/asr-experiments/qda-continuation-v2','/data/work/asr-pilot/long-context-pilot','/data/work/asr-pilot/VibeVoice','/data/work/asr-evaluation/periodic/code']
from contracts import sha,atomic_json
spec=importlib.util.spec_from_file_location('established_scoring','/data/work/asr-evaluation/periodic/code/evaluate.py');scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)

def main():
 panel=json.loads((HERE/'panel.json').read_text())
 for variant in ('vanilla','finetuned'):
  for row in panel['samples']:
   folder=HERE/variant/row['sample_id'];path=folder/'result.json'
   if not path.exists():continue
   result=json.loads(path.read_text());raw=folder/'result.inference.json'
   if not raw.exists():shutil.copy2(path,raw)
   try:
    assert result['termination']=='eos','Generation incomplete'
    hyp=json.loads((folder/'output.txt').read_text());assert isinstance(hyp,list) and hyp
    events=[];speech=[]
    for s in hyp:
     assert isinstance(s,dict) and {'Start','End','Content'}<=s.keys() and isinstance(s['Content'],str),'Invalid output fields'
     assert 0<=float(s['Start'])<=float(s['End'])<=row['audio_duration']+1,'Invalid timestamp'
     if re.fullmatch(r'\s*\[[^\[\]]+\]\s*',s['Content']):events.append(s)
     else:assert 'Speaker' in s,'Speech segment missing speaker';speech.append(s)
    reference=HERE/'inputs'/row['sample_id']/'reference.json';assert sha(reference)==result['reference_sha256']
    ref=json.loads(reference.read_text());metrics=scoring.metrics(ref,hyp,folder,row['audio_duration'],row['language']);assert 'der_250ms' in metrics
    result['metrics']=json.loads(json.dumps(metrics,default=str));result['complete']=True;result.pop('error',None)
    result['acoustic_event_count']=len(events);result['speech_segment_count']=len(speech);result['scorer_sha256']=sha(HERE/'rescore.py')
    atomic_json(folder/'hypothesis.json',hyp)
   except Exception as e:result['complete']=False;result['error']=repr(e)
   atomic_json(path,result)
if __name__=='__main__':main()
