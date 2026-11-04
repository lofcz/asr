"""Rescore saved inference; no model/API calls. Immutable run outputs remain intact."""
import pathlib,json,time,os,hashlib
from number_wer import score,VERSION
BASE=pathlib.Path('/data/work/asr-evaluation/periodic')
INITIAL=pathlib.Path('/data/work/asr-evaluation/after-100h')
def atomic(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 with temp.open('rb') as f:os.fsync(f.fileno())
 os.replace(temp,path)
def entry(root,variant):
 samples=[]
 for lang in ('cs','sk'):
  folder=root/variant/lang;r=json.loads((folder/'result.json').read_text())
  if r.get('termination')!='eos' or not r.get('valid_timestamps') or 'der_250ms' not in r.get('metrics',{}):raise ValueError(f'Incomplete evaluation: {folder}')
  ref=json.loads((folder/'reference.json').read_text());hyp=json.loads((folder/'hypothesis.json').read_text())
  import re
  hyp=[x for x in hyp if not re.fullmatch(r'\s*\[[^\[\]]+\]\s*',x['Content'])]
  m=dict(r['metrics']);m.update(score(''.join(x['Content'] for x in ref),' '.join(x['Content'] for x in hyp),lang))
  samples.append({'language':lang,'metrics':m,**{k:r[k] for k in ('audio_sha256','reference_sha256','feature_sha256','dtype','decoding','customized_context')}})
 denom=sum(s['metrics']['number_normalized_reference_words'] for s in samples)
 scored=sum(float(s['metrics']['der_details']['scored_speaker_time']) for s in samples)
 error=sum(sum(float(s['metrics']['der_details'][k]) for k in ('missed_speaker_time','falarm_speaker_time','speaker_error_time')) for s in samples)
 return {'id':'vanilla' if variant=='vanilla' else root.name,'variant':variant,'checkpoint_step':r['checkpoint_step'] or 0,'training_audio_hours':r['training_audio_hours'],'samples':samples,'aggregate':{'number_normalized_wer':sum(s['metrics']['number_normalized_errors'] for s in samples)/denom,'wer':sum(s['metrics']['substitutions']+s['metrics']['deletions']+s['metrics']['insertions'] for s in samples)/sum(s['metrics']['reference_words'] for s in samples),'der_250ms':error/scored},'source':str(root)}
def build():
 baseline=entry(INITIAL,'vanilla');runs=[entry(INITIAL,'finetuned')];failures=[]
 for path in sorted(BASE.glob('runs/*')):
  if (path/'completed.json').exists():runs.append(entry(path,'finetuned'))
  elif (path/'failed.json').exists():failures.append(json.loads((path/'failed.json').read_text()))
 for run in runs:
  for a,b in zip(baseline['samples'],run['samples']):
   for k in ('language','audio_sha256','reference_sha256','feature_sha256','dtype','decoding','customized_context'):assert a[k]==b[k],f'Benchmark mismatch: {k}'
 runs.sort(key=lambda r:r['training_audio_hours'])
 result={'normalization_version':VERSION,'baseline':baseline,'runs':runs,'failures':failures,'policy':json.loads((BASE/'policy.json').read_text()) if (BASE/'policy.json').exists() else {},'note':'Fixed user-approved CS/SK dialogue fixtures, 11.48 minutes total; anecdotal, outside training. Language-specific cardinal normalization preserves diacritics. DER: 250ms collar, scored-time weighting. Vanilla computed once.'}
 atomic(BASE/'history.json',result);return result
if __name__=='__main__':
 r=build();print(json.dumps({x['id']:x['aggregate'] for x in [r['baseline']]+r['runs']}))
