"""Paired, denominator-weighted WER/DER; uncertainty resamples recordings, not words."""
import pathlib,sys,json,csv,datetime,html
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,'/data/work/asr-experiments/qda-continuation-v2')
from contracts import sha,atomic_json

def der_errors(m):return sum(float(m['der_details'][k]) for k in ('missed_speaker_time','falarm_speaker_time','speaker_error_time'))
def aggregate(rows,variant):
 ms=[r[variant]['metrics'] for r in rows];words=sum(m['reference_words'] for m in ms);normalized=sum(m['number_normalized_reference_words'] for m in ms);speech=sum(float(m['der_details']['scored_speaker_time']) for m in ms)
 errors=sum(m['substitutions']+m['deletions']+m['insertions'] for m in ms);num_errors=sum(m['number_normalized_errors'] for m in ms)
 return {'samples':len(rows),'reference_words':words,'number_normalized_reference_words':normalized,'word_errors':errors,'number_normalized_errors':num_errors,'wer':errors/words,'number_normalized_wer':num_errors/normalized,'der_250ms':sum(der_errors(m) for m in ms)/speech,'scored_speaker_seconds':speech,'miss_seconds':sum(float(m['der_details']['missed_speaker_time']) for m in ms),'false_alarm_seconds':sum(float(m['der_details']['falarm_speaker_time']) for m in ms),'speaker_error_seconds':sum(float(m['der_details']['speaker_error_time']) for m in ms)}

def paired(rows,metric):
 if metric=='number_normalized_wer':
  den=np.array([r['vanilla']['metrics']['number_normalized_reference_words'] for r in rows],dtype=float)
  diff=np.array([r['finetuned']['metrics']['number_normalized_errors']-r['vanilla']['metrics']['number_normalized_errors'] for r in rows],dtype=float)
 else:
  den=np.array([float(r['vanilla']['metrics']['der_details']['scored_speaker_time']) for r in rows])
  diff=np.array([der_errors(r['finetuned']['metrics'])-der_errors(r['vanilla']['metrics']) for r in rows])
 rng=np.random.default_rng(20261023);indices=rng.integers(0,len(rows),(20000,len(rows)));values=diff[indices].sum(axis=1)/den[indices].sum(axis=1)
 return {'delta':float(diff.sum()/den.sum()),'paired_bootstrap_95_interval':np.quantile(values,[.025,.975]).tolist(),'bootstrap_draws':20000,'unit':'recording','improved_samples':int((diff<0).sum()),'worse_samples':int((diff>0).sum()),'tied_samples':int((diff==0).sum())}

def main():
 # Scoring is independent of decoding; valid non-speech tags need no speaker identity.
 import importlib.util
 spec=importlib.util.spec_from_file_location('panel_rescore',HERE/'rescore.py');rescore=importlib.util.module_from_spec(spec);spec.loader.exec_module(rescore);rescore.main()
 panel=json.loads((HERE/'panel.json').read_text());rows=[];failures=[]
 for sample in panel['samples']:
  row={'sample_id':sample['sample_id'],'language':sample['language'],'show':sample['show'],'title':sample['title'],'duration_seconds':sample['audio_duration'],'record_id':sample['record_id']}
  for variant in ('vanilla','finetuned'):
   p=HERE/variant/sample['sample_id']/'result.json'
   row[variant]=json.loads(p.read_text()) if p.exists() else {'complete':False,'error':'missing result'}
   if not row[variant].get('complete'):failures.append({'sample_id':sample['sample_id'],'variant':variant,'error':row[variant].get('error',row[variant].get('termination'))})
  if not all(row[v].get('complete') for v in ('vanilla','finetuned')):continue
  for key in ('input_sha256','feature_sha256','reference_sha256'):assert row['vanilla'][key]==row['finetuned'][key]
  for key in ('reference_words','number_normalized_reference_words'):assert row['vanilla']['metrics'][key]==row['finetuned']['metrics'][key]
  assert float(row['vanilla']['metrics']['der_details']['scored_speaker_time'])==float(row['finetuned']['metrics']['der_details']['scored_speaker_time'])
  rows.append(row)
 summary={'status':'complete' if not failures else 'incomplete','checkpoint':'checkpoint-000515','checkpoint_audio_hours':panel['checkpoint_audio_hours'],'audio_minutes':panel['audio_minutes'],'panel_sha256':sha(HERE/'panel.json'),'reference_quality':panel['reference_quality'],'failures':failures,'rows':rows,'interpretation_limits':['Soniox pseudo-references, not human gold.','Ten unseen recordings; series and speakers can overlap baseline training.','Approximately ten-minute windows; this does not establish hour-long session accuracy.','A developmental comparison; recordings are reserved from subsequent training extensions.']}
 if not failures:
  summary['aggregate']={v:aggregate(rows,v) for v in ('vanilla','finetuned')}
  summary['languages']={l:{v:aggregate([r for r in rows if r['language']==l],v) for v in ('vanilla','finetuned')} for l in ('cs','sk')}
  summary['paired']={m:paired(rows,m) for m in ('number_normalized_wer','der_250ms')}
  base=summary['aggregate']['vanilla']['number_normalized_wer'];ft=summary['aggregate']['finetuned']['number_normalized_wer'];summary['relative_word_error_reduction']=(base-ft)/base if base else None
 atomic_json(HERE/'summary.json',summary)
 def pct(x):return f'{100*x:.2f}%'
 lines=['# Vanilla versus checkpoint #515 (250.218 audio hours)','',f'Ten distinct unseen recordings, {panel["audio_minutes"]:.2f} audio minutes; six Czech and four Slovak. Both models use BF16, greedy decoding, identical frozen audio features, empty context, and the same Soniox references.','', 'Training remains paused. References are machine labels, not human gold.','']
 if failures:lines += ['Evaluation incomplete. No aggregate ranking was computed.',json.dumps(failures,ensure_ascii=False,indent=2)]
 else:
  lines+=['| Panel | Normalized WER vanilla | Normalized WER 250h | Strict WER vanilla | Strict WER 250h | DER vanilla | DER 250h |','|---|---:|---:|---:|---:|---:|---:|']
  for label,g in [('All',summary['aggregate']),('Czech',summary['languages']['cs']),('Slovak',summary['languages']['sk'])]:
   a,b=g['vanilla'],g['finetuned'];lines.append('| '+label+' | '+' | '.join(pct(v) for v in (a['number_normalized_wer'],b['number_normalized_wer'],a['wer'],b['wer'],a['der_250ms'],b['der_250ms']))+' |')
  lines+=['','Deltas are checkpoint minus vanilla; negative means improvement. Bootstrap resamples whole recordings (20,000 paired draws).','']
  for metric,label in [('number_normalized_wer','Normalized WER'),('der_250ms','DER')]:
   x=summary['paired'][metric];lo,hi=x['paired_bootstrap_95_interval'];lines.append(f'- {label}: {x["delta"]*100:+.2f} percentage points; paired 95% bootstrap interval [{lo*100:+.2f}, {hi*100:+.2f}]. Improved/worse/tied recordings: {x["improved_samples"]}/{x["worse_samples"]}/{x["tied_samples"]}.')
  lines+=['','| Sample | Language | Show | Normalized WER vanilla → 250h | DER vanilla → 250h |','|---|---|---|---:|---:|']
  for r in rows:
   a,b=r['vanilla']['metrics'],r['finetuned']['metrics'];lines.append(f'| {r["sample_id"]} | {r["language"]} | {r["show"]} | {pct(a["number_normalized_wer"])} → {pct(b["number_normalized_wer"])} | {pct(a["der_250ms"])} → {pct(b["der_250ms"])} |')
  lines+=['','## Interpretation limits','']+[f'- {x}' for x in summary['interpretation_limits']]
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
 with (HERE/'results.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['sample','language','show','vanilla_normalized_wer','finetuned_normalized_wer','vanilla_strict_wer','finetuned_strict_wer','vanilla_der','finetuned_der'])
  for r in rows:
   a,b=r['vanilla']['metrics'],r['finetuned']['metrics'];writer.writerow([r['sample_id'],r['language'],r['show'],a['number_normalized_wer'],b['number_normalized_wer'],a['wer'],b['wer'],a['der_250ms'],b['der_250ms']])
 print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)
if __name__=='__main__':main()
