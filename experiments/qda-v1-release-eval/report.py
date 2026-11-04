"""Progressive release comparison; preserve failures and paired uncertainty."""
import argparse,collections,csv,html,json,math,pathlib,shutil,time,unicodedata
from common import HERE,BASE,atomic_json,sha

def aggregate(rows,expected):
    good=[r for r in rows if r.get('complete') and 'metrics' in r]
    result={'expected':expected,'finished':len(rows),'scored':len(good),'failures':len(rows)-len(good),
            'strict_output_valid':sum(r.get('strict_output_valid',not r.get('output_issues')) for r in good)}
    if good:
        ms=[r['metrics'] for r in good];n=sum(m['number_normalized_reference_words'] for m in ms);e=sum(m['number_normalized_errors'] for m in ms)
        raw_n=sum(m['reference_words'] for m in ms);raw_e=sum(m[k] for m in ms for k in ['substitutions','deletions','insertions'])
        speaker_time=sum(float(m['der_details']['scored_speaker_time']) for m in ms)
        der_errors=sum(float(m['der_details'][k]) for m in ms for k in ['missed_speaker_time','falarm_speaker_time','speaker_error_time'])
        result.update(number_normalized_wer=e/n,number_normalized_errors=e,reference_words=n,ordinary_wer=raw_e/raw_n,
                      ordinary_errors=raw_e,ordinary_reference_words=raw_n,der_250ms=der_errors/speaker_time if speaker_time else None,
                      diarization_error_seconds=der_errors,speaker_seconds=speaker_time)
    result['metric_scope']='All expected recordings' if len(good)==expected else 'Completed valid outputs only; pending/failed recordings remain explicit and are not assigned fabricated WER'
    return result

def paired(a,b,by='record_id',draws=10000):
    import numpy as np
    shared=sorted(set(a)&set(b));shared=[k for k in shared if a[k].get('complete') and b[k].get('complete')]
    rng=np.random.default_rng(20261025);samples=[];units={}
    for lang in ['cs','sk']:
        groups=collections.defaultdict(list)
        for key in shared:
            if a[key]['language']==lang:groups[a[key][by]].append(key)
        if not groups:return {'matched_recordings':len(shared),'insufficient_data':True}
        values=[]
        for keys in groups.values():
            side=[]
            for source in [a,b]:
                m=aggregate([source[k] for k in keys],len(keys));side.append([m['number_normalized_errors'],m['reference_words'],m['diarization_error_seconds'],m['speaker_seconds']])
            values.append(side)
        x=np.asarray(values,dtype=np.float64);idx=rng.integers(0,len(x),size=(draws,len(x)));s=x[idx].sum(axis=1)
        samples.append(s);units[lang]=len(x)
    wer=np.mean([s[:,:,0]/s[:,:,1] for s in samples],axis=0);s=sum(samples);der=s[:,:,2]/s[:,:,3]
    return {'matched_recordings':len(shared),'cluster':by,'clusters_by_language':units,'draws':draws,
            'balanced_language_wer_delta_95':np.quantile(wer[:,1]-wer[:,0],[.025,.975]).tolist(),
            'der_delta_95':np.quantile(der[:,1]-der[:,0],[.025,.975]).tolist(),
            'direction':'second model minus first; negative favors second',
            'limitation':'Paired descriptive uncertainty during checkpoint selection; not a blind post-selection significance claim.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');args=p.parse_args()
    selection=json.loads((HERE/'selection.json').read_text());main_ids={r['window_id'] for r in selection['rows']};long_ids={r['window_id'] for r in selection['long_rows']}
    by_id={r['window_id']:r for r in selection['rows']+selection['long_rows']};result={'updated_at':time.time(),'final':args.final,'selection_sha256':sha(HERE/'selection.json'),'models':{},'failures':[]}
    records={};total=0
    for name,spec in selection['models'].items():
        rows={}
        for path in (HERE/'evaluations'/name).glob('*/result.json'):
            row=json.loads(path.read_text());wid=row['window_id']
            assert wid in by_id and row['adapter_sha256']==spec['adapter_sha256']
            rows[wid]=row
            if not row.get('complete'):result['failures'].append({'model':name,'window_id':wid,'language':row['language'],'cohort':row['cohort'],'termination':row.get('termination'),'error':row.get('error')})
        records[name]=rows;total+=len(rows)
        measured={k:rows[k] for k in main_ids if k in rows}
        scores={'main':aggregate(list(measured.values()),len(main_ids)),'long':aggregate([rows[k] for k in long_ids if k in rows],len(long_ids)),'languages':{},'cohorts':{}}
        for lang in ['cs','sk']:scores['languages'][lang]=aggregate([r for r in measured.values() if r['language']==lang],16)
        for cohort in ['existing-development','new-development','new-confirmation']:
            expected=sum(r['cohort']==cohort for r in selection['rows'])
            scores['cohorts'][cohort]=aggregate([r for r in measured.values() if r['cohort']==cohort],expected)
        if all('number_normalized_wer' in scores['languages'][lang] for lang in ['cs','sk']):
            scores['balanced_language_wer']=sum(scores['languages'][lang]['number_normalized_wer'] for lang in ['cs','sk'])/2
        result['models'][name]=scores
    result.update(finished=total,expected=(len(main_ids)+len(long_ids))*len(selection['models']))
    # Existing fixed markers remain visible, separate from this wider real panel.
    result['markers']={}
    for name in selection['models']:
        old_name={'300h':'r64-full-300h','450h':'r64-full-450h','final':'r64-full-final','vanilla':'vanilla'}[name]
        path=BASE/'markers'/old_name/'completed-v3.json'
        if path.exists():
            d=json.loads(path.read_text());assert d['adapter_sha256']==selection['models'][name]['adapter_sha256']
            result['markers'][name]={'languages':d['languages'],'source_sha256':sha(path)}
    if args.final:
        assert total==result['expected'],'Cannot finalize an unfinished panel'
        result['comparisons']={}
        for first,second in [('vanilla','300h'),('vanilla','450h'),('vanilla','final'),('300h','450h'),('300h','final'),('450h','final')]:
            a={k:v for k,v in records[first].items() if k in main_ids};b={k:v for k,v in records[second].items() if k in main_ids}
            result['comparisons'][first+'→'+second]={key:paired(a,b,key) for key in ['record_id','show']}
        candidates=['300h','450h','final']
        # Release recommendation is descriptive. Incomplete long decoding is
        # explicitly a capability limitation, never hidden by a short-clip WER.
        candidates.sort(key=lambda n:(-result['models'][n]['main']['scored'],result['models'][n].get('balanced_language_wer',math.inf)))
        best=candidates[0];ties=[best]
        for name in candidates[1:]:
            if result['models'][name]['main']['scored']!=result['models'][best]['main']['scored']:continue
            a={k:v for k,v in records[best].items() if k in main_ids};b={k:v for k,v in records[name].items() if k in main_ids}
            ci=paired(a,b,'show').get('balanced_language_wer_delta_95')
            if ci and ci[0]<=0<=ci[1]:ties.append(name)
        recommended=min(ties,key=lambda n:(-result['models'][n]['long']['scored'],result['models'][n]['main'].get('der_250ms',math.inf)))
        result['recommendation']={'checkpoint':recommended,'text_best_point_estimate':best,'statistically_unresolved_candidates':ties,
                                  'rule':'Prioritize completed main clips and lower balanced-language WER. When paired show-cluster differences include zero, favor completed long sessions then lower DER.',
                                  'main_all_completed':result['models'][recommended]['main']['scored']==32,
                                  'long_all_completed':result['models'][recommended]['long']['scored']==2,
                                  'automatic_deployment':False}
        atomic_json(HERE/'v1-candidate.json',{'model':selection['models'][recommended],**result['recommendation'],'evaluation_report':'report.json','selection_sha256':result['selection_sha256']})
        with (HERE/'word-errors.csv').open('w',newline='') as f:
            fields=['model','language','show','record_id','window_id','operation','reference_word','hypothesis_word','reference_index','hypothesis_index']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for name,rows in records.items():
                for wid,row in rows.items():
                    path=HERE/'evaluations'/name/wid/'word-errors.json'
                    if path.exists():
                        for error in json.loads(path.read_text()):writer.writerow({'model':name,**{k:row[k] for k in ['language','show','record_id','window_id']},**error})
    atomic_json(HERE/'report.json',result)
    if args.final:
        candidate=json.loads((HERE/'v1-candidate.json').read_text())
        candidate.update(report_sha256=sha(HERE/'report.json'),model_revision=json.loads((HERE/'feature-contract.json').read_text())['model_revision'],
                         evaluated_inference_dtype='bfloat16',evaluated_adapter_mode='merged into the pinned unquantized base')
        atomic_json(HERE/'v1-candidate.json',candidate)
    fmt=lambda x:'—' if x is None else f'{100*x:.3f}%'
    lines=['# QLoRA v1 release comparison','',f"{'Completed' if args.final else 'In progress'}: {total}/{result['expected']} model/recording evaluations.",'',
           '32 main recordings, 16 per language; 319.1 minutes. Two additional long sessions total 85.1 minutes. Soniox references are unchanged. Synthetic markers are reported separately.','',
           '| Checkpoint | Main scored / expected | CS WER | SK WER | Balanced WER | Ordinary WER | DER | Long scored / 2 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['vanilla','300h','450h','final']:
        s=result['models'][name];m=s['main'];cs=s['languages']['cs'];sk=s['languages']['sk']
        lines.append(f"| {name} | {m['scored']}/32 | {fmt(cs.get('number_normalized_wer'))} | {fmt(sk.get('number_normalized_wer'))} | {fmt(s.get('balanced_language_wer'))} | {fmt(m.get('ordinary_wer'))} | {fmt(m.get('der_250ms'))} | {s['long']['scored']}/2 |")
    lines+=['','WER/DER for incomplete panels cover only scored clips. Compare checkpoint values only on matched recordings; failures remain in the completion counts. Diacritics and nonnumeric inflections remain scored.','',f"Failures recorded: {len(result['failures'])}."]
    for failure in result['failures']:lines.append(f"- {failure['model']} / {failure['language']} / {failure['window_id']}: {failure['termination']} — {failure['error']}")
    if args.final:
        r=result['recommendation'];lines+=['',f"Recommended temporary v1 candidate: **{r['checkpoint']}**. Lowest text point estimate: {r['text_best_point_estimate']}. Unresolved candidates under the paired show-level interval: {', '.join(r['statistically_unresolved_candidates'])}.",
            '','Complete per-language/cohort results, paired confidence intervals and marker scores are in report.json. Word-level errors are in word-errors.csv. No model is automatically deployed.']
        lines+=['','| Comparison: second minus first | WER delta 95% interval, recordings | WER delta 95% interval, shows |','|---|---:|---:|']
        for name,comparison in result['comparisons'].items():
            intervals=[]
            for key in ['record_id','show']:
                ci=comparison[key].get('balanced_language_wer_delta_95')
                intervals.append(f'{100*ci[0]:+.3f} to {100*ci[1]:+.3f} pp' if ci else 'insufficient matched data')
            lines.append(f'| {name} | {intervals[0]} | {intervals[1]} |')
        lines+=['','Negative differences favor the second checkpoint. An interval crossing zero leaves that text difference unresolved. Intervals are descriptive after checkpoint selection.','',
                '| Checkpoint | Existing development WER | New development WER | Reserved confirmation WER |','|---|---:|---:|---:|']
        for name in ['vanilla','300h','450h','final']:
            c=result['models'][name]['cohorts'];lines.append(f"| {name} | {fmt(c['existing-development'].get('number_normalized_wer'))} | {fmt(c['new-development'].get('number_normalized_wer'))} | {fmt(c['new-confirmation'].get('number_normalized_wer'))} |")
        if not r['long_all_completed']:lines+=['','**Long-session limitation:** the recommended checkpoint did not complete both unchunked long sessions. This evaluation does not establish reliable 30–60 minute direct decoding.']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
