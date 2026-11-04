import pathlib,json,html,time,hashlib
ROOT=pathlib.Path('/data/work/asr-evaluation/after-100h');rows=[]
for lang in ('cs','sk'):
 base=json.loads((ROOT/'vanilla'/lang/'result.json').read_text());ft=json.loads((ROOT/'finetuned'/lang/'result.json').read_text())
 for key in ('feature_sha256','audio_sha256','reference_sha256','dtype','decoding','customized_context'):assert base[key]==ft[key],f'Unequal {key}'
 for result in (base,ft):assert result.get('termination')=='eos' and 'metrics' in result,result
 rows.append({'language':lang,'vanilla':base,'finetuned':ft,'wer_delta_percentage_points':100*(ft['metrics']['wer']-base['metrics']['wer']),'cpwer_delta_percentage_points':100*(ft['metrics']['cpwer']-base['metrics']['cpwer'])})
weights=sum(r['vanilla']['metrics']['reference_words'] for r in rows)
aggregate={v:{key:sum(r[v]['metrics'][key]*r[v]['metrics']['reference_words'] for r in rows)/weights for key in ('wer','cpwer')} for v in ('vanilla','finetuned')}
result={'status':'completed','checkpoint_step':325,'training_audio_hours':102.06957241319445,'created_at':time.time(),'adapter_sha256':hashlib.sha256((ROOT/'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),'samples':rows,'word_weighted_aggregate':aggregate,'reference':'User-approved Soniox gold; two generated dialogue fixtures, outside training. Anecdotal evaluation, not a representative QDA benchmark.','normalization':'Case/punctuation normalized; diacritics retained. Digits and spoken number spellings differ. cpWER permits speaker-ID permutation. DER uses Soniox speech intervals and 250ms collar; short within-turn gaps affect this metric.'}
(ROOT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n')
head='''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>102-hour checkpoint comparison</title><style>body{font:15px system-ui;background:#101722;color:#e6edf8;max-width:1400px;margin:auto;padding:28px}h1{font-size:29px}p{color:#aabbd0}a{color:#8de3bf}table{border-collapse:collapse;width:100%}td,th{padding:14px;text-align:left;border:1px solid #304157}th{background:#1b293c}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.box{background:#172233;padding:16px;border-radius:10px;white-space:pre-wrap;max-height:600px;overflow:auto;font-size:13px}.note{background:#30281d;border-left:3px solid #e7bb72;padding:14px}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><h1>Vanilla vs. the 102-hour checkpoint</h1><p>Update 325 · Same BF16 base, identical frozen audio features, greedy decoding, no context hints.</p><p class="note">Two user-approved gold dialogue fixtures. Useful checkpoint evidence, not a broad estimate of natural QDA accuracy. Numeral spelling differences count in WER. Lower is better for all error metrics.</p><table><tr><th>Sample</th><th>Model</th><th>WER</th><th>cpWER</th><th>CER</th><th>DER (250ms)</th><th>Speakers</th></tr>'''
for row in rows:
 for variant in ('vanilla','finetuned'):
  m=row[variant]['metrics'];pct=lambda k:f'{100*m[k]:.2f}%' if k in m else 'Unavailable'
  head+=f"<tr><td>{row['language'].upper()}</td><td>{variant}</td><td>{pct('wer')}</td><td>{pct('cpwer')}</td><td>{pct('cer')}</td><td>{pct('der_250ms')}</td><td>{m['predicted_speakers']}</td></tr>"
head+='</table>'
for row in rows:
 lang=row['language'];head+=f'<h2>{lang.upper()} transcripts</h2><p><a href="/references/{lang}/review.html">Play reference audio ↗</a></p><div class="grid">'
 for name,path in [('Gold',ROOT/'vanilla'/lang/'reference.json'),('Vanilla',ROOT/'vanilla'/lang/'hypothesis.json'),('Fine-tuned',ROOT/'finetuned'/lang/'hypothesis.json')]:
  segs=json.loads(path.read_text());text='\n\n'.join(f"[{s['Start']:.2f}–{s['End']:.2f}] Speaker {s.get('Speaker','event')}: {s['Content']}" for s in segs)
  head+=f'<section><h3>{name}</h3><div class="box">{html.escape(text)}</div></section>'
 head+='</div>'
head+='<p>Full metrics and provenance: <a href="/evaluations/100h/comparison.json">comparison.json</a></p></html>'
(ROOT/'report.html').write_text(head)
print(json.dumps({'samples':[{ 'language':r['language'],'vanilla':r['vanilla']['metrics'],'finetuned':r['finetuned']['metrics']} for r in rows],'aggregate':aggregate},default=str))
