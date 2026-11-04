import pathlib,json,re,difflib,html,hashlib,shutil,time
from label_production_samples import STORE,OUT,ROOT,rows

def words(text):return re.findall(r'\w+',text.casefold())
def distance(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  cur=[i]
  for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=cur
 return prev[-1]
summary=[];gold_index=[]
for r in rows():
 source=json.loads(((ROOT/r['path']).parent/'manifest.json').read_text())
 lang=r['language'];folder=OUT/lang;son=json.loads((folder/'soniox.segments.json').read_text());gem=json.loads((folder/'gemini-independent.parsed.json').read_text());audit=json.loads((folder/'gemini-adjudicate.parsed.json').read_text());candidate=[dict(s) for s in son];decisions=[]
 for d in audit['decisions']:
  d=dict(d);i=d['soniox_segment_index']
  if not (isinstance(i,int) and 0<=i<len(son) and d['original_text']==son[i]['text']):
   matches=[j for j,s in enumerate(son) if s['text']==d['original_text']]
   assert len(matches)==1,'Ambiguous original text; cannot pair Gemini correction'
   d['model_claimed_segment_index']=i;i=matches[0];d['soniox_segment_index']=i;d['index_resolution']='unique exact original-text match'
  assert isinstance(d['proposed_text'],str);assert i not in [x['soniox_segment_index'] for x in decisions]
  d={**d,'start':son[i]['start'],'end':son[i]['end'],'applied_to_candidate':d['confidence']=='high' and not d['needs_human_review']}
  if d['applied_to_candidate']:candidate[i]['text']=d['proposed_text']
  decisions.append(d)
 human_path=folder/'human-corrections.json'
 human=json.loads(human_path.read_text()) if human_path.exists() else []
 for correction in human:
  assert correction['audio_sha256']==r['sha256']
  i=correction['soniox_segment_index'];assert correction['scope']=='text_only'
  candidate[i]['text']=correction['text']
  decision=next((d for d in decisions if d['soniox_segment_index']==i),None)
  if decision is None:
   decision={'soniox_segment_index':i,'original_text':son[i]['text'],'start':son[i]['start'],'end':son[i]['end']};decisions.append(decision)
  decision.update(proposed_text=correction['text'],applied_to_candidate=True,needs_human_review=False,human_verified_text=True,reason='Text confirmed by user in conversation; speaker and timing not verified.',confidence='human_confirmed')
 STORE.put(f'{lang}/audio.playback.wav',(ROOT/r['path']).read_bytes())
 independent_timing_issues=[]
 for i,s in enumerate(gem['segments']):
  if 'start' not in s or 'end' not in s:independent_timing_issues.append({'segment_index':i,'issue':'Gemini omitted timestamp; not used for reference alignment'})
  else:assert 0<=s['start']<=s['end']<=r['seconds']+1
 for s in son:assert 0<=s['start']<=s['end']<=r['seconds']+1
 a=words(''.join(s['text'] for s in son));b=words(' '.join(s['text'] for s in gem['segments']));diff=[{'soniox':' '.join(a[i:j]),'gemini':' '.join(b[k:l]),'soniox_word_span':[i,j]} for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes() if tag!='equal']
 for name,segs in [('soniox.reference.json',son),('candidate.reference.json',candidate)]:STORE.obj(f'{lang}/{name}',[{'Start':s['start'],'End':s['end'],'Speaker':s['speaker'],'Content':s['text']} for s in segs])
 review=audit['unresolved']+audit['speaker_audit']['issues']+audit['coverage_issues']+[{'start':d['start'],'end':d['end'],'reason':d['reason']} for d in decisions if d['needs_human_review']]
 report={'source_title':source['title'],'production_transcript_id':source['production_transcript_id'],'source_kind':'generated dialogue fixture' if (folder/'authored-fixture-script.json').exists() else 'production test recording','language':lang,'duration_seconds':r['seconds'],'audio_sha256':r['sha256'],'status':'machine_cross_checked_reference_candidate','human_verified':False,'human_verified_text_segments':[x['soniox_segment_index'] for x in human],'gold_ready':False,'test_edits_used':False,'training_allowed':False,'reference_words':len(a),'independent_words':len(b),'independent_levenshtein_disagreement_rate':distance(a,b)/len(a),'metric_note':'Cross-system disagreement, NOT measured WER against human ground truth. Numeral spellings count as differences.','independent_differences':diff,'soniox_speakers':len({s['speaker'] for s in son}),'gemini_speakers':gem['speaker_count'],'decisions':decisions,'review_queue':review,'independent_uncertainties':gem.get('uncertainties',[]),'independent_timing_issues':independent_timing_issues,'speaker_audit':audit['speaker_audit'],'model_summary':audit['summary'],'applied_candidate_patches':sum(d['applied_to_candidate'] for d in decisions),'timing_provenance':'Soniox segment boundaries retained. Gemini timing is an approximate review aid. Corrected words do not inherit verified word alignment.','created_at':time.time()}
 approval_path=folder/'gold-approval.json'
 approval=json.loads(approval_path.read_text()) if approval_path.exists() else None
 if approval:
  assert approval['audio_sha256']==r['sha256']
  raw_reference=(folder/'soniox.reference.json').read_bytes()
  assert hashlib.sha256(raw_reference).hexdigest()==approval['reference_sha256']
  STORE.put(f'{lang}/gold.reference.json',raw_reference)
  report.update(status='user_approved_gold',gold_ready=True,user_approved_gold=True,gold_reference_file='gold.reference.json',gold_reference_sha256=approval['reference_sha256'],approval=approval,resolved_review_items=review,review_queue=[])
  gold_index.append({'language':lang,'title':source['title'],'production_transcript_id':source['production_transcript_id'],'audio_path':f'{lang}/audio.playback.wav','audio_sha256':r['sha256'],'reference_path':f'{lang}/gold.reference.json','reference_sha256':approval['reference_sha256'],'approval_path':f'{lang}/gold-approval.json','duration_seconds':r['seconds'],'split':'external_anecdote_gold','training_allowed':False})
 STORE.obj(f'{lang}/audit.json',report);summary.append({k:v for k,v in report.items() if k not in ('decisions','review_queue','independent_differences','independent_uncertainties')})
 # Persist the lossless audio and authored-script evidence on the independent disk too.
 for name in ('audio.lossless.flac','authored-fixture-script.json'):
  if (folder/name).exists():STORE.put(f'{lang}/{name}',(folder/name).read_bytes())
 blocks=[]
 for i,s in enumerate(son):
  proposed=candidate[i]['text'];decision=next((d for d in decisions if d['soniox_segment_index']==i),None)
  reason=decision['reason'] if decision else 'No lexical correction proposed by the audio audit.'
  status='Candidate change — verify by listening' if proposed!=s['text'] else 'Soniox retained'
  if decision and decision['needs_human_review']:status='Unresolved — needs listening';proposed=decision['proposed_text']
  if decision and decision.get('human_verified_text'):status='Text verified by user'
  if approval:status='User-approved gold · Soniox retained';proposed=s['text']
  blocks.append(f'<tr><td><button onclick="seek({s["start"]})">{s["start"]:.1f}–{s["end"]:.1f}s ▶</button><p>Speaker {s["speaker"]}</p></td><td>{html.escape(s["text"])}</td><td>{html.escape(proposed)}</td><td><b>{status}</b><p>{html.escape(reason)}</p></td></tr>')
 if approval:review=[]
 notes=''.join(f'<li><button onclick="seek({x["start"]})">{x["start"]:.1f}s ▶</button> {html.escape(x["reason"])}</li>' for x in review)
 page='''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Reference audio review</title><style>body{font:15px system-ui;background:#101722;color:#e4ebf5;max-width:1400px;margin:24px auto;padding:20px}header{position:sticky;top:0;background:#101722;padding:12px;border-bottom:1px solid #425}audio{width:100%}table{width:100%;border-collapse:collapse}td,th{padding:14px;border:1px solid #344054;vertical-align:top;text-align:left}td:nth-child(2),td:nth-child(3){width:28%}button{background:#273a50;color:#9ff1d5;border:0;border-radius:5px;padding:7px;cursor:pointer}p{color:#b0bfd1}a{color:#8cddc0}.warning{padding:14px;border-left:3px solid #ffb66a;background:#29221d}</style>'''+f'<header><h1>{lang.upper()} · {html.escape(source["title"])}</h1><p>{html.escape(report["source_kind"])} · {r["seconds"]:.2f} seconds</p><audio id="audio" controls preload="metadata" src="audio.playback.wav"></audio><span id="play-status" role="status"></span></header><p class="warning">Machine-cross-checked candidate; NOT human-verified gold. Test edits excluded. Audio and transcript are outside training. Soniox timestamps remain unverified by a human. Listen to the entire recording before promoting to gold.</p><p>{r["seconds"]:.2f} seconds · {len(a)} Soniox words · {report["soniox_speakers"]} Soniox / {report["gemini_speakers"]} Gemini speakers. <a href="audit.json">Full audit JSON</a> · <a href="candidate.reference.json">Candidate labels</a></p><h2>Review queue</h2><ul>{notes or "<li>No unresolved passages flagged by the models; human verification still pending.</li>"}</ul><table><tr><th>Audio interval</th><th>Fresh Soniox</th><th>Candidate / proposal</th><th>Audit</th></tr>'+''.join(blocks)+'</table><script>let seekRequest=0;async function seek(t){const id=++seekRequest,a=document.getElementById("audio"),status=document.getElementById("play-status");a.pause();try{if(a.readyState<1)await new Promise((resolve,reject)=>{a.addEventListener("loadedmetadata",resolve,{once:true});a.addEventListener("error",reject,{once:true});a.load()});if(id!==seekRequest)return;const target=Math.max(0,Math.min(t,a.duration-.01));status.textContent="Seeking to "+target.toFixed(2)+"s…";if(Math.abs(a.currentTime-target)>.01)await new Promise((resolve,reject)=>{const done=()=>{clearTimeout(timer);a.removeEventListener("seeked",done);resolve()};const timer=setTimeout(()=>{a.removeEventListener("seeked",done);reject(Error("Seek timed out. Please retry."))},10000);a.addEventListener("seeked",done);a.currentTime=target});if(id!==seekRequest)return;await a.play();status.textContent="Playing from "+target.toFixed(2)+"s"}catch(e){if(id===seekRequest)status.textContent=e.message||"Audio unavailable"}}</script></html>'
 if approval:
  page=page.replace('Machine-cross-checked candidate; NOT human-verified gold. Test edits excluded. Audio and transcript are outside training. Soniox timestamps remain unverified by a human. Listen to the entire recording before promoting to gold.','User-approved gold reference. Fresh Soniox wording, speaker labels and timestamps retained unchanged after Gemini cross-check and your acceptance. Production test edits excluded. Evaluation only; never training.')
  page=page.replace('href="candidate.reference.json">Candidate labels','href="gold.reference.json">Gold labels').replace('No unresolved passages flagged by the models; human verification still pending.','Approved by user. Soniox version retained; prior model flags remain in the audit record.').replace('<th>Candidate / proposal</th>','<th>Gold reference</th>').replace('<h2>Review queue</h2>','<h2>Approval status</h2>')
 STORE.put(f'{lang}/review.html',page.encode())
STORE.obj('summary.json',summary)
STORE.obj('gold-manifest.json',{'status':'user_approved_gold','samples':gold_index,'training_allowed':False})
checks={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.json'};STORE.obj('checksums.json',checks)
print(json.dumps([{'language':x['language'],'words':x['reference_words'],'candidate_patches':x['applied_candidate_patches'],'disagreement_rate':x['independent_levenshtein_disagreement_rate']} for x in summary]))
