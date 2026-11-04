"""Read-only review payload. Scored metrics remain authoritative and unchanged."""
import pathlib,json,re,difflib,unicodedata,sys
sys.path.insert(0,'/data/work/asr-evaluation/periodic/code')
from number_wer import tokens as number_tokens
EVAL=pathlib.Path('/data/work/asr-evaluation')
def words(text):
 matches=list(re.finditer(r'\w+',text,flags=re.UNICODE))
 return [{'key':unicodedata.normalize('NFC',m.group()).casefold(),'text':text[m.start():matches[i+1].start() if i+1<len(matches) else len(text)],'offset':m.start()} for i,m in enumerate(matches)]
def accent(s):return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
def align(reference,hypothesis,language):
 a=words(reference);b=words(hypothesis);out=[];errors=[]
 for op,i,j,k,l in difflib.SequenceMatcher(None,[x['key'] for x in a],[x['key'] for x in b],autojunk=False).get_opcodes():
  old=' '.join(x['key'] for x in a[i:j]);new=' '.join(x['key'] for x in b[k:l]);kind='equal'
  if op!='equal':
   kind='extra' if op=='insert' else 'missing' if op=='delete' else 'format' if number_tokens(old,language)==number_tokens(new,language) else 'accent' if accent(old)==accent(new) else 'wording'
   errors.append({'anchor':min(i,max(0,len(a)-1)),'end':j,'kind':kind,'reference':old,'hypothesis':new,'key':f'{i}:{j}:{kind}'})
  if op=='delete':out.append({'text':'['+''.join(x['text'] for x in a[i:j]).strip()+'] ','kind':kind,'anchor':i,'reference':old})
  else:
   for n,x in enumerate(b[k:l]):out.append({'text':x['text'],'kind':kind,'anchor':min(i+n,j-1) if op=='equal' else min(i,max(0,len(a)-1)),'reference':old if op!='equal' else ''})
 return out,errors

def build():
 history=json.loads((EVAL/'periodic/history.json').read_text());runs=[history['baseline']]+history['runs'];samples={};metadata=[]
 for run in runs:
  milestone=0 if run['variant']=='vanilla' else 100 if run['id']=='after-100h' else int(run['training_audio_hours']//50)*50
  root=EVAL/'after-100h' if run['id'] in ('vanilla','after-100h') else EVAL/'periodic/runs'/run['id']
  assert run['id'] in ('vanilla','after-100h') or re.fullmatch(r'checkpoint-\d+',run['id'])
  metadata.append({k:run[k] for k in ('id','variant','checkpoint_step','training_audio_hours','aggregate')}|{'milestone':milestone,'metrics':{s['language']:s['metrics'] for s in run['samples']}})
  for lang in ('cs','sk'):
   folder=root/run['variant']/lang
   if lang not in samples:
    ref=json.loads((folder/'reference.json').read_text());groups=[];text=''
    for seg in ref:
     if not groups or groups[-1]['speaker']!=seg['Speaker']:groups.append({'id':len(groups),'start':seg['Start'],'end':seg['End'],'speaker':seg['Speaker'],'offset':len(text),'text':'','segments':0})
     groups[-1]['text']+=seg['Content'];groups[-1]['end']=seg['End'];groups[-1]['segments']+=1;text+=seg['Content']
    ref_words=words(text)
    for i,g in enumerate(groups):
     end=groups[i+1]['offset'] if i+1<len(groups) else len(text)
     g['lo']=sum(w['offset']<g['offset'] for w in ref_words);g['hi']=sum(w['offset']<end for w in ref_words)
    samples[lang]={'text':text,'turns':groups,'models':{},'audio':f'/references/{lang}/audio.playback.wav'}
   hyp=json.loads((folder/'hypothesis.json').read_text());speech=[s for s in hyp if not re.fullmatch(r'\s*\[[^\[\]]+\]\s*',s['Content'])]
   aligned,errors=align(samples[lang]['text'],' '.join(s['Content'] for s in speech),lang)
   turns=[]
   for g in samples[lang]['turns']:
    segments=[s for s in speech if g['start']<=((s['Start']+s['End'])/2)<g['end']]
    turns.append({'tokens':[x for x in aligned if g['lo']<=x['anchor']<g['hi']], 'errors':[x for x in errors if g['lo']<=x['anchor']<g['hi']], 'segments':segments})
   gaps=[b['Start']-a['End'] for a,b in zip(speech,speech[1:]) if a.get('Speaker')==b.get('Speaker')]
   samples[lang]['models'][run['id']]={'turns':turns,'errors':errors,'segment_count':len(speech),'same_speaker_boundaries':len(gaps),'gap_seconds':sum(max(0,x) for x in gaps)}
 # Pairwise review alignments: independent of gold-error highlighting.
 for lang,sample in samples.items():
  sample['pairs']={}
  for left_id,left_model in sample['models'].items():
   for right_id,right_model in sample['models'].items():
    pair=[]
    for a,b in zip(left_model['turns'],right_model['turns']):
     text=lambda t:''.join(w['text'] for w in t['tokens'] if w['kind']!='missing')
     aa,bb=text(a),text(b)
     right_tokens,_=align(aa,bb,lang);left_tokens,_=align(bb,aa,lang)
     # Deletion placeholders belong in gold review, not in a model transcript.
     pair.append({'left':[w for w in left_tokens if w['kind']!='missing'],'right':[w for w in right_tokens if w['kind']!='missing']})
    sample['pairs'][left_id+'|'+right_id]=pair
 return {'snapshots':metadata,'samples':samples,'note':history['note'],'alignment_note':'Review highlights use case-insensitive word alignment; punctuation is retained for reading. Highlight spans are not WER edit counts. Metrics use the original evaluator. Speaker IDs are local to each output. Timing segments are grouped by their midpoint; transcript text is aligned to reference turns.'}
