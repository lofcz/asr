"""Small bilingual real-audio diagnostic; no relabeling or vocabulary admission gate."""
import collections,hashlib,json,pathlib,random
ROOT=pathlib.Path(__file__).resolve().parents[2]
HERE=pathlib.Path(__file__).resolve().parent
BASE=ROOT/'experiments/qda-full-continuous-v8'

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def main():
 plan=json.loads((BASE/'plan.json').read_text());rng=random.Random(20261025)
 train=[];selected_sources=set()
 # Short diagnostic clips are existing intact windows; no word-boundary surgery.
 for lang in ['cs','sk']:
  pool=[r for pair in plan['probe_pairs'][1729:] for r in pair if r['dataset_kind']=='real' and r['language']==lang and 10<=r['duration']<=60]
  rng.shuffle(pool);picked=[]
  for r in pool:
   if r['record_id'] in selected_sources:continue
   selected_sources.add(r['record_id']);picked.append(r)
   if len(picked)==48:break
  assert len(picked)==48,(lang,len(picked))
  train+=picked
 rng.shuffle(train)
 # For quick iteration, take intact consecutive reference turns near 25% and 70%
 # of each already held-out interview. Use the original full clips for promotion.
 dev=[]
 for row in plan['validation']:
  turns=row['target']
  for fraction in [.25,.7]:
   i=next(i for i,s in enumerate(turns) if s['Start']>=row['duration']*fraction)
   chosen=[turns[i]]
   for s in turns[i+1:]:
    if s['End']-chosen[0]['Start']>45:break
    chosen.append(s)
   start=chosen[0]['Start'];end=max(s['End'] for s in chosen)
   # Keep every overlapping turn, including other-speaker overlap, inside the crop.
   while True:
    included=[s for s in turns if s['Start']<end and s['End']>start]
    a=min(s['Start'] for s in included);b=max(s['End'] for s in included)
    if a==start and b==end:break
    start,end=a,b
   speakers={};target=[{'Start':round(s['Start']-start,2),'End':round(s['End']-start,2),
     'Speaker':speakers.setdefault(s['Speaker'],len(speakers)),'Content':s['Content']} for s in included]
   dev.append({**row,'window_id':digest([row['window_id'],start,end,'e2e-diagnostic'])[:24],
      'parent_window_id':row['window_id'],'source_start':row['source_start']+start,'source_end':row['source_start']+end,
      'duration':end-start,'target':target,'role':'heldout development diagnostic; not untouched confirmation'})
 assert not ({r['record_id'] for r in train}&{r['record_id'] for r in dev})
 assert all(r['split']=='train' for r in train) and all(r['split']=='development' for r in dev)
 def clean(r):return {k:r[k] for k in ['record_id','window_id','language','split','audio_file','audio_sha256','label_path','label_sha256','source_start','source_end','duration','target','customized_context'] if k in r}
 parent_hours=sum(r['duration'] for pair in plan['probe_pairs'][:1729] for r in pair)/3600
 value={'version':1,'parent_step':1729,'parent_audio_hours':parent_hours,'parent_adapter_sha256':'6de9ca639b31f94fbf8c17a8971f04c3f5a31616ad2c9844d52bb72e2251230b',
  'base_plan_sha256':sha(BASE/'plan.json'),'seed':20261025,'train':[clean(r) for r in train],'development':[clean(r) for r in dev],
  'train_source_count':len(selected_sources),'train_audio_hours':sum(r['duration'] for r in train)/3600,
  'development_source_count':len({r['record_id'] for r in dev}),'development_audio_minutes':sum(r['duration'] for r in dev)/60,
  'label_policy':'Exact existing Soniox words and speakers. No grammar normalization or second ASR relabeling.',
  'selection_policy':'48 distinct real sources per language, drawn only from windows after parent step 1729; fixed-seed sampling, no acoustic/speaker/vocabulary quality filters.',
  'long_form_policy':'This bounded diagnostic tests learning and gradients; it does not replace the 10–60 minute deployment evaluation. Full eight interviews, both markers and long sessions remain required.'}
 write(HERE/'manifest.json',value)
 print(json.dumps({k:v for k,v in value.items() if k not in ['train','development']}))
if __name__=='__main__':main()
