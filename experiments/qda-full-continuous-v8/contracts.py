"""Pure experiment contracts: reproducible targets, token masks and WSD schedule."""
import hashlib,json,os,pathlib,tempfile

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def atomic_json(path,value):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
  fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def target_with_spans(sample):
 text='[';spans=[]
 for i,s in enumerate(sample['segments']):
  if i:text+=','
  text+='{'
  row={'Start':round(s['start'],2),'End':round(s['end'],2),'Speaker':s['speaker'],'Content':s['text']}
  for j,(k,v) in enumerate(row.items()):
   if j:text+=','
   text+=json.dumps(k)+':'
   value=json.dumps(v,ensure_ascii=False,separators=(',',':'))
   if k=='Content':spans.append((len(text)+1,len(text)+len(value)-1))
   text+=value
  text+='}'
 return text+']',spans

def content_token_mask(offsets,spans):
 # Mixed boundary tokens count as content if any character overlaps the string value.
 from bisect import bisect_left
 ordered=sorted(spans);starts=[a for a,b in ordered];ends=[];extent=-1
 for a,b in ordered:extent=max(extent,b);ends.append(extent)
 result=[]
 for a,b in offsets:
  index=bisect_left(starts,b)-1
  result.append(b>a and index>=0 and ends[index]>a)
 return result

def wsd_lr(peak,seen,total,warmup,cooldown_fraction=.2,min_ratio=.1):
 if not 0<warmup<total*(1-cooldown_fraction):raise ValueError('Invalid WSD budget')
 if seen<warmup:return peak*max(0.,seen)/warmup
 start=total*(1-cooldown_fraction)
 if seen<=start:return peak
 progress=min(1.,max(0.,(seen-start)/(total-start)))
 return peak*(1-(1-min_ratio)*progress)
