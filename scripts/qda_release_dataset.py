"""Source-offset loader for the frozen Soniox speech/diarization pseudo-label profile."""
import hashlib,json,pathlib,sqlite3,subprocess,os
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
class ReleaseDataset:
 def __init__(self,root,split,nominal_seconds):
  if split not in ('train','validation','test'):raise ValueError('Invalid split')
  if nominal_seconds not in (120,600,1800,3600):raise ValueError('Choose a declared curriculum view')
  self.root=pathlib.Path(root);self.verified={};self.label_cache={};db=sqlite3.connect(f'file:{self.root}/training/catalog.sqlite3?mode=ro',uri=True);db.row_factory=sqlite3.Row
  self.rows=[dict(r) for r in db.execute('SELECT w.*,s.audio_path,s.audio_sha256,s.label_path,s.label_sha256 FROM windows w JOIN sources s USING(record_id) WHERE w.split=?',(split,)) if any(v['nominal_seconds']==nominal_seconds and not v['oversized'] for v in json.loads(r['views_json']))];db.close()
 def __len__(self):return len(self.rows)
 def label(self,index):
  row=self.rows[index];path=self.root/row['label_path']
  if row['record_id'] not in self.label_cache:
   if sha(path)!=row['label_sha256']:raise ValueError('Label checksum mismatch')
   doc=json.loads(path.read_bytes())
   if doc['record_id']!=row['record_id'] or doc['split']!=row['split'] or doc['audio_sha256']!=row['audio_sha256']:raise ValueError('Source/label pairing mismatch')
   self.label_cache[row['record_id']]=doc
  doc=self.label_cache[row['record_id']];w=json.loads(row['window_json']);segments=[]
  for s in doc['segments'][w['segment_lo']:w['segment_hi']]:
   if s.get('is_event'):raise ValueError('Wrong profile: speech baseline must not contain event proposals')
   segments.append({'start':s['start']-w['source_start'],'end':s['end']-w['source_start'],'speaker':w['session_to_window_speaker'][s['session_speaker']],'text':s['text']})
  return {'window_id':row['window_id'],'record_id':row['record_id'],'audio_duration':w['audio_duration'],'source_start':w['source_start'],'source_end':w['source_end'],'audio_path':row['audio_path'],'audio_sha256':row['audio_sha256'],'split':row['split'],'segments':segments,'customized_context':doc['customized_context'],'label_quality':'machine_validated_pseudo_label','profile':'speech-diarization-pseudo-v1'}
 def decode(self,index):
  sample=self.label(index);path=self.root/sample['audio_path'];signature=(path.stat().st_size,path.stat().st_mtime_ns)
  if self.verified.get(str(path))!=signature:
   if sha(path)!=sample['audio_sha256']:raise ValueError('Audio checksum mismatch')
   self.verified[str(path)]=signature
  ffmpeg=os.environ.get('FFMPEG')
  if not ffmpeg:
   import imageio_ffmpeg
   ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
  proc=subprocess.run([ffmpeg,'-nostdin','-v','error','-ss',str(sample['source_start']),'-i',str(path),'-t',str(sample['audio_duration']),'-vn','-ar','24000','-ac','1','-f','f32le','-'],capture_output=True,check=True)
  if proc.stderr:raise ValueError('Decode diagnostics require review: '+proc.stderr.decode(errors='replace')[:500])
  wave=np.frombuffer(proc.stdout,dtype='<f4').copy()
  if abs(len(wave)/24000-sample['audio_duration'])>.02:raise ValueError('Cropped duration mismatch')
  if not np.isfinite(wave).all():raise ValueError('Nonfinite audio')
  return sample,wave

def target_json(sample):
 rows=[{'Start':round(s['start'],2),'End':round(s['end'],2),'Speaker':s['speaker'],'Content':s['text']} for s in sample['segments']]
 return json.dumps(rows,ensure_ascii=False,separators=(',',':'))
