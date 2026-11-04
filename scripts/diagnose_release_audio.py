import pathlib,json,concurrent.futures,subprocess,tempfile,time,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from soniox_durable import atomic,encode
root=pathlib.Path(__file__).resolve().parents[1];rel=root/'datasets/releases/qda-496h-v1';base=root/'datasets/soniox-500h-v1';rows=json.loads((rel/'selection.json').read_bytes())
def run(r):
 dest=rel/'audio-diagnostics'/f"{r['record_id']}.json"
 if dest.exists():return json.loads(dest.read_bytes())
 t=time.monotonic();count=0
 with tempfile.TemporaryFile() as err:
  proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-i',str(root/r['path']),'-vn','-ar','16000','-ac','1','-f','f32le','-'],stdout=subprocess.PIPE,stderr=err)
  while b:=proc.stdout.read(1024*1024):count+=len(b)
  code=proc.wait();err.seek(0);errors=err.read().decode('utf-8',errors='replace')
 raw=json.loads((base/'records'/r['record_id']/'soniox.raw.json').read_bytes());last=max((t.get('end_ms',0) for t in raw['tokens']),default=0)/1000
 result={'record_id':r['record_id'],'audio_sha256':r['sha256'],'manifest_seconds':r['seconds'],'decoded_seconds':count/4/16000,'last_token_seconds':last,'ffmpeg_returncode':code,'decoder_stderr':errors,'elapsed_seconds':time.monotonic()-t}
 atomic(dest,encode(result));print(json.dumps(result),flush=True);return result
rows=[r for r in rows if (base/'records'/r['record_id']/'vad-error.json').exists()]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,rows))
