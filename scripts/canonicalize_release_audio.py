"""Recover a clean training container without changing the decoded 24kHz PCM timeline."""
import concurrent.futures,hashlib,json,pathlib,subprocess,sys,tempfile,time,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from soniox_durable import atomic,encode,file_hash
ROOT=pathlib.Path(__file__).resolve().parents[1];REL=ROOT/'datasets/releases/qda-496h-v1'
def pcm_hash(path):
 h=hashlib.sha256();n=0
 p=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-i',str(path),'-vn','-ar','24000','-ac','1','-f','s16le','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 while b:=p.stdout.read(1024*1024):h.update(b);n+=len(b)
 stderr=p.stderr.read().decode();code=p.wait()
 if code or stderr:raise ValueError(f'Canonical decode not clean: {code} {stderr}')
 return h.hexdigest(),n

def run(r):
 folder=REL/'canonical';folder.mkdir(exist_ok=True);receipt=folder/f"{r['record_id']}.json";dest=folder/f"{r['record_id']}.flac"
 if receipt.exists():return json.loads(receipt.read_bytes())
 source=ROOT/r['path']
 if file_hash(source)!=r['sha256']:raise ValueError('Original source hash mismatch')
 tmp=dest.with_suffix('.part.flac');h=hashlib.sha256();n=0;started=time.monotonic()
 with tempfile.TemporaryFile() as err1,tempfile.TemporaryFile() as err2:
  decoder=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-i',str(source),'-vn','-ar','24000','-ac','1','-f','s16le','-'],stdout=subprocess.PIPE,stderr=err1)
  encoder=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','s16le','-ar','24000','-ac','1','-i','-','-c:a','flac','-frame_size','2304',str(tmp)],stdin=subprocess.PIPE,stderr=err2)
  try:
   while block:=decoder.stdout.read(1024*1024):h.update(block);n+=len(block);encoder.stdin.write(block)
   encoder.stdin.close();dc=decoder.wait();ec=encoder.wait();err1.seek(0);original_errors=err1.read().decode();err2.seek(0);encode_errors=err2.read().decode()
   if dc or ec or encode_errors or not n:raise ValueError(f'Canonicalization failed: {dc}, {ec}, {encode_errors}')
  finally:
   for p in (decoder,encoder):
    if p.poll() is None:p.kill();p.wait()
 checks,n2=pcm_hash(tmp)
 if checks!=h.hexdigest() or n2!=n:raise ValueError('Canonical PCM differs from original decoded PCM')
 os.replace(tmp,dest)
 raw=json.loads((ROOT/'datasets/soniox-500h-v1/records'/r['record_id']/'soniox.raw.json').read_bytes());last=max(t.get('end_ms',0) for t in raw['tokens'])/1000
 result={'record_id':r['record_id'],'original_path':r['path'],'original_sha256':r['sha256'],'canonical_path':'canonical/'+dest.name,'canonical_sha256':file_hash(dest),'pcm_sha256':checks,'pcm_sample_rate':24000,'pcm_channels':1,'pcm_format':'s16le','pcm_samples':n//2,'decoded_seconds':n/48000,'original_manifest_seconds':r['seconds'],'last_soniox_token_seconds':last,'all_tokens_fit_decoded_audio':last<=n/48000+.001,'original_decoder_diagnostics':original_errors,'canonical_decode_clean':True,'waveform_equivalence':'Byte-identical decoded 24kHz mono signed-16 PCM, including exact sample count. No cuts, time shifts or silence removal. Original retained.','elapsed_seconds':time.monotonic()-started}
 atomic(receipt,encode(result));print(json.dumps({k:result[k] for k in ('record_id','decoded_seconds','all_tokens_fit_decoded_audio','canonical_decode_clean','elapsed_seconds')}),flush=True);return result
rows=json.loads((REL/'selection.json').read_bytes());rows=[r for r in rows if (REL/'audio-diagnostics'/f"{r['record_id']}.json").exists()]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,rows))
