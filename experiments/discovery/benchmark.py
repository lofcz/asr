import json,pathlib,time,urllib.request,subprocess
p=pathlib.Path('experiments/discovery');rows=[json.loads(s) for s in (p/'episodes.jsonl').read_text().splitlines()]
results=[]
for name in ['Na Výbornou','Knižní klub','Automatky']:
 row=next(r for r in rows if r['show']==name and (r['duration_seconds'] or 0)>900)
 target=p/f'sample-{len(results)}.mp3';start=time.monotonic()
 try:
  with urllib.request.urlopen(row['audio_url'],timeout=30) as r, target.open('wb') as f:
   while b:=r.read(1024*1024):
    f.write(b)
    if f.tell()>150*1024*1024 or time.monotonic()-start>90: raise RuntimeError('Sample limit exceeded')
  download=time.monotonic()-start
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-of','json',str(target)]))
  duration=float(probe['format']['duration']);start=time.monotonic()
  subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(target),'-ac','1','-ar','16000','-c:a','pcm_s16le',str(target.with_suffix('.wav'))],check=True)
  result=dict(show=name,title=row['title'],audio_seconds=duration,bytes=target.stat().st_size,download_seconds=download,decode_seconds=time.monotonic()-start)
 except Exception as e:result=dict(show=name,error=str(e))
 results.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
(p/'benchmark.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
