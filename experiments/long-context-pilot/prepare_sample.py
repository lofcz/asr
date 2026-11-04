import concurrent.futures, json, os, pathlib, subprocess, time
import requests
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent / 'sample'
def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)
def select():
    manifest=OUT/'sources.json'
    if not manifest.exists():
        raise FileNotFoundError('Missing sample/sources.json: select explicit source recordings before submitting paid Soniox jobs.')
    sources=json.loads(manifest.read_text())
    assert sources and all(s['duration'] > 0 for s in sources)
    return sources

def main():
    sources=select()
    # Read only the requested credential, never export or print the .env contents.
    env=ROOT.parent/'tazatelka/.env';key=None
    for line in env.read_text().splitlines():
        name,sep,value=line.partition('=')
        if name.strip()=='SONIOX_API_KEY':key=value.strip().strip('"').strip("'")
    if not key:raise RuntimeError('SONIOX_API_KEY missing')
    def process(s):
        folder=OUT/s['id'];folder.mkdir(exist_ok=True);clip=folder/'audio.flac';statepath=folder/'api-state.json';result=folder/'soniox.json'
        if not clip.exists():subprocess.run(['ffmpeg','-v','error','-ss',str(s['source_start']),'-i',str(ROOT/s['source_path']),'-t',str(s['duration']),'-ar','24000','-ac','1','-sample_fmt','s16','-c:a','flac','-frame_size','2304',str(clip)],check=True)
        session=requests.Session();session.headers['Authorization']='Bearer '+key
        def api(method,path,**kw):
            r=session.request(method,'https://api.soniox.com/v1'+path,timeout=120,**kw)
            if not r.ok:raise RuntimeError(f'Soniox {method} {path}: HTTP {r.status_code}: {r.text[:500]}')
            return r.json() if r.content else {}
        state=json.loads(statepath.read_text()) if statepath.exists() else {}
        if not result.exists():
            if 'file_id' not in state:
                with clip.open('rb') as f:state['file_id']=api('POST','/files',files={'file':('audio.flac',f,'audio/flac')})['id']
                save(statepath,state)
            if 'transcription_id' not in state:
                config={'model':'stt-async-v5','file_id':state['file_id'],'language_hints':[s['language']],'enable_speaker_diarization':True,'enable_language_identification':True,'client_reference_id':'asr-long-context-'+s['id']}
                state['request']=config;save(statepath,state)
                state['transcription_id']=api('POST','/transcriptions',json=config)['id'];save(statepath,state)
            deadline=time.monotonic()+1800
            while time.monotonic()<deadline:
                status=api('GET','/transcriptions/'+state['transcription_id'])
                if status['status']=='error':raise RuntimeError(str(status))
                if status['status']=='completed':break
                time.sleep(3)
            else:raise TimeoutError(s['id'])
            save(result,api('GET','/transcriptions/'+state['transcription_id']+'/transcript'))
            state['completed']=status;save(statepath,state)
        if not state.get('remote_deleted'):
            for endpoint,k in [('/transcriptions/','transcription_id'),('/files/','file_id')]:
                if k in state:
                    r=session.delete('https://api.soniox.com/v1'+endpoint+state[k],timeout=60)
                    if r.status_code not in (200,204,404):raise RuntimeError(f'Cleanup failed {r.status_code}')
            state['remote_deleted']=True;save(statepath,state)
        data=json.loads(result.read_text());tokens=data.get('tokens',[])
        print(json.dumps({'id':s['id'],'show':s['show'],'split':s['split'],'tokens':len(tokens),'speakers':sorted({str(t.get('speaker')) for t in tokens})}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(process,sources))
if __name__=='__main__':main()
