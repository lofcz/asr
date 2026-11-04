import concurrent.futures, json, os, pathlib, subprocess, time
import requests
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent / 'sample'
SHOWS = ['01736241338f4bc7','19cab80304f19311','381ae460b47ad272','4ba39fe1191b0566','57b06e02c5576b98','8bc7cdd6c4f37ec3','eb926a08347cc87c','40a489ad66fcaaa8','118970bc1b95ae3e','cb08dc6312d621bc','ce18a9524e04f1f2','73f41dcd5910ef50']
HELDOUT = {SHOWS[7], SHOWS[11]}
def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)
def select():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'sources.json'
    if manifest.exists():return json.loads(manifest.read_text())
    sources=[]
    for show in SHOWS:
        candidates=[]
        for path in sorted((ROOT/'data/audio').glob(f'*/{show}/*.json')):
            d=json.loads(path.read_text())
            if d.get('actual_duration_seconds',0)>600 and (ROOT/d['path']).is_file():candidates.append(d)
        if not candidates:raise RuntimeError(f'No audio for {show}')
        d=candidates[0]; start=round(min(d['actual_duration_seconds']*.3,d['actual_duration_seconds']-200),3)
        s={'id':d['id'],'show_id':show,'show':d['show'],'title':d['title'],'language':d['language'],'source_path':d['path'],'source_sha256':d['sha256'],'source_start':start,'duration':180,'split':'validation' if show in HELDOUT else 'train'}
        sources.append(s)
    save(manifest,sources);return sources

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
        if not clip.exists():subprocess.run(['ffmpeg','-v','error','-ss',str(s['source_start']),'-i',str(ROOT/s['source_path']),'-t','180','-ar','24000','-ac','1','-c:a','flac',str(clip)],check=True)
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
                config={'model':'stt-async-v5','file_id':state['file_id'],'language_hints':[s['language']],'enable_speaker_diarization':True,'enable_language_identification':True,'client_reference_id':'asr-pilot-'+s['id']}
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
