"""One bounded repair of a verified connection-reset session; retain all history."""
import signal,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from core import DATA,REPLICA,read,save,encode,digest
from cartesia import request_for,VERSION
from pipeline import run_session,progress,STOP

SID='v3-single_speaker_extra-sk-s1-033'


def main():
    rel=Path('sessions')/SID
    if read(rel/'receipt.json'):return
    state=read(rel/'state.json',{})
    if state.get('status')!='error' or 'Connection reset by peer' not in state.get('error',''):
        raise RuntimeError('Session is not the verified connection-reset failure')
    spec=read(rel/'spec.json');voices=read(rel/'voices.json')
    script=read('scripts/'+SID+'/part-000.json');repairs=[]
    for turn in script['turns']:
        voice=voices[turn['speaker']]
        request=request_for(turn['text'],voice['language'][0],voice['voice_id'],voice['speed'])
        ident=digest(encode({'api_version':VERSION,'body':request}));root=Path('utterances')/ident
        if read(root/'receipt.json'):continue
        intent=read(root/'intent.json')
        if not intent:continue
        if intent.get('status')!='submitted':raise RuntimeError('Unexpected submission state')
        for base in (DATA,REPLICA):
            if any((base/root/name).exists() for name in ['response.json','events.jsonl','audio.wav']):
                raise RuntimeError('A response exists; use response recovery instead')
        repairs.append((root,request,intent))
    if len(repairs)!=1 or len(repairs[0][1]['transcript'])!=476:
        raise RuntimeError('Recovery scope differs from inspected one-turn failure')
    root,request,intent=repairs[0]
    save(root/'network-recovery-001.json',{'source_session':SID,'request_sha256':digest(encode(request)),
        'max_retries':1,'max_additional_characters':476,'original_intent':intent,'at':time.time(),
        'reason':'Connection reset before any HTTP response or SSE event was saved on either disk; one bounded repair within the existing generation budget.'})
    save(rel/'recovery-history.json',{'original_failure':state,'retry_characters_reserved_separately':476,'at':time.time()})
    save(rel/'state.json',{'status':'running','recovery':'bounded_network_repair','updated_at':time.time()})
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:STOP.set())
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:run_session(spec,pool)
    except Exception as e:
        save(rel/'state.json',{'status':'error','error':str(e),'recovery_attempted':True,'updated_at':time.time()})
        progress();raise


if __name__=='__main__':main()
