"""Fill measured duration deficits with new sessions after the base queue finishes."""
from collections import defaultdict
import math
import subprocess
import time
from core import read, save
from plan import plan
from vocabulary import extend


def cycle():
    p=plan();sessions=p['sessions'];complete={}
    for s in sessions:
        r=read('sessions/'+s['id']+'/receipt.json')
        if r:complete[s['id']]=r
    if len(complete)!=len(sessions):return
    targets={(cohort,lang,n):(22500 if cohort=='core' else 18000)
        for cohort in ['core','single_speaker_extra'] for lang in ['cs','sk']
        for n in (range(1,5) if cohort=='core' else [1])}
    actual=defaultdict(float)
    for s in sessions:actual[(s['cohort'],s['language'][0],s['speaker_count'])]+=complete[s['id']]['duration_seconds']
    deficits={k:max(0,targets[k]-actual[k]) for k in targets}
    rows=[{'cohort':k[0],'language':[k[1]],'speaker_count':k[2],
           'target_hours':targets[k]/3600,'actual_hours':actual[k]/3600,'deficit_seconds':deficits[k]} for k in targets]
    if all(v<=30 for v in deficits.values()):
        coverage=read('domain-coverage.json',{});export=read('training/status.json',{})
        vocabulary_ready=coverage.get('vocabulary_targets_fulfilled',False)
        save('quota-status.json',{'status':'fulfilled' if vocabulary_ready and coverage.get('completed_sessions',0)==len(sessions) else 'duration_complete_awaiting_export_and_vocabulary_audit','rows':rows,'actual_hours':sum(actual.values())/3600,
                                 'vocabulary_targets_fulfilled':vocabulary_ready,
                                 'tolerance_seconds_per_stratum':30,'updated_at':time.time()})
        return
    active=subprocess.run(['systemctl','--user','is-active','asr-cartesia-v7.service'],capture_output=True,text=True).stdout.strip()
    if active in ('active','activating','deactivating'):return
    if read('control.json',{}).get('status')=='paused':return
    extensions=read('quota-extensions.json',[]);added=[]
    for key,remaining in deficits.items():
        if remaining<=30:continue
        template=next(s for s in sessions if (s['cohort'],s['language'][0],s['speaker_count'])==key and s['planned_seconds']>=600)
        count=math.ceil(remaining/1800)
        for _ in range(count):
            number=len(extensions)+len(added)+1
            seconds=max(30,min(1800,round(remaining/count)))
            added.append({**template,'id':f'quota-{key[0]}-{key[1]}-s{key[2]}-{number:03d}',
                'planned_seconds':seconds,'seed':20261024+number*7919,
                'quota_extension':True,'reason':'Actual generated duration below the original agreed stratum target'})
    for session in added:extend(session)
    save('quota-extensions.json',extensions+added)
    save('quota-status.json',{'status':'filling_measured_deficits','rows':rows,'added_sessions':len(added),'updated_at':time.time()})
    subprocess.run(['systemctl','--user','start','asr-cartesia-v7.service'],check=True)


if __name__=='__main__':
    while True:
        try:cycle()
        except Exception as e:print(type(e).__name__,str(e),flush=True)
        time.sleep(30)
