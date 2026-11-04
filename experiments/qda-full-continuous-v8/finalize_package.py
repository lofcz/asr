"""Freeze the concrete review package after mechanical checks; never train."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
from contracts import atomic_json,sha
from gates import check_data,demand
from prepare import DATA,require_old_jobs_stopped
from runtime_state import runtime_hashes

HERE=Path(__file__).resolve().parent


def verify_audio():
    selection=json.loads((DATA/'screening-selection.json').read_text())
    files={r['audio_file']:r['audio_sha256'] for r in selection['train']+selection['development']}
    def verify(item):
        path,digest=item;actual=sha(path)
        demand(actual==digest,'Training audio hash mismatch: '+path)
        return {'path':path,'sha256':actual,'bytes':Path(path).stat().st_size}
    with ThreadPoolExecutor(max_workers=4) as workers:rows=list(workers.map(verify,files.items()))
    atomic_json(HERE/'audio-integrity.json',{'passed':True,'verified_at':time.time(),
        'unique_audio_files':len(rows),'files':rows,'selection_sha256':sha(DATA/'screening-selection.json')})


def finalize():
    require_old_jobs_stopped();admission=check_data(DATA)
    plan=json.loads((HERE/'plan.json').read_text());spec=json.loads((HERE/'run-spec.json').read_text())
    preflight=json.loads((HERE/'preflight-r64.json').read_text());audio=json.loads((HERE/'audio-integrity.json').read_text())
    demand(preflight['status']=='passed' and preflight['optimizer_updates']==0 and preflight['adapter_unchanged'],'Two-GPU loader check failed')
    demand(preflight['plan_sha256']==sha(HERE/'plan.json') and preflight['admission_sha256']==sha(DATA/'admission.json'),'Loader check used different data')
    demand(audio['passed'] and audio['selection_sha256']==sha(DATA/'screening-selection.json'),'Audio integrity report used different data')
    demand(spec['initialization']=='vanilla' and spec['fresh_optimizer'] is True,'Fresh initialization not selected')
    demand(not (HERE/'runs/r64-full/latest.json').exists(),'Training was started before final approval')
    verification=json.loads((HERE/'verification.json').read_text())
    demand(verification['passed'] and verification['runtime_sha256']==runtime_hashes()
           and verification['plan_sha256']==sha(HERE/'plan.json'),'Package verification is stale')
    for row in plan['validation']:
        baseline=json.loads((HERE/'evaluations/vanilla'/row['window_id']/'result.json').read_text())
        demand(baseline['complete'] and baseline['adapter_sha256']=='vanilla'
               and baseline['input_sha256']==row['input_sha256'],'Fixed vanilla baseline is not ready')
    markers=json.loads((HERE/'markers/vanilla/completed-v3.json').read_text())
    from reuse_markers import contract
    demand(markers['contract']==contract() and set(markers['languages'])=={'cs','sk'},'Fixed marker baseline changed')
    long_panel=json.loads((HERE/'long-evaluation-selection.json').read_text())
    demand(long_panel['plan_sha256']==sha(HERE/'plan.json'),'Long evaluation plan is stale')
    preview=json.loads((HERE/'schedule-preview.json').read_text())
    demand(preview['terminal_lr']==0 and preview['optimizer_steps']==len(plan['probe_pairs']),'LR schedule is not one complete pass')
    demand(preview['name']=='linear' and preview['peak_lr']==.0003 and preview['warmup_ratio']==.1,'LR differs from agreed recipe')
    result={'status':'ready_for_user_approval','training_started':False,'training_autostart':False,
        'created_at':time.time(),'runtime_sha256':runtime_hashes(),'plan_sha256':sha(HERE/'plan.json'),
        'admission_sha256':sha(DATA/'admission.json'),'run_spec_sha256':sha(HERE/'run-spec.json'),
        'audio_integrity_sha256':sha(HERE/'audio-integrity.json'),'preflight_sha256':sha(HERE/'preflight-r64.json'),
        'verification_sha256':sha(HERE/'verification.json'),'tests_passed':verification['tests_run'],
        'long_panel_sha256':sha(HERE/'long-evaluation-selection.json'),
        'train_hours':plan['train_hours'],'real_hours':admission['real_train_hours'],'synthetic_hours':admission['synthetic_train_hours'],
        'optimizer_steps':len(plan['probe_pairs']),'warmup_steps':preview['warmup_steps'],
        'peak_lr':preview['peak_lr'],'terminal_lr':preview['terminal_lr'],'world_size':2,'rank':64,'alpha':32,
        'evaluations_every_audio_hours':50,'references':'Existing held-out interviews and fixed Czech/Slovak markers',
        'dataset_checks':admission,'approval_required_by':'Explicit user request for final approval before launch'}
    from collections import Counter
    selection=json.loads((DATA/'screening-selection.json').read_text())
    language=Counter();length=Counter();speakers=Counter()
    for row in selection['train']:
        language[row['language']]+=row['duration']/3600
        length['long' if row['duration']>=1800 else 'medium' if row['duration']>=600 else 'short']+=row['duration']/3600
        speakers[str(row['speaker_count'])]+=row['duration']/3600
    atomic_json(DATA/'selection-summary.json',{
        'train_hours':plan['train_hours'],'real_train_hours':admission['real_train_hours'],
        'synthetic_train_hours':admission['synthetic_train_hours'],'windows':len(selection['train']),
        'recordings':len({r['audio_sha256'] for r in selection['train']}),
        'length_hours':dict(length),'language_hours':dict(language),'speaker_hours':dict(speakers),
        'repeat_windows':False,'speaker_count_limits':None,'quality_filters':False})
    synthetic=Path('/data/work/asr-synthetic-v7')
    report=json.loads((synthetic/'domain-coverage.json').read_text())
    result['synthetic_recordings']=report['completed_sessions']
    result['synthetic_voices']=report['actual_voices_used']
    result['synthetic_vocabulary_counts']={name:{lang:report['scio'][name][lang]['occurrences'] for lang in ('cs','sk')}
        for name in ('ScioBot','ScioŠkola','ScioLink')}
    recovery=json.loads((HERE.parent/'qda-data-recovery-v5/result.json').read_text())
    result.update(heldout_hours=recovery['heldout_hours'],
                  existing_unresolved_real_hours=recovery['unresolved_source_hours'],
                  existing_unresolved_real_sources=recovery['unresolved_sources'])
    atomic_json(HERE/'prepared.json',result)
    atomic_json(HERE/'preparation-status.json',{'stage':'data-and-runtime-prepared','training_started':False,'updated_at':time.time()})
    atomic_json(HERE/'experiment-status.json',{'stage':'awaiting-final-approval','running':False,'updated_at':time.time()})
    print(json.dumps({k:v for k,v in result.items() if k not in ('runtime_sha256','dataset_checks')},indent=2))


if __name__=='__main__':
    import sys
    verify_audio() if '--audio-only' in sys.argv else finalize()
