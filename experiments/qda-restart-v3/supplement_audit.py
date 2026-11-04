"""Finish missing long-session and development acoustic checks, then the ASR queue.

This service ONLY curates data. It cannot start model training.
"""
import json
import pathlib
import subprocess
import time
from contracts import atomic_json,sha
from launch import parallel,run,PY,HERE
from prepare import DATA
from gates import lease_data

UNITS=('qda-v3-acoustic-original-0.service','qda-v3-acoustic-original-1.service',
       'qda-v3-acoustic-0.service','qda-v3-acoustic-1b.service','qda-v3-secondary-0.service','qda-v3-secondary-1.service')


def status(stage,**kwargs):atomic_json(HERE/'curation-status.json',{'stage':stage,'updated_at':time.time(),'training_started':False,**kwargs})


def main():
    status('waiting_for_current_audits')
    while any(subprocess.run(['systemctl','--user','is-active',u],capture_output=True,text=True).stdout.strip() in ('active','activating') for u in UNITS):time.sleep(15)
    for stage,option in [('long_context_audio','--long-only'),('heldout_audio','--development')]:
        status(stage);parallel([[PY,'acoustic_audit.py','--data',str(DATA),'--rank',str(i),option] for i in (0,1)],stage)
    for attempt in range(4):
        status('selecting_screened_windows',attempt=attempt)
        run([PY,'select_windows.py','candidates','--data',str(DATA)],'selection')
        run([PY,'select_windows.py','select','--data',str(DATA)],'selection')
        selection=json.loads((DATA/'screening-selection.json').read_text());rows=selection['train']+selection['development']
        if not json.loads((HERE/'policy.json').read_text()).get('secondary_asr_required',False):break
        def pending_row(r):
            path=DATA/'secondary'/(r['window_id']+'.json')
            return not path.exists() or json.loads(path.read_text()).get('code_sha256')!=sha(HERE/'secondary_audit.py')
        pending=[r for r in rows if pending_row(r)]
        if not pending:break
        status('secondary_asr',attempt=attempt,selected_train_hours=selection['train_hours'],pending_windows=len(pending))
        parallel([[str(HERE/'audit-venv/bin/python'),'secondary_audit.py','--data',str(DATA),'--rank',str(i)] for i in (0,1)],f'secondary-round-{attempt}')
    run([PY,'select_windows.py','select','--data',str(DATA)],'selection-final')
    result=subprocess.run([PY,'gates.py','--data',str(DATA)],cwd=HERE,capture_output=True,text=True)
    selection=json.loads((DATA/'screening-selection.json').read_text())
    status('ready_for_preparation' if result.returncode==0 else 'data_gate_blocked',machine_gate_passed=result.returncode==0,
           machine_gate_output=result.stdout if result.returncode==0 else result.stderr[-4000:],selected_train_hours=selection['train_hours'],
           length_hours=selection['length_hours'],development_minutes=selection['development_minutes'])
    if result.returncode:print(result.stderr,flush=True)


if __name__=='__main__':
    try:
        with lease_data(DATA,exclusive=True,wait=True):main()
    except Exception as error:status('audit_failed',error=str(error));raise
