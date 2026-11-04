"""Full-run utilities; use run_full.py for the gated training entry point."""
import os
import argparse
import fcntl
import json
import pathlib
import shutil
import subprocess
import sys
import time
import contextlib
from contracts import atomic_json,sha
from gates import check_data,demand,lease_data,validate_reference
from prepare import DATA,require_old_jobs_stopped
from runtime_state import runtime_hashes
from analyze import compare

HERE=pathlib.Path(__file__).resolve().parent
PY='/data/work/asr-pilot/venv/bin/python'
MONITOR='/data/work/asr-training/monitor-venv/bin/python'


class TrialRejected(RuntimeError):
    """A measured trajectory rejects one capacity without cancelling its peer."""


def progress(stage,**details):
    atomic_json(HERE/'experiment-status.json',{'stage':stage,'updated_at':time.time(),'running':True,**details})


def run(command,name):
    progress(name)
    (HERE/'logs').mkdir(exist_ok=True)
    with (HERE/'logs'/(name+'.log')).open('a') as f:subprocess.run(command,cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)


def parallel(commands,name):
    progress(name)
    files=[];children=[];(HERE/'logs').mkdir(exist_ok=True)
    try:
        for i,cmd in enumerate(commands):
            f=(HERE/'logs'/f'{name}-{i}.log').open('a');files.append(f);children.append(subprocess.Popen(cmd,cwd=HERE,stdout=f,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in children]
        demand(all(c==0 for c in codes),'Worker failed: '+name)
    finally:
        for p in children:
            if p.poll() is None:p.terminate();p.wait()
        for f in files:f.close()


def evaluate(name,adapter=None,small=False,stress=False):
    out=HERE/'evaluations'/name;out.mkdir(parents=True,exist_ok=True)
    flags=(['--adapter',str(adapter)] if adapter else [])+(['--small'] if small else [])+(['--stress'] if stress else [])
    parallel([[PY,'evaluate_panel.py','--out',str(out),'--rank',str(i),*flags] for i in (0,1)],'evaluation-'+name)
    plan=json.loads((HERE/'plan.json').read_text());panel=plan['long_stress'] if stress else plan['validation_small'] if small else plan['validation']
    results=[json.loads((out/r['window_id']/'result.json').read_text()) for r in panel]
    if adapter and not all(r.get('complete') for r in results):raise TrialRejected('Adapted evaluation has invalid/incomplete raw output: '+name)
    demand(all(r.get('complete') for r in results),'Evaluation invalid/incomplete; inspect raw output: '+name)
    return results


def markers(name,adapter=None):
    """Both original approved CS/SK fixtures remain visible, separately, at every trajectory checkpoint."""
    out=HERE/'markers'/name;out.mkdir(parents=True,exist_ok=True);variant='finetuned' if adapter else 'vanilla'
    if adapter:
        dest=out/'adapter';dest.mkdir(exist_ok=True)
        for filename in ('adapter_model.safetensors','adapter_config.json'):
            if (dest/filename).exists():demand(sha(dest/filename)==sha(adapter/filename),'Marker checkpoint changed')
            else:os.link(adapter/filename,dest/filename)
    stamp=out/'completed-v3.json';identity=sha(adapter/'adapter_model.safetensors') if adapter else 'vanilla'
    from reuse_markers import contract as marker_contract
    if not adapter and not stamp.exists():run([PY,'reuse_markers.py'],'reuse-vanilla-markers')
    if stamp.exists():
        saved=json.loads(stamp.read_text());demand(saved['adapter_sha256']==identity and saved['contract']==marker_contract(),'Marker identity/contract changed');return saved
    step=int(adapter.name.split('-')[-1]) if adapter else 0;hours=0.
    if adapter:
        events=[json.loads(l) for l in (adapter.parent/'events.jsonl').read_text().splitlines()]
        hours=next(e['audio_hours_seen'] for e in events if e['event']=='step' and e['step']==step)
    parallel([['env',f'CUDA_VISIBLE_DEVICES={i}',PY,'/data/work/asr-evaluation/periodic/code/evaluate.py','--variant',variant,'--out',str(out),'--language',lang,'--step',str(step),'--hours',str(hours)] for i,lang in enumerate(('cs','sk'))],'markers-'+name)
    rows={l:json.loads((out/variant/l/'result.json').read_text()) for l in ('cs','sk')}
    demand(all(r.get('termination')=='eos' and 'der_250ms' in r.get('metrics',{}) for r in rows.values()),'CS/SK marker evaluation failed')
    for lang,row in rows.items():
        raw=json.loads((out/variant/lang/'output.txt').read_text())
        demand(raw==json.loads((out/variant/lang/'hypothesis.json').read_text()),'Marker raw JSON differs from scored output')
        validate_reference(raw,row['duration_seconds'])
    provenance={l:{k:r[k] for k in ('audio_sha256','reference_sha256','feature_sha256')} for l,r in rows.items()}
    baseline=json.loads((HERE/'markers/vanilla/completed-v3.json').read_text())
    demand(provenance==baseline['provenance'],'Marker input/features differ from vanilla')
    result={'adapter_sha256':identity,'languages':{l:r['metrics'] for l,r in rows.items()},'provenance':provenance,'contract':marker_contract(),
            'step':step,'audio_hours_seen':hours,'parent_audio_hours':0.,'cumulative_audio_hours_seen':hours,'role':'Required fixed markers; user-approved generated dialogues; report separately'}
    atomic_json(stamp,result);return result

