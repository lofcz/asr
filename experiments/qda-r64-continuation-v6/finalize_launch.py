"""Record the measured LR decision, verify the real scheduler, pin the launch."""
import argparse,json,pathlib,subprocess,time
from contracts import sha,atomic_json
from gates import demand,check_data
from prepare import DATA
from runtime_state import runtime_hashes
from schedule import write_preview,require_decision
HERE=pathlib.Path(__file__).resolve().parent


def main(lr,rationale):
    history=HERE/'result/history.json'
    measured=json.loads(history.read_text())
    demand(not measured['running'] and measured.get('calibration_discarded') and measured.get('restoration_verified'),'LR calibration has not restored the parent')
    plan=json.loads((HERE/'plan.json').read_text())
    admission=check_data(DATA)
    demand(admission['remaining_hours']==plan['train_hours'],'Training plan coverage changed')
    demand(plan['dataset_admission_sha256']==sha(DATA/'admission.json'),'Training admission changed')
    decision=json.loads((HERE/'result/analysis.json').read_text())
    restored=json.loads((HERE/'result/parent-restore.json').read_text())
    provenance=json.loads((DATA/'provenance.json').read_text())
    demand(restored['adapter_sha256']==provenance['parent_checkpoint_files']['adapter_model.safetensors'] and restored['optimizer_restored'],'Wrong parent calibration')
    old=json.loads((HERE.parent/'qda-full-r64-v4/run-spec.json').read_text())
    spec={'version':6,'rank':64,'alpha':32,'learning_rate':lr,
        'scheduler':old['scheduler'],'schedule_evidence':old['schedule_evidence'],
        'initialization':provenance['parent_checkpoint'],'parent_audio_hours':restored['parent_audio_hours'],
        'parent_adapter_sha256':restored['adapter_sha256'],'optimizer':'adamw','weight_decay':0.,
        'optimizer_state':'Restored parent Adam moments; continuation LR schedule restarted',
        'world_size':2,'mode':'nf4-qlora-bf16-compute','repeat_windows':False,
        'training_audio_hours':plan['train_hours'],'evaluation_every_audio_hours':50,'fixed_markers':['cs','sk'],
        'long_session_evaluation':'Reserved 30-minute CS and 55-minute SK development sessions at completion',
        'lr_evidence':{'history_path':str(history),'history_sha256':sha(history),
            'analysis_path':str(HERE/'result/analysis.json'),'analysis_sha256':sha(HERE/'result/analysis.json'),
            'upstream_steepest_descent_lr':decision['upstream_steepest_descent_lr'],
            'best_raw_content_lr':decision['best_raw_content_lr'],'rationale':rationale},
        'status':'Authorized continuation; measured LR recorded; actual activity is in experiment-status.json'}
    dest=HERE/'run-spec.json'
    if dest.exists():demand(json.loads(dest.read_text())==spec,'Run specification already pinned')
    else:atomic_json(dest,spec)
    require_decision(spec);write_preview(HERE)
    cmd=['/data/work/asr-pilot/venv/bin/python','-m','unittest','test_continuation','test_disk_retention','test_scheduler_resume','-v']
    with (HERE/'launch-tests.log').open('w') as log:subprocess.run(cmd,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic_json(HERE/'launch-receipt.json',{'runtime_sha256':runtime_hashes(),
        'plan_sha256':sha(HERE/'plan.json'),'run_spec_sha256':sha(dest),
        'admission_sha256':sha(DATA/'admission.json'),'lr_history_sha256':sha(history),
        'test_command':cmd,'tests_sha256':sha(HERE/'launch-tests.log'),'passed_at':time.time(),
        'parent_restored_and_verified':restored,'interval_partition':provenance['partition'],
        'optimizer_steps':len(plan['probe_pairs']),'new_audio_hours':plan['train_hours']})
    print(json.dumps({'status':'ready','steps':len(plan['probe_pairs']),'audio_hours':plan['train_hours'],
                      'peak_lr':lr,'scheduler':spec['scheduler']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lr',type=float,required=True);parser.add_argument('--rationale',required=True)
    args=parser.parse_args();main(args.lr,args.rationale)
