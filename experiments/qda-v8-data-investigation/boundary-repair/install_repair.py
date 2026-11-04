"""Install the tested future-row amendment only while the service is stopped."""
import hashlib,json,os,pathlib,shutil,subprocess,sys,time
import torch
HERE=pathlib.Path('/data/work/asr-experiments/qda-full-continuous-v8');STAGE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from contracts import atomic_json,sha
DATA=pathlib.Path('/data/work/asr-datasets/qda-full-continuous-v8');RUN=HERE/'runs/r64-full'
assert subprocess.run(['systemctl','--user','is-active','--quiet','qda-v8-training.service']).returncode!=0
assert not (HERE/'runtime-repair.json').exists()
amendment=DATA/'amendments/zero-duration-boundary-v1/amendment.json';value=json.loads(amendment.read_text())
verified=json.loads(amendment.with_name('verification.json').read_text());assert verified['status']=='passed' and verified['amendment_sha256']==sha(amendment)
latest=json.loads((RUN/'latest.json').read_text());cp=RUN/latest['checkpoint'];old=json.loads((RUN/'config.json').read_text())
assert latest['step']<value['update'] and old['steps']==3110 and old['warmup_steps']==311 and old['lr']==.0003
assert all(sha(pathlib.Path(p))==v for p,v in old['runtime_sha256'].items())
assert sha(HERE/'plan.json')==old['plan_sha256']==value['base_plan_sha256']
state=torch.load(cp/'state.pt',map_location='cpu',mmap=True,weights_only=False)
assert state['step']==latest['step'] and state['config']==old and state['scheduler']['last_epoch']==latest['step']
assert state['adapter_sha256']==sha(cp/'adapter_model.safetensors') and len(state['rng'])==2
assert state['optimizer']['state'] and all(int(v['step'])==latest['step'] for v in state['optimizer']['state'].values())
incident=HERE/'incidents/zero-duration-boundary-v1';before=incident/'before';before.mkdir(parents=True,exist_ok=False)
names=['train_restart.py','runtime_state.py','disk_retention.py','future_data_repair.py','test_future_data_repair.py']
backup=[n for n in names+['prepared.json','approval.json'] if (HERE/n).exists()]
for name in backup:shutil.copy2(HERE/name,before/name)
atomic_json(before/'run-config.json',old);atomic_json(before/'latest.json',latest)
atomic_json(RUN/'backup-in-progress/boundary-repair.json',{'checkpoint':cp.name,'expires_at':time.time()+86400,'reason':'Future data correction resume anchor'})
anchor={'checkpoint':cp.name,'step':latest['step'],'state_sha256':sha(cp/'state.pt'),'adapter_sha256':state['adapter_sha256'],
 'optimizer_parameters':len(state['optimizer']['state']),'optimizer_lr':state['optimizer']['param_groups'][0]['lr'],
 'scheduler':state['scheduler'],'tokens_seen':state['tokens_seen'],'rng_ranks':len(state['rng']),
 'old_config_sha256':sha(RUN/'config.json'),'approved_base_plan_sha256':sha(HERE/'plan.json')}
atomic_json(incident/'resume-anchor.json',anchor)
def replace_file(src,dest):
 temp=dest.with_suffix(dest.suffix+'.repair-building');shutil.copy2(src,temp)
 with temp.open('rb') as f:os.fsync(f.fileno())
 os.replace(temp,dest)
try:
 for name in names:replace_file(STAGE/'patch'/name,HERE/name)
 from runtime_state import runtime_hashes
 receipt={'version':1,'kind':'single-future-row-label-correction','authorization':'User: fix this before we train on it',
  'old_config':old,'new_runtime_sha256':runtime_hashes(),'amendment_path':str(amendment),'amendment_sha256':sha(amendment),
  'target_token_delta':value['target_token_delta'],'cache_contract_sha256':sha(HERE/'cache/contract.json'),
  'checkpoint_step':latest['step'],'checkpoint':cp.name,'resume_anchor_sha256':sha(incident/'resume-anchor.json'),
  'installed_at':time.time(),'no_optimizer_or_scheduler_reset':True}
 atomic_json(HERE/'runtime-repair.json',receipt)
 from future_data_repair import load_repair,apply_repair,compatible_resume,validate_resume_step
 repair=load_repair();base=json.loads((HERE/'plan.json').read_text());effective=apply_repair(base,repair)
 differences=[(i+1,j) for i,(a,b) in enumerate(zip(base['probe_pairs'],effective['probe_pairs'])) for j,(x,y) in enumerate(zip(a,b)) if x!=y]
 assert differences==[(value['update'],value['rank'])]
 assert all(sum(r['duration'] for r in a)==sum(r['duration'] for r in b) for a,b in zip(base['probe_pairs'],effective['probe_pairs']))
 assert max(r['total_tokens'] for p in base['probe_pairs'] for r in p)==max(r['total_tokens'] for p in effective['probe_pairs'] for r in p)
 current={**old,'runtime_sha256':receipt['new_runtime_sha256'],'code_sha256':sha(HERE/'train_restart.py'),
  'runtime_repair_sha256':sha(HERE/'runtime-repair.json'),'target_tokens':old['target_tokens']+value['target_token_delta']}
 assert compatible_resume(state['config'],current,repair);validate_resume_step(state['step'],state['config'],repair)
 from prepare import checked_input
 checked_input(value['replacement'],effective)
 ready=json.loads((HERE/'prepared.json').read_text());ready['runtime_sha256']=runtime_hashes()
 ready['future_data_repair']={'receipt_sha256':sha(HERE/'runtime-repair.json'),'amendment_sha256':sha(amendment),
  'verification_sha256':sha(amendment.with_name('verification.json')),'preflight_base_plan_unchanged':True,'max_sequence_length_unchanged':True,
  'optimizer_schedule_audio_crops_unchanged':True,'changed_future_update':value['update']}
 atomic_json(HERE/'prepared.json',ready)
 approval=json.loads((HERE/'approval.json').read_text());approval['files']['prepared.json']=sha(HERE/'prepared.json')
 approval['amendments']=[{'authorization':'User: fix this before we train on it','runtime_repair_sha256':sha(HERE/'runtime-repair.json'),
  'amendment_sha256':sha(amendment),'approved':True,'time':time.time()}];atomic_json(HERE/'approval.json',approval)
 from approval import require_approval
 require_approval()
 from gates import check_data
 check_data(DATA,reviewed=True)
 tests=subprocess.run([sys.executable,'-m','unittest','-v','test_future_data_repair','test_scheduler_resume','test_launch_and_retention'],cwd=HERE,capture_output=True,text=True)
 (incident/'tests.log').write_text(tests.stdout+tests.stderr);assert tests.returncode==0,(tests.stdout+tests.stderr)
 assert sha(cp/'state.pt')==anchor['state_sha256'] and sha(cp/'adapter_model.safetensors')==anchor['adapter_sha256']
 result={'status':'installed-and-verified','time':time.time(),'checkpoint':cp.name,'resume_step':latest['step'],
  'changed_rows':differences,'amendment_sha256':sha(amendment),'runtime_repair_sha256':sha(HERE/'runtime-repair.json'),
  'base_plan_unchanged':True,'all_audio_crops_unchanged':True,'prior_consumed_rows_unchanged':True,'optimizer_state_untouched':True,
  'scheduler_state_untouched':True,'lr_at_resume':anchor['optimizer_lr'],'optimizer_steps':old['steps'],'warmup_steps':old['warmup_steps'],
  'tests_exit_code':tests.returncode,'restored_words':5,'target_token_delta':33}
 atomic_json(incident/'deployment.json',result)
 atomic_json(DATA/'amendments/index.json',{'base_selection_sha256':sha(DATA/'screening-selection.json'),
  'entries':[{'path':str(amendment.relative_to(DATA)),'sha256':sha(amendment),'active_run':'qda-full-continuous-v8'}]})
 print(json.dumps(result),flush=True)
except BaseException:
 for name in names+['prepared.json','approval.json']:
  if (before/name).exists():replace_file(before/name,HERE/name)
  elif name in names:(HERE/name).unlink(missing_ok=True)
 if (HERE/'runtime-repair.json').exists():shutil.move(HERE/'runtime-repair.json',incident/'failed-runtime-repair.json')
 raise
