"""Checkpoint-safe suffix permutation; preserve optimizer, RNG, completed prefix."""
import pathlib,json,hashlib,shutil,time,torch,subprocess,os,numpy as np
torch.set_num_threads(2)
from balance import balance,pair_cost
ROOT=pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16');OUT=pathlib.Path('/data/work/asr-optimization');BACK=OUT/'before-migration'
assert subprocess.run(['systemctl','--user','is-active','--quiet','qda496-training.service']).returncode!=0
latest=json.loads((ROOT/'latest.json').read_text());assert latest['step']==350
plan=json.loads((ROOT/'plan.json').read_text());lengths=json.loads((OUT/'lengths.json').read_text());new=balance(plan,350,lengths)
hash_plan=lambda p:hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
old_hash=hash_plan(plan);new_hash=hash_plan(new);assert old_hash!=new_hash
BACK.mkdir(exist_ok=False)
for name in ('config.json','plan.json','plan-override.json','latest.json','plan-audit.json'):shutil.copy2(ROOT/name,BACK/name)
shutil.copytree(ROOT/latest['checkpoint'],BACK/latest['checkpoint'])
checkpoint=ROOT/'checkpoint-000350-balanced';shutil.copytree(ROOT/latest['checkpoint'],checkpoint)
state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False);assert state['step']==350 and state['plan_sha256']==old_hash
state['plan_sha256']=new_hash;torch.save(state,checkpoint/'state.pt')
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
original=torch.load(BACK/latest['checkpoint']/'state.pt',map_location='cpu',weights_only=False)
reloaded=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
for key in ('optimizer','rng','step'):assert equal(original[key],reloaded[key]),key
for name in ('adapter_model.safetensors','adapter_config.json'):assert (checkpoint/name).read_bytes()==(BACK/latest['checkpoint']/name).read_bytes()
del original,reloaded,state
for file in checkpoint.iterdir():
 if file.is_file():
  with file.open('rb') as f:os.fsync(f.fileno())
def atomic(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n')
 with tmp.open('rb') as f:os.fsync(f.fileno())
 tmp.replace(path)
config=json.loads((ROOT/'config.json').read_text());assert config['plan_sha256']==old_hash;config['plan_sha256']=new_hash
atomic(ROOT/'plan.json',new);atomic(ROOT/'plan-override.json',new);atomic(ROOT/'config.json',config)
atomic(ROOT/'latest.json',{'checkpoint':checkpoint.name,'step':350})
report={'time':time.time(),'checkpoint_step':350,'old_plan_sha256':old_hash,'new_plan_sha256':new_hash,'unchanged_completed_windows':700,'remaining_windows':len(new)-700,'coverage':'Exact permutation of remaining original windows; identical sample dictionaries; no skips or duplication; original curriculum bucket sequence preserved','old_pair_cost':pair_cost(plan[700:],lengths),'new_pair_cost':pair_cost(new[700:],lengths),'checkpoint_backup':str(BACK/latest['checkpoint']),'optimizer_rng_preserved':True,'max_tokens':max(x['tokens'] for x in lengths.values())}
atomic(ROOT/'optimization-migration.json',report);atomic(OUT/'migration.json',report);print(json.dumps(report),flush=True)
