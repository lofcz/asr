import pathlib,json,sys,concurrent.futures
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from soniox_durable import Store,file_hash
from soniox_vad_audit import audit
root=pathlib.Path(__file__).resolve().parents[1];release=root/'datasets/releases/qda-496h-v1';backup='/home/lofcz/.local/state/asr-soniox/releases/qda-496h-v1';store=Store(root,release,backup,'33ba0a9e-6e77-441d-96fc-82f4270b7140');rows=json.loads((release/'selection.json').read_bytes());model=root/'datasets/soniox-500h-v1/references/silero_vad.onnx';model_hash=file_hash(model)
def run(r):
 p=release/'canonical'/f"{r['record_id']}.json"
 if not p.exists():return
 c=json.loads(p.read_bytes());adapted={**r,'path':str((release/c['canonical_path']).relative_to(root)),'sha256':c['canonical_sha256'],'seconds':c['decoded_seconds']}
 audit(store,adapted,model,model_hash);print(r['record_id'],'canonical VAD verified',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,rows))
