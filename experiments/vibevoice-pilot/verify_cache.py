import json,pathlib,numpy as np,torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
BASE=pathlib.Path(__file__).resolve().parent
rows=[json.loads(l) for l in (BASE/'cache/index.jsonl').read_text().splitlines()]
a={k:np.load(BASE/'cache'/f'{k}.npy',mmap_mode='r') for k in ('tokens','labels','mask','features')}
for r in rows:
 o,n=r['token_offset'],r['token_length'];f,m=r['feature_offset'],r['feature_length'];p=r['prompt_length']
 assert a['mask'][o:o+n].sum()==m
 assert (a['labels'][o:o+p]==-100).all()
 assert np.array_equal(a['labels'][o+p:o+n],a['tokens'][o+p:o+n])
 assert np.isfinite(a['features'][f:f+m]).all()
train={r['show_id'] for r in rows if r['split']=='train'};val={r['show_id'] for r in rows if r['split']=='validation'};assert train.isdisjoint(val)
torch.manual_seed(42);h=torch.randn(2,19,16,requires_grad=True);w=torch.randn(31,16,requires_grad=True);labels=torch.randint(0,31,(2,19));labels[:,:8]=-100
reference=F.cross_entropy(F.linear(h,w)[:,:-1].reshape(-1,31),labels[:,1:].reshape(-1));gref=torch.autograd.grad(reference,(h,w))
y=labels[:,1:].reshape(-1);x=h[:,:-1].reshape(-1,16);valid=y!=-100;x=x[valid];y=y[valid]
def part(xx,yy):return F.cross_entropy(F.linear(xx,w),yy,reduction='sum')
chunked=sum(checkpoint(part,x[i:i+3],y[i:i+3],use_reentrant=False) for i in range(0,len(y),3))/len(y)
gchunk=torch.autograd.grad(chunked,(h,w));torch.testing.assert_close(reference,chunked)
for r,c in zip(gref,gchunk):torch.testing.assert_close(r,c,atol=1e-6,rtol=1e-5)
result={'rows_checked':len(rows),'show_split_disjoint':True,'masked_loss_and_gradients_equivalent':True,'reference_loss':reference.item(),'chunked_loss':chunked.item()}
(BASE/'cache/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
