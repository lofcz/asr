import torch,pathlib,json
torch.set_num_threads(2)
root=pathlib.Path('/data/work/asr-optimization');results=[]
for bucket in (1800,3600):
 old=root/f'grads-old-{bucket}.pt';new=root/f'grads-new-{bucket}.pt'
 if not old.exists() or not new.exists():continue
 a=torch.load(old,weights_only=True,map_location='cpu');b=torch.load(new,weights_only=True,map_location='cpu');assert a.keys()==b.keys()
 aa=bb=ab=dd=0.
 for k in a:
  x=a[k].float();y=b[k].float();aa+=float((x*x).sum());bb+=float((y*y).sum());ab+=float((x*y).sum());dd+=float(((x-y)**2).sum())
 row={'bucket':bucket,'gradient_cosine':ab/(aa*bb)**.5,'gradient_relative_l2':(dd/aa)**.5,'gradient_norm_ratio':(bb/aa)**.5,'finite':all(torch.isfinite(v).all().item() for v in b.values())};results.append(row);print(json.dumps(row))
 assert row['finite'] and row['gradient_cosine']>.995 and .98<row['gradient_norm_ratio']<1.02
 del a,b
(root/'gradient-comparison.json').write_text(json.dumps(results,indent=2))
