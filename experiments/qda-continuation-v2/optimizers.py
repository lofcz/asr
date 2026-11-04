"""Pinned PoLoRA with complete durable state; upstream numerical update unchanged."""
import copy, pathlib,sys
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'vendor/polora'))
from polora import Polora

class ResumablePolora(Polora):
    FORMAT=1
    FIELDS=('epsilon','delta','beta1','curvature_beta','ns_steps','higham_iters')
    def state_dict(self):
        result=super().state_dict()
        result['polora_complete']={'version':self.FORMAT,'shapes':[(tuple(a.shape),tuple(b.shape)) for a,b in self.pairs],
            'hyperparameters':{k:getattr(self,k) for k in self.FIELDS},
            'pairs':{i:{k:v.detach().cpu().clone() for k,v in s.items()} for i,s in self.pair_state.items()}}
        return result
    def load_state_dict(self,state_dict):
        state=dict(state_dict);extra=state.pop('polora_complete',None)
        if not extra or extra['version']!=self.FORMAT:raise ValueError('Incomplete PoLoRA optimizer checkpoint')
        if extra['shapes']!=[(tuple(a.shape),tuple(b.shape)) for a,b in self.pairs]:raise ValueError('PoLoRA pair ordering/shape mismatch')
        if extra['hyperparameters']!={k:getattr(self,k) for k in self.FIELDS}:raise ValueError('PoLoRA hyperparameters changed')
        if set(extra['pairs'])!=set(range(len(self.pairs))):raise ValueError('Missing PoLoRA pair state')
        for i,(a,b) in enumerate(self.pairs):
            s=extra['pairs'][i]
            if not {'M_A','M_B','Q','P'}<=s.keys():raise ValueError('Missing PoLoRA momentum/preconditioner')
            if s['M_A'].shape!=a.shape or s['M_B'].shape!=b.shape or s['Q'].shape!=(a.shape[1],) or s['P'].shape!=(b.shape[0],):raise ValueError('Corrupt PoLoRA state shape')
            if not all(torch.isfinite(v).all() for v in s.values()):raise ValueError('Nonfinite PoLoRA state')
        super().load_state_dict(state)
        self.pair_state={i:{k:v.to(device=self.pairs[i][0].device,dtype=torch.float32).clone() for k,v in s.items()} for i,s in extra['pairs'].items()}

def build_optimizer(model,name,lr):
    params=[p for p in model.parameters() if p.requires_grad]
    if not params:raise ValueError('No trainable parameters')
    if name=='adamw':return torch.optim.AdamW(params,lr=lr,weight_decay=0.)
    if name=='polora':
        opt=ResumablePolora(model,lr=lr,compile=False)
        if {id(p) for p in params}!={id(p) for a,b in opt.pairs for p in (a,b)}:raise ValueError('PoLoRA would miss trainable parameters')
        return opt
    raise ValueError(name)
