"""Vanilla initialization, fixed alpha across ranks, no previous adapter loading."""
import pathlib
import sys
sys.path[:0]=['/data/work/asr-pilot/long-context-pilot','/data/work/asr-pilot','/data/work/asr-pilot/VibeVoice']
from train import load_model


def fresh_model(rank,mode='qlora'):
    if rank not in (16,64):raise ValueError('Unplanned rank')
    model=load_model(mode,rank,2048)
    model.peft_config['default'].lora_alpha=32
    modules=0
    for module in model.modules():
        if hasattr(module,'lora_A') and 'default' in module.lora_A:
            module.lora_alpha['default']=32
            module.scaling['default']=32/rank
            if module.lora_B['default'].weight.detach().count_nonzero():raise RuntimeError('Nonzero initial adapter: not vanilla')
            modules+=1
    if not modules:raise RuntimeError('No LoRA modules')
    actual={name.rsplit('.',2)[0].split('.')[-1] for name,p in model.named_parameters() if p.requires_grad}
    # Check targeted module families using the concrete adapter keys, not name heuristics.
    for name in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'):
        if not any('.'+name+'.lora_' in key and p.requires_grad for key,p in model.named_parameters()):raise RuntimeError('Missing adapter family '+name)
    if any(p.requires_grad and '.lora_' not in name for name,p in model.named_parameters()):raise RuntimeError('Unexpected trainable base weight')
    return model
