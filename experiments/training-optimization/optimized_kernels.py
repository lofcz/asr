"""Memory-bounded kernels; preserve token-weighted objective and frozen base."""
import contextlib,torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import sdpa_kernel,SDPBackend
class AdaptiveOffload:
 def __init__(self,threshold=36000,keep_gib=1):self.threshold=threshold;self.budget=int(keep_gib*2**30);self.total_bytes=0;self.kept=0;self.enabled=True
 def pack(self,tensor):
  if tensor.ndim==3 and tensor.shape[-1]==3584 and tensor.shape[1]>512 and tensor.device.type=='cuda':
   size=tensor.numel()*tensor.element_size()
   if self.kept+size<=self.budget:self.kept+=size;return ('keep',tensor.detach())
   cpu=torch.empty_like(tensor,device='cpu');cpu.copy_(tensor.detach());self.total_bytes+=size;return ('host',cpu,tensor.device)
  return ('keep',tensor.detach())
 def unpack(self,x):return x[1].to(x[2]) if x[0]=='host' else x[1]
 def context(self,length):
  self.kept=0;self.total_bytes=0;self.enabled=length>self.threshold
  return torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack) if self.enabled else contextlib.nullcontext()
def loss_for(model,item,offload,loss_chunk=512):
 ids,mask,features,labels=item;base=model.get_base_model()
 with offload.context(ids.shape[1]),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
  embeddings=base.get_input_embeddings()(ids).clone();embeddings[mask]=features
  hidden=base.model(inputs_embeds=embeddings,use_cache=False,return_dict=True).last_hidden_state
 targets=labels[:,1:].reshape(-1);states=hidden[:,:-1].reshape(-1,hidden.shape[-1]);valid=targets!=-100;states=states[valid];targets=targets[valid]
 def chunk(x,y):return F.cross_entropy(base.lm_head(x.to(base.lm_head.weight.dtype)).float(),y,reduction='sum')
 total=states.new_zeros((),dtype=torch.float32)
 for i in range(0,len(targets),loss_chunk):
  x,y=states[i:i+loss_chunk],targets[i:i+loss_chunk]
  total=total+(checkpoint(chunk,x,y,use_reentrant=False) if model.training else chunk(x,y))
 return total/len(targets)
