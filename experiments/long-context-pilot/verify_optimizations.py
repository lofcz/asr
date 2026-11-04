import copy,json,types,pathlib
import torch
from torch.utils.checkpoint import checkpoint
from train import BoundaryOffload,chunk_mlps

torch.manual_seed(42);torch.set_num_threads(2)
reference=torch.nn.Sequential(torch.nn.Linear(3584,64),torch.nn.GELU(),torch.nn.Linear(64,3584)).cuda()
optimized=copy.deepcopy(reference)
wrapper=types.SimpleNamespace(get_base_model=lambda:types.SimpleNamespace(model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=[types.SimpleNamespace(mlp=optimized)]))))
chunk_mlps(wrapper,128)
x=torch.randn(1,600,3584,device='cuda',requires_grad=True);z=x.detach().clone().requires_grad_()
y=reference(x);loss=y.square().mean();loss.backward()
offload=BoundaryOffload(True)
with offload.context():y2=checkpoint(optimized,z,use_reentrant=False)
loss2=y2.square().mean();loss2.backward()
torch.testing.assert_close(y,y2,atol=1e-5,rtol=1e-4);torch.testing.assert_close(x.grad,z.grad,atol=1e-7,rtol=1e-3)
for a,b in zip(reference.parameters(),optimized.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-3)
result={'output_and_input_and_parameter_gradients_equivalent':True,'loss_reference':loss.item(),'loss_chunked_offloaded':loss2.item(),'offloaded_bytes':offload.total_bytes}
pathlib.Path(__file__).with_name('optimization-verification.json').write_text(json.dumps(result,indent=2));print(result)
