import copy,unittest,torch
from optimizers import ResumablePolora
class ResumeTests(unittest.TestCase):
 def test_exact_continuation(self):
  torch.manual_seed(42);a=torch.nn.Parameter(torch.randn(4,12)*.1);b=torch.nn.Parameter(torch.randn(9,4)*.01)
  x=torch.randn(16,12);y=torch.randn(16,9)
  def step(opt,a,b):
   opt.zero_grad();loss=((x@a.T@b.T-y)**2).mean();loss.backward();opt.step()
  opt=ResumablePolora(pairs=[(a,b)],lr=.002)
  for _ in range(3):step(opt,a,b)
  state=copy.deepcopy(opt.state_dict());aa=torch.nn.Parameter(a.detach().clone());bb=torch.nn.Parameter(b.detach().clone())
  other=ResumablePolora(pairs=[(aa,bb)],lr=.002);other.load_state_dict(state)
  for _ in range(3):step(opt,a,b);step(other,aa,bb)
  self.assertTrue(torch.equal(a,aa));self.assertTrue(torch.equal(b,bb))
  for k,v in opt.pair_state[0].items():self.assertTrue(torch.equal(v,other.pair_state[0][k]),k)
 def test_incomplete_checkpoint_rejected(self):
  a=torch.nn.Parameter(torch.randn(2,4));b=torch.nn.Parameter(torch.zeros(3,2));o=ResumablePolora(pairs=[(a,b)])
  s=o.state_dict();del s['polora_complete']
  with self.assertRaises(ValueError):o.load_state_dict(s)
 def test_zero_b_finite(self):
  a=torch.nn.Parameter(torch.randn(4,12)*.1);b=torch.nn.Parameter(torch.zeros(9,4));o=ResumablePolora(pairs=[(a,b)],lr=.002)
  x=torch.randn(16,12);((x@a.T@b.T-1)**2).mean().backward();o.step()
  self.assertTrue(torch.isfinite(a).all());self.assertTrue(torch.isfinite(b).all());self.assertGreater(b.abs().sum(),0)
 def test_bfloat16_resume_with_independent_rng(self):
  torch.manual_seed(4);a=torch.nn.Parameter((torch.randn(4,12)*.1).bfloat16());b=torch.nn.Parameter((torch.randn(9,4)*.01).bfloat16())
  x=torch.randn(16,12).bfloat16();y=torch.randn(16,9)
  def step(opt,a,b):
   opt.zero_grad();loss=((x@a.T@b.T).float()-y).square().mean();loss.backward();opt.step()
  first=ResumablePolora(pairs=[(a,b)],lr=.002)
  for _ in range(2):step(first,a,b)
  aa=torch.nn.Parameter(a.detach().clone());bb=torch.nn.Parameter(b.detach().clone());second=ResumablePolora(pairs=[(aa,bb)],lr=.002);second.load_state_dict(copy.deepcopy(first.state_dict()))
  for _ in range(3):
   torch.manual_seed(19);step(first,a,b);torch.manual_seed(99);step(second,aa,bb)
  self.assertTrue(torch.equal(a,aa));self.assertTrue(torch.equal(b,bb))
  for k,v in first.pair_state[0].items():self.assertTrue(torch.equal(v,second.pair_state[0][k]),k)
if __name__=='__main__':unittest.main()
