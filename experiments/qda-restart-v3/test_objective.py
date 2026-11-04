"""The diagnostic masks must preserve the original causal loss and its gradient."""
import copy, types, unittest
import torch
from train_restart import Objective
from optimized_kernels import loss_for

class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(8,8)
    def forward(self, inputs_embeds, **kwargs):
        return types.SimpleNamespace(last_hidden_state=self.layer(inputs_embeds).tanh())

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(19,8);self.model=TinyBackbone();self.lm_head=torch.nn.Linear(8,19,bias=False)
    def get_base_model(self):return self
    def get_input_embeddings(self):return self.embedding

class NoOffload:
    def context(self, length):
        import contextlib
        return contextlib.nullcontext()

class ObjectiveTests(unittest.TestCase):
    def test_diagnostics_preserve_loss_and_gradients(self):
        torch.manual_seed(17);original=TinyModel();candidate=copy.deepcopy(original)
        ids=torch.randint(0,19,(1,700));mask=torch.zeros_like(ids,dtype=torch.bool);mask[:,2:6]=True
        features=torch.randn(4,8);labels=ids.clone();labels[:,:13]=-100
        content=torch.zeros_like(mask);content[:,17::3]=True
        expected=loss_for(original,(ids,mask,features,labels),NoOffload());expected.backward()
        actual,stats=Objective(candidate)((ids,mask,features,labels,content));actual.backward()
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(stats[0],stats[1]+stats[2])
        self.assertEqual(int(stats[3]),687);self.assertEqual(int(stats[4]),int(content.sum()))
        for a,b in zip(original.parameters(),candidate.parameters()):torch.testing.assert_close(a.grad,b.grad)

if __name__=='__main__':unittest.main()
