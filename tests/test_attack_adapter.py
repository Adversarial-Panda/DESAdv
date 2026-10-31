import torch
from torch import nn
from paper_ensemble import run_ensemble_attack

class Tiny(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.linear=nn.Linear(3*8*8,4)
        with torch.no_grad(): self.linear.bias.fill_(offset)
    def forward(self,x):return self.linear(x.flatten(1))

def test_native_ensattack_budget():
    torch.manual_seed(9)
    models=[Tiny(0.1),Tiny(0.2)]
    x=torch.rand((1,3,8,8))
    y=torch.tensor([1])
    adv=run_ensemble_attack('ens','mifgsm',models,x,y,eps=8/255,alpha=2/255,steps=2,momentum=1.)
    assert adv.shape==x.shape
    assert (adv-x).abs().max()<=8/255+1e-6
    assert adv.min()>=0 and adv.max()<=1
