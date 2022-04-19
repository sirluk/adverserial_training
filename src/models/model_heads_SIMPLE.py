# The Classes in this script are just use to showcase the general architecture

import torch
from torch import nn
from torch.autograd import Function

from typing import Union

class ReverseLayerF(Function):
    # https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/2
    @staticmethod
    def forward(ctx, input_, lmbda):
        ctx.lmbda = lmbda
        return input_.view_as(input_)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.lmbda
        return grad_input, None


class ClfHeadSimple(nn.Module):
    
    def __init__(self, hid_size: int, num_labels: int):
        super().__init__()
                
        # create sequential class from hid_size and out_sizes
        layers = [
            nn.Dropout(.3),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Dropout(.3),
            nn.Linear(hid_size, num_labels)
        ] 
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.classifier(x)       


class AdvHead(nn.Module):
    def __init__(self, adv_count: int = 1, **kwargs):
        super().__init__()    
        
        # initialize n=adv_count heads
        self.heads = nn.ModuleList()
        for i in range(adv_count):
            self.heads.append(ClfHead(**kwargs))
            
    def forward(self, x):
        # iterate over heads and store ouputs in list
        out = []
        for head in self.heads:
            out.append(head(x))
        return out
            
    def forward_reverse(self, x, lmbda = 1.):
        # add gradient reversal layer to modify gradients when they are coming out of the heads into the model "underneath"
        x_ = ReverseLayerF.apply(x, lmbda)
        return self(x_)

        
