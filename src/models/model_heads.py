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

        

class ClfHead(nn.Module):
    
    ACTIVATIONS = nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['gelu', nn.GELU()],
        ['tanh', nn.Tanh()]
    ])
    
    def __init__(self, hid_sizes: Union[int, list], num_labels: int, activation: str = 'tanh', dropout: bool = True, dropout_prob: float = 0.3):
        super().__init__()
        
        # get input sizes and output sizes for layers as two lists
        if isinstance(hid_sizes, int):
            hid_sizes = [hid_sizes]
            out_sizes = [num_labels]
        elif isinstance(hid_sizes, list):
            if len(hid_sizes)==1:
                out_sizes = [num_labels]
            else:
                out_sizes = hid_sizes[1:] + [num_labels]
        else:
            raise ValueError(f"hid_sizes has to be of type int or float but got {type(hid_sizes)}")
        
        # create sequential class from hid_sizes and out_sizes
        layers = []
        for i, (hid_size, out_size) in enumerate(zip(hid_sizes, out_sizes)):
            if dropout:
                layers.append(nn.Dropout(dropout_prob))
            layers.extend([
                nn.Linear(hid_size, out_size),
                self.ACTIVATIONS[activation]
            ])
        layers = layers[:-1] # remove last activation        
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.classifier(x)
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()        


class AdvHead(nn.Module):
    def __init__(self, adv_count: int = 1, **kwargs):
        super().__init__()    
        self.heads = nn.ModuleList()
        for i in range(adv_count):
            self.heads.append(ClfHead(**kwargs))
            
    def forward(self, x):
        out = []
        for head in self.heads:
            out.append(head(x))
        return out
            
    def forward_reverse(self, x, lmbda = 1.):
        x_ = ReverseLayerF.apply(x, lmbda)
        return self(x_)
    
    def reset_parameters(self):
        for head in self.heads:
            head.reset_parameters()
        
        
