"""DANet-20 raw-feature shortcuts and a three-layer regression head."""
import torch
from torch import nn
from torch.nn import functional as F

from .danet_terms import AbstractLayer


class BasicBlock(nn.Module):
    def __init__(self,input_dim,raw_dim,width,groups,ghost_size,dropout,learn_masks):
        super().__init__()
        self.first=AbstractLayer(input_dim,width//2,groups,ghost_size,learn_masks=learn_masks)
        self.second=AbstractLayer(width//2,width,groups,ghost_size,learn_masks=learn_masks)
        self.raw_dropout=nn.Dropout(dropout)
        self.shortcut=AbstractLayer(raw_dim,width,groups,ghost_size,learn_masks=learn_masks)

    def forward(self,raw,previous):
        return F.leaky_relu(self.second(self.first(previous))+self.shortcut(self.raw_dropout(raw)),.01)


class Network(nn.Module):
    def __init__(self,input_dim,*,layers=20,width=64,groups=5,ghost_size=256,dropout=.1,learn_masks=True):
        super().__init__()
        if (type(layers) is not int or layers<2 or layers%2 or type(width) is not int or width<2 or width%2
                or not 0<=dropout<1):raise ValueError('Even DANet main-path depth/width and valid dropout required')
        self.blocks=nn.ModuleList([BasicBlock(input_dim if i==0 else width,input_dim,width,groups,ghost_size,dropout,learn_masks)
                                   for i in range(layers//2)])
        self.dropout=nn.Dropout(.1)
        self.head=nn.Sequential(nn.Linear(width,256),nn.ReLU(),nn.Linear(256,512),nn.ReLU(),nn.Linear(512,1))

    def forward(self,features):
        previous=features
        for block in self.blocks:previous=block(features,previous)
        return self.head(self.dropout(previous)).squeeze(-1)
