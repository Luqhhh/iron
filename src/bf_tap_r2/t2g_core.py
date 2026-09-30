"""Frozen T2G-Former mechanism adaptation; upstream attribution in third_party."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("LEARNED_GRAPH", "DENSE_CONTROL")

class GraphAttention(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        self.arm = arm
        self.q = nn.Linear(64,64)
        self.v = nn.Linear(64,64)
        self.out = nn.Linear(64,64)
        self.relation = nn.Parameter(torch.ones(4,16))
        self.column_head = nn.Parameter(torch.empty(4,22,10))
        self.column_tail = nn.Parameter(torch.empty(4,22,10))
        self.topology_bias = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(.1)

    def topology(self):
        h = F.normalize(self.column_head,dim=-1,eps=1e-12)
        t = F.normalize(self.column_tail,dim=-1,eps=1e-12)
        p = torch.sigmoid(h @ t.transpose(-1,-2) + self.topology_bias)
        learned = (p > .5).to(p.dtype) - p.detach() + p
        feature = learned if self.arm == "LEARNED_GRAPH" else torch.ones_like(p)
        eye = torch.eye(22,device=p.device,dtype=p.dtype).unsqueeze(0)
        feature = feature * (1-eye) + eye
        # Concatenation fixes readout edges without allowing their topology gradients.
        top = torch.cat((torch.zeros(4,1,1,device=p.device),
                         torch.ones(4,1,22,device=p.device)),dim=-1)
        bottom = torch.cat((torch.zeros(4,22,1,device=p.device),feature),dim=-1)
        return torch.cat((top,bottom),dim=-2)

    def weights(self,x,last=False):
        n,k,_ = x.shape
        q = self.q(x).reshape(n,k,4,16).transpose(1,2)
        query = q[:,:,:1] if last else q
        score = (query*self.relation[None,:,None,:]) @ q.transpose(-1,-2) / 4
        a = self.topology()[:,:1] if last else self.topology()
        return torch.softmax(score - 10000*(1-a[None]),dim=-1)

    def forward(self,x,last=False):
        n,k,_ = x.shape
        v = self.v(x).reshape(n,k,4,16).transpose(1,2)
        z = self.dropout(self.weights(x,last)) @ v
        z = z.transpose(1,2).reshape(n,1 if last else k,64)
        return self.out(z)

class GraphBlock(nn.Module):
    def __init__(self,arm,last):
        super().__init__()
        self.last=last
        self.norm_attention=nn.LayerNorm(64,eps=1e-5)
        self.attention=GraphAttention(arm)
        self.norm_ffn=nn.LayerNorm(64,eps=1e-5)
        self.ffn_in=nn.Linear(64,256)
        self.ffn_out=nn.Linear(128,64)
        self.dropout=nn.Dropout(.1)

    def forward(self,x):
        residual = x[:,:1] if self.last else x
        x = residual+self.attention(self.norm_attention(x),self.last)
        value,gate = self.ffn_in(self.norm_ffn(x)).chunk(2,dim=-1)
        return x+self.ffn_out(self.dropout(value*F.relu(gate)))

class T2GRegressor(nn.Module):
    def __init__(self,arm,n_categories):
        super().__init__()
        if arm not in ARMS or type(n_categories) is not int or n_categories<1:
            raise ValueError("Invalid arm or category cardinality")
        self.arm=arm
        self.n_categories=n_categories
        self.numeric_weight=nn.Parameter(torch.empty(21,64))
        self.numeric_bias=nn.Parameter(torch.zeros(21,64))
        self.category=nn.Embedding(n_categories,64)
        self.category_bias=nn.Parameter(torch.zeros(64))
        self.readout=nn.Parameter(torch.empty(64))
        self.blocks=nn.ModuleList([GraphBlock(arm,False),GraphBlock(arm,True)])
        self.norm=nn.LayerNorm(64,eps=1e-5)
        self.head=nn.Linear(64,1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight,gain=1/math.sqrt(2))
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        for p in (self.numeric_weight,self.category.weight):
            nn.init.kaiming_uniform_(p,a=math.sqrt(5))
        for b in self.blocks:
            nn.init.kaiming_uniform_(b.attention.column_head,a=math.sqrt(5))
            nn.init.kaiming_uniform_(b.attention.column_tail,a=math.sqrt(5))
            nn.init.ones_(b.attention.relation)
            nn.init.zeros_(b.attention.topology_bias)
        nn.init.zeros_(self.numeric_bias); nn.init.zeros_(self.category_bias)
        nn.init.normal_(self.readout,mean=0,std=1)

    def topology(self):
        return self.blocks[0].attention.topology()

    def forward(self,numeric,category):
        if (numeric.ndim!=2 or numeric.shape[1]!=21 or numeric.dtype!=torch.float32
            or category.ndim!=1 or len(category)!=len(numeric)
            or category.dtype!=torch.int64 or not torch.isfinite(numeric).all()
            or (category<0).any() or (category>=self.n_categories).any()):
            raise ValueError("Invalid numeric/category query")
        tokens=numeric.unsqueeze(-1)*self.numeric_weight+self.numeric_bias
        categorical=self.category(category)+self.category_bias
        x=torch.cat((self.readout.expand(len(numeric),1,64),tokens,
                     categorical.unsqueeze(1)),dim=1)
        for block in self.blocks:
            x=block(x)
        return self.head(F.relu(self.norm(x[:,0]))).squeeze(-1)

def make_model(arm: str,n_categories: int,seed: int=42) -> nn.Module:
    torch.manual_seed(seed)
    return T2GRegressor(arm,n_categories)
