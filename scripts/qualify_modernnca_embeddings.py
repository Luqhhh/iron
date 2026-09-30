"""Compare the complete default encoder against pinned author code.

Artificial formula/gradient/one-update witnesses only; no competition reads.
"""
import argparse
import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from bf_tap_r2.modernnca_model import NeighborNetwork,Settings
from bf_tap_r2.modernnca_audit import encode_numpy
from bf_tap_r2.data import FEATURES

MODEL_SHA='02fce6a107998ab7212774507998e77535f630fbf4ee328acf8519ad7c10632f'
UTILS_SHA='46f26474f559cdda4eda92c5d28d504905c3180ad0ead1bc3f296af72c74230b'


def checked(path,sha):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Pinned author source differs')
    return data


def qualify(source,utils):
    torch.set_num_threads(1)
    namespace={};exec(compile(checked(utils,UTILS_SHA),str(utils),'exec'),namespace)
    def imports(name,*args,**kwargs):
        if name=='TALENT.model.lib.tabr.utils':return SimpleNamespace(make_module=namespace['make_module'])
        return builtins.__import__(name,*args,**kwargs)
    model_ns={'__builtins__':dict(vars(builtins),__import__=imports)}
    exec(compile(checked(source,MODEL_SHA),str(source),'exec'),model_ns)
    s=Settings();categories=4
    torch.manual_seed(63001)
    author=model_ns['ModernNCA'](d_in=len(FEATURES)+categories,d_num=len(FEATURES),d_out=1,
        dim=s.dim,dropout=.1,d_block=512,n_blocks=0,temperature=s.temperature,sample_rate=s.sample_rate,
        num_embeddings=dict(type='PLREmbeddings',n_frequencies=s.frequencies,frequency_scale=s.frequency_scale,
            d_embedding=s.embedding,lite=True)).double()
    torch.manual_seed(63001);native=NeighborNetwork(s,categories)
    if set(author.state_dict())!=set(native.state_dict()) or any(not torch.equal(author.state_dict()[k],v) for k,v in native.state_dict().items()):
        raise ValueError('Author initialization or parameter layout differs')
    rng=np.random.default_rng(63002);x=rng.normal(size=(43,len(FEATURES)))
    cat=np.eye(categories)[np.arange(len(x))%categories]
    values=torch.as_tensor(np.column_stack([x,cat]),dtype=torch.float64)
    labels=torch.as_tensor(rng.normal(size=len(values)),dtype=torch.float64)
    author.eval();native.eval();query=values[:7];bank=values[7:];ybank=labels[7:]
    with torch.no_grad():
        expected=author(query,None,bank,ybank,False);actual=native(query,None,bank,ybank,training=False)
    difference=float((actual-expected).abs().max())
    state={k:v.detach().numpy() for k,v in native.state_dict().items()}
    q=encode_numpy(query.numpy(),state);c=encode_numpy(bank.numpy(),state)
    logits=-np.sqrt(np.sum((q[:,None]-c[None])**2,2))/s.temperature
    weights=np.exp(logits-logits.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
    independent=weights@ybank.numpy();independent_diff=float(np.abs(independent-actual.numpy()).max())
    author.train();native.train();batch=values[:6];yb=labels[:6];pool=values[6:];yp=labels[6:]
    torch.manual_seed(63003);a=author(batch,yb,pool,yp,True);a.square().mean().backward()
    torch.manual_seed(63003);b=native(batch,yb,pool,yp,training=True);b.square().mean().backward()
    training_difference=float((a-b).abs().max().detach())
    grad=max(float((p.grad-dict(author.named_parameters())[k].grad).abs().max()) for k,p in native.named_parameters())
    if max(difference,training_difference,independent_diff)>1e-8 or grad>1e-9:
        raise ValueError('Author/independent formula or gradient differs')
    for model in (author,native):
        torch.optim.AdamW(model.parameters(),lr=s.learning_rate,weight_decay=s.weight_decay).step()
    step=max(float((v-author.state_dict()[k]).abs().max()) for k,v in native.state_dict().items())
    if step>1e-8:raise ValueError('One-update author trajectory differs')
    return dict(status='default_PLR_encoder_source_equivalence_passed',scope='author_default_zero_post_blocks',
        parameter_count=sum(p.numel() for p in native.parameters()),inference_difference=difference,
        independent_numpy_difference=independent_diff,training_prediction_difference=training_difference,
        gradient_difference=grad,one_AdamW_update_difference=step,source_sha256=MODEL_SHA,utils_sha256=UTILS_SHA,
        artificial_optimizer_steps=2,official_label_reads=0,new_official_fits=0,packages=0,uploads=0,
        unqualified=['full_size_resource','complete_official_phase','G1'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--utils',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=qualify(a.source,a.utils)
    with a.output.open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
