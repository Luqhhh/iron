"""Synthetic-only evidence for the historical PLE-B activation/scale defect."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
import rtdl_num_embeddings
from rtdl_num_embeddings import compute_bins, PiecewiseLinearEmbeddings

from bf_tap_r2.v7_periodic import file_hash, write_new


def diagnose():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    raw = torch.randn(100, 3)*torch.tensor([100., 1000., 10000.])+torch.tensor([4000., 8000., 10000.])
    standardized = (raw-raw.mean(0))/raw.std(0, correction=0)
    results = {}
    for coordinate, x in [('raw',raw), ('standardized',standardized)]:
        bins = compute_bins(x,n_bins=8)
        for activation in [True,False]:
            torch.manual_seed(42)
            embedding = PiecewiseLinearEmbeddings(bins,d_embedding=8,activation=activation,version='B')
            optimizer = torch.optim.AdamW(embedding.parameters(),lr=.001)
            target = torch.randn(100,3,8)
            steps=[]
            for step in range(3):
                optimizer.zero_grad()
                output=embedding(x)
                (output-target).square().mean().backward()
                steps.append({'step':step,
                    'nonlinear_gradient_max':float(embedding.linear.weight.grad.abs().max()),
                    'linear_gradient_max':float(embedding.linear0.weight.grad.abs().max()),
                    'embedding_rms':float(output.detach().square().mean().sqrt())})
                optimizer.step()
            nonzero=int(torch.count_nonzero(embedding.linear.weight.detach()))
            assert (nonzero==0) if activation else (nonzero>0)
            results[f'{coordinate}_activation_{activation}']={'steps':steps,'nonlinear_weights_nonzero':nonzero}
    return {'status':'passed','synthetic_only':True,'official_model_fits':0,
            'runtime_version':importlib.metadata.version('rtdl-num-embeddings'),
            'library_source_sha256':file_hash(rtdl_num_embeddings.__file__),
            'diagnostic_source_sha256':file_hash(__file__),'results':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(Path.cwd()/'local'):
        parser.error('Private output required')
    result=diagnose(); write_new(args.output,result); print(json.dumps(result,indent=2))
