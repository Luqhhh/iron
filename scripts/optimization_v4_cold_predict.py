"""Predict from a frozen v4 composite, using public process files and stored histories only."""
import argparse
import json
from pathlib import Path
import pandas as pd
from bf_tap.artifacts import file_sha256, stable_digest, file_identities, build_inference_source_contract
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.models.baseline import DualTargetBaseline
from bf_tap.offline import _load_process_sources
from bf_tap.optimization.final_lifecycle import sample_metadata, feature_frame, component_features
from bf_tap.optimization.history_stable import stable_features
from bf_tap.optimization.snapshot_ensemble import blend


def predict(root, data_config, stage):
    root=Path(root)
    m=json.loads((root/'composite.json').read_text())
    if file_sha256(root/'composite.json')!=json.loads((root/'identity.json').read_text())['sha256']:
        raise ContractError('composite identity mismatch')
    if m['candidate'] not in ('R1','R2','R3') or m['calibration']!='none':
        raise ContractError('unregistered composite')
    anchors=m['anchors']
    if len(anchors)!=(2 if m['candidate']=='R3' else 1):
        raise ContractError('wrong anchor count')
    paths=load_yaml(data_config)['paths']
    samples=sample_metadata(paths[stage+'_samples'])
    a=m['algorithm']
    inputs=file_identities({k:paths[k] for k in ('operation_hourly','burden_change')})
    contract=build_inference_source_contract(inputs,semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
    if contract!=m['source_contract']:
        raise ContractError('public source contract mismatch')
    op,burden,_=_load_process_sources(paths,a['semantic'],a['features'])
    outputs=[]
    for anchor in anchors:
        cutoff=pd.Timestamp(anchor['cutoff'])
        if (samples.reference_time<cutoff).any():
            raise ContractError('prediction before anchor cutoff')
        if set(anchor['components'])!={'E09_PROCESS_CHANGE_E02','E04'}:
            raise ContractError('component set mismatch')
        parts=[]
        for c in ('E09_PROCESS_CHANGE_E02','E04'):
            identity=anchor['components'][c]
            directory=root/identity['path']
            if not directory.resolve().is_relative_to(root.resolve()):
                raise ContractError('component path escapes bundle')
            if file_sha256(directory/'bundle.json')!=identity['sha256']:
                raise ContractError('component identity mismatch')
            model=DualTargetBaseline.load(directory)
            tr=model.bundle_metadata_['training']
            if any(pd.Timestamp(tr[k])!=cutoff for k in ('fit_cutoff','history_cutoff','label_available_cutoff')):
                raise ContractError('component cutoff mismatch')
            if model.parameters!=a['baseline']['parameters']:
                raise ContractError('component parameters mismatch')
            history=model.load_history_snapshot()
            if (history.available_at>cutoff).any() or set(history.sample_id.astype(str))&set(samples.sample_id.astype(str)):
                raise ContractError('illegal stored history')
            variant=anchor['variant']
            if variant!=('raw' if m['candidate']=='R3' else m['candidate']):
                raise ContractError('component strategy mismatch')
            X=component_features(feature_frame(samples,history,op,burden,cutoff,a),c,a)
            X=stable_features(X,samples,history,cutoff,variant)
            p=model.predict_raw(X).clip(lower=0)
            p.insert(0,'sample_id',samples.sample_id.astype(str))
            parts.append(p)
        outputs.append(blend(*parts,.8))
    return blend(*outputs) if len(outputs)==2 else outputs[0]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--data-config',required=True)
    parser.add_argument('--stage',default='test_a',choices=['test_a','test_b','test_c'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    frame=predict(args.bundle,args.data_config,args.stage)
    with args.output.open('x') as f: frame.to_csv(f,index=False)

if __name__=='__main__': main()
