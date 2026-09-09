"""OPT-16 mixed-variant component bundle: one gated challenger, no activation/upload."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from ..artifacts import atomic_write_json, file_sha256, file_identities, stable_digest, verify_file_identities, build_inference_source_contract
from ..config import load_yaml
from ..exceptions import ContractError
from ..models.baseline import DualTargetBaseline
from ..offline import _load_process_sources
from ..submission import write_submission, pack_submission, validate_submission
from .component_export import META, PRED, ComponentFeatures, load_verified_component, forbid_fit, read_json
from .component_followup import COMPONENTS, fit_component, append_access
from .final_lifecycle import algorithm, source_files, sample_metadata, label_metadata, eligible, _read_eligible_rows
from .history_component_v5 import transform, MAIN, AUX
from .snapshot_ensemble import blend


def implementation_files():
    return {str(p):p for p in Path('src/bf_tap').rglob('*.py')}


class Predictor:
    def __init__(self,root):
        self.root=Path(root);self.m=read_json(self.root/'composite.json')
        if file_sha256(self.root/'composite.json')!=read_json(self.root/'identity.json')['sha256']:
            raise ContractError('mixed composite digest mismatch')
        m=self.m
        if m['schema']!='v5-component-composite-v1' or m['candidate'] not in ('T1','T2') or m['weights']!={MAIN:.8,AUX:.2} or m['calibration']!='none':
            raise ContractError('unregistered mixed composite')
        verify_file_identities(m['implementation'])
        self.models={};self.history={}
        if set(m['components'])!={MAIN,AUX}:
            raise ContractError('mixed composite component set mismatch')
        for component,entry in m['components'].items():
            expected=m['candidate'] if component==COMPONENTS[m['candidate']] else 'R2'
            if entry['variant']!=expected:
                raise ContractError('component-specific variant mismatch')
            path=self.root/entry['path']
            if not path.resolve().is_relative_to(self.root.resolve()) or file_sha256(path/'bundle.json')!=entry['bundle_sha256']:
                raise ContractError('component identity mismatch')
            model=DualTargetBaseline.load(path);md=model.bundle_metadata_;tr=md['training']
            if tr.get('variant')!=entry['variant'] or any(pd.Timestamp(tr[k])!=pd.Timestamp(m['cutoff']) for k in ('fit_cutoff','history_cutoff','label_available_cutoff')):
                raise ContractError('component metadata cutoff/variant mismatch')
            if md['inference_source_contract']!=m['source_contract'] or model.parameters!=m['algorithm']['baseline']['parameters']:
                raise ContractError('component source/parameter mismatch')
            if stable_digest(model.feature_schema_)!=entry['schema_sha256'] or file_sha256(path/'history_snapshot.csv')!=entry['history_snapshot_sha256']:
                raise ContractError('component schema/history digest mismatch')
            history=model.load_history_snapshot()
            if (history.available_at>pd.Timestamp(m['cutoff'])).any():
                raise ContractError('stored future history')
            self.models[component]=model;self.history[component]=history

    def predict(self,samples,op,burden):
        builder=ComponentFeatures(self.m['algorithm'],op,burden);parts=[]
        for component in (MAIN,AUX):
            entry=self.m['components'][component]
            base=builder.X(samples[META],dict(cutoff=self.m['cutoff'],component=component,variant='R2'),self.history[component])
            X=transform(base,samples[META],self.history[component],pd.Timestamp(self.m['cutoff']),entry['variant'],component)
            p=self.models[component].predict_raw(X).clip(lower=0)
            p.insert(0,'sample_id',samples.sample_id.astype(str));parts.append(p)
        return blend(*parts,.8)


def cold_predict(bundle,data_config,stage,output):
    if output.exists():raise FileExistsError(output)
    m=Predictor(bundle);paths=load_yaml(data_config)['paths']
    inputs=file_identities({k:paths[k] for k in ('operation_hourly','burden_change')})
    contract=build_inference_source_contract(inputs,semantic_contract_sha256=m.m['algorithm']['contract_digests']['semantic_contract_sha256'])
    if contract!=m.m['source_contract']:raise ContractError('prediction process source contract mismatch')
    samples=sample_metadata(paths[stage+'_samples'])
    op,burden,_=_load_process_sources(paths,m.m['algorithm']['semantic'],m.m['algorithm']['features'])
    counter={'attempted_target_fits':0}
    with forbid_fit(counter):p=m.predict(samples,op,burden)
    with output.open('x') as f:p.to_csv(f,index=False)
    return p


def release(run,data_config,output):
    gate=read_json(run/'acceptance.json');state=read_json(run/'final_status.json')
    candidate=gate['selected_STAGE_A']
    if state.get('G0')!='PASS_OPT15' or candidate not in ('T1','T2') or not gate['candidates'][candidate]['STAGE_A']['passed']:
        raise ContractError('complete STAGE_A passing challenger required')
    output.mkdir(parents=True,exist_ok=False)
    try:
        cfg=load_yaml('configs/optimization_v0_5/opt15.yaml');a=algorithm()
        paths=load_yaml(data_config)['paths'];samples=sample_metadata(paths['test_a_samples'])
        cutoff=samples.reference_time.min();metadata=label_metadata(paths,a['semantic'])
        metadata=metadata.loc[(metadata.reference_time>=pd.Timestamp(cfg['training_reference_start'])) &
                              (metadata.reference_time<pd.Timestamp('2024-12-01',tz='Asia/Shanghai'))]
        train_meta=eligible(metadata,cutoff)
        inputs=file_identities({k:paths[k] for k in ('train_samples','tap_history_train','operation_hourly','burden_change','data_dictionary','test_a_samples')})
        dev=read_json(run/'manifest.json')
        for k,v in dev['inputs'].items():
            if inputs[k]['sha256']!=v['sha256']:raise ContractError('release data differs from development')
        sources=file_identities({**source_files(),**{str(p):p for p in Path('configs/optimization_v0_5').glob('*.yaml')}})
        contract=build_inference_source_contract(inputs,semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
        active=load_yaml('configs/optimization_v0_4/active_release.yaml');oldroot=Path(active['bundle'])
        if file_sha256(oldroot/'composite.json')!=active['composite_sha256']:raise ContractError('active R2 digest changed')
        old=read_json(oldroot/'composite.json');anchor=old['anchors'][0]
        if old['candidate']!='R2' or pd.Timestamp(anchor['cutoff'])!=cutoff or anchor['variant']!='R2':
            raise ContractError('final R2 anchor does not match legal release cutoff')
        manifest=dict(candidate=candidate,cutoff=str(cutoff),sources=sources,inputs=inputs,source_contract=contract,
            acceptance_sha256=file_sha256(run/'acceptance.json'),selection='registered_STAGE_A_then_H1_J_complexity',
            active_release_sha256=file_sha256('configs/optimization_v0_4/active_release.yaml'),
            eligible_samples_sha256=stable_digest(train_meta.sample_id.astype(str).tolist()),holdout_consumed=True)
        atomic_write_json(output/'manifest.json',manifest)
        append_access(cfg['access_ledger'],output/'manifest.json','OPT16_frozen_winner_final_training')
        labels=_read_eligible_rows(paths['train_samples'],train_meta.sample_id).merge(train_meta[['sample_id','label_available_at']],on='sample_id',validate='one_to_one')
        loaded={};counter={'attempted_target_fits':0}
        with forbid_fit(counter):
            for component in (MAIN,AUX):
                entry=anchor['components'][component]
                identity=dict(path=str(oldroot/entry['path']),bundle_sha256=entry['sha256'],cutoff=str(cutoff),variant='R2',component=component)
                loaded[component]=load_verified_component(identity,a,contract,metadata)
        op,burden,_=_load_process_sources(paths,a['semantic'],a['features']);builder=ComponentFeatures(a,op,burden)
        component=COMPONENTS[candidate];history=loaded[component][1]
        fresh,fit_identity=fit_component(builder,labels,history,cutoff,candidate,output/'fitted_component',manifest,
            'local/ledgers/optimization-v0.5-opt16-fit-budget.jsonl',cfg['max_final_target_fits'])
        models={c:loaded[c][0] for c in (MAIN,AUX)};models[component]=fresh
        root=output/'bundle';root.mkdir()
        composite=dict(schema='v5-component-composite-v1',candidate=candidate,cutoff=str(cutoff),
            algorithm=a,weights={MAIN:.8,AUX:.2},calibration='none',source_contract=contract,components={},
            acceptance_sha256=manifest['acceptance_sha256'],implementation=file_identities(implementation_files()))
        for c,model in models.items():
            destination=root/c;shutil.copytree(model.bundle_directory_,destination)
            composite['components'][c]=dict(path=c,variant=candidate if c==component else 'R2',
                bundle_sha256=file_sha256(destination/'bundle.json'),schema_sha256=stable_digest(model.feature_schema_),
                history_snapshot_sha256=file_sha256(destination/'history_snapshot.csv'),
                model_fit_cutoff=str(cutoff),inference_history_cutoff=str(cutoff))
        atomic_write_json(root/'composite.json',composite);atomic_write_json(root/'identity.json',dict(sha256=file_sha256(root/'composite.json')))
        local=Predictor(root)
        with forbid_fit(counter):expected=local.predict(samples,op,burden)
        cold_paths={k:paths[k] for k in ('test_a_samples','test_b_samples','test_c_samples','operation_hourly','burden_change','data_dictionary')}
        cold_cfg=output/'cold_data.yaml';cold_cfg.write_text(yaml.safe_dump(dict(schema_version=1,paths=cold_paths)))
        subprocess.run([sys.executable,'-m','bf_tap.optimization.component_release_v5','predict','--bundle',str(root),
            '--data-config',str(cold_cfg),'--stage','test_a','--output',str(output/'cold_test_a.csv')],check=True)
        cold=pd.read_csv(output/'cold_test_a.csv',dtype={'sample_id':str}).set_index('sample_id').sort_index()
        direct=expected.set_index('sample_id').sort_index()
        if not cold.index.equals(direct.index) or not np.allclose(cold.to_numpy(),direct.to_numpy(),rtol=0,atol=1e-10):
            raise ContractError('cold-process candidate mismatch')
        ordered=samples[['sample_id']].astype(str).merge(expected,on='sample_id',validate='one_to_one',sort=False)
        validate_submission(ordered,samples.sample_id)
        write_submission(ordered,samples.sample_id,output/'result.csv')
        archive=pack_submission(output/'result.csv',stage='test_a',team_name='Luqhhh',output_dir=output,expected_ids=samples.sample_id)
        verify_file_identities(inputs);verify_file_identities(sources)
        if file_sha256('configs/optimization_v0_4/active_release.yaml')!=manifest['active_release_sha256']:
            raise ContractError('active R2 pointer changed')
        atomic_write_json(output/'final_status.json',dict(status='READY_CHALLENGER',candidate=candidate,
            new_final_target_fits=2,reused_final_component_count=1,training_rows=len(train_meta),test_a_rows=len(samples),
            cutoff=str(cutoff),archive_sha256=file_sha256(archive),cold_max_abs_difference=float(np.abs(cold.to_numpy()-direct.to_numpy()).max()),
            cold_official_label_paths=False,active_R2_unchanged=True,uploaded=False))
    except Exception as exc:
        atomic_write_json(output/'final_status.json',dict(status='FAILED',error=str(exc)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('release');r.add_argument('--run',type=Path,required=True);r.add_argument('--data-config',default='configs/data.local.yaml');r.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('predict');c.add_argument('--bundle',type=Path,required=True);c.add_argument('--data-config',required=True);c.add_argument('--stage',choices=['test_a','test_b','test_c'],default='test_a');c.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='release':release(args.run,args.data_config,args.output)
    else:cold_predict(args.bundle,args.data_config,args.stage,args.output)

if __name__=='__main__':main()
