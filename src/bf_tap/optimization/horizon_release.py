"""OPT-17 zero-fit composite deployment; test A challenger, B/C dry runs only."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from ..artifacts import atomic_write_json,file_identities,file_sha256,stable_digest,verify_file_identities,build_inference_source_contract
from ..config import load_yaml
from ..exceptions import ContractError
from ..models.baseline import DualTargetBaseline
from ..offline import _load_process_sources
from ..submission import write_submission,pack_submission,validate_submission
from .component_export import META,PRED,ROLES,read_json,forbid_fit,ComponentFeatures
from .final_lifecycle import algorithm,sample_metadata
from .horizon_router import S1,TARGETS,load_component,predict_parts,route,horizons
from .horizon_evaluation import append_access


def predict(bundle,data_config,stage,output):
    if output.exists():raise FileExistsError(output)
    root=Path(bundle);m=read_json(root/'router.json')
    if file_sha256(root/'router.json')!=read_json(root/'identity.json')['sha256']:raise ContractError('router digest mismatch')
    if m['candidate']!='S1_HORIZON_ROUTER' or m['routes']!={str(k):v for k,v in S1.items()} or set(m['components'])!={'O0','OR','HR'}:raise ContractError('unregistered router')
    verify_file_identities(m['implementation'])
    paths=load_yaml(data_config)['paths']
    if set(paths)-{'test_a_samples','test_b_samples','test_c_samples','operation_hourly','burden_change','data_dictionary'}:raise ContractError('label path forbidden in deployment config')
    contract=build_inference_source_contract(file_identities({k:paths[k] for k in ('operation_hourly','burden_change')}),semantic_contract_sha256=m['algorithm']['contract_digests']['semantic_contract_sha256'])
    if contract!=m['source_contract']:raise ContractError('public source mismatch')
    entries={}
    for role,entry in m['components'].items():
        e=dict(entry);path=root/e['path']
        if not path.resolve().is_relative_to(root.resolve()) or e['role']!=role or pd.Timestamp(e['cutoff'])!=pd.Timestamp(m['cutoff']):raise ContractError('component path/cutoff/role mismatch')
        e['path']=str(path);entries[role]=e
    samples=sample_metadata(paths[stage+'_samples']);op,burden,_=_load_process_sources(paths,m['algorithm']['semantic'],m['algorithm']['features'])
    counter={'attempted_target_fits':0}
    with forbid_fit(counter):
        loaded={k:load_component(v,m['algorithm'],contract) for k,v in entries.items()}
        parts=predict_parts(samples[META],entries,loaded,ComponentFeatures(m['algorithm'],op,burden))
        pred=route(samples[META],m['cutoff'],parts)
    with output.open('x') as f:pred.to_csv(f,index=False)
    atomic_write_json(output.with_suffix('.audit.json'),dict(**counter,completed_target_fits=0,label_paths_present=False,rows=len(pred)))
    return samples,parts,pred


def release(run,data_config,output):
    gate=read_json(run/'acceptance.json');status=read_json(run/'final_status.json')
    if not gate['passed'] or status.get('G0')!='PASS_OPT17':raise ContractError('S1 deployment pass required')
    output.mkdir(parents=True,exist_ok=False)
    try:
        a=algorithm();paths=load_yaml(data_config)['paths'];active=load_yaml('configs/optimization_v0_4/active_release.yaml')
        rroot=Path(active['bundle']);r=read_json(rroot/'composite.json');oroot=Path('local/runs/optimization-v0.3-r2-opt10-final-r1/bundle');o=read_json(oroot/'bundle.json')
        if file_sha256(rroot/'composite.json')!=active['composite_sha256'] or r['candidate']!='R2' or len(r['anchors'])!=1:raise ContractError('active R2 mismatch')
        cutoff=r['anchors'][0]['cutoff']
        if pd.Timestamp(o['training']['fit_cutoff'])!=pd.Timestamp(cutoff) or stable_digest(o['algorithm'])!=stable_digest(a):raise ContractError('raw final anchor mismatch')
        protected={str(p):p for p in [Path('configs/optimization_v0_4/active_release.yaml'),rroot.parent/'Luqhhh_bf_tap_predict_prelim.zip',Path('local/ledgers/optimization-v0.3-r2-protected.jsonl')]}
        manifest=dict(candidate='S1_HORIZON_ROUTER',cutoff=cutoff,acceptance_sha256=file_sha256(run/'acceptance.json'),
            registration_sha256=file_sha256('configs/optimization_v0_6/experiment.yaml'),protected=file_identities(protected),
            raw_root_sha256=file_sha256(oroot/'bundle.json'),R2_root_sha256=file_sha256(rroot/'composite.json'),new_fits=0)
        atomic_write_json(output/'manifest.json',manifest);append_access(output,'OPT17_gated_existing_final_component_recovery_no_official_target_reads')
        root=output/'bundle';root.mkdir();entries={};counter={'attempted_target_fits':0}
        with forbid_fit(counter):
            for role in ('O0','OR','HR'):
                variant,component=ROLES[role]
                if role=='O0':src=oroot/component;digest=o['components'][component]
                else:en=r['anchors'][0]['components'][component];src=rroot/en['path'];digest=en['sha256']
                if file_sha256(src/'bundle.json')!=digest:raise ContractError('final component archive mismatch')
                model=DualTargetBaseline.load(src);md=model.bundle_metadata_
                relevant={k:v for k,v in md['code_identity'].items() if k.startswith(('src/bf_tap/features/','src/bf_tap/models/','configs/')) or k in ('src/bf_tap/optimization/history_stable.py','src/bf_tap/optimization/features.py','src/bf_tap/optimization/process_change.py','src/bf_tap/optimization/final_lifecycle.py','src/bf_tap/availability.py')}
                e=dict(path=str(src),role=role,variant=variant,component=component,cutoff=cutoff,bundle_sha256=digest,
                    schema_sha256=stable_digest(model.feature_schema_),history_snapshot_sha256=file_sha256(src/'history_snapshot.csv'),verified_source_files=relevant)
                load_component(e,a,r['source_contract'])
                if md['training']['sample_ids_sha256']!=o['training']['sample_ids_sha256']:raise ContractError('final training samples differ')
                shutil.copytree(src,root/role);e['path']=role;entries[role]=e
        m=dict(candidate='S1_HORIZON_ROUTER',routes=S1,cutoff=cutoff,algorithm=a,source_contract=r['source_contract'],components=entries,
            implementation=file_identities({str(p):p for p in Path('src/bf_tap').rglob('*.py')}),calibration='none',component_weights=[.8,.2])
        atomic_write_json(root/'router.json',m);atomic_write_json(root/'identity.json',dict(sha256=file_sha256(root/'router.json')))
        cold={k:paths[k] for k in ('test_a_samples','test_b_samples','test_c_samples','operation_hourly','burden_change','data_dictionary')}
        cfg=output/'no_labels.yaml';cfg.write_text(yaml.safe_dump(dict(schema_version=1,paths=cold)))
        stage_rows=[]
        for stage in ('test_a','test_b','test_c'):
            samples,parts,direct=predict(root,cfg,stage,output/f'{stage}_direct.csv')
            if stage=='test_a' and samples.reference_time.min()!=pd.Timestamp(cutoff):raise ContractError('cutoff differs from earliest test reference')
            coldfile=output/f'{stage}_cold.csv'
            subprocess.run([sys.executable,'-m','bf_tap.optimization.horizon_release','predict','--bundle',str(root),'--data-config',str(cfg),'--stage',stage,'--output',str(coldfile)],check=True)
            coldpred=pd.read_csv(coldfile,dtype={'sample_id':str}).set_index('sample_id').sort_index();d=direct.set_index('sample_id').sort_index()
            if not d.index.equals(coldpred.index) or not np.allclose(d[PRED],coldpred[PRED],rtol=0,atol=1e-10):raise ContractError('cold-process router mismatch')
            # Independently recover the untouched incumbent with its old official loader.
            oldfile=output/f'{stage}_archived_R2.csv'
            subprocess.run([sys.executable,'scripts/optimization_v4_cold_predict.py','--bundle',str(rroot),'--data-config',str(cfg),'--stage',stage,'--output',str(oldfile)],check=True)
            archive=pd.read_csv(oldfile,dtype={'sample_id':str}).set_index('sample_id').sort_index();rp=parts['R2'].set_index('sample_id').sort_index()
            difference=float(np.abs(archive[PRED]-rp[PRED]).to_numpy().max())
            if not archive.index.equals(rp.index) or difference>=1e-8:raise ContractError('final R2 endpoint mismatch')
            h=horizons(samples[META],cutoff);mask=h.to_numpy()==4
            if not np.array_equal(direct[PRED].to_numpy()[mask],parts['R2'][PRED].to_numpy()[mask]):raise ContractError('H4 changed')
            validate_submission(direct,samples.sample_id)
            stage_rows.append(dict(stage=stage,rows=len(samples),horizons={str(k):int(v) for k,v in h.value_counts().sort_index().items()},R2_endpoint_max_difference=difference,cold_max_difference=float(np.abs(d[PRED]-coldpred[PRED]).to_numpy().max()),scored=False))
            if stage=='test_a':
                write_submission(direct,samples.sample_id,output/'result.csv')
                archive_path=pack_submission(output/'result.csv',stage='test_a',team_name='Luqhhh',output_dir=output,expected_ids=samples.sample_id)
            print(f'{stage}: S1 label-free cold recovery passed',flush=True)
        verify_file_identities(manifest['protected'])
        atomic_write_json(output/'final_status.json',dict(status='READY_CHALLENGER',candidate='S1_HORIZON_ROUTER',stages=stage_rows,**counter,completed_target_fits=0,archive_sha256=file_sha256(archive_path),active_R2_unchanged=True,uploaded=False))
    except Exception as e:
        atomic_write_json(output/'failure.json',dict(error=str(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    for name in ('release','predict'):
        q=s.add_parser(name);q.add_argument('--data-config',default='configs/data.local.yaml');q.add_argument('--output',type=Path,required=True)
        if name=='release':q.add_argument('--run',type=Path,required=True)
        else:q.add_argument('--bundle',type=Path,required=True);q.add_argument('--stage',required=True,choices=['test_a','test_b','test_c'])
    args=p.parse_args()
    if args.command=='release':release(args.run,args.data_config,args.output)
    else:predict(args.bundle,args.data_config,args.stage,args.output)
