"""Build at most one fully gated v4 challenger; never activate it or upload it."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from bf_tap.artifacts import atomic_write_json, file_sha256, verify_file_identities
from bf_tap.exceptions import ContractError
from bf_tap.optimization.refresh_factorial import Context, stamp, COMPONENTS
from bf_tap.optimization.final_lifecycle import sample_metadata
from bf_tap.optimization.snapshot_ensemble import blend
from bf_tap.submission import write_submission, pack_submission


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--data-config',default='configs/data.local.yaml')
    args=parser.parse_args()
    status=json.loads((args.run/'final_status.json').read_text())
    gate=json.loads((args.run/'acceptance.json').read_text())
    if status.get('G0')!='PASS_RESEARCH_EXECUTION' or gate['status']!='PASS':
        raise ContractError('complete successful development and accepted candidate required')
    accepted=[(v['J'],c) for c,v in gate['candidates'].items() if v['pass']]
    if not accepted: raise ContractError('no candidate passed')
    _,candidate=min(accepted)
    args.output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(args.output/'selection.json',dict(candidate=candidate,rule='minimum_J_among_full_gate_passes',
        acceptance_sha256=file_sha256(args.run/'acceptance.json'),incumbent='E16',activate=False))
    ctx=Context(args.data_config,args.output)
    anchors=[(ctx.release_cutoff,candidate)] if candidate!='R3' else [(ctx.release_cutoff,'raw'),(stamp(11),'raw')]
    sample=sample_metadata(ctx.paths['test_a_samples'])
    parts=[]
    manifest=dict(schema_version=1,candidate=candidate,algorithm=ctx.a,calibration='none',
                  source_contract=ctx.contract,anchors=[],holdout_consumed=True,
                  development_acceptance_sha256=file_sha256(args.run/'acceptance.json'),
                  implementation_sha256={str(p):file_sha256(p) for p in Path('scripts').glob('optimization_v4_*.py')})
    bundle=args.output/'bundle'
    bundle.mkdir()
    for i,(cutoff,variant) in enumerate(anchors):
        models=ctx.fit(cutoff,variant)
        parts.append(ctx.predict(sample,cutoff,cutoff,variant)['E12-raw'])
        anchor=dict(cutoff=str(cutoff),variant=variant,components={})
        for c in COMPONENTS:
            dest=bundle/f'anchor_{i}'/c
            shutil.copytree(models[c].bundle_directory_,dest)
            anchor['components'][c]=dict(path=str(dest.relative_to(bundle)),sha256=file_sha256(dest/'bundle.json'))
        manifest['anchors'].append(anchor)
    atomic_write_json(bundle/'composite.json',manifest)
    atomic_write_json(bundle/'identity.json',dict(sha256=file_sha256(bundle/'composite.json')))
    expected=blend(*parts) if len(parts)==2 else parts[0]
    # Deliberately omit official label and history files from the cold subprocess config.
    data=dict(schema_version=1,paths={k:ctx.paths[k] for k in ('test_a_samples','operation_hourly','burden_change','data_dictionary')})
    import yaml
    cold_config=args.output/'cold_data.yaml'
    cold_config.write_text(yaml.safe_dump(data))
    cold_path=args.output/'cold_predictions.csv'
    subprocess.run([sys.executable,'scripts/optimization_v4_cold_predict.py','--bundle',str(bundle),
        '--data-config',str(cold_config),'--output',str(cold_path)],check=True)
    cold=pd.read_csv(cold_path,dtype={'sample_id':str})
    a=expected.set_index('sample_id').sort_index()
    b=cold.set_index('sample_id').sort_index()
    if not a.index.equals(b.index) or not np.allclose(a.to_numpy(),b.to_numpy(),rtol=0,atol=1e-10):
        raise ContractError('cold prediction mismatch')
    ordered=sample[['sample_id']].astype(str).merge(expected,on='sample_id',validate='one_to_one',sort=False)
    write_submission(ordered,sample.sample_id,args.output/'result.csv')
    archive=pack_submission(args.output/'result.csv',stage='test_a',team_name='Luqhhh',output_dir=args.output,expected_ids=sample.sample_id)
    verify_file_identities(ctx.inputs)
    atomic_write_json(args.output/'release_validation.json',dict(status='PASS_CHALLENGER_READY',candidate=candidate,
        rows=len(sample),cold_process_max_difference=float(np.abs(a.to_numpy()-b.to_numpy()).max()),
        cold_config_has_official_labels=False,archive_sha256=file_sha256(archive),target_fits=ctx.fit_count,
        incumbent='E16',activated=False,uploaded=False))

if __name__=='__main__': main()
