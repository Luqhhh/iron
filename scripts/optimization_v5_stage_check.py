"""Unlabeled B/C runnability and calendar-horizon mapping after candidate selection."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import yaml
from bf_tap.artifacts import atomic_write_json, file_sha256, file_identities, verify_file_identities
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.final_lifecycle import sample_metadata
from bf_tap.submission import validate_submission


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opt15-run',required=True,type=Path)
    p.add_argument('--challenger',type=Path)
    p.add_argument('--data-config',default='configs/data.local.yaml')
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    status=json.loads((args.opt15_run/'final_status.json').read_text())
    if status.get('G0')!='PASS_OPT15':raise ContractError('complete development selection required first')
    args.output.mkdir(parents=True,exist_ok=False)
    if args.challenger:
        release=json.loads((args.challenger/'final_status.json').read_text())
        if release.get('status')!='READY_CHALLENGER':raise ContractError('challenger must pass release validation')
        bundle=args.challenger/'bundle';candidate=release['candidate'];cutoff=pd.Timestamp(release['cutoff'])
        command=[sys.executable,'-m','bf_tap.optimization.component_release_v5','predict']
    else:
        if status['selected_STAGE_A'] is not None:raise ContractError('check the selected challenger after OPT16')
        active=load_yaml('configs/optimization_v0_4/active_release.yaml')
        bundle=Path(active['bundle']);candidate='R2';cutoff=pd.Timestamp(active['fit_cutoff'])
        if file_sha256(bundle/'composite.json')!=active['composite_sha256']:raise ContractError('active R2 changed')
        command=[sys.executable,'scripts/optimization_v4_cold_predict.py']
    paths=load_yaml(args.data_config)['paths']
    allowed={k:paths[k] for k in ('test_b_samples','test_c_samples','operation_hourly','burden_change','data_dictionary')}
    identities=file_identities(allowed)
    cfg=args.output/'no_official_labels.yaml';cfg.write_text(yaml.safe_dump(dict(schema_version=1,paths=allowed)))
    rows=[]
    for stage in ('test_b','test_c'):
        samples=sample_metadata(allowed[stage+'_samples'])
        dest=args.output/f'{stage}_predictions.csv'
        subprocess.run([*command,'--bundle',str(bundle),'--data-config',str(cfg),'--stage',stage,'--output',str(dest)],check=True)
        pred=pd.read_csv(dest,dtype={'sample_id':str})
        ordered=samples[['sample_id']].astype(str).merge(pred,on='sample_id',validate='one_to_one',sort=False)
        validate_submission(ordered,samples.sample_id)
        if len(pred)!=len(samples):raise ContractError('extra stage prediction IDs')
        horizons=(samples.reference_time.dt.year-cutoff.year)*12+samples.reference_time.dt.month-cutoff.month+1
        rows.append(dict(stage=stage,candidate=candidate,rows=len(samples),reference_min=str(samples.reference_time.min()),
            reference_max=str(samples.reference_time.max()),fit_cutoff=str(cutoff),
            calendar_horizon_counts={str(k):int(v) for k,v in horizons.value_counts().sort_index().items()},
            predictions_sha256=file_sha256(dest),official_target_paths_present=False,new_fits=0,scored=False,zip_created=False))
        print(f'{stage}: {len(samples)} valid predictions; horizons={rows[-1]["calendar_horizon_counts"]}',flush=True)
    verify_file_identities(identities)
    atomic_write_json(args.output/'stage_mapping.json',dict(status='PASS_RUNNABILITY_ONLY',results=rows,
        input_identities=identities,selection_already_frozen=True,platform_gain_claimed=False))

if __name__=='__main__':main()
