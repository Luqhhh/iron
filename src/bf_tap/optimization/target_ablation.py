"""OPT-21 diagnostic-only target ablation: no model/coefficient fits or releases."""
import argparse
from pathlib import Path
import numpy as np
from ..artifacts import atomic_write_json
from ..exceptions import ContractError
from .component_export import PRED,forbid_fit,read_json
from .dual_ratio_common import registry,freeze,verify_manifest,score,week_intervals


def reconstruct(parts):
    a=parts['R2'].set_index('sample_id').sort_index();b=parts['V1'].set_index('sample_id').sort_index()
    if not a.index.equals(b.index):raise ContractError('ablation IDs differ')
    w=a.copy();w[PRED[1]]=b[PRED[1]]
    if not np.array_equal(w[PRED[0]],a[PRED[0]]) or not np.array_equal(w[PRED[1]],b[PRED[1]]):raise ContractError('ablation endpoint mismatch')
    return w.reset_index()


def run(output):
    output.mkdir(parents=True,exist_ok=False);counter={'attempted_target_fits':0}
    try:
        reg=registry();manifest=freeze(output,'OPT21_archived_label_target_ablation_zero_fit')
        with forbid_fit(counter):
            metrics,summary,errors,_=score(output,reg,lambda unit,cutoff,metadata,parts:{'W0_TIME_ABLATION':reconstruct(parts)})
            intervals={f'{c}_vs_{r}':week_intervals(errors,c,r,**reg['bootstrap']) for c,r in [('V1','R2'),('W0_TIME_ABLATION','R2'),('W0_TIME_ABLATION','V1')]}
        old=read_json(Path(reg['source_v8'])/'summary.json')
        for c,key in [('R2','U0'),('V1','U1')]:
            if abs(summary[c]['J']-old[key]['J'])>1e-12:raise ContractError('archived summary failed reconstruction')
        gain=summary['R2']['J']-summary['V1']['J'];ablation_gain=summary['R2']['J']-summary['W0_TIME_ABLATION']['J']
        atomic_write_json(output/'diagnosis.json',dict(V1_J_improvement_vs_R2=gain,W0_J_improvement_vs_R2=ablation_gain,
            retained_gain_fraction=ablation_gain/gain if gain else None,diagnostic_only=True,eligible_for_platform=False))
        atomic_write_json(output/'week_bootstrap.json',intervals);verify_manifest(manifest)
        atomic_write_json(output/'final_status.json',dict(G0='PASS_OPT21',scope='DIAGNOSTIC_ONLY',**counter,completed_target_fits=0,coefficient_fits=0,
            units=20,platform_package=False,active_V1_and_R2_unchanged=True))
        print(read_json(output/'diagnosis.json'),flush=True)
    except Exception as exc:
        atomic_write_json(output/'failure.json',dict(error=str(exc),**counter));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
