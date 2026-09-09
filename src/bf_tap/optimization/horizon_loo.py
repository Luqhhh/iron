"""OPT-18 target/horizon selection excluding each evaluation origin."""
import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from ..artifacts import atomic_write_json,file_identities,verify_file_identities,file_sha256
from ..config import load_yaml
from ..exceptions import ContractError
from ..metrics import score_predictions
from .component_export import META,PRED,read_json,units,forbid_fit
from .component_ablation import validate_frame
from .horizon_router import TARGETS,route
from .horizon_evaluation import append_access
from .validation import aggregate_grid,error_contributions


def choose(metrics,target,horizon,excluded_origin=None):
    entries=[v for v in metrics.values() if v['horizon']==horizon and v['origin_id']!=excluded_origin]
    if len(entries)<2:raise ContractError('at least two other origins required')
    means={c:float(np.mean([v['candidates'][c]['overall'][target]['wmape'] for v in entries])) for c in ('P2','R2')}
    return ('P2' if means['P2']<means['R2'] else 'R2'),means


def run(source,challenger,output):
    if not read_json(source/'acceptance.json')['passed'] or read_json(challenger/'final_status.json')['status']!='READY_CHALLENGER':raise ContractError('S1 must pass and challenger must be ready first')
    output.mkdir(parents=True,exist_ok=False);reg=load_yaml('configs/optimization_v0_6/experiment.yaml')
    paths=[source/'metrics.json',source/'acceptance.json',challenger/'final_status.json',Path(__file__),Path('configs/optimization_v0_6/experiment.yaml')]
    for unit,*_ in units(reg):
        for c in ('P2','R2'):paths.append(source/'units'/unit/f'{c}_errors.csv')
    manifest=dict(inputs=file_identities({str(p):p for p in paths}),registration=reg,holdout_consumed=True,
        statistical_identity='cross_origin_selection_on_consumed_development_not_independent_confirmation')
    atomic_write_json(output/'manifest.json',manifest);append_access(output,'OPT18_leave_one_origin_out_existing_predictions_scoring')
    base=read_json(source/'metrics.json');metrics=copy.deepcopy(base);decisions=[];rows=[];counter={'attempted_target_fits':0}
    with forbid_fit(counter):
        for unit,cutoff,horizon,*_ in units(reg):
            actual=validate_frame(pd.read_csv(source/'units'/unit/'R2_errors.csv',dtype={'sample_id':str}),labeled=True)
            parts={c:validate_frame(pd.read_csv(source/'units'/unit/f'{c}_errors.csv',dtype={'sample_id':str}),labeled=True)[['sample_id',*PRED]] for c in ('P2','R2')}
            origin=metrics[unit]['origin_id'];mapping={t:{} for t in TARGETS}
            for h in (1,2,3,4):
                for target,key in zip(TARGETS,('iron','time')):
                    chosen,means=choose(base,key,h,origin);mapping[target][h]=chosen
                    decisions.append(dict(unit=unit,target=target,horizon=h,excluded_origin=origin,choice=chosen,training_origin_means=means))
            pred=route(actual[META],cutoff,parts,mapping)
            metrics[unit]['candidates']['S2_LOO']={'overall':score_predictions(actual,pred)}
            err=error_contributions(actual,pred);err['unit']=unit;err['candidate']='S2_LOO';err['horizon']=horizon;err['origin']=origin;rows.append(err)
        summary=aggregate_grid(metrics);s=summary['S2_LOO'];r=summary['S1'];p=reg['opt18']
        target_deltas={h:{t:s['horizons'][h][t+'_mean_wmape']-r['horizons'][h][t+'_mean_wmape'] for t in ('iron','time')} for h in ('H1','H2','H3','H4')}
        J_delta=s['J']-r['J'];H1_delta=s['horizons']['H1']['mean_loss']-r['horizons']['H1']['mean_loss']
        checks=dict(J_improvement=-J_delta>=p['J_min_improvement_vs_S1'],H1_no_regression=H1_delta<=p['H1_max_regression_vs_S1'],target_horizon=max(v for x in target_deltas.values() for v in x.values())<=p['target_horizon_max_regression_vs_S1'])
        gate=dict(passed=all(checks.values()),checks=checks,J=s['J'],J_delta_vs_S1=J_delta,H1_delta_vs_S1=H1_delta,target_horizon_deltas=target_deltas,
            DEV_deltas={u:metrics[u]['candidates']['S2_LOO']['overall']['loss']-metrics[u]['candidates']['S1']['overall']['loss'] for u in ('DEV_LONG','DEV_SHORT')})
        atomic_write_json(output/'acceptance.json',gate);atomic_write_json(output/'decisions.json',decisions);atomic_write_json(output/'metrics.json',metrics);atomic_write_json(output/'summary.json',summary)
        pd.concat(rows,ignore_index=True).to_csv(output/'routed_errors.csv',index=False)
        if gate['passed']:
            mapping={t:{h:choose(base,key,h)[0] for h in (1,2,3,4)} for t,key in zip(TARGETS,('iron','time'))}
            atomic_write_json(output/'final_mapping.json',dict(mapping=mapping,selection='full_development_after_LOO_gate',not_independent_performance_estimate=True))
    verify_file_identities(manifest['inputs'])
    atomic_write_json(output/'final_status.json',dict(G0='PASS_OPT18',G1='PASS' if gate['passed'] else 'FAIL',**counter,completed_target_fits=0,selected='S2' if gate['passed'] else 'S1',scope=manifest['statistical_identity']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--challenger',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.challenger,a.output)
