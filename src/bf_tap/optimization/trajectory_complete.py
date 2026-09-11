"""Complete saved OPT-24 evidence after a documented cold timestamp-type repair.

No model or coefficient estimation. Original fits, predictions, source snapshots,
manifest and failure evidence remain immutable.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from ..artifacts import atomic_write_json,file_sha256,verify_file_identities
from ..exceptions import ContractError
from .component_export import PRED,read_json,forbid_fit
from .dual_ratio_common import frame,score,week_intervals,verify_manifest
from .structural_run import append_ledger
from .trajectory_candidate import CANDIDATE,acceptance


def canonical_history_time(series):
    """Same instants and exact Shanghai dtype as original bundle history loader."""
    return pd.to_datetime(series,utc=True).dt.tz_convert('Asia/Shanghai')


def verify_execution(source):
    source=Path(source);manifest=read_json(source/'manifest.json')
    repair_path=source/'cold_repair_manifest.json'
    if not repair_path.exists():
        verify_manifest(manifest)
        return manifest
    repair=read_json(repair_path)
    if file_sha256(source/'manifest.json')!=repair['original_manifest_sha256'] or file_sha256(source/'predictions_complete.json')!=repair['predictions_complete_sha256']:
        raise ContractError('original execution identity changed')
    if file_sha256(source/'failure.json')!=repair['preserved_failure_sha256'] or file_sha256(source/'fit_counts.json')!=repair['fit_counts_sha256']:
        raise ContractError('original failure or counts changed')
    verify_file_identities(repair['repair_sources'])
    replacements=repair['source_replacements']
    if set(replacements)!={'scripts/optimization_v11_cold_check.py'}:
        raise ContractError('cold repair must not replace training/features/candidate/gates')
    for path,identity in manifest['sources'].items():
        if path in replacements:
            original=replacements[path]['original']
            if original['sha256']!=identity['sha256']:
                raise ContractError('source snapshot differs from pre-fit registration')
            verify_file_identities({'original':original,'repaired':replacements[path]['repaired']})
        else:
            verify_file_identities({path:identity})
    for k in ('inputs','evidence'):
        verify_file_identities(manifest[k])
    if file_sha256(manifest['scope']['old_ledger'])!=manifest['old_ledger_sha256']:
        raise ContractError('original protected ledger changed')
    verify_file_identities(read_json(source/'predictions_complete.json')['predictions'])
    return manifest


def complete(source):
    from .trajectory_run import equal,diagnostic
    manifest=verify_execution(source);reg=manifest['registration']
    reg['origins']={int(k):v for k,v in reg['origins'].items()}
    counts=read_json(source/'fit_counts.json')
    if (counts['development_time_CatBoost_attempted']!=8 or counts['development_time_CatBoost_completed']!=8
            or counts['development_time_LAD']!=6 or counts['forbidden_fits']!=0):
        raise ContractError('completed registered 8+6 budget required')
    atomic_write_json(source/'completion_started.json',dict(
        repair_manifest_sha256=file_sha256(source/'cold_repair_manifest.json'),new_fits_authorized=0))
    append_ledger({**manifest['scope'],'authorization':'OPT24_cold_timestamp_representation_repair_'+file_sha256(source/'cold_repair_manifest.json')},source)
    subprocess.run([sys.executable,'scripts/optimization_v11_cold_check.py','--source',str(source)],check=True)
    cold=read_json(source/'cold_validation.json')
    if cold['status']!='PASS' or cold['attempted_target_fits'] or cold['LAD_fits']:
        raise ContractError('complete zero-fit cold validation required')
    verify_execution(source)
    append_ledger({**manifest['scope'],'authorization':'OPT24_scoring_after_all_frozen_predictions_and_repaired_cold_PASS'},source)
    counter={'attempted_target_fits':0};predictions={}
    with forbid_fit(counter):
        for month in reg['origins']:
            predictions[month]=frame(source/'predictions'/f'{month}_V5.csv')
        def provider(unit,cutoff,metadata,baselines):
            p=predictions[cutoff.month];p=p.loc[p.sample_id.isin(metadata.sample_id)]
            equal(p,baselines['V1'],[PRED[0]],tolerance=0.)
            return {CANDIDATE:p}
        metrics,summary,errors,_=score(source,reg,provider)
        gate=acceptance(metrics,summary,reg,True,True)
        atomic_write_json(source/'acceptance.json',gate)
        atomic_write_json(source/'week_bootstrap.json',week_intervals(errors,CANDIDATE,'V1',**reg['bootstrap']))
        paired={}
        for month,pred in predictions.items():
            parts=frame(source/'predictions'/f'{month}_inputs.csv').set_index('sample_id')
            p=pred.set_index('sample_id').sort_index();parts=parts.loc[p.index]
            original=frame(Path(reg['source_v8'])/'predictions'/f'{month}_V1.csv').set_index('sample_id').loc[p.index]
            coeff=read_json(source/'corrections'/f'{month}.json')
            paired[str(month)]=dict(alpha_T=coeff['alpha_T'],unusable_rate_rows=int((parts.pred_rate<=reg['rate_floor']).sum()),
                correction_delta_quantiles=dict(zip(('p05','median','p95'),np.quantile(p[PRED[1]]-parts[PRED[1]],[.05,.5,.95]).tolist())),
                prediction_delta_vs_V1_quantiles=dict(zip(('p05','median','p95'),np.quantile(p[PRED[1]]-original[PRED[1]],[.05,.5,.95]).tolist())))
        atomic_write_json(source/'paired_changes.json',paired)
        diagnostic(source,errors)
    verify_execution(source)
    atomic_write_json(source/'final_status.json',dict(G0='PASS_OPT24_DEVELOPMENT_AND_REPAIRED_COLD',G1=gate['status'],
        selected=gate['selected'],holdout_consumed=True,active_release_unchanged=True,
        final_stage='AUTHORIZED_IF_FULL_TEST_SUITE_PASSES' if gate['passed'] else 'NOT_AUTHORIZED_QUALITY_FAIL'))
    atomic_write_json(source/'completion.json',dict(status='PASS',original_fit_counts=counts,
        additional_CatBoost_fits=0,additional_LAD_fits=0,inference_fit_attempts=counter['attempted_target_fits'],
        repair_manifest_sha256=file_sha256(source/'cold_repair_manifest.json'),
        cold_validation_sha256=file_sha256(source/'cold_validation.json'),acceptance_sha256=file_sha256(source/'acceptance.json')))
    print(__import__('json').dumps(gate,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);complete(p.parse_args().source)
