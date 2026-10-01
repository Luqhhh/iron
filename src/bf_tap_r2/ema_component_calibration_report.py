"""Report-only closure of retained matched-component arrays; never fits."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

from .candidate_tiers import classify_candidates
from .ema_component_calibration import ORDER, choose_candidate
from .ema_evaluation_diagnostics import sha, write_new
from .ema_fusion_selection import verify_files
from .q75_combination_review import score
from .v49_run import metric_detail


def build_report(arrays, policy):
    records = {}; gains = {k: {} for k in ORDER}; matched = {k: {} for k in ORDER}
    metrics = {'tap_time_len': {k: {} for k in ('Q75', *ORDER)}}
    if set(arrays) != {'42', '3407'}: raise ValueError('Complete frozen split pool required')
    for seed, values in arrays.items():
        actual = values['actual']; baseline = score(actual, values['iron'], values['q75'])
        control = score(actual, values['iron'], values['UNCORRECTED'])
        record = dict(control_gain=control-baseline, gains={}, matched_gains={}, metrics={})
        metrics['tap_time_len']['Q75'][seed] = metric_detail(actual[:, 1], values['q75'], values['folds'], values['spouts'])
        for family in ORDER:
            value = score(actual, values['iron'], values[family])
            gains[family][seed] = record['gains'][family] = value-baseline
            matched[family][seed] = record['matched_gains'][family] = value-control
            record['metrics'][family] = metrics['tap_time_len'][family][seed] = metric_detail(
                actual[:, 1], values[family], values['folds'], values['spouts'])
        records[seed] = record
    spec = dict(split_seeds=[42, 3407], folds=5, candidates={'tap_time_len': list(ORDER)},
        tie_preference_by_target={'tap_time_len': list(ORDER)}, reference_by_target={'tap_time_len': 'Q75'})
    return dict(status='report_only_recovery_not_formal_promotion', records=records,
        selected_for_confirmation=choose_candidate(gains, matched),
        candidate_tiers=classify_candidates(metrics, spec, policy),
        G0='retained_cold_coverage_bound_independent_scalar_pending',
        G1='two_complete_development_splits_not_formal_promotion',
        correction_fit_calls=80, recovery_correction_fit_calls=0, new_model_fits=0,
        optimizer_runs=0, new_confirmation_seeds=0, full_data_fits=0, packages=0, uploads=0)


def recover(failed_run, output):
    old = Path(failed_run).resolve(); out = Path(output).resolve()
    manifest = json.loads((old/'manifest.json').read_text()); main = Path(manifest['spec']['main_root'])
    terminal = json.loads((old/'terminal-verification.json').read_text())
    if terminal['status'] != 'failed' or terminal['actual_exit_codes'] != [1]:
        raise ValueError('Original failed actual terminal required')
    if out.exists() or not out.is_relative_to(main/'local/runs') or out == old:
        raise ValueError('New private recovery namespace required')
    verify_files(manifest['files'])
    units = [old/f's{seed}-f{fold}' for seed in (42, 3407) for fold in range(5)]
    fits = 0; witnesses = 0; retained = {}
    for unit in units:
        receipt = json.loads((unit/'complete.json').read_text()); fits += receipt['correction_fit_calls']
        if receipt['new_model_fits'] or receipt['optimizer_runs']: raise ValueError('Unexpected model fit')
        if sha(unit/'predictions.npz') != receipt['predictions_sha256']: raise ValueError('Retained columns changed')
        retained[str(unit/'complete.json')] = sha(unit/'complete.json')
        for name, expected in receipt['witnesses'].items():
            p = unit/name; cold = json.loads((p/'cold-audit.json').read_text())
            if sha(p/'complete.json') != expected or cold['status'] != 'passed' or cold['receipt_sha256'] != expected:
                raise ValueError('Retained cold binding differs')
            witnesses += 1; retained[str(p/'cold-audit.json')] = sha(p/'cold-audit.json')
    if fits != 80 or witnesses != 20: raise ValueError('Original complete budget required')
    arrays = {}
    for seed in (42, 3407):
        p = old/f'oof-s{seed}.npz'; retained[str(p)] = sha(p)
        with np.load(p, allow_pickle=False) as values: arrays[str(seed)] = {k: values[k].copy() for k in values.files}
    report = build_report(arrays, yaml.safe_load((main/'configs/candidate_tiers.yaml').read_text()))
    out.mkdir(parents=True, exist_ok=False)
    # Read-only aliases retain original physical cached fit/prediction identities.
    shutil.copyfile(old/'manifest.json', out/'manifest.json')
    for unit in units: (out/unit.name).symlink_to(unit, target_is_directory=True)
    for seed in (42, 3407): (out/f'oof-s{seed}.npz').symlink_to(old/f'oof-s{seed}.npz')
    work = Path(__file__).resolve().parents[2]
    write_new(out/'recovery-manifest.json', dict(original_run=str(old), recovery_output=str(out),
        original_manifest_sha256=sha(old/'manifest.json'), original_failed_terminal_sha256=sha(old/'terminal-verification.json'),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=work, text=True).strip(),
        recovery_source_sha256=sha(Path(__file__)), retained_inputs=retained,
        model_fits=0, optimizer_runs=0, correction_fit_calls=0, original_failure_unchanged=True))
    report.update(manifest_sha256=sha(out/'manifest.json'),
        vector_sha256={str(seed): sha(out/f'oof-s{seed}.npz') for seed in (42, 3407)},
        peak_rss_mib=max(e['peak_rss_mib'] for e in terminal['events']))
    write_new(out/'report.json', report)
    audit = Path(next(p for p in manifest['files'] if p.endswith('/scripts/audit_ema_component_calibration.py')))
    subprocess.run([sys.executable, str(audit), '--output', str(out)], check=True)
    verify_files(manifest['files']); verify_files(retained)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--failed-run', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); recover(args.failed_run, args.output)


if __name__ == '__main__': main()
