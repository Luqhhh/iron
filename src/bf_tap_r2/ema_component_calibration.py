"""Matched cached EMA component calibration; no base-model training."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from .component_regularization import ComponentRegressor
from .component_regularization_audit import verify_saved
from .data import TARGETS
from .ema_average_span import candidate_column, old_cache
from .ema_evaluation_diagnostics import sha, write_new
from .ema_fusion_selection import runtime, verify_files
from .ema_reference_artifacts import audit_witness, save_witness
from .ema_training_scale import require_memory
from .q75_combination_review import score
from .q75_pressure_calibration import apply_calibration, learn_calibration, partition
from .v3_4_bags import group_safe_inner_folds
from .v5_library import fold_vector, load_v5_training_frame
from .v5_spec import load_v5_spec
from .v7_periodic import digest
from .v49_run import metric_detail

SPEC = 'configs/ema_component_calibration/SPEC.json'
PROTOCOL = 'docs/ema_component_calibration/PREREGISTRATION.md'
ORDER = ('GLOBAL', 'PRESSURE')
WORK = Path(__file__).resolve().parents[2]


def validate_scope(spec):
    expected = dict(split_seeds=[42, 3407], folds=5, families=list(ORDER),
        calibration_cache='local/runs/q75-error-relocation-20261001/calibration-development-r1',
        old_development='local/runs/strong-component-regularization/development-r2',
        replacement_weight=.75, numerical_threads=1, workers=1, monitor_seconds=600,
        maximum_runtime_seconds=None, automatic_retries=False, max_worker_rss_mib=1536,
        cold_predict_atol=.0005, new_model_fits=0, optimizer_runs=0,
        correction_fit_calls=80, new_confirmation_seeds=0, full_data_fits=0,
        packages=0, desktop_writes=0, uploads=0,
        correction=dict(cv_seed=961048, minimum_bin_rows=30, gamma_grid=[0, .25, .5, 1]))
    main = Path(spec['main_root'])
    expected['previous_terminals'] = [str(main/f'local/runs/{name}/development-r1/terminal-verification.json')
        for name in ('ema-retraining-initialization-20261002', 'ema-retraining-step-matching-20261002')]
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Unregistered matched component scope')
    if spec['reference']['candidate'] != 'EMA_TIME_Q75':
        raise ValueError('A new incumbent requires a matched reference implementation')


def choose_candidate(gains, matched_gains):
    if set(gains) != set(ORDER) or set(matched_gains) != set(ORDER):
        raise ValueError('Complete frozen candidate pool required')
    for records in (gains, matched_gains):
        if any(set(r) != {'42', '3407'} or not all(math.isfinite(x) for x in r.values())
                for r in records.values()):
            raise ValueError('Complete finite within-split gains required')
    eligible = [k for k in ORDER if min(gains[k].values()) > 0 and min(matched_gains[k].values()) > 0]
    if not eligible:
        return None
    best = max(np.mean(list(gains[k].values())) for k in eligible)
    return next(k for k in ORDER if k in eligible and np.mean(list(gains[k].values())) >= best-1e-12)


def component_heads(calibration, actual, cal_prediction, query, prediction, fitting, q75, old_ema, settings):
    """Only C labels fit heads; correction is applied to the same EMA component."""
    if any(t in calibration or t in query for t in TARGETS):
        raise ValueError('Calibration and application inputs must be label-free')
    cuts = np.quantile(fitting.total_press_diff.to_numpy(float), [.25, .5, .75])
    columns = {'UNCORRECTED': candidate_column(q75, old_ema, prediction)}
    fitted = {}
    import bf_tap_r2.q75_pressure_calibration as calibration_module
    calls = []
    original = calibration_module.fit_offsets
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    with patch.object(calibration_module, 'fit_offsets', counted):
        for family in ORDER:
            fitted[family] = learn_calibration(calibration, actual, cal_prediction, cuts, family, settings)
            corrected = apply_calibration(query, prediction, fitted[family])
            columns[family] = candidate_column(q75, old_ema, corrected)
    if len(calls) != 8:
        raise ValueError('Declared correction fit budget differs')
    return columns, fitted


@contextmanager
def no_model_training():
    import torch
    def forbidden(*args, **kwargs):
        raise ValueError('Cached component experiment attempted model training')
    with patch.object(ComponentRegressor, 'fit', forbidden), patch.object(ComponentRegressor, '_train', forbidden), \
            patch.object(ComponentRegressor, '_initialize', forbidden), patch.object(torch.optim.AdamW, 'step', forbidden):
        yield


def source_files():
    paths = list((WORK/'src').rglob('*.py'))
    paths += [WORK/SPEC, WORK/PROTOCOL, WORK/'tests/test_ema_component_calibration.py',
        WORK/'scripts/audit_ema_component_calibration.py', WORK/'pyproject.toml', WORK/'uv.lock']
    return {str(p.resolve()): sha(p) for p in paths}


def previous_success(spec):
    bindings = {}
    for filename in spec['previous_terminals']:
        p = Path(filename)
        receipt = json.loads(p.read_text())
        if receipt.get('status') != 'passed' or receipt.get('actual_exit_codes') != [0, 0]:
            raise ValueError('Both preceding phases need successful actual terminals')
        if sha(p.parent/'report.json') != receipt['report_sha256']:
            raise ValueError('Previous report identity differs')
        if sha(p.parent/'independent-score.json') != receipt['independent_score_sha256']:
            raise ValueError('Previous independent audit differs')
        bindings[str(p)] = sha(p)
    return bindings


def prepare(output, tests_receipt):
    spec = json.loads((WORK/SPEC).read_text()); validate_scope(spec)
    main = Path(spec['main_root']); out = Path(output).resolve()
    if out.exists() or not out.is_relative_to(main/'local/runs'):
        raise ValueError('Fresh private output required')
    current = json.loads((main/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if any(current.get(k) != v for k, v in spec['reference'].items()):
        raise ValueError('Update unfrozen protocol to latest reference before freezing')
    prior = previous_success(spec); runtime(spec); require_memory(spec)
    files = source_files(); tested = json.loads(Path(tests_receipt).read_text())
    if tested.get('status') != 'passed' or tested.get('sources') != files or tested.get('python') != sys.version.split()[0]:
        raise ValueError('Exact-source locked Python3.12 checks required')
    if subprocess.check_output(['git', 'status', '--porcelain', '--',
            *[str(Path(p).relative_to(WORK)) for p in files]], cwd=WORK, text=True).strip():
        raise ValueError('Commit tested scientific source before freeze')
    files[str(Path(tests_receipt).resolve())] = sha(tests_receipt)
    grant = main/spec['standing_grant']
    if sha(grant) != spec['standing_grant_sha256']:
        raise ValueError('Standing authorization identity differs')
    files[str(grant)] = sha(grant)
    old = main/spec['calibration_cache']; original = json.loads((old/'manifest.json').read_text())
    audit = json.loads((old/'audit.json').read_text())
    if audit['status'] != 'passed' or audit['manifest_sha256'] != sha(old/'manifest.json'):
        raise ValueError('Original calibration audit differs')
    for name in ('component_regularization.py', 'v12_joint.py', 'v7_periodic.py', 'v3_6_networks.py'):
        relative = 'src/bf_tap_r2/'+name
        if sha(WORK/relative) != original['sources'][relative]:
            raise ValueError('Original scientific dependency changed')
    files.update({str(main/p): h for p, h in original['spec']['inputs'].items()})
    for p in (old/'manifest.json', old/'audit.json', main/'configs/candidate_tiers.yaml',
              main/'configs/round2_v5/SPEC.yaml', main/'configs/data.local.yaml'):
        files[str(p)] = sha(p)
    for seed in spec['split_seeds']:
        for fold in range(5):
            p = old/f's{seed}-f{fold}'
            complete = json.loads((p/'complete.json').read_text())
            files.update({str(p/name): h for name, h in complete['hashes'].items()})
            files[str(p/'complete.json')] = sha(p/'complete.json')
    files.update(prior); verify_files(files)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(spec=spec, output=str(out), files=files, previous_terminals=prior,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK, text=True).strip(),
        original_calibration_manifest_sha256=sha(old/'manifest.json'))
    manifest['identity'] = digest(manifest)
    write_new(out/'manifest.json', manifest)


def run(output):
    out = Path(output).resolve(); manifest = json.loads((out/'manifest.json').read_text())
    core = dict(manifest); identity = core.pop('identity')
    spec = manifest['spec']
    if (digest(core) != identity or out != Path(manifest['output'])
            or not out.is_relative_to(Path(spec['main_root'])/'local/runs')
            or spec != json.loads((WORK/SPEC).read_text())):
        raise ValueError('Frozen run location or metadata differs')
    validate_scope(spec); verify_files(manifest['files']); runtime(spec)
    main = Path(spec['main_root']); old = main/spec['calibration_cache']
    original = json.loads((old/'manifest.json').read_text())
    frame = load_v5_training_frame(main); records = {}; all_gains = {k: {} for k in ORDER}; paired = {k: {} for k in ORDER}
    with no_model_training():
        for seed in spec['split_seeds']:
            fv = fold_vector(main, frame, seed, load_v5_spec(main))
            vectors = {k: np.full(len(frame), np.nan) for k in ('q75', 'iron', 'UNCORRECTED', *ORDER)}
            for fold in range(5):
                require_memory(spec)
                unit = out/f's{seed}-f{fold}'; unit.mkdir(exist_ok=False)
                fitting, calibration, training, query = partition(frame, seed, fold, dict(original['spec'], main_root=str(main)))
                inner = np.asarray(group_safe_inner_folds(fitting, seed=original['spec']['training']['inner_seed'])['fold'])
                inner_fit, inner_cal = fitting.loc[inner != 0], fitting.loc[inner == 0]
                cached = old/unit.name
                selector = verify_saved(cached/'selection.pt', inner_fit, inner_fit[['tap_time_len']].to_numpy(),
                    'EMA', original['spec']['training'], original['spec']['mechanisms'], inner_cal)
                model = verify_saved(cached/'refit.pt', fitting, fitting[['tap_time_len']].to_numpy(),
                    'EMA', original['spec']['training'], original['spec']['mechanisms'], expected_epoch=selector.saved['trace']['selected_epoch'])
                reference = old_cache(dict(main_root=str(main), old_development=spec['old_development']), seed, fold, training, query)
                cal_query = calibration.drop(columns=list(TARGETS)); cp = model.predict(cal_query)[:, 0]; qp = model.predict(query)[:, 0]
                with np.load(cached/'predictions.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['calibration_ids'], cal_query.sample_id.to_numpy(str))
                    np.testing.assert_array_equal(saved['ema_calibration'], cp)
                witnesses = {}
                for name, q, p in (('calibration', cal_query, cp), ('outer', query, qp)):
                    witnesses[name] = save_witness(model, q, p[:, None], unit/name,
                        identity=dict(source_directory=str(cached.resolve()), split_seed=seed, fold=fold, trial_id='EMA', fit_call_id='refit'),
                        training_ids=fitting.sample_id.tolist(), source_hashes=source_files(),
                        full_batch_atol=0., row_atol=spec['cold_predict_atol'], fit_metadata=model.saved['trace'])
                    subprocess.run([sys.executable, '-m', 'bf_tap_r2.ema_component_calibration', '--cold', '--directory', str(unit/name),
                        '--receipt-sha256', witnesses[name]], check=True, stdout=subprocess.DEVNULL)
                columns, fitted = component_heads(cal_query, calibration.tap_time_len.to_numpy(), cp, query, qp,
                    fitting, reference['q75'], reference['ema'], spec['correction'])
                held = fv == fold
                for key in ('q75', 'iron'): vectors[key][held] = reference[key]
                for key, value in columns.items(): vectors[key][held] = value
                with (unit/'predictions.npz').open('xb') as stream:
                    np.savez_compressed(stream, query_ids=query.sample_id.to_numpy(str), ema_F=qp,
                        q75=reference['q75'], old_ema=reference['ema'], pressure=query.total_press_diff.to_numpy(float), **columns)
                write_new(unit/'complete.json', dict(witnesses=witnesses, fitted=fitted, correction_fit_calls=8,
                    new_model_fits=0, optimizer_runs=0, original_refit_sha256=sha(cached/'refit.pt'),
                    head_source_directory=str(out), split_seed=seed, fold=fold,
                    predictions_sha256=sha(unit/'predictions.npz')))
            if any(not np.isfinite(v).all() for v in vectors.values()): raise ValueError('Incomplete same-split OOF')
            actual = frame[list(TARGETS)].to_numpy(); baseline = score(actual, vectors['iron'], vectors['q75'])
            control = score(actual, vectors['iron'], vectors['UNCORRECTED'])
            record = {'control_gain': control-baseline, 'gains': {}, 'matched_gains': {}, 'metrics': {}}
            for family in ORDER:
                value = score(actual, vectors['iron'], vectors[family])
                all_gains[family][str(seed)] = record['gains'][family] = value-baseline
                paired[family][str(seed)] = record['matched_gains'][family] = value-control
                record['metrics'][family] = metric_detail(actual[:, 1], vectors[family], fv, frame.spout_no.to_numpy())
            with (out/f'oof-s{seed}.npz').open('xb') as stream:
                np.savez_compressed(stream, actual=actual, query_ids=frame.sample_id.to_numpy(str), folds=fv,
                    spouts=frame.spout_no.to_numpy(), **vectors)
            records[str(seed)] = record
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak > spec['max_worker_rss_mib']: raise ValueError('RSS gate failed')
    import yaml
    from .candidate_tiers import classify_candidates
    metrics = {'tap_time_len': {k: {} for k in ('Q75', *ORDER)}}
    for seed in spec['split_seeds']:
        with np.load(out/f'oof-s{seed}.npz', allow_pickle=False) as saved:
            for name in metrics['tap_time_len']:
                p = saved['q75'] if name == 'Q75' else saved[name]
                metrics['tap_time_len'][name][str(seed)] = metric_detail(saved['actual'][:, 1], p, saved['folds'], saved['spouts'])
    tier_spec = dict(split_seeds=spec['split_seeds'], folds=5, candidates={'tap_time_len': list(ORDER)},
        tie_preference_by_target={'tap_time_len': list(ORDER)}, reference_by_target={'tap_time_len': 'Q75'})
    tiers = classify_candidates(tier_spec, metrics, yaml.safe_load((main/'configs/candidate_tiers.yaml').read_text()))
    report = dict(status='complete_development_not_formal_promotion', records=records,
        G0='cold_coverage_passed_scalar_audit_pending', G1='two_development_splits_not_formal_promotion',
        selected_for_confirmation=choose_candidate(all_gains, paired), candidate_tiers=tiers, peak_rss_mib=peak,
        correction_fit_calls=80, new_model_fits=0, optimizer_runs=0, packages=0, uploads=0,
        manifest_sha256=sha(out/'manifest.json'), vector_sha256={str(s): sha(out/f'oof-s{s}.npz') for s in spec['split_seeds']})
    verify_files(manifest['files']); write_new(out/'report.json', report)
    subprocess.run([sys.executable, str(WORK/'scripts/audit_ema_component_calibration.py'), '--output', str(out)], check=True)
    print(json.dumps(report), flush=True)


def main():
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    p = argparse.ArgumentParser(); p.add_argument('--output'); p.add_argument('--tests-receipt')
    p.add_argument('--directory'); p.add_argument('--receipt-sha256')
    mode = p.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'run', 'cold'): mode.add_argument('--'+name, action='store_true')
    args = p.parse_args()
    if args.cold:
        spec = json.loads((WORK/SPEC).read_text()); runtime(spec)
        result = audit_witness(args.directory, args.receipt_sha256)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if peak > spec['max_worker_rss_mib']: raise ValueError('Cold process RSS gate failed')
        write_new(Path(args.directory)/'cold-rss.json', dict(status='passed', peak_rss_mib=peak,
            receipt_sha256=args.receipt_sha256, new_model_fits=0, optimizer_runs=0))
        print(json.dumps(result), flush=True)
    elif args.prepare: prepare(args.output, args.tests_receipt)
    else: run(args.output)


if __name__ == '__main__':
    main()
