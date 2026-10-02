"""One explicit EMA three-training-seed exploration and desktop handoff."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
sys.path.insert(0, str(WORK / 'scripts'))
import numpy as np
from bf_tap_r2.ema_reference_artifacts import cold_only, sha, write_new
from bf_tap_r2.ema_reference_ledger import binding_sources, reference_bindings
from bf_tap_r2.ema_span_confirmation import partition
from bf_tap_r2.ema_training_scale import require_memory
from bf_tap_r2.sam_ema_development import runtime
from bf_tap_r2.v7_periodic import digest
from ema_short_release import release_payload, zip_payload
from observe_ema_dropout_development import wait_actual
from sam_ema_release import current_reference, official_frames, peak, verify_files

SPEC = 'configs/ema_mean3_release/SPEC.json'
read = lambda p: json.loads(Path(p).read_text())


def validate_scope(spec, science):
    expected = dict(candidate='EMA_MEAN3_FULL_Q75', target='tap_time_len',
        training_seeds=[42, 1042, 2042], new_training_seeds=[1042, 2042], reuse_training_seeds=[42],
        replacement_weight=.75, full_training_procedures=2, torch_optimizer=4,
        new_native_states=4, reused_native_states=2, new_CV=0, new_confirmation_seeds=0,
        packages=1, desktop_copies=1, agent_uploads=0, workers=1, numerical_threads=1,
        torch_interop_threads=1, monitor_seconds=600, max_worker_rss_mib=1536,
        cold_predict_atol=.0005, maximum_runtime_seconds=None, automatic_scientific_retries=False,
        formal_promotion=False, full_scope_identity_seed=-1, full_scope_identity_fold=-1,
        authorization_mode='explicit_user_single_manual_exploration_and_desktop_delivery',
        pause_after_delivery=True)
    if any(spec.get(k) != v or type(spec.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError('Unregistered EMA mean3 release scope')
    if any(spec[k] != science[k] for k in ('training', 'mechanisms', 'runtime_versions', 'training_seeds')):
        raise ValueError('Completed scientific training recipe changed')


def explicit_authority(spec):
    p = Path(spec['explicit_grant'])
    if sha(p) != spec['explicit_grant_sha256']:
        raise ValueError('Explicit desktop release authorization changed')
    grant = read(p)
    scope = {k: spec[k] for k in ('candidate', 'training_seeds', 'new_training_seeds',
        'reuse_training_seeds', 'full_training_procedures', 'torch_optimizer', 'packages',
        'desktop_copies', 'new_CV', 'new_confirmation_seeds', 'agent_uploads',
        'replacement_weight', 'formal_promotion')}
    if (grant.get('source') != 'explicit_user_task' or grant.get('user_authorized') is not True
            or grant.get('user_request') != 'EMA_MEAN3 写桌面' or grant.get('scope') != scope):
        raise ValueError('Explicit fixed EMA mean3 task required')
    return {str(p): sha(p)}


def sources(work):
    work = Path(work)
    paths = list((work / 'src').rglob('*.py')) + [work / p for p in (
        SPEC, 'configs/ema_retraining_validation/initialization.json',
        'docs/ema_mean3_release/PREREGISTRATION.md', 'scripts/ema_mean3_release.py',
        'scripts/sam_ema_release.py', 'scripts/ema_short_release.py',
        'scripts/observe_ema_dropout_development.py', 'tests/test_ema_mean3_release.py',
        'tests/test_ema_span_confirmation_models.py', 'tests/test_component_regularization.py',
        'tests/test_ema_short_release.py', 'uv.lock', 'pyproject.toml')]
    return {str(p.resolve()): sha(p) for p in sorted(set(paths))}


def development_evidence(spec):
    verify_files(spec['scientific_evidence'])
    dev = Path(spec['scientific_development'])
    report, audit, terminal = [read(dev / p) for p in
        ('report.json', 'independent-score.json', 'terminal-verification.json')]
    if (audit['status'] != 'passed' or terminal['status'] != 'passed'
            or terminal['actual_exit_codes'] != [0, 0] or terminal['native_optimizer_runs'] != 80
            or terminal['cold_states'] != 120 or terminal['report_sha256'] != sha(dev / 'report.json')
            or terminal['independent_score_sha256'] != sha(dev / 'independent-score.json')
            or terminal['manifest_sha256'] != sha(dev / 'manifest.json')
            or audit['report_sha256'] != sha(dev / 'report.json')
            or report['candidate'] != 'EMA_MEAN3' or report['confirmation_eligible']
            or report['formal_promotion'] or set(report['records']) != {'42', '3407'}
            or not all(r['gains_vs_q75']['EMA_MEAN3'] > 0 for r in report['records'].values())
            or not report['records']['42']['equal_mean_ema_minus_equal_mean_base'] > 0
            or not report['records']['3407']['equal_mean_ema_minus_equal_mean_base'] < 0):
        raise ValueError('Complete audited development and original failed paired gate required')
    return read(dev / 'manifest.json')['files']


def prepare(work, out, tests):
    work, out = Path(work).resolve(), Path(out).resolve()
    spec = read(work / SPEC)
    validate_scope(spec, read(work / spec['scientific_config']))
    current_reference(spec)
    if out.exists() or not out.is_relative_to(Path(spec['main_root']) / 'local/runs'):
        raise ValueError('Fresh private run required')
    if sha(work / spec['scientific_config']) != spec['scientific_config_sha256']:
        raise ValueError('Science configuration changed')
    source = sources(work)
    receipt = read(tests)
    if (receipt['status'] != 'passed' or receipt['sources'] != source or receipt['skipped'] != 0
            or receipt['full_suite'] is not True or receipt['python'] != sys.version.split()[0]):
        raise ValueError('Exact-source locked Python3.12 full tests required')
    if subprocess.check_output(['git', 'status', '--porcelain', '--', *source], cwd=work, text=True).strip():
        raise ValueError('Commit tested source before freeze')
    main = Path(spec['main_root'])
    grant = main / spec['standing_grant']
    if sha(grant) != spec['standing_grant_sha256']:
        raise ValueError('Standing grant changed')
    inputs = development_evidence(spec) | spec['scientific_evidence'] | spec['old_inputs'] | explicit_authority(spec)
    inputs[str(grant)] = sha(grant)
    inputs[str(main / 'configs/protection.yaml')] = sha(main / 'configs/protection.yaml')
    for stage in ('train', 'test'):
        for kind in ('samples', 'features'):
            p = main / f'复赛_{stage}/{stage}_{kind}.csv'
            inputs[str(p)] = sha(p)
    p = main / '复赛_test/result_template.csv'
    inputs[str(p)] = sha(p)
    verify_files(inputs)
    # Exact original trainer dependencies are reused; only the training seed changes.
    for name in ('component_regularization.py', 'v12_joint.py', 'v3_6_networks.py', 'v7_periodic.py', 'data.py'):
        matches = [h for p, h in spec['old_inputs'].items() if p.endswith('/src/bf_tap_r2/' + name)]
        if len(matches) != 1 or sha(work / 'src/bf_tap_r2' / name) != matches[0]:
            raise ValueError('Original model source changed: ' + name)
    versions = runtime(spec)
    available = require_memory(spec)
    training, query = official_frames(spec)
    part = partition(training, query, spec['training'])
    old = Path(spec['old_ema_directory'])
    if read(old / 'fit.json')['metadata']['fit_ids_digest'] != part['training']:
        raise ValueError('Old full EMA training partition differs')
    import pandas as pd
    if digest(pd.read_pickle(old / 'query.pkl').to_dict(orient='list')) != part['query_frame']:
        raise ValueError('Old full query identity differs')
    zip_payload(spec['parent_zip'], spec['platform_reference']['zip_sha256'], query.sample_id.tolist())
    models = source | binding_sources(reference_bindings())
    verify_files(models)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'query.pkl').open('xb') as f:
        query.to_pickle(f)
    write_new(out / 'manifest.json', dict(spec=spec, workspace=str(work),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=work, text=True).strip(),
        sources=source, inputs=inputs, model_sources=models, partition=part,
        query_sha256=sha(out / 'query.pkl'), tests_receipt_sha256=sha(tests), versions=versions,
        available_memory_mib=available, created_ns=time.time_ns(), formal_promotion=False))


def context(out, cold=False):
    out = Path(out).resolve()
    m = read(out / 'manifest.json')
    spec, work = m['spec'], Path(m['workspace'])
    if not out.is_relative_to(Path(spec['main_root']) / 'local/runs') or spec != read(work / SPEC):
        raise ValueError('Bound release context differs')
    verify_files(m['sources'])
    verify_files(m['model_sources'])
    verify_files({p: h for p, h in m['inputs'].items() if not cold or not p.lower().endswith('.csv')})
    validate_scope(spec, read(work / spec['scientific_config']))
    explicit_authority(spec)
    runtime(spec)
    if sha(out / 'query.pkl') != m['query_sha256']:
        raise ValueError('Frozen query changed')
    if (out / 'failure.json').exists():
        raise ValueError('Failed release retained; no automatic retry')
    return m


def member_directory(out, seed):
    return Path(out) / f'EMA_FULL_INIT{seed}'


def warm(out, seed):
    from bf_tap_r2.ema_span_confirmation_models import fit_component
    m = context(out)
    s = m['spec']
    if seed not in s['new_training_seeds']:
        raise ValueError('Only preregistered new training seeds may fit')
    training, query = official_frames(s)
    if partition(training, query, s['training']) != m['partition']:
        raise ValueError('Frozen partitions changed')
    settings = dict(s['training'], random_seed=seed)
    identity = dict(source_directory=str(Path(out).resolve()), split_seed=-1, fold=-1,
        trial_id=f'EMA_FULL_INIT{seed}')
    _, receipt = fit_component(member_directory(out, seed), training, query, identity=identity,
        settings=settings, mechanisms=s['mechanisms'], source_hashes=m['model_sources'])
    if receipt['native_counts']['torch_optimizer'] != 2 or any(
            v for k, v in receipt['native_counts'].items() if k != 'torch_optimizer'):
        raise ValueError('Member native budget changed')
    write_new(Path(out) / f'warm-complete-{seed}.json', dict(status='passed', pid=os.getpid(),
        training_seed=seed, manifest_sha256=sha(Path(out) / 'manifest.json'),
        component_sha256=sha(member_directory(out, seed) / 'complete.json'), peak_rss_mib=peak(s)))


def cold(out, receipt):
    from bf_tap_r2.ema_span_confirmation_models import audit_component
    m = context(out, cold=True)
    out = Path(out)
    if sha(out / 'warm-complete.json') != receipt:
        raise ValueError('External aggregate warm receipt changed')
    aggregate = read(out / 'warm-complete.json')
    expected_members = {str(seed): sha(out / f'warm-complete-{seed}.json')
        for seed in m['spec']['new_training_seeds']}
    if aggregate.get('members') != expected_members or aggregate.get('optimizer_runs') != 4:
        raise ValueError('Externally bound complete warm member inventory changed')
    audits = {}
    with cold_only():
        for seed in m['spec']['new_training_seeds']:
            w = read(out / f'warm-complete-{seed}.json')
            if (w['pid'] == os.getpid() or w['status'] != 'passed'
                    or w['manifest_sha256'] != sha(out / 'manifest.json') or w['training_seed'] != seed):
                raise ValueError('Independent cold process required')
            audits[str(seed)] = audit_component(member_directory(out, seed), w['component_sha256'])
    if sum(a['retained_states'] for a in audits.values()) != 4:
        raise ValueError('Four new selected/refit states required')
    write_new(out / 'cold-complete.json', dict(status='passed', warm_sha256=receipt,
        audits=audits, peak_rss_mib=peak(m['spec']), training_csv_reads=0, new_fits=0))


def audit(out):
    from bf_tap_r2.component_regularization_audit import verify_saved
    from bf_tap_r2.v3_4_bags import group_safe_inner_folds
    out = Path(out)
    m = context(out)
    s = m['spec']
    training, query = official_frames(s)
    inner = np.asarray(group_safe_inner_folds(training, seed=s['training']['inner_seed'])['fold'])
    fitting = training.loc[inner != 0].reset_index(drop=True)
    validation = training.loc[inner == 0]
    if partition(training, query, s['training']) != m['partition']:
        raise ValueError('Audit partition differs')
    epochs = {}
    for seed in s['training_seeds']:
        directory = Path(s['old_ema_directory']) if seed == 42 else member_directory(out, seed)
        settings = dict(s['training'], random_seed=seed)
        selector = verify_saved(directory / 'selection.pt', fitting, fitting[['tap_time_len']].to_numpy(),
            'EMA', settings, s['mechanisms'], validation)
        epoch = selector.saved['trace']['selected_epoch']
        model = verify_saved(directory / 'refit.pt', training, training[['tap_time_len']].to_numpy(),
            'EMA', settings, s['mechanisms'], expected_epoch=epoch)
        prediction = model.predict(query)
        if seed == 42:
            np.testing.assert_array_equal(prediction, np.load(directory / 'cold.npy', allow_pickle=False))
        else:
            receipt = read(directory / 'complete.json')
            metadata = receipt['model_metadata']
            if (metadata['fit_ids_digest'] != digest(training.sample_id.tolist())
                    or metadata['selected_epoch'] != epoch or metadata['optimizer_runs'] != 2
                    or metadata['traces']['selection'] != selector.saved['trace']
                    or metadata['traces']['refit'] != model.saved['trace']):
                raise ValueError('Native model metadata differs')
            with np.load(directory / 'predictions.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['query_ids'], query.sample_id.to_numpy(str))
                np.testing.assert_array_equal(prediction, z['prediction'])
        epochs[str(seed)] = epoch
    write_new(out / 'native-audit.json', dict(status='passed', new_native_states=4,
        old_native_states=2, optimizer_runs=4, selected_epochs=epochs,
        cold_sha256=sha(out / 'cold-complete.json'), peak_rss_mib=peak(s), new_fits=0,
        formal_promotion=False))


def mean_prediction(spec, predictions):
    if list(predictions) != spec['training_seeds']:
        raise ValueError('Fixed member ordering and all three training seeds required')
    values = [np.asarray(v, dtype=float) for v in predictions.values()]
    if any(v.shape != values[0].shape or v.ndim != 1 or not np.isfinite(v).all() for v in values):
        raise ValueError('Finite matching one-column member predictions required')
    return np.mean(np.stack(values), axis=0)


def build(out):
    import pandas as pd
    out = Path(out)
    m = context(out, cold=True)
    s = m['spec']
    if read(out / 'native-audit.json')['status'] != 'passed':
        raise ValueError('Independent native audit required before packaging')
    ids = pd.read_pickle(out / 'query.pkl').sample_id.tolist()
    parent = zip_payload(s['parent_zip'], s['platform_reference']['zip_sha256'], ids)
    old = np.load(Path(s['old_ema_directory']) / 'cold.npy', allow_pickle=False)[:, 0]
    predictions = {42: old}
    for seed in s['new_training_seeds']:
        with np.load(member_directory(out, seed) / 'predictions.npz', allow_pickle=False) as z:
            predictions[seed] = z['prediction'][:, 0].copy()
    mean = mean_prediction(s, predictions)
    payload = release_payload(parent, ids, old, mean)
    directory = out / 'package'
    directory.mkdir(exist_ok=False)
    with (directory / 'result.csv').open('xb') as f:
        f.write(payload)
    path = directory / 'Luqhhh_bf_tap_predict_round2.zip'
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('result.csv', payload)
    write_new(out / 'release.json', dict(candidate=s['candidate'], zip=str(path),
        zip_sha256=sha(path), result_sha256=sha(directory / 'result.csv'),
        parent_sha256=s['platform_reference']['zip_sha256'], training_seeds=s['training_seeds'],
        replacement_weight=.75, formal_promotion=False, platform_score=None))


def verify(out, receipt):
    import pandas as pd
    from bf_tap_r2.component_regularization import ComponentRegressor
    from bf_tap_r2.submission import validate_result
    out = Path(out)
    m = context(out, cold=True)
    s = m['spec']
    query = pd.read_pickle(out / 'query.pkl')
    ids = query.sample_id.tolist()
    if sha(out / 'release.json') != receipt:
        raise ValueError('External package receipt changed')
    release = read(out / 'release.json')
    payload = zip_payload(release['zip'], release['zip_sha256'], ids)
    rows = validate_result(payload, ids)
    parent = validate_result(zip_payload(s['parent_zip'], s['platform_reference']['zip_sha256'], ids), ids)
    with cold_only():
        predictions = {}
        for seed in s['training_seeds']:
            directory = Path(s['old_ema_directory']) if seed == 42 else member_directory(out, seed)
            predictions[seed] = ComponentRegressor.load(directory / 'refit.pt').predict(query)[:, 0]
        # Independent scalar arithmetic rather than the vectorized builder.
        expected = [float(row['pred_tap_time_len']) + .75 * (
            (float(predictions[42][i]) + float(predictions[1042][i]) + float(predictions[2042][i])) / 3
            - float(predictions[42][i])) for i, row in enumerate(parent)]
    np.testing.assert_array_equal([float(r['pred_tap_time_len']) for r in rows], expected)
    if [r['pred_tap_iron'] for r in rows] != [r['pred_tap_iron'] for r in parent]:
        raise ValueError('Parent iron field strings changed')
    if payload != (out / 'package/result.csv').read_bytes():
        raise ValueError('ZIP/result payload differs')
    write_new(out / 'package-audit.json', dict(status='passed', rows=322,
        zip_sha256=release['zip_sha256'], release_sha256=receipt,
        unmodified_iron_field_string_mismatches=0, arithmetic_difference=0,
        new_fits=0, training_csv_reads=0, peak_rss_mib=peak(s)))


def execute(out):
    out = Path(out)
    m = context(out)
    current_reference(m['spec'])
    require_memory(m['spec'])
    write_new(out / 'activation.json', dict(pid=os.getpid(), started_ns=time.time_ns(),
        manifest_sha256=sha(out / 'manifest.json')))
    command = [sys.executable, str(Path(m['workspace']) / 'scripts/ema_mean3_release.py')]
    try:
        for seed in m['spec']['new_training_seeds']:
            subprocess.run(command + ['warm', '--output', str(out), '--training-seed', str(seed)],
                cwd=m['workspace'], check=True)
        write_new(out / 'warm-complete.json', dict(status='passed', optimizer_runs=4,
            members={str(seed): sha(out / f'warm-complete-{seed}.json')
                for seed in m['spec']['new_training_seeds']}))
        for mode in ('cold', 'audit', 'build', 'verify'):
            args = command + [mode, '--output', str(out)]
            if mode in ('cold', 'verify'):
                args += ['--receipt', sha(out / ('warm-complete.json' if mode == 'cold' else 'release.json'))]
            subprocess.run(args, cwd=m['workspace'], check=True)
        verify_files(m['sources'])
        verify_files(m['inputs'])
        write_new(out / 'completion-event.json', dict(status='passed', optimizer_runs=4,
            full_training_procedures=2, reused_full_models=1, new_CV=0, new_confirmation_seeds=0,
            packages=1, formal_promotion=False, release_sha256=sha(out / 'release.json'),
            native_audit_sha256=sha(out / 'native-audit.json'),
            package_audit_sha256=sha(out / 'package-audit.json')))
    except BaseException as error:
        write_new(out / 'failure.json', dict(error=repr(error), automatic_retries=0))
        raise


def observe(out):
    out = Path(out)
    m = context(out)
    env = dict(os.environ, PYTHONPATH=str(WORK / 'src'))
    with (out / 'controller.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__)), 'execute', '--output', str(out)],
            cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT)
        launch = dict(supervisor_pid=os.getpid(), controller_pid=child.pid,
            started_ns=time.time_ns(), monitor_seconds=600)
        write_new(out / 'process-launch.json', launch)
        print(json.dumps(launch), flush=True)
        terminal = wait_actual(child, out, 'release')
    result = dict(status='passed' if terminal['exit_code'] == 0 else 'failed',
        actual_main=terminal, manifest_sha256=sha(out / 'manifest.json'), automatic_retries=0)
    if terminal['exit_code'] == 0:
        complete = read(out / 'completion-event.json')
        if complete['status'] != 'passed' or complete['optimizer_runs'] != 4:
            raise ValueError('Full release completion budget differs')
        if terminal['peak_rss_mib'] > m['spec']['max_worker_rss_mib']:
            raise ValueError('Actual process memory gate failed')
        result.update(completion_sha256=sha(out / 'completion-event.json'),
            release_sha256=sha(out / 'release.json'), package_audit_sha256=sha(out / 'package-audit.json'),
            optimizer_runs=4, packages=1)
    write_new(out / 'terminal-verification.json', result)
    print(json.dumps(result), flush=True)
    return terminal['exit_code']


def deliver(out):
    from bf_tap_r2.submission import validate_result
    import pandas as pd
    out = Path(out)
    m = context(out, cold=True)
    s = m['spec']
    terminal = read(out / 'terminal-verification.json')
    if (terminal['status'] != 'passed' or terminal['actual_main']['exit_code'] != 0
            or terminal['release_sha256'] != sha(out / 'release.json')
            or terminal['package_audit_sha256'] != sha(out / 'package-audit.json')):
        raise ValueError('Real successful terminal and package audit required before desktop handoff')
    ids = pd.read_pickle(out / 'query.pkl').sample_id.tolist()
    release = read(out / 'release.json')
    source = Path(release['zip'])
    payload = zip_payload(source, release['zip_sha256'], ids)
    parent = validate_result(zip_payload(s['parent_zip'], s['platform_reference']['zip_sha256'], ids), ids)
    destination = Path(s['desktop_root'])
    destination.mkdir(exist_ok=False)
    target = destination / source.name
    with source.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    copied = zip_payload(target, release['zip_sha256'], ids)
    rows = validate_result(copied, ids)
    if copied != payload or [r['pred_tap_iron'] for r in rows] != [r['pred_tap_iron'] for r in parent]:
        raise ValueError('Desktop byte or fixed-field identity changed')
    with (destination / 'README.txt').open('x', encoding='utf-8') as f:
        f.write('EMA_MEAN3_FULL_Q75（2026-10-02）\n\n'
            '训练seed42、1042、2042的EMA等权平均，时长替换权重0.75；铁量保持Q75原字段。\n'
            'G0工程审计通过，G1为用户指定人工探索，未正式晋级，平台分数尚未测得。\n'
            '上传本目录ZIP，并回传EMA_MEAN3分数。本地增益不能直接换算为平台得分。\n')
    write_new(out / 'desktop-delivery.json', dict(status='passed', candidate=s['candidate'],
        desktop_zip=str(target), zip_sha256=sha(target), source_package=str(source),
        source_desktop_bytes_equal=source.read_bytes() == target.read_bytes(),
        csv_bytes_equal=True, rows=322, unchanged_iron_string_mismatches=0,
        new_fits=0, new_packages=0, desktop_copies=1, agent_uploads=0))
    print(json.dumps(dict(desktop=str(destination), candidate=s['candidate'], zip_sha256=sha(target))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'observe', 'execute', 'warm', 'cold', 'audit', 'build', 'verify', 'deliver'])
    p.add_argument('--output', required=True)
    p.add_argument('--tests')
    p.add_argument('--receipt')
    p.add_argument('--training-seed', type=int)
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(WORK, args.output, args.tests)
    elif args.mode == 'observe':
        raise SystemExit(observe(args.output))
    elif args.mode == 'warm':
        warm(args.output, args.training_seed)
    elif args.mode in ('cold', 'verify'):
        globals()[args.mode](args.output, args.receipt)
    else:
        globals()[args.mode](args.output)
