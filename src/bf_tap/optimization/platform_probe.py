"""Limited V11 probe and independent V10 inference using certified final assets."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
from ..artifacts import (atomic_write_json, build_inference_source_contract,
    file_identities, file_sha256, runtime_environment, stable_digest, verify_file_identities)
from ..config import load_yaml
from ..exceptions import ContractError
from ..offline import _load_process_sources
from .component_export import ComponentFeatures, META, PRED
from .final_lifecycle import sample_metadata
from .qrf_time_run import pack
from .recency_model import RecencyModel
from .structural_predict import StructuralPredictor
from .structural_run import append_ledger
from .v13_common import zero_fit

REPO = Path(__file__).resolve().parents[3]
V11 = 'V11_V6I_IRON_QRF_MEAN_TIME'
V10 = 'V10_V6I_IRON_V8_TIME'
COLS = ['sample_id', *PRED]
spec = importlib.util.spec_from_file_location('frozen_v13_stage', REPO/'scripts/optimization_v13_stage_predict.py')
stage_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage_api)


def read(path):
    return json.loads(Path(path).read_text())


def identities(paths):
    return file_identities({str(p): p for p in paths})


def align(frame, ids):
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ContractError('unique nonmissing prediction IDs required')
    if frame.sample_id.dtype.kind not in ('O', 'U', 'S') or any(not isinstance(s, str) for s in frame.sample_id):
        raise ContractError('string prediction IDs required')
    if len(set(ids)) != len(ids) or set(frame.sample_id) != set(ids):
        raise ContractError('prediction ID coverage differs')
    return frame.set_index('sample_id').loc[ids].reset_index()


def freeze(root, purpose, evidence, inputs=None):
    root = Path(root).resolve()
    if not root.is_relative_to(REPO/'local/runs'):
        raise ContractError('private evidence and predictions must stay under ignored local/runs')
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    scope = load_yaml(REPO/'configs/platform_probes_r2/access_scope.yaml')
    if not read(REPO/'EVIDENCE_STATUS.json')['holdout_consumed']:
        raise ContractError('consumed historical evidence required')
    files = [*REPO.glob('src/bf_tap/**/*.py'), *REPO.glob('configs/platform_probes_r2/*.yaml'),
             REPO/'configs/protection.yaml', REPO/'uv.lock', REPO/'pyproject.toml',
             REPO/'scripts/platform_probe_r2.py', REPO/'docs/platform_probes_r2/PLAN.md',
             *REPO.glob('workers/qrf_v015/*.py'), REPO/'workers/qrf_v015/uv.lock',
             REPO/'workers/qrf_v015/pyproject.toml', REPO/'scripts/optimization_v13_stage_predict.py',
             REPO/'scripts/optimization_v4_cold_predict.py', REPO/'tests/test_platform_probe.py',
             REPO/'workers/qrf_v015/tests/test_inference.py']
    manifest = dict(purpose=purpose, scope=scope,
                    registration=load_yaml(REPO/'configs/platform_probes_r2/experiment.yaml'),
                    environment=runtime_environment(), protection=load_yaml(REPO/'configs/protection.yaml'),
                    sources=identities(files), evidence=identities(evidence),
                    inputs=file_identities(inputs or {}), new_model_fits=0,
                    new_preprocessor_fits=0, new_LAD_fits=0, automatic_uploads=0, desktop_writes=0)
    atomic_write_json(root/'manifest.json', manifest)
    ledger = REPO/'local/ledgers'/('platform-probes-r2-'+root.parent.name+'-'+root.name+'.jsonl')
    if ledger.exists():
        raise ContractError('do not reuse a sealed access ledger')
    append_ledger(dict(new_ledger=str(ledger), authorization=scope['authorization']+':'+purpose), root)
    return manifest, ledger


def verify(manifest):
    for key in ('sources', 'evidence', 'inputs'):
        verify_file_identities(manifest[key])


def bundle_assets(bundle, expected_sha256):
    if file_sha256(bundle) != expected_sha256:
        raise ContractError('uncertified component manifest')
    value = read(bundle)
    if value['kind'] != 'V10_CERTIFIED_FINAL_COMPONENT_REFERENCES_v1':
        raise ContractError('unknown component manifest')
    verify_file_identities(value['assets'])
    return value


def iron_branch(samples, predictor, builder, model, beta, op, burden, contract):
    """Align caller-order direct/E04 arrays to the original sorted comparator."""
    v1, old = predictor.predict(samples, op, burden, contract)
    x = builder.X(samples, predictor.entries['OR'], predictor.loaded['OR'][1])
    ids = pd.Index(samples.sample_id.tolist(), name='sample_id')
    direct = pd.Series(model.predict(x), index=ids)
    hr_model, hr_history = predictor.loaded['HR']
    e04 = hr_model.predict_raw(builder.X(samples, predictor.entries['HR'], hr_history)).clip(lower=0)
    e04.index = ids
    base = .8*direct.loc[old.sample_id].to_numpy()+(1.-.8)*e04.loc[old.sample_id, PRED[0]].to_numpy()
    rate = old.pred_rate.to_numpy()
    direction = np.zeros(len(base))
    usable = rate > 1e-6
    direction[usable] = (rate*old[PRED[1]].to_numpy()-base)[usable]
    result = old[['sample_id']].copy()
    result[PRED[0]] = np.maximum(0., base+beta*direction)
    result = align(result, ids.tolist())
    if not np.isfinite(result[PRED[0]]).all():
        raise ContractError('nonfinite iron branch')
    return result, align(v1, ids.tolist()), x


def combine(iron, arrays, ids, statistic='median'):
    if statistic not in ('median', 'mean') or set(arrays) != {'ids', 'median', 'mean'}:
        raise ContractError('only frozen median/mean QRF outputs permitted')
    if arrays['ids'].dtype.kind not in ('U', 'S'):
        raise ContractError('string worker IDs required')
    time = align(pd.DataFrame({'sample_id': arrays['ids'].tolist(), PRED[1]: arrays[statistic]}), ids)
    result = align(iron[['sample_id', PRED[0]]], ids)
    result[PRED[1]] = time[PRED[1]].to_numpy()
    if not np.isfinite(result[PRED].to_numpy()).all() or (result[PRED].to_numpy() < 0).any():
        raise ContractError('finite nonnegative prediction columns required')
    return result


def serialize(prediction):
    rows = [dict(sample_id=row.sample_id, pred_tap_iron=f'{row.pred_tap_iron:.6f}',
                 pred_tap_time_len=f'{row.pred_tap_time_len:.6f}') for row in prediction.itertuples(index=False)]
    return serialize_rows(rows)


def serialize_rows(rows):
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=COLS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode('utf-8')


def stage_predict(bundle, bundle_sha256, config, stage, output):
    output = Path(output).resolve()
    if output.exists() or Path(str(output)+'.json').exists():
        raise FileExistsError(output)
    paths = stage_api.stage_paths(config, stage)
    manifest, ledger = freeze(Path(str(output)+'.evidence'), 'V10_'+stage+'_ENGINEERING_ONLY',
                              [Path(bundle), Path(config)], paths)
    root = Path(str(output)+'.evidence')
    try:
        refs = bundle_assets(bundle, bundle_sha256)
        # Before restoring model histories, bind every referenced asset to this run.
        atomic_write_json(root/'source_manifest.json', refs)
        with zero_fit() as counts:
            predictor = StructuralPredictor(Path(refs['original_V1']))
            samples = sample_metadata(paths[stage+'_samples'])[META]
            stage_api.validate_metadata(samples, predictor)
            contract = build_inference_source_contract(file_identities({k:v for k,v in paths.items() if k != stage+'_samples'}),
                        semantic_contract_sha256=predictor.a['contract_digests']['semantic_contract_sha256'])
            if contract != predictor.contract:
                raise ContractError('BLOCKED_SOURCE_CONTRACT_CHANGE')
            if file_sha256(paths['data_dictionary']) != refs['dictionary_sha256']:
                raise ContractError('BLOCKED_SOURCE_CONTRACT_CHANGE: dictionary changed')
            op, burden, _ = _load_process_sources(paths, predictor.a['semantic'], predictor.a['features'])
            builder = ComponentFeatures(predictor.a, op, burden)
            model = RecencyModel.load(Path(refs['recency_iron']))
            coefficient = read(refs['iron_coefficient'])
            cutoff = pd.Timestamp(predictor.m['training']['fit_cutoff'])
            training = model.metadata['training']
            if (model.target != 'tap_iron' or pd.Timestamp(training['cutoff']) != cutoff or
                training['training_ids_sha256'] != predictor.m['training']['sample_ids_sha256'] or
                pd.Timestamp(training['reference_max']) >= cutoff or pd.Timestamp(training['available_max']) > cutoff or
                coefficient['target'] != 'tap_iron' or pd.Timestamp(coefficient['cutoff']) != cutoff or
                pd.Timestamp(coefficient['available_max']) > cutoff or not coefficient['certificate']['smallest_minimizer_verified'] or
                not 0 <= coefficient['beta'] <= 1):
                raise ContractError('certified iron training/coefficient identity changed')
            iron, v1, x = iron_branch(samples, predictor, builder, model, coefficient['beta'], op, burden, contract)
            checks = {}
            groups = {'reverse':samples.iloc[::-1], 'subset':samples.iloc[::max(1,len(samples)//7)],
                      'single':samples.iloc[[len(samples)//2]]}
            chunks = []
            for i in range(0, len(samples), 127):
                group, _, gx = iron_branch(samples.iloc[i:i+127], predictor, builder, model, coefficient['beta'], op, burden, contract)
                if not gx.equals(x.iloc[i:i+127]):
                    raise ContractError('chunked raw features differ')
                chunks.append(group)
            if not np.array_equal(align(pd.concat(chunks), samples.sample_id.tolist())[PRED[0]], iron[PRED[0]]):
                raise ContractError('chunked iron differs')
            checks['chunks'] = True
            for name, group in groups.items():
                part, _, gx = iron_branch(group, predictor, builder, model, coefficient['beta'], op, burden, contract)
                expected = iron.set_index('sample_id').loc[group.sample_id, PRED[0]].to_numpy()
                if not np.array_equal(part[PRED[0]], expected) or not gx.equals(x.loc[group.index]):
                    raise ContractError('iron/raw features differ with caller order')
                checks[name] = True
            handoff = pack(root/'evaluation.npz', x, samples, cutoff)
            audit = stage_api.source_audit(samples, predictor, op, burden)
            atomic_write_json(root/'root_validation.json', dict(status='PASS', checks=checks, zero_fit=counts, source_audit=audit))
            v1.to_csv(root/'V1_full_precision.csv', index=False, mode='x')
        worker = REPO/'workers/qrf_v015'
        subprocess.run([str(worker/'.venv/bin/python'), str(worker/'inference.py'), '--model', refs['qrf_model'],
                        '--bundle-sha256', refs['qrf_bundle_sha256'], '--input', str(root/'evaluation.npz'),
                        '--input-sha256', handoff['sha256'], '--output', str(root/'qrf.npz')], check=True, cwd=REPO)
        with np.load(root/'qrf.npz', allow_pickle=False) as saved:
            arrays = {k:saved[k] for k in saved.files}
        prediction = combine(iron, arrays, samples.sample_id.tolist())
        # Combining independently aligned columns must also be invariant to order/subsets.
        for group in [samples.iloc[::-1], samples.iloc[::max(1,len(samples)//7)], samples.iloc[[len(samples)//2]]]:
            mask = np.isin(arrays['ids'], group.sample_id)
            part = combine(iron[iron.sample_id.isin(group.sample_id)], {k:v[mask] for k,v in arrays.items()}, group.sample_id.tolist())
            stage_api.exact(part, prediction[prediction.sample_id.isin(group.sample_id)])
        verify(manifest)
        verify_file_identities(refs['assets'])
        output.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_csv(output, index=False, mode='x')
        (root/'result_replay.csv').write_bytes(serialize(prediction))
        worker_check = read(root/'qrf.npz.json')
        status = dict(status='PASS', candidate=V10, stage=stage, rows=len(prediction),
                      engineering_valid=True, official_data_identity_verified=False,
                      quality_evaluated=False, platform_verified=False,
                      data_status='PREVIEW_ENGINEERING_ONLY', model_cutoff=str(cutoff),
                      zero_fit=counts, worker_zero_fit=worker_check['zero_fit'], checks=checks,
                      sample_id_sha256=stable_digest(samples.sample_id.tolist()),
                      reference_min=str(samples.reference_time.min()), reference_max=str(samples.reference_time.max()),
                      component_bundle_sha256=bundle_sha256, manifest_sha256=file_sha256(root/'manifest.json'),
                      result_sha256=file_sha256(root/'result_replay.csv'), source_audit=audit)
        atomic_write_json(str(output)+'.json', status)
        atomic_write_json(root/'completion.json', status)
        atomic_write_json(root/'receipt.json', dict(evidence=identities([p for p in root.rglob('*') if p.is_file()]+[output,Path(str(output)+'.json'),ledger])))
        return prediction, arrays, status
    except Exception as exc:
        atomic_write_json(root/'failure.json', dict(exception=repr(exc), evidence_preserved=True, new_fits=0))
        raise


def zip_rows(path, expected_sha256=None):
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ContractError('original package hash differs')
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ['result.csv']:
            raise ContractError('ZIP must contain only result.csv')
        raw = archive.read('result.csv')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8'), newline=''))
    if reader.fieldnames != COLS:
        raise ContractError('official three-column format required')
    rows = list(reader)
    ids = [r['sample_id'] for r in rows]
    if not ids or len(set(ids)) != len(ids) or any(not s for s in ids):
        raise ContractError('unique complete submission IDs required')
    for row in rows:
        values = np.array([float(row[p]) for p in PRED])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ContractError('invalid original submission prediction')
    return rows, raw


def mean_rows(original_rows, arrays):
    """Copy iron text unchanged, join the same-forest mean by string sample ID."""
    ids = [row['sample_id'] for row in original_rows]
    means = align(pd.DataFrame({'sample_id':arrays['ids'].tolist(), 'mean':arrays['mean']}), ids)
    if not np.isfinite(means['mean']).all() or (means['mean'] < 0).any():
        raise ContractError('invalid mean prediction')
    return [dict(sample_id=row['sample_id'], pred_tap_iron=row[PRED[0]],
                 pred_tap_time_len=f'{mean:.6f}') for row, mean in zip(original_rows, means['mean'], strict=True)]


def prepare_probe(output):
    """P0, independent A/B replay, then exactly one mean candidate package."""
    root = Path(output).resolve()
    runs = REPO/'local/runs'
    original = runs/'optimization-v0.8-v1-challenger-r1'
    recovered = runs/'optimization-v0.15-platform-recovery-r1'
    forest = runs/'optimization-v0.15-v8-user-test-a-r1'
    composition = runs/'optimization-v0.15-v10-target-composition-r1'
    packages = dict(
        V10=(composition/'submission/Luqhhh_bf_tap_predict_prelim.zip', 'be83f1f623c12f4ef03479f2fcacdfb756920d41c4e09d1c4f6b55a2279761ac'),
        V8=(forest/'submission/Luqhhh_bf_tap_predict_prelim.zip', '409218c364dac20f59169b08f95af54c1a9e206eb19b71a6115f05ce4889b402'),
        V6I=(recovered/'submissions/V6I_RECENCY60_IRON/Luqhhh_bf_tap_predict_prelim.zip', '7c28b6c76fe05eebf90be2560b52d03111a2d0e2d9b1b44f80fed085092e385c'))
    configs = {stage:REPO/'local/runs/optimization-v0.13-opt28-preview-r1'/f'{stage}_cold.yaml' for stage in ('test_a','test_b')}
    paths = {stage:stage_api.stage_paths(config, stage) for stage, config in configs.items()}
    model_files = [*original.joinpath('bundle').rglob('*'),
                   *recovered.joinpath('models/tap_iron/bundle').rglob('*'),
                   recovered/'coefficients/V6I_RECENCY60_IRON.json',
                   *forest.joinpath('models/12').rglob('*')]
    model_files = [p for p in model_files if p.is_file()]
    receipts = [source/'receipt.json' for source in (forest, recovered, composition)]
    evidence = [*model_files, *receipts, *configs.values(), *[p[0] for p in packages.values()],
                forest/'worker_predictions/12.npz', forest/'worker_predictions/12.npz.json',
                forest/'V8_full_precision.csv', forest/'V1_full_precision.csv', forest/'models_complete.json',
                recovered/'predictions/V6I_RECENCY60_IRON.csv', original/'cold_predictions.csv',
                REPO/'configs/optimization_v0_8/active_release.yaml', REPO/'configs/optimization_v0_4/active_release.yaml']
    input_paths = {stage+'__'+key:value for stage, pp in paths.items() for key,value in pp.items()}
    manifest, ledger = freeze(root, 'V11_PREPARE_AND_V10_INDEPENDENT_REPLAY', evidence, input_paths)
    try:
        for receipt in receipts:
            verify_file_identities(read(receipt)['evidence'])
        source_rows = {}
        source_csv = {}
        for name, (path, digest) in packages.items():
            source_rows[name], source_csv[name] = zip_rows(path, digest)
        ids = sample_metadata(paths['test_a']['test_a_samples']).sample_id.tolist()
        if len(ids) != 335:
            raise ContractError('certified original test_a identity/335 rows required')
        source_by_id = {k:{r['sample_id']:r for r in v} for k,v in source_rows.items()}
        if any(set(v) != set(ids) for v in source_by_id.values()):
            raise ContractError('original package coverage differs')
        original_rows = [source_by_id['V10'][s] for s in ids]
        for row in original_rows:
            s = row['sample_id']
            if row[PRED[0]] != source_by_id['V6I'][s][PRED[0]] or row[PRED[1]] != source_by_id['V8'][s][PRED[1]]:
                raise ContractError('original V10 component columns differ')
        with np.load(forest/'worker_predictions/12.npz', allow_pickle=False) as saved:
            arrays = {k:saved[k] for k in saved.files}
        if set(arrays) != {'ids','median','mean'} or arrays['ids'].dtype.kind != 'U':
            raise ContractError('only final string-ID median/mean output required')
        full = pd.read_csv(forest/'V8_full_precision.csv', dtype={'sample_id':str}, float_precision='round_trip')
        normalized = align(pd.DataFrame({'sample_id':arrays['ids'].tolist(), 'median':arrays['median']}), ids)
        if not np.array_equal(normalized['median'], align(full, ids)[PRED[1]]):
            raise ContractError('final NPZ median does not reproduce original full precision V8')
        if any(f'{median:.6f}' != source_by_id['V8'][s][PRED[1]] for s,median in zip(ids, normalized['median'], strict=True)):
            raise ContractError('final median differs at submission precision')
        bundle = dict(kind='V10_CERTIFIED_FINAL_COMPONENT_REFERENCES_v1',
                      original_V1=str(original/'bundle'), recency_iron=str(recovered/'models/tap_iron/bundle'),
                      iron_coefficient=str(recovered/'coefficients/V6I_RECENCY60_IRON.json'),
                      qrf_model=str(forest/'models/12'), qrf_bundle_sha256=read(forest/'models_complete.json')['12'],
                      dictionary_sha256=file_sha256(paths['test_a']['data_dictionary']), assets=identities(model_files),
                      provenance_receipts=identities(receipts), source_v10_zip_sha256=packages['V10'][1],
                      historical_quality_passed=False, platform_verified=False)
        atomic_write_json(root/'component_bundle.json', bundle)
        bundle_sha = file_sha256(root/'component_bundle.json')
        atomic_write_json(root/'p0.json', dict(status='PASS', original_receipts_verified=True,
                          packages={k:dict(path=str(v[0]),sha256=v[1]) for k,v in packages.items()},
                          rows=335, final_median_full_precision_exact=True, median_submission_exact=True,
                          V10_component_columns_exact=True, component_bundle_sha256=bundle_sha,
                          npz_identity=identities([forest/'worker_predictions/12.npz'])))
        for stage in ('test_a','test_b'):
            subprocess.run([sys.executable, str(REPO/'scripts/platform_probe_r2.py'), 'predict', '--bundle', str(root/'component_bundle.json'),
                            '--bundle-sha256', bundle_sha, '--data-config', str(configs[stage]), '--stage', stage,
                            '--output', str(root/f'{stage}_V10_full_precision.csv')], check=True, cwd=REPO)
        cold_root = root/'test_a_V10_full_precision.csv.evidence'
        replay = pd.read_csv(root/'test_a_V10_full_precision.csv', dtype={'sample_id':str}, float_precision='round_trip')
        legacy_iron = pd.read_csv(recovered/'predictions/V6I_RECENCY60_IRON.csv', dtype={'sample_id':str}, float_precision='round_trip')
        if not np.array_equal(align(replay, ids)[PRED[0]], align(legacy_iron, ids)[PRED[0]]):
            raise ContractError('cold V6I iron differs from original full precision')
        if not np.array_equal(align(replay, ids)[PRED[1]], align(full, ids)[PRED[1]]):
            raise ContractError('cold QRF median differs from original full precision')
        cold_v1 = pd.read_csv(cold_root/'V1_full_precision.csv', dtype={'sample_id':str}, float_precision='round_trip')
        legacy_v1 = pd.read_csv(original/'cold_predictions.csv', dtype={'sample_id':str}, float_precision='round_trip')
        stage_api.exact(cold_v1, legacy_v1)
        with np.load(cold_root/'qrf.npz', allow_pickle=False) as loaded:
            cold_arrays = {k:loaded[k] for k in loaded.files}
        for key in ('median','mean'):
            cold = align(pd.DataFrame({'sample_id':cold_arrays['ids'].tolist(), key:cold_arrays[key]}), arrays['ids'].tolist())
            if not np.array_equal(cold[key], arrays[key]):
                raise ContractError('independent final median/mean differs from saved NPZ')
        if (cold_root/'result_replay.csv').read_bytes() != source_csv['V10']:
            raise ContractError('V10 replay serialization differs from original result.csv bytes')
        validation = dict(status='PASS', independent_V6I_full_precision_exact=True,
                          independent_V8_full_precision_exact=True, independent_final_median_mean_exact=True,
                          independent_original_V1_full_precision_exact=True,
                          V10_original_result_bytes_exact=True, new_fit_attempts=0,
                          stage_validations={stage:read(root/f'{stage}_V10_full_precision.csv.json') for stage in ('test_a','test_b')})
        atomic_write_json(root/'cold_validation.json', validation)
        rows = mean_rows(original_rows, arrays)
        raw = serialize_rows(rows)
        changed = sum(a[PRED[1]] != b[PRED[1]] for a,b in zip(rows, original_rows, strict=True))
        # Only after cold validation passes do we create the single new candidate ZIP.
        candidate = dict(candidate=V11, stage='test_a', rows=335, iron_original_text_exact=True,
                         changed_time_rows_at_submission_precision=changed, engineering_valid=True,
                         historical_quality_passed=False, platform_verified=False)
        if raw == source_csv['V10']:
            status = 'NO_CHANGE_SKIP_SUBMISSION'
            candidate['zip_created'] = False
        else:
            folder = root/'submission'
            folder.mkdir()
            (folder/'result.csv').write_bytes(raw)
            target = folder/'Luqhhh_bf_tap_predict_prelim_V11_V6I_IRON_QRF_MEAN_TIME.zip'
            with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
                info = zipfile.ZipInfo('result.csv', (2026,9,12,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, raw)
            readback, restored = zip_rows(target)
            if restored != raw or readback != rows:
                raise ContractError('candidate ZIP round-trip differs')
            candidate.update(zip_created=True, zip=identities([target]), result_sha256=file_sha256(folder/'result.csv'))
            status = 'PREPARED_PENDING_MANUAL_PLATFORM_PROBE'
        atomic_write_json(root/'candidate_validation.json', candidate)
        atomic_write_json(root/'platform_feedback.json', dict(candidate=V11, stage='test_a', score=None,
                          status='PENDING_NOT_SUBMITTED', submission_id=None, submission_time=None,
                          platform_displayed_status=None, platform_verified=False, current_best_user_reported_score=83.1951,
                          remaining_account_submissions='UNKNOWN', manual_probe_requires_remaining_submissions=2))
        atomic_write_json(root/'fit_counts.json', dict(model_fit_attempts=0, preprocessor_fit_attempts=0,
                          LAD_fit_attempts=0, automatic_uploads=0, desktop_writes=0, candidate_zips=int(candidate['zip_created'])))
        verify(manifest)
        verify_file_identities(bundle['assets'])
        completion = dict(status=status, candidate=V11, engineering_valid=True,
                          historical_quality_passed=False, official_round2_identity_verified=False,
                          platform_verified=False, best_user_reported_test_a=V10,
                          best_user_reported_test_a_score=83.1951, v16='PAUSED_NOT_IMPLEMENTED', V2='PAUSED',
                          V10_independent_test_a_replay=True, old_test_b_preview_rows=322,
                          new_model_fit_attempts=0, new_preprocessor_fit_attempts=0, new_LAD_fit_attempts=0,
                          platform_submissions=0, desktop_writes=0, current_release_pointer_changed=False)
        atomic_write_json(root/'completion.json', completion)
        atomic_write_json(root/'receipt.json', dict(status=status, evidence=identities([p for p in root.rglob('*') if p.is_file()]+[ledger])))
        print(json.dumps(completion, ensure_ascii=False), flush=True)
        return completion
    except Exception as exc:
        atomic_write_json(root/'failure.json', dict(exception=repr(exc), evidence_preserved=True, new_fit_attempts=0))
        raise
