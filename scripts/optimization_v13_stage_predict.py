"""OPT-28: frozen V1 stage-aware preview inference; never trains or makes a ZIP."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from bf_tap.artifacts import atomic_write_json, file_identities, build_inference_source_contract, stable_digest
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.offline import _load_process_sources
from bf_tap.optimization.component_export import META, PRED
from bf_tap.optimization.final_lifecycle import sample_metadata
from bf_tap.optimization.rate_model import schema
from bf_tap.optimization.structural_predict import StructuralPredictor
from bf_tap.optimization.component_export import ComponentFeatures
from bf_tap.optimization.v13_common import freeze, verify, zero_fit

STAGES = ('test_a', 'test_b', 'test_c')
SOURCES = {'operation_hourly', 'burden_change', 'data_dictionary'}


def stage_paths(config, stage):
    if stage not in STAGES:
        raise ContractError('unknown prediction stage')
    data = load_yaml(config)
    if set(data) != {'schema_version', 'paths'} or data['schema_version'] != 1:
        raise ContractError('schema_version=1 and label-free paths required')
    paths = data['paths']
    if not isinstance(paths, dict) or set(paths) != SOURCES | {stage+'_samples'}:
        raise ContractError('only current-stage metadata and three label-free sources permitted')
    if any(not isinstance(v, str) or not v for v in paths.values()):
        raise ContractError('nonempty source paths required')
    return paths


def validate_metadata(samples, predictor):
    if list(samples) != META or samples.empty or samples.isna().any().any() or samples.sample_id.duplicated().any():
        raise ContractError('unique nonempty metadata-only samples required')
    if samples.reference_time.dt.tz is None:
        raise ContractError('timezone-aware reference_time required')
    cutoff = pd.Timestamp(predictor.m['training']['fit_cutoff'])
    if (samples.reference_time < cutoff).any():
        raise ContractError('prediction before frozen model cutoff')
    histories = [h for _, h in predictor.loaded.values()] + [predictor.history]
    if any(set(samples.sample_id.astype(str)) & set(h.sample_id.astype(str)) for h in histories):
        raise ContractError('inference IDs intersect stored training histories')


def exact(left, right):
    a, b = [x.set_index('sample_id').sort_index() for x in (left, right)]
    if a.index.duplicated().any() or b.index.duplicated().any() or not a.index.equals(b.index):
        raise ContractError('prediction IDs differ')
    if not np.array_equal(a[PRED].to_numpy(), b[PRED].to_numpy()):
        raise ContractError('prediction changes with order/subset/chunk')
    return 0.0


def checked_predictions(predictor, samples, op, burden, contract, chunk_size=127):
    validate_metadata(samples, predictor)
    if chunk_size < 1:
        raise ContractError('positive chunk size required')
    pred, parts = predictor.predict(samples, op, burden, contract)
    exact(pred, predictor.predict(samples.iloc[::-1], op, burden, contract)[0])
    chunks = [predictor.predict(samples.iloc[i:i+chunk_size], op, burden, contract)[0]
              for i in range(0, len(samples), chunk_size)]
    exact(pred, pd.concat(chunks, ignore_index=True))
    subset = samples.iloc[::max(1, len(samples)//7)]
    exact(pred.loc[pred.sample_id.isin(subset.sample_id)], predictor.predict(subset, op, burden, contract)[0])
    for i in sorted({0, len(samples)//2, len(samples)-1}):
        one = samples.iloc[i:i+1]
        exact(pred.loc[pred.sample_id.isin(one.sample_id)], predictor.predict(one, op, burden, contract)[0])
    values = pred[PRED].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or list(pred) != ['sample_id', *PRED]:
        raise ContractError('finite nonnegative three-column predictions required')
    if not pred.sample_id.astype(str).tolist() == samples.sample_id.astype(str).tolist():
        raise ContractError('output order differs from current-stage input')
    return pred, parts


def source_audit(samples, predictor, op, burden):
    builder = ComponentFeatures(predictor.a, op, burden)
    roles = {}
    for role, (model, history) in predictor.loaded.items():
        x = builder.X(samples, predictor.entries[role], history)
        if schema(x) != model.feature_schema_:
            raise ContractError('frozen component schema changed')
        age_flags = {c: {'missing': int(x[c].isna().sum()),
                         'minimum': float(x[c].min()) if x[c].notna().any() else None,
                         'maximum': float(x[c].max()) if x[c].notna().any() else None,
                         'flagged': int((x[c] > 0).sum()) if c.endswith(('__missing', '__stale')) else None}
                     for c in x if c.endswith(('__missing', '__stale', '__event_age_hours', '__latest_available_age_minutes'))}
        roles[role] = {'feature_schema': model.feature_schema_, 'schema_sha256': stable_digest(schema(x)),
                       'missing_stale_age': age_flags, 'history_available_max': str(history.available_at.max())}
    return {'roles': roles, 'operation_file_event_max': str(op.event_time.max()),
            'operation_file_available_max': str(op.available_at.max()),
            'burden_file_event_max': str(burden.event_time.max()),
            'burden_file_available_max': str(burden.available_at.max()),
            'restriction': 'each sample uses only event_time and available_at <= reference_time; full-file maxima are inventory only'}


def main(bundle, config, stage, output):
    bundle, output = Path(bundle), Path(output)
    if output.exists() or Path(str(output)+'.json').exists():
        raise FileExistsError(output)
    paths = stage_paths(config, stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = [p for p in bundle.rglob('*') if p.is_file()] + [Path(config), Path('scripts/optimization_v8_cold_predict.py'), Path('scripts/optimization_v4_cold_predict.py')]
    evidence += [Path('configs/optimization_v0_8/active_release.yaml'), Path('configs/optimization_v0_4/active_release.yaml')]
    for path in evidence[-2:]:
        evidence.append(Path(load_yaml(path)['test_a_zip']))
    # Freeze all identities before loading metadata/process sources or model histories.
    root = Path(str(output)+'.evidence')
    manifest = freeze(root, 'OPT28_'+stage+'_PREVIEW_ENGINEERING_ONLY', evidence, paths)
    try:
        with zero_fit() as counter:
            predictor = StructuralPredictor(bundle)
            samples = sample_metadata(paths[stage+'_samples'])[META]
            inputs = file_identities({k: v for k, v in paths.items() if k != stage+'_samples'})
            contract = build_inference_source_contract(inputs, semantic_contract_sha256=predictor.a['contract_digests']['semantic_contract_sha256'])
            if contract != predictor.contract:
                raise ContractError('BLOCKED_SOURCE_CONTRACT_CHANGE: register official version separately; migration not authorized')
            op, burden, _ = _load_process_sources(paths, predictor.a['semantic'], predictor.a['features'])
            pred, parts = checked_predictions(predictor, samples, op, burden, contract)
            spec = importlib.util.spec_from_file_location('v13_original_r2', 'scripts/optimization_v4_cold_predict.py')
            comparator = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparator)
            baseline = comparator.predict(bundle/'base_R2', config, stage)
            a, b = [p.set_index('sample_id').sort_index() for p in (baseline, parts)]
            if not a.index.equals(b.index):
                raise ContractError('R2 stage IDs differ')
            delta = float(np.abs(a[PRED].to_numpy()-b[PRED].to_numpy()).max())
            if not np.isfinite(delta) or delta >= 1e-8:
                raise ContractError('original R2 comparison exceeds frozen tolerance')
            audit = source_audit(samples, predictor, op, burden)
        verify(manifest)
        with output.open('x', encoding='utf-8') as f: pred.to_csv(f, index=False)
        parts.to_csv(root/'R2_rate_parts.csv', index=False)
        atomic_write_json(root/'source_feature_audit.json', audit)
        status = {'status': 'PREVIEW_ENGINEERING_ONLY', 'stage': stage, 'engineering_valid': True,
                  'official_data_identity_verified': False, 'quality_evaluated': False, 'platform_verified': False,
                  'rows': len(samples), 'sample_ids_sha256': stable_digest(samples.sample_id.astype(str).tolist()),
                  'schema': list(samples), 'reference_min': str(samples.reference_time.min()),
                  'reference_max': str(samples.reference_time.max()), 'model_cutoff': predictor.m['training']['fit_cutoff'],
                  'reverse_exact': True, 'chunks_exact': True, 'single_and_subset_exact': True,
                  'R2_original_loader_max_difference': delta, 'rate_fallback_rows': int((parts.pred_rate <= 1e-6).sum()),
                  'source_contract': predictor.contract, 'inputs': inputs, **counter,
                  'ZIP_created': False, 'uploaded': False, 'original_bundle_unchanged': True}
        atomic_write_json(str(output)+'.json', status)
        print(f'OPT28 {stage}: {len(samples)} rows PREVIEW_ENGINEERING_ONLY; fits=0', flush=True)
        return status
    except Exception as exc:
        atomic_write_json(root/'failure.json', {'exception': repr(exc), 'engineering_valid': False,
                          'official_data_identity_verified': False, 'quality_evaluated': False, 'platform_verified': False})
        raise

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=STAGES)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--data-config', required=True)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(); main(a.bundle, a.data_config, a.stage, a.output)
