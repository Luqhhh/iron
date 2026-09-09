"""OPT-14: verified saved-component inference. No training or official label reads."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from ..artifacts import atomic_write_json, file_sha256, file_identities, stable_digest, verify_file_identities, build_inference_source_contract
from ..config import load_yaml
from ..exceptions import ContractError
from ..io import read_csv
from ..models.baseline import DualTargetBaseline
from ..offline import _load_process_sources
from ..schema import validate_history
from .final_lifecycle import algorithm, label_metadata, eligible
from .refresh_factorial import Context, stamp, snapshot_identity

META = ['sample_id', 'spout_no', 'reference_time']
PRED = ['pred_tap_iron', 'pred_tap_time_len']
ROLES = {'O0': ('raw','E09_PROCESS_CHANGE_E02'), 'H0': ('raw','E04'),
         'OR': ('R2','E09_PROCESS_CHANGE_E02'), 'HR': ('R2','E04')}


def read_json(path):
    return json.loads(Path(path).read_text())


def units(registration):
    result = [(f'O2024{m:02d}_H{h}', stamp(m), h,
               stamp(m)+pd.DateOffset(months=h-1), stamp(m)+pd.DateOffset(months=h))
              for m,n in registration['origins'].items() for h in range(1,n+1)]
    return result + [('DEV_LONG',stamp(7),None,stamp(7),stamp(11)),
                     ('DEV_SHORT',stamp(9),None,stamp(9),stamp(11))]


@contextmanager
def forbid_fit(counter):
    from catboost import CatBoostRegressor
    def reject(*args, **kwargs):
        counter['attempted_target_fits'] += 1
        raise ContractError('OPT-14 forbids fit; repair missing artifacts separately')
    with patch.object(DualTargetBaseline,'fit',reject), patch.object(CatBoostRegressor,'fit',reject), patch.object(Context,'fit',reject):
        yield


def inventory(source, registration):
    cache = [json.loads(x) for x in (source/'cache_reuse.jsonl').read_text().splitlines()]
    registry = [json.loads(x) for x in (source/'registry.jsonl').read_text().splitlines()]
    result = {}
    for month in registration['origins']:
        cutoff = stamp(month)
        for role,(variant,component) in ROLES.items():
            if variant == 'raw':
                matches = [r for r in cache if pd.Timestamp(r['cutoff']) == cutoff]
                if len(matches) != 1:
                    raise ContractError('missing or ambiguous raw cache identity')
                directory = Path(matches[0]['root'])/component
                digest = matches[0]['bundles'][component]
            else:
                matches = [r for r in registry if r.get('variant') == variant and
                           r.get('component') == component and pd.Timestamp(r['fit_cutoff']) == cutoff]
                if len(matches) != 1:
                    raise ContractError('missing or ambiguous R2 registry identity')
                directory = source/'models'/variant/cutoff.strftime('%Y%m%dT%H%M%S')/component
                digest = matches[0]['bundle_sha256']
            if not (directory/'bundle.json').is_file() or file_sha256(directory/'bundle.json') != digest:
                raise ContractError('missing component or registered bundle hash mismatch')
            result[f'{month}/{role}'] = dict(path=str(directory),bundle_sha256=digest,
                cutoff=str(cutoff),variant=variant,component=component,role=role)
    return result


def load_verified_component(identity, a, contract, metadata):
    path = Path(identity['path'])
    if not (path/'bundle.json').is_file() or file_sha256(path/'bundle.json') != identity['bundle_sha256']:
        raise ContractError('registered component missing or changed')
    model = DualTargetBaseline.load(path)
    md = model.bundle_metadata_
    cutoff = pd.Timestamp(identity['cutoff'])
    tr = md['training']
    if any(pd.Timestamp(tr[k]) != cutoff for k in ('fit_cutoff','history_cutoff','label_available_cutoff')):
        raise ContractError('component cutoff mismatch')
    if tr.get('variant','raw') != identity['variant']:
        raise ContractError('component variant mismatch')
    if tr.get('component',identity['component']) != identity['component']:
        raise ContractError('component role mismatch')
    if md['inference_source_contract'] != contract or md['contract_digests'] != a['contract_digests']:
        raise ContractError('component source contract mismatch')
    if model.parameters != a['baseline']['parameters'] or md['feature_config'] != a['features']:
        raise ContractError('component parameter or feature contract mismatch')
    training = eligible(metadata,cutoff)
    if tr['sample_ids_sha256'] != stable_digest(training.sample_id.astype(str).tolist()):
        raise ContractError('component training sample identity mismatch')
    relevant = {k:v for k,v in md['code_identity'].items() if k.startswith('src/bf_tap/features/') or
                k.startswith('src/bf_tap/models/') or k in (
                    'src/bf_tap/optimization/history_stable.py','src/bf_tap/optimization/features.py',
                    'src/bf_tap/optimization/process_change.py','src/bf_tap/optimization/final_lifecycle.py',
                    'src/bf_tap/availability.py') or k.startswith('configs/')}
    if not relevant:
        raise ContractError('component lacks source identity')
    verify_file_identities(relevant)
    history = model.load_history_snapshot()
    validate_history(history)
    if (history.available_at > cutoff).any() or (history.reference_time >= cutoff).any():
        raise ContractError('component has future historical results')
    if set(history.sample_id.astype(str)) != set(training.sample_id.astype(str)):
        raise ContractError('component history sample identity mismatch')
    h = history.set_index('sample_id').sort_index()
    t = training.set_index('sample_id').sort_index()
    for column in ('reference_time','spout_no'):
        if not np.array_equal(h[column].astype(str),t[column].astype(str)):
            raise ContractError('history metadata differs from official source')
    if not np.array_equal(h.available_at.astype(str),t.label_available_at.astype(str)):
        raise ContractError('history availability differs from official source')
    details = {**identity, 'history_snapshot_sha256':file_sha256(path/'history_snapshot.csv'),
        'history_values_sha256':snapshot_identity(history), 'training_samples_sha256':tr['sample_ids_sha256'],
        'schema_sha256':stable_digest(model.feature_schema_), 'feature_schema':model.feature_schema_,
        'model_files_sha256':md['component_sha256'], 'source_contract_sha256':stable_digest(contract),
        'training_history_policy':'original_per_sample_asof_and_fit_cutoff',
        'verified_source_files':relevant}
    return model,history,details


class ComponentFeatures:
    # Reuse the old pure feature-cache method, never its initialization or fit path.
    def __init__(self,a,op,burden):
        self.a,self.op,self.burden = a,op,burden
        self.cache,self.public_metadata = {},{}
        self.public_features = None

    def X(self,samples,identity,history):
        if list(samples) != META or samples.sample_id.isna().any() or samples.sample_id.duplicated().any():
            raise ContractError('inference accepts unique metadata only, never labels')
        cutoff = pd.Timestamp(identity['cutoff'])
        if (samples.reference_time < cutoff).any() or (history.available_at > cutoff).any():
            raise ContractError('future history or model for prediction reference')
        if set(samples.sample_id.astype(str)) & set(history.sample_id.astype(str)):
            raise ContractError('current evaluation sample appears in history')
        return Context.X(self,samples,cutoff,identity['component'],identity['variant'],history)


def run(data_config,output):
    output.mkdir(parents=True,exist_ok=False)
    counter = {'attempted_target_fits':0,'completed_target_fits':0}
    try:
        registration = load_yaml('configs/optimization_v0_5/experiment.yaml')
        scope = load_yaml('configs/optimization_v0_5/access_scope.yaml')
        source = Path(registration['source_run'])
        source_manifest = read_json(source/'factorial_manifest.json')
        if read_json(source/'final_status.json').get('G0') != 'PASS_RESEARCH_EXECUTION':
            raise ContractError('source research run is not complete')
        if read_json('EVIDENCE_STATUS.json').get('holdout_consumed') is not True or scope['holdout_consumed'] is not True:
            raise ContractError('post-consumption development requires consumed holdout')
        a = algorithm()
        if stable_digest(a) != stable_digest(source_manifest['algorithm']):
            raise ContractError('old algorithm identity changed')
        paths = load_yaml(data_config)['paths']
        keys = ('train_samples','tap_history_train','operation_hourly','burden_change','data_dictionary')
        inputs = file_identities({k:paths[k] for k in keys})
        for k,v in inputs.items():
            if any(v[x] != source_manifest['inputs'][k][x] for x in ('sha256','bytes')):
                raise ContractError('official input differs from v4 source contract')
        entries = inventory(source,registration)
        evidence_paths = {str(source/n):source/n for n in ('factorial_manifest.json','final_status.json','registry.jsonl','cache_reuse.jsonl')}
        for unit,*_ in units(registration):
            for c in ('E12-raw','R2','B0','B1'):
                p = source/'units'/unit/c/'errors.csv'
                evidence_paths[str(p)] = p
        evidence = file_identities(evidence_paths)
        old_ledger = Path(scope['old_ledger'])
        if not any(json.loads(x)['lifecycle']=='holdout_scoring' for x in old_ledger.read_text().splitlines()):
            raise ContractError('old consumption ledger missing')
        sources = file_identities({str(p):p for p in [*Path('src/bf_tap').rglob('*.py'),
                                   *Path('configs/optimization_v0_5').glob('*.yaml')]})
        manifest = dict(registration=registration,access_scope=scope,inputs=inputs,evidence=evidence,
            inventory=entries,source_files=sources,old_ledger_sha256=file_sha256(old_ledger),
            active_release_sha256=file_sha256('configs/optimization_v0_4/active_release.yaml'),
            official_target_columns_read=False,test_a_read=False)
        atomic_write_json(output/'manifest.json',manifest)
        ledger = Path(scope['new_ledger']);ledger.parent.mkdir(parents=True,exist_ok=True)
        with ledger.open('a+') as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX);f.seek(0);prior=f.read().splitlines()
            f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),purpose=scope['purpose'],
                authorization=scope['authorization'],manifest_sha256=file_sha256(output/'manifest.json'),
                previous_sha256=stable_digest(prior[-1]) if prior else None,holdout_consumed=True))+'\n');f.flush()
        metadata = label_metadata(paths,a['semantic'])
        metadata = metadata.loc[(metadata.reference_time >= pd.Timestamp(scope['reference_start'])) &
                                (metadata.reference_time < pd.Timestamp(scope['reference_end_exclusive']))]
        op,burden,_ = _load_process_sources(paths,a['semantic'],a['features'])
        contract = build_inference_source_contract(inputs,semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
        builder = ComponentFeatures(a,op,burden)
        loaded,identities,rows = {},{},[]
        with forbid_fit(counter):
            for key,identity in entries.items():
                model,history,details = load_verified_component(identity,a,contract,metadata)
                loaded[key] = model,history
                identities[key] = details
            atomic_write_json(output/'component_identities.json',identities)
            for unit,cutoff,horizon,start,end in units(registration):
                p = source/'units'/unit/'R2'/'errors.csv'
                samples = read_csv(p,usecols=META,time_columns=['reference_time'])[META]
                if samples.empty or not ((samples.reference_time>=start)&(samples.reference_time<end)).all():
                    raise ContractError('evaluation unit time boundary mismatch')
                for role in ROLES:
                    key = f'{cutoff.month}/{role}'
                    model,history = loaded[key]
                    X = builder.X(samples,entries[key],history)
                    raw = model.predict_raw(X)
                    reordered = model.predict_raw(X.iloc[::-1]).iloc[::-1]
                    if not np.array_equal(raw.to_numpy(),reordered.to_numpy()):
                        raise ContractError('component prediction changes under input reorder')
                    result = samples.copy()
                    for col in PRED:
                        result['raw_'+col] = raw[col].to_numpy()
                        result[col] = raw[col].clip(lower=0).to_numpy()
                    result['unit'],result['role'] = unit,role
                    result['fit_cutoff'],result['horizon'] = str(cutoff),horizon
                    rows.append(result)
                print(f'exported {unit}: four verified components; fits=0',flush=True)
        pd.concat(rows,ignore_index=True).to_csv(output/'component_predictions.csv',index=False)
        verify_file_identities(inputs);verify_file_identities(evidence);verify_file_identities(sources)
        for identity in entries.values():
            if file_sha256(Path(identity['path'])/'bundle.json') != identity['bundle_sha256']:
                raise ContractError('component changed during export')
        if file_sha256(old_ledger) != manifest['old_ledger_sha256']:
            raise ContractError('old access ledger changed')
        atomic_write_json(output/'final_status.json',dict(G0='PASS_COMPONENT_EXPORT',**counter,
            units=len(units(registration)),components=len(entries),input_reorder_verified=True,
            official_target_columns_read=False,test_a_read=False,holdout_consumed=True,
            predictions_sha256=file_sha256(output/'component_predictions.csv')))
    except Exception as exc:
        atomic_write_json(output/'final_status.json',dict(status='FAILED',error=str(exc),**counter))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config',default='configs/data.local.yaml')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.data_config,args.output)

if __name__=='__main__': main()
