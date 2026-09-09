"""Frozen v0.6 calendar routing and verified, label-free component recovery."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..artifacts import file_sha256, stable_digest, verify_file_identities
from ..exceptions import ContractError
from ..io import parse_local_time
from ..models.baseline import DualTargetBaseline
from ..schema import validate_history
from .component_export import META, PRED, ROLES, ComponentFeatures
from .component_ablation import validate_frame
from .snapshot_ensemble import blend

S1={1:'P2',2:'P2',3:'P2',4:'R2'}
TARGETS=['tap_iron','tap_time_len']


def horizons(samples,cutoff):
    if list(samples)!=META or samples.empty or samples[META].isna().any().any() or samples.sample_id.astype(str).duplicated().any():
        raise ContractError('unique, complete metadata required')
    t=parse_local_time(samples.reference_time,'reference_time');c=pd.Timestamp(cutoff)
    if c.tzinfo is None:raise ContractError('timezone-aware cutoff required')
    c=c.tz_convert('Asia/Shanghai')
    if (t<c).any():raise ContractError('sample before model cutoff')
    h=(t.dt.year-c.year)*12+t.dt.month-c.month+1
    if not h.isin([1,2,3,4]).all():raise ContractError('unregistered calendar horizon')
    return h


def route(samples,cutoff,parts,mapping=None):
    mapping=mapping or {t:S1 for t in TARGETS}
    if set(mapping)!=set(TARGETS) or any(set(m)!=set(S1) or set(m.values())-{'P2','R2'} for m in mapping.values()):
        raise ContractError('incomplete or unknown target routing map')
    if set(parts)!={'P2','R2'}:raise ContractError('both router endpoints required')
    ids=samples.sample_id.astype(str).tolist();h=horizons(samples,cutoff)
    p={k:validate_frame(v).set_index('sample_id') for k,v in parts.items()}
    if any(set(v.index)!=set(ids) for v in p.values()):raise ContractError('router endpoint IDs differ')
    out=pd.DataFrame({'sample_id':ids})
    for target,col in zip(TARGETS,PRED):
        use=np.array([mapping[target][int(x)]=='P2' for x in h])
        out[col]=np.where(use,p['P2'].loc[ids,col],p['R2'].loc[ids,col])
    return out


def load_component(entry,algorithm,contract):
    path=Path(entry['path'])
    if file_sha256(path/'bundle.json')!=entry['bundle_sha256']:raise ContractError('bundle digest mismatch')
    model=DualTargetBaseline.load(path);md=model.bundle_metadata_;tr=md['training'];cutoff=pd.Timestamp(entry['cutoff'])
    if entry['role'] not in ROLES or (entry['variant'],entry['component'])!=ROLES[entry['role']]:raise ContractError('role/variant mismatch')
    if any(pd.Timestamp(tr[k])!=cutoff for k in ('fit_cutoff','history_cutoff','label_available_cutoff')) or tr.get('variant','raw')!=entry['variant'] or tr.get('component',entry['component'])!=entry['component']:raise ContractError('component training boundary mismatch')
    if md['contract_digests']!=algorithm['contract_digests'] or md['inference_source_contract']!=contract or model.parameters!=algorithm['baseline']['parameters'] or md['feature_config']!=algorithm['features']:raise ContractError('component source/algorithm mismatch')
    if stable_digest(model.feature_schema_)!=entry['schema_sha256'] or file_sha256(path/'history_snapshot.csv')!=entry['history_snapshot_sha256']:raise ContractError('schema/history identity mismatch')
    verify_file_identities(entry['verified_source_files'])
    history=model.load_history_snapshot();validate_history(history)
    if (history.available_at>cutoff).any() or (history.reference_time>=cutoff).any():raise ContractError('future history')
    if stable_digest(history.sort_values(['reference_time','sample_id'],kind='mergesort').sample_id.astype(str).tolist())!=tr['sample_ids_sha256']:raise ContractError('training/history IDs differ')
    return model,history


def predict_parts(samples,entries,loaded,builder):
    if set(entries)!={'O0','OR','HR'} or set(loaded)!=set(entries):raise ContractError('router requires three exact components')
    if len({pd.Timestamp(e['cutoff']) for e in entries.values()})!=1:raise ContractError('mixed cutoff')
    horizons(samples,next(iter(entries.values()))['cutoff'])
    parts={}
    for role,entry in entries.items():
        model,history=loaded[role];X=builder.X(samples,entry,history)
        raw=model.predict_raw(X)
        if not np.array_equal(raw.to_numpy(),model.predict_raw(X.iloc[::-1]).iloc[::-1].to_numpy()):raise ContractError('order-dependent inference')
        p=raw.clip(lower=0);p.insert(0,'sample_id',samples.sample_id.astype(str).to_numpy());parts[role]=p
    return {'P2':blend(parts['O0'],parts['HR'],.8),'R2':blend(parts['OR'],parts['HR'],.8)}
