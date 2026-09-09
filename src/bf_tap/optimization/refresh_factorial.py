"""OPT-11 isolated inference evaluator and post-consumption development runner."""
from __future__ import annotations
import argparse
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from ..artifacts import atomic_write_json, file_sha256, file_identities, stable_digest, verify_file_identities, runtime_environment
from ..config import load_yaml
from ..exceptions import ContractError
from ..io import parse_local_time
from ..models.baseline import DualTargetBaseline
from ..models.sanity import MedianControls
from ..offline import _load_process_sources
from ..schema import validate_samples, validate_history, validate_cross_table_consistency
from ..artifacts import build_inference_source_contract
from .final_lifecycle import algorithm, source_files, sample_metadata, label_metadata, eligible, _read_eligible_rows, feature_frame, component_features
from .history_stable import stable_features
from .snapshot_ensemble import blend
from .validation import error_contributions, aggregate_grid, evaluate_acceptance
from ..metrics import score_predictions
from ..features.history import build_history_features
from ..availability import assert_no_future

TARGETS = ('tap_iron', 'tap_time_len')
COMPONENTS = ('E09_PROCESS_CHANGE_E02', 'E04')

def stamp(month):
    return pd.Timestamp(f'2024-{month:02d}-01', tz='Asia/Shanghai')


def decomposition(loss):
    a,b,c,d = [loss[k] for k in ('00','01','10','11')]
    result = dict(model=((c-a)+(d-b))/2, history=((b-a)+(d-c))/2,
                  joint=d-a, interaction=d-c-b+a)
    if not np.isclose(result['model']+result['history'], result['joint'], atol=1e-14):
        raise ContractError('factorial identity failed')
    return result


def score_factorial(frame):
    required = {'eval_month','sample_id','model_state','history_state', *TARGETS, *['pred_'+t for t in TARGETS]}
    if required-set(frame) or frame.empty or frame[list(required)].isna().any().any():
        raise ContractError('incomplete factorial inputs')
    frame = frame.copy()
    frame['sample_id'] = frame.sample_id.astype(str)
    if not set(frame.model_state).issubset({'old','new'}) or not set(frame.history_state).issubset({'old','new'}):
        raise ContractError('unknown factorial state')
    frame['arm'] = frame.model_state.map({'old':'0','new':'1'})+frame.history_state.map({'old':'0','new':'1'})
    nums = frame[[*TARGETS,*['pred_'+t for t in TARGETS]]].to_numpy(float)
    if not np.isfinite(nums).all() or (nums<0).any():
        raise ContractError('factorial requires finite nonnegative values')
    output = {}
    for month, rows in frame.groupby('eval_month'):
        arms = {}
        for arm in ('00','01','10','11'):
            part = rows.loc[rows.arm == arm].set_index('sample_id').sort_index()
            if part.empty or part.index.duplicated().any():
                raise ContractError('missing or duplicate arm')
            arms[arm] = part
        first = arms['00']
        for part in arms.values():
            if not first.index.equals(part.index) or not first[list(TARGETS)].equals(part[list(TARGETS)]):
                raise ContractError('factorial IDs or labels mismatch')
        metrics = {}
        for arm, part in arms.items():
            metrics[arm] = {}
            for target in TARGETS:
                denominator = float(part[target].sum())
                if denominator <= 0:
                    raise ContractError('nonpositive target sum')
                error = part['pred_'+target]-part[target]
                metrics[arm][target] = dict(abs_error_sum=float(error.abs().sum()), target_sum=denominator,
                                           wmape=float(error.abs().sum()/denominator), signed_error_sum=float(error.sum()))
            metrics[arm]['E'] = float(np.mean([metrics[arm][t]['wmape'] for t in TARGETS]))
        output[str(month)] = dict(samples=len(first), arms=metrics,
            decomposition={**{t:decomposition({a:metrics[a][t]['wmape'] for a in arms}) for t in TARGETS},
                           'E':decomposition({a:metrics[a]['E'] for a in arms})})
    return output


def snapshot_identity(history):
    return stable_digest(history.sort_values('sample_id').astype(str).to_dict('records'))


def validate_snapshot(snapshot, official, cutoff, samples):
    validate_history(snapshot)
    if samples.empty or (snapshot.available_at > cutoff).any() or (samples.reference_time < cutoff).any():
        raise ContractError('future snapshot or prediction reference')
    if set(snapshot.sample_id.astype(str)) & set(samples.sample_id.astype(str)):
        raise ContractError('evaluation IDs intersect inference history')
    known = official.set_index('sample_id').sort_index()
    supplied = snapshot.set_index('sample_id').sort_index()
    if not supplied.index.isin(known.index).all() or not supplied.equals(known.loc[supplied.index, supplied.columns]):
        raise ContractError('snapshot differs from official source')


class Context:
    def __init__(self, data_config, output):
        self.output = output
        self.a = algorithm()
        self.registration = load_yaml('configs/optimization_v0_4/experiment.yaml')
        scope = load_yaml('configs/optimization_v0_4/access_scope.yaml')
        status = json.loads(Path('EVIDENCE_STATUS.json').read_text())
        if not status.get('holdout_consumed') or scope['holdout_consumed'] is not True:
            raise ContractError('post-consumption access requires consumed global holdout')
        old = Path(scope['old_ledger'])
        records = [json.loads(x) for x in old.read_text().splitlines()]
        if not {'holdout_scoring','final_training'}.issubset({x['lifecycle'] for x in records}):
            raise ContractError('missing historical consumption evidence')
        self.paths = load_yaml(data_config)['paths']
        keys = ('train_samples','tap_history_train','operation_hourly','burden_change','data_dictionary','test_a_samples')
        self.inputs = file_identities({k:self.paths[k] for k in keys})
        self.sources = file_identities({**source_files(), **{str(p):p for p in Path('configs/optimization_v0_4').glob('*.yaml')}})
        self.release_cutoff = sample_metadata(self.paths['test_a_samples']).reference_time.min()
        meta = label_metadata(self.paths, self.a['semantic'])
        selected = meta.loc[(meta.reference_time >= pd.Timestamp(scope['reference_start'])) &
                            (meta.reference_time < pd.Timestamp(scope['reference_end_exclusive'])) &
                            (meta.label_available_at <= self.release_cutoff)]
        self.manifest = dict(registration=self.registration, access_scope=scope, algorithm=self.a,
            inputs=self.inputs, source_files=self.sources, old_ledger_sha256=file_sha256(old),
            eligible_ids_sha256=stable_digest(selected.sample_id.astype(str).tolist()),
            model_fit_cutoffs=[str(stamp(m)) for m in range(5,12)],
            release_anchors=[str(stamp(11)), str(self.release_cutoff)],
            training_history_policy='per_sample_as_of_and_model_cutoff', holdout_consumed=True,
            evaluation_identity='RETROSPECTIVE_DEVELOPMENT_POST_HOLDOUT_CONSUMPTION')
        atomic_write_json(output/'factorial_manifest.json', self.manifest)
        ledger = Path(scope['new_ledger'])
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open('a+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            previous = handle.read().splitlines()
            entry = dict(purpose=scope['purpose'], authorization=scope['authorization'],
                at=datetime.now(timezone.utc).isoformat(), manifest_sha256=file_sha256(output/'factorial_manifest.json'),
                holdout_consumed=True, previous_entry_sha256=stable_digest(previous[-1]) if previous else None)
            handle.write(json.dumps(entry,sort_keys=True)+'\n')
            handle.flush()
        self.labels = _read_eligible_rows(self.paths['train_samples'], selected.sample_id).merge(
            selected[['sample_id','label_available_at']], on='sample_id',validate='one_to_one').sort_values(['reference_time','sample_id'],kind='mergesort')
        self.history = _read_eligible_rows(self.paths['tap_history_train'], selected.sample_id)
        available = self.a['semantic']['targets']['available_at_column']
        for c in {available, self.a['semantic']['sources']['history']['end_time_column']}:
            self.history[c] = parse_local_time(self.history[c],c)
        self.history['available_at'] = self.history[available]
        validate_samples(self.labels,labeled=True)
        validate_history(self.history)
        validate_cross_table_consistency(self.labels,self.history)
        self.op,self.burden,_ = _load_process_sources(self.paths,self.a['semantic'],self.a['features'])
        self.cache = {}
        self.public_features = None
        self.public_metadata = {}
        self.models = {}
        self.fit_count = 0
        self.contract = build_inference_source_contract(self.inputs,
            semantic_contract_sha256=self.a['contract_digests']['semantic_contract_sha256'])

    def X(self,samples,cutoff,component,variant='raw',history=None):
        h = self.history if history is None else history
        key=(str(cutoff),snapshot_identity(h),stable_digest(samples[['sample_id','spout_no','reference_time']].astype(str).to_dict('records')),tuple(samples.index))
        if key not in self.cache:
            identities = {str(r.sample_id):(str(r.spout_no),str(r.reference_time)) for r in samples.itertuples()}
            if len(identities) != len(samples):
                raise ContractError('duplicate feature sample IDs')
            for sid, metadata in identities.items():
                if sid in self.public_metadata and self.public_metadata[sid] != metadata:
                    raise ContractError('cached public sample metadata differs')
            missing = samples.loc[~samples.sample_id.astype(str).isin(self.public_metadata)]
            if len(missing):
                computed = feature_frame(missing,h,self.op,self.burden,cutoff,self.a)
                computed.index = missing.sample_id.astype(str)
                self.public_features = computed if self.public_features is None else pd.concat([self.public_features,computed])
                self.public_metadata.update({str(r.sample_id):(str(r.spout_no),str(r.reference_time)) for r in missing.itertuples()})
            frame = self.public_features.loc[samples.sample_id.astype(str)].copy()
            frame.index = samples.index
            history_features, audit = build_history_features(samples,h,fit_cutoff=cutoff,
                last_k=tuple(self.a['features']['history']['last_k_mean']))
            assert_no_future(audit)
            if set(history_features) != {c for c in frame if c.startswith('history__')}:
                raise ContractError('history cache replacement is incomplete')
            frame.loc[:,history_features.columns] = history_features
            self.cache[key] = frame
        base = component_features(self.cache[key],component,self.a)
        return stable_features(base,samples,h,cutoff,variant)

    def reuse(self, cutoff, train, snapshot, variant):
        if variant != 'raw':
            return None
        candidates = []
        if cutoff == self.release_cutoff:
            candidates.append(Path('local/runs/optimization-v0.3-r2-opt10-final-r1/bundle'))
        if cutoff.day == 1 and cutoff.hour == 0 and 8 <= cutoff.month <= 11:
            candidates.append(Path(f'local/runs/optimization-v0.3-r2-opt10-prepared-r1/HOLDOUT_H{12-cutoff.month}/bundle'))
        for previous in ('r2','r3'):
            candidates.append(Path(f'local/runs/optimization-v0.4-{previous}/models/raw')/cutoff.strftime('%Y%m%dT%H%M%S'))
        for root in candidates:
            if not all((root/c/'bundle.json').is_file() for c in COMPONENTS):
                continue
            result = {}
            try:
                for c in COMPONENTS:
                    model = DualTargetBaseline.load(root/c)
                    md = model.bundle_metadata_
                    tr = md['training']
                    if any(pd.Timestamp(tr[k]) != cutoff for k in ('fit_cutoff','history_cutoff','label_available_cutoff')):
                        raise ContractError('cache cutoff mismatch')
                    if md['inference_source_contract'] != self.contract:
                        raise ContractError('cache process source contract mismatch')
                    if model.parameters != self.a['baseline']['parameters'] or md['contract_digests'] != self.a['contract_digests']:
                        raise ContractError('cache parameter or contract mismatch')
                    if tr['sample_ids_sha256'] != stable_digest(train.sample_id.astype(str).tolist()):
                        raise ContractError('cache sample mismatch')
                    if snapshot_identity(model.load_history_snapshot()) != snapshot_identity(snapshot):
                        raise ContractError('cache history mismatch')
                    # Verify original model and feature implementation identity. The new evaluator
                    # may evolve without changing the old training/feature code being reused.
                    relevant = {k:v for k,v in md['code_identity'].items() if
                        k.startswith('src/bf_tap/features/') or k.startswith('src/bf_tap/models/') or
                        k in ('src/bf_tap/optimization/features.py','src/bf_tap/optimization/process_change.py',
                              'src/bf_tap/optimization/final_lifecycle.py','src/bf_tap/availability.py') or k.startswith('configs/')}
                    if not relevant:
                        raise ContractError('cache lacks source identity')
                    verify_file_identities(relevant)
                    X = self.X(train,cutoff,c)
                    if model.feature_names_ != list(X) or model.feature_schema_ != [
                        dict(name=k,dtype=str(X[k].dtype),categorical=k in model.categorical) for k in X]:
                        raise ContractError('cache feature mismatch')
                    result[c] = model
                with (self.output/'cache_reuse.jsonl').open('a') as f:
                    f.write(json.dumps(dict(cutoff=str(cutoff),root=str(root),
                        bundles={c:file_sha256(root/c/'bundle.json') for c in COMPONENTS},
                        validated=['parameters','contracts','training_samples','history_values','feature_schema','source_files'],
                        training_history_policy='original_per_sample_asof',actual_target_fits=0))+'\n')
                print(f'reused verified raw anchor {cutoff}',flush=True)
                return result
            except ContractError as exc:
                with (self.output/'cache_rejections.jsonl').open('a') as f:
                    f.write(json.dumps(dict(root=str(root),reason=str(exc)))+'\n')
        return None

    def fit(self,cutoff,variant='raw'):
        key=(str(cutoff),variant)
        if key in self.models:
            return self.models[key]
        train=eligible(self.labels,cutoff)
        snapshot=self.history.loc[self.history.available_at <= cutoff].copy()
        reused = self.reuse(cutoff,train,snapshot,variant)
        if reused is not None:
            self.models[key] = reused
            return reused
        root=self.output/'models'/variant/cutoff.strftime('%Y%m%dT%H%M%S')
        result={}
        for c in COMPONENTS:
            X=self.X(train,cutoff,c,variant)
            model=DualTargetBaseline(self.a['baseline']['parameters'],('spout_no',))
            model.fit(X,train[list(TARGETS)])
            training=dict(fit_cutoff=str(cutoff),history_cutoff=str(cutoff),label_available_cutoff=str(cutoff),
                model_fit_cutoff=str(cutoff),training_label_available_cutoff=str(cutoff),
                training_history_policy='per_sample_as_of_and_model_cutoff',variant=variant,
                sample_ids_sha256=stable_digest(train.sample_id.astype(str).tolist()),rows=len(train))
            model.save(root/c,history_snapshot=snapshot,metadata=dict(baseline_config=self.a['baseline'],
                feature_config=self.a['features'],semantic_contract=self.a['semantic'],contract_digests=self.a['contract_digests'],
                inference_source_contract=self.contract,training=training,code_identity=self.sources,
                environment=runtime_environment(),lockfile_sha256=file_sha256('uv.lock')))
            loaded=DualTargetBaseline.load(root/c)
            if not np.array_equal(model.predict_raw(X).to_numpy(),loaded.predict_raw(X).to_numpy()):
                raise ContractError('saved model mismatch')
            result[c]=loaded
            self.fit_count+=2
            with (self.output/'registry.jsonl').open('a') as f:
                f.write(json.dumps(dict(**training,component=c,actual_target_fits=2,bundle_sha256=file_sha256(root/c/'bundle.json')))+'\n')
            print(f'fit {variant} {cutoff} {c}: {len(train)} rows; target fits={self.fit_count}',flush=True)
        self.models[key]=result
        return result

    def predict(self,samples,model_cutoff,history_cutoff,variant='raw',history=None):
        key=(str(model_cutoff),variant)
        if key not in self.models:
            raise ContractError('prediction cannot fit a missing endpoint')
        if samples.reference_time.min() < model_cutoff:
            raise ContractError('prediction precedes model fit cutoff')
        snapshot=self.history.loc[self.history.available_at<=history_cutoff].copy() if history is None else history
        validate_snapshot(snapshot,self.history,history_cutoff,samples)
        predictions={}
        for c,model in self.models[key].items():
            p=model.predict_raw(self.X(samples,history_cutoff,c,variant,snapshot)).clip(lower=0)
            p.insert(0,'sample_id',samples.sample_id.astype(str))
            predictions[c]=p
        predictions['E12-raw']=blend(predictions[COMPONENTS[0]],predictions[COMPONENTS[1]],.8)
        with (self.output/'inference_registry.jsonl').open('a') as f:
            f.write(json.dumps(dict(model_fit_cutoff=str(model_cutoff),training_label_available_cutoff=str(model_cutoff),
                training_history_policy='per_sample_as_of_and_model_cutoff',inference_history_cutoff=str(history_cutoff),
                inference_history_snapshot_sha256=snapshot_identity(snapshot),source_contract_sha256=stable_digest(self.contract),
                prediction_reference_time=samples[['sample_id','reference_time']].astype(str).to_dict('records'),variant=variant))+'\n')
        return predictions


def factorial(ctx):
    for m in range(6,12):
        ctx.fit(stamp(m))
    rows=[]
    for m in range(7,12):
        samples=ctx.labels.loc[(ctx.labels.reference_time>=stamp(m)) & (ctx.labels.reference_time<stamp(m+1))]
        for mi,mc in enumerate((stamp(m-1),stamp(m))):
            for hi,hc in enumerate((stamp(m-1),stamp(m))):
                predictions=ctx.predict(samples,mc,hc)
                for c,p in predictions.items():
                    part=error_contributions(samples,p)
                    part['component']=c
                    part['eval_month']=f'2024-{m:02d}'
                    part['model_state']=('old','new')[mi]
                    part['history_state']=('old','new')[hi]
                    rows.append(part)
        print(f'factorial month {m} complete',flush=True)
    frame=pd.concat(rows,ignore_index=True)
    frame.to_csv(ctx.output/'factorial_predictions.csv',index=False)
    report={c:score_factorial(f) for c,f in frame.groupby('component')}
    atomic_write_json(ctx.output/'factorial_scores.json',report)
    atomic_write_json(ctx.output/'factorial_by_spout.json',{
        f'{c}/{s}':score_factorial(f) for (c,s),f in frame.groupby(['component','spout_no'])})
    comparisons=[]
    old=Path('local/runs/optimization-v0.3-opt07-reference-r2')
    for m in range(7,11):
        for state,origin,horizon in [('old',m-1,2),('new',m,1)]:
            for c,old_c in [(COMPONENTS[0],'E09'),('E04','E04'),('E12-raw','E12-raw')]:
                path=old/'units'/f'O2024{origin:02d}_H{horizon}'/old_c/'errors.csv'
                prior=pd.read_csv(path,dtype={'sample_id':str}).set_index('sample_id').sort_index()
                now=frame.loc[(frame.eval_month==f'2024-{m:02d}') & (frame.model_state==state) &
                    (frame.history_state==state) & (frame.component==c)].set_index('sample_id').sort_index()
                cols=[*TARGETS,*['pred_'+t for t in TARGETS]]
                if not prior.index.equals(now.index):
                    raise ContractError('legacy endpoint IDs differ')
                delta=float(np.abs(prior[cols].to_numpy()-now[cols].to_numpy()).max())
                comparisons.append(dict(month=m,state=state,component=c,max_abs_difference=delta,source_sha256=file_sha256(path)))
    atomic_write_json(ctx.output/'legacy_endpoint_alignment.json',comparisons)
    if max(r['max_abs_difference'] for r in comparisons)>1e-9:
        raise ContractError('legacy endpoint predictions differ')
    return frame


def extended(ctx):
    ctx.fit(stamp(5))
    for variant in ('R1','R2'):
        for m in range(6,12):
            ctx.fit(stamp(m),variant)
    metrics={}
    units=[(f'O2024{m:02d}_H{h}',m,h,stamp(m)+pd.DateOffset(months=h-1),stamp(m)+pd.DateOffset(months=h))
           for m,n in ctx.registration['origins'].items() for h in range(1,n+1)]
    units += [('DEV_LONG',7,None,stamp(7),stamp(11)),('DEV_SHORT',9,None,stamp(9),stamp(11))]
    for unit,m,h,start,end in units:
        samples=ctx.labels.loc[(ctx.labels.reference_time>=start)&(ctx.labels.reference_time<end)]
        raw=ctx.predict(samples,stamp(m),stamp(m))
        pred={'E09':raw[COMPONENTS[0]],'E12-raw':raw['E12-raw']}
        for variant in ('R1','R2'):
            pred[variant]=ctx.predict(samples,stamp(m),stamp(m),variant)['E12-raw']
        pred['R3']=blend(raw['E12-raw'],ctx.predict(samples,stamp(m-1),stamp(m-1))['E12-raw'])
        controls=MedianControls(min_group_count=20).fit(eligible(ctx.labels,stamp(m)))
        pred.update({c:controls.predict(samples,c) for c in ('B0','B1')})
        metrics[unit]=dict(origin_id=f'O2024{m:02d}',horizon=h,candidates={})
        for c,p in pred.items():
            dest=ctx.output/'units'/unit/c
            dest.mkdir(parents=True,exist_ok=False)
            error_contributions(samples,p).to_csv(dest/'errors.csv',index=False)
            metrics[unit]['candidates'][c]={'overall':score_predictions(samples,p)}
        print(f'extended {unit} complete',flush=True)
    summary=aggregate_grid(metrics)
    mapped={k:{**v,'candidates':{**v['candidates'],'E00':v['candidates']['E12-raw']}} for k,v in metrics.items()}
    gate=evaluate_acceptance(mapped,{**summary,'E00':summary['E12-raw']},ctx.registration['acceptance'])
    gate['candidates']={k:v for k,v in gate['candidates'].items() if k in ('R1','R2','R3')}
    gate['status']='PASS' if any(v['pass'] for v in gate['candidates'].values()) else 'FAIL'
    atomic_write_json(ctx.output/'candidate_metrics.json',metrics)
    atomic_write_json(ctx.output/'extended_18_grid_summary.json',summary)
    legacy={k:v for k,v in metrics.items() if k.startswith('O') and int(k[5:7])+v['horizon']<=11}
    atomic_write_json(ctx.output/'legacy_14_grid_summary.json',aggregate_grid(legacy))
    atomic_write_json(ctx.output/'acceptance.json',gate)
    return gate


def sensitivity(ctx):
    samples=sample_metadata(ctx.paths['test_a_samples'])
    results=[]
    for cutoff in (stamp(11),ctx.release_cutoff):
        if (str(cutoff),'raw') not in ctx.models:
            snapshot = ctx.history.loc[ctx.history.available_at<=cutoff].copy()
            reused = ctx.reuse(cutoff,eligible(ctx.labels,cutoff),snapshot,'raw')
            if reused is None:
                raise ContractError('sensitivity requires an existing verified anchor; cannot fit')
            ctx.models[(str(cutoff),'raw')] = reused
        snapshot=ctx.history.loc[ctx.history.available_at<=cutoff].sort_values(['available_at','reference_time','sample_id'],kind='mergesort')
        base=ctx.predict(samples,cutoff,cutoff)
        tail=snapshot.tail(3).sample_id.astype(str).tolist()
        masks=[('last1',[tail[-1]]),('last3',tail)]+[(f'individual_{i}',[sid]) for i,sid in enumerate(tail[:-1])]
        for label,ids in masks:
            altered=snapshot.loc[~snapshot.sample_id.astype(str).isin(ids)]
            preds=ctx.predict(samples,cutoff,cutoff,history=altered)
            for c,p in preds.items():
                a=p.set_index('sample_id').sort_index()
                b=base[c].set_index('sample_id').sort_index()
                for t in TARGETS:
                    delta=np.abs(a['pred_'+t]-b['pred_'+t])
                    results.append(dict(cutoff=str(cutoff),mask=label,removed_ids=ids,component=c,target=t,
                        median=float(delta.median()),p95=float(delta.quantile(.95)),max=float(delta.max())))
    atomic_write_json(ctx.output/'sensitivity.json',dict(scope='UNLABELED_DEPENDENCE_ONLY',results=results))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config',default='configs/data.local.yaml')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        ctx=Context(args.data_config,args.output)
        factorial(ctx)
        gate=extended(ctx)
        sensitivity(ctx)
        verify_file_identities(ctx.inputs)
        verify_file_identities(ctx.sources)
        atomic_write_json(args.output/'final_status.json',dict(G0='PASS_RESEARCH_EXECUTION',G1=gate['status'],
            holdout_consumed=True,target_fits=ctx.fit_count,incumbent='E16',
            release_status='PENDING_FULL_RELEASE_VALIDATION' if gate['status']=='PASS' else 'NO_CHALLENGER_KEEP_E16'))
    except Exception as exc:
        atomic_write_json(args.output/'final_status.json',dict(status='FAILED',error=str(exc),error_type=type(exc).__name__))
        raise

if __name__=='__main__':
    main()
