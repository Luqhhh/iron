"""Bounded OPT-25/26 stages using existing legacy components, LAD and scoring."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import subprocess
import sys
import numpy as np
import pandas as pd
from ..artifacts import atomic_write_json,file_sha256,file_identities,stable_digest,verify_file_identities,runtime_environment
from ..config import load_yaml
from ..exceptions import ContractError
from .component_export import META,PRED,read_json,forbid_fit,units
from .dual_ratio_common import frame,verify_manifest,score,week_intervals
from .refresh_factorial import Context,stamp
from .structural import apply_correction,select_oof
from .structural_run import append_ledger,predict_inputs
from .trajectory_run import load_fold,builder_for,equal
from .trajectory_complete import verify_execution as verify_v11
from .residual_stack import ResidualFitBudget
from .rate_model import schema
from .recency_model import RecencyModel,weights,weight_audit,original_schema,TARGETS
from .recency_candidate import CANDIDATES,PARTS,isolated_directions,fit_coefficients,outputs,validate_outputs,acceptance

DATES=('reference_time','label_available_at','fold_cutoff','train_reference_max','train_available_max','history_available_max','available_at')


def archive(path):
    result=frame(path)
    for c in DATES:
        if c in result:result[c]=pd.to_datetime(result[c],utc=True).dt.tz_convert('Asia/Shanghai')
    return result


def registration():
    reg=load_yaml('configs/optimization_v0_12/experiment.yaml')
    if reg['weight']['half_life_days']!=60. or reg['oof_months']!=list(range(4,12)) or {k:tuple(v) for k,v in reg['candidates'].items()}!=CANDIDATES:
        raise ContractError('registered recency identity changed')
    return reg


def restored(root):
    manifest=read_json(root/'manifest.json');verify_manifest(manifest)
    reg=manifest['registration']
    for k in ('origins','expected_training_rows'):reg[k]={int(m):v for m,v in reg[k].items()}
    return manifest


def access(root,manifest,purpose):
    append_ledger({**manifest['scope'],'authorization':manifest['scope']['authorization']+':'+purpose},root)


@contextmanager
def no_estimators():
    from . import structural,recency_candidate
    count={'attempted_target_fits':0,'attempted_LAD_fits':0}
    def reject(*a,**k):
        count['attempted_LAD_fits']+=1;raise ContractError('inference forbids LAD fitting')
    with forbid_fit(count),patch.object(structural,'lad_coefficient',reject),patch.object(recency_candidate,'lad_coefficient',reject),patch.object(recency_candidate,'fit_coefficients',reject):
        yield count


def freeze(root):
    if (root/'manifest.json').exists():raise ContractError('manifest exists; use a specific saved stage, never restart fits')
    reg=registration();scope=load_yaml('configs/optimization_v0_12/access_scope.yaml');protection=load_yaml(scope['protection_contract'])
    old=verify_v11(Path(reg['source_v11']))
    receipt=read_json(reg['source_v11_receipt'])
    for p,d in receipt['evidence_sha256'].items():
        if file_sha256(p)!=d:raise ContractError('old completed evidence changed')
    if read_json(Path(reg['source_v11'])/'cold_validation.json')['status']!='PASS':raise ContractError('prior repaired G0 required')
    paths=[*Path('src/bf_tap').rglob('*.py'),*Path('configs/optimization_v0_12').glob('*.yaml'),
           Path('docs/optimization_v0_12/PLAN.md'),Path('docs/optimization_v0_12/DATA_PUBLICATION_REVIEW.md'),
           Path('scripts/optimization_v12_cold_check.py'),*Path('tests').glob('test_recency*.py'),Path('uv.lock')]
    evidence=dict(old['evidence'])
    extras=[Path(reg['source_v11_receipt']),Path(reg['source_v11'])/'cold_validation.json',Path(reg['source_v11'])/'cold_repair_manifest.json',
            Path(reg['source_v11'])/'manifest.json',*list((root/'boundary').glob('*'))]
    evidence.update(file_identities({str(p):p for p in extras if p.is_file()}))
    # Inspect only registry/config metadata; no CSV target reads or test distributions.
    pattern=__import__('re').compile(r'recency60|half.?life|age_days|time.?decay',__import__('re').I)
    inspected=[];matches=[]
    for p in Path('local').rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.yaml') and any(t in p.name for t in ('registry','manifest','fit_record','training_record','experiment')) and not p.is_relative_to(root):
            inspected.append(str(p))
            if pattern.search(p.read_text()):matches.append(str(p))
    if matches:raise ContractError('possible prior recency identity requires inspection before fitting: '+str(matches))
    atomic_write_json(root/'registry_audit.json',dict(inspected_files=inspected,matches=matches,same_definition_found=False))
    evidence.update(file_identities({str(root/'registry_audit.json'):root/'registry_audit.json'}))
    manifest=dict(registration=reg,scope=scope,protection=protection,algorithm=old['algorithm'],inventory=old['inventory'],
        inputs=old['inputs'],evidence=evidence,sources=file_identities({str(p):p for p in paths}),
        old_ledger_sha256=file_sha256(scope['old_ledger']),environment=runtime_environment(),
        holdout_consumed=True,official_target_columns_read=False,test_distribution_used=False,publication_hold=True)
    atomic_write_json(root/'manifest.json',manifest);access(root,manifest,'freeze_before_label_access')
    return manifest


def fold_for(manifest,month):
    fold,_=load_fold(manifest,month)
    old=archive(Path(manifest['registration']['source_v8'])/'oof'/f'{month}.csv')
    return fold,old


def matrix(builder,samples,fold,training=False):
    entry=fold[0]['OR'];h=fold[1]['OR'][1]
    if training:
        x=Context.X(builder,samples,pd.Timestamp(entry['cutoff']),entry['component'],'R2',h)
    else:x=builder.X(samples,entry,h)
    original_schema(x,fold[1]['OR'][0].feature_schema_)
    before=x.copy();x=x.copy();x.index=pd.Index(samples.sample_id.astype(str),name='sample_id')
    if not x.reset_index(drop=True).equals(before.reset_index(drop=True)):raise ContractError('old feature values changed')
    return x


def matrix_digest(x):
    return stable_digest(dict(schema=schema(x),row_hashes=pd.util.hash_pandas_object(x,index=False).astype(str).tolist()))


def metadata_all(manifest):
    return pd.concat([archive(Path(manifest['registration']['source_v8'])/'oof'/f'{m}.csv') for m in range(4,12)],ignore_index=True)


def outer_samples(combined,month,n):
    cutoff=stamp(month)
    return combined.loc[(combined.reference_time>=cutoff)&(combined.reference_time<cutoff+pd.DateOffset(months=n)),META]


def diagnostic_rows(errors):
    errors=errors.copy();errors['month']=pd.to_datetime(errors.reference_time,utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m');rows=[]
    for dim in (None,'month','spout_no'):
        grouping=['unit','candidate']+([dim] if dim else [])
        for keys,p in errors.groupby(grouping):
            for target in TARGETS:
                d=float(p[target].sum());e=p['pred_'+target]-p[target];n=float(e.abs().sum())
                rows.append(dict(zip(grouping,keys))|dict(dimension=dim or 'overall',target=target,rows=len(p),absolute_error_sum=n,target_sum=d,
                    wmape=n/d if d>0 else None,signed_bias=float(e.mean()),signed_error_sum=float(e.sum())))
    return rows


def preflight(root,manifest):
    reg=manifest['registration'];dest=root/'p0';dest.mkdir(exist_ok=False)
    builder=builder_for(manifest);checks=[]
    for name,script in (('active_release','scripts/optimization_v8_cold_predict.py'),('fallback_release','scripts/optimization_v4_cold_predict.py')):
        release=load_yaml(reg[name]);prior=Path(release['bundle']).parent
        config=prior/('cold_data_repaired.yaml' if name=='active_release' else 'cold_data.yaml')
        out=dest/f'{name}.csv'
        subprocess.run([sys.executable,script,'--bundle',release['bundle'],'--data-config',str(config),'--output',str(out)],check=True)
        diff=equal(frame(out),frame(prior/'cold_predictions.csv'),PRED)
        checks.append(dict(release=name,rows=len(frame(out)),maximum_difference=diff))
    with no_estimators() as counter:
        for month in reg['oof_months']:
            fold,old=fold_for(manifest,month);h=fold[1]['OR'][1]
            if len(h)!=reg['expected_training_rows'][month]:raise ContractError('old training identity/count mismatch; do not pad/truncate')
            x=matrix(builder,h[META],fold,training=True);w,detail=weights(h,stamp(month));audit=weight_audit(h,w)
            detail.to_csv(dest/f'{month}_weights.csv',index=False)
            audit.update(month=month,old_feature_matrix_sha256=matrix_digest(x),schema_sha256=stable_digest(schema(x)),
                training_ids_sha256=stable_digest(h.sample_id.astype(str).tolist()),weights_sha256=file_sha256(dest/f'{month}_weights.csv'),
                original_columns_values_unchanged=True,no_trajectory_columns=True)
            atomic_write_json(dest/f'{month}_audit.json',audit)
            diff=equal(predict_inputs(old[META],fold,builder,reg),old,PRED+['pred_rate'])
            checks.append(dict(fold=month,rows=len(h),OOF_rows=len(old),maximum_difference=diff))
            builder.cache.clear();print(f'OPT25 fold {month}: original OOF/schema/weights PASS; ESS/n={audit["global_summary"]["ESS_over_n"]:.6f}; zero fits',flush=True)
        combined=metadata_all(manifest)
        for month,n in reg['origins'].items():
            fold,_=fold_for(manifest,month);samples=outer_samples(combined,month,n)
            old=predict_inputs(samples,fold,builder,reg);source=Path(reg['source_v8'])
            equal(old,frame(source/'predictions'/f'{month}_inputs.csv'),PRED+['pred_rate'])
            alpha=read_json(source/'corrections'/f'{month}.json')['alpha']
            diff=equal(apply_correction(old,alpha),frame(source/'predictions'/f'{month}_V1.csv'),PRED)
            checks.append(dict(origin=month,rows=len(samples),maximum_difference=diff));builder.cache.clear()
            print(f'OPT25 origin {month}: original V1/R2 replay PASS',flush=True)
    access(root,manifest,'P0_consumed_V1_background_errors_no_design_changes')
    background=[]
    for unit,*_ in units(reg):
        p=archive(Path(reg['source_v8'])/'units'/unit/'U1_errors.csv');p['unit']=unit;p['candidate']='V1';background.append(p)
    atomic_write_json(dest/'V1_background.json',diagnostic_rows(pd.concat(background,ignore_index=True)))
    verify_manifest(manifest)
    atomic_write_json(dest/'complete.json',dict(status='PASS',checks=checks,**counter,supervised_fits=0,LAD_fits=0,
        audits=file_identities({str(p):p for p in dest.glob('*') if p.is_file()})))


def candidate_parts(builder,samples,fold,models):
    old=predict_inputs(samples,fold,builder,registration())
    x=matrix(builder,samples,fold)
    hr,h=fold[1]['HR'];x04=builder.X(samples,fold[0]['HR'],h);p04=hr.predict_raw(x04).clip(lower=0)
    p04.index=pd.Index(samples.sample_id.astype(str),name='sample_id')
    ids=old.sample_id;parts=old.copy()
    for target,short in zip(TARGETS,('iron','time')):
        if models[target].target!=target:raise ContractError('direct target mismatch')
        direct=pd.Series(models[target].predict(x),index=x.index)
        parts['direct_'+short]=direct.loc[ids].to_numpy()
        parts['base_'+short]=.8*parts['direct_'+short]+(1.-.8)*p04.loc[ids,'pred_'+target].to_numpy()
    direction,_=isolated_directions(old,parts[['sample_id','base_iron','base_time']])
    return parts.merge(direction,on='sample_id',validate='one_to_one')[PARTS]


def fit_counts(root):
    attempted=len(list((root/'models').glob('*/*/fit_intent.json')))
    complete=len(list((root/'models').glob('*/*/bundle/bundle.json')))
    lad_attempted=2*len(list((root/'corrections').glob('*_intent.json')))
    lad_complete=2*len([p for p in (root/'corrections').glob('*.json') if not p.name.endswith('_intent.json')])
    if attempted>16 or lad_attempted>12:raise ContractError('development fit budget exceeded')
    return dict(development_CatBoost_attempted=attempted,development_CatBoost_completed=complete,
        development_LAD_attempted=lad_attempted,development_LAD_completed=lad_complete,
        new_E04_rate_q_old_V1_R2_fits=0,final_CatBoost=0,final_LAD=0,challengers=0)


def trained_models(root,month):
    return {t:RecencyModel.load(root/'models'/str(month)/t/'bundle') for t in TARGETS}


def selected_oof(combined,history,cutoff,minimum):
    labeled=combined.merge(history[['sample_id',*TARGETS,'available_at']],on='sample_id',validate='one_to_one')
    if not (labeled.label_available_at==labeled.available_at).all():raise ContractError('OOF label availability differs')
    return select_oof(labeled,cutoff,minimum)


def train(root,manifest):
    reg=manifest['registration'];p0=read_json(root/'p0/complete.json');verify_file_identities(p0['audits'])
    if p0['status']!='PASS':raise ContractError('P0 must pass')
    builder=builder_for(manifest);records=[];new_oof=[]
    existing=fit_counts(root)['development_CatBoost_attempted']
    budget=ResidualFitBudget(16-existing)
    with budget:
        for month in reg['oof_months']:
            fold,old=fold_for(manifest,month);h=fold[1]['OR'][1];x=matrix(builder,h[META],fold,training=True);w,detail=weights(h,stamp(month))
            audit=read_json(root/'p0'/f'{month}_audit.json')
            if matrix_digest(x)!=audit['old_feature_matrix_sha256']:raise ContractError('P0 training feature matrix changed')
            pd.testing.assert_frame_equal(detail,frame(root/'p0'/f'{month}_weights.csv'),check_exact=True)
            training=dict(cutoff=str(stamp(month)),rows=len(h),training_ids_sha256=stable_digest(h.sample_id.astype(str).tolist()),
                reference_max=str(h.reference_time.max()),available_max=str(h.available_at.max()),
                original_OR=fold[0]['OR'],schema_sha256=stable_digest(schema(x)),matrix_sha256=matrix_digest(x),
                weights_sha256=audit['weights_sha256'],manifest_sha256=file_sha256(root/'manifest.json'))
            for target in TARGETS:
                destination=root/'models'/str(month)/target;bundle=destination/'bundle'
                identity=dict(**training,target=target,label_sha256=stable_digest(h[target].tolist()))
                if (bundle/'bundle.json').exists():
                    model=RecencyModel.load(bundle)
                    if model.metadata['training']!=identity:raise ContractError('saved successful fit identity differs')
                else:
                    if (destination/'fit_intent.json').exists():raise ContractError('unfinished fit attempt cannot be retrained as recovery')
                    atomic_write_json(destination/'fit_intent.json',identity)
                    y=pd.Series(h[target].to_numpy(),index=x.index,name=target)
                    model=RecencyModel().fit(x,y,w,h,stamp(month),target,fold[1]['OR'][0].feature_schema_,budget)
                    model.save(bundle,identity)
                restored_model=RecencyModel.load(bundle)
                if not np.array_equal(model.predict(x),restored_model.predict(x)):raise ContractError('direct model persistence differs')
                record=dict(**identity,bundle_sha256=file_sha256(bundle/'bundle.json'))
                if not (destination/'fit_record.json').exists():atomic_write_json(destination/'fit_record.json',record)
                records.append(record)
                print(f'OPT26 fold {month} {target}: saved direct fit; completed={fit_counts(root)["development_CatBoost_completed"]}/16',flush=True)
            models=trained_models(root,month)
            with no_estimators():parts=candidate_parts(builder,old[META],fold,models)
            oof=old[[*META,*[c for c in DATES if c in old and c not in META]]].merge(parts,on='sample_id',validate='one_to_one')
            path=root/'oof'/f'{month}.csv';path.parent.mkdir(exist_ok=True)
            if path.exists():pd.testing.assert_frame_equal(oof,archive(path),check_exact=True)
            else:oof.to_csv(path,index=False)
            new_oof.append(oof);builder.cache.clear()
    if not (root/'training_records.json').exists():atomic_write_json(root/'training_records.json',records)
    combined=pd.concat(new_oof,ignore_index=True);inference={'attempted_target_fits':0,'attempted_LAD_fits':0}
    for month,n in reg['origins'].items():
        fold,_=fold_for(manifest,month);cutoff=stamp(month);selected=selected_oof(combined,fold[1]['OR'][1],cutoff,reg['minimum_OOF_rows'])
        dest=root/'corrections';dest.mkdir(exist_ok=True);coeff_path=dest/f'{month}.json'
        if coeff_path.exists():
            coeff=read_json(coeff_path);beta=coeff['beta']
            if coeff['OOF_ids_sha256']!=stable_digest(selected.sample_id.tolist()):raise ContractError('saved coefficient identities differ')
        else:
            if (dest/f'{month}_intent.json').exists():raise ContractError('unfinished LAD attempt cannot be refitted as recovery')
            selected.to_csv(dest/f'{month}_OOF.csv',index=False)
            atomic_write_json(dest/f'{month}_intent.json',dict(cutoff=str(cutoff),targets=list(TARGETS),OOF_sha256=file_sha256(dest/f'{month}_OOF.csv')))
            beta,used=fit_coefficients(selected,cutoff,reg['minimum_OOF_rows'])
            coeff=dict(beta=beta,rows=len(used),cutoff=str(cutoff),OOF_ids_sha256=stable_digest(used.sample_id.tolist()),
                OOF_sha256=file_sha256(dest/f'{month}_OOF.csv'),reference_max=str(used.reference_time.max()),available_max=str(used.label_available_at.max()),
                source_history_sha256=fold[0]['OR']['history_snapshot_sha256'])
            atomic_write_json(coeff_path,coeff)
        with no_estimators() as counter:
            samples=outer_samples(combined,month,n);models=trained_models(root,month)
            parts=candidate_parts(builder,samples,fold,models)
            original=frame(Path(reg['source_v8'])/'predictions'/f'{month}_V1.csv')
            result=outputs(parts,original,beta)
            reverse=outputs(candidate_parts(builder,samples.iloc[::-1],fold,models),original,beta)
            subset=samples.iloc[::max(1,len(samples)//7)]
            sub=outputs(candidate_parts(builder,subset,fold,models),original.loc[original.sample_id.isin(subset.sample_id)],beta)
            d=root/'predictions';d.mkdir(exist_ok=True)
            for candidate,pred in result.items():
                equal(pred,reverse[candidate],PRED);equal(pred.loc[pred.sample_id.isin(subset.sample_id)],sub[candidate],PRED)
                path=d/f'{month}_{candidate}.csv'
                if path.exists():equal(pred,frame(path),PRED,tolerance=0.)
                else:pred.to_csv(path,index=False)
            path=d/f'{month}_parts.csv'
            if path.exists():equal(parts,frame(path),PARTS[1:],tolerance=0.)
            else:parts.to_csv(path,index=False)
            for k,v in counter.items():inference[k]+=v
        builder.cache.clear();print(f'OPT26 outer {month}: LAD rows={coeff["rows"]}, beta={beta}; isolated/reverse/subset PASS',flush=True)
    counts=fit_counts(root)
    if counts['development_CatBoost_completed']!=16 or counts['development_LAD_completed']!=12 or any(inference.values()):raise ContractError('completed fit budget/inference count differs')
    verify_manifest(manifest)
    atomic_write_json(root/'predictions_complete.json',dict(status='PASS',fit_counts=counts,inference=inference,
        files=file_identities({str(p):p for p in (root/'predictions').glob('*.csv')}),
        training_evidence=file_identities({str(p):p for folder in ('models','oof','corrections') for p in (root/folder).rglob('*') if p.is_file()}),
        holdout_consumed=True,candidate_scoring_labels_read=False))


def coefficient_certificate(selected,beta):
    for target,short in zip(TARGETS,('iron','time')):
        b=beta[target];d=selected['direction_'+short].to_numpy();res=selected['base_'+short].to_numpy()+b*d-selected[target].to_numpy()
        zero=np.isclose(res,0.,atol=1e-10,rtol=0.)
        fixed=float(np.sum(np.sign(res[~zero])*d[~zero]));kink=float(np.abs(d[zero]).sum())
        if not 0<=b<=1 or (b>0 and fixed-kink>1e-8) or (b<1 and fixed+kink< -1e-8):raise ContractError('LAD optimality certificate failed')


def cold(root,manifest):
    reg=manifest['registration'];complete=read_json(root/'predictions_complete.json')
    verify_file_identities(complete['files']);verify_file_identities(complete['training_evidence'])
    if set(manifest['inputs'])!={'operation_hourly','burden_change','data_dictionary'}:raise ContractError('cold input config contains official training label paths')
    builder=builder_for(manifest);checks=[];combined=[]
    with no_estimators() as counter:
        for month in reg['oof_months']:
            fold,old=fold_for(manifest,month);h=fold[1]['OR'][1];x=matrix(builder,h[META],fold,training=True);w,detail=weights(h,stamp(month))
            audit=read_json(root/'p0'/f'{month}_audit.json')
            if matrix_digest(x)!=audit['old_feature_matrix_sha256']:raise ContractError('cold original training matrix differs')
            pd.testing.assert_frame_equal(detail,frame(root/'p0'/f'{month}_weights.csv'),check_exact=True)
            models=trained_models(root,month)
            for target,model in models.items():
                tr=model.metadata['training']
                if tr['training_ids_sha256']!=stable_digest(h.sample_id.astype(str).tolist()) or tr['weights_sha256']!=audit['weights_sha256'] or tr['label_sha256']!=stable_digest(h[target].tolist()) or tr['manifest_sha256']!=file_sha256(root/'manifest.json') or tr['cutoff']!=str(stamp(month)):
                    raise ContractError('cold direct training provenance differs')
                original_schema(x,model.schema)
            saved=archive(root/'oof'/f'{month}.csv')
            certificate=[*META,*[c for c in DATES if c in old and c not in META]]
            pd.testing.assert_frame_equal(saved[certificate],old[certificate],check_exact=True)
            fresh=candidate_parts(builder,saved[META],fold,models)
            delta=equal(fresh,saved,PARTS[1:]);combined.append(saved)
            checks.append(dict(fold=month,rows=len(saved),maximum_difference=delta));builder.cache.clear()
            print(f'OPT26 cold fold {month}: old schema/weights/training/OOF PASS',flush=True)
        combined=pd.concat(combined,ignore_index=True)
        for month,n in reg['origins'].items():
            fold,_=fold_for(manifest,month);selected=selected_oof(combined,fold[1]['OR'][1],stamp(month),reg['minimum_OOF_rows'])
            saved=archive(root/'corrections'/f'{month}_OOF.csv')
            pd.testing.assert_frame_equal(selected.reset_index(drop=True),saved.reset_index(drop=True),check_exact=True)
            coeff=read_json(root/'corrections'/f'{month}.json');coefficient_certificate(selected,coeff['beta'])
            if coeff['OOF_sha256']!=file_sha256(root/'corrections'/f'{month}_OOF.csv') or coeff['OOF_ids_sha256']!=stable_digest(selected.sample_id.tolist()):raise ContractError('cold beta provenance mismatch')
            samples=outer_samples(combined,month,n);models=trained_models(root,month);parts=candidate_parts(builder,samples,fold,models)
            original=frame(Path(reg['source_v8'])/'predictions'/f'{month}_V1.csv')
            result=outputs(parts,original,coeff['beta']);reverse=outputs(candidate_parts(builder,samples.iloc[::-1],fold,models),original,coeff['beta'])
            subset=samples.iloc[::max(1,len(samples)//7)];sub=outputs(candidate_parts(builder,subset,fold,models),original.loc[original.sample_id.isin(subset.sample_id)],coeff['beta'])
            for c,p in result.items():
                delta=equal(p,frame(root/'predictions'/f'{month}_{c}.csv'),PRED)
                equal(p,reverse[c],PRED);equal(p.loc[p.sample_id.isin(subset.sample_id)],sub[c],PRED)
                checks.append(dict(origin=month,candidate=c,maximum_difference=delta,unchanged_targets_exact=True,reverse=True,subset=True))
            builder.cache.clear();print(f'OPT26 cold origin {month}: beta/three candidates/reverse/subset PASS',flush=True)
    verify_manifest(manifest)
    atomic_write_json(root/'cold_validation.json',dict(status='PASS',checks=checks,**counter,CatBoost_fits=0,LAD_fits=0,
        manifest_sha256=file_sha256(root/'manifest.json'),prediction_receipt_sha256=file_sha256(root/'predictions_complete.json')))


def evaluate(root,manifest):
    reg=manifest['registration'];cold_result=read_json(root/'cold_validation.json')
    if cold_result['status']!='PASS' or cold_result['CatBoost_fits'] or cold_result['LAD_fits']:raise ContractError('zero-fit cold PASS required before candidate scoring')
    receipt=read_json(root/'predictions_complete.json');verify_file_identities(receipt['files']);verify_file_identities(receipt['training_evidence'])
    access(root,manifest,'candidate_scoring_after_all_frozen_predictions_and_cold_PASS')
    with no_estimators() as counter:
        def provider(unit,cutoff,metadata,baselines):
            result={c:frame(root/'predictions'/f'{cutoff.month}_{c}.csv') for c in CANDIDATES}
            result={c:p.loc[p.sample_id.isin(metadata.sample_id)] for c,p in result.items()};validate_outputs(result,baselines['V1'])
            return result
        metrics,summary,errors,_=score(root,reg,provider)
        gate=acceptance(metrics,summary,reg,True,True);atomic_write_json(root/'acceptance.json',gate)
        atomic_write_json(root/'bootstrap.json',{c:week_intervals(errors,c,'V1',**reg['bootstrap']) for c in CANDIDATES})
        atomic_write_json(root/'diagnostics.json',diagnostic_rows(errors))
        changes={}
        for month in reg['origins']:
            parts=frame(root/'predictions'/f'{month}_parts.csv');original=frame(Path(reg['source_v8'])/'predictions'/f'{month}_V1.csv').set_index('sample_id')
            changes[str(month)]={'rate_fallback_rows':int((parts.pred_rate<=reg['rate_floor']).sum()),'candidates':{}}
            for c in CANDIDATES:
                pred=frame(root/'predictions'/f'{month}_{c}.csv').set_index('sample_id').sort_index();old=original.loc[pred.index]
                changes[str(month)]['candidates'][c]={t:{str(q):float(np.quantile(pred['pred_'+t]-old['pred_'+t],q)) for q in (.05,.5,.95)} for t in TARGETS}
        atomic_write_json(root/'prediction_changes.json',changes)
    verify_manifest(manifest)
    atomic_write_json(root/'fit_counts.json',fit_counts(root))
    atomic_write_json(root/'OOF_provenance.json',dict(training=read_json(root/'training_records.json'),
        coefficients={str(m):read_json(root/'corrections'/f'{m}.json') for m in reg['origins']},
        manifest_sha256=file_sha256(root/'manifest.json'),cold_sha256=file_sha256(root/'cold_validation.json'),holdout_consumed=True))
    atomic_write_json(root/'final_status.json',dict(G0='PASS_OPT25_OPT26_AND_COLD',G1=gate['status'],selected=gate['selected'],
        publication='BLOCKED_PENDING_SEPARATE_DATA_HISTORY_REMEDIATION',holdout_consumed=True,**counter,
        final_stage='CONDITIONAL_SELECTED_TARGETS_ONLY' if gate['selected'] else 'NOT_AUTHORIZED_NO_WINNER'))
    print(__import__('json').dumps(gate,indent=2),flush=True)


def main(root,phase):
    try:
        if phase=='run':
            manifest=freeze(root);preflight(root,manifest);train(root,manifest)
            subprocess.run([sys.executable,'scripts/optimization_v12_cold_check.py','--source',str(root)],check=True)
            evaluate(root,manifest)
        else:
            manifest=restored(root)
            {'p0':preflight,'train':train,'cold':cold,'score':evaluate}[phase](root,manifest)
    except Exception as exc:
        d=root/'failures';d.mkdir(exist_ok=True)
        atomic_write_json(d/f'{len(list(d.glob("*.json"))):03d}_{phase}.json',dict(error=str(exc),phase=phase,counts=fit_counts(root)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--phase',choices=('run','p0','train','cold','score'),default='run')
    args=parser.parse_args();main(args.output,args.phase)
