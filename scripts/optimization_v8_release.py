"""One registered final rate fit and one V1 challenger after strict development PASS."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import yaml
from bf_tap.artifacts import atomic_write_json,file_identities,file_sha256,stable_digest,verify_file_identities,build_inference_source_contract
from bf_tap.config import load_yaml
from bf_tap.io import parse_local_time
from bf_tap.offline import _load_process_sources
from bf_tap.protection import load_protection_policy
from bf_tap.schema import validate_samples,validate_history,validate_cross_table_consistency
from bf_tap.optimization.component_export import META,PRED,ComponentFeatures,read_json,forbid_fit
from bf_tap.optimization.final_lifecycle import algorithm,label_metadata,eligible,_read_eligible_rows,sample_metadata
from bf_tap.optimization.refresh_factorial import Context
from bf_tap.optimization.structural import fit_correction
from bf_tap.optimization.rate_model import RateModel
from bf_tap.optimization.structural_predict import StructuralPredictor
from bf_tap.submission import write_submission,pack_submission


def run(development,cold_check,output):
    if not read_json(development/'acceptance.json')['passed'] or read_json(development/'final_status.json')['G0']!='PASS_OPT20':raise ValueError('complete strict development PASS required')
    checked=read_json(cold_check/'validation.json')
    if checked['status']!='PASS' or len(checked['checks'])!=6 or checked['attempted_target_fits']!=0:raise ValueError('six-origin cold OOF audit required')
    if checked['source_manifest_sha256']!=file_sha256(development/'manifest.json'):raise ValueError('cold audit does not bind development')
    output.mkdir(parents=True,exist_ok=False);fits=0
    try:
        prior=read_json(development/'manifest.json')
        for key in ('inputs','sources','evidence'):verify_file_identities(prior[key])
        reg=prior['registration'];release=load_yaml('configs/optimization_v0_8/final_release.yaml')
        policy=load_protection_policy('configs/protection.yaml')
        if release['protected_lifecycle'] not in policy.allowed_protected_lifecycles:raise ValueError('unregistered protected lifecycle')
        active=load_yaml(release['base_release']);base=Path(active['bundle']);cutoff=pd.Timestamp(active['fit_cutoff'])
        if file_sha256(base/'composite.json')!=active['composite_sha256']:raise ValueError('active R2 composite changed')
        if file_sha256(active['test_a_zip'])!='e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf':raise ValueError('R2 archive changed')
        paths=load_yaml('configs/data.local.yaml')['paths'];paths={k:paths[k] for k in ('train_samples','tap_history_train','operation_hourly','burden_change','data_dictionary','test_a_samples')}
        a=algorithm();metadata=label_metadata(paths,a['semantic']);selected=eligible(metadata,cutoff)
        original=read_json(base/'composite.json');anchor=original['anchors'][0]
        for entry in anchor['components'].values():
            md=read_json(base/entry['path']/'bundle.json')
            if md['training']['sample_ids_sha256']!=stable_digest(selected.sample_id.astype(str).tolist()):raise ValueError('final R2/selected training IDs differ')
        ev=[development/'manifest.json',development/'acceptance.json',cold_check/'validation.json',Path(release['base_release']),Path(active['test_a_zip']),base/'composite.json',*sorted((development/'oof').glob('[0-9]*.csv'))]
        sources=[*Path('src/bf_tap').rglob('*.py'),*Path('scripts').glob('optimization_v8_*.py'),*Path('configs/optimization_v0_8').glob('*.yaml'),Path('configs/protection.yaml')]
        manifest=dict(release=release,registration=reg,policy_digest=policy.digest,lifecycle='final_training',
            cutoff=str(cutoff),inputs=file_identities(paths),evidence=file_identities({str(p):p for p in ev}),
            sources=file_identities({str(p):p for p in sources}),old_ledger_sha256=file_sha256(prior['scope']['old_ledger']),
            desktop_sha256=file_sha256('/mnt/c/Users/lqh22/Desktop/Luqhhh_bf_tap_predict_prelim.zip'))
        atomic_write_json(output/'manifest.json',manifest)
        import fcntl
        ledger=Path(release['ledger']);ledger.parent.mkdir(parents=True,exist_ok=True)
        with ledger.open('a+') as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX);f.seek(0);old=f.read().splitlines()
            f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),lifecycle='final_training',policy_digest=policy.digest,
                authorization=release['authorization'],frozen_manifest_sha256=file_sha256(output/'manifest.json'),
                previous_sha256=stable_digest(old[-1]) if old else None))+'\n')
        # Protected November target reads occur only after the final manifest and ledger.
        labels=_read_eligible_rows(paths['train_samples'],selected.sample_id).merge(selected[['sample_id','label_available_at']],on='sample_id',validate='one_to_one')
        labels=eligible(labels,cutoff)
        history=_read_eligible_rows(paths['tap_history_train'],selected.sample_id)
        for c in ('reference_time','tap_end_time'):history[c]=parse_local_time(history[c],c)
        history['available_at']=history.tap_end_time
        validate_samples(labels,labeled=True);validate_history(history);validate_cross_table_consistency(labels,history)
        if (history.reference_time>=cutoff).any() or (history.available_at>cutoff).any():raise ValueError('future final history')
        training=dict(fit_cutoff=str(cutoff),history_cutoff=str(cutoff),label_available_cutoff=str(cutoff),
            variant='R2',rows=len(labels),sample_ids_sha256=stable_digest(labels.sample_id.astype(str).tolist()),
            reference_max=str(labels.reference_time.max()),label_available_max=str(labels.label_available_at.max()),
            history_available_max=str(history.available_at.max()),training_history_policy='original_per_sample_asof')
        oof=pd.concat([pd.read_csv(development/'oof'/f'{m}.csv,float_precision='round_trip',dtype={'sample_id':'string'}) for m in release['OOF_months']],ignore_index=True)
        for c in ('reference_time','label_available_at','fold_cutoff','train_reference_max','train_available_max','history_available_max'):oof[c]=pd.to_datetime(oof[c])
        oof=oof.merge(labels[['sample_id','tap_iron','tap_time_len']],on='sample_id',validate='one_to_one')
        alpha,used=fit_correction(oof,cutoff,reg['correction']['minimum_OOF_rows'],reg['rate']['unusable_predicted_rate_max'])
        used.to_csv(output/'final_correction_OOF.csv',index=False)
        op,burden,_=_load_process_sources(paths,a['semantic'],a['features']);builder=ComponentFeatures(a,op,burden)
        X=Context.X(builder,labels[META],cutoff,reg['rate']['component'],'R2',history)
        model=RateModel().fit(X,labels.tap_iron,labels.tap_time_len,a['baseline']['parameters']);fits+=1
        contract=build_inference_source_contract(manifest['inputs'],semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
        bundle=output/'bundle';bundle.mkdir();shutil.copytree(base,bundle/'base_R2')
        model.save(bundle/'rate',dict(parameters=a['baseline']['parameters'],training=training,
            inference_source_contract=contract,manifest_sha256=file_sha256(output/'manifest.json'),rate_spec=reg['rate'],algorithm_contract=a['contract_digests']),history)
        if not np.array_equal(model.predict(X),RateModel.load(bundle/'rate').predict(X)):raise ValueError('final rate roundtrip differs')
        composite=dict(candidate='V1_RATE_STRUCTURAL',algorithm=a,registration=reg,training=training,alpha=alpha,
            base_R2_composite_sha256=active['composite_sha256'],rate_bundle_sha256=file_sha256(bundle/'rate'/'bundle.json'),
            correction_OOF_rows=len(used),correction_OOF_ids_sha256=stable_digest(used.sample_id.astype(str).tolist()),
            correction_OOF_sha256=file_sha256(output/'final_correction_OOF.csv'),development_acceptance_sha256=file_sha256(development/'acceptance.json'))
        atomic_write_json(bundle/'structural.json',composite);atomic_write_json(bundle/'identity.json',dict(sha256=file_sha256(bundle/'structural.json')))
        samples=sample_metadata(paths['test_a_samples'])[META];counter={'attempted_target_fits':0}
        with forbid_fit(counter):expected,parts=StructuralPredictor(bundle).predict(samples,op,burden,contract)
        cold_config=output/'cold_data.yaml';cold_config.write_text(yaml.safe_dump(dict(paths={k:v for k,v in paths.items() if k not in ('train_samples','tap_history_train')})))
        cold_path=output/'cold_predictions.csv'
        subprocess.run([sys.executable,'scripts/optimization_v8_cold_predict.py','--bundle',str(bundle),'--data-config',str(cold_config),'--output',str(cold_path)],check=True)
        cold=pd.read_csv(cold_path,float_precision='round_trip',dtype={'sample_id':str}).set_index('sample_id').sort_index()
        compare=expected.set_index('sample_id').sort_index();delta=float(np.abs(cold[PRED]-compare[PRED]).to_numpy().max())
        if not cold.index.equals(compare.index) or delta>1e-10:raise ValueError('final cold process mismatch')
        ordered=samples[['sample_id']].merge(expected,on='sample_id',validate='one_to_one',sort=False)
        write_submission(ordered,samples.sample_id,output/'result.csv')
        archive=pack_submission(output/'result.csv',stage='test_a',team_name='Luqhhh',output_dir=output,expected_ids=samples.sample_id)
        for key in ('inputs','sources','evidence'):verify_file_identities(manifest[key])
        if file_sha256(prior['scope']['old_ledger'])!=manifest['old_ledger_sha256']:raise ValueError('old ledger changed')
        if file_sha256('/mnt/c/Users/lqh22/Desktop/Luqhhh_bf_tap_predict_prelim.zip')!=manifest['desktop_sha256']:raise ValueError('desktop incumbent changed')
        atomic_write_json(output/'release_validation.json',dict(status='READY_CHALLENGER',candidate='V1_RATE_STRUCTURAL',rows=len(samples),
            new_rate_fits=fits,new_R2_fits=0,structural_LAD_target_fits=2,inference_fit_attempts=counter['attempted_target_fits'],
            training=training,alpha=alpha,correction_OOF_rows=len(used),cold_max_difference=delta,
            original_R2_loader_max_difference=read_json(str(cold_path)+'.json')['R2_original_loader_max_difference'],
            archive_sha256=file_sha256(archive),active_R2_unchanged=True,desktop_unchanged=True,uploaded=False,
            test_labels_read=False,rate_zero_duration_excluded=model.excluded_zero_duration))
        print(json.dumps(read_json(output/'release_validation.json'),indent=2),flush=True)
    except Exception as exc:
        atomic_write_json(output/'failure.json',dict(error=str(exc),completed_final_rate_fits=fits));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--development',required=True,type=Path);p.add_argument('--cold-check',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.development,a.cold_check,a.output)
