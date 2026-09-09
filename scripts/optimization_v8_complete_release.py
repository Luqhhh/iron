"""Resume packaging an already trained V1 after the documented cold YAML repair."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import yaml
from bf_tap.artifacts import atomic_write_json,file_sha256,verify_file_identities,file_identities,build_inference_source_contract
from bf_tap.offline import _load_process_sources
from bf_tap.optimization.component_export import META,PRED,read_json,forbid_fit
from bf_tap.optimization.final_lifecycle import sample_metadata
from bf_tap.optimization.structural_predict import StructuralPredictor
from bf_tap.submission import write_submission,pack_submission


def complete(root):
    if (root/'release_validation.json').exists() or list(root.glob('*.zip')):raise FileExistsError('release already exists')
    old=read_json(root/'manifest.json');failure=read_json(root/'failure.json')
    assert failure['completed_final_rate_fits']==1
    changed='scripts/optimization_v8_release.py'
    old_source=subprocess.check_output(['git','show',f'451c042:{changed}'])
    assert hashlib.sha256(old_source).hexdigest()==old['sources'][changed]['sha256']
    # Only add the missing schema version to the archived launcher. No training or
    # model algorithm change is permitted by this recovery path.
    expected=old_source.replace(b'yaml.safe_dump(dict(paths={k:v for k,v in paths.items()',b'yaml.safe_dump(dict(schema_version=1,paths={k:v for k,v in paths.items()')
    assert Path(changed).read_bytes()==expected
    for key in ('inputs','evidence'):verify_file_identities(old[key])
    verify_file_identities({k:v for k,v in old['sources'].items() if k!=changed})
    counter={'attempted_target_fits':0}
    with forbid_fit(counter):
        predictor=StructuralPredictor(root/'bundle');m=predictor.m;a=predictor.a
        assert m['rate_bundle_sha256']==file_sha256(root/'bundle'/'rate'/'bundle.json')
        archived=root/'release_launcher_at_training.py';archived.write_bytes(old_source)
        raw=yaml.safe_load((root/'cold_data.yaml').read_text())
        repaired=root/'cold_data_repaired.yaml';repaired.write_text(yaml.safe_dump(dict(schema_version=1,paths=raw['paths'])))
        paths=raw['paths']
        assert set(paths)<={'test_a_samples','operation_hourly','burden_change','data_dictionary'}
        receipt=dict(repair='add_missing_cold_config_schema_version_only',additional_model_fits=0,
            training_manifest_sha256=file_sha256(root/'manifest.json'),original_failure_sha256=file_sha256(root/'failure.json'),
            original_launcher_sha256=hashlib.sha256(old_source).hexdigest(),fixed_launcher_sha256=file_sha256(changed),
            repaired_config_sha256=file_sha256(repaired),bundle_sha256=file_sha256(root/'bundle'/'structural.json'),
            sources=file_identities({str(p):p for p in [Path(__file__),Path(changed),Path('scripts/optimization_v8_cold_predict.py')]}))
        atomic_write_json(root/'packaging_repair.json',receipt)
        cold_path=root/'cold_predictions.csv'
        subprocess.run([sys.executable,'scripts/optimization_v8_cold_predict.py','--bundle',str(root/'bundle'),'--data-config',str(repaired),'--output',str(cold_path)],check=True)
        samples=sample_metadata(paths['test_a_samples'])[META]
        op,burden,_=_load_process_sources(paths,a['semantic'],a['features'])
        inputs=file_identities({k:v for k,v in paths.items() if k!='test_a_samples'})
        contract=build_inference_source_contract(inputs,semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
        expected,parts=predictor.predict(samples,op,burden,contract)
        cold=pd.read_csv(cold_path,float_precision='round_trip',dtype={'sample_id':str}).set_index('sample_id').sort_index()
        aligned=expected.set_index('sample_id').sort_index();assert cold.index.equals(aligned.index)
        delta=float(np.abs(cold[PRED]-aligned[PRED]).to_numpy().max());assert delta<=1e-10
        ordered=samples[['sample_id']].merge(expected,on='sample_id',sort=False,validate='one_to_one')
        write_submission(ordered,samples.sample_id,root/'result.csv')
        archive=pack_submission(root/'result.csv',stage='test_a',team_name='Luqhhh',output_dir=root,expected_ids=samples.sample_id)
    verify_file_identities(receipt['sources'])
    for key in ('inputs','evidence'):verify_file_identities(old[key])
    assert file_sha256('/mnt/c/Users/lqh22/Desktop/Luqhhh_bf_tap_predict_prelim.zip')==old['desktop_sha256']
    assert file_sha256('local/ledgers/optimization-v0.3-r2-protected.jsonl')==old['old_ledger_sha256']
    atomic_write_json(root/'release_validation.json',dict(status='READY_CHALLENGER',candidate='V1_RATE_STRUCTURAL',rows=len(samples),
        new_rate_fits=1,new_R2_fits=0,structural_LAD_target_fits=2,inference_fit_attempts=counter['attempted_target_fits'],
        training=m['training'],alpha=m['alpha'],correction_OOF_rows=m['correction_OOF_rows'],cold_max_difference=delta,
        original_R2_loader_max_difference=read_json(str(cold_path)+'.json')['R2_original_loader_max_difference'],
        archive_sha256=file_sha256(archive),active_R2_unchanged=True,desktop_unchanged=True,uploaded=False,test_labels_read=False,
        rate_zero_duration_excluded=predictor.rate.metadata_['excluded_zero_duration'],
        packaging_recovered=True,repair_receipt_sha256=file_sha256(root/'packaging_repair.json'),additional_recovery_fits=0))
    print(read_json(root/'release_validation.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);complete(p.parse_args().run)
