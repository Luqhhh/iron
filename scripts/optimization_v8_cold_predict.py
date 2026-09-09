"""V1 final inference in a fresh process; no label or official history paths."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
from bf_tap.artifacts import atomic_write_json,file_identities,build_inference_source_contract
from bf_tap.config import load_yaml
from bf_tap.offline import _load_process_sources
from bf_tap.optimization.component_export import META,PRED,forbid_fit
from bf_tap.optimization.final_lifecycle import sample_metadata
from bf_tap.optimization.structural_predict import StructuralPredictor


def main(bundle,config,output):
    if output.exists():raise FileExistsError(output)
    paths=load_yaml(config)['paths']
    if set(paths)-{'test_a_samples','operation_hourly','burden_change','data_dictionary'}:raise ValueError('label-free inference config required')
    samples=sample_metadata(paths['test_a_samples'])[META];counter={'attempted_target_fits':0}
    with forbid_fit(counter):
        predictor=StructuralPredictor(bundle);a=predictor.a
        inputs=file_identities({k:v for k,v in paths.items() if k!='test_a_samples'})
        contract=build_inference_source_contract(inputs,semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
        op,burden,_=_load_process_sources(paths,a['semantic'],a['features'])
        predicted,parts=predictor.predict(samples,op,burden,contract)
        reverse,_=predictor.predict(samples.iloc[::-1],op,burden,contract)
        assert np.array_equal(predicted.set_index('sample_id').sort_index()[PRED].to_numpy(),reverse.set_index('sample_id').sort_index()[PRED].to_numpy())
        spec=importlib.util.spec_from_file_location('original_r2','scripts/optimization_v4_cold_predict.py')
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        baseline=original.predict(Path(bundle)/'base_R2',config,'test_a').set_index('sample_id').sort_index()
        recovered=parts.set_index('sample_id').sort_index()
        delta=float(np.abs(baseline[PRED]-recovered[PRED]).to_numpy().max())
        assert delta<1e-8
    predicted.to_csv(output,index=False)
    atomic_write_json(str(output)+'.json',dict(status='PASS',rows=len(samples),reversal_exact=True,**counter,
        R2_original_loader_max_difference=delta,label_paths_present=False))
    print(f'PASS V1 cold inference: {len(samples)} rows; original R2 delta={delta}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True,type=Path);p.add_argument('--data-config',required=True);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();main(a.bundle,a.data_config,a.output)
