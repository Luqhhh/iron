import json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from bf_tap.artifacts import atomic_write_json,file_sha256
from bf_tap.optimization.final_lifecycle import Predictor,sample_metadata
from bf_tap.optimization.snapshot_ensemble import blend
from bf_tap.offline import _load_process_sources

out=Path('local/runs/optimization-v0.4-composite-check-r1');out.mkdir(parents=True,exist_ok=False)
roots=[Path('local/runs/optimization-v0.3-r2-opt10-final-r1/bundle'),Path('local/runs/optimization-v0.3-r2-opt10-prepared-r1/HOLDOUT_H1/bundle')]
old=[Predictor(r) for r in roots]
a=old[0].m['algorithm']
m=dict(schema_version=1,candidate='R3',algorithm=a,calibration='none',source_contract=old[0].m['inference_source_contract'],anchors=[])
bundle=out/'bundle';bundle.mkdir()
for i,root in enumerate(roots):
 anchor=dict(cutoff=old[i].m['training']['fit_cutoff'],variant='raw',components={})
 for c in ('E09_PROCESS_CHANGE_E02','E04'):
  dest=bundle/f'anchor_{i}'/c;shutil.copytree(root/c,dest)
  anchor['components'][c]=dict(path=str(dest.relative_to(bundle)),sha256=file_sha256(dest/'bundle.json'))
 m['anchors'].append(anchor)
atomic_write_json(bundle/'composite.json',m)
atomic_write_json(bundle/'identity.json',dict(sha256=file_sha256(bundle/'composite.json')))
paths=yaml.safe_load(Path('configs/data.local.yaml').read_text())['paths']
data=dict(schema_version=1,paths={k:paths[k] for k in ('test_a_samples','operation_hourly','burden_change','data_dictionary')})
config=out/'no_official_labels.yaml';config.write_text(yaml.safe_dump(data))
subprocess.run([sys.executable,'scripts/optimization_v4_cold_predict.py','--bundle',str(bundle),'--data-config',str(config),'--output',str(out/'cold.csv')],check=True)
samples=sample_metadata(paths['test_a_samples'])
op,burden,_=_load_process_sources(paths,a['semantic'],a['features'])
expected=blend(*[p.predict(samples,op,burden) for p in old]).set_index('sample_id').sort_index()
cold=pd.read_csv(out/'cold.csv',dtype={'sample_id':str}).set_index('sample_id').sort_index()
assert cold.index.equals(expected.index)
delta=float(np.abs(cold.to_numpy()-expected.to_numpy()).max());assert delta<1e-10
atomic_write_json(out/'validation.json',dict(status='PASS_RESEARCH_COMPOSITE',rows=len(cold),max_abs_difference=delta,
    original_bundles_restored=True,cold_process=True,official_label_paths_present=False,new_fits=0,submission_created=False,incumbent='E16'))
print(json.dumps(dict(status='PASS',max_abs_difference=delta)))
