"""Independently reconstruct visibility/completion/provenance from saved policy evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from bf_tap.artifacts import atomic_write_json,stable_digest,file_sha256
from bf_tap.optimization.pseudo_history import ROW


def check(root,output):
    if output.exists():raise FileExistsError(output)
    identities=json.loads(Path('local/runs/optimization-v0.5-opt14-export-r1/component_identities.json').read_text())
    results=[]
    for directory in sorted((root/'origins').iterdir()):
        pseudo=pd.read_csv(directory/'pseudo_rows.csv',float_precision='round_trip',dtype={'sample_id':str})
        for c in ('reference_time','tap_end_time','available_at'):
            pseudo[c]=pd.to_datetime(pseudo[c],format='mixed')
        audits=[json.loads(line) for line in (directory/'audit.jsonl').read_text().splitlines()]
        predictions=pd.read_csv(directory/'U1.csv',float_precision='round_trip').set_index('sample_id')
        assert len(audits)==len(pseudo)==len(predictions)
        assert pseudo.sample_id.is_unique
        indexed=pseudo.set_index('sample_id')
        for row in audits:
            for role in ('OR','HR'):
                identity=identities[f'{directory.name}/{role}']
                assert row['base_bundle_sha256'][role]==identity['bundle_sha256']
                assert row['base_history_sha256'][role]==identity['history_snapshot_sha256']
            sample=indexed.loc[row['sample_id']];t=pd.Timestamp(row['sample_reference_time'])
            assert sample.reference_time==t
            values=np.array(row['prediction_before_update'])
            assert np.array_equal(values,sample[['tap_iron','tap_time_len']].to_numpy(dtype=float))
            assert np.array_equal(values,predictions.loc[row['sample_id']].to_numpy())
            assert sample.available_at==sample.tap_end_time==t+pd.to_timedelta(values[1],unit='min')
            visible=pseudo.loc[(pseudo.reference_time<t)&(pseudo.available_at<=t),ROW]
            assert len(visible)==row['pseudo_history_rows_visible']
            assert stable_digest(visible.astype(str).to_dict('records'))==row['pseudo_history_sha256']
            assert (str(visible.available_at.max()) if len(visible) else None)==row['pseudo_available_at_max']
            assert (int(visible.depth.max()) if len(visible) else 0)==row['pseudo_history_depth']
            pending=pseudo.loc[(pseudo.reference_time<t)&(pseudo.available_at>t)]
            assert len(pending)==row['predicted_completion_overlap']
            assert len(pending.loc[pending.spout_no==sample.spout_no])==row['same_spout_completion_overlap']
        results.append(dict(origin=directory.name,rows=len(pseudo),audits_sha256=file_sha256(directory/'audit.jsonl'),pseudo_sha256=file_sha256(directory/'pseudo_rows.csv')))
    atomic_write_json(output,dict(status='PASS',origins=results,complete_six_origins=len(results)==6))
    print(f'PASS independent provenance/visibility reconstruction: {len(results)} origins',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();check(a.run,a.output)
