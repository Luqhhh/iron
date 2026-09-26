import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip('torch')
pytest.importorskip('tabm')
pytest.importorskip('rtdl_num_embeddings')

from bf_tap_r2.data import FEATURES, SUBMISSION_COLUMNS
from bf_tap_r2.v7_periodic import file_hash
from bf_tap_r2.v12_joint import JointRegressor
from bf_tap_r2.v12_release import IronView,check_authorization,run
from bf_tap_r2.v12_release import authorization_record, blended_payload, parent_payload, save_and_cold, verify_payload


def parent_fixture():
    ids = [f'R2S2_TEST_{i:012X}' for i in range(322)]
    stream = io.StringIO(newline=''); writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(SUBMISSION_COLUMNS)
    writer.writerows((sid, '500.12345678901234500', '100.50000000000000') for sid in ids)
    return ids, stream.getvalue().encode()


def test_yaml_authorization_date_can_be_serialized_without_changing_source():
    raw = yaml.safe_load('authorization:\n  date: 2026-09-26\n  desktop_write: true\n  uploads: false\n')
    record = json.loads(json.dumps(authorization_record(raw), allow_nan=False))
    assert record == {'date': '2026-09-26', 'desktop_write': True, 'uploads': False}
    assert not isinstance(raw['authorization']['date'], str)


def test_release_preserves_time_text_and_exact_blend_with_frozen_clip(tmp_path):
    ids, parent = parent_fixture()
    member = np.linspace(480., 530., 322); member[0] = -1000.
    payload, clip = blended_payload(parent, ids, member, .5)
    result = verify_payload(payload, parent, ids, member, .5)
    assert result['time_string_mismatches'] == 0 and result['blend_maximum_absolute_difference'] == 0
    assert clip['negative_rows_clipped'] == 1
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    assert all(r['pred_tap_time_len'] == '100.50000000000000' for r in rows)
    with pytest.raises(ValueError, match='readback'):
        verify_payload(payload.replace(b'100.50000000000000', b'100.500000000000'), parent, ids, member, .5)
    with pytest.raises(ValueError, match='order'):
        verify_payload(payload, parent, ids[::-1], member, .5)
    archive = tmp_path/'parent.zip'
    with zipfile.ZipFile(archive, 'x') as z: z.writestr('result.csv', parent)
    assert parent_payload(archive, file_hash(archive), ids) == parent
    with pytest.raises(ValueError, match='identity'):
        parent_payload(archive, '0'*64, ids)


def test_saved_original_estimator_has_exact_cold_inference(tmp_path):
    rng = np.random.default_rng(721)
    frame = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f'synthetic-{i}' for i in range(len(frame))]
    frame['spout_no'] = np.arange(len(frame)) % 3 + 1
    frame['tap_time_len'] = 100 + 5*np.sin(frame.air_volume)
    frame['tap_iron'] = 500 + frame.air_volume
    settings = yaml.safe_load(Path('configs/round2_v12/SPEC.yaml').read_text())['training']
    settings.update(width=16, tabm_k=2, max_epochs=3, patience=2, embedding_dim=4, n_frequencies=4)
    model = JointRegressor({'backbone': 'tabm', 'frequency': .01}, settings).fit(frame, frame[['tap_iron','tap_time_len']].values)
    result = save_and_cold(model, frame, model.predict(frame), tmp_path/'cold', 1e-6)
    assert result['bit_identical'] and result['maximum_absolute_difference'] == 0
    assert result['training_reads_prohibited']
    query = pd.read_pickle(tmp_path/'cold/query.pkl')
    assert 'tap_time_len' not in query and 'tap_iron' not in query
    before = (tmp_path/'cold/model.pkl').read_bytes()
    with pytest.raises(FileExistsError):
        save_and_cold(model, frame, model.predict(frame), tmp_path/'cold', 1e-6)
    assert (tmp_path/'cold/model.pkl').read_bytes() == before
    view=IronView(model)
    np.testing.assert_array_equal(view.predict(frame),model.predict(frame)[:,0])
    result=save_and_cold(view,frame,view.predict(frame),tmp_path/'iron-cold',1e-6)
    assert result['bit_identical']
    ids,parent=parent_fixture()
    query=frame.iloc[np.arange(322)%len(frame)].reset_index(drop=True)
    query['sample_id']=ids
    prediction=view.predict(query)
    result=save_and_cold(view,query,prediction,tmp_path/'payload-cold',1e-6,parent,.5)
    payload=(tmp_path/'payload-cold/cold-result.csv').read_bytes()
    assert verify_payload(payload,parent,ids,prediction,.5)['time_string_mismatches']==0


def test_cold_training_read_guard_blocks_filesystem_access(tmp_path):
    forbidden = tmp_path/'train_samples.csv'; forbidden.write_text('private')
    code = "import sys;from bf_tap_r2.submission import deny_training_reads;sys.addaudithook(deny_training_reads);open(sys.argv[1]).read()"
    completed = subprocess.run([sys.executable, '-c', code, str(forbidden)], capture_output=True, text=True)
    assert completed.returncode != 0 and 'Inference must not read training inputs' in completed.stderr


def test_pending_proposal_blocks_before_any_fit_or_output(tmp_path):
    proposal=yaml.safe_load(Path('configs/round2_v12/RELEASE_PROPOSAL.yaml').read_text())
    with pytest.raises(ValueError,match='authorization'):
        check_authorization(proposal)
    spec=tmp_path/'configs/round2_v12/RELEASE_PROPOSAL.yaml'
    spec.parent.mkdir(parents=True);spec.write_text(yaml.safe_dump(proposal))
    with pytest.raises(ValueError,match='authorization'):
        run(tmp_path,Path('configs/round2_v12/RELEASE_PROPOSAL.yaml'))
    assert not (tmp_path/'local').exists()
    proposal['authorization'].update(user_requested_release=True,local_working_gate_exception='this_candidate_only',desktop_write=True)
    check_authorization(proposal)
    proposal['blend_weight']=.35
    with pytest.raises(ValueError,match='identity'):
        check_authorization(proposal)
