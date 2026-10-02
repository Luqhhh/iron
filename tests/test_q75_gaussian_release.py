import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from bf_tap_r2.q75_gaussian_release import validate_spec, verify_payload, zip_payload, release_ledger
from bf_tap_r2.ema_evaluation_diagnostics import sha
from bf_tap_r2.q75_gaussian_time import compose
from bf_tap_r2.submission import package, ZIP_NAME
from bf_tap_r2.v5_package import payload_with_parent_other_column


def example():
    ids = [f'R2S2_TEST_{i:012X}' for i in range(322)]
    iron = [('1.2300000000000000' if i % 2 else '9.800000000000001e+02') for i in range(322)]
    parent = payload_with_parent_other_column(ids, 'tap_time_len', np.full(322, 20.), iron)
    member = np.linspace(18., 23., 322)
    payload = payload_with_parent_other_column(ids, 'tap_time_len', compose(np.full(322,20.),member), iron)
    return ids, iron, parent, member, payload


def test_roundtrip_preserves_iron_text_and_fixed_time_arithmetic(tmp_path):
    ids, iron, parent, member, payload = example()
    package(tmp_path, payload, ids)
    reopened = zip_payload(tmp_path / ZIP_NAME)
    assert reopened == payload
    assert verify_payload(reopened, parent, ids, member)['iron_string_mismatches'] == 0
    rows = list(csv.DictReader(io.StringIO(reopened.decode())))
    assert [r['pred_tap_iron'] for r in rows] == iron
    assert float(rows[0]['pred_tap_time_len']) == .8*20+.2*18


def test_unchanged_column_float_equivalence_is_insufficient():
    ids, _, parent, member, payload = example()
    changed = payload.replace(b'1.2300000000000000', b'1.23')
    with pytest.raises(ValueError, match='iron field string'):
        verify_payload(changed, parent, ids, member)


def test_fixed_formula_and_template_order_are_required():
    ids, iron, parent, member, payload = example()
    wrong = payload_with_parent_other_column(ids, 'tap_time_len', .7*20+.3*member, iron)
    with pytest.raises(ValueError, match='endpoint differs'):
        verify_payload(wrong, parent, ids, member)
    with pytest.raises(ValueError, match='IDs or sample order'):
        verify_payload(payload, parent, ids[::-1], member)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -1.])
def test_packaging_rejects_invalid_predictions_without_clipping(bad):
    ids, iron, *_ = example()
    values = np.full(322,20.)
    values[13] = bad
    with pytest.raises(ValueError, match='finite and non-negative'):
        payload_with_parent_other_column(ids, 'tap_time_len', values, iron)


def test_extra_zip_member_is_rejected(tmp_path):
    path = tmp_path / 'extra.zip'
    with zipfile.ZipFile(path,'x') as archive:
        archive.writestr('result.csv',b'')
        archive.writestr('metadata.json',b'{}')
    with pytest.raises(ValueError, match='ZIP members'):
        zip_payload(path)


def test_release_scope_cannot_add_training_weights_or_upload():
    spec = json.loads(Path('configs/q75_gaussian_release/SPEC.json').read_text())
    validate_spec(spec)
    for key,value in [('weight',.3),('optimizer_runs',4),('new_cv_fits',1),('agent_uploads',1)]:
        with pytest.raises(ValueError, match='scope or budget'):
            validate_spec(dict(spec,**{key:value}))


def test_full_release_ledger_admits_identity_before_any_optimizer_and_rejects_overlap(tmp_path):
    sources = {str(Path(__file__).resolve()):sha(__file__)}
    ledger = release_ledger(tmp_path, ['T0','T1'], ['Q0'], sources)
    with ledger.partition(['T0'],['T1']):
        assert not any(ledger.counts.values())
    with pytest.raises(ValueError,match='did not close'):
        ledger.close()
    with pytest.raises(ValueError,match='overlap'):
        release_ledger(tmp_path / 'overlap', ['T0','T1'], ['T1'], sources)
