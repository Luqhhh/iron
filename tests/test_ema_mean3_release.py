import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location('ema_mean3_release', WORK / 'scripts/ema_mean3_release.py')
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)


def spec():
    return json.loads((WORK / m.SPEC).read_text())


@pytest.mark.parametrize('field,value', [('torch_optimizer', 6), ('full_training_procedures', 3),
    ('replacement_weight', .5), ('training_seeds', [42, 1042]), ('new_training_seeds', [42, 1042, 2042]),
    ('new_CV', 1), ('formal_promotion', True), ('maximum_runtime_seconds', 600), ('desktop_copies', 2)])
def test_only_fixed_three_seed_explicit_exploration_can_be_released(field, value):
    s = spec()
    science = json.loads((WORK / s['scientific_config']).read_text())
    m.validate_scope(s, science)
    s[field] = value
    with pytest.raises(ValueError, match='scope'):
        m.validate_scope(s, science)


def test_equal_mean_requires_all_members_in_frozen_order_and_finite_values():
    s = spec()
    values = {42: np.array([100., 200.]), 1042: np.array([103., 194.]), 2042: np.array([97., 206.])}
    np.testing.assert_array_equal(m.mean_prediction(s, values), [100., 200.])
    with pytest.raises(ValueError, match='ordering'):
        m.mean_prediction(s, {1042: values[1042], 42: values[42], 2042: values[2042]})
    with pytest.raises(ValueError, match='ordering'):
        m.mean_prediction(s, {42: values[42], 1042: values[1042]})
    values[2042] = np.array([float('nan'), 206.])
    with pytest.raises(ValueError, match='Finite'):
        m.mean_prediction(s, values)


def test_three_member_scalar_replay_preserves_parent_iron_field_strings():
    import csv
    import io
    ids = [f'query-{i}' for i in range(322)]
    iron = ['123.4500', '00456.0'] * 161
    bases = [100., 200.] * 161
    parent = ('sample_id,pred_tap_iron,pred_tap_time_len\n' + ''.join(
        f'{name},{other},{base}\n' for name, other, base in zip(ids, iron, bases))).encode()
    values = {42: np.tile([98., 198.], 161), 1042: np.tile([101., 201.], 161),
        2042: np.tile([104., 204.], 161)}
    payload = m.release_payload(parent, ids, values[42], m.mean_prediction(spec(), values))
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    assert [r['pred_tap_iron'] for r in rows] == iron
    expected = [float(base) + .75 * ((float(a) + float(b) + float(c)) / 3 - float(a))
        for base, a, b, c in zip(bases, *values.values())]
    assert [float(r['pred_tap_time_len']) for r in rows] == expected


def test_unregistered_seed_is_rejected_before_any_scientific_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'context', lambda out: {'spec': spec()})
    with pytest.raises(ValueError, match='preregistered'):
        m.warm(tmp_path, 42)
    assert not list(tmp_path.iterdir())


def test_development_failure_is_preserved_and_real_successful_audit_required(tmp_path):
    def save(name, value):
        (tmp_path / name).write_text(json.dumps(value))
    save('manifest.json', {'files': {}})
    records = {str(seed): {'gains_vs_q75': {'EMA_MEAN3': .001},
        'equal_mean_ema_minus_equal_mean_base': .001 if seed == 42 else -.001}
        for seed in (42, 3407)}
    report = {'candidate': 'EMA_MEAN3', 'confirmation_eligible': False,
        'formal_promotion': False, 'records': records}
    save('report.json', report)
    save('independent-score.json', {'status': 'passed', 'report_sha256': m.sha(tmp_path / 'report.json')})
    terminal = {'status': 'passed', 'actual_exit_codes': [0, 0], 'native_optimizer_runs': 80,
        'cold_states': 120, 'report_sha256': m.sha(tmp_path / 'report.json'),
        'independent_score_sha256': m.sha(tmp_path / 'independent-score.json'),
        'manifest_sha256': m.sha(tmp_path / 'manifest.json')}
    save('terminal-verification.json', terminal)
    s = {'scientific_evidence': {}, 'scientific_development': str(tmp_path)}
    assert m.development_evidence(s) == {}
    terminal['actual_exit_codes'] = [0, 1]
    save('terminal-verification.json', terminal)
    with pytest.raises(ValueError, match='failed paired gate'):
        m.development_evidence(s)
