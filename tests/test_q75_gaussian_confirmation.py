import json
from pathlib import Path

import pytest

from bf_tap_r2.q75_gaussian_confirmation import gate, validate_spec


def test_positive_mean_does_not_override_negative_seed():
    result = gate(dict(zip(('42','3407','271828','314159'), [.01,.02,-.001,.01])))
    assert result['paired']['mean'] > 0
    assert not result['formal_promoted']
    assert 'nonpositive_seed_gain' in result['failed_conditions']


def test_positive_seeds_do_not_override_negative_lcb_or_missing_split():
    result = gate(dict(zip(('42','3407','271828','314159'), [.001,.001,.001,.1])))
    assert result['paired']['positive'] == 4
    assert not result['formal_promoted']
    assert 'nonpositive_seed_lcb95' in result['failed_conditions']
    with pytest.raises(ValueError, match='Four complete'):
        gate({'42': .01, '3407': .02})


def test_stable_positive_four_seed_gain_can_pass():
    assert gate(dict(zip(('42','3407','271828','314159'), [.01,.012,.009,.011])))['formal_promoted']


def test_scope_rejects_added_optimizer_reference_fit_or_weight_scan():
    spec = json.loads(Path('configs/q75_gaussian_confirmation/SPEC.json').read_text())
    validate_spec(spec)
    for key, value in [('optimizer_runs', 22), ('reference_fits', 1), ('weight', .3), ('confirmation_seeds', [7777, 12011])]:
        with pytest.raises(ValueError, match='scope or budget'):
            validate_spec(dict(spec, **{key: value}))
