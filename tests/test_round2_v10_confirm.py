from copy import deepcopy

import pytest

pytest.importorskip('torch')
pytest.importorskip('rtdl_revisiting_models')
from bf_tap_r2.v10_confirm import selected_complete


def fixture():
    spec = {'recipes': {'mae': {}, 'smooth_l1_01': {}}, 'split_seeds': [42, 3407],
            'folds': 5, 'tie_preference_by_target': {'tap_time_len': ['mae', 'smooth_l1_01']}}
    records = []
    for name, gain in [('mae', .001), ('smooth_l1_01', .003)]:
        comparisons = {label: {'seed_gains': {'42': gain, '3407': gain}, 'seed_summary': {'mean': gain},
                        'cells': [{'seed': s, 'fold': f} for s in [42, 3407] for f in range(5)]}
                       for label in ['A35', 'Q20', 'V7_TIME']}
        records.append({'recipe': name, 'target': 'tap_time_len', 'comparisons': comparisons})
    return {'status': 'development_complete', 'records': records, 'selected_confirmation': 'smooth_l1_01'}, spec


def test_selection_rejects_partial_duplicate_and_foreign_coverage():
    summary, spec = fixture()
    assert selected_complete(summary, spec) == 'smooth_l1_01'
    partial = deepcopy(summary); partial['records'].pop()
    with pytest.raises(ValueError, match='pool'):
        selected_complete(partial, spec)
    for alteration in ['missing_fold', 'duplicate_fold', 'foreign_seed']:
        broken = deepcopy(summary)
        data = broken['records'][1]['comparisons']['V7_TIME']
        if alteration == 'missing_fold': data['cells'].pop()
        if alteration == 'duplicate_fold': data['cells'][-1] = data['cells'][0]
        if alteration == 'foreign_seed': data['seed_gains'] = {'42': .003, '7777': .003}
        with pytest.raises(ValueError, match='coverage'):
            selected_complete(broken, spec)


def test_selected_recipe_cannot_bypass_incremental_gate():
    summary, spec = fixture()
    summary['records'][1]['comparisons']['V7_TIME']['seed_gains']['3407'] = -.001
    with pytest.raises(ValueError, match='incremental'):
        selected_complete(summary, spec)
