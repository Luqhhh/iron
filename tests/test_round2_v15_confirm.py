import copy
from pathlib import Path

import pytest
import yaml

from bf_tap_r2.v15_confirm import selected_complete


def frozen_pool():
    spec = yaml.safe_load(Path('configs/round2_v15/SPEC.yaml').read_text())
    cells = [{'seed': s, 'fold': f} for s in spec['split_seeds'] for f in range(5)]
    rows = []
    for target in spec['targets']:
        for recipe in spec['recipes']:
            gain = .2 if (target, recipe) == ('tap_time_len', 'task_gated_experts') else .1
            rows.append({'target': target, 'recipe': recipe, 'comparisons': {
                label: {'seed_gains': {'42': gain, '3407': gain}, 'seed_summary': {'mean': gain},
                        'cells': copy.deepcopy(cells)} for label in ['A60', 'A35', 'Q20', 'CURRENT']}})
    return spec, {'status': 'development_complete', 'records': rows,
                  'selected_confirmation': {'target': 'tap_time_len', 'recipe': 'task_gated_experts'}}


def test_complete_pool_selects_incremental_winner():
    spec, summary = frozen_pool()
    assert selected_complete(summary, spec) == summary['selected_confirmation']


@pytest.mark.parametrize('defect', ['duplicate', 'missing', 'seed', 'fold', 'negative', 'negative_platform', 'selection'])
def test_confirmation_refuses_incomplete_or_unearned_selection(defect):
    spec, summary = frozen_pool()
    if defect == 'duplicate':
        summary['records'][-1] = copy.deepcopy(summary['records'][0])
    elif defect == 'missing':
        summary['records'].pop()
    elif defect == 'seed':
        summary['records'][0]['comparisons']['CURRENT']['seed_gains'].pop('42')
    elif defect == 'fold':
        cells = summary['records'][0]['comparisons']['A35']['cells']
        cells[0] = copy.deepcopy(cells[1])
    elif defect == 'negative':
        for row in summary['records']:
            row['comparisons']['CURRENT']['seed_gains']['42'] = -1
    elif defect == 'negative_platform':
        for row in summary['records']:
            row['comparisons']['A60']['seed_gains']['3407'] = -1
    elif defect == 'selection':
        summary['selected_confirmation']['recipe'] = 'uniform_experts'
    with pytest.raises(ValueError):
        selected_complete(summary, spec)


def test_joint_confirmation_requires_both_columns_and_selects_time():
    import numpy as np
    from bf_tap_r2.v15_confirm import time_column
    values=np.array([[600.,120.],[500.,90.]])
    np.testing.assert_array_equal(time_column(values,2),values[:,1])
    for invalid in [values[:,0],values[:,:1],values[:1],np.full((2,2),np.nan)]:
        with pytest.raises(ValueError):
            time_column(invalid,2)
