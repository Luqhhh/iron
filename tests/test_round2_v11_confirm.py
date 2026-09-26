import copy
from pathlib import Path

import pytest
import yaml

from bf_tap_r2.v11_confirm import selected_complete


def frozen_pool():
    spec = yaml.safe_load(Path('configs/round2_v11/SPEC.yaml').read_text())
    cells = [{'seed': s, 'fold': f} for s in spec['split_seeds'] for f in range(5)]
    rows = []
    for target in spec['targets']:
        for recipe in spec['recipes']:
            gain = .2 if (target, recipe) == ('tap_time_len', 'plr_quniform') else .1
            rows.append({'target': target, 'recipe': recipe, 'comparisons': {
                label: {'seed_gains': {'42': gain, '3407': gain}, 'seed_summary': {'mean': gain},
                        'cells': copy.deepcopy(cells)} for label in ['A35', 'Q20', 'CURRENT']}})
    return spec, {'status': 'development_complete', 'records': rows,
                  'selected_confirmation': {'target': 'tap_time_len', 'recipe': 'plr_quniform'}}


def test_complete_pool_selects_incremental_winner():
    spec, summary = frozen_pool()
    assert selected_complete(summary, spec) == summary['selected_confirmation']


@pytest.mark.parametrize('defect', ['duplicate', 'missing', 'seed', 'fold', 'negative', 'selection'])
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
    elif defect == 'selection':
        summary['selected_confirmation']['recipe'] = 'raw_qnormal'
    with pytest.raises(ValueError):
        selected_complete(summary, spec)
