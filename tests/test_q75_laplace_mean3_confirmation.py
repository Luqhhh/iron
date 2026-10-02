import ast
import json
from pathlib import Path

import pytest

from bf_tap_r2.q75_laplace_mean3_confirmation import ALL_SEEDS, admission, validate_spec


def values(seq):
    return dict(zip(map(str,ALL_SEEDS),seq))


def test_q75_formal_promotion_does_not_hide_negative_increment_to_ready_single():
    result=admission(values([.01,.012,.008,.016]),values([.001,.002,-.0001,.003]))
    assert result['formal_promoted'] and not result['release_eligible']
    assert 'nonpositive_seed_gain' in result['replacement_failed_conditions']


def test_all_positive_increments_still_require_positive_seed_lcb():
    result=admission(values([.01,.012,.008,.016]),values([.0001,.0001,.0001,.1]))
    assert result['formal_promoted'] and not result['release_eligible']
    assert 'nonpositive_seed_lcb95' in result['replacement_failed_conditions']


def test_both_four_seed_gates_can_pass_without_changing_weights():
    result=admission(values([.01,.012,.013,.014]),values([.001,.0012,.0013,.0014]))
    assert result['formal_promoted'] and result['release_eligible']
    assert result['paired']['positive']==4 and result['paired_over_single']['positive']==4


def test_missing_or_foreign_seed_rejected():
    with pytest.raises(ValueError,match='Four complete'):admission({'42':.01},values([.001]*4))


def test_training_cold_and_mean_procedures_unchanged_from_development():
    def bodies(name):
        tree=ast.parse(Path('src/bf_tap_r2/'+name+'.py').read_text())
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    old=bodies('q75_laplace_mean3');new=bodies('q75_laplace_mean3_confirmation')
    for function in ['train','cold','mean_three','unit_settings','unit_parts','laplace_cold_state']:
        assert old[function]==new[function]


def test_confirmation_scope_cannot_repeat_development_or_add_optimizer():
    spec=json.loads(Path('configs/q75_laplace_mean3_confirmation/SPEC.json').read_text());validate_spec(spec)
    for key,value in [('split_seeds',[42,3407]),('optimizer_runs',42),('training_seeds',[42,1042,2043]),('confirmation_fits',0)]:
        with pytest.raises(ValueError,match='scope or budget'):validate_spec(dict(spec,**{key:value}))
