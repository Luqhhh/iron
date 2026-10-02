import json
from pathlib import Path

import pytest

from bf_tap_r2.q75_laplace_confirmation import validate_spec
from bf_tap_r2.q75_gaussian_confirmation import gate


def test_confirmation_cannot_silently_change_arm_weight_seeds_or_budget():
    spec=json.loads(Path('configs/q75_laplace_confirmation/SPEC.json').read_text())
    validate_spec(spec)
    for key,value in [('arm','LAPLACE_SCALE'),('weight',.3),('optimizer_runs',22),
        ('confirmation_seeds',[42,3407]),('engineering_optimizer_runs',2),('reference_fits',1)]:
        with pytest.raises(ValueError,match='scope'):validate_spec(dict(spec,**{key:value}))


def test_large_mean_cannot_hide_one_negative_confirmation_split():
    result=gate({'42':.00895,'3407':.01217,'271828':-.001,'314159':.02})
    assert result['paired']['mean']>0 and not result['formal_promoted']
