from copy import deepcopy
import json
from pathlib import Path
import pytest
from bf_tap_r2.realmlp_mse_user_release import SCOPE,require_scope,require_closed_history
from bf_tap_r2.realmlp_time_release import require_quality


def test_exact_exploration_grant_cannot_expand_weight_or_desktop_scope():
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/realmlp_mse_user_release/SPEC.json').read_text())
    grant=dict(source='explicit_user_task',user_authorized=True,user_request='- **RealMLP MSE A20** 写桌面',scope=deepcopy(SCOPE),desktop_directory=spec['desktop_directory'],historical_quality_failure_preserved=True)
    require_scope(spec,grant)
    for field,value in [('blend_weight',.1),('desktop_copies',2),('native_adam',4),('formal_promoted',True),('agent_uploads',1)]:
        bad=deepcopy(spec);bad[field]=value
        with pytest.raises(ValueError):require_scope(bad,grant)
    for field,value in [('user_authorized',False),('user_request','继续'),('desktop_directory','/tmp/other'),('historical_quality_failure_preserved',False)]:
        bad=deepcopy(grant);bad[field]=value
        with pytest.raises(ValueError):require_scope(spec,bad)


def test_exploration_preserves_original_failure_and_does_not_override_quality_gate():
    report=dict(candidate='REALMLP_SINGLE_A20',gains={'42':.004,'3407':.008,'271828':-.0003,'314159':.009},paired=dict(passed=False,lcb95=.0001))
    audit=dict(status='passed',four_seed_gate_passed=False)
    terminal=dict(status='passed',actual_supervisor_exit_code=0,four_seed_gate_passed=False)
    require_closed_history(report,audit,terminal)
    with pytest.raises(ValueError):require_quality(report,audit,terminal)
    for source,key,value in [(0,'candidate','REALMLP_MAE_A20'),(0,'paired',dict(passed=True,lcb95=.0001)),(1,'status','pending'),(2,'actual_supervisor_exit_code',1),(2,'four_seed_gate_passed',True)]:
        items=deepcopy([report,audit,terminal]);items[source][key]=value
        with pytest.raises(ValueError):require_closed_history(*items)
