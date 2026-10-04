from copy import deepcopy
import pytest
from bf_tap_r2.realmlp_mae_release import require_quality


def test_mae_release_needs_its_own_completed_four_seed_gate():
    report=dict(candidate='REALMLP_MAE_A20',gains={str(s):.004 for s in [42,3407,271828,314159]},paired=dict(passed=True,lcb95=.001))
    audit=dict(status='passed',four_seed_gate_passed=True)
    terminal=dict(status='passed',actual_supervisor_exit_code=0,four_seed_gate_passed=True)
    require_quality(report,audit,terminal)
    for source,key,value in [(0,'candidate','REALMLP_SINGLE_A20'),(0,'gains',{'42':.01,'3407':.01,'271828':-.0003,'314159':.01}),
            (0,'gains',{'42':.004,'3407':.004}),(0,'paired',dict(passed=True,lcb95=-.001)),(1,'four_seed_gate_passed',False),
            (2,'actual_supervisor_exit_code',1),(2,'four_seed_gate_passed',False)]:
        items=deepcopy([report,audit,terminal]);items[source][key]=value
        with pytest.raises(ValueError):require_quality(*items)
