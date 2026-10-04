from copy import deepcopy
import pytest
from bf_tap_r2.realmlp_mae_confirmation import require_development


def test_mae_confirmation_requires_closed_complete_positive_development():
    report=dict(candidate='REALMLP_MAE_A20',gains={'42':.001,'3407':.002},confirmation_qualified=True)
    audit=dict(status='passed',confirmation_qualified=True)
    terminal=dict(status='passed',actual_supervisor_exit_code=0,confirmation_qualified=True)
    require_development(report,audit,terminal)
    for source,key,value in [(0,'candidate','REALMLP_SINGLE_A20'),(0,'gains',{'42':.001,'3407':-.001}),
            (0,'gains',{'42':.001}),(1,'status','pending'),(2,'actual_supervisor_exit_code',1),(2,'confirmation_qualified',False)]:
        items=deepcopy([report,audit,terminal]);items[source][key]=value
        with pytest.raises(ValueError):require_development(*items)
