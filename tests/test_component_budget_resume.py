from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from bf_tap_r2.component_budget_resume import validate_amendment,OLD_SPEC,NEW_SPEC


def test_only_explicit_budget_change_and_runtime_failure_are_admissible():
    old=yaml.safe_load(Path(OLD_SPEC).read_text());new=yaml.safe_load(Path(NEW_SPEC).read_text())
    report={'status':'failed','checks':{'learnability':True,'runtime':False},'projected_hours':7.928570820616652}
    validate_amendment(old,new,report)
    for changed in ('training','mechanisms','promotion'):
        bad=deepcopy(new);bad[changed]['unapproved_change']=True
        with pytest.raises(ValueError,match='Only the authorized'):validate_amendment(old,bad,report)
    bad=deepcopy(report);bad['checks']['learnability']=False
    with pytest.raises(ValueError,match='runtime-only'):validate_amendment(old,new,bad)
    bad=deepcopy(report);bad['projected_hours']=8.01
    with pytest.raises(ValueError,match='exceeds'):validate_amendment(old,new,bad)
