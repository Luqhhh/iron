from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v34_crps import CRPSRegressor
from bf_tap_r2.v35_run import verify_convergence_contract


def test_only_iteration_cap_changes_and_bad_recipe_is_rejected():
    old=yaml.safe_load(Path("configs/round2_v34/SPEC.yaml").read_text())
    new=yaml.safe_load(Path("configs/round2_v35/SPEC.yaml").read_text())
    assert new["training"] == dict(old["training"],max_epochs=3000)
    new["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError,match="Only the declared"):
        verify_convergence_contract(Path.cwd(),new)


@pytest.mark.parametrize("recipe",["CRPS_FIXED","CRPS_SCALE"])
def test_longer_budget_preserves_model_prediction_and_training_prefixes(recipe):
    settings=yaml.safe_load(Path("configs/round2_v34/SPEC.yaml").read_text())["training"]
    settings=dict(settings,min_samples_leaf=3,max_epochs=5)
    x=np.random.default_rng(535).normal(size=(90,len(FEATURES)))
    d=pd.DataFrame(x,columns=FEATURES)
    d["sample_id"]=[f"prefix-{i}" for i in range(len(d))]
    d["spout_no"]=1+np.arange(len(d))%2
    y=100+3*x[:,0]+np.sin(x[:,1])
    old=CRPSRegressor(recipe,settings).initialize(d.iloc[:70],y[:70]);old.train(5)
    new=CRPSRegressor(recipe,dict(settings,max_epochs=12)).initialize(d.iloc[:70],y[:70]);new.train(12)
    assert new.history_[:5] == old.history_
    new.trees_,new.steps_ = new.trees_[:5],new.steps_[:5]
    assert np.array_equal(new.predict(d.iloc[70:]),old.predict(d.iloc[70:]))
