import json
from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.ema_evaluation_diagnostics import sha
from bf_tap_r2.q75_retained_v10_mae import checked_vector, decision, ledger_entries, validate_spec


def test_positive_cached_gains_never_authorize_confirmation_or_release():
    values={'42':.01,'3407':.02}
    result=decision(values,values,values)
    assert result['fresh_engineered_development_worth_registering']
    assert not result['confirmation_eligible'] and not result['formal_promoted'] and not result['release_eligible']
    assert result['evidence_limit']=='retained_predictions_only_no_scientific_checkpoint_replay'


def test_existing_ready_candidate_and_matched_control_are_binding():
    good={'42':.01,'3407':.02}
    result=decision(good,{'42':.001,'3407':-.002},{'42':-.01,'3407':.001})
    assert not result['fresh_engineered_development_worth_registering']
    assert result['failed_conditions']==['not_both_positive_vs_ready_Laplace','nonpositive_mean_vs_matched_MSE']
    with pytest.raises(ValueError,match='Two complete'):decision({'42':.01},good,good)


def test_ledger_rejects_duplicate_or_failed_units(tmp_path):
    path=tmp_path/'ledger.jsonl';good={'event':'complete','key':'same-source-seed-trial','metadata':{}}
    path.write_text(json.dumps(good)+'\n');assert len(ledger_entries(path))==1
    path.write_text(json.dumps(good)+'\n'+json.dumps(good)+'\n')
    with pytest.raises(ValueError,match='unique'):ledger_entries(path)
    path.write_text(json.dumps(dict(good,event='failed'))+'\n')
    with pytest.raises(ValueError,match='complete|Complete'):ledger_entries(path)


def test_original_fold_shape_finiteness_and_hash_are_required(tmp_path):
    path=tmp_path/'prediction.npy';np.save(path,np.arange(4.));entry={'prediction_sha256':sha(path)}
    np.testing.assert_array_equal(checked_vector(path,entry,4),np.arange(4.))
    with pytest.raises(ValueError,match='fold vector'):checked_vector(path,entry,5)
    np.save(path,np.array([0.,1.,np.nan,3.]))
    with pytest.raises(ValueError,match='identity changed'):checked_vector(path,entry,4)
    with pytest.raises(ValueError,match='finite'):checked_vector(path,{'prediction_sha256':sha(path)},4)


def test_scope_cannot_silently_fit_or_claim_cold_models():
    spec=json.loads(Path('configs/q75_retained_v10_mae/SPEC.json').read_text());validate_spec(spec)
    for key,value in [('new_fits',1),('optimizer_runs',2),('cold_states',10),('weight',.35),('automatic_confirmation',True)]:
        with pytest.raises(ValueError,match='zero-fit scope'):validate_spec(dict(spec,**{key:value}))
