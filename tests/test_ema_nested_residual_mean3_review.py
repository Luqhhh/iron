import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest

p=Path(__file__).resolve().parents[1]/'scripts/ema_nested_residual_mean3_review.py'
s=importlib.util.spec_from_file_location('nested_mean3',p);review=importlib.util.module_from_spec(s);s.loader.exec_module(review)

def test_only_seed42_member_is_corrected_at_its_actual_portfolio_weight():
    q75=np.array([100.,200.]);members=np.array([[10.,20.],[12.,22.],[14.,24.]]);head=np.array([4.,-8.])
    base=q75+.75*(members.mean(axis=0)-members[0]);changed=members.copy();changed[0]+=.25*head
    expected=q75+.75*(changed.mean(axis=0)-members[0])
    np.testing.assert_array_equal(review.corrected(base,head),expected)
    np.testing.assert_array_equal(review.corrected(base,head)-base,(.75*.25*head)/3)
    np.testing.assert_array_equal(review.corrected(base,np.zeros(2)),base)

def test_invalid_or_misaligned_predictions_are_rejected():
    with pytest.raises(ValueError,match='Aligned'):review.corrected([1,2],[1])
    with pytest.raises(ValueError,match='Invalid'):review.corrected([1],[np.nan])
    with pytest.raises(ValueError,match='Invalid'):review.corrected([1],[-100])

def test_complete_two_seed_positive_gate_and_frozen_tie_order():
    assert review.choose({'RIDGE':{'42':1.,'3407':1.},'GBM':{'42':1.,'3407':1.}})=='RIDGE'
    assert review.choose({'RIDGE':{'42':1.,'3407':-1.},'GBM':{'42':.1,'3407':.2}})=='GBM'
    assert review.choose({'RIDGE':{'42':1.,'3407':0.},'GBM':{'42':0.,'3407':1.}}) is None
    with pytest.raises(ValueError,match='Complete'):review.choose({'RIDGE':{'42':1.},'GBM':{'42':1.,'3407':1.}})

def test_original_controller_and_audit_must_both_be_actually_closed(tmp_path):
    e=tmp_path/'execution';e.mkdir()
    for name,v in [('terminal',{'status':'passed'}),('independent-terminal-audit',{'status':'passed'}),
                   ('final-reconciliation',{'status':'running','actual_supervisor_exec_exit_code':0})]:
        (e/(name+'.json')).write_text(json.dumps(v))
    with pytest.raises(ValueError,match='Actual successful'):review.require_terminal(tmp_path)
