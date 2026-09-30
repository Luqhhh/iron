import numpy as np
import pandas as pd
import pytest
from bf_tap_r2.t2g_reference import ReferenceBundle
from bf_tap_r2.t2g_verify import evaluate_target,qualifies_development,qualifies_confirmation

def fixture(gains):
    seeds=list(gains)
    f=pd.DataFrame({"tap_iron":[100.]*3,"tap_time_len":[25.]*3})
    parent={s:{"tap_iron":np.array([101.]*3),"tap_time_len":np.array([26.]*3)} for s in seeds}
    reference=ReferenceBundle(f,{},parent,parent,{})
    # Time endpoint reduction: gain=50*.2*(1-member_error)/25=.4*(1-error).
    pred={s:{"LEARNED_GRAPH":np.full(3,26.-g/.4),"DENSE_CONTROL":np.full(3,26.)} for s,g in gains.items()}
    return evaluate_target(reference,pred,"tap_time_len",seeds)

def test_hand_computed_current_endpoint_score():
    r=fixture({42:.2,3407:.2})
    assert r["seed_gains"]==pytest.approx([.2,.2])
    assert r["mean_score"]==pytest.approx(97.7)
    assert r["mean_mechanism_gain"]==pytest.approx(.2)
    assert qualifies_development(r)

def test_negative_seed():
    assert not qualifies_development(fixture({42:.2,3407:-.01}))

def test_control_ineligible():
    r=fixture({42:.2,3407:.2})
    assert not qualifies_development(r,arm="DENSE_CONTROL")

def test_four_seed_lcb():
    r=fixture({42:.1,3407:.11,7777:.12,12011:.13})
    g=np.array([.1,.11,.12,.13])
    assert r["lcb95"]==pytest.approx(g.mean()-2.3533634348018264*g.std(ddof=1)/2)
    assert qualifies_confirmation(r)
    r=fixture({42:.001,3407:.001,7777:.001,12011:.2})
    assert not qualifies_confirmation(r)

def test_nan_member():
    f=pd.DataFrame({"tap_iron":[100.],"tap_time_len":[25.]})
    parent={42:{"tap_iron":np.array([101.]),"tap_time_len":np.array([26.])}}
    b=ReferenceBundle(f,{},parent,parent,{})
    with pytest.raises(ValueError): evaluate_target(b,{42:{"LEARNED_GRAPH":np.array([np.nan]),"DENSE_CONTROL":np.array([26.])}},"tap_time_len",[42])

# Artificial artifact tree: exercises row assembly and decision arithmetic, not fits.
def build_phase(root,reference):
    import json
    from pathlib import Path
    from bf_tap_r2.t2g_model import file_hash
    from bf_tap_r2.t2g_protocol import expected_units
    root.mkdir()
    records=[]
    for key in expected_units("development",["tap_iron","tap_time_len"]):
        p=root/f"{key.target}-{key.arm}-{key.seed}-{key.fold}"; p.mkdir()
        mask=reference.folds[key.seed]==key.fold
        ids=reference.frame.loc[mask,"sample_id"].tolist()
        truth=reference.frame.loc[mask,key.target].to_numpy()
        prediction=truth+(.5 if key.arm=="LEARNED_GRAPH" else 1.)
        np.savez(p/"predictions.npz",query_ids=np.asarray(ids),prediction=prediction)
        unit={"key":key.__dict__,"manifest_digest":"a"*64,"query_ids":ids,
            "hashes":{"predictions.npz":file_hash(p/"predictions.npz")},"optimizer_starts":2}
        (p/"complete.json").write_text(json.dumps(unit))
        records.append({**unit,"directory":p.name,"complete_sha256":file_hash(p/"complete.json")})
    (root/"phase.json").write_text(json.dumps({"phase":"development","selected_targets":["tap_iron","tap_time_len"],"manifest_digest":"a"*64}))
    (root/"complete.json").write_text(json.dumps({"records":records,"manifest_digest":"a"*64}))
    (root/"audit.json").write_text(json.dumps({"status":"passed","phase":"development",
        "manifest_digest":"a"*64,"complete_sha256":file_hash(root/"complete.json"),"artificial_fixture":True}))
    (root/"audit.complete.json").write_text(json.dumps({"audit_sha256":file_hash(root/"audit.json")}))
    return root

@pytest.fixture
def artificial_phase(tmp_path):
    f=pd.DataFrame({"sample_id":[f"SYNTH_{i}" for i in range(10)],"tap_iron":[100.]*10,"tap_time_len":[25.]*10})
    folds={s:np.arange(10)%5 for s in (42,3407,7777,12011)}
    parent={s:{"tap_iron":np.full(10,101.),"tap_time_len":np.full(10,26.)} for s in folds}
    b=ReferenceBundle(f,folds,parent,parent,{})
    return build_phase(tmp_path/"phase",b),b

def test_both_targets_eligible_and_forged_summary_ignored(artificial_phase):
    from bf_tap_r2.t2g_verify import decide
    root,b=artificial_phase
    (root/"summary.json").write_text('{"selected_for_confirmation":[],"score":999999}')
    r=decide(root,None,b)
    assert r["selected_for_confirmation"]==["tap_iron","tap_time_len"]

def test_missing_nan_duplicate_rows(artificial_phase):
    from bf_tap_r2.t2g_verify import read_predictions
    root,b=artificial_phase
    unit=next(p for p in root.iterdir() if p.is_dir())
    # Substitution is rejected even without trusting its changed content.
    np.savez(unit/"predictions.npz",query_ids=np.array(["duplicate","duplicate"]),prediction=np.array([np.nan,np.nan]))
    with pytest.raises(ValueError): read_predictions(root,b,"development")

def test_missing_cell_rejected(artificial_phase):
    import json
    from bf_tap_r2.t2g_verify import read_predictions
    root,b=artificial_phase
    c=json.loads((root/"complete.json").read_text()); c["records"].pop()
    (root/"complete.json").write_text(json.dumps(c))
    with pytest.raises(ValueError): read_predictions(root,b,"development")
