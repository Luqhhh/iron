from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
import torch
from torch import nn
import yaml
from bf_tap_r2.data import FEATURES,TARGETS
from bf_tap_r2.v49_gradients import attribution_terms,latent_attributions,GradientRegressor,sampled_units,fit_partition
from test_round2_v47_factorized import frame


def settings():
    s=yaml.safe_load(Path('configs/round2_v49/SPEC.yaml').read_text())['training']
    return dict(s,width=16,tabm_k=2,embedding_dim=4,n_frequencies=4,blocks=2,auxiliary_units=3,
                max_epochs=3,batch_size=16,auxiliary_rows=4)


def test_attribution_terms_analytic_zero_and_second_gradient():
    g=torch.tensor([[[1.,0],[0,2.]]],dtype=torch.float64,requires_grad=True)
    spec,orth=attribution_terms(g,1e-8)
    assert spec.item()==1.5 and orth.item()==0
    z=torch.zeros(2,3,4,dtype=torch.float64,requires_grad=True)
    a,b=attribution_terms(z,1e-8);grad=torch.autograd.grad(a+b,z,create_graph=True)[0]
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad)==0
    assert torch.isfinite(torch.autograd.grad(grad.sum(),z)[0]).all()


def test_parameter_gradient_matches_finite_difference_and_descends():
    w=torch.tensor([[[1.,.3,.2],[.2,1.1,.4],[.5,.6,1.2]]],dtype=torch.float64,requires_grad=True)
    def objective(v):
        a,b=attribution_terms(v,1e-8);return .2*a+.3*b
    val=objective(w);grad=torch.autograd.grad(val,w)[0];epsilon=1e-6
    for idx in np.ndindex(tuple(w.shape)):
        plus=w.detach().clone();minus=plus.clone();plus[idx]+=epsilon;minus[idx]-=epsilon
        fd=float((objective(plus)-objective(minus))/(2*epsilon))
        assert abs(fd-grad[idx].item())<1e-8
    assert objective(w-.01*grad)<val


def test_tabm_hook_attribution_matches_input_finite_difference_and_isolates_rng():
    s=settings();data=frame(12);y=3+data[FEATURES[0]].to_numpy()
    m=GradientRegressor('TANGOS',s).initialize(data,y);x,cat=m._inputs(data.iloc[:2])
    m.model_.train();rng=torch.get_rng_state().clone();modes=[a.training for a in m.model_.modules()]
    member,units=sampled_units(s,0);attr=latent_attributions(m.model_,x,cat,member,units)
    assert torch.equal(rng,torch.get_rng_state())
    assert modes==[a.training for a in m.model_.modules()]
    captures=[];hook=m.model_.backbone.register_forward_hook(lambda _m,_a,o:captures.append(o.detach().clone()))
    m.model_.eval();epsilon=1e-5
    try:
        plus=x.clone();minus=x.clone();plus[0,0]+=epsilon;minus[0,0]-=epsilon
        with torch.no_grad():m.model_(plus,cat);m.model_(minus,cat)
        fd=(captures[0][0,member,units]-captures[1][0,member,units])/(2*epsilon)
        torch.testing.assert_close(attr[0,:,0],fd,atol=1e-7,rtol=1e-5)
    finally:hook.remove()
    spec,orth=attribution_terms(attr,s['norm_epsilon']);(spec+orth).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.model_.parameters())


def test_zero_lambdas_exact_control_and_live_penalty_changes_state():
    data=frame(30);y=3+data[FEATURES[0]].to_numpy();s=settings()
    base=GradientRegressor('BASE',s).initialize(data,y);base.train(2)
    zero=GradientRegressor('TANGOS',dict(s,lambda_specialization=0.,lambda_orthogonalization=0.)).initialize(data,y);zero.train(2)
    for key,value in base.model_.state_dict().items():torch.testing.assert_close(value,zero.model_.state_dict()[key],atol=0,rtol=0)
    active=GradientRegressor('TANGOS',s).initialize(data,y);active.train(2)
    assert active.auxiliary_updates_>0 and base.auxiliary_updates_==0
    assert any(not torch.equal(v,active.model_.state_dict()[k]) for k,v in base.model_.state_dict().items())


def test_cold_order_chunk_and_query_labels(tmp_path):
    data=frame(40);y=2+data[FEATURES[0]].to_numpy();s=settings()
    model=GradientRegressor('TANGOS',s).initialize(data.iloc[:30],y[:30]);model.train(2)
    p=tmp_path/'model.pt';model.save(p);cold=GradientRegressor.load(p);query=data.iloc[30:]
    expected=model.predict(query)
    for actual in [cold.predict(query),cold.predict(query.iloc[::-1])[::-1],np.concatenate([cold.predict(query.iloc[i:i+1]) for i in range(len(query))])]:
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
    with pytest.raises(ValueError,match='labels'):cold.predict(query.assign(tap_iron=0))
    with pytest.raises(FileExistsError):model.save(p)


def test_train_partition_identity_and_fresh_epoch_refit():
    data=frame(40)
    for target in TARGETS:data[target]=4+data[FEATURES[0]]
    s=settings();outer=data.iloc[:30];query=data.iloc[30:].drop(columns=list(TARGETS))
    m,p,meta,cp=fit_partition(outer.iloc[:20],outer.iloc[20:],outer,query,TARGETS[0],'TANGOS',s,np.full(10,4.),[0.,.5,1.])
    assert meta['refit']['selected_epoch']==meta['calibration']['selected_epoch']
    assert meta['refit']['fit_ids_digest']!=meta['calibration']['fit_ids_digest']
    assert p.shape==cp.shape==(10,)
    with pytest.raises(ValueError,match='overlaps'):fit_partition(outer.iloc[:20],outer.iloc[20:],outer,outer.iloc[:1].drop(columns=list(TARGETS)),TARGETS[0],'TANGOS',s,np.full(10,4.),[0.,1.])


def test_saved_audit_rejects_tampered_auxiliary_and_preprocessing(tmp_path):
    from bf_tap_r2.v49_verify import verify_model
    data=frame(30);y=2+data[FEATURES[0]].to_numpy();s=settings()
    model=GradientRegressor('TANGOS',s).initialize(data,y);model.train(2)
    original=tmp_path/'original.pt';model.save(original)
    verify_model(original,data,y,'TANGOS',s,model.metadata())
    payload=torch.load(original,weights_only=True)
    for label,mutate in [
        ('count',lambda p:p['metadata']['history'][0].update(auxiliary_updates=999)),
        ('preprocessing',lambda p:p['metadata']['preprocessing']['means'].__setitem__(0,123.)),
        ('dtype',lambda p:p['state'].__setitem__(next(iter(p['state'])),next(iter(p['state'].values())).float())),
        ('epoch',lambda p:p['metadata'].update(selected_epoch=999))]:
        bad=deepcopy(payload);mutate(bad);path=tmp_path/f'{label}.pt';torch.save(bad,path)
        with pytest.raises(ValueError):verify_model(path,data,y,'TANGOS',s)


def test_selector_audit_rejects_wrong_epoch_and_incomplete_trace(tmp_path):
    from bf_tap_r2.v49_verify import verify_model
    data=frame(40);y=2+data[FEATURES[0]].to_numpy();s=settings()
    model=GradientRegressor('BASE',s).initialize(data.iloc[:30],y[:30])
    model.train(s['max_epochs'],(data.iloc[30:],y[30:]))
    path=tmp_path/'selector.pt';model.save(path)
    verify_model(path,data.iloc[:30],y[:30],'BASE',s)
    payload=torch.load(path,weights_only=True)
    payload['metadata']['history'].pop(0);bad=tmp_path/'bad.pt';torch.save(payload,bad)
    with pytest.raises(ValueError,match='epochs'):verify_model(bad,data.iloc[:30],y[:30],'BASE',s)


def test_v49_finalist_requires_gain_over_current_reference_and_control():
    from bf_tap_r2.v49_run import select_finalists
    spec=yaml.safe_load(Path('configs/round2_v49/SPEC.yaml').read_text())
    records=[dict(target=t,recipe=r,both_seeds_positive=True,paired_seed_summary={'mean':g})
             for t in TARGETS for r,g in [('BASE',.011),('TANGOS',.012)]]
    assert all(v=='TANGOS' for v in select_finalists(records,spec).values())
    for row in records:
        if row['recipe']=='TANGOS':row['paired_seed_summary']['mean']=.009
    assert all(v is None for v in select_finalists(records,spec).values())


def test_runner_saves_auditable_unit_and_refuses_changed_artifact(tmp_path,monkeypatch):
    import json
    from bf_tap_r2 import v49_run
    from bf_tap_r2.v49_verify import verify_model
    data=frame(40)
    for target in TARGETS:data[target]=4+data[FEATURES[0]]
    training=data.iloc[:30];query=data.iloc[30:].drop(columns=list(TARGETS))
    fitting=training.iloc[:20];calibration=training.iloc[20:]
    monkeypatch.setattr(v49_run,'partitions',lambda *args:(training,query,fitting,calibration))
    spec=yaml.safe_load(Path('configs/round2_v49/SPEC.yaml').read_text());spec['training']=settings()
    ref=tmp_path/'reference-calibration-s42-f0';ref.mkdir()
    np.savez(ref/'predictions.npz',query_ids=calibration.sample_id.to_numpy(dtype=str),tap_iron=np.full(10,4.))
    out=tmp_path/'tap_iron-TANGOS-s42-f0'
    v49_run.run_unit(tmp_path,out,data,None,42,0,spec,'synthetic-identity','candidate','tap_iron','TANGOS')
    assert v49_run.verified_unit(out,'synthetic-identity')
    meta=json.loads((out/'metadata.json').read_text())
    for name,part,key in [('model.pt',training,'refit'),('calibration_model.pt',fitting,'calibration')]:
        verify_model(out/name,part,part.tap_iron,'TANGOS',settings(),meta[key])
    (out/'metadata.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):v49_run.verified_unit(out,'synthetic-identity')
