from copy import deepcopy
from fractions import Fraction as F

import pytest

from bf_tap_r2.v46_family_bound import analyze, canonical_cells, exact_coarse, partitioned_bound
from bf_tap_r2.v46_bound_audit import verify, verify_partition


def affine_design():
    points={'O':['0','0'],'X':['1','0'],'Y':['0','1'],'M':['1/3','1/3']}
    return {k:[v,str(F(90)+F(v[0])/10+F(v[1])/5),'0'] for k,v in points.items()}


def peak_design(error='0'):
    points={'O':['0','0'],'X':['1','0'],'Y':['0','1'],
            'A':['1/5','1/5'],'B':['3/5','1/5'],'C':['1/5','3/5']}
    return {k:[v,str(1-abs(F(v[0])-F(2,5))-abs(F(v[1])-F(3,10))),error] for k,v in points.items()}


@pytest.mark.parametrize('n',[1,2,4])
def test_exact_affine_domain_bound_and_tiling(n):
    design=affine_design();result=partitioned_bound(design,n,2)
    bound,count=verify_partition(design,result,n,2)
    assert bound==F('90.2') and count==3*n*n
    assert result['vertex_count']==(n+1)*(n+2)//2
    assert len(canonical_cells(n,1))==n


def test_known_concave_interior_peak_refinement_and_uncertainty():
    design=peak_design();coarse=exact_coarse(design,[[0,0],[1,0],[0,1]])
    fine=partitioned_bound(design,4,2)
    assert F(1)<=F(fine['exact_upper'])<=F(coarse['exact_upper'])
    wide=partitioned_bound(peak_design('0.01'),4,2)
    assert F(wide['exact_upper'])>=F(fine['exact_upper'])
    verify_partition(design,fine,4,2)


def test_audit_rejects_holes_duplicate_cells_and_mixed_anchors():
    design=affine_design();result=partitioned_bound(design,2,2)
    bad=deepcopy(result);bad['cells'].pop()
    with pytest.raises(ValueError,match='coverage'):verify_partition(design,bad,2,2)
    bad=deepcopy(result);bad['cells'][1]=deepcopy(bad['cells'][0])
    with pytest.raises(ValueError,match='coverage'):verify_partition(design,bad,2,2)
    bad=deepcopy(result)
    different=next(k for k in design if k!=bad['cells'][0]['anchor'])
    bad['cells'][0]['vertex_certificates'][0]['anchor']=different
    with pytest.raises(ValueError,match='common anchor'):verify_partition(design,bad,2,2)


def test_audit_rejects_invalid_dual_and_inward_presentation():
    design=peak_design();result=partitioned_bound(design,2,2)
    bad=deepcopy(result)
    cert=next(c for cell in bad['cells'] for c in cell['vertex_certificates'] if c['dual_weights'])
    cert['dual_weights'][next(iter(cert['dual_weights']))]='-1'
    with pytest.raises(ValueError,match='negative'):verify_partition(design,bad,2,2)
    bad=deepcopy(result);bad['upper']-=1
    with pytest.raises(ValueError,match='inward'):verify_partition(design,bad,2,2)


def test_full_exact_family_and_verifier_does_not_call_optimizer(monkeypatch):
    design=affine_design()
    base={'time_observations':design,'time_domain_vertices':[[0,0],[1,0],[0,1]],
          'iron_observations':{'W0':[[0],90,0],'W05':[[.5],90.1,0],'W1':[[1],90.2,0]},
          'iron_domain_vertices':[[0],[1]]}
    spec=dict(base,grid_denominator=2,platform_milestones=[90.25,90.3,90.4])
    report=analyze(spec,base)
    assert F(report['family_exact_upper'])==F('90.3')
    assert [r['excluded_under_assumptions'] for r in report['milestones']]==[False,False,True]
    def forbidden(*a,**k):raise AssertionError('Optimizer must not run during verification')
    monkeypatch.setattr('scipy.optimize.linprog',forbidden)
    monkeypatch.setattr('bf_tap_r2.v41_concavity_bound.linprog',forbidden)
    assert verify(spec,base,report)['status']=='passed'
    bad=deepcopy(report);bad['milestones'][0]['excluded_under_assumptions']=True
    with pytest.raises(ValueError,match='Milestone'):verify(spec,base,bad)


def test_inconsistent_concavity_is_rejected():
    with pytest.raises(ValueError,match='Inconsistent'):
        partitioned_bound({'A':[[0],0,0],'B':[[.5],-1,0],'C':[[1],0,0]},2,1)
