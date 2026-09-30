from dataclasses import replace

import numpy as np
import pytest

from bf_tap_r2.incumbent_reference_columns import Member,assemble_columns,SPLIT_SEEDS,TRAINING_SEEDS


def example():
    ids=tuple(f'row-{i}' for i in range(10))
    folds={s:(np.arange(10)+k)%5 for k,s in enumerate(SPLIT_SEEDS)}
    parent={s:dict(tap_iron=np.arange(10)-3.,tap_time_len=np.arange(10)+100.) for s in SPLIT_SEEDS}
    members=[]
    for k,s in enumerate(SPLIT_SEEDS):
        for f in range(5):
            ix=np.flatnonzero(folds[s]==f)
            for t,value in zip(TRAINING_SEEDS,[10.,12.+k,14.+2*k]):
                members.append(Member(s,f,t,tuple(ids[i] for i in ix),np.full(len(ix),value)))
    return ids,folds,parent,members


def test_exact_delivery_endpoint_and_seed_specific_mean_preserve_time():
    ids,folds,parent,members=example();result=assemble_columns(ids,folds,parent,members)
    for k,s in enumerate(SPLIT_SEEDS):
        np.testing.assert_array_equal(result[s]['tap_iron'],np.maximum(parent[s]['tap_iron']+1+.5*k,0))
        np.testing.assert_array_equal(result[s]['tap_time_len'],parent[s]['tap_time_len'])
        assert not np.shares_memory(result[s]['tap_time_len'],parent[s]['tap_time_len'])


@pytest.mark.parametrize('defect',['missing','duplicate','foreign','row_order','nonfinite','row_count'])
def test_member_coverage_and_partition_defects_are_rejected(defect):
    ids,folds,parent,members=example()
    if defect=='missing':members.pop()
    elif defect=='duplicate':members.append(members[0])
    elif defect=='foreign':members[0]=replace(members[0],split_seed=271828)
    elif defect=='row_order':members[0]=replace(members[0],query_ids=members[0].query_ids[::-1])
    elif defect=='nonfinite':members[0]=replace(members[0],iron=np.full(2,np.nan))
    else:members[0]=replace(members[0],iron=np.ones(3))
    with pytest.raises(ValueError):assemble_columns(ids,folds,parent,members)


def test_cross_split_member_substitution_cannot_fill_missing_unit():
    ids,folds,parent,members=example()
    foreign=next(m for m in members if m.split_seed==3407 and m.fold==0 and m.training_seed==42)
    members[0]=foreign
    with pytest.raises(ValueError,match='duplicate'):assemble_columns(ids,folds,parent,members)


@pytest.mark.parametrize('defect',['split','fold','parent','ids'])
def test_complete_four_split_context_required(defect):
    ids,folds,parent,members=example()
    if defect=='split':del folds[12011]
    elif defect=='fold':folds[42]=np.zeros(10,int)
    elif defect=='parent':parent[42]['tap_iron']=np.ones(9)
    else:ids=ids[:-1]+(ids[0],)
    with pytest.raises(ValueError):assemble_columns(ids,folds,parent,members)
