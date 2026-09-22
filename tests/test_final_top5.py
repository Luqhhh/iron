import numpy as np
from bf_tap_r2.final_top5 import canonical_metrics,recipes
from bf_tap_r2.v2_selected_release import predict_node


def test_metric_catalog_preserves_targets_and_splits():
    old={'R':{'42':{'tap_iron':{'wmape':.1},'tap_time_len':{'wmape':.2}},'3407':{'tap_iron':{'wmape':.3},'tap_time_len':{'wmape':.4}}}}
    new=canonical_metrics(old)
    assert new['tap_iron']['R']['3407']['wmape']==.3
    assert new['tap_time_len']['R']['42']['wmape']==.2
    assert canonical_metrics(new)==new


def test_joint_top5_recipes_keep_seed_mean_and_original_blend_order(tmp_path):
    import pandas as pd
    from bf_tap_r2.v2_selected_release import file_node
    ids=[str(i) for i in range(322)];frame=pd.DataFrame({'sample_id':ids})
    def node(name,value):
        p=tmp_path/name
        pd.DataFrame({'sample_id':ids,'pred_tap_iron':np.repeat(value,322)}).to_csv(p,index=False)
        return file_node(tmp_path,p,'tap_iron')
    # CSV loader infers numeric IDs; use nonnumeric identifiers just as official IDs.
    ids[:]=['ID_'+s for s in ids];frame['sample_id']=ids
    a=node('anchor.csv',10.);i2=node('i2.csv',14.)
    members=[node('j42.csv',20.),node('j2026.csv',26.),node('j2027.csv',29.)]
    nodes=recipes(a,i2,members,node('dj.csv',13.),node('aj.csv',15.))
    expected={'DJ':13.,'J3':25.,'AJ3':17.5,'DJ3':15.75,'AJ':15.}
    for route,value in expected.items():
        np.testing.assert_array_equal(predict_node(tmp_path,nodes[route],frame),np.repeat(value,322))
