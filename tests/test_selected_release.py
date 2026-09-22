import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.v2_selected_release import file_node,predict_node


def test_fixed_csv_blend_uses_ids_and_roundtrip_precision(tmp_path):
    ids=[f'R2S2_TEST_{i:012X}' for i in range(322)]
    values=np.linspace(500.0000000000001,600,322)
    nodes=[]
    for name,scale in [('a',1.),('b',2.)]:
        path=tmp_path/(name+'.csv')
        pd.DataFrame({'sample_id':ids,'pred_tap_iron':values*scale,'pred_tap_time_len':100}).to_csv(path,index=False,float_format='%.17g')
        nodes.append(file_node(tmp_path,path,'tap_iron'))
    blend={'kind':'mean','weights':[.5,.5],'members':nodes}
    frame=pd.DataFrame({'sample_id':ids[::-1]})
    np.testing.assert_array_equal(predict_node(tmp_path,blend,frame),(.5*values+.5*(2*values))[::-1])
    nodes[0]['sha256']='wrong'
    with pytest.raises(ValueError,match='changed'):
        predict_node(tmp_path,blend,frame)


def test_invalid_blend_weights_rejected(tmp_path):
    with pytest.raises(ValueError,match='blend'):
        predict_node(tmp_path,{'kind':'mean','weights':[-1,2],'members':[{},{}]},pd.DataFrame())
