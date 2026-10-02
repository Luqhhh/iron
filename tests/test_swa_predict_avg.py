import numpy as np
import pytest
from bf_tap_r2.swa_predict_avg import prediction_mean, replay_window

def test_predict_average_is_not_parameter_average_and_restores_model():
    torch=pytest.importorskip('torch')
    class Model:
        def __init__(self):
            self.model_=torch.nn.Sequential(torch.nn.Linear(1,1,bias=False),torch.nn.ReLU())
            self.model_[0].weight.data.zero_();self.saved={'state':self.model_.state_dict(),'trace':{'selected_epoch':2}}
        def predict(self,frame):
            with torch.no_grad():return self.model_(torch.as_tensor(frame,dtype=torch.float32)).numpy()
    m=Model();before=m.model_[0].weight.detach().clone()
    witness={'window':10,'selected_epoch':2,'epochs':[1,2],'states':[{'0.weight':torch.tensor([[-2.]])},{'0.weight':torch.tensor([[2.]])}]}
    matrix,average=replay_window(m,witness,np.array([[1.],[3.]]),column=0)
    np.testing.assert_array_equal(matrix,[[0.,0.],[2.,6.]])
    np.testing.assert_array_equal(average,[1.,3.]);np.testing.assert_array_equal(m.predict([[1.],[3.]]),[[0.],[0.]])
    assert torch.equal(before,m.model_[0].weight)

def test_nonfinite_predictions_are_rejected():
    with pytest.raises(ValueError,match='finite'):prediction_mean([[1.,float('nan')]])

def test_wrong_prediction_shape_is_rejected():
    with pytest.raises(ValueError,match='matrix'):prediction_mean(np.array([1.,2.]))

def test_prediction_mean_is_equal_epoch_weight_not_last_only():
    np.testing.assert_array_equal(prediction_mean([[1.,4.],[2.,8.],[9.,0.]]),[4.,4.])

def test_window_identity_rejected_before_inference():
    torch=pytest.importorskip('torch')
    class Model:
        saved={'state':{'w':torch.tensor([1.])},'trace':{'selected_epoch':11}}
        def predict(self,*args):raise AssertionError('invalid window must not predict')
    with pytest.raises(ValueError,match='window'):
        replay_window(Model(),{'window':10,'selected_epoch':11,'epochs':[11],'states':[{'w':torch.tensor([1.])}]},[],column=0)
