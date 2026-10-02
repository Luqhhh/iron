"""Equal raw-epoch prediction averaging without training or checkpoint selection."""
import numpy as np

def prediction_mean(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=2 or a.shape[0]<1 or a.shape[1]<1:
        raise ValueError('Nonempty epoch by row matrix required')
    if not np.isfinite(a).all():raise ValueError('Predictions must be finite')
    return a.mean(axis=0)

def replay_window(model,witness,frame,column=1):
    from .tabm_swa_audit import verify_window
    verify_window(model.saved,witness)
    original={k:v.detach().clone() for k,v in model.model_.state_dict().items()}
    values=[]
    try:
        for state in witness['states']:
            model.model_.load_state_dict(state);model.model_.eval()
            values.append(model.predict(frame)[:,column])
    finally:
        model.model_.load_state_dict(original);model.model_.eval()
    matrix=np.stack(values).astype(float)
    return matrix,prediction_mean(matrix)
