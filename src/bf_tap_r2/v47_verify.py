"""Independent NumPy interpolation/ANOVA DP and saved training-trace audit."""
from pathlib import Path
import json
import numpy as np
from .data import FEATURES, TARGETS
from .v7_periodic import digest


def independent_inputs(frame, info):
    if any(t in frame for t in TARGETS): raise ValueError('Query carries labels')
    width=info['width'];p=len(FEATURES)
    basis=np.zeros((len(frame),p+1,width))
    for j,name in enumerate(FEATURES):
        knots=np.array(info['knots'][j]);identity=np.eye(len(knots))
        for k in range(len(knots)):
            basis[:,j,k]=np.interp(frame[name].to_numpy(float),knots,identity[:,k])
    for k,category in enumerate(info['categories']):
        basis[:,-1,k]=frame.spout_no.to_numpy()==category
    return basis-np.array(info['centers']),(frame[list(FEATURES)].to_numpy(float)-info['mean'])/info['std']


def independent_prediction(path,frame):
    saved=json.loads(Path(path).read_text());info=saved['metadata'];state=saved['state']
    basis,linear=independent_inputs(frame,info['preprocessing'])
    pred=np.full(len(frame),state['bias'],dtype=float)
    for j in range(len(FEATURES)): pred+=linear[:,j]*state['linear'][j]
    additive=np.array(state['additive'])
    for j in range(basis.shape[1]):
        for k in range(basis.shape[2]): pred+=basis[:,j,k]*additive[j,k]
    orders=saved['settings']['control_orders' if saved['recipe']=='AFM2' else 'candidate_orders']
    for degree in orders:
        factors=np.array(state[f'factors.{degree}']);rank=factors.shape[2]
        # Coefficient of t^degree in product_j (1 + phi_j*t), no power-sum identity.
        dp=[np.ones((len(frame),rank))]+[np.zeros((len(frame),rank)) for _ in range(degree)]
        for j in range(basis.shape[1]):
            phi=np.zeros((len(frame),rank))
            for k in range(basis.shape[2]): phi+=basis[:,j,k,None]*factors[j,k]
            for d in range(min(degree,j+1),0,-1): dp[d]+=phi*dp[d-1]
        weights=state[f'rank_weights.{degree}']
        for r in range(rank): pred+=dp[degree][:,r]*weights[r]
    return pred*info['target_std']+info['target_mean']


def verify_model(path,training,y,recipe,settings,trace=None):
    saved=json.loads(Path(path).read_text());meta=saved['metadata'];info=meta['preprocessing']
    if saved['recipe']!=recipe or saved['settings']!=settings: raise ValueError('Model spec mismatch')
    if meta['fit_ids_digest']!=digest(training.sample_id.astype(str).tolist()):
        raise ValueError('Fit identity mismatch')
    x=training[list(FEATURES)].to_numpy(float)
    expected_knots=[np.unique(np.quantile(x[:,j],np.linspace(0,1,settings['quantile_knots']))).tolist()
                    for j in range(len(FEATURES))]
    categories=sorted(int(c) for c in training.spout_no.unique())
    if info['knots']!=expected_knots or info['categories']!=categories:
        raise ValueError('Training-only basis identity mismatch')
    sizes=[len(k) for k in expected_knots]+[len(categories)]
    mean=x.mean(0);std=x.std(0);std[std<1e-12]=1.
    if (info['sizes']!=sizes or info['width']!=max(sizes)
            or info['fit_ids_digest']!=meta['fit_ids_digest']
            or info['knot_count']!=settings['quantile_knots']
            or not np.array_equal(mean,info['mean']) or not np.array_equal(std,info['std'])):
        raise ValueError('Training-only moments/shape mismatch')
    clean=training.drop(columns=[t for t in TARGETS if t in training])
    values,_=independent_inputs(clean,info)
    if not np.allclose(values.mean(0),0,atol=1e-14,rtol=0): raise ValueError('Training-only basis centering mismatch')
    y=np.asarray(y,float)
    if meta['target_mean']!=float(y.mean()) or meta['target_std']!=float(y.std()):
        raise ValueError('Target scaling mismatch')
    if meta['fit_rows']!=len(training): raise ValueError('Fit row count mismatch')
    expected_keys={'bias','linear','additive'}
    orders=settings['control_orders' if recipe=='AFM2' else 'candidate_orders']
    shapes={'bias':(),'linear':(len(FEATURES),),'additive':(len(sizes),max(sizes))}
    for degree in orders:
        expected_keys.update({f'factors.{degree}',f'rank_weights.{degree}'})
        shapes[f'factors.{degree}']=(len(sizes),max(sizes),settings['rank_per_order'])
        shapes[f'rank_weights.{degree}']=(settings['rank_per_order'],)
    if set(saved['state'])!=expected_keys: raise ValueError('Wrong order/parameter keys')
    for key,value in saved['state'].items():
        array=np.asarray(value)
        if array.shape!=shapes[key] or not np.isfinite(array).all(): raise ValueError('Invalid parameter shape/value')
    if meta['parameter_count']!=sum(np.asarray(v).size for v in saved['state'].values()):
        raise ValueError('Parameter count mismatch')
    if trace is not None:
        # Peak memory is sampled twice and may increase during serialization.
        if {k:v for k,v in meta.items() if k!='peak_rss_mib'}!={k:v for k,v in trace.items() if k!='peak_rss_mib'}:
            raise ValueError('Saved training trace mismatch')
    history=meta['history']
    if [r['epoch'] for r in history]!=list(range(1,meta['stopped_epoch']+1)):
        raise ValueError('Missing/reordered training epochs')
    if any(not np.isfinite(r['standardized_training_mae']) for r in history):
        raise ValueError('Nonfinite training trace')
    if 'calibration_standardized_mae' in history[0]:
        best=float('inf');selected=0;stale=0
        for row in history:
            if stale>=settings['patience']: raise ValueError('Training continued after patience')
            value=row['calibration_standardized_mae']
            if not np.isfinite(value): raise ValueError('Nonfinite calibration trace')
            if value<best-settings['min_delta_standardized_mae']:
                best=value;selected=row['epoch'];stale=0
            else: stale+=1
        if meta['selected_epoch']!=selected: raise ValueError('Wrong selected epoch')
        if meta['stopped_epoch']!=settings['max_epochs'] and stale!=settings['patience']:
            raise ValueError('Premature selector stop')
    elif meta['selected_epoch']!=meta['stopped_epoch']:
        raise ValueError('Refit epoch mismatch')
    return meta
