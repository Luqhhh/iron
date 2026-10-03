"""Closed same-split affine correction primitives; no fitting or clipping."""
import numpy as np

def affine(columns, coefficients):
    a=[np.asarray(v,dtype=np.float64) for v in columns]
    if len(a)!=len(coefficients) or not a or a[0].ndim!=1 or any(v.shape!=a[0].shape for v in a):
        raise ValueError("Column identity differs")
    if not all(np.isfinite(v).all() for v in a) or not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite affine input")
    result=sum(c*v for c,v in zip(coefficients,a))
    if not np.isfinite(result).all() or (result<0).any():
        raise ValueError("Invalid affine endpoint; no clipping permitted")
    return result

def assemble(ids, folds, records):
    ids=np.asarray(ids);folds=np.asarray(folds)
    if ids.ndim!=1 or folds.shape!=ids.shape or len(set(ids))!=len(ids):
        raise ValueError("Invalid full-population identity")
    result=np.full(len(ids),np.nan);count=np.zeros(len(ids),int)
    for q,query_ids,pred,fold in records:
        q=np.asarray(q);pred=np.asarray(pred)
        if q.ndim!=1 or not np.issubdtype(q.dtype,np.integer) or len(set(q))!=len(q) or (q<0).any() or (q>=len(ids)).any():
            raise ValueError("Invalid query indices")
        if pred.shape!=q.shape or not np.isfinite(pred).all() or (pred<0).any() or not np.array_equal(query_ids,ids[q]) or not np.all(folds[q]==fold):
            raise ValueError("Prediction/ID/fold identity differs")
        result[q]=pred;count[q]+=1
    if not np.all(count==1) or not np.isfinite(result).all():
        raise ValueError("OOF must cover each ID exactly once")
    return result

def choose(records, order):
    eligible=[n for n in order if records[n]["mean_gain"]>0]
    return sorted(eligible,key=lambda n:(-records[n]["mean_gain"],-records[n]["minimum_gain"],order.index(n)))[:1]
