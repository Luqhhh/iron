"""Same-split DE3 incumbent arithmetic, without labels, scoring or fitting.

Every member is explicitly identified by outer split, held-out fold and training
seed. Training-seed averaging is confined to one identical held-out partition.
"""
from dataclasses import dataclass

import numpy as np

SPLIT_SEEDS=(42,3407,7777,12011)
TRAINING_SEEDS=(42,104729,130363)
ENDPOINT='maximum(parent_iron+0.5*(mean(J42,J104729,J130363)-J42),0)'


@dataclass(frozen=True)
class Member:
    split_seed: int
    fold: int
    training_seed: int
    query_ids: tuple[str,...]
    iron: np.ndarray


def assemble_columns(row_ids,folds,parent,members):
    """Return complete four-split columns and unchanged parent time copies.

    This function cannot establish provenance on its own. The phase runner
    must first audit native/reused/new artifacts and bind their external hashes.
    """
    row_ids=tuple(row_ids)
    if (not row_ids or len(set(row_ids))!=len(row_ids)
            or any(not isinstance(v,str) or not v for v in row_ids)):
        raise ValueError('Unique nonempty string row identities required')
    if set(folds)!=set(SPLIT_SEEDS) or set(parent)!=set(SPLIT_SEEDS):
        raise ValueError('Complete four-split parent required')
    expected={(s,f,t) for s in SPLIT_SEEDS for f in range(5) for t in TRAINING_SEEDS}
    library={}
    for m in members:
        key=m.split_seed,m.fold,m.training_seed
        if key not in expected or key in library:
            raise ValueError('Foreign or duplicate member identity')
        library[key]=m
    if set(library)!=expected:raise ValueError('Incomplete same-split member coverage')
    result={}
    for seed in SPLIT_SEEDS:
        fv=np.asarray(folds[seed])
        if fv.shape!=(len(row_ids),) or set(fv.tolist())!=set(range(5)):
            raise ValueError('Complete five-fold partition required')
        if set(parent[seed])!={'tap_iron','tap_time_len'}:
            raise ValueError('Isolated parent targets required')
        old={t:np.asarray(p,float) for t,p in parent[seed].items()}
        if any(p.shape!=(len(row_ids),) or not np.isfinite(p).all() for p in old.values()):
            raise ValueError('Incomplete finite parent columns')
        iron=np.full(len(row_ids),np.nan)
        for fold in range(5):
            indices=np.flatnonzero(fv==fold)
            ids=tuple(row_ids[i] for i in indices)
            predictions=[]
            for training_seed in TRAINING_SEEDS:
                m=library[seed,fold,training_seed]
                values=np.asarray(m.iron,float)
                if m.query_ids!=ids or values.shape!=(len(ids),) or not np.isfinite(values).all():
                    raise ValueError('Held-out identity/finite member mismatch')
                predictions.append(values)
            # Preserve the delivered arithmetic order exactly.
            iron[indices]=np.maximum(old['tap_iron'][indices]+.5*(np.mean(predictions,axis=0)-predictions[0]),0)
        result[seed]=dict(tap_iron=iron,tap_time_len=old['tap_time_len'].copy())
    return result
