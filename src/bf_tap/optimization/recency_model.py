"""Fixed 60-day sample weights for original-schema direct E09/R2 regressors."""
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from ..artifacts import atomic_write_json,file_sha256,stable_digest
from ..exceptions import ContractError
from ..models.baseline import FROZEN_PARAMETERS
from .component_export import read_json
from .rate_model import schema

TARGETS=('tap_iron','tap_time_len')
HALF_LIFE_DAYS=60.


def canonical_time(values):
    if not isinstance(values.dtype,pd.DatetimeTZDtype) or values.isna().any():
        raise ContractError('complete timezone-aware times required')
    return values.dt.tz_convert('Asia/Shanghai')


def weights(metadata,cutoff):
    required={'sample_id','reference_time','available_at'}
    if required-set(metadata) or metadata.empty or metadata.sample_id.isna().any() or metadata.sample_id.astype(str).duplicated().any():
        raise ContractError('unique complete training IDs and timing metadata required')
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:raise ContractError('timezone-aware cutoff required')
    reference=canonical_time(metadata.reference_time);available=canonical_time(metadata.available_at)
    if (reference>=cutoff).any() or (available>cutoff).any() or (available<reference).any():
        raise ContractError('training time or label availability boundary violation')
    try:
        age=(cutoff-reference).dt.total_seconds().to_numpy(dtype=float)/86400.
        raw=np.exp2(-age/HALF_LIFE_DAYS)
        # Canonical ID summation makes normalization invariant to metadata order;
        # the actual training rows retain their original chronological order.
        order=np.argsort(metadata.sample_id.astype(str).to_numpy(),kind='mergesort')
        value=raw/raw[order].mean()
    except (ValueError,OverflowError) as exc:
        raise ContractError('invalid weight times') from exc
    if not np.isfinite(age).all() or not np.isfinite(value).all() or (raw<=0).any() or (value<=0).any():
        raise ContractError('finite strictly positive recency weights required')
    if not np.isclose(value.mean(),1.,rtol=0.,atol=1e-12):raise ContractError('weight normalization differs')
    index=pd.Index(metadata.sample_id.astype(str),name='sample_id')
    return pd.Series(value,index=index,name='weight'),pd.DataFrame(dict(sample_id=index,age_days=age,raw_weight=raw,weight=value))


def weight_audit(metadata,weight):
    index=pd.Index(metadata.sample_id.astype(str),name='sample_id')
    if not weight.index.equals(index):raise ContractError('weight audit IDs differ')
    data=metadata.copy();data['weight']=weight.to_numpy();data['month']=canonical_time(data.reference_time).dt.strftime('%Y-%m')
    def summarize(part):
        w=part.weight.to_numpy();ess=float(w.sum()**2/np.dot(w,w))
        return dict(rows=len(w),sum=float(w.sum()),mean=float(w.mean()),ESS=ess,ESS_over_n=ess/len(w),
            quantiles={str(q):float(np.quantile(w,q)) for q in (0.,.05,.25,.5,.75,.95,1.)},
            month_weight_share=(part.groupby('month').weight.sum()/w.sum()).to_dict())
    return dict(global_summary=summarize(data),by_spout={str(k):summarize(v) for k,v in data.groupby('spout_no')},
                interpretation='training distribution diagnostics, not independent sample counts')


def original_schema(x,expected):
    if schema(x)!=expected or any(c.startswith('trajectory__') for c in x) or set(x)&{'weight','age_days','raw_weight','reference_time','available_at','sample_id'}:
        raise ContractError('original E09/R2 schema required; no trajectory/time/weight features')


class RecencyModel:
    def fit(self,x,y,weight,metadata,cutoff,target,expected_schema,budget):
        original_schema(x,expected_schema)
        expected,_=weights(metadata,cutoff)
        if target not in TARGETS or y.name!=target or not all(z.index.equals(expected.index) for z in (x,y,weight)):
            raise ContractError('target/feature/weight sample IDs are misaligned')
        if not np.array_equal(weight.to_numpy(),expected.to_numpy()):raise ContractError('fixed recency weights differ')
        values=y.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values<0).any():raise ContractError('finite nonnegative direct labels required')
        self.target=target;self.schema=schema(x)
        self.model=CatBoostRegressor(**FROZEN_PARAMETERS);budget.allow(self.model)
        self.model.fit(x,y,cat_features=['spout_no'],sample_weight=weight)
        return self

    def predict(self,x):
        original_schema(x,self.schema)
        value=np.asarray(self.model.predict(x),dtype=float)
        if not np.isfinite(value).all():raise ContractError('nonfinite recency prediction')
        return np.maximum(value,0.)

    def save(self,root,training):
        root=Path(root);root.mkdir(parents=True,exist_ok=False)
        self.model.save_model(root/'direct.cbm')
        atomic_write_json(root/'bundle.json',dict(kind='E09_R2_RECENCY60_DIRECT',target=self.target,
            parameters=FROZEN_PARAMETERS,half_life_days=HALF_LIFE_DAYS,feature_schema=self.schema,
            training=training,model_sha256=file_sha256(root/'direct.cbm')))

    @classmethod
    def load(cls,root):
        root=Path(root);md=read_json(root/'bundle.json')
        if md['kind']!='E09_R2_RECENCY60_DIRECT' or md['parameters']!=FROZEN_PARAMETERS or md['half_life_days']!=HALF_LIFE_DAYS or md['target'] not in TARGETS or file_sha256(root/'direct.cbm')!=md['model_sha256']:
            raise ContractError('recency model identity differs')
        obj=cls();obj.target=md['target'];obj.schema=md['feature_schema'];obj.metadata=md
        obj.model=CatBoostRegressor();obj.model.load_model(root/'direct.cbm')
        return obj
