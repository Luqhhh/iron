"""Independent NumPy inference and saved-state audits; no estimator fitting."""
from pathlib import Path
import json

import numpy as np
import torch

from .data import FEATURES
from .dnnr_model import validate_frame
from .dnnr_ledger import file_hash
from .v3_4_bags import group_safe_inner_folds
from .danet_model import Regressor,ARMS,make_network,state_identity,batch_sizes

ABSOLUTE=1e-4
RELATIVE=1e-6


def close(actual,expected):
    actual,expected=np.asarray(actual,float),np.asarray(expected,float)
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Finite aligned DANet audit vectors required')
    difference=np.abs(actual-expected);bound=ABSOLUTE+RELATIVE*np.abs(expected)
    if np.any(difference>bound):raise ValueError('DANet independent/cold inference exceeds frozen numerical bound')
    return dict(maximum_absolute=float(difference.max()),maximum_bound_fraction=float(np.max(difference/bound)))


def simplex(logits):
    values=np.asarray(logits,dtype=np.float64)/2;values-=values.max()
    low,high=-1.,0.
    for _ in range(100):
        middle=(low+high)/2
        mass=float(np.maximum(values-middle,0).dot(np.maximum(values-middle,0)))
        if mass>1:low=middle
        else:high=middle
    return np.maximum(values-(low+high)/2,0)**2


def encode(frame,metadata):
    raw,spout=validate_frame(frame);means=np.asarray(metadata['means']);stds=np.asarray(metadata['stds'])
    categories=metadata['spout_categories'];dummy=np.zeros((len(frame),len(categories)+1))
    for row,value in enumerate(spout):dummy[row,categories.index(int(value))+1 if int(value) in categories else 0]=1.
    return np.column_stack(((raw-means)/stds,dummy)).astype(np.float32)


def sigmoid(value):
    exponential=np.exp(-np.abs(value))
    return np.where(value>=0,1/(1+exponential),exponential/(1+exponential))


def independent_predict(model,query):
    if set(query.sample_id.astype(str))&set(model.ids_):raise ValueError('Audit inference overlaps training identities')
    raw=encode(query,model.saved_metadata_['encoder']).astype(np.float64)
    state={k:v.detach().cpu().numpy().astype(np.float64) for k,v in model.model_.state_dict().items()}
    groups=model.settings.groups;width=model.settings.width
    def abstract(values,prefix,output_dim):
        masks=np.stack([simplex(row) for row in state[prefix+'.logits']])
        weight=state[prefix+'.projection.weight'].reshape(groups,2*output_dim,values.shape[1])
        bias=state[prefix+'.projection.bias'].reshape(groups,2*output_dim)
        gamma=state[prefix+'.normalization.bn.weight'].reshape(groups,2*output_dim)
        beta=state[prefix+'.normalization.bn.bias'].reshape(groups,2*output_dim)
        mean=state[prefix+'.normalization.bn.running_mean'].reshape(groups,2*output_dim)
        var=state[prefix+'.normalization.bn.running_var'].reshape(groups,2*output_dim)
        outputs=[]
        for group in range(groups):
            projected=(values*masks[group]).dot(weight[group].T)+bias[group]
            normalized=(projected-mean[group])*gamma[group]/np.sqrt(var[group]+1e-5)+beta[group]
            outputs.append(np.maximum(sigmoid(normalized[:,:output_dim])*normalized[:,output_dim:],0))
        return sum(outputs)
    previous=raw
    for index in range(model.settings.layers//2):
        prefix=f'blocks.{index}'
        main=abstract(abstract(previous,prefix+'.first',width//2),prefix+'.second',width)
        combined=main+abstract(raw,prefix+'.shortcut',width)
        previous=np.where(combined>=0,combined,.01*combined)
    value=previous
    for index in (0,2,4):
        value=value.dot(state[f'head.{index}.weight'].T)+state[f'head.{index}.bias']
        if index!=4:value=np.maximum(value,0)
    return model.center_+model.scale_*value[:,0]


def verify_saved(path,sha256,training,y,query,expected,*,calibration=None):
    model=Regressor.load(path,sha256);metadata=model.saved_metadata_;y=np.asarray(y,float)
    raw,spout=validate_frame(training)
    std=raw.std(0,ddof=0);std[std==0]=1.
    prep=metadata['encoder']
    if (training.sample_id.astype(str).tolist()!=model.ids_ or not np.array_equal(y,model.y_)
            or not np.array_equal(raw.mean(0),prep['means']) or not np.array_equal(std,prep['stds'])
            or sorted(set(spout.tolist()))!=prep['spout_categories']
            or not np.array_equal(encode(training,prep),model.x_)
            or float(y.mean())!=model.center_ or (float(y.std(ddof=0)) or 1.)!=model.scale_):
        raise ValueError('DANet saved training-only arrays/statistics differ')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model.settings.random_seed)
        initial=make_network(model.x_.shape[1],model.arm,model.settings).cpu().float()
    original={k:v.detach().cpu().numpy() for k,v in initial.state_dict().items()}
    initial_digest=state_identity(np.concatenate([v.ravel().astype(float) for v in original.values()]))
    if initial_digest!=metadata['initial_state_digest']:raise ValueError('DANet initial state differs')
    mask_change=sum(float((value-initial.state_dict()[name]).abs().sum())
        for name,value in model.model_.state_dict().items() if name.endswith('.logits'))
    if mask_change!=model.trace_['mask_change'][model.selected_epoch_-1]:raise ValueError('Saved selected mask-change witness differs')
    if any(np.any(model.trace_[key]<0) for key in ('training_loss','gradient_norm','mask_change')):
        raise ValueError('Negative training trajectory witness')
    if model.arm=='DANET_FIXED':
        for name,value in model.model_.state_dict().items():
            if name.endswith('.logits') and not np.array_equal(value.numpy(),original[name]):raise ValueError('Frozen mask control changed')
        if np.any(model.trace_['mask_change']!=0):raise ValueError('Frozen mask trace changed')
    batches=batch_sizes(len(training),model.settings)
    for name,value in model.model_.state_dict().items():
        if name.endswith('num_batches_tracked'):
            expected_count=model.selected_epoch_*sum(len(torch.empty(size).chunk(int(np.ceil(size/model.settings.ghost_size)))) for size in batches)
            if int(value)!=expected_count:raise ValueError('Saved normalization epoch/batch count differs')
    generator=torch.Generator().manual_seed(model.settings.random_seed)
    for permutation in model.trace_['permutations']:
        if not np.array_equal(permutation,torch.randperm(len(training),generator=generator).numpy()):raise ValueError('Training permutation trace differs')
    rates=[model.settings.learning_rate*model.settings.learning_rate_gamma**(i//model.settings.learning_rate_step_epochs) for i in range(model.actual_epochs_)]
    if not np.array_equal(model.trace_['learning_rate'],rates):raise ValueError('Training learning-rate trace differs')
    if calibration is not None:
        valid,target=calibration
        if model.calibration_ids_!=valid.sample_id.astype(str).tolist():raise ValueError('Calibration identity differs')
        losses=np.abs(model.trace_['calibration_predictions']-np.asarray(target)[None,:]).mean(1)
        selected=int(np.argmin(losses))+1
        if model.selected_epoch_!=selected or metadata['selection']!='first_minimum_raw_MAE':raise ValueError('Raw-MAE first-minimum epoch selection differs')
        close(model.predict(valid),model.trace_['calibration_predictions'][selected-1])
    elif model.calibration_ids_ or model.selected_epoch_!=model.actual_epochs_ or metadata['selection']!='fixed_epochs_no_calibration':
        raise ValueError('Whole-training refit unexpectedly used calibration')
    cold=model.predict(query)
    if not np.array_equal(cold,np.asarray(expected)):raise ValueError('Exact saved cold replay differs')
    checks=dict(independent=close(independent_predict(model,query),cold),
        reverse=close(model.predict(query.iloc[::-1].reset_index(drop=True))[::-1],cold),
        chunk=close(np.concatenate([model.predict(query.iloc[i:i+7].reset_index(drop=True)) for i in range(0,len(query),7)]),cold))
    return dict(status='passed',arm=model.arm,saved_state_sha256=sha256,selected_epoch=model.selected_epoch_,
        actual_epochs=model.actual_epochs_,checks=checks,new_estimator_fits=0,new_optimizer_runs=0,
        numerical_initializations=1,training_only_statistics_verified=True)


def verify_pair(directory,receipt_sha256,training,y,query,predictions,*,settings=None):
    directory=Path(directory);path=directory/'receipt.json'
    if not receipt_sha256 or file_hash(path)!=receipt_sha256:raise ValueError('Anchored DANet pair required')
    pair=json.loads(path.read_text());receipt=pair['receipt'];hashes=pair['model_hashes']
    if (pair['format']!='danet-matched-pair-v1' or set(hashes)!={f'{r}-{a}.npz' for r in ('inner','outer') for a in ARMS}
            or set(predictions)!=set(ARMS) or type(receipt['resource_upper_bound']) is not bool):raise ValueError('DANet pair artifact schema differs')
    assignment=group_safe_inner_folds(training,n_splits=5,seed=42);mask=assignment['fold']!=0
    fitting,calibration=training.loc[mask].reset_index(drop=True),training.loc[~mask].reset_index(drop=True);y=np.asarray(y,float)
    checks=dict(inner_seed=42,held_fold=0,inner_folds=assignment['fold'].tolist(),inner_group_hash=assignment['group_hash'],
        inner_fold_hash=assignment['inner_fold_hash'],fitting_ids=fitting.sample_id.astype(str).tolist(),
        calibration_ids=calibration.sample_id.astype(str).tolist(),outer_ids=training.sample_id.astype(str).tolist(),
        estimator_runs=4,optimizer_runs=4,selection='first_minimum_raw_MAE',
        outer_refit='fresh_encoder_target_scale_network_and_optimizer',outer_query_labels='absent')
    if any(receipt[k]!=v for k,v in checks.items()):raise ValueError('DANet pair partition/refit/count identity differs')
    reports={};epochs=0;initials={}
    for arm in ARMS:
        inner_name=f'inner-{arm}.npz';outer_name=f'outer-{arm}.npz'
        inner=Regressor.load(directory/inner_name,hashes[inner_name]);outer=Regressor.load(directory/outer_name,hashes[outer_name])
        if settings is None:settings=inner.settings
        if inner.arm!=arm or outer.arm!=arm or inner.settings!=settings or outer.settings!=settings:raise ValueError('DANet matched state settings/arm differ')
        selected=receipt['selected_epochs'][arm];wanted=inner.settings.max_epochs if receipt['resource_upper_bound'] else selected
        if (inner.actual_epochs_!=inner.settings.max_epochs or inner.selected_epoch_!=selected
                or outer.actual_epochs_!=wanted or receipt['outer_epochs'][arm]!=wanted):raise ValueError('DANet selected/fresh outer epoch identity differs')
        reports[inner_name]=verify_saved(directory/inner_name,hashes[inner_name],fitting,y[mask],calibration,
            inner.trace_['calibration_predictions'][selected-1],calibration=(calibration,y[~mask]))
        reports[outer_name]=verify_saved(directory/outer_name,hashes[outer_name],training,y,query,predictions[arm])
        initials[arm]=inner.initial_state_digest_,outer.initial_state_digest_
        epochs+=inner.actual_epochs_+outer.actual_epochs_
    if initials[ARMS[0]]!=initials[ARMS[1]] or receipt['optimizer_epochs']!=epochs:raise ValueError('Matched initialization or optimizer epoch counts differ')
    return dict(status='passed',models=reports,saved_models=4,estimator_runs=4,optimizer_runs=4,optimizer_epochs=epochs,
        new_estimator_fits=0,new_optimizer_runs=0)
