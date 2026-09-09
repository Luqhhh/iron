"""Label-free V1 final composite loader. Models and correction are already frozen."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..artifacts import file_sha256,stable_digest
from ..exceptions import ContractError
from ..models.baseline import DualTargetBaseline
from ..schema import validate_history
from .component_export import ComponentFeatures,META,read_json
from .rate_model import RateModel
from .structural_run import predict_inputs
from .structural import apply_correction


class StructuralPredictor:
    def __init__(self,root):
        self.root=Path(root);self.m=read_json(self.root/'structural.json')
        if file_sha256(self.root/'structural.json')!=read_json(self.root/'identity.json')['sha256']:raise ContractError('structural identity mismatch')
        if self.m['candidate']!='V1_RATE_STRUCTURAL':raise ContractError('unknown structural candidate')
        base=self.root/'base_R2';b=read_json(base/'composite.json')
        if file_sha256(base/'composite.json')!=self.m['base_R2_composite_sha256']:raise ContractError('base R2 identity mismatch')
        if b['candidate']!='R2' or len(b['anchors'])!=1 or b['algorithm']!=self.m['algorithm']:raise ContractError('invalid base algorithm')
        self.a=self.m['algorithm'];self.contract=b['source_contract']
        anchor=b['anchors'][0];cutoff=pd.Timestamp(anchor['cutoff']);self.entries={};self.loaded={}
        if anchor['variant']!='R2' or set(anchor['components'])!={'E09_PROCESS_CHANGE_E02','E04'}:raise ContractError('invalid R2 pair')
        for role,c in [('OR','E09_PROCESS_CHANGE_E02'),('HR','E04')]:
            identity=anchor['components'][c];directory=base/identity['path']
            if not directory.resolve().is_relative_to(base.resolve()):raise ContractError('component path escapes bundle')
            if file_sha256(directory/'bundle.json')!=identity['sha256']:raise ContractError('base component changed')
            model=DualTargetBaseline.load(directory);history=model.load_history_snapshot();validate_history(history)
            tr=model.bundle_metadata_['training']
            if any(pd.Timestamp(tr[k])!=cutoff for k in ('fit_cutoff','history_cutoff','label_available_cutoff')):raise ContractError('base cutoff mismatch')
            if model.parameters!=self.a['baseline']['parameters'] or model.bundle_metadata_['inference_source_contract']!=self.contract:raise ContractError('base parameter/source drift')
            if tr['sample_ids_sha256']!=self.m['training']['sample_ids_sha256']:raise ContractError('base training ID mismatch')
            if (history.reference_time>=cutoff).any() or (history.available_at>cutoff).any():raise ContractError('base future history')
            self.entries[role]=dict(cutoff=str(cutoff),component=c,variant='R2')
            self.loaded[role]=(model,history)
        rate_dir=self.root/'rate'
        if file_sha256(rate_dir/'bundle.json')!=self.m['rate_bundle_sha256']:raise ContractError('rate identity mismatch')
        self.rate=RateModel.load(rate_dir);md=self.rate.metadata_
        if md['training']!=self.m['training'] or md['inference_source_contract']!=self.contract or md['rate_spec']!=self.m['registration']['rate']:raise ContractError('rate contract mismatch')
        self.history=pd.read_csv(rate_dir/'history_snapshot.csv',float_precision='round_trip',dtype={'sample_id':'string'})
        for c in ('reference_time','tap_end_time','available_at'):self.history[c]=pd.to_datetime(self.history[c])
        validate_history(self.history)
        if (self.history.reference_time>=cutoff).any() or (self.history.available_at>cutoff).any():raise ContractError('rate future history')
        if stable_digest(self.history.sort_values(['reference_time','sample_id']).sample_id.astype(str).tolist())!=self.m['training']['sample_ids_sha256']:raise ContractError('rate history IDs differ')

    def predict(self,samples,op,burden,source_contract):
        if source_contract!=self.contract:raise ContractError('public source contract mismatch')
        if list(samples)!=META:raise ContractError('prediction accepts metadata only')
        builder=ComponentFeatures(self.a,op,burden)
        parts=predict_inputs(samples,(self.entries,self.loaded,self.rate,self.history),builder,self.m['registration'])
        pred=apply_correction(parts,self.m['alpha'],self.m['registration']['rate']['unusable_predicted_rate_max'])
        return pred,parts
