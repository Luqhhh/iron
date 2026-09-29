"""Training-only local Mixup and small coupled periodic input channels."""
from __future__ import annotations
import math
import numpy as np
import torch
from scipy.spatial.distance import cdist

from .component_regularization import ComponentRegressor
from .v12_joint import make_network, joint_loss
from .v7_periodic import digest


def neighbor_index(numeric, spout, ids, k):
    numeric=np.asarray(numeric,float);spout=np.asarray(spout);ids=np.asarray(ids,dtype=str)
    if len(set(ids))!=len(ids) or len(ids)!=len(numeric) or len(spout)!=len(ids) or k<1:
        raise ValueError("Invalid neighbor identity")
    if not np.isfinite(numeric).all():raise ValueError("Nonfinite neighbor inputs")
    neighbors=np.repeat(np.arange(len(ids))[:,None],k,axis=1);counts=np.zeros(len(ids),int)
    for value in sorted(set(spout)):
        group=np.flatnonzero(spout==value)
        group=group[np.argsort(ids[group],kind="stable")]
        count=min(k,len(group)-1)
        if count<=0:continue
        distances=cdist(numeric[group],numeric[group],metric="sqeuclidean")
        np.fill_diagonal(distances,np.inf)
        ranked=np.argsort(distances,axis=1,kind="stable")[:,:count]
        neighbors[group,:count]=group[ranked];counts[group]=count
    return neighbors,counts


def interpolate(x,y,partners,lam):
    lam=lam[:,None]
    return lam*x+(1-lam)*partners[0],lam*y+(1-lam)*partners[1]


class CoupledNetwork(torch.nn.Module):
    """Extra inputs to the first affine layer, mathematically concatenation."""
    def __init__(self, native, mechanisms, learned):
        super().__init__();self.native=native
        rank=mechanisms["coupled_rank"];coordinates=mechanisms["coupled_coordinates"]
        first=native.backbone.blocks[0][0]
        self.scale=mechanisms["coupled_scale"]
        generator=torch.Generator().manual_seed(mechanisms["coupled_initialization_seed"])
        n_features=native._n_num_features
        u=torch.randn(coordinates,rank,generator=generator)/math.sqrt(rank)
        v=torch.randn(rank,n_features,generator=generator)/math.sqrt(n_features)
        self.u=torch.nn.Parameter(u,requires_grad=learned)
        self.v=torch.nn.Parameter(v,requires_grad=learned)
        self.extra_weight=torch.nn.Parameter(torch.zeros(first.weight.shape[0],2*coordinates))
        scaling=torch.randn(native.k,coordinates,generator=generator)
        self.extra_r=torch.nn.Parameter(torch.cat([scaling,scaling],dim=1))

    def forward(self,x_num,x_cat):
        native=self.native
        numeric=native.num_module(x_num).flatten(1)
        categorical=native.cat_module(x_cat).flatten(1)
        x=native.ensemble_view(torch.cat([numeric,categorical],dim=1))
        u=torch.nn.functional.linear(torch.nn.functional.linear(x_num,self.v),self.u)*self.scale
        periodic=torch.cat([u.sin(),u.cos()],dim=1)
        first=native.backbone.blocks[0][0]
        addition=torch.nn.functional.linear(periodic[:,None,:]*self.extra_r,self.extra_weight)*first.s
        x=first(x)+addition
        for layer in native.backbone.blocks[0][1:]:x=layer(x)
        for block in native.backbone.blocks[1:]:x=block(x)
        return native.output(x)


class AugmentedRegressor(ComponentRegressor):
    def __init__(self,recipe,settings,arm,mechanisms,directory=None):
        if arm not in ("D-LMIX","R-FIXED","R-LEARNED"):
            raise ValueError("Unknown augmentation/representation arm")
        super().__init__(recipe,settings,"BASE",mechanisms,directory)
        self.arm=arm;self.auxiliary={}

    @staticmethod
    def network_for_load(recipe,settings,categories,outputs,arm,mechanisms):
        native=make_network(recipe,settings,categories,outputs)
        return CoupledNetwork(native,mechanisms,arm=="R-LEARNED") if arm.startswith("R-") else native

    def _initialize(self,frame,y):
        super()._initialize(frame,y)
        if self.arm.startswith("R-"):
            self.model_=CoupledNetwork(self.model_,self.mechanisms,self.arm=="R-LEARNED")
            self.optimizer_=torch.optim.AdamW(self.model_.parameters(),lr=self.settings["learning_rate"],weight_decay=self.settings["weight_decay"])

    def prepare_training(self,frame,x,cat,target):
        self.auxiliary={"arm":self.arm}
        if self.arm=="D-LMIX":
            self.neighbors,self.counts=neighbor_index(self.preprocessor_._standardized(frame),
                frame.spout_no.to_numpy(),frame.sample_id.to_numpy(),self.mechanisms["mixup_k"])
            self.mix_rng=np.random.default_rng(np.random.SeedSequence([self.settings["random_seed"],self.mechanisms["mixup_seed_stream"]]))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.settings["random_seed"]+self.mechanisms["mixup_dropout_stream"])
                self.mix_dropout_state=torch.get_rng_state()
            self.auxiliary.update(neighbor_digest=digest(self.neighbors.tolist()),counts_digest=digest(self.counts.tolist()),
                fit_ids_digest=digest(frame.sample_id.tolist()),singleton_rows=int(np.sum(self.counts==0)),synthetic_rows=0,synthetic_forward_calls=0)
        else:
            self.auxiliary.update(coordinates=self.mechanisms["coupled_coordinates"],rank=self.mechanisms["coupled_rank"],
                projection_trainable=self.arm=="R-LEARNED",initialization_seed=self.mechanisms["coupled_initialization_seed"])

    def training_loss(self,x,cat,target,idx):
        original=joint_loss(self.model_(x[idx],cat[idx]),target[idx])
        weight=self.mechanisms["mixup_loss_weight"] if self.arm=="D-LMIX" else 0.
        if not weight:return original
        valid=idx[self.counts[idx]>0]
        if not len(valid):return original
        choice=(self.mix_rng.random(len(valid))*self.counts[valid]).astype(int)
        partner=self.neighbors[valid,choice]
        lam=torch.as_tensor(self.mix_rng.beta(self.mechanisms["mixup_beta"],self.mechanisms["mixup_beta"],len(valid)),dtype=x.dtype)
        mx,my=interpolate(x[valid],target[valid],(x[partner],target[partner]),lam)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.mix_dropout_state)
            mixed=joint_loss(self.model_(mx,cat[valid]),my)
            self.mix_dropout_state=torch.get_rng_state()
        self.auxiliary["synthetic_rows"]+=len(valid);self.auxiliary["synthetic_forward_calls"]+=1
        return (1-weight)*original+weight*mixed

    def extra_saved_metadata(self):
        return self.auxiliary

    def audit_fit(self,training):
        saved=self.saved;extra=saved["auxiliary"]
        if extra["arm"]!=self.arm:raise ValueError("Auxiliary identity mismatch")
        if self.arm=="D-LMIX":
            neighbors,counts=neighbor_index(self.preprocessor_._standardized(training),training.spout_no.to_numpy(),
                training.sample_id.to_numpy(),self.mechanisms["mixup_k"])
            if extra["neighbor_digest"]!=digest(neighbors.tolist()) or extra["counts_digest"]!=digest(counts.tolist()):
                raise ValueError("Training-only neighbor mismatch")
            expected_rows=int(np.sum(counts>0))*saved["trace"]["stopped_epoch"]
            if self.mechanisms["mixup_loss_weight"]>0 and extra["synthetic_rows"]!=expected_rows:
                raise ValueError("Synthetic-row update count mismatch")
        else:
            network=self.model_
            if network.u.requires_grad!=(self.arm=="R-LEARNED") or network.v.requires_grad!=(self.arm=="R-LEARNED"):
                raise ValueError("Projection freeze mismatch")
            if self.arm=="R-FIXED":
                with torch.random.fork_rng(devices=[]):
                    native=make_network(self.recipe,self.settings,self.preprocessor_.n_spout_categories_,len(self.mean_))
                    expected=CoupledNetwork(native,self.mechanisms,False)
                if not torch.equal(network.u,expected.u) or not torch.equal(network.v,expected.v):
                    raise ValueError("Fixed projection changed")
