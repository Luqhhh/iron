"""Partition-local prototype auxiliary training on the strong periodic TabM.

This follows the pinned author-code direct-head route, not projected-head PTaRL.
There is intentionally no official-data runner or implicit phase scheduler here.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.cluster import KMeans

from .data import TARGETS
from .ptarl_terms import auxiliary_terms, sampled_rows
from .v3_6_networks import NumericPreprocessor
from .v7_periodic import digest, file_hash
from .v49_gradients import GradientRegressor, architecture, clean

ARMS = ("CONTROL", "PTARL_AUX")


def latent_forward(backbone, x, c):
    values = []
    hook = backbone.backbone.register_forward_hook(lambda _m, _args, out: values.append(out))
    try:
        prediction = backbone(x, c)
    finally:
        hook.remove()
    if len(values) != 1 or values[0].ndim != 3:
        raise ValueError("Exactly one TabM latent output required")
    return prediction, values[0].mean(1)


def validate_frame(frame):
    if (not len(frame) or frame.sample_id.isna().any()
            or frame.sample_id.astype(str).duplicated().any()):
        raise ValueError("Nonempty unique training identities required")


def prototype_bank(teacher, frame, count, random_seed):
    """Only latent rows from THIS teacher's fitting partition may enter KMeans."""
    validate_frame(frame)
    if any(t in frame for t in TARGETS):
        raise ValueError("Prototype extraction accepts unlabeled inputs")
    if frame.sample_id.astype(str).tolist() != teacher.fit_ids_:
        raise ValueError("Teacher/prototype fit row identity mismatch")
    if type(count) is not int or not 2 <= count <= len(frame):
        raise ValueError("Invalid prototype count")
    teacher.model_.eval()
    inputs = teacher._inputs(frame)
    if (not torch.equal(inputs[0],teacher.x_train_)
            or not torch.equal(inputs[1],teacher.cat_train_)):
        raise ValueError("Teacher/prototype training feature contents differ")
    with torch.no_grad():
        _, hidden = latent_forward(teacher.model_, *inputs)
    matrix = hidden.numpy()
    km = KMeans(n_clusters=count, init="k-means++", n_init=1, max_iter=100,
                tol=1e-4, random_state=random_seed, algorithm="lloyd").fit(matrix)
    if len(np.unique(km.labels_)) != count or not np.isfinite(km.cluster_centers_).all():
        raise ValueError("Degenerate or nonfinite prototype initialization")
    if (np.linalg.norm(km.cluster_centers_, axis=1) <= 1e-12).any():
        raise ValueError("Zero prototype initialization")
    # Persist witnesses for a zero-fit centroid/inertia audit. A tolerance-stop
    # can leave a small final reassignment residual; preserve it, never hide it.
    means = np.stack([matrix[km.labels_ == i].mean(0) for i in range(count)])
    return km.cluster_centers_.copy(), {"fit_ids_digest": digest(teacher.fit_ids_),
        "teacher_selected_epoch": teacher.selected_epoch_, "latent_rows": len(matrix),
        "latent_width": matrix.shape[1], "prototype_count": count,
        "centers_digest": digest(km.cluster_centers_.tolist()), "kmeans_iterations": int(km.n_iter_),
        "cluster_labels": km.labels_.tolist(), "inertia": float(km.inertia_),
        "centroid_mean_max_difference": float(np.abs(means-km.cluster_centers_).max())}


def teacher_pair(fitting, calibration, outer_training, target, settings):
    """Two warm-up optimizer runs; outer teacher starts fresh at inner-selected epoch."""
    if target not in TARGETS:
        raise ValueError("Unknown teacher target")
    for part in (fitting,calibration,outer_training):
        validate_frame(part)
    ids = [set(p.sample_id.astype(str)) for p in (fitting,calibration,outer_training)]
    if ids[0] & ids[1] or ids[2] != ids[0] | ids[1]:
        raise ValueError("Invalid fitting/calibration/outer teacher partition")
    combined = pd.concat([fitting,calibration]).set_index('sample_id').sort_index()
    if not combined.equals(outer_training.set_index('sample_id').sort_index()):
        raise ValueError("Teacher union changed row contents")
    with torch.random.fork_rng(devices=[]):
        inner = GradientRegressor('BASE',settings).initialize(fitting,fitting[target].to_numpy())
        epoch = inner.train(settings['max_epochs'],(clean(calibration),calibration[target].to_numpy()))
        outer = GradientRegressor('BASE',settings).initialize(outer_training,outer_training[target].to_numpy())
        outer.train(epoch)
    inner_bank = prototype_bank(inner,clean(fitting),settings['prototype_count'],settings['random_seed'])
    outer_bank = prototype_bank(outer,clean(outer_training),settings['prototype_count'],settings['random_seed'])
    return inner,outer,inner_bank,outer_bank


class GeometryNetwork(nn.Module):
    def __init__(self, settings, categories, centers):
        super().__init__()
        self.native = architecture(settings, categories)
        width = settings["width"]
        centers = torch.as_tensor(centers, dtype=torch.float64)
        if centers.shape != (settings["prototype_count"], width) or not torch.isfinite(centers).all():
            raise ValueError("Prototype shape/value mismatch")
        if (centers.square().sum(1) <= settings["norm_epsilon"]**2).any():
            raise ValueError("Nonzero prototype initialization required")
        self.register_buffer("initial_prototypes", centers.clone())
        self.prototypes = nn.Parameter(centers.clone())
        layers = []
        for _ in range(3):
            layers.extend([nn.Linear(width, width, dtype=torch.float64), nn.GELU(), nn.Dropout(.1)])
        layers.append(nn.Linear(width, settings["prototype_count"], dtype=torch.float64))
        self.coordinates = nn.Sequential(*layers)

    def forward(self, x, c):
        # Author-code route: prediction DOES NOT use r @ prototypes at inference.
        return self.native(x, c)

    def training_forward(self, x, c):
        pred, hidden = latent_forward(self.native, x, c)
        return pred, hidden, self.coordinates(hidden).softmax(1)


class PrototypeRegressor(GradientRegressor):
    def __init__(self, arm, settings):
        if arm not in ARMS:
            raise ValueError("Unknown prototype arm")
        super().__init__("BASE", settings)
        self.arm = arm

    def initialize(self, frame, y, centers, center_receipt):
        validate_frame(frame)
        y = np.asarray(y, float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or y.std() <= 0:
            raise ValueError("Finite nonconstant targets with matching rows required")
        if center_receipt["fit_ids_digest"] != digest(frame.sample_id.astype(str).tolist()):
            raise ValueError("Prototype/preprocessing fitting partitions differ")
        if center_receipt["centers_digest"] != digest(np.asarray(centers).tolist()):
            raise ValueError("Prototype initialization receipt differs")
        if (center_receipt['latent_rows'] != len(frame)
                or center_receipt['latent_width'] != self.settings['width']
                or center_receipt['prototype_count'] != self.settings['prototype_count']
                or not 1 <= center_receipt['teacher_selected_epoch'] <= self.settings['max_epochs']):
            raise ValueError("Prototype receipt shape/epoch mismatch")
        torch.set_num_threads(1)
        self.preprocessor_ = NumericPreprocessor(structure="raw_tabm").fit(clean(frame))
        self.mean_, self.std_ = float(y.mean()), float(y.std())
        self.fit_ids_ = frame.sample_id.astype(str).tolist()
        self.x_train_, self.cat_train_ = self._inputs(clean(frame))
        self.y_train_ = torch.tensor((y-self.mean_)/self.std_,dtype=torch.float64)
        with torch.random.fork_rng(devices=[]):
            # Fresh actual stage2 backbone; no teacher weights imported.
            torch.manual_seed(self.settings["random_seed"])
            self.model_ = GeometryNetwork(self.settings, self.preprocessor_.n_spout_categories_, centers)
        self.optimizer_ = torch.optim.AdamW(self.model_.parameters(), lr=self.settings["learning_rate"],
                                           weight_decay=self.settings["weight_decay"])
        self.center_receipt_ = deepcopy(center_receipt)
        self.trained_ = False
        self.initial_native_digest_ = digest({k: v.detach().numpy().tolist()
                                             for k,v in self.model_.native.state_dict().items()})
        return self

    def train(self, epochs, validation=None):
        if self.trained_:
            raise ValueError("No repeated fit or implicit checkpoint resume")
        if type(epochs) is not int or not 1 <= epochs <= self.settings["max_epochs"]:
            raise ValueError("Epoch count outside frozen budget")
        if validation is not None:
            validate_frame(validation[0])
            if set(self.fit_ids_) & set(validation[0].sample_id.astype(str)):
                raise ValueError("Calibration overlaps fitting rows")
            vy = np.asarray(validation[1], float)
            if vy.shape != (len(validation[0]),) or not np.isfinite(vy).all():
                raise ValueError("Invalid calibration targets")
        self.trained_ = True
        weights = self.settings["auxiliary_weights"]
        if set(weights) != {"projection", "diversity", "orthogonalization"} or any(
                not np.isfinite(v) or v < 0 for v in weights.values()):
            raise ValueError("Invalid auxiliary weights")
        active = self.arm == "PTARL_AUX" and any(weights.values())
        rng = np.random.default_rng(self.settings["random_seed"])
        best, stale, selected, checkpoint = float("inf"), 0, 0, None
        step = 0
        self.history_, self.auxiliary_updates_ = [], 0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.settings["random_seed"])
            for epoch in range(1, epochs+1):
                self.model_.train()
                order = rng.permutation(len(self.y_train_)); total = 0.
                totals = dict.fromkeys(weights, 0.)
                updates = 0
                for start in range(0, len(order), self.settings["batch_size"]):
                    idx = order[start:start+self.settings["batch_size"]]
                    self.optimizer_.zero_grad(set_to_none=True)
                    # Both arms run the coordinate dropout, so native RNG paths match.
                    p, hidden, coordinates = self.model_.training_forward(self.x_train_[idx], self.cat_train_[idx])
                    mse = (p[:,:,0]-self.y_train_[idx,None]).square().mean()
                    loss = mse
                    if active:
                        terms = auxiliary_terms(hidden, coordinates, self.model_.prototypes,
                                                self.y_train_[idx], sampled_rows(len(idx),
                                                self.settings["random_seed"], step), self.settings["norm_epsilon"])
                        loss = loss + sum(weights[k]*v for k,v in terms.items())
                        for k,v in terms.items():
                            totals[k] += float(v.detach())
                        updates += 1; self.auxiliary_updates_ += 1
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite prototype objective")
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.settings["gradient_norm_clip"],
                                             error_if_nonfinite=True)
                    self.optimizer_.step()
                    total += float(mse.detach())*len(idx); step += 1
                row = {"epoch":epoch, "standardized_training_mse":total/len(order),
                       "auxiliary_updates":updates, **{k:v/updates if updates else 0. for k,v in totals.items()}}
                if validation is not None:
                    value = float(np.abs(self.predict(validation[0])-vy).mean()/self.std_)
                    row["calibration_standardized_mae"] = value
                    if value < best-self.settings["min_delta_standardized_mae"]:
                        best, stale, selected, checkpoint = value, 0, epoch, deepcopy(self.model_.state_dict())
                    else:
                        stale += 1
                self.history_.append(row)
                if validation is not None and stale >= self.settings["patience"]:
                    break
        if validation is not None:
            if checkpoint is None:
                raise ValueError("No finite selected prototype epoch")
            self.model_.load_state_dict(checkpoint)
        else:
            selected = epoch
        self.model_.eval()
        self.selected_epoch_, self.stopped_epoch_ = selected, epoch
        return selected

    def metadata(self):
        return {**super().metadata(), "arm":self.arm, "prototype_initialization":self.center_receipt_,
                "initial_native_digest":self.initial_native_digest_, "task_head":"native_direct_latent",
                "prototype_state_digest":digest(self.model_.prototypes.detach().numpy().tolist())}

    def save(self, path):
        payload = {"arm":self.arm, "settings":self.settings, "metadata":self.metadata(),
                   "state":self.model_.state_dict()}
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)
        return file_hash(path)

    @classmethod
    def load(cls, path, expected_sha256):
        if not expected_sha256 or file_hash(path) != expected_sha256:
            raise ValueError("Externally anchored prototype model hash required")
        data = torch.load(path,map_location="cpu",weights_only=True)
        obj = cls(data["arm"],data["settings"])
        meta, state = data["metadata"], data["state"]
        obj.saved_metadata_ = meta
        prep = meta["preprocessing"]
        obj.preprocessor_ = NumericPreprocessor(structure="raw_tabm")
        obj.preprocessor_.means_, obj.preprocessor_.stds_ = np.array(prep["means"]), np.array(prep["stds"])
        obj.preprocessor_.spout_to_index_ = {int(k):v for k,v in prep["spout_vocabulary"].items()}
        obj.preprocessor_.n_spout_categories_ = prep["n_spout_categories"]
        obj.mean_, obj.std_ = meta["target_mean"], meta["target_std"]
        if not np.isfinite([obj.mean_,obj.std_]).all() or obj.std_ <= 0:
            raise ValueError("Invalid target scale")
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Nonfinite saved prototype parameters")
        with torch.random.fork_rng(devices=[]):
            obj.model_ = GeometryNetwork(obj.settings,prep["n_spout_categories"],state["initial_prototypes"])
        obj.model_.load_state_dict(state, strict=True)
        if digest(state["prototypes"].numpy().tolist()) != meta["prototype_state_digest"]:
            raise ValueError("Prototype state digest differs")
        if digest(state["initial_prototypes"].numpy().tolist()) != meta["prototype_initialization"]["centers_digest"]:
            raise ValueError("Prototype initial state differs")
        obj.model_.eval()
        return obj
