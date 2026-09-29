"""Independent scalar-equation NumPy inference and saved partition verification."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import FEATURES, TARGETS
from .v7_periodic import digest, file_hash


def saved(path, expected_sha256):
    if not expected_sha256 or file_hash(path) != expected_sha256:
        raise ValueError("External model hash mismatch")
    return json.loads(Path(path).read_text())


def affine(x, w, b):
    # Independent sum of coordinate products, not torch/BLAS linear layers.
    w = np.asarray(w, float)
    b = np.asarray(b, float)
    result = np.broadcast_to(b, (len(x), len(b))).copy()
    for j in range(x.shape[1]):
        result += x[:, j:j+1] * w[:, j]
    return result


def independent_prediction(path, frame, expected_sha256):
    if any(t in frame for t in TARGETS):
        raise ValueError("Query labels forbidden")
    data = saved(path, expected_sha256)
    meta, settings, state = data["metadata"], data["settings"], data["state"]
    prep = meta["preprocessing"]
    numeric = (frame[list(FEATURES)].to_numpy(float) - np.array(prep["mean"])) / np.array(prep["std"])
    x = np.column_stack([numeric, *[(frame.spout_no.to_numpy() == c) for c in prep["categories"]]])
    h = x.copy()
    for i in range(settings["cross_layers"]):
        update = affine(h, state[f"cross.{i}.weight"], state[f"cross.{i}.bias"])
        if data["arm"] == "CROSS":
            h += x * update
        elif data["arm"] == "ADDITIVE":
            h += update
        else:
            raise ValueError("Unknown cross arm")
    deep = x
    for i in range(len(settings["deep_widths"])):
        deep = np.maximum(affine(deep, state[f"deep.{3*i}.weight"], state[f"deep.{3*i}.bias"]), 0.)
    prediction = affine(np.column_stack([h, deep]), state["head.weight"], state["head.bias"])[:, 0]
    result = prediction * meta["target_std"] + meta["target_mean"]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite independent prediction")
    return result


def verify_model(path, training, y, arm, settings, expected_sha256, trace=None):
    data = saved(path, expected_sha256)
    if data["arm"] != arm or data["settings"] != settings:
        raise ValueError("Model differs from frozen recipe")
    meta, prep = data["metadata"], data["metadata"]["preprocessing"]
    x = training[list(FEATURES)].to_numpy(float)
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-12] = 1.
    ids = digest(training.sample_id.astype(str).tolist())
    if (meta["fit_ids_digest"] != ids or prep["fit_ids_digest"] != ids
            or meta["fit_rows"] != len(training) or not np.array_equal(prep["mean"], mean)
            or not np.array_equal(prep["std"], std)
            or prep["categories"] != sorted(int(c) for c in training.spout_no.unique())):
        raise ValueError("Training-only preprocessing/row identity mismatch")
    y = np.asarray(y, float)
    if y.shape != (len(training),) or meta["target_mean"] != float(y.mean()) or meta["target_std"] != float(y.std()):
        raise ValueError("Target scale mismatch")
    width = len(FEATURES) + len(prep["categories"])
    shapes = {}
    dim = width
    for i, hidden in enumerate(settings["deep_widths"]):
        shapes[f"deep.{3*i}.weight"] = (hidden, dim)
        shapes[f"deep.{3*i}.bias"] = (hidden,)
        dim = hidden
    shapes["head.weight"], shapes["head.bias"] = (1, width + dim), (1,)
    for i in range(settings["cross_layers"]):
        shapes[f"cross.{i}.weight"], shapes[f"cross.{i}.bias"] = (width, width), (width,)
    if set(data["state"]) != set(shapes):
        raise ValueError("Saved state keys mismatch")
    for k, shape in shapes.items():
        a = np.asarray(data["state"][k], float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError("Saved parameter shape/value mismatch")
    state_hash = hashlib.sha256()
    for k in sorted(shapes):
        a = np.ascontiguousarray(data["state"][k], dtype=np.float64)
        state_hash.update(k.encode()); state_hash.update(str(a.shape).encode()); state_hash.update(a.tobytes())
    if state_hash.hexdigest() != meta["state_digest"]:
        raise ValueError("Independent parameter state digest mismatch")
    if meta["parameter_count"] != sum(np.prod(shape) for shape in shapes.values()):
        raise ValueError("Parameter count mismatch")
    history = meta["history"]
    if not history or [r["epoch"] for r in history] != list(range(1, meta["stopped_epoch"] + 1)):
        raise ValueError("Missing/reordered epoch trace")
    calibration = "calibration_standardized_mae" in history[0]
    expected_keys = {"epoch", "standardized_training_mse"}
    if calibration:
        expected_keys.add("calibration_standardized_mae")
    if any(set(r) != expected_keys for r in history):
        raise ValueError("Mixed selector/refit training traces")
    if not 1 <= meta["stopped_epoch"] <= settings["max_epochs"]:
        raise ValueError("Frozen training budget exceeded")
    if any(not np.isfinite(r["standardized_training_mse"]) or r["standardized_training_mse"] < 0 for r in history):
        raise ValueError("Invalid training trace")
    if "calibration_standardized_mae" in history[0]:
        best, selected, stale = float("inf"), 0, 0
        for r in history:
            if stale >= settings["patience"]:
                raise ValueError("Training continued after patience")
            value = r["calibration_standardized_mae"]
            if not np.isfinite(value) or value < 0:
                raise ValueError("Invalid calibration trace")
            if value < best - settings["min_delta_standardized_mae"]:
                best, selected, stale = value, r["epoch"], 0
            else:
                stale += 1
        if meta["selected_epoch"] != selected:
            raise ValueError("Selected epoch differs from trace")
        if meta["stopped_epoch"] != settings["max_epochs"] and stale != settings["patience"]:
            raise ValueError("Premature selector stop")
    elif meta["selected_epoch"] != meta["stopped_epoch"]:
        raise ValueError("Fresh refit epoch mismatch")
    if trace is not None and trace != meta:
        raise ValueError("External training trace differs")
    return meta
