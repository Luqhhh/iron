"""Label-free, source-bound joint feature test; never reads sample/target tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist, pdist

from .data import FEATURES

SPEC_PATH = "configs/joint_support_audit/SPEC.json"
INPUTS = ("复赛_train/train_features.csv", "复赛_test/test_features.csv")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, obj: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)


def validate_spec(spec: dict) -> None:
    if (spec["identity"] != "XJN_JOINT_SUPPORT_AUDIT_V1"
            or (spec["train_features"], spec["test_features"]) != INPUTS
            or spec["id_prefixes"] != ["R2S2_TRAIN_", "R2S2_TEST_"]
            or spec["expected_rows"] != [2754, 322]
            or spec["permutations"] != 199 or spec["permutation_seed"] != 61002
            or spec["bandwidth_seed"] != 20261002
            or spec["bandwidth_sample_size"] != 256
            or spec["primary_alpha"] != 0.05
            or spec["maximum_runtime_seconds"] is not None
            or spec["automatic_importance_weighting"]):
        raise ValueError("Frozen diagnostic specification differs")


def read_features(path: Path, prefix: str, count: int) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sample_id", *FEATURES]:
            raise ValueError("Only ID and exact feature columns are allowed; no targets")
        rows = list(reader)
    ids = [row["sample_id"] for row in rows]
    if (len(ids) != count or len(set(ids)) != count
            or any(not value.startswith(prefix) for value in ids)):
        raise ValueError("Unexpected row count, duplicate or foreign IDs")
    x = np.array([[float(row[name]) for name in FEATURES] for row in rows], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite feature")
    return ids, x


def exclusive_units(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    if (train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]
            or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError("Invalid feature arrays")
    a = set(map(tuple, train.tolist()))
    b = set(map(tuple, test.tolist()))
    shared = a & b
    left, right = sorted(a - shared), sorted(b - shared)
    if len(left) < 2 or len(right) < 2:
        raise ValueError("Fewer than two exclusive units in a domain")
    x = np.array(left + right)
    domain = np.r_[np.zeros(len(left), dtype=int), np.ones(len(right), dtype=int)]
    return x, domain, {
        "train_rows": len(train), "test_rows": len(test),
        "train_unique_vectors": len(a), "test_unique_vectors": len(b),
        "train_duplicate_rows": len(train) - len(a), "test_duplicate_rows": len(test) - len(b),
        "shared_unique_vectors_excluded": len(shared),
        "train_exclusive_units": len(left), "test_exclusive_units": len(right),
        "statistical_unit": "exclusive unique feature vector, not all sample rows",
    }


def kernel_matrix(x: np.ndarray, sample_size: int = 256, seed: int = 20261002):
    if x.ndim != 2 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("Invalid pooled features")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    z = (x - mean) / scale
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(z), min(sample_size, len(z)), replace=False))
    distances = pdist(z[selected])
    positive = distances[distances > 0]
    if len(positive) == 0:
        raise ValueError("No positive pair distance")
    bandwidth = float(np.median(positive))
    kernel = cdist(z, z, metric="sqeuclidean")
    kernel *= -1. / (2. * bandwidth * bandwidth)
    np.exp(kernel, out=kernel)
    np.fill_diagonal(kernel, 0.)
    return kernel, {"bandwidth": bandwidth, "mean": mean.tolist(),
                    "scale": scale.tolist(), "bandwidth_selected_rows": selected.tolist()}


def validate_kernel(k: np.ndarray, labels: np.ndarray) -> None:
    n = len(labels)
    if (k.shape != (n, n) or not np.isfinite(k).all()
            or set(np.unique(labels)) != {0, 1}
            or min(np.bincount(labels.astype(int))) < 2
            or not np.allclose(k, k.T, rtol=0, atol=1e-14)
            or not np.all(np.diag(k) == 0)):
        raise ValueError("Invalid kernel or domain labels")


def direct_mmd(k: np.ndarray, labels: np.ndarray) -> float:
    a, b = labels == 0, labels == 1
    n, m = int(a.sum()), int(b.sum())
    return float(k[np.ix_(a, a)].sum() / (n*(n-1))
                 + k[np.ix_(b, b)].sum() / (m*(m-1))
                 - 2*k[np.ix_(a, b)].sum() / (n*m))


def quadratic_mmd(k: np.ndarray, labels: np.ndarray) -> float:
    u = labels.astype(float)
    m, n = int(u.sum()), len(u) - int(u.sum())
    row = k.sum(axis=1)
    bb = float(u @ (k @ u))
    ab = float(row @ u) - bb
    aa = float(row.sum()) - bb - 2*ab
    return aa/(n*(n-1)) + bb/(m*(m-1)) - 2*ab/(n*m)


def permutation_test(k: np.ndarray, labels: np.ndarray, count: int = 199, seed: int = 61002):
    validate_kernel(k, labels)
    if count < 1:
        raise ValueError("At least one permutation required")
    observed = direct_mmd(k, labels)
    if not math.isclose(observed, quadratic_mmd(k, labels), abs_tol=2e-14, rel_tol=0):
        raise ValueError("Independent statistic differs")
    rng = np.random.default_rng(seed)
    null = np.array([quadratic_mmd(k, rng.permutation(labels)) for _ in range(count)])
    p = float((1 + np.count_nonzero(null >= observed))/(count + 1))
    return {"mmd2_unbiased": observed, "permutation_p": p, "permutations": count,
            "permutation_seed": seed, "null_mean": float(null.mean()),
            "null_95_quantile_descriptive": float(np.quantile(null, .95)),
            "reject_exchangeable_exclusive_unit_null_at_005": p <= .05}, null


def local_output(root: Path, value: str) -> Path:
    out = (root/value).resolve()
    if not out.is_relative_to((root/"local").resolve()) or out == (root/"local").resolve():
        raise ValueError("Output must be a new subdirectory under local")
    return out


def inventory(root: Path) -> dict:
    paths = [SPEC_PATH, *INPUTS, "src/bf_tap_r2/joint_support_audit.py", "src/bf_tap_r2/data.py"]
    return {p: sha(root/p) for p in paths}


def verify(root: Path, out: Path) -> dict:
    manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    if inventory(root) != manifest["input_and_source_sha256"]:
        raise ValueError("Source or inputs changed")
    recorded = json.loads((out/"result.json").read_text(encoding="utf-8"))
    for name, digest in recorded["output_sha256"].items():
        if sha(out/name) != digest:
            raise ValueError("Output bytes changed")
    k = np.load(out/"kernel.npy", allow_pickle=False)
    labels = np.load(out/"domain.npy", allow_pickle=False)
    recalculated, null = permutation_test(k, labels)
    if recalculated != recorded["test"] or not np.array_equal(null, np.load(out/"null.npy", allow_pickle=False)):
        raise ValueError("Saved statistic or permutations differ")
    return {"status": "passed_saved_array_recalculation", "regression_fits": 0,
            "target_reads": 0, "manifest_sha256": sha(out/"manifest.json"),
            "result_sha256": sha(out/"result.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "verify"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    out = local_output(root, args.output)
    if args.action == "verify":
        receipt = verify(root, out)
        write_new(out/"independent-readback.json", receipt)
        print(json.dumps(receipt))
        return
    spec = json.loads((root/SPEC_PATH).read_text(encoding="utf-8"))
    validate_spec(spec)
    frozen = inventory(root)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/"manifest.json", {"identity": spec["identity"], "input_and_source_sha256": frozen,
              "python": sys.version, "numpy": np.__version__,
              "created_utc": datetime.now(timezone.utc).isoformat(), "target_reads": 0})
    try:
        a_ids, a = read_features(root/INPUTS[0], spec["id_prefixes"][0], spec["expected_rows"][0])
        b_ids, b = read_features(root/INPUTS[1], spec["id_prefixes"][1], spec["expected_rows"][1])
        if set(a_ids) & set(b_ids):
            raise ValueError("Domain ID overlap")
        x, labels, units = exclusive_units(a, b)
        k, prep = kernel_matrix(x)
        result, null = permutation_test(k, labels)
        for name, arr in (("kernel.npy", k), ("domain.npy", labels), ("null.npy", null)):
            with (out/name).open("xb") as handle:
                np.save(handle, arr, allow_pickle=False)
        if inventory(root) != frozen:
            raise ValueError("Sources or input bytes changed during run")
        write_new(out/"result.json", {"status": "complete_diagnostic_not_model_quality", "test": result,
                  "units": units, "preprocessing": prep, "regression_fits": 0, "target_reads": 0,
                  "packages": 0, "platform_forecast": None,
                  "output_sha256": {name: sha(out/name) for name in ("kernel.npy", "domain.npy", "null.npy")}})
        print(json.dumps({"status": "complete", "test": result, "units": units}))
    except Exception as exc:
        write_new(out/"FAILED.json", {"error_type": type(exc).__name__, "error": str(exc), "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
