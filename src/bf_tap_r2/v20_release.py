#!/usr/bin/env python3
"""V20: release the V17 P-LL time expert on the verified combined B0 parent.

One replay fit, one four-seed design check against the frozen B0 reference, one
full-data fit, a cold-process audit and one single-target replacement package.
Never uploads; refuses to write outside ``local/runs/round2-v20``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .data import TARGETS
from .submission import ZIP_NAME, package
from .v2_release import load_v2
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import package_score, paired_summary, wmape
from .v5_spec import load_v5_spec
from .v7_periodic import file_hash, write_new
from .v7_release import blended_payload, parent_payload, save_and_cold, verify_payload
from .v17_confirm import confirmed_references
from .v17_models import V17TimeRegressor, fit_fold

ROOT_DEFAULT = Path("/home/lux1/iron")
SPEC_DEFAULT = Path("configs/round2_v20/SPEC.yaml")
OUTPUT_DEFAULT = Path("local/runs/round2-v20/release-r1")
TARGET = "tap_time_len"
OTHER = "tap_iron"


def read_predictions(directory: Path, key: str, frame, folds, seeds) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for seed in seeds:
        vector = np.full(len(frame), np.nan)
        for fold in range(5):
            path = directory / f"{key}-s{seed}-f{fold}.npy"
            values = np.load(path, allow_pickle=False).ravel()
            mask = folds[seed] == fold
            if values.shape != (int(mask.sum()),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid V17 prediction: {path}")
            vector[mask] = values
        if not np.isfinite(vector).all():
            raise ValueError(f"Incomplete V17 predictions for seed {seed}")
        out[seed] = vector
    return out


def design_check(root: Path, spec20: Mapping[str, Any], spec17: Mapping[str, Any],
                 frame, folds, b0) -> dict[str, Any]:
    alpha = float(spec20["blend"]["alpha"])
    dev_seeds = [int(seed) for seed in spec17["split_seeds"]]
    conf_seeds = [int(seed) for seed in spec17["confirmation_seeds"]]
    dev_dir = root / "local/runs/round2-v17/development-r1"
    conf_dir = root / "local/runs/round2-v17/confirmation-r1"
    dev = read_predictions(dev_dir, "P_LL_T", frame, folds, dev_seeds)
    conf = read_predictions(conf_dir, "P_LL_T", frame, folds, conf_seeds)
    members = {**dev, **conf}
    y_time = frame[TARGET].to_numpy()
    rows: dict[str, Any] = {}
    for seed in dev_seeds + conf_seeds:
        blended = (1 - alpha) * b0[seed][TARGET] + alpha * members[seed]
        base = package_score(wmape(frame[OTHER].to_numpy(), b0[seed][OTHER]),
                             wmape(y_time, b0[seed][TARGET]))
        candidate = package_score(wmape(frame[OTHER].to_numpy(), b0[seed][OTHER]),
                                  wmape(y_time, blended))
        cells = []
        for fold in range(5):
            mask = folds[seed] == fold
            cells.append({"fold": fold, "rows": int(mask.sum()),
                          "gain": 50.0 * (wmape(y_time[mask], b0[seed][TARGET][mask])
                                          - wmape(y_time[mask], blended[mask]))})
        rows[str(seed)] = {"b0_score": base, "candidate_score": candidate,
                           "gain": candidate - base, "folds_descriptive": cells}
    gains = [rows[str(seed)]["gain"] for seed in dev_seeds + conf_seeds]
    paired = paired_summary(gains)
    dev_score = float(np.mean([rows[str(seed)]["candidate_score"] for seed in dev_seeds]))
    gate = float(spec20["gates"]["local_working_gate"])
    return {"alpha": alpha, "seed_results": rows, "paired_seed_summary": paired,
            "development_mean_candidate_score": dev_score,
            "local_working_gate": gate, "local_working_gate_met": bool(dev_score >= gate),
            "all_seeds_positive": bool(all(gain > 0 for gain in gains)),
            "paired_lcb95_positive": bool(paired["lcb95"] > 0),
            "promoted": bool(all(gain > 0 for gain in gains) and paired["lcb95"] > 0
                             and dev_score >= gate)}


def replay_check(root: Path, spec17: Mapping[str, Any], frame, folds, settings,
                 seed: int, fold: int) -> dict[str, Any]:
    recipe = spec17["recipes"]["P_LL_T"]
    prediction, metadata = fit_fold(frame, folds[seed], "P_LL_T", recipe, settings, fold)
    stored = np.load(root / f"local/runs/round2-v17/development-r1/P_LL_T-s{seed}-f{fold}.npy",
                     allow_pickle=False).ravel()
    if prediction.shape != stored.shape:
        raise ValueError("Replay shape mismatch")
    difference = float(np.max(np.abs(prediction - stored)))
    if not np.array_equal(prediction, stored):
        raise ValueError(f"Replay is not bit-identical: max |diff| = {difference}")
    return {"seed": seed, "fold": fold, "rows": int(prediction.shape[0]),
            "maximum_absolute_difference": difference, "bit_identical": True,
            "selected_epoch": metadata.get("selection_stopped_epoch"),
            "parameter_count": metadata.get("parameter_count")}


def run(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = SPEC_DEFAULT,
        output: Path | str = OUTPUT_DEFAULT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec_path = (root / spec_path).resolve()
    if not spec_path.is_relative_to(root / "configs/round2_v20"):
        raise ValueError("V20 specification required")
    spec = yaml.safe_load(spec_path.read_text())
    spec17 = yaml.safe_load((root / "configs/round2_v17/SPEC.yaml").read_text())
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1")
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v20"):
        raise ValueError("V20 release output must stay private under local/runs/round2-v20")
    if out.exists():
        raise FileExistsError("V20 output already exists; refusing overwrite")

    versions = {name: importlib.metadata.version(name) for name in spec17["runtime_versions"]}
    frame = load_v5_training_frame(root)
    seeds = [int(s) for s in spec17["split_seeds"]] + [int(s) for s in spec17["confirmation_seeds"]]
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root)) for seed in seeds}
    a35, b0, reference_hashes = confirmed_references(root, frame, folds, spec17)
    time_settings = yaml.safe_load((root / spec17["training_source_time"]).read_text())["training"]

    out.mkdir(parents=True)
    design = design_check(root, spec, spec17, frame, folds, b0)
    write_new(out / "design-check.json", design)
    if not design["promoted"]:
        write_new(out / "STOPPED.json", {"reason": "four-seed design gate not met", "design": design})
        raise ValueError("V20 four-seed design gate not met; no full-data fit performed")
    print(json.dumps({"design_check": "passed", "development_mean_candidate_score":
                      design["development_mean_candidate_score"],
                      "paired_seed_summary": design["paired_seed_summary"]}, indent=2), flush=True)

    replay = replay_check(root, spec17, frame, folds, time_settings, 42, 0)
    write_new(out / "replay.json", replay)
    print(json.dumps({"replay": "passed", "selected_epoch": replay["selected_epoch"]}), flush=True)

    test = load_v2(root / "复赛_test", "test", 322)
    ids = test.sample_id.tolist()
    parent_path = root / spec["parent"]["zip"]
    parent = parent_payload(parent_path, spec["parent"]["sha256"], ids)
    alpha = float(spec["blend"]["alpha"])

    full = V17TimeRegressor(spec17["recipes"]["P_LL_T"], time_settings)
    full.fit(frame.reset_index(drop=True), frame[TARGET].to_numpy())
    member = full.predict(test)
    cold = save_and_cold(full, test, member, out / "full", 1e-6, parent, alpha)
    payload, clipping = blended_payload(parent, ids, member, alpha)
    if payload != (out / "full" / "cold-result.csv").read_bytes():
        raise ValueError("Cold-process submission CSV bytes differ")

    package_dir = out / "V20_B0_PLLT_A325"
    package_dir.mkdir(exist_ok=False)
    package(package_dir, payload, ids)
    import zipfile

    with zipfile.ZipFile(package_dir / ZIP_NAME) as archive:
        if archive.namelist() != ["result.csv"] or archive.testzip() is not None:
            raise ValueError("Invalid V20 archive")
        archived = archive.read("result.csv")
    if archived != payload:
        raise ValueError("V20 archive content differs")
    readback = verify_payload(archived, parent, ids, member, alpha)

    verification = {
        "G0": "PASS",
        "candidate": "V20_B0_PLLT_A325",
        "target": TARGET,
        "alpha": alpha,
        "parent": {"zip": str(parent_path), "sha256": spec["parent"]["sha256"]},
        "readback": readback,
        "clipping": clipping,
        "replay": replay,
        "cold": cold,
        "full_fit_metadata": full.metadata_,
        "design_check": {"seed_results": design["seed_results"],
                         "paired_seed_summary": design["paired_seed_summary"],
                         "development_mean_candidate_score": design["development_mean_candidate_score"]},
        "zip_sha256": file_hash(package_dir / ZIP_NAME),
        "result_sha256": file_hash(package_dir / "result.csv"),
        "new_fits": {"replay": 1, "full_data": 1},
        "desktop_writes": 0,
        "agent_uploads": 0,
    }
    write_new(out / "verification.json", verification)
    write_new(out / "manifest.json", {
        "spec_sha256": file_hash(spec_path),
        "data_digest": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        "versions": versions,
        "reference_hashes": reference_hashes,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    })
    print(json.dumps({"status": "released", "zip": str(package_dir / ZIP_NAME),
                      "sha256": verification["zip_sha256"],
                      "development_mean_candidate_score": design["development_mean_candidate_score"]},
                     indent=2))
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--spec", type=Path, default=SPEC_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args.root, args.spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
