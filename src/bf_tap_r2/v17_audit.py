"""Independent read-only audit of V17 OOF evidence and fixed-slot arithmetic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .v7_periodic import digest, file_hash, write_new
from .v17_run import context, summarize


def audit(root, directory, report):
    root = Path(root).resolve()
    directory = (root / directory).resolve()
    report = (root / report).resolve()
    if not directory.is_relative_to(root / "local/runs/round2-v17") or not report.is_relative_to(directory):
        raise ValueError("Private V17 run and new audit report required")
    spec_path = root / "configs/round2_v17/SPEC.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if file_hash(spec_path) != manifest["spec_sha256"] or file_hash(root / spec["source_plan"]) != manifest["plan_sha256"]:
        raise ValueError("V17 freeze changed")
    for name, sha in manifest["source_hashes"].items():
        if file_hash(root / "src/bf_tap_r2" / name) != sha:
            raise ValueError(f"V17 implementation changed: {name}")
    frame, folds, a35, b0, hashes = context(root, spec, spec["split_seeds"])
    if hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest() != manifest["data_digest"]:
        raise ValueError("Training data digest changed")
    if {str(seed): digest(fv.tolist()) for seed, fv in folds.items()} != manifest["fold_digests"]:
        raise ValueError("Split identity changed")
    if hashes != manifest["reference_hashes"]:
        raise ValueError("Reference identity changed")
    events = [json.loads(line) for line in (directory / "fit_ledger.jsonl").read_text().splitlines()]
    expected = {f"{recipe}-s{seed}-f{fold}" for recipe in spec["recipes"]
                for seed in spec["split_seeds"] for fold in range(5)}
    if len(events) != len(expected) or {e["key"] for e in events} != expected or any(e["event"] != "complete" for e in events):
        raise ValueError("Incomplete or duplicate V17 fit ledger")
    for event in events:
        key = event["key"]
        recipe_name, suffix = key.rsplit("-s", 1)
        seed_text, fold_text = suffix.split("-f")
        seed, fold = int(seed_text), int(fold_text)
        mask = folds[seed] == fold
        path = directory / f"{key}.npy"
        if file_hash(path) != event["prediction_sha256"]:
            raise ValueError(f"Prediction hash changed: {key}")
        prediction = np.load(path, allow_pickle=False)
        metadata = event["metadata"]
        if (prediction.shape != (int(mask.sum()),) or not np.isfinite(prediction).all()
                or metadata["fit_ids_digest"] != digest(frame.loc[~mask, "sample_id"].tolist())
                or metadata["fit_rows"] != int((~mask).sum())
                or metadata["optimizer_runs"] != 2
                or metadata["selected_epoch"] > 240):
            raise ValueError(f"Prediction or fit identity mismatch: {key}")
        if recipe_name.startswith("P_") and "dual_periodic" not in metadata:
            raise ValueError(f"Missing dual-branch diagnostics: {key}")
        if recipe_name == "J2" and metadata["gradient_diagnostics"]["batches"] <= 0:
            raise ValueError(f"Projection did not run: {key}")
    fresh = summarize(directory, frame, folds, a35, b0, spec)
    recorded = json.loads((directory / "summary.json").read_text())
    if fresh != recorded:
        raise ValueError("Fixed-slot score or finalist summary mismatch")
    result = {"status": "passed", "fits": len(events), "prediction_matrices": len(expected),
              "spec_sha256": manifest["spec_sha256"],
              "summary_sha256": file_hash(directory / "summary.json"),
              "selected_for_confirmation": fresh["selected_for_confirmation"],
              "packages": 0, "uploads": 0}
    write_new(report, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(Path.cwd(), args.run, args.report), indent=2))


if __name__ == "__main__":
    main()
