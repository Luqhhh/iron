"""Independent disk readback, calibration arithmetic and cold V27 inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .data import FEATURES, TARGETS
from .v5_library import fold_vector, load_v5_training_frame
from .v5_spec import load_v5_spec
from .v7_periodic import digest, file_hash, write_new
from .v27_deep_kernel import DeepKernelRegressor
from .v27_run import SPEC, partitions, unit_id, verified_unit, verify_hashes


def run(root, directory):
    root = Path(root).resolve()
    out = (root / directory).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v27"):
        raise ValueError("Private V27 audit required")
    manifest = json.loads((out / "manifest.json").read_text())
    verify_hashes(root, manifest["source_hashes"])
    verify_hashes(root, manifest["data_hashes"])
    spec = yaml.safe_load((root / SPEC).read_text())
    summary = json.loads((out / "summary.json").read_text())
    frame = load_v5_training_frame(root)
    max_diff, models, units = 0., 0, 0
    base, columns, raw_columns = {}, {}, {}
    for seed in manifest["seeds"]:
        fv = fold_vector(root, frame, seed, load_v5_spec(root))
        if digest(fv.tolist()) != manifest["fold_hashes"][str(seed)]:
            raise ValueError("Outer fold identity changed")
        base[seed] = {t: np.full(len(frame), np.nan) for t in TARGETS}
        for target, recipes in manifest["candidates"].items():
            for recipe in recipes:
                columns[seed, target, recipe] = np.full(len(frame), np.nan)
                raw_columns[seed, target, recipe] = np.full(len(frame), np.nan)
        for fold in range(5):
            mask = fv == fold
            training, query, fitting, calibration = partitions(frame, fv, fold, spec)
            refs = {}
            for role, train_part, query_part in [("outer", training, query),
                                                ("calibration", fitting, calibration)]:
                key = f"reference-{role}-s{seed}-f{fold}"
                verified_unit(out / key, unit_id(manifest, key))
                units += 1
                meta = json.loads((out / key / "metadata.json").read_text())
                if meta["fit_ids_digest"] != digest(train_part.sample_id.tolist()):
                    raise ValueError("Reference fit identity mismatch")
                with np.load(out / key / "predictions.npz") as saved:
                    refs[role] = {t: saved[t].copy() for t in TARGETS}
                    if not np.array_equal(saved["query_ids"], query_part.sample_id.to_numpy(dtype=str)):
                        raise ValueError("Reference query identity mismatch")
                    expected_i = .5 * saved["v36_iron"] + .5 * saved["v12_iron"]
                    expected_t = .325 * saved["v36_time"] + .175 * saved["n_time"] + .5 * saved["v7_time"]
                    if not (np.array_equal(expected_i, saved["tap_iron"])
                            and np.array_equal(expected_t, saved["tap_time_len"])):
                        raise ValueError("B0 endpoint arithmetic mismatch")
            for t in TARGETS:
                base[seed][t][mask] = refs["outer"][t]
            for target, recipes in manifest["candidates"].items():
                for recipe in recipes:
                    key = f"{target}-{recipe}-s{seed}-f{fold}"
                    verified_unit(out / key, unit_id(manifest, key))
                    units += 1
                    meta = json.loads((out / key / "metadata.json").read_text())
                    with np.load(out / key / "predictions.npz") as saved:
                        pred, cp = saved["prediction"].copy(), saved["calibration_prediction"].copy()
                        if not np.array_equal(saved["query_ids"], query.sample_id.to_numpy(dtype=str)):
                            raise ValueError("Candidate query identity mismatch")
                    for name, fit_frame, query_frame, expected, m in [
                        ("model.pt", training, query, pred, meta["refit"]),
                        ("calibration_model.pt", fitting, calibration.drop(columns=list(TARGETS)), cp, meta["calibration"])]:
                        model = DeepKernelRegressor.load(out / key / name)
                        if m["fit_ids_digest"] != digest(fit_frame.sample_id.tolist()):
                            raise ValueError("Model fit row digest mismatch")
                        if not np.allclose(model.preprocessor_.means_, fit_frame[list(FEATURES)].mean().to_numpy(),
                                           atol=1e-10, rtol=1e-12):
                            raise ValueError("Preprocessor training subset mismatch")
                        observed = model.predict(query_frame)
                        reverse = model.predict(query_frame.iloc[::-1])[::-1]
                        chunks = np.concatenate([model.predict(query_frame.iloc[i:i+37])
                                                 for i in range(0, len(query_frame), 37)])
                        difference = max(float(np.max(np.abs(v - expected))) for v in [observed, reverse, chunks])
                        if difference > 1e-9:
                            raise ValueError(f"Cold/order/batch inference mismatch: {key} {difference}")
                        max_diff = max(max_diff, difference)
                        models += 1
                    # Independent numpy arithmetic, never call the runner's selector.
                    losses = [np.abs(calibration[target].to_numpy() - ((1-a)*refs["calibration"][target]+a*cp)).mean()
                              for a in spec["calibration"]["blend_grid"]]
                    alpha = spec["calibration"]["blend_grid"][int(np.argmin(losses))]
                    if alpha != meta["weight"]:
                        raise ValueError("Calibration-only weight selection mismatch")
                    columns[seed, target, recipe][mask] = (1-alpha)*refs["outer"][target] + alpha*pred
                    raw_columns[seed, target, recipe][mask] = pred
    score_differences = []
    for row in summary["records"]:
        target, recipe = row["target"], row["recipe"]
        y = frame[target].to_numpy()
        for seed in manifest["seeds"]:
            candidate = columns[seed, target, recipe]
            if not np.isfinite(candidate).all(): raise ValueError("Incomplete candidate coverage")
            actual_gain = 50 * (np.abs(y-base[seed][target]).sum() - np.abs(y-candidate).sum()) / np.abs(y).sum()
            difference = abs(actual_gain-row["seed_results"][str(seed)]["gain"])
            if difference > 1e-10: raise ValueError("Pooled score mismatch")
            score_differences.append(difference)
    report = {"status": "passed", "units": units, "cold_models": models,
              "max_inference_difference": max_diff, "maximum_score_difference": max(score_differences, default=0),
              "summary_sha256": file_hash(out / "summary.json"), "manifest_sha256": file_hash(out / "manifest.json"),
              "reference_audit_scope": "fit/query identities and saved endpoint arithmetic; no extra baseline refits",
              "new_fits": 0, "packages": 0, "agent_uploads": 0}
    write_new(out / "audit.json", report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    run(Path.cwd(), args.directory)


if __name__ == "__main__":
    main()
