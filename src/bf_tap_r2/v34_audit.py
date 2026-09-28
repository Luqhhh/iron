"""Independent disk readback, calibration arithmetic and cold V34 inference."""
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
from .v34_crps import CRPSRegressor
from .v34_run import SPEC, partitions, unit_id, verified_unit, verify_hashes


def run(root, directory):
    root = Path(root).resolve()
    out = (root / directory).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v34"):
        raise ValueError("Private V34 audit required")
    manifest = json.loads((out / "manifest.json").read_text())
    verify_hashes(root, manifest["source_hashes"])
    verify_hashes(root, manifest["data_hashes"])
    spec = yaml.safe_load((root / SPEC).read_text())
    summary = json.loads((out / "summary.json").read_text())
    expected_identity = dict(manifest)
    expected_identity.pop("identity")
    if digest(expected_identity) != manifest["identity"]:
        raise ValueError("Manifest identity mismatch")
    if manifest["spec_sha256"] != file_hash(root / SPEC):
        raise ValueError("Spec identity mismatch")
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
                        ("model.pkl", training, query, pred, meta["refit"]),
                        ("calibration_model.pkl", fitting, calibration.drop(columns=list(TARGETS)), cp, meta["calibration"])]:
                        model = CRPSRegressor.load(out / key / name)
                        if m["fit_ids_digest"] != digest(fit_frame.sample_id.tolist()):
                            raise ValueError("Model fit row digest mismatch")
                        if not np.allclose(model.preprocessor_.means_, fit_frame[list(FEATURES)].mean().to_numpy(),
                                           atol=1e-10, rtol=1e-12):
                            raise ValueError("Preprocessor training subset mismatch")
                        if model.settings != spec["training"] or model.recipe != recipe:
                            raise ValueError("Model settings differ from spec")
                        if not np.isclose(model.mean_, fit_frame[target].mean(), rtol=0, atol=1e-10):
                            raise ValueError("Target mean differs from training subset")
                        if not np.isclose(model.std_, fit_frame[target].std(ddof=0), rtol=0, atol=1e-10):
                            raise ValueError("Target scale differs from training subset")
                        if name == "model.pkl" and m["selected_epoch"] != meta["calibration"]["selected_epoch"]:
                            raise ValueError("Refit did not use selected epoch")
                        if not np.allclose(model.preprocessor_.stds_, fit_frame[list(FEATURES)].std(ddof=0), atol=1e-10, rtol=1e-12):
                            raise ValueError("Training-only feature scale mismatch")
                        expected_vocab = {int(k):i for i,k in enumerate(sorted(set(fit_frame.spout_no.astype(int))),start=1)}
                        if model.preprocessor_.spout_to_index_ != expected_vocab:
                            raise ValueError("Training-only vocabulary mismatch")
                        if len(model.trees_) != m["selected_epoch"] or len(model.steps_) != m["selected_epoch"]:
                            raise ValueError("Persisted iteration count mismatch")
                        if recipe == "CRPS_FIXED" and any(len(v) != 1 for v in model.trees_):
                            raise ValueError("Fixed-scale control has extra trees")
                        if recipe == "CRPS_SCALE" and any(len(v) != 2 for v in model.trees_):
                            raise ValueError("Adaptive-scale arm lacks a parameter tree")
                        if any(b["training_crps"] > a["training_crps"]+1e-12 for a,b in zip(m["history"],m["history"][1:])):
                            raise ValueError("Training score increased")
                        observed = model.predict(query_frame)
                        reverse = model.predict(query_frame.iloc[::-1])[::-1]
                        chunks = np.concatenate([model.predict(query_frame.iloc[i:i+37])
                                                 for i in range(0, len(query_frame), 37)])
                        difference = max(float(np.max(np.abs(v - expected))) for v in [observed, reverse, chunks])
                        if difference > 1e-8:
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
    selected = {}
    for target in TARGETS:
        eligible = []
        for row in summary["records"]:
            if row["target"] != target: continue
            gains = []
            for seed in manifest["seeds"]:
                y = frame[target].to_numpy()
                gains.append(float(50*(np.abs(y-base[seed][target]).sum()-np.abs(y-columns[seed,target,row["recipe"]]).sum())/np.abs(y).sum()))
            if row["recipe"] in spec["promotion"]["eligible_recipes"] and min(gains) > 0:
                eligible.append((float(np.mean(gains)),row["recipe"]))
        # Mirror the frozen cost order, but recompute gains directly above.
        eligible.sort(key=lambda r:(-r[0],spec["tie_preference_by_target"][target].index(r[1])))
        selected[target] = eligible[0][1] if eligible else None
    if selected != summary["selected_for_confirmation"]:
        raise ValueError("Independent selection differs")
    if manifest["development_summary_sha256"] is not None:
        from .v5_resolution import paired_summary
        dev = root / manifest["development_directory"]
        if file_hash(dev / "summary.json") != manifest["development_summary_sha256"]:
            raise ValueError("Development summary identity mismatch")
        previous = json.loads((dev / "summary.json").read_text())
        for decision in summary["four_seed_decisions"]:
            match = lambda r: (r["target"],r["recipe"]) == (decision["target"],decision["recipe"])
            old = next(r for r in previous["records"] if match(r))
            current = next(r for r in summary["records"] if match(r))
            expected = {**old["seed_results"], **current["seed_results"]}
            if decision["seed_results"] != expected or len(expected) != 4:
                raise ValueError("Final decision does not use all four saved seeds")
            expected_dev_score = float(np.mean([r["candidate_score"] for r in old["seed_results"].values()]))
            if abs(decision["development_mean_score"] - expected_dev_score) > 1e-10:
                raise ValueError("Development gate score mismatch")
            paired = paired_summary([v["gain"] for v in decision["seed_results"].values()])
            if any(abs(paired[k]-decision["paired_seed_summary"][k]) > 1e-10 for k in ("mean","lcb95")):
                raise ValueError("Paired summary mismatch")
            failed = []
            if paired["positive"] != 4: failed.append("not_all_four_seeds_positive")
            if paired["lcb95"] <= 0: failed.append("nonpositive_seed_lcb95")
            if decision["development_mean_score"] < spec["promotion"]["local_working_gate"]: failed.append("local_working_gate")
            if failed != decision["failed_conditions"] or decision["promoted"] != (not failed):
                raise ValueError("Final gate decision mismatch")
    report = {"status": "passed", "selection_verified": True, "units": units, "cold_models": models,
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
