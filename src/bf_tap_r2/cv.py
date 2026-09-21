"""Append-only P1/P2 run: 50 main fits, at most 20 recheck fits; no uploads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .audit import digest, write_json
from .data import FEATURES, TARGETS, load_config, load_snapshot, resolve_file
from .metrics import wmape
from .models import SnapshotRegressor
from .splits import make_folds


def append_event(path, event):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **event}, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def target_metrics(frame, target, prediction, folds):
    return {"wmape": wmape(frame[target], prediction),
            "by_fold": {str(f): wmape(frame.loc[folds == f, target], prediction[folds == f]) for f in sorted(set(folds))},
            "by_spout": {str(s): wmape(frame.loc[frame.spout_no == s, target], prediction[frame.spout_no == s]) for s in sorted(frame.spout_no.unique())}}


def baseline_predictions(frame, target, folds, by_spout):
    result = np.full(len(frame), np.nan)
    for fold in sorted(set(folds)):
        train, valid = frame.loc[folds != fold], frame.loc[folds == fold]
        median = float(train[target].median())
        result[folds == fold] = (valid.spout_no.map(train.groupby("spout_no")[target].median()).fillna(median)
                                 if by_spout else median)
    return result


def combined_metrics(score):
    if not all(t in score for t in TARGETS):
        return {"J": None}
    return {"J": sum(score[t]["wmape"] for t in TARGETS) / 2,
            **{f"J_{group}": {key: sum(score[t][group][key] for t in TARGETS) / 2
                              for key in score[TARGETS[0]][group]} for group in ("by_fold", "by_spout")}}


def verify_audit(root, audit_dir):
    audit = json.loads((audit_dir / "data_audit.json").read_text())
    manifest_path = audit_dir / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if audit["data_gate"] != "pass" or digest(manifest_path) != audit["manifest_sha256"]:
        raise ValueError("P0 data gate or manifest identity failed")
    for name, record in manifest["files"].items():
        if digest(resolve_file(root, name)) != record["sha256"]:
            raise ValueError(f"Data changed since P0: {name}")
    # Nonzero screening pairs need reviewed group assignments, not blind random CV.
    if any(audit["features"]["near_duplicates"]["counts"].values()):
        raise ValueError("Review near-duplicate groups before running CV")
    return manifest


def run(root: Path, output: Path, audit_dir: Path):
    if not output.resolve().is_relative_to(root.resolve() / "local/runs/round2-v0.1"):
        raise ValueError("Run artifacts must remain private")
    manifest = verify_audit(root, audit_dir)
    cfg_dir = root / "configs/round2_v0_1"
    cfg = load_config(cfg_dir / "data.yaml")
    model_cfg = yaml.safe_load((cfg_dir / "models.yaml").read_text())
    validation = yaml.safe_load((cfg_dir / "validation.yaml").read_text())
    if model_cfg["fit_budget"] != {"main": 50, "recheck": 20, "full": 4, "total": 74}:
        raise ValueError("Unexpected fit budget")
    frame, _ = load_snapshot(root, cfg, "train")
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    if not np.isfinite(frame[[*FEATURES, "spout_no", *TARGETS]]).all().all():
        raise ValueError("Training requires complete finite snapshots")
    output.mkdir(parents=True, exist_ok=False)
    ledger = output / "fit_ledger.jsonl"
    fits = 0
    try:
        for part in ("oof", "metrics", "models", "configs"):
            (output / part).mkdir()
        for path in cfg_dir.glob("*.yaml"):
            with (output / "configs" / path.name).open("xb") as handle:
                handle.write(path.read_bytes())
        write_json(output / "data_manifest.json", manifest)
        identity = {"code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    "python": platform.python_version(),
                    "dependencies": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scikit-learn", "catboost", "scipy", "joblib")},
                    "uv_lock_sha256": digest(root / "uv.lock"),
                    "source_sha256": {str(p.relative_to(root)): digest(p) for p in sorted((root / "src/bf_tap_r2").glob("*.py"))},
                    "config_sha256": {p.name: digest(p) for p in sorted(cfg_dir.glob("*.yaml"))},
                    "data_manifest_sha256": digest(output / "data_manifest.json"),
                    "input_columns": [*FEATURES, "spout_no"], "no_preliminary_inputs": True,
                    "selection": "each target top two main OOF WMAPE; tie broken by route name",
                    "model_fit_budget": 70, "outer_validation_early_stopping": False}
        write_json(output / "run_manifest.json", identity)
        folds_by_seed = {}
        assignments = []
        for protocol in ("main", "recheck"):
            spec = validation[protocol]
            assignment = make_folds(frame, spec["seed"], spec["n_splits"])
            assignments.append(assignment)
            folds_by_seed[spec["seed"]] = assignment.set_index("sample_id").loc[frame.sample_id, "fold"].to_numpy()
        pd.concat(assignments).to_csv(output / "fold_assignments.csv", index=False, mode="x")
        write_json(output / "fold_identity.json", {"sha256": digest(output / "fold_assignments.csv"),
                   "method": "StratifiedKFold; exact duplicate vectors grouped if present",
                   "same_folds_for_both_targets": True,
                   "counts": pd.concat(assignments).groupby(["seed", "fold", "spout_no"]).size().rename("rows").reset_index().to_dict("records")})
        all_metrics = {}
        for seed, folds in folds_by_seed.items():
            for name, by_spout in (("median_global", False), ("median_spout", True)):
                predictions = frame[["sample_id", "spout_no", *TARGETS]].copy()
                predictions["fold"] = folds
                score = {}
                for target in TARGETS:
                    pred = baseline_predictions(frame, target, folds, by_spout)
                    predictions[f"pred_{target}"] = pred
                    score[target] = target_metrics(frame, target, pred, folds)
                    append_event(ledger, {"kind": "baseline_complete", "seed": seed, "route": name,
                                         "target": target, "fold_fits": 5})
                score.update(combined_metrics(score))
                predictions.to_csv(output / "oof" / f"{seed}-{name}.csv", index=False, mode="x")
                write_json(output / "metrics" / f"{seed}-{name}.json", score)
                all_metrics[f"{seed}-{name}"] = score
        selected = None
        main_scores = {t: {} for t in TARGETS}
        recheck_scores = {t: {} for t in TARGETS}
        for phase in ("main", "recheck"):
            seed = validation[phase]["seed"]
            folds = folds_by_seed[seed]
            candidates = {t: list(model_cfg["models"]) if phase == "main" else selected[t] for t in TARGETS}
            for route in model_cfg["models"]:
                score = {}
                for target in TARGETS:
                    if route not in candidates[target]:
                        continue
                    prediction = np.full(len(frame), np.nan)
                    for fold in range(validation[phase]["n_splits"]):
                        if fits >= 70:
                            raise ValueError("Fit budget exhausted")
                        tag = f"{seed}-{route}-{target}-fold{fold}"
                        train_mask, valid_mask = folds != fold, folds == fold
                        fits += 1
                        event = {"phase": phase, "seed": seed, "route": route, "target": target, "fold": fold, "fit_number": fits}
                        append_event(ledger, {**event, "kind": "regressor_fit_start", "training_rows": int(train_mask.sum()),
                                             "preprocessor_fit_count": int(route in ("L1", "L2", "S1"))})
                        started = time.monotonic()
                        with threadpool_limits(limits=1):
                            model = SnapshotRegressor(route, model_cfg).fit(frame.loc[train_mask], frame.loc[train_mask, target])
                            pred = model.predict(frame.loc[valid_mask])
                        prediction[valid_mask] = pred
                        model_path = output / "models" / f"{tag}.joblib"
                        joblib.dump(model, model_path)
                        fold_oof = frame.loc[valid_mask, ["sample_id", "spout_no", target]].copy()
                        fold_oof[f"pred_{target}"] = pred
                        fold_oof.to_csv(output / "oof" / f"{tag}.csv", index=False, mode="x")
                        append_event(ledger, {**event, "kind": "regressor_fit_complete", "seconds": time.monotonic()-started,
                                             "model_sha256": digest(model_path), "target_scale": model.target_scale_})
                        print(json.dumps({**event, "wmape": wmape(frame.loc[valid_mask, target], pred)}, ensure_ascii=False), flush=True)
                    score[target] = target_metrics(frame, target, prediction, folds)
                    oof = frame[["sample_id", "spout_no", target]].copy()
                    oof["fold"] = folds
                    oof[f"pred_{target}"] = prediction
                    oof.to_csv(output / "oof" / f"{seed}-{route}-{target}.csv", index=False, mode="x")
                    (main_scores if phase == "main" else recheck_scores)[target][route] = score[target]["wmape"]
                if score:
                    score.update(combined_metrics(score))
                    write_json(output / "metrics" / f"{seed}-{route}.json", score)
                    all_metrics[f"{seed}-{route}"] = score
            if phase == "main":
                selected = {t: sorted(main_scores[t], key=lambda route: (main_scores[t][route], route))[:2] for t in TARGETS}
                write_json(output / "recheck_selection.json", {"selected": selected, "main_scores": main_scores,
                           "rule": identity["selection"], "frozen_before_recheck": True})
        summary = {"status": "complete", "main_scores": main_scores, "recheck_scores": recheck_scores,
                   "recheck_selection": selected, "metrics": all_metrics,
                   "regressor_fits": fits, "baseline_fold_fits": 40,
                   "preprocessor_fits": sum(json.loads(line).get("preprocessor_fit_count", 0) for line in ledger.read_text().splitlines()),
                   "platform_uploads": 0, "full_training_fits": 0}
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(output / "FAILED.json", {"type": type(exc).__name__, "error": str(exc), "started_regressor_fits": fits})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path("local/runs/round2-v0.1/p0-audit-r2"))
    args = parser.parse_args()
    run(Path.cwd(), args.output, args.audit)


if __name__ == "__main__":
    main()
