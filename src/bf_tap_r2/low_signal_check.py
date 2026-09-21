"""Zero-fit independent verification of v0.3 evidence and frozen decisions."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from .audit import digest, write_json
from .cv import target_metrics
from .data import TARGETS, load_config, load_snapshot
from .weak_models import assess_candidate


def check(root, run):
    summary = json.loads((run / "summary.json").read_text())
    cfg = yaml.safe_load((run / "experiment.yaml").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    for name, sha in manifest["source_sha256"].items():
        assert digest(root / "src/bf_tap_r2" / name) == sha
    train, _ = load_snapshot(root, load_config(root / "configs/round2_v0_1/data.yaml"), "train")
    train = train.sort_values("sample_id").reset_index(drop=True)
    old = root / "local/runs/round2-v0.1/p2-cv-r1"
    fold_table = pd.read_csv(old / "fold_assignments.csv").set_index(["seed", "sample_id"])
    model_count = 0
    records = []
    for seed in cfg["seeds"]:
        folds = fold_table.loc[seed].loc[train.sample_id, "fold"].to_numpy()
        old_anchor = pd.read_csv(old / "oof" / f"{seed}-median_global.csv", float_precision="round_trip").set_index("sample_id").loc[train.sample_id]
        for target in TARGETS:
            anchor = np.empty(len(train))
            for fold in range(5):
                anchor[folds == fold] = np.median(train.loc[folds != fold, target])
            np.testing.assert_array_equal(anchor, old_anchor[f"pred_{target}"].to_numpy())
            assert target_metrics(train, target, anchor, folds) == summary["metrics"]["M0"][str(seed)][target]
            for route in ("M1_HD_MEDIAN", "K1_RBF_L1"):
                oof = pd.read_csv(run / "oof" / f"{seed}-{route}-{target}.csv", float_precision="round_trip")
                assert oof.sample_id.tolist() == train.sample_id.tolist()
                np.testing.assert_array_equal(oof[target], train[target])
                np.testing.assert_array_equal(oof.fold, folds)
                pred = oof[f"pred_{target}"].to_numpy()
                assert target_metrics(train, target, pred, folds) == summary["metrics"][route][str(seed)][target]
                for fold in range(5):
                    record = json.loads((run / "metrics" / f"{seed}-{route}-{target}-{fold}.json").read_text())
                    records.append(record)
                    valid = train.loc[folds == fold]
                    if route == "M1_HD_MEDIAN":
                        np.testing.assert_array_equal(pred[folds == fold], np.full(len(valid), record["hd_median"]))
                    else:
                        path = run / "models" / f"{seed}-{target}-{fold}.joblib"
                        assert digest(path) == record["model_sha256"]
                        model = joblib.load(path)
                        assert model.fit_report_["fit_status"] == 0 and not model.fit_report_["convergence_warning"]
                        np.testing.assert_array_equal(pred[folds == fold], model.predict(valid))
                        model_count += 1
    for target in TARGETS:
        for route in ("M1_HD_MEDIAN", "K1_RBF_L1"):
            decision = assess_candidate({s: summary["metrics"][route][str(s)][target] for s in cfg["seeds"]},
                                        {s: summary["metrics"]["M0"][str(s)][target] for s in cfg["seeds"]}, cfg["selection"])
            assert decision == summary["decisions"][target][route]
    starts = [json.loads(line) for line in (run / "fit_ledger.jsonl").read_text().splitlines() if json.loads(line)["event"] == "start"]
    assert len(starts) == model_count == summary["cv_fits"] == 20
    assert summary["total_regression_fits"] == 22
    result = {"G0": "PASS", "model_readbacks": model_count, "new_fits": 0,
              "old_m0_oof_exact_match": True, "oof_metrics_and_decisions_recomputed": True,
              "summary_sha256": digest(run / "summary.json"),
              "clipped_predictions": sum(r["clipped_count"] for r in records),
              "k1_iterations_range": [min(r["iterations"] for r in records if "iterations" in r), max(r["iterations"] for r in records if "iterations" in r)],
              "k1_support_range": [min(r["support_vectors"] for r in records if "support_vectors" in r), max(r["support_vectors"] for r in records if "support_vectors" in r)]}
    write_json(run / "independent_verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    check(Path.cwd(), args.run)
