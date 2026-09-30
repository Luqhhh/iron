"""Explicitly requested DE3 iron exploration release; original gates unchanged."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from .component_regularization import ComponentRegressor, replacement
from .component_regularization_audit import verify_saved
from .component_regularization_run import RECIPE
from .data import FEATURES, TARGETS
from .submission import ZIP_NAME, package, validate_result
from .v12_release import IronView, parent_payload, save_and_cold
from .v2_release import load_v2
from .v3_4_bags import group_safe_inner_folds
from .v3_6_networks import NumericPreprocessor
from .v5_library import load_v5_training_frame
from .v5_package import payload_with_parent_other_column
from .v7_periodic import digest, file_hash, write_new
from .v49_run import check_runtime

SPEC = "configs/de3_user_release/RELEASE.yaml"


class NativeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("__main__", "IronView"):
            return IronView
        return super().find_class(module, name)


class DE3Iron:
    def __init__(self, members):
        if len(members) != 3:
            raise ValueError("Exactly three frozen members required")
        self.members = members

    def predict(self, frame):
        if any(t in frame for t in TARGETS):
            raise ValueError("Inference query contains labels")
        return np.stack([m.predict(frame)[:, 0] for m in self.members]).mean(axis=0)


def release_payload(parent, ids, old_member, new_member):
    rows = validate_result(parent, ids)
    current = np.asarray([float(r["pred_tap_iron"]) for r in rows])
    # Replace the existing .5 component; do not blend the entire package again.
    iron = np.maximum(replacement(current, old_member, new_member, .5), 0.)
    payload = payload_with_parent_other_column(
        ids, "tap_iron", iron, [r["pred_tap_time_len"] for r in rows])
    actual = validate_result(payload, ids)
    np.testing.assert_array_equal([float(r["pred_tap_iron"]) for r in actual], iron)
    if [r["pred_tap_time_len"] for r in actual] != [r["pred_tap_time_len"] for r in rows]:
        raise ValueError("Time strings changed")
    return payload


def checked_hash(root, path, expected):
    resolved = root / path
    if file_hash(resolved) != expected:
        raise ValueError(f"Frozen artifact hash changed: {path}")
    return resolved


def run(root):
    root = Path(root).resolve()
    cfg = yaml.safe_load((root / SPEC).read_text())
    if (cfg["candidate"] != "DE3_IRON_USER_REQUESTED"
            or cfg["training_seeds"] != [42, 104729, 130363]
            or cfg["authorization"]["user_requested_release"] is not True
            or cfg["authorization"]["uploads"] is not False):
        raise ValueError("Explicit frozen DE3 release authorization required")
    out = root / cfg["output"]
    if not out.resolve().is_relative_to(root / "local/runs"):
        raise ValueError("Private release output required")
    out.mkdir(parents=True, exist_ok=False)
    try:
        paths = {name: checked_hash(root, row["path"], row["sha256"])
                 for name, row in cfg["artifacts"].items()}
        development = json.loads(paths["development_summary"].read_text())
        audit = json.loads(paths["development_audit"].read_text())
        manifest = json.loads(paths["development_manifest"].read_text())
        if (audit["status"] != "passed" or audit["native_replays"] != 20
                or audit["summary_sha256"] != file_hash(paths["development_summary"])
                or audit["manifest_sha256"] != file_hash(paths["development_manifest"])):
            raise ValueError("Audited complete DE3 evidence required")
        for name, expected in manifest["data_hashes"].items():
            checked_hash(root, name, expected)
        # The full-fit estimator and training code must match the audited DE3 run.
        for name in ("component_regularization.py", "v12_joint.py", "v3_6_networks.py"):
            key = f"src/bf_tap_r2/{name}"
            checked_hash(root, key, manifest["source_hashes"][key])
        native_spec = yaml.safe_load((root / "configs/strong_component_regularization/SPEC.yaml").read_text())
        if check_runtime(native_spec) != manifest["versions"]:
            raise ValueError("Frozen runtime changed")
        record = next(r for r in development["records"] if r["target"] == "tap_iron")
        if development["selected_for_confirmation"]["tap_iron"] is not None:
            raise ValueError("Historical no-finalist decision changed")
        frozen_files = {str(p): file_hash(p) for p in paths.values()}
        frozen_files.update({str(root / name): file_hash(root / name)
            for name in manifest["data_hashes"]})
        frozen_files.update({str(p): file_hash(p) for p in (root / "src/bf_tap_r2").glob("*.py")})
        frozen_files.update({str(root / p): file_hash(root / p) for p in
            (SPEC, "uv.lock", "configs/strong_component_regularization/SPEC.yaml",
             "复赛_test/test_samples.csv", "复赛_test/test_features.csv", "复赛_test/result_template.csv")})
        write_new(out / "manifest.json", {
            "candidate": cfg["candidate"], "authorization": cfg["authorization"],
            "files": frozen_files, "training_seeds": cfg["training_seeds"],
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "historical_no_finalist_preserved": True, "development": record,
            "four_seed_promotion": False, "platform_score": None})
        train = load_v5_training_frame(root)
        test = load_v2(root / "复赛_test", "test", 322)
        query = test[["sample_id", "spout_no", *FEATURES]].copy()
        ids = query.sample_id.tolist()
        parent = parent_payload(paths["parent"], cfg["artifacts"]["parent"]["sha256"], ids)
        with paths["native_full_model"].open("rb") as stream:
            native = NativeUnpickler(stream).load().joint
        settings = native_spec["training"]["tap_iron"]
        if native.settings != settings or native.recipe != RECIPE:
            raise ValueError("Native full-data recipe changed")
        if native.metadata_["fit_ids_digest"] != digest(train.sample_id.tolist()):
            raise ValueError("Native full-data training rows changed")
        if native.preprocessor_.metadata() != NumericPreprocessor(structure="raw_tabm").fit(train).metadata():
            raise ValueError("Native preprocessing changed")
        old = native.predict(query)[:, 0]
        np.testing.assert_array_equal(old, np.load(paths["native_full_prediction"], allow_pickle=False))
        np.testing.assert_array_equal(old, np.load(paths["native_full_warm"], allow_pickle=False))
        models = [native]
        mask = np.asarray(group_safe_inner_folds(train, seed=settings["inner_seed"])["fold"]) != 0
        y = train[list(TARGETS)].to_numpy()
        inner = train.loc[mask].reset_index(drop=True)
        selection_info = []
        for seed in cfg["training_seeds"][1:]:
            directory = out / f"training-seed-{seed}"
            directory.mkdir(exist_ok=False)
            updated = dict(settings, random_seed=seed)
            print(json.dumps({"event": "full_fit_started", "training_seed": seed}), flush=True)
            model = ComponentRegressor(RECIPE, updated, "BASE", {}, directory).fit(train, y)
            selector = verify_saved(directory / "selection.pt", inner, y[mask], "BASE", updated, {}, train.loc[~mask])
            saved = verify_saved(directory / "refit.pt", train, y, "BASE", updated, {},
                                 expected_epoch=selector.saved["trace"]["selected_epoch"])
            np.testing.assert_array_equal(model.predict(query), saved.predict(query))
            info = {"seed": seed, "metadata": model.metadata_,
                    "selection_sha256": file_hash(directory / "selection.pt"),
                    "refit_sha256": file_hash(directory / "refit.pt")}
            write_new(directory / "verification.json", info)
            selection_info.append(info)
            models.append(model)
            print(json.dumps({"event": "full_fit_complete", "training_seed": seed,
                              "selected_epoch": model.metadata_["selected_epoch"]}), flush=True)
        ensemble = DE3Iron(models)
        new = ensemble.predict(query)
        cold = save_and_cold(ensemble, query, new, out / "ensemble", 1e-6)
        payload = release_payload(parent, ids, old, new)
        cold_payload = release_payload(parent, ids, old, np.load(out / "ensemble/cold.npy"))
        if payload != cold_payload:
            raise ValueError("Cold submission bytes changed")
        destination = out / cfg["candidate"]
        destination.mkdir(exist_ok=False)
        package(destination, payload, ids)
        for path, expected in frozen_files.items():
            if file_hash(Path(path)) != expected:
                raise ValueError(f"Frozen source changed during release: {path}")
        report = {"G0": "passed", "G1": "user_requested_exploration_not_four_seed_promoted",
                  "candidate": cfg["candidate"], "rows": 322, "time_string_mismatches": 0,
                  "component_replacement_readback_difference": 0.0,
                  "cold": cold, "new_full_data_fits": 2, "new_optimizer_runs": 4,
                  "reused_native_full_models": 1, "new_CV_fits": 0,
                  "models": selection_info, "local_mean_gain": record["paired"]["mean"],
                  "package": str(destination / ZIP_NAME), "zip_sha256": file_hash(destination / ZIP_NAME),
                  "parent_zip_sha256": cfg["artifacts"]["parent"]["sha256"],
                  "platform_score": None, "agent_uploads": 0, "desktop_writes": 0}
        write_new(out / "verification.json", report)
        print(json.dumps({k:v for k,v in report.items() if k not in ("models", "cold")}), flush=True)
        return report
    except BaseException as exc:
        write_new(out / "FAILED.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    # Stable module name for the ensemble's inference pickle.
    from bf_tap_r2.de3_user_release import run as release_run
    release_run(args.root)
