"""User-authorized four isolated local/platform diagnostic releases.

Use each development worktree's unchanged trainer, not a mixed source tree.
All models, predictions, ledgers and release receipts stay under local/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/local_platform_diagnostic_release/RELEASE.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def config():
    cfg = json.loads(SPEC.read_text())
    if (cfg["authorization"]["user_request"] != "四项写桌面"
            or cfg["authorization"]["agent_uploads"] != 0
            or [(r["target"], r["arm"]) for r in cfg["candidates"]] != [
                ("tap_time_len", "EMA"), ("tap_iron", "D-LMIX"),
                ("tap_iron", "SAM"), ("tap_time_len", "SAM")]
            or cfg["replacement_weight"] != 0.5):
        raise ValueError("Frozen four-candidate authorization required")
    return cfg


def activate(row):
    source = ROOT / row["source_root"]
    sys.path.insert(0, str(source / "src"))
    import torch
    from bf_tap_r2.v49_run import check_runtime
    import yaml
    spec = yaml.safe_load((source / row["model_spec"]).read_text())
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError(f"Set {key}=1")
    torch.set_num_threads(1)
    check_runtime(spec)
    return source, spec


def check_frozen(out):
    manifest = json.loads((out / "manifest.json").read_text())
    for path, expected in manifest["files"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen identity changed: {path}")
    return manifest


def prepare(cfg, out):
    out.mkdir(parents=True, exist_ok=False)
    files = {str(SPEC): sha(SPEC), str(Path(__file__).resolve()): sha(__file__)}
    for name, artifact in cfg["artifacts"].items():
        path = ROOT / artifact["path"]
        if sha(path) != artifact["sha256"]:
            raise ValueError(f"Pinned artifact changed: {name}")
        files[str(path)] = sha(path)
    records = []
    for row in cfg["candidates"]:
        source = ROOT / row["source_root"]
        dev = ROOT / row["development"]
        manifest = json.loads((dev / "manifest.json").read_text())
        audit = json.loads((dev / "audit.json").read_text())
        if (audit["status"] != "passed" or audit["native_replays"] != 20
                or audit["manifest_sha256"] != sha(dev / "manifest.json")
                or audit["summary_sha256"] != sha(dev / "summary.json")
                or manifest["spec_sha256"] != sha(source / row["model_spec"])):
            raise ValueError("Audited complete development identity required")
        for relative in ["src/bf_tap_r2/component_regularization.py",
                         "src/bf_tap_r2/v12_joint.py", "src/bf_tap_r2/v3_6_networks.py",
                         "src/bf_tap_r2/v7_periodic.py", "src/bf_tap_r2/component_regularization_audit.py"]:
            if sha(source / relative) != manifest["source_hashes"][relative]:
                raise ValueError(f"Development dependency changed: {relative}")
        if row["arm"] == "D-LMIX":
            relative = "src/bf_tap_r2/component_augmentation.py"
            if sha(source / relative) != manifest["source_hashes"][relative]:
                raise ValueError("Frozen Mixup implementation changed")
        for relative, expected in manifest["data_hashes"].items():
            if sha(ROOT / relative) != expected:
                raise ValueError("Development data changed")
            files[str(ROOT / relative)] = expected
        # Anchor the entire historical evidence, including saved OOF states.
        for path in dev.rglob("*"):
            if path.is_file():
                files[str(path)] = sha(path)
        for path in (source / "src/bf_tap_r2").glob("*.py"):
            files[str(path)] = sha(path)
        for relative in [row["model_spec"], "uv.lock", "pyproject.toml"]:
            files[str(source / relative)] = sha(source / relative)
        summary = json.loads((dev / "summary.json").read_text())
        if any(v is not None for v in summary["selected_for_confirmation"].values()):
            raise ValueError("Historical no-finalist decision changed")
        record = next(r for r in summary["records"] if (r["target"], r["arm"]) == (row["target"], row["arm"]))
        records.append({"name": row["name"], "development": record})
    for stage in ("train", "test"):
        for kind in ("samples", "features"):
            path = ROOT / f"复赛_{stage}/{stage}_{kind}.csv"
            files[str(path)] = sha(path)
    template = ROOT / "复赛_test/result_template.csv"
    files[str(template)] = sha(template)
    write(out / "manifest.json", {"authorization": cfg["authorization"], "files": files,
          "candidates": cfg["candidates"], "development_records": records,
          "four_seed_promotion": False, "historical_decisions_unchanged": True,
          "parent_score": 96.3727, "current_best_at_freeze": 96.3749,
          "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    print(json.dumps({"prepared": str(out), "frozen_files": len(files)}), flush=True)


def model_type(row):
    if row["arm"] == "D-LMIX":
        from bf_tap_r2.component_augmentation import AugmentedRegressor
        return AugmentedRegressor
    from bf_tap_r2.component_regularization import ComponentRegressor
    return ComponentRegressor


def load_native(path):
    from bf_tap_r2.v12_release import IronView

    class NativeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("__main__", "IronView"):
                return IronView
            return super().find_class(module, name)

    with path.open("rb") as stream:
        return NativeUnpickler(stream).load()


def fit(cfg, out, row):
    import numpy as np
    from bf_tap_r2.component_regularization_run import RECIPE, outputs
    from bf_tap_r2.data import FEATURES
    from bf_tap_r2.v2_release import load_v2
    from bf_tap_r2.v5_library import load_v5_training_frame
    from bf_tap_r2.v7_periodic import digest
    from bf_tap_r2.v3_6_networks import NumericPreprocessor
    _, spec = activate(row)
    check_frozen(out)
    work = out / row["name"]
    work.mkdir(exist_ok=False)
    train = load_v5_training_frame(ROOT)
    test = load_v2(ROOT / "复赛_test", "test", 322)
    query = test[["sample_id", "spout_no", *FEATURES]].copy()
    kind = "iron" if row["target"] == "tap_iron" else "time"
    native_view = load_native(ROOT / cfg["artifacts"][f"native_{kind}_model"]["path"])
    native = getattr(native_view, "joint", native_view)
    if (native.settings != spec["training"][row["target"]] or native.recipe != RECIPE
            or native.metadata_["fit_ids_digest"] != digest(train.sample_id.tolist())
            or native.preprocessor_.metadata() != NumericPreprocessor(structure="raw_tabm").fit(train).metadata()):
        raise ValueError("Native full-data recipe/preprocessing/rows changed")
    old = native_view.predict(query)
    np.testing.assert_array_equal(old, np.load(ROOT / cfg["artifacts"][f"native_{kind}_prediction"]["path"], allow_pickle=False))
    query.to_pickle(work / "query.pkl")
    with (work / "native.npy").open("xb") as stream:
        np.save(stream, old)
    print(json.dumps({"event": "full_fit_started", "name": row["name"]}), flush=True)
    model = model_type(row)(RECIPE, spec["training"][row["target"]], row["arm"], spec["mechanisms"], work)
    y = train[outputs(row["target"])].to_numpy()
    model.fit(train, y)
    with (work / "warm.npy").open("xb") as stream:
        np.save(stream, model.predict(query))
    write(work / "fit.json", {"name": row["name"], "metadata": model.metadata_,
          "selection_sha256": sha(work / "selection.pt"), "refit_sha256": sha(work / "refit.pt"),
          "query_sha256": sha(work / "query.pkl"), "native_sha256": sha(work / "native.npy"),
          "warm_sha256": sha(work / "warm.npy"), "full_data_fits": 1, "optimizer_runs": 2,
          "new_CV_fits": 0, "training_seed": 42, "platform_score": None})
    check_frozen(out)
    print(json.dumps({"event": "full_fit_complete", "name": row["name"],
                      "selected_epoch": model.metadata_["selected_epoch"]}), flush=True)


def audit(cfg, out, row):
    import numpy as np
    from bf_tap_r2.component_regularization_audit import verify_saved
    from bf_tap_r2.component_regularization_run import outputs
    from bf_tap_r2.v3_4_bags import group_safe_inner_folds
    from bf_tap_r2.v5_library import load_v5_training_frame
    _, spec = activate(row)
    check_frozen(out)
    work = out / row["name"]
    receipt = json.loads((work / "fit.json").read_text())
    for name in ("selection", "refit", "query", "native", "warm"):
        suffix = ".pt" if name in ("selection", "refit") else ".pkl" if name == "query" else ".npy"
        if sha(work / (name + suffix)) != receipt[name + "_sha256"]:
            raise ValueError("Fitted artifact changed")
    train = load_v5_training_frame(ROOT)
    settings = spec["training"][row["target"]]
    y = train[outputs(row["target"])].to_numpy()
    mask = np.asarray(group_safe_inner_folds(train, seed=settings["inner_seed"])["fold"]) != 0
    # Match the trainer's native target-array slicing, avoiding layout changes.
    inner = train.loc[mask].reset_index(drop=True)
    extra = {"model_type": model_type(row)} if row["arm"] == "D-LMIX" else {}
    selector = verify_saved(work / "selection.pt", inner, y[mask], row["arm"], settings,
                            spec["mechanisms"], train.loc[~mask], **extra)
    full = verify_saved(work / "refit.pt", train, y, row["arm"], settings, spec["mechanisms"],
                        expected_epoch=selector.saved["trace"]["selected_epoch"], **extra)
    if selector.saved["trace"] != receipt["metadata"]["traces"]["selection"] or full.saved["trace"] != receipt["metadata"]["traces"]["refit"]:
        raise ValueError("Fit metadata differs from saved state")
    import pandas as pd
    np.testing.assert_array_equal(full.predict(pd.read_pickle(work / "query.pkl")), np.load(work / "warm.npy", allow_pickle=False))
    write(work / "audit.json", {"G0": "passed", "saved_models": 2,
          "selected_epoch": selector.saved["trace"]["selected_epoch"], "fit_sha256": sha(work / "fit.json"),
          "new_fits": 0, "train_only_preprocessing": True, "train_only_epoch_selection": True})
    check_frozen(out)
    print(json.dumps({"audited": row["name"]}), flush=True)


def cold(cfg, out, row):
    import numpy as np
    import pandas as pd
    from bf_tap_r2.submission import deny_training_reads
    from bf_tap_r2.data import TARGETS
    _, spec = activate(row)
    # Do not check training file hashes after installing the inference guard.
    sys.addaudithook(deny_training_reads)
    work = out / row["name"]
    query = pd.read_pickle(work / "query.pkl")
    if any(t in query for t in TARGETS):
        raise ValueError("Cold query contains labels")
    model = model_type(row).load(work / "refit.pt")
    prediction = model.predict(query)
    np.testing.assert_array_equal(prediction, np.load(work / "warm.npy", allow_pickle=False))
    variants = [model.predict(query.iloc[::-1])[::-1],
                np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0, len(query), 37)]),
                model.predict(query.iloc[:1])]
    differences = [float(np.max(np.abs(variants[0]-prediction))),
                   float(np.max(np.abs(variants[1]-prediction))),
                   float(np.max(np.abs(variants[2]-prediction[:1])))]
    if max(differences) > spec["preflight"]["cold_predict_atol"]:
        raise ValueError("Frozen cold order/chunk tolerance failed")
    kind = "iron" if row["target"] == "tap_iron" else "time"
    native = load_native(ROOT / cfg["artifacts"][f"native_{kind}_model"]["path"])
    np.testing.assert_array_equal(native.predict(query), np.load(work / "native.npy", allow_pickle=False))
    with (work / "cold.npy").open("xb") as stream:
        np.save(stream, prediction)
    write(work / "cold.json", {"G0": "passed", "training_reads_prohibited": True,
          "warm_cold_max_difference": 0.0, "native_replay_max_difference": 0.0,
          "reverse_chunk_singleton_max_differences": differences,
          "absolute_tolerance": spec["preflight"]["cold_predict_atol"],
          "prediction_sha256": sha(work / "cold.npy"), "new_fits": 0})
    print(json.dumps({"cold_passed": row["name"]}), flush=True)


def payload(parent, ids, target, old, new):
    import numpy as np
    from bf_tap_r2.submission import validate_result
    from bf_tap_r2.v5_package import payload_with_parent_other_column
    rows = validate_result(parent, ids)
    field = "pred_" + target
    other = "pred_tap_time_len" if target == "tap_iron" else "pred_tap_iron"
    base = np.array([float(r[field]) for r in rows])
    old, new = np.asarray(old), np.asarray(new)
    if old.shape != base.shape or new.shape != base.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError("Invalid component prediction shape/value")
    values = base + .5*(new-old)
    if np.min(values) < 0:
        # The development protocol had no clipping. Refuse a changed experiment.
        raise ValueError("Negative diagnostic replacement; no postprocessing change allowed")
    result = payload_with_parent_other_column(ids, target, values, [r[other] for r in rows])
    check = validate_result(result, ids)
    np.testing.assert_array_equal([float(r[field]) for r in check], values)
    if [r[other] for r in check] != [r[other] for r in rows]:
        raise ValueError("Unchanged target strings differ")
    return result


def build(cfg, out):
    import numpy as np
    import pandas as pd
    from bf_tap_r2.submission import ZIP_NAME, package, validate_result
    from bf_tap_r2.v12_release import parent_payload
    check_frozen(out)
    reports = []
    for row in cfg["candidates"]:
        work = out / row["name"]
        audit = json.loads((work / "audit.json").read_text())
        cold_report = json.loads((work / "cold.json").read_text())
        if audit["G0"] != "passed" or cold_report["G0"] != "passed" or audit["fit_sha256"] != sha(work / "fit.json") or cold_report["prediction_sha256"] != sha(work / "cold.npy"):
            raise ValueError("Audited cold release required")
        query = pd.read_pickle(work / "query.pkl")
        ids = query.sample_id.tolist()
        parent = parent_payload(ROOT / cfg["artifacts"]["parent"]["path"], cfg["artifacts"]["parent"]["sha256"], ids)
        new = np.load(work / "cold.npy", allow_pickle=False)[:, 0]
        old = np.load(work / "native.npy", allow_pickle=False)
        result = payload(parent, ids, row["target"], old, new)
        destination = work / "package"
        destination.mkdir(exist_ok=False)
        package(destination, result, ids)
        with zipfile.ZipFile(destination / ZIP_NAME) as archive:
            if archive.namelist() != ["result.csv"] or archive.testzip() is not None or archive.read("result.csv") != result:
                raise ValueError("Archive verification failed")
        rows = validate_result(result, ids)
        report = {"name": row["name"], "target": row["target"], "arm": row["arm"],
                  "G0": "passed", "G1": "user_requested_diagnostic_not_promoted",
                  "rows": len(rows), "unchanged_field_mismatches": 0, "readback_difference": 0.0,
                  "zip": str(destination / ZIP_NAME), "zip_sha256": sha(destination / ZIP_NAME),
                  "csv_sha256": sha(destination / "result.csv"), "selected_epoch": audit["selected_epoch"],
                  "local_mean_gain": row["local_mean_gain"], "platform_score": None,
                  "full_data_fits": 1, "optimizer_runs": 2, "new_CV_fits": 0, "agent_uploads": 0}
        write(work / "release.json", report)
        reports.append(report)
    write(out / "release-summary.json", {"packages": reports, "full_data_fits": 4,
          "optimizer_runs": 8, "new_CV_fits": 0, "agent_uploads": 0, "parent_score": 96.3727})
    check_frozen(out)
    print(json.dumps({"built": [r["name"] for r in reports]}), flush=True)


def verify_packages(cfg, out, selected):
    """Separate fresh process recomputes CSV from saved cold states and parent."""
    import csv
    import io
    import numpy as np
    import pandas as pd
    from bf_tap_r2.submission import deny_training_reads
    check_frozen(out)
    sys.addaudithook(deny_training_reads)
    report = json.loads((out / "release-summary.json").read_text())
    with zipfile.ZipFile(ROOT / cfg["artifacts"]["parent"]["path"]) as archive:
        original = list(csv.DictReader(io.StringIO(archive.read("result.csv").decode())))
    released_by_name = {r["name"]: r for r in report["packages"]}
    for row in [selected]:
        released = released_by_name[row["name"]]
        work = out / row["name"]
        if sha(released["zip"]) != released["zip_sha256"]:
            raise ValueError("Release archive hash changed")
        query = pd.read_pickle(work / "query.pkl")
        model = model_type(row).load(work / "refit.pt")
        member = model.predict(query)[:, 0]
        kind = "iron" if row["target"] == "tap_iron" else "time"
        native = load_native(ROOT / cfg["artifacts"][f"native_{kind}_model"]["path"])
        old = native.predict(query)
        field = "pred_" + row["target"]
        other = "pred_tap_time_len" if row["target"] == "tap_iron" else "pred_tap_iron"
        expected = np.asarray([float(r[field]) for r in original]) - old/2 + member/2
        with zipfile.ZipFile(released["zip"]) as archive:
            if archive.namelist() != ["result.csv"] or archive.testzip() is not None:
                raise ValueError("Invalid diagnostic archive")
            actual = list(csv.DictReader(io.StringIO(archive.read("result.csv").decode())))
        if len(actual) != 322 or [r["sample_id"] for r in actual] != query.sample_id.tolist() or len({r["sample_id"] for r in actual}) != 322:
            raise ValueError("Template row identity changed")
        if [r[other] for r in actual] != [r[other] for r in original]:
            raise ValueError("Other target changed")
        values = np.array([float(r[field]) for r in actual])
        np.testing.assert_allclose(values, expected, atol=1e-10, rtol=0)
        if not np.isfinite(values).all() or values.min() < 0:
            raise ValueError("Invalid submission values")
    write(out / selected["name"] / "independent-package-audit.json", {"G0": "passed", "packages": 1,
          "saved_state_replayed": True, "arithmetic_atol": 1e-10, "training_reads_prohibited": True,
          "unchanged_field_mismatches": 0, "new_fits": 0, "release_summary_sha256": sha(out / "release-summary.json")})
    print(f"Independent package audit passed: {selected['name']}", flush=True)


def deliver(cfg, out):
    # This mode requires filesystem escalation for the user's desktop path.
    from bf_tap_r2.submission import ZIP_NAME
    check_frozen(out)
    audit = json.loads((out / "independent-package-audit.json").read_text())
    if audit["G0"] != "passed" or audit["release_summary_sha256"] != sha(out / "release-summary.json"):
        raise ValueError("Independent package audit required before desktop delivery")
    reports = json.loads((out / "release-summary.json").read_text())["packages"]
    dest = Path(cfg["desktop"])
    dest.mkdir(parents=True, exist_ok=False)
    receipts = []
    explanation = ["本地筛选可信度：四项隔离诊断实验", "",
        "四个包统一父包 V32_TIME_A60V7_50，已回传参照分数 96.3727。",
        "当前最佳 DE3=96.3749；先比较相对 V32 的方法增益，再判断是否超过当前最佳。",
        "收益指分数点。只替换指定组件：parent + 0.5*(新组件-原组件)。另一目标字符串不变。",
        "不是四 seed 晋级候选；此次用户明确授权用于检验本地漏筛，历史门槛和失败决定保留。",
        "请上传各子目录中的 ZIP，分别按完整候选名回传分数。助手未上传。", ""]
    for i, report in enumerate(reports, 1):
        directory = dest / f"{i:02d}_{report['name']}"
        directory.mkdir(exist_ok=False)
        target = directory / ZIP_NAME
        with Path(report["zip"]).open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        if sha(target) != report["zip_sha256"]:
            raise ValueError("Desktop copy hash mismatch")
        with zipfile.ZipFile(target) as archive:
            if archive.namelist() != ["result.csv"] or archive.testzip() is not None:
                raise ValueError("Desktop archive readback failed")
            with zipfile.ZipFile(report["zip"]) as original:
                if archive.read("result.csv") != original.read("result.csv"):
                    raise ValueError("Desktop payload differs")
        explanation.append(f"{i}. {report['name']}：本地均值 {report['local_mean_gain']:+.9f}；ZIP SHA256 {report['zip_sha256']}")
        receipts.append({"name": report["name"], "desktop_zip": str(target), "zip_sha256": sha(target), "readback_passed": True})
    with (dest / "README.txt").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(explanation)+"\n")
    write(out / "desktop-delivery.json", {"desktop": str(dest), "packages": receipts, "agent_uploads": 0})
    print(json.dumps({"delivered": str(dest), "packages": receipts}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "fit", "audit", "cold", "build", "verify", "deliver"])
    parser.add_argument("--candidate", type=int, choices=range(1, 5))
    args = parser.parse_args()
    cfg = config()
    out = ROOT / cfg["output"]
    if not out.resolve().is_relative_to(ROOT / "local/runs"):
        raise ValueError("Private release root required")
    row = cfg["candidates"][args.candidate-1] if args.candidate else None
    if row:
        activate(row)
    else:
        sys.path.insert(0, str(ROOT / "src"))
    if args.mode == "prepare":
        prepare(cfg, out)
    elif args.mode in ("fit", "audit", "cold"):
        if row is None:
            parser.error("--candidate required")
        try:
            {"fit": fit, "audit": audit, "cold": cold}[args.mode](cfg, out, row)
        except BaseException as exc:
            path = out / row["name"] / f"FAILED-{args.mode}.json"
            if path.parent.exists() and not path.exists():
                write(path, {"error": repr(exc), "evidence_preserved": True})
            raise
    elif args.mode == "build":
        build(cfg, out)
    elif args.mode == "verify":
        if row:
            verify_packages(cfg, out, row)
        else:
            verify_dispatch(cfg, out)
    else:
        deliver(cfg, out)


def verify_dispatch(cfg, out):
    # Each fresh process loads only its historical worktree's exact trainer.
    check_frozen(out)
    audits = []
    for i, row in enumerate(cfg["candidates"], 1):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "verify", "--candidate", str(i)], check=True)
        path = out / row["name"] / "independent-package-audit.json"
        audit = json.loads(path.read_text())
        if audit["G0"] != "passed" or audit["release_summary_sha256"] != sha(out / "release-summary.json"):
            raise ValueError("Independent child audit failed")
        audits.append({"name": row["name"], "audit_sha256": sha(path)})
    write(out / "independent-package-audit.json", {"G0": "passed", "packages": 4,
          "audits": audits, "release_summary_sha256": sha(out / "release-summary.json")})
    check_frozen(out)


if __name__ == "__main__":
    main()
