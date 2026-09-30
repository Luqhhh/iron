"""Frozen, zero-fit sparse EMA time experiments; artifacts remain private."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from local_platform_diagnostic_release import sha, write, load_native

SPEC = ROOT / "configs/ema_time_followup/SPEC.json"


def replacement_time(parent, old, ema, q):
    import numpy as np
    parent, old, ema = [np.asarray(x, dtype=float) for x in (parent, old, ema)]
    if (parent.ndim != 1 or old.shape != parent.shape or ema.shape != parent.shape
            or not all(np.isfinite(x).all() for x in (parent, old, ema))
            or q not in (0., .25, .5, .75, 1.)):
        raise ValueError("Frozen sparse weights and finite aligned vectors required")
    result = parent + q * (ema - old)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Invalid replacement: no clipping or postprocessing allowed")
    return result


def time_payload(parent, ids, old, ema, q):
    import numpy as np
    from bf_tap_r2.submission import validate_result
    from bf_tap_r2.v5_package import payload_with_parent_other_column
    rows = validate_result(parent, ids)
    values = replacement_time([float(r["pred_tap_time_len"]) for r in rows], old, ema, q)
    result = payload_with_parent_other_column(ids, "tap_time_len", values,
                                             [r["pred_tap_iron"] for r in rows])
    actual = validate_result(result, ids)
    np.testing.assert_array_equal([float(r["pred_tap_time_len"]) for r in actual], values)
    if [r["pred_tap_iron"] for r in actual] != [r["pred_tap_iron"] for r in rows]:
        raise ValueError("Iron strings changed")
    return result


def combined_payload(iron_parent, time_parent, ids):
    from bf_tap_r2.submission import SUBMISSION_COLUMNS, validate_result
    iron = validate_result(iron_parent, ids)
    time = validate_result(time_parent, ids)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(SUBMISSION_COLUMNS)
    writer.writerows([sid, i["pred_tap_iron"], t["pred_tap_time_len"]]
                    for sid, i, t in zip(ids, iron, time))
    payload = stream.getvalue().encode()
    validate_result(payload, ids)
    return payload


def de3_full_column(selected, original):
    """Exported DE3 members already contain the shipped .5 replacement."""
    import numpy as np
    np.testing.assert_allclose(selected.current_prediction.to_numpy(), original, atol=1e-10, rtol=0)
    np.testing.assert_allclose(selected.seed_42_prediction.to_numpy(), original, atol=1e-10, rtol=0)
    columns = selected[["seed_42_prediction", "seed_104729_prediction", "seed_130363_prediction"]].to_numpy()
    if not np.isfinite(columns).all():
        raise ValueError("Nonfinite DE3 full deployment columns")
    return columns.mean(axis=1)


def config():
    cfg = json.loads(SPEC.read_text())
    if ([(r["name"], r["q"]) for r in cfg["candidates"]] != [
            ("EMA_TIME_Q25", .25), ("EMA_TIME_Q75", .75),
            ("EMA_TIME_Q100", 1.), ("DE3_IRON_EMA_TIME_Q50", .5)]
            or cfg["budget"] != {"new_fits": 0, "new_split_seeds": 0,
                                  "agent_uploads": 0, "desktop_writes": 0}
            or cfg["reference"]["candidate"] != "LOC_DIAG_EMA_TIME"):
        raise ValueError("Frozen experiment pool/authorization changed")
    return cfg


def check(out):
    manifest = json.loads((out / "manifest.json").read_text())
    for path, expected in manifest["files"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen identity changed: {path}")
    return manifest


def read_zip(cfg, key, ids):
    from bf_tap_r2.v12_release import parent_payload
    artifact = cfg["artifacts"][key]
    return parent_payload(ROOT / artifact["path"], artifact["sha256"], ids)


def prepare(cfg, out):
    out.mkdir(parents=True, exist_ok=False)
    files = {str(SPEC): sha(SPEC), str(Path(__file__).resolve()): sha(__file__)}
    for item in cfg["artifacts"].values():
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"Pinned identity changed: {path}")
        files[str(path)] = item["sha256"]
    # Preserve the original complete audit and all source/data identities.
    prior = json.loads((ROOT / cfg["artifacts"]["diagnostic_manifest"]["path"]).read_text())
    files.update(prior["files"])
    dev = ROOT / cfg["development"]
    for path in dev.rglob("*"):
        if path.is_file():
            files[str(path)] = sha(path)
    de3 = ROOT / cfg["de3_development"]
    for name in ("manifest.json", "summary.json", "audit.json"):
        files[str(de3 / name)] = sha(de3 / name)
    for path in files:
        if sha(path) != files[path]:
            raise ValueError(f"Historical dependency changed: {path}")
    for name in ("audit", "independent_audit", "cold_receipt"):
        if json.loads((ROOT / cfg["artifacts"][name]["path"]).read_text())["G0"] != "passed":
            raise ValueError("Previously audited EMA release required")
    write(out / "manifest.json", {"files": files, "spec": cfg,
          "G1": "user_authorized_sparse_platform_experiment_not_formal_promotion",
          "new_fits": 0, "platform_scores": None, "historical_gates_unchanged": True})
    print(json.dumps({"prepared": str(out), "frozen_files": len(files)}), flush=True)


def descriptive_local(cfg, out):
    import numpy as np
    import pandas as pd
    from bf_tap_r2.component_regularization_run import collect
    from bf_tap_r2.v5_library import load_v5_training_frame, fold_vector
    from bf_tap_r2.v5_spec import load_v5_spec
    check(out)
    frame = load_v5_training_frame(ROOT)
    folds = {s: fold_vector(ROOT, frame, s, load_v5_spec(ROOT)) for s in (42, 3407)}
    base, members = collect(ROOT / cfg["development"], frame, folds, {"tap_time_len": ["EMA"]})
    de3 = pd.read_csv(ROOT / cfg["artifacts"]["de3_oof"]["path"])
    records = []
    for row in cfg["candidates"]:
        seeds = {}
        for seed, fv in folds.items():
            y = frame.tap_time_len.to_numpy()
            original = base[seed]["tap_time_len"]
            old = base[seed]["v7_time"]
            new = members[seed, "tap_time_len", "EMA"]
            incumbent = replacement_time(original, old, new, .5)
            candidate = replacement_time(original, old, new, row["q"])
            iron_original = base[seed]["tap_iron"]
            iron_candidate = iron_original
            if row["kind"] == "column_control":
                selected = de3.loc[(de3.target == "tap_iron") & (de3.split_seed == seed)]
                if selected.sample_id.duplicated().any() or set(selected.sample_id) != set(frame.sample_id):
                    raise ValueError("DE3 OOF rows/seed mismatch")
                selected = selected.set_index("sample_id").loc[frame.sample_id]
                np.testing.assert_array_equal(selected.fold.to_numpy(), fv)
                np.testing.assert_array_equal(selected.actual.to_numpy(), frame.tap_iron.to_numpy())
                # These exported columns already use parent+.5*(component-old).
                # Mean of these FULL deployment columns equals the shipped DE3.
                iron_candidate = de3_full_column(selected, iron_original)
            iron_y = frame.tap_iron.to_numpy()
            if not np.isfinite(iron_candidate).all() or (iron_candidate < 0).any():
                raise ValueError("Invalid DE3 local replacement")
            def score(time, iron, mask):
                return float(100 - 50 * (np.abs(y[mask] - time[mask]).sum()/np.abs(y[mask]).sum()
                    + np.abs(iron_y[mask] - iron[mask]).sum()/np.abs(iron_y[mask]).sum()))
            all_rows = np.ones(len(frame), dtype=bool)
            candidate_score = score(candidate, iron_candidate, all_rows)
            seeds[str(seed)] = {"rows": len(frame), "folds": 5,
                "reference_score": score(incumbent, iron_original, all_rows),
                "candidate_score": candidate_score,
                "gain_vs_ema_incumbent": candidate_score-score(incumbent, iron_original, all_rows),
                "gain_vs_v32": candidate_score-score(original, iron_original, all_rows),
                "fold_gains_vs_ema": [score(candidate, iron_candidate, fv == f)-score(incumbent, iron_original, fv == f) for f in range(5)]}
        records.append({"name": row["name"], "seeds": seeds,
            "mean_gain_vs_ema": float(np.mean([s["gain_vs_ema_incumbent"] for s in seeds.values()])),
            "mean_gain_vs_v32": float(np.mean([s["gain_vs_v32"] for s in seeds.values()]))})
    # Pin the original EMA reproduction before any weight interpretation.
    gains = []
    for seed in folds:
        y = frame.tap_time_len.to_numpy()
        original = base[seed]["tap_time_len"]
        incumbent = replacement_time(original, base[seed]["v7_time"], members[seed, "tap_time_len", "EMA"], .5)
        gains.append(float(50*(np.abs(y-original).sum()-np.abs(y-incumbent).sum())/np.abs(y).sum()))
    np.testing.assert_allclose(np.mean(gains), cfg["original_local_mean_gain"], atol=1e-11, rtol=0)
    write(out / "local-evaluation.json", {"records": records, "reference": cfg["reference"],
        "original_ema_gain_reproduced": float(np.mean(gains)), "new_fits": 0,
        "cross_seed_prediction_averaging": False, "selection_use": "descriptive_only_no_local_veto",
        "complete_rows_per_seed": len(frame), "split_seeds": [42, 3407],
        "formal_four_seed_promotion": False})
    check(out)
    print(json.dumps({"local": [{"name": r["name"], "gain_vs_ema": r["mean_gain_vs_ema"]} for r in records]}), flush=True)


def inference(cfg):
    import numpy as np
    import pandas as pd
    import torch
    from bf_tap_r2.component_regularization import ComponentRegressor
    from bf_tap_r2.v49_run import check_runtime
    from bf_tap_r2.submission import deny_training_reads
    import yaml
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1")
    torch.set_num_threads(1)
    check_runtime(yaml.safe_load((ROOT / "configs/strong_component_regularization/SPEC.yaml").read_text()))
    sys.addaudithook(deny_training_reads)
    artifact = lambda key: ROOT / cfg["artifacts"][key]["path"]
    query = pd.read_pickle(artifact("query"))
    if any(t in query for t in ("tap_iron", "tap_time_len")):
        raise ValueError("Cold query contains targets")
    model = ComponentRegressor.load(artifact("ema_checkpoint"))
    ema = model.predict(query)
    np.testing.assert_array_equal(ema, np.load(artifact("ema_prediction"), allow_pickle=False))
    differences = [float(np.max(np.abs(model.predict(query.iloc[::-1])[::-1]-ema))),
        float(np.max(np.abs(np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0, len(query), 37)])-ema)))]
    if max(differences) > cfg["cold_predict_atol"]:
        raise ValueError("Frozen order/chunk tolerance exceeded")
    old = load_native(artifact("v7_model")).predict(query)
    np.testing.assert_array_equal(old, np.load(artifact("v7_prediction"), allow_pickle=False))
    return query.sample_id.tolist(), old, ema[:, 0], differences


def expected_payloads(cfg, ids, old, ema):
    parent = read_zip(cfg, "v32", ids)
    incumbent = read_zip(cfg, "incumbent", ids)
    de3 = read_zip(cfg, "de3", ids)
    if time_payload(parent, ids, old, ema, .5) != incumbent:
        raise ValueError("Scored EMA q=.5 CSV fails exact reconstruction")
    from bf_tap_r2.submission import validate_result
    p, d = validate_result(parent, ids), validate_result(de3, ids)
    if [r["pred_tap_time_len"] for r in p] != [r["pred_tap_time_len"] for r in d]:
        raise ValueError("DE3 parent time identity differs")
    return {r["name"]: combined_payload(de3, incumbent, ids) if r["kind"] == "column_control"
            else time_payload(parent, ids, old, ema, r["q"]) for r in cfg["candidates"]}


def build(cfg, out):
    from bf_tap_r2.submission import ZIP_NAME, package
    check(out)
    local = json.loads((out / "local-evaluation.json").read_text())
    ids, old, ema, differences = inference(cfg)
    payloads = expected_payloads(cfg, ids, old, ema)
    reports = []
    for row, desc in zip(cfg["candidates"], local["records"]):
        dest = out / row["name"]
        dest.mkdir(exist_ok=False)
        package(dest, payloads[row["name"]], ids)
        reports.append({**row, "zip": str(dest / ZIP_NAME), "zip_sha256": sha(dest / ZIP_NAME),
            "csv_sha256": sha(dest / "result.csv"), "mean_local_gain_vs_ema": desc["mean_gain_vs_ema"],
            "G0": "passed_cold_build_independent_verification_pending",
            "G1": "platform_experiment_not_formal_promotion", "platform_score": None})
    write(out / "release-summary.json", {"packages": reports, "reference": cfg["reference"],
        "q50_exact_csv_reconstruction": True, "cold_full_batch_difference": 0.,
        "cold_order_chunk_differences": differences, "new_fits": 0, "agent_uploads": 0,
        "desktop_writes": 0, "conditional_combination_arithmetic": 96.3917,
        "combination_arithmetic_is_not_measured_score": True})
    print(json.dumps({"built": [r["name"] for r in reports]}), flush=True)


def verify(cfg, out):
    from bf_tap_r2.submission import validate_result
    check(out)
    summary = json.loads((out / "release-summary.json").read_text())
    ids, old, ema, differences = inference(cfg)
    expected = expected_payloads(cfg, ids, old, ema)
    for report in summary["packages"]:
        path = Path(report["zip"])
        if sha(path) != report["zip_sha256"] or sha(path.parent / "result.csv") != report["csv_sha256"]:
            raise ValueError("Release identity changed")
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != ["result.csv"] or archive.testzip() is not None:
                raise ValueError("Invalid ZIP structure/CRC")
            payload = archive.read("result.csv")
        validate_result(payload, ids)
        if payload != expected[report["name"]] or payload != (path.parent / "result.csv").read_bytes():
            raise ValueError("Independent cold package reconstruction failed")
    write(out / "independent-audit.json", {"G0": "passed", "packages": 4, "rows_per_package": 322,
        "training_reads_prohibited": True, "fresh_saved_state_replayed": True,
        "cold_order_chunk_differences": differences, "q50_exact_reconstruction": True,
        "unchanged_target_string_mismatches": 0, "new_fits": 0,
        "release_summary_sha256": sha(out / "release-summary.json")})
    print("Independent fresh-process EMA package audit passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "local", "build", "verify"))
    args = parser.parse_args()
    cfg = config()
    out = ROOT / cfg["output"]
    if not out.resolve().is_relative_to(ROOT / "local/runs"):
        raise ValueError("Private output required")
    {"prepare": prepare, "local": descriptive_local, "build": build, "verify": verify}[args.mode](cfg, out)


if __name__ == "__main__":
    main()
