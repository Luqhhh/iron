"""Registered v0.24 lifecycle: two burden-lag models, two isolated packages."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, file_identities, file_sha256, runtime_environment, stable_digest
from ..config import load_yaml
from ..exceptions import ContractError
from ..io import parse_local_time
from .burden_lag_v24 import FEATURE_COLUMNS, append_burden_lag, build_burden_lag
from .component_export import read_json
from .dual_burden_v24 import (BurdenRecencyIron, CANDIDATE_A, CANDIDATE_B, compose_iron,
                              isolate, roundtrip_six)
from .dual_ratio_common import week_intervals
from .qrf_support_shrink import replay_v21, serialize_six_decimals
from .recency_model import RecencyModel, weights
from .structural_run import append_ledger


DEVELOPMENT = Path("local/runs/optimization-v0.15-opt32-r1")
FINAL_QRF = Path("local/runs/optimization-v0.15-v8-user-test-a-r1")
RECENCY_DEV = Path("local/runs/optimization-v0.12-opt25-26-r1")
RECENCY_FINAL = Path("local/runs/optimization-v0.15-platform-recovery-r1")
V22 = Path("local/runs/optimization-v0.22-causal-h2-r8")
V23 = Path("local/runs/optimization-v0.23-recovery-and-stage-validation-r6")
V10_FINAL = Path("local/runs/platform-probes-r2-r1/test_a_V10_full_precision.csv")
V1_FINAL_PARTS = Path("local/runs/optimization-v0.13-opt28-preview-r1/test_a.csv.evidence/R2_rate_parts.csv")
WORKER = Path("workers/qrf_v024")
ORIGINS = range(6, 12)
PRED = ["pred_tap_iron", "pred_tap_time_len"]


def registration():
    value = load_yaml("configs/optimization_v0_24/experiment.yaml")
    if value["candidates"] != {"A": CANDIDATE_A, "B": CANDIDATE_B} or value["burden_lag"]["feature_count"] != 30:
        raise ContractError("v0.24 registration differs")
    if value["budget"] != {"development_catboost_fits": 6, "final_catboost_fits": 1,
            "development_qrf_fits": 6, "final_qrf_fits": 1,
            "development_qrf_preprocessor_fits": 6, "final_qrf_preprocessor_fits": 1,
            "qrf_internal_trees_total": 1792, "LAD_beta_lambda_bias_fits": 0,
            "new_candidate_packages": 2, "new_candidate_platform_tests": 2}:
        raise ContractError("v0.24 budget differs")
    return value


def worker(command, *args):
    # Do not resolve this symlink: the venv path is what activates its site-packages.
    executable = Path("workers/qrf_v015/.venv/bin/python")
    return subprocess.check_output([str(executable), str(WORKER / "worker.py"), command, *map(str, args)], text=True)


def original_root(slot: int) -> Path:
    return FINAL_QRF if slot == 12 else DEVELOPMENT


def original_handoff(slot: int, kind: str) -> Path:
    return original_root(slot) / "features" / str(slot) / f"{kind}.npz"


def _source_files():
    return [Path("configs/optimization_v0_24/experiment.yaml"), Path("configs/optimization_v0_24/access_scope.yaml"),
            Path("src/bf_tap/optimization/burden_lag_v24.py"), Path("src/bf_tap/optimization/dual_burden_v24.py"),
            Path("src/bf_tap/optimization/dual_burden_v24_run.py"), Path("scripts/optimization_v24_dual_burden.py"),
            Path("scripts/optimization_v24_cold_check.py"), Path("tests/test_optimization_v24_burden_lag.py"),
            Path("docs/optimization_v0_24/PLAN.md"), WORKER / "worker.py", WORKER / "test_adapter.py"]


def register(root: Path):
    reg = registration(); scope = load_yaml("configs/optimization_v0_24/access_scope.yaml")
    if root.exists(): raise ContractError("never overwrite a v0.24 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ContractError("clean preregistered working tree required")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != reg["branch"]:
        raise ContractError("registered v0.24 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", reg["base_commit"], "HEAD"], check=True)
    equivalents = []
    for path in Path("local/runs").glob("**/*manifest*.json"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if CANDIDATE_A in text or CANDIDATE_B in text: equivalents.append(str(path))
    if equivalents: raise ContractError("equivalent v0.24 experiment already exists: " + str(equivalents))
    paths = load_yaml("configs/data.local.yaml")["paths"]
    inputs = {key: paths[key] for key in ("train_samples", "tap_history_train", "burden_change", "test_a_samples", "data_dictionary")}
    evidence = {
        "v23_completion": V23 / "completion.json", "v23_v21_zip": V23 / "recovery/Luqhhh_bf_tap_predict_prelim_TGATE_SPOUT1.zip",
        "v23_v21_result": V23 / "recovery/result.csv", "v22_errors": V22 / "all_errors.csv",
        "recency_manifest": RECENCY_DEV / "manifest.json", "qrf_manifest": DEVELOPMENT / "manifest.json",
        "final_qrf_manifest": FINAL_QRF / "manifest.json", "final_recency_completion": RECENCY_FINAL / "completion.json",
    }
    manifest = dict(registration=reg, scope=scope, protection=load_yaml(scope["protection_contract"]),
        inputs=file_identities(inputs), evidence=file_identities(evidence), sources=file_identities({str(p): p for p in _source_files()}),
        worker_environment=json.loads(worker("environment")), worker_sources=json.loads(worker("identity")),
        qrf_parameters=reg["qrf_parameters"], code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runtime=runtime_environment(), equivalent_registry_matches=equivalents, holdout_consumed=True,
        official_target_column_reads=False, test_targets_read=False, platform_feedback_may_change_second_candidate=False)
    root.mkdir(parents=True, exist_ok=False); root.chmod(0o700)
    atomic_write_json(root / "manifest.json", manifest); append_ledger(scope, root)
    atomic_write_json(root / "registration.json", dict(status="FROZEN_BEFORE_FIT", candidates=reg["candidates"],
        platform_order=reg["platform_order"], candidate_platform_budget=2, platform_uploads=0,
        v21_zip_sha256=file_sha256(V23 / "recovery/Luqhhh_bf_tap_predict_prelim_TGATE_SPOUT1.zip")))
    return manifest


def load_original(slot: int, kind: str):
    path = original_handoff(slot, kind); info = read_json(str(path) + ".json")
    if file_sha256(path) != info["sha256"]: raise ContractError("original QRF handoff identity differs")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if arrays["ids"].tolist() != info["ids"]: raise ContractError("original QRF handoff IDs differ")
    return arrays, info


def frame_from(arrays, info):
    index = pd.Index(arrays["ids"].tolist(), name="sample_id")
    numeric = pd.DataFrame(arrays["numeric"], index=index, columns=info["numeric_columns"])
    result = numeric.copy(); result.insert(0, "spout_no", pd.Series(arrays["spout"], index=index, dtype="string"))
    expected = [entry["name"] for entry in info["raw_schema"]]
    if list(result) != expected: raise ContractError("reconstructed original feature schema differs")
    return result


def query_frame(arrays):
    return pd.DataFrame({"sample_id": arrays["ids"].astype(str), "spout_no": arrays["spout"].astype(str),
                         "reference_time": pd.to_datetime(arrays["reference_ns"], utc=True).tz_convert("Asia/Shanghai")})


def _write_expanded(root: Path, slot: int, kind: str, burden: pd.DataFrame):
    arrays, old = load_original(slot, kind); query = query_frame(arrays)
    lag = build_burden_lag(query, burden); old_frame = frame_from(arrays, old)
    lag.index = old_frame.index; expanded = append_burden_lag(old_frame, lag)
    numeric_columns = [c for c in expanded if c != "spout_no"]
    output = root / "features" / str(slot) / f"{kind}.npz"; output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise ContractError("never overwrite expanded handoff")
    packed = {key: value for key, value in arrays.items() if key != "numeric"}
    packed["numeric"] = expanded[numeric_columns].to_numpy(dtype=np.float64)
    np.savez(output, **packed)
    info = dict(sha256=file_sha256(output), ids=arrays["ids"].tolist(), numeric_columns=numeric_columns,
        raw_schema=[dict(name=c, dtype=str(expanded[c].dtype), categorical=c == "spout_no") for c in expanded],
        cutoff_ns=old["cutoff_ns"], cutoff=old["cutoff"], training=kind == "train",
        original_input_sha256=old["sha256"], original_schema_sha256=old["raw_schema_sha256"],
        original_columns_unchanged=True, appended_columns=list(FEATURE_COLUMNS),
        feature_values_sha256=stable_digest(lag.where(lag.notna(), None).to_dict("records")))
    atomic_write_json(str(output) + ".json", info)
    return info


def prepare(root: Path):
    if (root / "p0_complete.json").exists(): raise ContractError("P0 already complete")
    burden = pd.read_csv(load_yaml("configs/data.local.yaml")["paths"]["burden_change"])
    handoffs = {}
    for slot in range(6, 13):
        handoffs[str(slot)] = {kind: _write_expanded(root, slot, kind, burden) for kind in ("train", "evaluation")}
        if len(handoffs[str(slot)]["train"]["ids"]) != registration()["expected_training_rows"][slot]:
            raise ContractError("certified training row count differs")
    atomic_write_json(root / "p0_complete.json", dict(status="PASS", handoffs=handoffs,
        old_columns_unchanged=True, feature_count=len(FEATURE_COLUMNS), timezone="Asia/Shanghai",
        availability_contract="burden.cal_time_is_event_time_and_available_at_ASSUMED"))


def _expanded(root: Path, slot: int, kind: str):
    path = root / "features" / str(slot) / f"{kind}.npz"; info = read_json(str(path) + ".json")
    if file_sha256(path) != info["sha256"]: raise ContractError("expanded handoff changed")
    with np.load(path, allow_pickle=False) as source: arrays = {key: source[key] for key in source.files}
    return arrays, info, frame_from(arrays, info)


def _iron_labels(arrays):
    path = load_yaml("configs/data.local.yaml")["paths"]["train_samples"]
    labels = pd.read_csv(path, dtype={"sample_id": str}, usecols=["sample_id", "tap_iron"]).set_index("sample_id")
    ids = arrays["ids"].tolist()
    if not set(ids) <= set(labels.index): raise ContractError("certified training iron labels missing")
    result = pd.Series(labels.loc[ids, "tap_iron"].to_numpy(dtype=float), index=pd.Index(ids, name="sample_id"), name="tap_iron")
    if not np.isfinite(result).all() or (result < 0).any(): raise ContractError("invalid training iron labels")
    return result


def _fit_A(root: Path, slot: int):
    attempted = len(list((root / "models/A").glob("*/fit_intent.json")))
    if attempted >= 7: raise ContractError("seven-fit CatBoost budget exhausted")
    arrays, info, x = _expanded(root, slot, "train")
    cutoff = pd.Timestamp(info["cutoff"]); metadata = pd.DataFrame({"sample_id": arrays["ids"].astype(str),
        "spout_no": arrays["spout"].astype(str), "reference_time": pd.to_datetime(arrays["reference_ns"], utc=True).tz_convert("Asia/Shanghai"),
        "available_at": pd.to_datetime(arrays["available_ns"], utc=True).tz_convert("Asia/Shanghai")}).set_index("sample_id", drop=False)
    weight, audit = weights(metadata.reset_index(drop=True), cutoff); weight = weight.loc[x.index]
    target = _iron_labels(arrays)
    folder = root / "models/A" / str(slot); folder.mkdir(parents=True, exist_ok=False)
    common = dict(slot=slot, cutoff=str(cutoff), rows=len(x), input_sha256=info["sha256"],
                  training_ids_sha256=stable_digest(arrays["ids"].tolist()), target_sha256=stable_digest(target.tolist()),
                  weight_sha256=stable_digest(weight.tolist()))
    atomic_write_json(folder / "fit_intent.json", common)
    model = BurdenRecencyIron().fit(x, target, weight)
    bundle = model.save(folder / "bundle", common)
    atomic_write_json(folder / "fit_record.json", dict(status="COMPLETED", **common,
        bundle_sha256=file_sha256(folder / "bundle/bundle.json"), weight_audit={"mean": float(weight.mean()),
        "min": float(weight.min()), "max": float(weight.max()), "ESS": float(weight.sum() ** 2 / np.dot(weight, weight))}))
    return bundle


def _old_e04(slot: int, x: pd.DataFrame, old: pd.DataFrame):
    """Recover the frozen E04 component without rebuilding any old feature matrix."""
    if slot != 12:
        return (old.base_iron.to_numpy(dtype=float) - 0.8 * old.direct_iron.to_numpy(dtype=float)) / 0.2
    old_direct = RecencyModel.load(RECENCY_FINAL / "models/tap_iron/bundle").predict(x)
    beta = _beta(slot); rate = old.pred_rate.to_numpy(dtype=float); time = old.pred_tap_time_len.to_numpy(dtype=float)
    delivered = pd.read_csv(RECENCY_FINAL / "predictions/V6I_RECENCY60_IRON.csv", dtype={"sample_id": str}).set_index("sample_id").loc[old.index]
    usable = rate > 1e-6; base = delivered.pred_tap_iron.to_numpy(dtype=float).copy()
    base[usable] = (base[usable] - beta * rate[usable] * time[usable]) / (1.0 - beta)
    e04 = (base - 0.8 * old_direct) / 0.2
    replay, _ = compose_iron(old_direct, e04, rate, time, beta)
    if not np.allclose(replay, delivered.pred_tap_iron.to_numpy(dtype=float), rtol=0, atol=2e-12):
        raise ContractError("recovered final E04 component does not reproduce V6I")
    return e04


def _old_parts(slot: int):
    if slot == 12: return pd.read_csv(V1_FINAL_PARTS, dtype={"sample_id": str})
    return pd.read_csv(RECENCY_DEV / f"predictions/{slot}_parts.csv", dtype={"sample_id": str})


def _beta(slot: int):
    if slot == 12: return float(read_json(RECENCY_FINAL / "coefficients/V6I_RECENCY60_IRON.json")["beta"])
    return float(read_json(RECENCY_DEV / f"corrections/{slot}.json")["beta"]["tap_iron"])


def _parent(slot: int):
    path = V23 / "recovery/result.csv" if slot == 12 else V22 / f"predictions/{slot}/V21_REPLAY.csv"
    return pd.read_csv(path, dtype={"sample_id": str})[["sample_id", *PRED]]


def _v10(slot: int):
    path = V10_FINAL if slot == 12 else V22 / f"predictions/{slot}/V10.csv"
    value = pd.read_csv(path, dtype={"sample_id": str})[["sample_id", *PRED]]
    for column in PRED: value[column] = roundtrip_six(value[column])
    return value


def _history(slot: int):
    if slot == 12:
        path = RECENCY_FINAL / "bundle/original_V1/base_R2/anchor_0/E09_PROCESS_CHANGE_E02/history_snapshot.csv"
    else:
        manifest = read_json(RECENCY_DEV / "manifest.json")
        path = Path(manifest["inventory"][str(slot)]["base_components"]["OR"]["path"]) / "history_snapshot.csv"
    return pd.read_csv(path, dtype={"sample_id": str})


def _old_support(slot: int, ids):
    if slot == 12:
        value = pd.read_csv(Path("local/runs/optimization-v0.22-test-a-platform-r1/directions.csv"), dtype={"sample_id": str})
    else:
        value = pd.read_csv(V22 / f"directions/outer-{slot}.csv", dtype={"sample_id": str})
    if value.sample_id.duplicated().any() or set(ids) != set(value.sample_id): raise ContractError("old QRF support IDs differ")
    return value.set_index("sample_id").loc[ids].effective_neighbors.to_numpy(dtype=float)


def _replay(slot: int, qrf_time=None):
    # The actual root-independent query comes from the certified original handoff.
    original, _ = load_original(slot, "evaluation"); query = query_frame(original); ids = query.sample_id.tolist()
    base = _v10(slot)
    if base.sample_id.tolist() != ids: base = base.set_index("sample_id").loc[ids].reset_index()
    if qrf_time is not None: base["pred_tap_time_len"] = roundtrip_six(qrf_time)
    prediction, audit = replay_v21(base, query, _history(slot), _old_support(slot, ids), cutoff=pd.Timestamp(read_json(str(original_handoff(slot, "evaluation")) + ".json")["cutoff"]))
    prediction["pred_tap_time_len"] = roundtrip_six(prediction.pred_tap_time_len)
    return prediction, audit


def _predict_slot(root: Path, slot: int):
    arrays, _, x = _expanded(root, slot, "evaluation"); ids = arrays["ids"].tolist(); parent = _parent(slot)
    if parent.sample_id.tolist() != ids: parent = parent.set_index("sample_id").loc[ids].reset_index()
    bundle_sha = file_sha256(root / f"models/A/{slot}/bundle/bundle.json")
    model = BurdenRecencyIron.load(root / f"models/A/{slot}/bundle", bundle_sha)
    direct = model.predict(x); old = _old_parts(slot).set_index("sample_id").loc[ids]
    iron, a_audit = compose_iron(direct, _old_e04(slot, x, old), old.pred_rate, old.pred_tap_time_len, _beta(slot))
    candidate_a = isolate(parent, iron=iron, candidate=CANDIDATE_A)
    qrf_path = root / "worker_predictions" / f"{slot}.npz"
    worker("predict", "--root", root.resolve(), "--slot", slot, "--output", qrf_path.resolve())
    with np.load(qrf_path, allow_pickle=False) as source:
        if source["ids"].tolist() != ids: raise ContractError("new QRF prediction IDs differ")
        qrf = source["median"]
    candidate_b_raw, b_audit = _replay(slot, qrf)
    candidate_b = isolate(parent, time=candidate_b_raw.pred_tap_time_len.to_numpy(), candidate=CANDIDATE_B)
    old_replay, old_audit = _replay(slot)
    if serialize_six_decimals(old_replay) != serialize_six_decimals(parent): raise ContractError("old QRF substitution does not reproduce V21")
    destination = root / "predictions"; destination.mkdir(exist_ok=True)
    candidate_a.to_csv(destination / f"{slot}_{CANDIDATE_A}.csv", index=False)
    candidate_b.to_csv(destination / f"{slot}_{CANDIDATE_B}.csv", index=False)
    atomic_write_json(destination / f"{slot}_audit.json", dict(A=a_audit, B=b_audit, old_replay=old_audit,
        A_unchanged_time_exact=bool(np.array_equal(candidate_a.pred_tap_time_len, parent.pred_tap_time_len)),
        B_unchanged_iron_exact=bool(np.array_equal(candidate_b.pred_tap_iron, parent.pred_tap_iron))))


def develop(root: Path):
    if (root / "development_complete.json").exists(): raise ContractError("development already complete")
    for slot in ORIGINS: _fit_A(root, slot)
    for slot in ORIGINS: worker("fit", "--root", root.resolve(), "--slot", slot)
    atomic_write_json(root / "development_models_complete.json", {str(slot): file_sha256(root / f"models/B/{slot}/bundle.json") for slot in ORIGINS})
    for slot in ORIGINS: _predict_slot(root, slot)
    atomic_write_json(root / "development_complete.json", dict(status="PASS_G0_DEVELOPMENT_PREDICTIONS_FROZEN",
        A_fits=6, B_forest_fits=6, B_preprocessor_fits=6, B_internal_trees=1536, calibration_fits=0))


def _candidate_errors(root: Path, candidate: str):
    source = pd.read_csv(V22 / "all_errors.csv", dtype={"sample_id": str})
    base = source.loc[source.candidate == "V1"].copy()
    frames = []
    for origin, part in base.groupby("origin", sort=False):
        slot = int(str(origin)[-2:]); pred = pd.read_csv(root / f"predictions/{slot}_{candidate}.csv", dtype={"sample_id": str}).set_index("sample_id")
        p = pred.loc[part.sample_id]
        current = part.copy(); current["pred_tap_iron"] = roundtrip_six(p.pred_tap_iron); current["pred_tap_time_len"] = roundtrip_six(p.pred_tap_time_len)
        for target in ("tap_iron", "tap_time_len"):
            current[f"error_{target}"] = current[f"pred_{target}"] - current[target]
            current[f"abs_error_{target}"] = current[f"error_{target}"].abs()
        current["candidate"] = candidate; frames.append(current)
    return pd.concat(frames, ignore_index=True)


def _reference_errors(candidate: str):
    source = pd.read_csv(V22 / "all_errors.csv", dtype={"sample_id": str})
    value = source.loc[source.candidate == candidate].copy()
    for target in ("tap_iron", "tap_time_len"):
        value[f"pred_{target}"] = roundtrip_six(value[f"pred_{target}"])
        value[f"error_{target}"] = value[f"pred_{target}"] - value[target]
        value[f"abs_error_{target}"] = value[f"error_{target}"].abs()
    return value


def _score_rows(errors: pd.DataFrame, algorithm: str):
    rows = []
    for unit, part in errors.groupby("unit", sort=False):
        target_metrics = {}
        for target in ("tap_iron", "tap_time_len"):
            denom = float(part[target].sum()); numerator = float(part[f"abs_error_{target}"].sum()); signed = float(part[f"error_{target}"].sum())
            target_metrics[target] = numerator / denom
            rows.append(dict(algorithm=algorithm, unit=unit, cutoff=part.origin.iloc[0],
                horizon=None if pd.isna(part.horizon.iloc[0]) else int(part.horizon.iloc[0]), target=target, n=len(part),
                target_sum=denom, absolute_error_sum=numerator, signed_error_sum=signed, signed_bias=signed / len(part),
                wmape=numerator / denom, E=0.0))
        E=float(np.mean(list(target_metrics.values())))
        for row in rows[-2:]: row["E"] = E
    return rows


def _summary(scorecard: pd.DataFrame):
    cells = scorecard[["algorithm", "unit", "cutoff", "horizon", "E"]].drop_duplicates()
    rows=[]
    for algorithm, part in cells.groupby("algorithm"):
        for row in part.itertuples(index=False): rows.append(dict(algorithm=algorithm, scope=row.unit, scope_type="CELL", E=row.E))
        for h in range(1,5): rows.append(dict(algorithm=algorithm, scope=f"H{h}", scope_type="HORIZON_MEAN", E=float(part.loc[part.horizon==h,"E"].mean())))
        hmeans=[r["E"] for r in rows if r["algorithm"]==algorithm and r["scope_type"]=="HORIZON_MEAN"]
        rows.append(dict(algorithm=algorithm, scope="J", scope_type="GRID", E=float(np.mean(hmeans))))
        for unit in ("DEV_LONG","DEV_SHORT"):
            rows.append(dict(algorithm=algorithm, scope=unit, scope_type="DEVELOPMENT", E=float(part.loc[part.unit==unit,"E"].iloc[0])))
    return pd.DataFrame(rows)


def score(root: Path):
    candidates = {CANDIDATE_A: _candidate_errors(root, CANDIDATE_A), CANDIDATE_B: _candidate_errors(root, CANDIDATE_B)}
    references = {name: _reference_errors(name) for name in ("V1", "V10", "V21_REPLAY")}
    all_errors = pd.concat([*references.values(), *candidates.values()], ignore_index=True)
    all_errors.to_csv(root / "all_errors.csv", index=False)
    scorecard = pd.DataFrame([row for name, errors in {**references, **candidates}.items() for row in _score_rows(errors, name)])
    scorecard.to_csv(root / "canonical_scorecard.csv", index=False)
    summary = _summary(scorecard); summary.to_csv(root / "canonical_summary.csv", index=False)
    # DEV units are retained as CELL rows for a complete scorecard and as the
    # registered DEVELOPMENT summaries.  Only the latter enters the unique
    # top-level comparison lookup.
    reported = summary.loc[summary.scope_type.isin(("HORIZON_MEAN", "GRID", "DEVELOPMENT"))].copy()
    if reported.duplicated(["algorithm", "scope"]).any():
        raise ContractError("non-unique registered summary scope")
    lookup = reported.set_index(["algorithm", "scope"]).E.sort_index()
    deltas=[]
    for candidate in (CANDIDATE_A,CANDIDATE_B):
        for scope in reported.scope.unique():
            if (candidate,scope) not in lookup: continue
            deltas.append(dict(candidate=candidate,scope=scope,E=lookup[candidate,scope],
                delta_vs_V1=lookup[candidate,scope]-lookup["V1",scope],
                delta_vs_V21=lookup[candidate,scope]-lookup["V21_REPLAY",scope]))
    pd.DataFrame(deltas).to_csv(root / "reference_deltas.csv", index=False)
    bootstrap = {candidate: {ref: week_intervals(all_errors,candidate,ref,1000,2026) for ref in ("V1","V21_REPLAY")} for candidate in candidates}
    atomic_write_json(root / "bootstrap.json", bootstrap)
    diagnostics={}
    for candidate, errors in candidates.items():
        diagnostics[candidate]={}
        for spout, part in errors.groupby("spout_no"):
            diagnostics[candidate][str(spout)]={target:{"n":len(part),"target_sum":float(part[target].sum()),
                "abs_error_sum":float(part[f"abs_error_{target}"].sum()),"wmape":float(part[f"abs_error_{target}"].sum()/part[target].sum()),
                "signed_bias":float(part[f"error_{target}"].mean())} for target in ("tap_iron","tap_time_len")}
    atomic_write_json(root / "spout_diagnostics.json", diagnostics)
    atomic_write_json(root / "offline_assessment.json", dict(status="COMPLETED_CONSUMED_RETROSPECTIVE_NOT_A_PLATFORM_GATE",
        candidates={c:{scope:float(lookup[c,scope]) for scope in ("H1","H2","H3","H4","J","DEV_LONG","DEV_SHORT")} for c in candidates},
        references={r:{scope:float(lookup[r,scope]) for scope in ("H1","H2","H3","H4","J","DEV_LONG","DEV_SHORT")} for r in references},
        both_candidates_retain_preregistered_platform_slot=True))


def finalize(root: Path):
    if not (root / "offline_assessment.json").exists(): raise ContractError("score development before final fits")
    _fit_A(root, 12); worker("fit", "--root", root.resolve(), "--slot", 12)
    atomic_write_json(root / "final_models_complete.json", {"12": file_sha256(root / "models/B/12/bundle.json")})
    _predict_slot(root, 12)
    parent = _parent(12)
    package_receipts={}
    for key,candidate in (("A",CANDIDATE_A),("B",CANDIDATE_B)):
        value=pd.read_csv(root/f"predictions/12_{candidate}.csv",dtype={"sample_id":str})
        folder=root/"submissions"/candidate;folder.mkdir(parents=True,exist_ok=False)
        csv_path=folder/"result.csv";csv_path.write_bytes(serialize_six_decimals(value))
        archive=folder/"Luqhhh_bf_tap_predict_prelim.zip"
        with ZipFile(archive,"x",compression=ZIP_DEFLATED) as handle: handle.write(csv_path,arcname="result.csv")
        with ZipFile(archive) as handle:
            if handle.namelist()!=["result.csv"] or handle.read("result.csv")!=csv_path.read_bytes(): raise ContractError("submission archive payload differs")
        reread=pd.read_csv(csv_path,dtype={"sample_id":str})
        unchanged="pred_tap_time_len" if key=="A" else "pred_tap_iron"
        if not np.array_equal(reread[unchanged].to_numpy(),parent[unchanged].to_numpy()): raise ContractError("unchanged final target differs from V21 CSV")
        package_receipts[key]=dict(candidate=candidate,rows=len(value),result_sha256=file_sha256(csv_path),zip_sha256=file_sha256(archive),
            unchanged_target=unchanged,unchanged_target_csv_exact=True,platform_uploads=0,platform_feedback=None)
    atomic_write_json(root/"packages_frozen_before_feedback.json",dict(status="PASS_TWO_PACKAGES_FROZEN",order=["A","B"],receipts=package_receipts,
        definitions_may_change_after_A_feedback=False,platform_upload_budget=2,platform_uploads=0))


def cold(root: Path):
    slot=12; arrays,_,x=_expanded(root,slot,"evaluation"); bundle_sha=file_sha256(root/"models/A/12/bundle/bundle.json")
    model=BurdenRecencyIron.load(root/"models/A/12/bundle",bundle_sha); full=model.predict(x)
    if not np.array_equal(model.predict(x.iloc[::-1])[::-1],full): raise ContractError("CatBoost reverse inference differs")
    if not np.array_equal(np.concatenate([model.predict(x.iloc[i:i+73]) for i in range(0,len(x),73)]),full): raise ContractError("CatBoost chunk inference differs")
    if not np.array_equal(model.predict(x.iloc[[len(x)//2]]),full[[len(x)//2]]): raise ContractError("CatBoost single inference differs")
    cold_path=root/"cold/qrf.npz";worker("cold","--root",root.resolve(),"--slot",12,"--output",cold_path.resolve())
    atomic_write_json(root/"cold_validation.json",dict(status="PASS",A_full_reverse_chunk_single_exact=True,
        B_full_reverse_chunk_subset_single_exact=True,fit_attempts={"CatBoost":0,"QRF":0,"preprocessor":0,"calibration":0},
        trusted_private_bundles_only=True))
    atomic_write_json(root/"fit_counts.json",dict(CatBoost_attempted=7,CatBoost_completed=7,QRF_attempted=7,QRF_completed=7,
        QRF_preprocessor_attempted=7,QRF_preprocessor_completed=7,QRF_internal_trees=1792,LAD_beta_lambda_bias=0))
    atomic_write_json(root/"completion.json",dict(status="READY_FOR_TWO_EXPLICIT_PLATFORM_SUBMISSIONS",G0="PASS",G1="REPORTED_SEPARATELY_NOT_A_GATE",
        candidates=[CANDIDATE_A,CANDIDATE_B],platform_order=["A","B"],platform_uploads=0,platform_budget=2,
        packages_sha256=file_sha256(root/"packages_frozen_before_feedback.json"),cold_validation_sha256=file_sha256(root/"cold_validation.json"),
        fit_counts_sha256=file_sha256(root/"fit_counts.json"),test_targets_read=False,automatic_upload=False,desktop_writes=0,public_pushes=0))


def run(root: Path):
    register(root); prepare(root); develop(root); score(root); finalize(root); cold(root)
