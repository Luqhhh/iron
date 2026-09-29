"""Isolated RFM outer units and complete-coverage evidence collection.

No phase launch or automatic confirmation is exposed here. A future controller
must bind source, reference and resource admission before scheduling units.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from .candidate_tiers import classify_candidates
from .data import FEATURES, TARGETS
from .rfm_audit import audit_partition
from .rfm_model import SavedRFM, array_digest, feature_frame, fit_ids, fit_partition
from .rfm_protocol import ARMS, ReservationLedger, canonical, file_hash, phase_tasks, phase_limits, write_new
from .rfm_scoring import score_seed, decide_development, decide_confirmation, wmape


def task_name(task):
    if (set(task) != {"target", "arm", "seed", "fold"}
            or task["target"] not in TARGETS or task["arm"] not in ARMS
            or type(task["seed"]) is not int or task["seed"] not in (42,3407,7777,12011)
            or type(task["fold"]) is not int or task["fold"] not in range(5)):
        raise ValueError("invalid frozen outer task")
    return f"{task['target']}-{task['arm']}-s{task['seed']}-f{task['fold']}"


def validate_partition(training, query):
    training_ids, query_ids = fit_ids(training), fit_ids(query)
    if not training_ids or not query_ids or set(training_ids) & set(query_ids):
        raise ValueError("outer training/query IDs overlap or are empty")
    if set(query.columns) & set(TARGETS):
        raise ValueError("outer query must not contain labels")
    feature_frame(training)
    feature_frame(query)
    train_groups = set(pd.util.hash_pandas_object(training[list(FEATURES)], index=False))
    query_groups = set(pd.util.hash_pandas_object(query[list(FEATURES)], index=False))
    if train_groups & query_groups:
        raise ValueError("outer duplicate-feature group leakage")
    return training_ids, query_ids


def execute_unit(task, training, query, output, ledger_root, policy_sha256):
    name = task_name(task)
    ledger = ReservationLedger.open(ledger_root, policy_sha256)
    with ledger.event("outer_fit", (name,), {"task": task}) as event:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        train_ids, query_ids = validate_partition(training, query)
        model, selection = fit_partition(training, task["target"], task["arm"],
                                        ledger.scoped((name,)), directory=output / "models")
        prediction = model.predict(query)
        with (output / "predictions.npz").open("xb") as handle:
            np.savez(handle, query_ids=np.asarray(query_ids, dtype=str), prediction=prediction)
        complete = {"task": task, "name": name,
            "training_ids_sha256": hashlib.sha256(canonical(train_ids)).hexdigest(),
            "query_ids_sha256": hashlib.sha256(canonical(query_ids)).hexdigest(),
            "query_features_sha256": array_digest(feature_frame(query).to_numpy(dtype=float)),
            "selection_sha256": file_hash(output / "models" / "selection.json"),
            "prediction_sha256": file_hash(output / "predictions.npz"),
            "selected_state": selection["selected_state"], "training_rows": len(training),
            "query_rows": len(query), "ledger_policy_sha256": policy_sha256}
        write_new(output / "complete.json", complete)
        anchor = file_hash(output / "complete.json")
        event.update(unit_complete_sha256=anchor, selected_state=selection["selected_state"])
    return {"name": name, "complete_sha256": anchor}


def audit_unit(directory, task, training, query, *, expected_sha256):
    directory = Path(directory)
    name = task_name(task)
    train_ids, query_ids = validate_partition(training, query)
    if {p.name for p in directory.iterdir()} != {"models", "predictions.npz", "complete.json"}:
        raise ValueError("unexpected outer unit artifact")
    if file_hash(directory / "complete.json") != expected_sha256:
        raise ValueError("outer unit completion identity mismatch")
    complete = json.loads((directory / "complete.json").read_text())
    if (complete["task"] != task or complete["name"] != name
            or complete["training_ids_sha256"] != hashlib.sha256(canonical(train_ids)).hexdigest()
            or complete["query_ids_sha256"] != hashlib.sha256(canonical(query_ids)).hexdigest()
            or complete["query_features_sha256"] != array_digest(feature_frame(query).to_numpy(dtype=float))
            or complete["training_rows"] != len(training) or complete["query_rows"] != len(query)):
        raise ValueError("outer task/row provenance mismatch")
    audit = audit_partition(directory / "models", training, task["target"],
                            expected_selection_sha256=complete["selection_sha256"])
    if audit["selected_state"] != complete["selected_state"]:
        raise ValueError("outer selected state mismatch")
    if file_hash(directory / "predictions.npz") != complete["prediction_sha256"]:
        raise ValueError("outer predictions changed")
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"query_ids", "prediction"}:
            raise ValueError("unexpected outer prediction schema")
        if not np.array_equal(saved["query_ids"], np.asarray(query_ids, dtype=str)):
            raise ValueError("outer prediction row identity mismatch")
        prediction = saved["prediction"].copy()
    selection = json.loads((directory / "models" / "selection.json").read_text())
    state = complete["selected_state"]
    model = SavedRFM.load(directory / "models" / "refit" / f"state-{state}",
                          expected_sha256=selection["artifacts"]["refit"][state]["complete_sha256"])
    expected = model.predict(query)
    if prediction.shape != expected.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid outer prediction values")
    delta = float(np.max(np.abs(prediction-expected)))
    if delta > 1e-8:
        raise ValueError("outer saved/cold prediction mismatch")
    return prediction, {**audit, "name": name, "query_rows": len(query),
                        "outer_cold_difference": delta, "ledger_policy_sha256": complete["ledger_policy_sha256"]}


def _expected_events(name, selected, arm):
    if type(selected) is not int or selected not in range(4) or (arm == "FIXED_KRR" and selected != 0):
        raise ValueError("invalid selected state for reservation accounting")
    result = {("outer_fit", canonical([name]))}
    for stage, updates in (("inner", 3 if arm == "FULL_RFM" else 0), ("refit", selected)):
        result.add(("procedure", canonical([name, "procedure", [stage, None]])))
        result.update(("solve", canonical([name, "solve", [stage, s]])) for s in range(updates+1))
        result.update(("update", canonical([name, "update", [stage, s]])) for s in range(1,updates+1))
    return result


def collect_audited_phase(workspace, output, phase, frame, folds, current, historical,
                          unit_anchors, ledger_policy_sha256, development_records=None):
    """Rebuild every held-out column and decision from cold-audited artifacts.

    The future phase audit must separately bind caller data/references/source
    hashes to its frozen manifest before using this collector.
    """
    if phase == "development":
        if development_records is not None:
            raise ValueError("development cannot inherit selection records")
        eligible = None
    else:
        if development_records is None:
            raise ValueError("confirmation requires audited development records")
        eligible = decide_development(development_records)["eligible_targets"]
    tasks = phase_tasks(phase, eligible)
    expected_names = {task_name(t) for t in tasks}
    if set(unit_anchors) != expected_names:
        raise ValueError("incomplete or extra outer task anchors")
    output, workspace = Path(output), Path(workspace)
    if {p.name for p in (output / "units").iterdir()} != expected_names:
        raise ValueError("incomplete or extra outer unit directories")
    ledger = ReservationLedger.open(output / "ledger", ledger_policy_sha256)
    if ledger.limits != phase_limits(phase, eligible):
        raise ValueError("phase reservation budget differs from frozen limits")
    counts = ledger.inspect()
    if any(counts[k][kind] for k in ("failed", "incomplete") for kind in ledger.limits):
        raise ValueError("phase contains failed/incomplete reservations")
    n = len(frame)
    y = {t: frame[t].to_numpy(dtype=float) for t in TARGETS}
    members, audits, expected_events = {}, [], set()
    for task in tasks:
        name, seed, fold, target, arm = task_name(task), task["seed"], task["fold"], task["target"], task["arm"]
        fv = np.asarray(folds[seed])
        if fv.shape != (n,) or set(fv) != set(range(5)):
            raise ValueError("invalid complete outer fold vector")
        mask = fv == fold
        training = frame.loc[~mask].reset_index(drop=True)
        query = frame.loc[mask, ["sample_id", "spout_no", *FEATURES]].reset_index(drop=True)
        prediction, audit = audit_unit(output / "units" / name, task, training, query,
                                       expected_sha256=unit_anchors[name])
        if audit["ledger_policy_sha256"] != ledger_policy_sha256:
            raise ValueError("outer unit belongs to another ledger")
        vector = members.setdefault((target,arm,seed), np.full(n,np.nan))
        if np.isfinite(vector[mask]).any():
            raise ValueError("duplicate OOF coverage")
        vector[mask] = prediction
        audits.append(audit)
        expected_events.update(_expected_events(name,audit["selected_state"],arm))
    actual_events = set()
    audited_states = {a["name"]: a["selected_state"] for a in audits}
    for path in (output / "ledger" / "events").glob("*.started.json"):
        record = json.loads(path.read_text())
        actual_events.add((record["kind"], canonical(record["key"])))
        if record["kind"] == "outer_fit":
            name = record["key"][0]
            receipt = json.loads(path.with_name(path.name.replace("started.json", "complete.json")).read_text())
            if receipt["result"] != {"unit_complete_sha256": unit_anchors.get(name),
                                     "selected_state": audited_states.get(name)}:
                raise ValueError("outer receipt does not bind audited completion/state")
    if actual_events != expected_events:
        raise ValueError("unaccounted or missing fit/solver/update starts")
    targets = list(TARGETS) if phase == "development" else eligible
    seeds = [42,3407] if phase == "development" else [7777,12011]
    records = []
    metrics = {}
    for target in targets:
        metrics[target] = {r:{} for r in ("CURRENT", "FULL_RFM")}
        for seed in seeds:
            values={arm: members[target,arm,seed] for arm in ARMS}
            records.extend(score_seed(y,folds[seed],current[seed],historical[seed],values,target=target,seed=seed))
            for route, prediction in (("CURRENT",current[seed][target]),
                                       ("FULL_RFM",.8*current[seed][target]+.2*values["FULL_RFM"])):
                metrics[target][route][str(seed)]={"wmape":wmape(y[target],prediction),
                    "by_fold":{str(f):wmape(y[target][folds[seed]==f],prediction[folds[seed]==f]) for f in range(5)},
                    "by_spout":{str(s):wmape(y[target][frame.spout_no.to_numpy()==s],prediction[frame.spout_no.to_numpy()==s])
                                for s in sorted(frame.spout_no.unique())}}
    if phase == "development":
        decision = decide_development(records)
        spec={"split_seeds":seeds,"folds":5,"candidates":{t:["FULL_RFM"] for t in targets},
              "reference_by_target":{t:"CURRENT" for t in targets},
              "tie_preference_by_target":{t:["FULL_RFM"] for t in targets}}
        tiers=classify_candidates(metrics,spec,yaml.safe_load((workspace/'configs/candidate_tiers.yaml').read_text()))
    else:
        decision=decide_confirmation(development_records,records)
        tiers=None
    return {"status":"passed","phase":phase,"records":records,"decision":decision,"tiers":tiers,
            "counts":counts,"audited_units":audits,"new_audit_fits":0,"release_authorized":False}
