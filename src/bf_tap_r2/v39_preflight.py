"""Synthetic full-shape admission under the explicit V37 time authorization."""
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import yaml

from .data import FEATURES
from .v7_periodic import file_hash, write_new
from .v39_regressor import TreeRegressor
from .v39_run import SPEC, append_event, check_runtime, source_hashes, verify_reference_cache


def synthetic():
    rng = np.random.default_rng(25001)
    x = rng.normal(size=(1024, len(FEATURES)))
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["sample_id"] = [f"synthetic-v39-{i}" for i in range(len(frame))]
    frame["spout_no"] = rng.integers(1, 4, len(frame))
    y = 10+3*(x[:, 0] > 0)+2*(x[:, 1] > 0)-(x[:, 2] > 0).astype(int)
    return frame.iloc[:768], frame.iloc[768:], y[:768].astype(float), y[768:].astype(float)


def check_engineering(root, spec):
    authorization = spec["engineering_authorization"]
    path = root/authorization["audit_path"]
    if file_hash(path) != authorization["audit_sha256"]:
        raise ValueError("V37 audit changed")
    audit = json.loads(path.read_text())
    if (audit["audit_status"] != "passed" or not audit["checks"]["worker_rss"]
            or audit["projected_hours"] > authorization["accepted_projection_hours"]
            or authorization["user_quote"] != "7.61小时可接受"):
        raise ValueError("Explicit resource authorization does not cover the measured run")
    if file_hash(root/authorization["model_source"]) != authorization["model_source_sha256"]:
        raise ValueError("Frozen V37 model changed")
    old = yaml.safe_load((root/"configs/round2_v37/SPEC.yaml").read_text())
    if any(spec["training"][k] != v for k, v in old["training"].items()):
        raise ValueError("Original training settings changed")
    if spec["training"]["model"] != old["model"]:
        raise ValueError("Original model settings changed")
    if 4*audit["peak_worker_rss_mib"]+1024 > psutil.virtual_memory().available/1024**2:
        raise ValueError("Available RAM does not support four workers")
    for name in ("equivalence-GLOBAL.json", "equivalence-INSTANCE.json"):
        evidence = path.parent/name
        if file_hash(evidence) != audit["evidence_sha256"][name]:
            raise ValueError("Original equivalence record changed")
        record = json.loads(evidence.read_text())
        for source, sha in record["source_hashes"].items():
            if file_hash(root/source) != sha:
                raise ValueError(f"Original equivalence source changed: {source}")
    return {"original_decision": audit["decision"], "accepted_hours": authorization["accepted_projection_hours"],
            "projection_hours": audit["projected_hours"], "new_resource_optimizer_runs": 0}


def cold(root, out):
    spec = yaml.safe_load((root/SPEC).read_text())
    _, query, _, _ = synthetic()
    results = {}
    for recipe in spec["recipes"]:
        model = TreeRegressor.load(out/f"{recipe}.pt")
        expected = np.load(out/f"{recipe}-prediction.npy", allow_pickle=False)
        predictions = [model.predict(query), model.predict(query.iloc[::-1])[::-1],
                       np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0, len(query), 37)])]
        difference = max(float(np.max(np.abs(p-expected))) for p in predictions)
        if difference > spec["preflight"]["cold_predict_atol"]:
            raise ValueError(f"Cold/order/chunk failure: {recipe} {difference}")
        results[recipe] = difference
    print(json.dumps(results))


def run(root):
    spec = yaml.safe_load((root/SPEC).read_text())
    check_runtime(spec)
    engineering = check_engineering(root, spec)
    verify_reference_cache(root, spec)
    out = root/"local/runs/round2-v39/preflight-r1"
    out.mkdir(parents=True, exist_ok=False)
    train, query, y_train, y_query = synthetic()
    constant = float(np.abs(y_query-np.median(y_train)).mean())
    results = []
    for recipe in spec["recipes"]:
        append_event(out/"events.jsonl", {"event": "optimizer_started", "recipe": recipe, "synthetic": True})
        start = time.monotonic()
        try:
            model = TreeRegressor(recipe, spec["training"]).initialize(train, y_train)
            model.train(80)
            prediction = model.predict(query)
            mae = float(np.abs(prediction-y_query).mean())
            model.save(out/f"{recipe}.pt")
            with (out/f"{recipe}-prediction.npy").open("xb") as stream:
                np.save(stream, prediction, allow_pickle=False)
            write_new(out/f"{recipe}-metadata.json", model.metadata())
            if mae >= constant:
                raise ValueError(f"Synthetic learnability failed: {recipe} {mae} >= {constant}")
            row = {"recipe": recipe, "mae": mae, "constant_mae": constant,
                   "seconds": time.monotonic()-start, "peak_rss_mib": model.metadata()["peak_rss_mib"]}
            results.append(row)
            append_event(out/"events.jsonl", {"event": "complete", **row})
            print(json.dumps(row), flush=True)
            del model
        except BaseException as error:
            append_event(out/"events.jsonl", {"event": "failed", "recipe": recipe, "error": repr(error)})
            raise
    completed = subprocess.run([sys.executable, "-m", "bf_tap_r2.v39_preflight", "--cold", str(out)],
                               cwd=root, check=True, text=True, capture_output=True)
    differences = json.loads(completed.stdout)
    write_new(out/"cold.json", differences)
    report = {"status": "passed", "spec_sha256": file_hash(root/SPEC), "source_hashes": source_hashes(root),
              "results": results, "engineering_authorization": engineering, "cold_differences": differences,
              "reference_cache_verified": True, "official_candidate_fits": 0, "synthetic_optimizer_runs": 2}
    write_new(out/"report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "source_hashes"}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--cold":
        cold(Path.cwd(), Path(sys.argv[2]))
    else:
        run(Path.cwd())
