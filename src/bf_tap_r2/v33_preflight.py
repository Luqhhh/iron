"""Synthetic learnability and provenance checks; no official candidate fits."""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import yaml

from .data import FEATURES, TARGETS
from .v7_periodic import file_hash, write_new
from .v33_mixture import MixtureRegressor
from .v33_run import SPEC, append_event, check_runtime, source_hashes, verify_reference_cache


def run(root):
    root = Path(root).resolve()
    spec = yaml.safe_load((root / SPEC).read_text())
    check_runtime(spec)
    verify_reference_cache(root, spec)
    out = root / "local/runs/round2-v33/preflight-r1"
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(33001)
    x = rng.normal(size=(640, len(FEATURES)))
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["sample_id"] = [f"synthetic-preflight-{i}" for i in range(len(frame))]
    frame["spout_no"] = 1 + np.arange(len(frame)) % 2
    y = 100 + 10*x[:,0] + 2*np.sin(x[:,1]) + rng.normal(size=len(frame))
    train, query = frame.iloc[:480], frame.iloc[480:]
    constant = float(np.abs(y[480:] - np.median(y[:480])).mean())
    results = []
    for recipe in spec["recipes"]:
        append_event(out / "events.jsonl", {"event":"optimizer_started", "recipe":recipe, "synthetic":True})
        start = time.monotonic()
        model = MixtureRegressor(recipe, spec["training"]).initialize(train, y[:480])
        model.train(80)
        predicted = model.predict(query)
        mae = float(np.abs(predicted-y[480:]).mean())
        if not mae < constant:
            append_event(out / "events.jsonl", {"event":"failed", "recipe":recipe, "mae":mae})
            raise ValueError("Synthetic learnability failed")
        model.save(out / f"{recipe}.pt")
        cold = MixtureRegressor.load(out / f"{recipe}.pt")
        difference = float(np.max(np.abs(cold.predict(query.iloc[::-1])[::-1]-predicted)))
        if difference > spec["preflight"]["cold_predict_atol"]:
            raise ValueError("Synthetic persistence/order failed")
        row = {"recipe":recipe,"mae":mae,"constant_mae":constant,"cold_difference":difference,
               "seconds":time.monotonic()-start,"parameter_count":model.metadata()["parameter_count"]}
        results.append(row)
        append_event(out / "events.jsonl", {"event":"complete", **row})
    report = {"status":"passed", "spec_sha256":file_hash(root / SPEC),
              "source_hashes":source_hashes(root), "results":results,
              "reference_cache_verified":True, "official_candidate_fits":0, "synthetic_optimizer_runs":2}
    write_new(out / "report.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "source_hashes"}), flush=True)


if __name__ == "__main__":
    run(Path.cwd())
