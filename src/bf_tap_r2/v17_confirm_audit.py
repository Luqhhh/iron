"""Read-only independent identity and arithmetic audit of V17 confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import paired_summary
from .v5_spec import load_v5_spec
from .v7_periodic import digest, file_hash, write_new
from .v17_confirm import confirmed_references
from .v17_run import oof_from_folds, score_candidate


def audit(root, development, confirmation, output):
    root = Path(root).resolve()
    dev = (root / development).resolve()
    run = (root / confirmation).resolve()
    output = (root / output).resolve()
    allowed = root / "local/runs/round2-v17"
    if not dev.is_relative_to(allowed) or not run.is_relative_to(allowed) or not output.is_relative_to(run):
        raise ValueError("Private V17 run and audit paths required")
    spec_path = root / "configs/round2_v17/SPEC.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    dev_manifest = json.loads((dev / "manifest.json").read_text())
    dev_summary = json.loads((dev / "summary.json").read_text())
    dev_audit = json.loads((dev / "audit-r1.json").read_text())
    if (dev_audit["status"] != "passed" or manifest["spec_sha256"] != file_hash(spec_path)
            or manifest["development_manifest_sha256"] != file_hash(dev / "manifest.json")
            or manifest["development_summary_sha256"] != file_hash(dev / "summary.json")
            or manifest["development_audit_sha256"] != file_hash(dev / "audit-r1.json")
            or manifest["source_sha256"] != file_hash(root / "src/bf_tap_r2/v17_confirm.py")):
        raise ValueError("Confirmation freeze or development evidence changed")
    for name, sha in dev_manifest["source_hashes"].items():
        if file_hash(root / "src/bf_tap_r2" / name) != sha:
            raise ValueError(f"Development source changed: {name}")
    selected = {target: name for target, name in dev_summary["selected_for_confirmation"].items() if name}
    if selected != manifest["selected"] or summary["status"] != "confirmation_complete":
        raise ValueError("Finalist selection changed")
    frame = load_v5_training_frame(root)
    if hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest() != manifest["data_digest"]:
        raise ValueError("Training frame changed")
    seeds = spec["split_seeds"] + spec["confirmation_seeds"]
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root)) for seed in seeds}
    if {str(s): digest(v.tolist()) for s, v in folds.items()} != manifest["fold_digests"]:
        raise ValueError("Confirmation fold identity changed")
    a35, b0, hashes = confirmed_references(root, frame, folds, spec)
    if hashes != manifest["reference_hashes"]:
        raise ValueError("Four-seed reference identity changed")
    events = [json.loads(line) for line in (run / "fit_ledger.jsonl").read_text().splitlines()]
    expected = {f"{name}-s{seed}-f{fold}" for name in selected.values()
                for seed in spec["confirmation_seeds"] for fold in range(5)}
    if len(events) != len(expected) or {e["key"] for e in events} != expected or any(e["event"] != "complete" for e in events):
        raise ValueError("Incomplete or duplicate confirmation ledger")
    for event in events:
        key = event["key"]
        _, suffix = key.rsplit("-s", 1)
        seed_text, fold_text = suffix.split("-f")
        seed, fold = int(seed_text), int(fold_text)
        mask = folds[seed] == fold
        path = run / f"{key}.npy"
        values = np.load(path, allow_pickle=False)
        metadata = event["metadata"]
        if (file_hash(path) != event["prediction_sha256"]
                or values.shape != (int(mask.sum()),) or not np.isfinite(values).all()
                or metadata["fit_ids_digest"] != digest(frame.loc[~mask, "sample_id"].tolist())
                or metadata["fit_rows"] != int((~mask).sum())
                or metadata["optimizer_runs"] != 2):
            raise ValueError(f"Confirmation prediction or row identity mismatch: {key}")
    if set(summary["results"]) != set(selected):
        raise ValueError("Confirmation summary target mismatch")
    for target, name in selected.items():
        dev_pred = oof_from_folds(dev, frame, {s: folds[s] for s in spec["split_seeds"]}, name)
        confirm_pred = oof_from_folds(run, frame, {s: folds[s] for s in spec["confirmation_seeds"]}, name)
        cells = score_candidate(frame, folds, a35, b0, {**dev_pred, **confirm_pred}, target,
                                spec["blend_weight"])
        gains = [cells[str(seed)]["gain"] for seed in seeds]
        paired = paired_summary(gains)
        dev_score = float(np.mean([cells[str(seed)]["candidate_score"] for seed in spec["split_seeds"]]))
        expected_result = {"recipe": name, "seed_results": cells, "paired_seed_summary": paired,
            "local_working_gate_met": dev_score >= spec["promotion"]["local_working_gate"],
            "promoted": bool(all(g > 0 for g in gains) and paired["lcb95"] > 0
                             and dev_score >= spec["promotion"]["local_working_gate"])}
        if expected_result != summary["results"][target]:
            raise ValueError(f"Confirmation score or gate mismatch: {target}")
    result = {"status": "passed", "fits": len(events), "prediction_matrices": len(expected),
              "summary_sha256": file_hash(run / "summary.json"),
              "promoted_targets": [target for target, row in summary["results"].items() if row["promoted"]],
              "packages": 0, "uploads": 0}
    write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(Path.cwd(), args.development, args.confirmation, args.report), indent=2))


if __name__ == "__main__":
    main()
