#!/usr/bin/env python3
"""Read-only V4 OOF audit. No .fit(), no new split, no test input, no ZIP.

Run inside the original iron environment. Imports only the existing train
loader; reads its frozen fold assignments and V4 coarse-r2 arrays. This script
cannot certify the historic top-level A weights were independently selected.
It treats them as the same descriptive development anchor used by V4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from increment_diagnostic import diagnose_direction

SEEDS = (42, 3407)
TARGETS = ("tap_iron", "tap_time_len")
DEFAULT_SOURCE = "local/runs/round2-v4-mechanism-search/coarse-r2"
DEFAULT_OUTPUT = "local/runs/round2-v4.1-strong-increment/diagnostic-r1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def v4_frame_hash(frame: pd.DataFrame, features: list[str]) -> str:
    # Exact compatibility with v4_run._frame_hash(train, (0,1,2,3,4)).
    h = hashlib.sha256()
    for sample_id in frame.sample_id.astype(str):
        h.update(sample_id.encode("utf-8")); h.update(b"\x00")
    numeric = frame[[*features, *TARGETS]].to_numpy(dtype=float)
    h.update(np.ascontiguousarray(numeric).tobytes())
    h.update(b"0,1,2,3,4")
    return h.hexdigest()


def close(a: float, b: float, label: str) -> None:
    if not np.isclose(a, b, rtol=0.0, atol=1e-9):
        raise ValueError(f"Metric replay mismatch ({label}): {a!r} != {b!r}")


def load_fold_assignment(path: Path, train: pd.DataFrame, seed: int) -> np.ndarray:
    a = pd.read_csv(path, dtype={"sample_id": str, "group_id": str})
    needed = {"sample_id", "group_id", "fold", "seed"}
    if not needed.issubset(a.columns) or a[list(needed)].isna().any().any():
        raise ValueError("Incomplete frozen fold assignment")
    if a.sample_id.duplicated().any() or set(a.sample_id) != set(train.sample_id.astype(str)):
        raise ValueError("Frozen fold sample-ID mismatch")
    if not (a.seed == seed).all() or set(a.fold) != set(range(5)):
        raise ValueError("Invalid frozen fold/seed values")
    if (a.groupby("group_id").fold.nunique() > 1).any():
        raise ValueError("Duplicate group crosses frozen fold boundary")
    return a.set_index("sample_id").loc[train.sample_id.astype(str), "fold"].to_numpy(dtype=np.int64)


def run(root: Path, source_relative: str, output_relative: str) -> dict[str, Any]:
    root = root.resolve()
    source, output = (root/source_relative).resolve(), (root/output_relative).resolve()
    private = (root/"local/runs").resolve()
    if not source.is_relative_to(private) or not output.is_relative_to(private):
        raise ValueError("Both source and output must be beneath this repository's local/runs")
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the source directory")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    must_exist = [source/"environment.json", source/"fit_ledger.jsonl", source/"reference/a_dev_oof.npz"]
    must_exist += [root/"local/runs/round2-v2/comparison-r1"/f"folds-{s}.csv" for s in SEEDS]
    must_exist += [root/"复赛_train/train_samples.csv", root/"复赛_train/train_features.csv"]
    absent = [str(p) for p in must_exist if not p.is_file()]
    if absent:
        raise FileNotFoundError("Missing original training/OOF inputs; nothing trained or written:\n" + "\n".join(absent))

    sys.path.insert(0, str(root/"src"))
    from bf_tap_r2.v2_release import load_v2
    train = load_v2(root/"复赛_train", "train", 2754)
    if not train.sample_id.astype(str).str.startswith("R2S2_TRAIN_").all():
        raise ValueError("Not synthetic_round2_v2 training IDs")
    environment = json.loads((source/"environment.json").read_text(encoding="utf-8"))
    features = environment["feature_order"]
    if len(features) != 21 or len(set(features)) != 21 or set(features) & {"sample_id", *TARGETS}:
        raise ValueError("Unexpected source feature order")
    if v4_frame_hash(train, features) != environment["data_hash"]:
        raise ValueError("Source data hash mismatch, including original row order")
    folds = {}
    for seed in SEEDS:
        path = root/"local/runs/round2-v2/comparison-r1"/f"folds-{seed}.csv"
        folds[seed] = load_fold_assignment(path, train, seed)
        digest = hashlib.sha256(folds[seed].tobytes()).hexdigest()
        if digest != environment["fold_hashes"][str(seed)]:
            raise ValueError(f"Frozen fold hash mismatch for {seed}")
    with np.load(source/"reference/a_dev_oof.npz", allow_pickle=False) as f:
        refs = {key: np.asarray(f[key], dtype=float) for key in f.files}
    for seed in SEEDS:
        for target in TARGETS:
            a = refs[f"{seed}_{target}"]
            if a.shape != (len(train),) or not np.isfinite(a).all():
                raise ValueError("Invalid A OOF reference array")

    rows, blocked, completed = [], [], set()
    hashes = {str(p.relative_to(root)): sha(p) for p in must_exist}
    for line in (source/"fit_ledger.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") != "complete":
            continue
        method, seed, target = str(e["method_id"]), int(e["seed"]), str(e["target"])
        if seed not in SEEDS or target not in TARGETS or not re.fullmatch(r"v4-[FSPJ]-\d{3}", method):
            raise ValueError("Unexpected method, seed or target identity")
        key = (method, seed)
        if key in completed:
            raise ValueError(f"Repeated completed source event: {key}")
        completed.add(key)
        if e.get("status") == "blocked":
            blocked.append({"method_id": method, "seed": seed, "reason": e.get("reason")})
            continue
        if e.get("status") != "available" or list(e.get("folds", [])) != [0, 1]:
            raise ValueError("Unexpected V4 status or coverage")
        if e.get("data_hash") != environment["data_hash"] or e.get("fold_hash") != environment["fold_hashes"][str(seed)]:
            raise ValueError("Source ledger/identity mismatch")
        path = source/"oof"/f"{method}-seed{seed}.npy"
        c = np.load(path, allow_pickle=False)
        mask = np.isin(folds[seed], [0, 1])
        if c.shape != (len(train),) or not np.isfinite(c[mask]).all() or not np.isnan(c[~mask]).all():
            raise ValueError(f"Unexpected coarse OOF coverage: {path}")
        y = train.loc[mask, target].to_numpy(dtype=float)
        diag = diagnose_direction(y, refs[f"{seed}_{target}"][mask], c[mask])
        old = e["metrics"]
        close(diag["baseline_target_wmape"], old["baseline_target_wmape"], "A WMAPE")
        close(diag["single_package_delta"], old["package_delta_single"], "single delta")
        close(diag["half_blend_package_delta"], old["package_delta_equal_blend"], "half blend delta")
        rows.append({"method_id": method, "family": e["family"], "target": target,
                     "mechanism": e["mechanism"], "seed": seed, **diag})
        hashes[str(path.relative_to(root))] = sha(path)
    if len(rows) != 76 or len(blocked) != 4 or len(completed) != 80:
        raise ValueError("Expected the complete coarse-r2 ledger: 76 available and 4 blocked seed-unit events")
    table = pd.DataFrame(rows)
    groups = []
    for (method, target), part in table.groupby(["method_id", "target"], sort=True):
        if set(part.seed) != set(SEEDS) or len(part) != 2:
            raise ValueError("Incomplete two-seed diagnostic")
        gain = float(part.oracle_package_delta_same_labels.mean())
        both = bool((part.oracle_package_delta_same_labels > 0).all())
        groups.append({"method_id": method, "target": target,
                       "single_mean_delta": float(part.single_package_delta.mean()),
                       "half_blend_mean_delta": float(part.half_blend_package_delta.mean()),
                       "oracle_mean_delta_SAME_LABELS": gain,
                       "oracle_positive_both_seeds": both,
                       "worth_nested_retest_NOT_PROMOTION": bool(both and gain >= 0.02)})
    summary = pd.DataFrame(groups).sort_values(["oracle_mean_delta_SAME_LABELS", "method_id"], ascending=[False, True])
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    manifest = {
        "status": "DESCRIPTIVE_ONLY_NOT_VALIDATION_NOT_SUBMISSION",
        "source": str(source.relative_to(root)), "source_hashes": hashes,
        "source_data_hash": environment["data_hash"], "git_head_at_diagnosis": head,
        "available_method_target_units": len(summary), "available_seed_unit_events": len(rows),
        "blocked_seed_unit_events": blocked, "model_training_fits": 0,
        "same_label_scalar_optimizations": len(rows), "platform_uploads": 0,
        "submission_packages": 0, "held_out_seeds_consumed": [],
        "warning": "Oracle alpha uses the SAME OOF labels. It is not unbiased validation, a deployed coefficient, or evidence of platform improvement. Full nested re-evaluation is required.",
        "identity_limitation": "Arrays inherit V4 positional identity, checked against V4 data/fold hashes and replayed metrics; the source npy arrays do not carry embedded sample IDs.",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "diagnostic_source_sha256": sha(Path(__file__)),
        "core_source_sha256": sha(Path(__file__).with_name("increment_diagnostic.py")),
    }
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output/"direction_diagnostics.csv", index=False)
    summary.to_csv(output/"diagnostic_shortlist_NOT_PROMOTION.csv", index=False)
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"output": str(output), "status": manifest["status"], "model_training_fits": 0,
            "available_seed_unit_events": len(rows), "blocked_seed_unit_events": len(blocked)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = run(args.root, args.source, args.output)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(2, f"Audit stopped: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
