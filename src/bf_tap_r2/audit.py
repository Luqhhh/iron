"""P0 only: descriptive statistics, no model fitting or preliminary data reads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

from .data import FEATURES, TARGETS, FILES, SUBMISSION_COLUMNS, load_config, load_snapshot, resolve_file


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def distribution(series: pd.Series) -> dict:
    finite = series[np.isfinite(series)]
    return {"count": len(series), "finite_count": len(finite),
            "quantiles": {str(q): (float(finite.quantile(q)) if len(finite) else None)
                          for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)},
            "unique_finite_values": int(finite.nunique())}


def duplicates(frame: pd.DataFrame, columns: list[str]) -> dict:
    sizes = frame.groupby(columns, dropna=False).size()
    repeated = sizes[sizes > 1]
    return {"groups": len(repeated), "rows_in_groups": int(repeated.sum()),
            "excess_rows": int((repeated - 1).sum())}


def near_pairs(train: pd.DataFrame, test: pd.DataFrame, threshold: float, block: int) -> dict:
    # Label-free audit scale only; never reused as a CV preprocessor.
    x = pd.concat([train, test], ignore_index=True)
    a = train[list(FEATURES)]
    scale = (a.quantile(.75) - a.quantile(.25)).to_numpy()
    span = (a.max() - a.min()).to_numpy()
    scale = np.where(scale > 0, scale, np.where(span > 0, span, 1.0))
    values = x[list(FEATURES)].to_numpy(dtype=float) / scale
    pairs = []
    for start in range(0, len(x), block):
        distances = np.max(np.abs(values[start:start+block, None] - values[None]), axis=2)
        for row, col in np.argwhere(distances <= threshold):
            row = int(row) + start
            col = int(col)
            if row < col:
                pairs.append({"left": str(x.sample_id.iloc[row]), "right": str(x.sample_id.iloc[col]),
                              "distance": float(distances[row-start, col]),
                              "same_spout": bool(x.spout_no.iloc[row] == x.spout_no.iloc[col]),
                              "scope": "train" if col < len(train) else ("test" if row >= len(train) else "cross")})
    return {"definition": "max absolute numeric difference / train IQR <= threshold; zero IQR uses train range then 1",
            "threshold": threshold, "spout_restriction": "none; same_spout recorded per pair",
            "scale": dict(zip(FEATURES, scale.tolist())), "pairs": pairs,
            "counts": {scope: sum(p["scope"] == scope for p in pairs) for scope in ("train", "test", "cross")},
            "interpretation": "screening only; absence does not establish independence"}


def run(root: Path, config: Path, output: Path) -> dict:
    if not output.resolve().is_relative_to(root.resolve() / "local/runs/round2-v0.1"):
        raise ValueError("Audit evidence must stay under local/runs/round2-v0.1")
    output.mkdir(parents=True, exist_ok=False)
    try:
        cfg = load_config(config)
        paths = [*FILES.values(), *cfg["support_files"]]
        manifest = {"version": cfg["version"], "created_at": datetime.now(timezone.utc).isoformat(),
                    "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    "files": {p: {"sha256": digest(resolve_file(root, p)), "bytes": resolve_file(root, p).stat().st_size} for p in paths},
                    "numeric_features": list(FEATURES), "categorical_features": ["spout_no"],
                    "targets": list(TARGETS),
                    "implementation_sha256": {str(p.relative_to(root)): digest(p) for p in sorted((root / "src/bf_tap_r2").glob("*.py"))},
                    "config_sha256": {str(p.relative_to(root)): digest(p) for p in sorted(config.parent.glob("*.yaml"))}}
        write_json(output / "data_manifest.json", manifest)
        train, ti = load_snapshot(root, cfg, "train")
        test, vi = load_snapshot(root, cfg, "test")
        template = pd.read_csv(resolve_file(root, cfg["template"]))
        if tuple(template.columns) != SUBMISSION_COLUMNS or len(template):
            raise ValueError("Expected the documented header-only result template")
        audit = {"version": cfg["version"], "manifest_sha256": digest(output / "data_manifest.json"),
                 "integrity": {"train": ti, "test": vi}, "targets": {}, "features": {}, "shift": {}}
        blockers = []
        for stage, frame in (("train", train), ("test", test)):
            nums = frame.drop(columns="sample_id")
            audit["integrity"][stage].update({"missing": frame.isna().sum().to_dict(),
                "infinite": np.isinf(nums).sum().to_dict()})
            if frame.isna().any().any() or not np.isfinite(nums).all().all():
                blockers.append(f"{stage}: missing/nonfinite values need an explicit processing rule")
            if not ((frame.spout_no.dropna() % 1) == 0).all():
                blockers.append(f"{stage}: noninteger spout_no")
            audit["features"][stage] = {
                "constant_columns": [c for c in FEATURES if frame[c].nunique(dropna=False) <= 1],
                "duplicate_numeric_vectors": duplicates(frame, list(FEATURES)),
                "duplicate_model_inputs": duplicates(frame, ["spout_no", *FEATURES])}
        for name in TARGETS:
            groups = {"all": train, **{f"spout_{k}": v for k, v in train.groupby("spout_no")}}
            audit["targets"][name] = {}
            for key, frame in groups.items():
                s = frame[name]
                counts = s.value_counts()
                audit["targets"][name][key] = {**distribution(s),
                    "repeated_values": [{"value": float(v), "count": int(n)} for v, n in counts.items() if n > 1 and np.isfinite(v)],
                    "minimum_count": int(s.eq(s.min()).sum()), "maximum_count": int(s.eq(s.max()).sum()),
                    "count_61_53": int(s.eq(61.53).sum()) if name == "tap_time_len" else None}
        corr = train[list(FEATURES)].corr()
        audit["features"]["train_high_correlations"] = [
            {"left": a, "right": b, "pearson": float(corr.loc[a, b])}
            for i, a in enumerate(FEATURES) for b in FEATURES[i+1:]
            if abs(corr.loc[a, b]) >= cfg["audit"]["absolute_pearson_threshold"]]
        audit["features"]["absolute_pearson_threshold"] = cfg["audit"]["absolute_pearson_threshold"]
        for stage, frame in (("train", train), ("test", test)):
            audit["shift"][f"{stage}_spout_counts"] = {str(k): int(v) for k, v in frame.spout_no.value_counts().items()}
            audit["shift"][f"{stage}_spout_proportions"] = {str(k): float(v) for k, v in frame.spout_no.value_counts(normalize=True).items()}
        audit["shift"]["unseen_test_spouts"] = sorted(set(test.spout_no.dropna()) - set(train.spout_no.dropna()))
        for name in FEATURES:
            audit["shift"][name] = {"train": distribution(train[name]), "test": distribution(test[name]),
                "test_below_train_min_fraction": float((test[name] < train[name].min()).mean()),
                "test_above_train_max_fraction": float((test[name] > train[name].max()).mean())}
        if not blockers:
            audit["features"]["near_duplicates"] = near_pairs(train, test,
                cfg["audit"]["near_duplicate_max_train_iqr_distance"], cfg["audit"]["distance_block_rows"])
        audit["training_data_blockers"] = blockers
        audit["data_gate"] = "pass" if not blockers else "blocked"
        audit["platform_rules"] = yaml.safe_load((config.parent / "metric_contract.yaml").read_text(encoding="utf-8"))["platform"]
        audit["model_fits"] = 0
        audit["platform_submissions"] = 0
        for p, identity in manifest["files"].items():
            if digest(resolve_file(root, p)) != identity["sha256"]:
                raise ValueError(f"Input changed during audit: {p}")
        write_json(output / "data_audit.json", audit)
        return audit
    except Exception as exc:
        write_json(output / "FAILED.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root, args.root / "configs/round2_v0_1/data.yaml", args.output)
    print(json.dumps({"data_gate": report["data_gate"], "output": str(args.output), "model_fits": 0}))


if __name__ == "__main__":
    main()
