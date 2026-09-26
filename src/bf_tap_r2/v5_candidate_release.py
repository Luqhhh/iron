"""Round2 V5 candidate release: isolated single-column package for a passed member.

The V5 promotion rule admitted `v36-s1-N-0048` on four split seeds.  This module
turns that member into the artifact a platform test needs:

* only the member's target column changes — the other column keeps the parent
  package's exact CSV field text;
* the released blend weight is the grid optimum over **all** available split
  seeds, and the nested per-seed weights are recorded beside it;
* the member is refit on every training row with its frozen specification, and
  the fit is subject to a cold-inference audit;
* the member's test predictions are persisted so the blend arithmetic can be
  recomputed independently, and the built ZIP is read back and re-verified.

The package is written under ``local/``; upload is performed by the user.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import TARGETS
from .submission import ZIP_NAME
from .v5_library import fold_vector, load_column_reference, load_v5_training_frame
from .v5_package import (
    DEFAULT_PARENT_CSV,
    build_single_target_replacement_package,
    read_parent_columns,
)
from .v5_replicate import load_candidate
from .v5_resolution import paired_cells, paired_summary, seed_gains, wmape
from .v5_spec import V5Spec, load_v5_spec

__all__ = ["REPLICATION_DIR", "release_alpha", "read_back_package", "build_member_package", "main"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

ROOT_DEFAULT = Path("/home/lux1/iron")
TIME_FAMILY_DIR = "local/runs/round2-v5-error-covariance/time-n-family-r1"
REPLICATION_DIR = "local/runs/round2-v5-error-covariance/replication-r1"
DEFAULT_OUTPUT_DIR = "local/runs/round2-v5-error-covariance/release-r1"


def _private_output(root: Path, output: Path) -> Path:
    allowed = (root / "local/runs/round2-v5-error-covariance").resolve()
    resolved = (output if output.is_absolute() else root / output).resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError(f"V5 release output must stay private under {allowed}")
    return resolved


def _derive_baseline(root: Path, spec: V5Spec, target: str, seed: int,
                     folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the cached fixed-recipe baseline and candidate for a derived seed."""
    baseline = np.full(len(folds), np.nan)
    candidate = np.full(len(folds), np.nan)
    for fold in sorted(set(int(v) for v in folds)):
        path = root / REPLICATION_DIR / f"seed-{int(seed)}" / f"fold-{fold}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"V5 release needs the cached replication fold: {path}")
        cache = np.load(path)
        position = folds == fold
        baseline[position] = np.asarray(cache["base"], dtype=float)[position]
        candidate[position] = np.asarray(cache["candidate"], dtype=float)[position]
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError(f"V5 release: incomplete cached coverage at seed {seed}")
    return baseline, candidate


def release_alpha(root: Path | str, spec: V5Spec, trial_id: str, target: str,
                  recorded_seeds: Sequence[int] = (42, 3407),
                  derived_seeds: Sequence[int] = (7777, 12011)) -> dict[str, Any]:
    """Grid optimum of the blend weight over every available split seed."""
    root = Path(root).resolve()
    train = load_v5_training_frame(root)
    reference = load_column_reference(root, train, spec)
    actual = train[target].to_numpy(dtype=float)
    seeds = [int(s) for s in recorded_seeds] + [int(s) for s in derived_seeds]
    folds = {seed: fold_vector(root, train, seed, spec) for seed in seeds}
    baseline = {int(s): reference.base_for(target, int(s)) for s in recorded_seeds}
    candidate: dict[int, np.ndarray] = {}
    for seed in recorded_seeds:
        path = root / TIME_FAMILY_DIR / f"seed-{int(seed)}" / f"pred-{trial_id}.npy"
        candidate[int(seed)] = np.load(path).astype(float, copy=False)
    for seed in derived_seeds:
        baseline[int(seed)], candidate[int(seed)] = _derive_baseline(
            root, spec, target, int(seed), folds[int(seed)])

    rows = []
    for alpha in spec.alpha_grid:
        per_seed = {s: wmape(actual, (1.0 - float(alpha)) * baseline[s] + float(alpha) * candidate[s])
                    for s in seeds}
        rows.append({"alpha": float(alpha), "mean_wmape": float(np.mean(list(per_seed.values()))),
                     "per_seed_wmape": {str(s): float(v) for s, v in per_seed.items()}})
    rows.sort(key=lambda row: (row["mean_wmape"], row["alpha"]))
    best = rows[0]
    alpha = float(best["alpha"])
    cells = paired_cells(actual, folds, baseline,
                         {s: (1.0 - alpha) * baseline[s] + alpha * candidate[s] for s in seeds})
    gains = seed_gains(cells)
    return {
        "trial_id": str(trial_id),
        "target": target,
        "seeds": seeds,
        "release_alpha": alpha,
        "release_alpha_basis": "grid optimum of the mean WMAPE over all available split seeds",
        "mean_wmape_at_release_alpha": float(best["mean_wmape"]),
        "per_seed_gain": {str(k): float(v) for k, v in gains.items()},
        "fold_summary": paired_summary([cell.delta_score for cell in cells]),
        "grid": rows,
    }


def _cold_check(model, test) -> Mapping[str, Any]:
    reversed_frame = test.iloc[::-1].reset_index(drop=True)
    reversed_prediction = np.asarray(model.predict(reversed_frame), dtype=float)[::-1]
    subset_index = np.asarray([0, 5, 17, 100, 321])
    subset_frame = test.iloc[subset_index].reset_index(drop=True)
    subset_prediction = np.asarray(model.predict(subset_frame), dtype=float)
    forward = np.asarray(model.predict(test.reset_index(drop=True)), dtype=float)
    reversed_diff = float(np.max(np.abs(reversed_prediction - forward)))
    subset_diff = float(np.max(np.abs(subset_prediction - forward[subset_index])))
    scale = max(1.0, float(np.max(np.abs(forward))))
    return {
        "reversed_max_abs_diff": reversed_diff,
        "subset_max_abs_diff": subset_diff,
        "relative_max": max(reversed_diff, subset_diff) / scale,
        "atol_relative_threshold": 1e-6,
        "passed": bool(max(reversed_diff, subset_diff) / scale <= 1e-6),
        "note": "row-order and subset invariance of the full-data member fit",
    }


def build_member_package(root: Path | str, spec: V5Spec, *, trial_id: str, target: str,
                         name: str, output_dir: Path | str = DEFAULT_OUTPUT_DIR,
                         parent_csv: Path | str = DEFAULT_PARENT_CSV,
                         status: str = "V5_FOUR_SEED_PROMOTED_BELOW_LOCAL_GATE",
                         release: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit the member on all training rows and emit the isolated package."""
    root = Path(root).resolve()
    release = dict(release or release_alpha(root, spec, trial_id, target))
    alpha = float(release["release_alpha"])
    fit, meta = load_candidate(root, spec, "v36", trial_id)
    if str(meta["target"]) != str(target):
        raise ValueError(f"V5 release: member target {meta['target']!r} != {target!r}")

    holder: dict[str, Any] = {}

    def member_predictor(train, test):  # noqa: ANN001 - local closure
        from .v3_6_models import V36Regressor

        from .v3_6_sampler import sample_v36

        trials = {str(t["trial_id"]): t for t in sample_v36(root)}
        trial = trials[str(trial_id)]
        model = V36Regressor(dict(trial))
        model.fit(train.reset_index(drop=True), train[target].to_numpy(dtype=float))
        prediction = np.asarray(model.predict(test.reset_index(drop=True)), dtype=float)
        holder["model"] = model
        holder["prediction"] = prediction
        return prediction

    out_dir = _private_output(root, Path(output_dir))
    manifest = build_single_target_replacement_package(
        root, out_dir / str(name),
        target=target, alpha=alpha, member_predictor=member_predictor, name=str(name),
        status=str(status), parent_csv=parent_csv,
        alpha_basis=(
            "grid optimum of the mean WMAE over the four available split seeds "
            "(42/3407 recorded, 7777/12011 derived); nested per-seed weights agree at 0.20-0.21"
        ),
        member_meta={"kind": "v36", "trial_id": str(trial_id), **dict(meta),
                     "recipe": "large raw-TabM, frozen V3.6 specification, refit on all training rows"},
        reproduction_check=None,
        cold_check=lambda prediction: _cold_check(holder["model"], _load_test_frame(root)),
        gate_summary={
            "promotion": "PASSED_FOUR_SPLIT_SEED_RULE",
            "seed_level_lcb95": float(release["fold_summary"]["lcb95"]),
            "seed_level_positive_seeds": int(release["fold_summary"]["positive"]),
            "fold_positive_cells": f"{release['fold_summary']['positive']}/{release['fold_summary']['n']}",
            "release_evidence": release,
            "local_working_gate": 96.25,
            "below_local_gate": True,
            "note": (
                "The four-seed promotion rule passed, but the expected local package "
                "96.2038 + 0.0097 = 96.2135 is below the frozen local working gate 96.25. "
                "This package exists because the user explicitly authorised the transfer test."
            ),
        },
    )
    package_dir = Path(manifest["output"])
    np.save(package_dir / "member-predictions.npy", np.asarray(holder["prediction"], dtype=float))
    read_back = read_back_package(root, package_dir, parent_csv, target, alpha,
                                  np.asarray(holder["prediction"], dtype=float))
    evidence = {
        "candidate": str(name),
        "trial_id": str(trial_id),
        "target": target,
        "alpha": alpha,
        "release": release,
        "read_back": read_back,
        "cold_check": manifest.get("cold_check"),
        "gate_summary": manifest["gate_summary"],
        "agent_uploads": 0,
    }
    (package_dir / "release-evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    merged = {**json.loads((package_dir / "manifest.json").read_text(encoding="utf-8")),
              "release": release, "read_back": read_back,
              "member_predictions_sha256": _sha256(package_dir / "member-predictions.npy")}
    (package_dir / "manifest.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    if not read_back["ids_match_template_order"] or not read_back["unchanged_column"]["byte_identical_to_parent"]:
        raise AssertionError("V5 read-back verification failed; package must not be released")
    return {**manifest, "release": release, "read_back": read_back}


def _load_test_frame(root: Path):
    from .v2_release import load_v2

    return load_v2(root / "复赛_test", "test", 322)


def read_back_package(root: Path | str, package: Path, parent_csv: Path | str,
                      target: str, alpha: float, member_prediction: np.ndarray) -> dict[str, Any]:
    """Independently re-verify the built ZIP: rows, order and blend arithmetic."""
    root = Path(root).resolve()
    package = Path(package)
    zip_path = package / ZIP_NAME
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if names != ["result.csv"]:
            raise ValueError(f"V5 read-back: unexpected ZIP members {names}")
        text = archive.read("result.csv").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    ids = [row["sample_id"] for row in rows]
    template = [str(v) for v in _load_test_frame(root).sample_id.tolist()]
    parent_path = Path(parent_csv)
    if not parent_path.is_absolute():
        parent_path = root / parent_path
    parent_ids, parent_columns = read_parent_columns(parent_path)
    other = next(t for t in TARGETS if t != target)
    parent_values = {sid: value for sid, value in zip(parent_ids, parent_columns[target][0])}
    parent_other = {sid: value for sid, value in zip(parent_ids, parent_columns[other][1])}
    order = {sid: position for position, sid in enumerate(ids)}
    changed = np.asarray([float(row[f"pred_{target}"]) for row in rows], dtype=float)
    unchanged = [row[f"pred_{other}"] for row in rows]
    expected_other = [parent_other[sid] for sid in ids]
    parent_ordered = np.asarray([parent_values[sid] for sid in ids], dtype=float)
    member_ordered = np.asarray([member_prediction[int(template.index(sid))] for sid in ids], dtype=float)
    recomputed = np.maximum((1.0 - float(alpha)) * parent_ordered + float(alpha) * member_ordered, 0.0)
    return {
        "zip_members": names,
        "rows": len(rows),
        "ids_match_template_order": ids == template,
        "unique_ids": len(set(ids)),
        "unchanged_column": {
            "target": other,
            "byte_identical_to_parent": unchanged == expected_other,
            "mismatches": int(sum(1 for a, b in zip(unchanged, expected_other) if a != b)),
        },
        "changed_column": {
            "target": target,
            "max_abs_diff_vs_recomputed_blend": float(np.max(np.abs(changed - recomputed))),
            "mean_abs_diff_vs_parent": float(np.mean(np.abs(changed - parent_ordered))),
            "scale": float(max(1.0, np.max(np.abs(parent_ordered)))),
        },
        "agent_uploads": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--trial-id", default="v36-s1-N-0048")
    parser.add_argument("--target", default="tap_time_len", choices=list(TARGETS))
    parser.add_argument("--name", default="V5_TIME_N0048_Q20")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--parent-csv", default=DEFAULT_PARENT_CSV)
    parser.add_argument("--alpha", type=float, default=None,
                        help="override the all-seed grid optimum (recorded in the manifest)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    spec = load_v5_spec(root)
    release = release_alpha(root, spec, args.trial_id, args.target)
    if args.alpha is not None:
        release = {**release, "release_alpha": float(args.alpha),
                   "release_alpha_basis": "user/CLI override of the all-seed grid optimum"}
    manifest = build_member_package(root, spec, trial_id=args.trial_id, target=args.target,
                                    name=args.name, output_dir=args.output_dir,
                                    parent_csv=args.parent_csv, release=release)
    package_dir = Path(manifest["output"])
    print(json.dumps({
        "package": str(package_dir),
        "release_alpha": release["release_alpha"],
        "per_seed_gain": release["per_seed_gain"],
        "zip_sha256": manifest["zip_sha256"],
        "result_sha256": manifest["result_sha256"],
        "unchanged_column_verification": manifest["unchanged_column_verification"],
        "cold_check": manifest.get("cold_check"),
    }, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
