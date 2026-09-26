"""Round2 V6 Stage A: the zero-fit signature screen.

Stage A spends no fits.  It reads the folds 0/1 predictions that the frozen V3.6
batch already recorded for every N trial, and judges each one against the
released column on exactly those rows:

* residual correlation with the released column (lower is better),
* standalone accuracy ratio against the released column (bounded loss),
* the projected gain of a weight-``alpha`` blend chosen on one fold and scored on
  the other, in both directions.

A candidate is admissible when it is decorrelated enough, bounded in accuracy
loss, and projects a positive gain.  Selection then enforces structural
diversity: at most one member per ``(structure, capacity)`` family and pairwise
residual correlation at or below the cap — the V5 lesson was that four training
settings of one family are near-duplicates (rho 0.979-0.998).

Nothing here fits, promotes, packages or uploads.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .data import TARGETS
from .metrics import wmape as _wmape
from .v3_6_sampler import sample_v36
from .v5_library import fold_vector, load_column_reference, load_v5_training_frame
from .v5_resolution import select_alpha
from .v6_spec import V6Spec, load_v6_spec

__all__ = ["StageARecord", "stage_a_screen", "screen_records_for_target", "main"]

DEFAULT_OUTPUT = "local/runs/round2-v6-iron-capacity-networks/stage-a-r1"
V36_FIXED_DIR = "local/runs/round2-v3.6-loss-training-and-numeric-encoding/fixed-r2-final"
V36_DEV_CACHE = "local/runs/round2-v3.6-loss-training-and-numeric-encoding/complete-dev-r2-final"
V5_TIME_FAMILY = "local/runs/round2-v5-error-covariance/time-n-family-r1"
RECORDED_SEED = 42
RECORDED_FOLDS = (0, 1)


def _private_output(root: Path, output: Path) -> Path:
    allowed = (root / "local/runs/round2-v6-iron-capacity-networks").resolve()
    resolved = (output if output.is_absolute() else root / output).resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError(f"V6 output must stay private under {allowed}")
    return resolved


def wmape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return float(_wmape(actual, predicted))


@dataclass
class StageARecord:
    trial_id: str
    target: str
    structure: str
    capacity_name: str
    training_setting: str
    family: str
    source: str
    residual_correlation: float
    single_wmape: float
    base_wmape: float
    accuracy_ratio: float
    projected_gain_score: float
    projected_fold_gains: list[float] = field(default_factory=list)
    projected_alphas: list[float] = field(default_factory=list)
    admissible: bool = False
    reasons: list[str] = field(default_factory=list)
    selected: bool = False
    selection_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "target": self.target,
            "structure": self.structure,
            "capacity_name": self.capacity_name,
            "training_setting": self.training_setting,
            "family": self.family,
            "source": self.source,
            "residual_correlation": float(self.residual_correlation),
            "single_wmape": float(self.single_wmape),
            "base_wmape": float(self.base_wmape),
            "accuracy_ratio": float(self.accuracy_ratio),
            "projected_gain_score": float(self.projected_gain_score),
            "projected_fold_gains": [float(v) for v in self.projected_fold_gains],
            "projected_alphas": [float(v) for v in self.projected_alphas],
            "admissible": bool(self.admissible),
            "reasons": list(self.reasons),
            "selected": bool(self.selected),
            "selection_note": self.selection_note,
        }


def _completed_trials(root: Path) -> set[str]:
    """Trials that already have complete coverage somewhere in the repository."""
    complete: set[str] = set()
    for directory in (root / V36_DEV_CACHE, root / V5_TIME_FAMILY):
        for path in directory.glob("seed-42/pred-v36-s1-N-*.npy"):
            complete.add(path.name[len("pred-"):-len(".npy")])
    return complete


def screen_records_for_target(target: str, actual: np.ndarray, base: np.ndarray,
                              mask: np.ndarray, folds: np.ndarray,
                              candidates: Mapping[str, dict[str, Any]],
                              prediction_dir: Path, spec: V6Spec) -> list[StageARecord]:
    """Judge one target's recorded folds 0/1 predictions against the released column."""
    signature = spec.stage_a
    ratio_max = float(signature["accuracy_ratio_max"])
    correlation_max = float(signature["residual_correlation_max"])
    grid = spec.alpha_grid
    base_wmape = wmape(actual[mask], base[mask])
    records: list[StageARecord] = []
    for trial_id, trial in sorted(candidates.items()):
        path = prediction_dir / f"pred-{trial_id}.npy"
        if not path.is_file():
            continue
        vector = np.load(path).astype(float, copy=False)
        if vector.shape != actual.shape or not np.isfinite(vector[mask]).all():
            continue
        correlation = float(np.corrcoef((actual - base)[mask], (actual - vector)[mask])[0, 1])
        single = wmape(actual[mask], vector[mask])
        ratio = single / base_wmape if base_wmape > 0 else float("inf")
        gains: list[float] = []
        alphas: list[float] = []
        for fit_fold, eval_fold in ((0, 1), (1, 0)):
            fit_mask = folds == fit_fold
            eval_mask = folds == eval_fold
            alpha, _ = select_alpha(actual[fit_mask], base[fit_mask], vector[fit_mask], grid)
            delta = wmape(actual[eval_mask], base[eval_mask]) - wmape(
                actual[eval_mask],
                (1.0 - float(alpha)) * base[eval_mask] + float(alpha) * vector[eval_mask],
            )
            gains.append(float(50.0 * delta))
            alphas.append(float(alpha))
        reasons: list[str] = []
        if not np.isfinite(correlation) or correlation > correlation_max:
            reasons.append("correlation_above_max")
        if ratio > ratio_max:
            reasons.append("accuracy_loss_above_max")
        if float(np.mean(gains)) <= float(signature["projected_nested_fold_gain_min"]):
            reasons.append("projected_gain_not_positive")
        records.append(StageARecord(
            trial_id=str(trial_id),
            target=str(target),
            structure=str(trial["structure"]),
            capacity_name=str(trial["capacity_name"]),
            training_setting=str(trial["training_setting"]),
            family=f"{trial['structure']}|{trial['capacity_name']}",
            source="recorded_folds01",
            residual_correlation=correlation,
            single_wmape=single,
            base_wmape=base_wmape,
            accuracy_ratio=ratio,
            projected_gain_score=float(np.mean(gains)),
            projected_fold_gains=gains,
            projected_alphas=alphas,
            admissible=not reasons,
            reasons=reasons,
        ))
    records.sort(key=lambda record: (-record.projected_gain_score, record.residual_correlation))
    return records


def _residual(vector: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return actual - vector


def select_with_diversity(records: Sequence[StageARecord], vectors: Mapping[str, np.ndarray],
                          actual: np.ndarray, mask: np.ndarray, mutual_max: float,
                          limit: int) -> list[StageARecord]:
    """Greedy selection enforcing *both* diversity constraints.

    A candidate is skipped when its ``(structure, capacity)`` family is already
    represented or when its residual correlates above the cap with an already
    selected candidate — the near-duplicate case V5 measured (rho 0.979-0.998
    among four training settings of one family).
    """
    selected: list[StageARecord] = []
    used_families: set[str] = set()
    for record in records:
        if len(selected) >= int(limit):
            break
        if not record.admissible:
            continue
        if record.family in used_families:
            record.selection_note = "family_already_selected"
            continue
        blocked_by = None
        for other in selected:
            correlation = float(np.corrcoef(_residual(vectors[record.trial_id], actual)[mask],
                                            _residual(vectors[other.trial_id], actual)[mask])[0, 1])
            if correlation > float(mutual_max):
                blocked_by = (other.trial_id, correlation)
                break
        if blocked_by is not None:
            record.selection_note = (
                f"mutual_residual_correlation_above_max:{blocked_by[0]}:{blocked_by[1]:.4f}")
            continue
        selected.append(record)
        used_families.add(record.family)
    return selected


def stage_a_screen(root: Path | str, spec: V6Spec | None = None,
                   output: Path | str = DEFAULT_OUTPUT,
                   limit_per_target: int = 6) -> dict[str, Any]:
    """Run the zero-fit Stage A screen over every N trial with recorded folds 0/1."""
    root = Path(root).resolve()
    spec = spec or load_v6_spec(root)
    started = time.time()
    out = _private_output(root, Path(output))
    train = load_v5_training_frame(root)
    reference = load_column_reference(root, train, spec)  # noqa: SLF001 - reused frozen replay
    trials = {str(t["trial_id"]): t for t in sample_v36(root)}
    complete = _completed_trials(root)
    prediction_dir = root / V36_FIXED_DIR
    folds = fold_vector(root, train, RECORDED_SEED, None)
    mask = np.isin(folds, list(RECORDED_FOLDS))

    payload: dict[str, Any] = {
        "version": spec.version,
        "stage": "stage_a_zero_fit_signature_screen",
        "recorded_seed": RECORDED_SEED,
        "recorded_folds": list(RECORDED_FOLDS),
        "excluded_complete_trials": sorted(complete),
        "targets": {},
        "agent_uploads": 0,
    }
    selection: dict[str, Any] = {}
    for target in TARGETS:
        candidates = {
            trial_id: trial for trial_id, trial in trials.items()
            if str(trial["target"]) == target
            and str(trial.get("line")) == "N"          # only the numeric-encoding line has this space
            and trial_id not in complete
        }
        actual = train[target].to_numpy(dtype=float)
        base = reference.base_for(target, RECORDED_SEED)
        records = screen_records_for_target(target, actual, base, mask, folds,
                                            candidates, prediction_dir, spec)
        vectors = {record.trial_id: np.load(prediction_dir / f"pred-{record.trial_id}.npy")
                   .astype(float, copy=False) for record in records}
        chosen = select_with_diversity(
            records, vectors, actual, mask,
            mutual_max=float(spec.diversity["mutual_residual_correlation_max"]),
            limit=int(limit_per_target),
        )
        pairwise: list[dict[str, Any]] = []
        for i, left in enumerate(chosen):
            for right in chosen[i + 1:]:
                correlation = float(np.corrcoef((actual - vectors[left.trial_id])[mask],
                                                (actual - vectors[right.trial_id])[mask])[0, 1])
                pairwise.append({"left": left.trial_id, "right": right.trial_id,
                                 "mutual_residual_correlation": correlation,
                                 "within_cap": bool(correlation <= float(spec.diversity[
                                     "mutual_residual_correlation_max"]))})
        for record in chosen:
            record.selected = True
        payload["targets"][target] = {
            "recorded_candidate_count": len(candidates),
            "screened_count": len(records),
            "admissible_count": int(sum(1 for record in records if record.admissible)),
            "records": [record.as_dict() for record in records],
            "selected": [record.trial_id for record in chosen],
            "pairwise_mutual_correlation": pairwise,
        }
        selection[target] = chosen

    payload["stage_a_admissible_any"] = bool(any(
        payload["targets"][target]["admissible_count"] > 0 for target in TARGETS))
    payload["stage_a2_unlocked"] = payload["stage_a_admissible_any"]
    payload["status"] = "COMPLETE_NO_PROMOTION"
    payload["elapsed_seconds"] = float(time.time() - started)

    out.mkdir(parents=True, exist_ok=True)
    (out / "stage_a.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    rows = [row for target in TARGETS for row in payload["targets"][target]["records"]]
    with (out / "stage_a.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "target", "trial_id", "structure", "capacity_name", "training_setting", "family",
            "residual_correlation", "single_wmape", "base_wmape", "accuracy_ratio",
            "projected_gain_score", "admissible", "selected", "selection_note", "reasons"],
            extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["reasons"] = ";".join(row["reasons"])
            writer.writerow(row)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--limit-per-target", type=int, default=6)
    args = parser.parse_args(argv)
    payload = stage_a_screen(args.root, output=args.output,
                             limit_per_target=args.limit_per_target)
    summary = {
        "status": payload["status"],
        "excluded_complete_trials": len(payload["excluded_complete_trials"]),
        "stage_a_admissible_any": payload["stage_a_admissible_any"],
        "stage_a2_unlocked": payload["stage_a2_unlocked"],
        "targets": {
            target: {
                "screened": payload["targets"][target]["screened_count"],
                "admissible": payload["targets"][target]["admissible_count"],
                "selected": payload["targets"][target]["selected"],
            } for target in TARGETS
        },
        "elapsed_seconds": payload["elapsed_seconds"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
