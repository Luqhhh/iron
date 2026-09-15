"""Development-only replay of V1 plus state-local structural calibration."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from bf_tap.artifacts import atomic_write_json, runtime_environment, stable_digest
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.features import build_features
from bf_tap.metrics import score_predictions
from bf_tap.models.baseline import DualTargetBaseline
from bf_tap.offline import _load_process_sources
from bf_tap.optimization.config import Candidate, load_experiment, load_feature_selection
from bf_tap.optimization.features import select_candidate_features
from bf_tap.optimization.final_lifecycle import algorithm, component_features, eligible
from bf_tap.optimization.history_stable import stable_features
from bf_tap.optimization.process_change import add_process_change_features
from bf_tap.optimization.rate_model import RateModel
from bf_tap.optimization.snapshot_ensemble import blend
from bf_tap.optimization.state_analog import StateAlphaCalibrator, StateAnalogCalibrator
from bf_tap.optimization.structural import INPUT, PRED, apply_correction, fit_correction
from bf_tap.optimization.run import _load_inputs
from bf_tap.protection import load_protection_policy

TARGETS = ["tap_iron", "tap_time_len"]
META = ["sample_id", "tap_no", "spout_no", "reference_time"]
COMPONENTS = ("E09_PROCESS_CHANGE_E02", "E04")


def stamp(month: int) -> pd.Timestamp:
    return pd.Timestamp(f"2024-{month:02d}-01", tz="Asia/Shanghai")


class DevelopmentContext:
    def __init__(self, data_config: Path):
        self.registration = load_yaml("configs/optimization_v0_16/experiment.yaml")
        self.access = load_yaml("configs/optimization_v0_16/access_scope.yaml")
        self.paths = load_yaml(data_config)["paths"]
        self.algorithm = algorithm()
        protection = load_protection_policy("configs/protection.yaml")
        requested_end = pd.Timestamp(
            self.access["development_label_end_exclusive"]
        )
        (
            self.labels,
            self.history,
            self.operation,
            self.burden,
            self.consistency,
        ) = _load_inputs(
            data_cfg={"paths": self.paths},
            semantic_cfg=self.algorithm["semantic"],
            feature_cfg=self.algorithm["features"],
            protection=protection,
            requested_end=requested_end,
        )
        if (self.labels.reference_time >= requested_end).any():
            raise ContractError("v0.16 development loaded a protected label")
        _, candidates = load_experiment("configs/optimization_v0_2/experiment.yaml")
        self.candidates = {candidate.id: candidate for candidate in candidates}
        self.selection = load_feature_selection("configs/optimization_v0_2/features.yaml")
        self.cache: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, pd.DataFrame]]] = {}
        self.models: dict[int, tuple[dict[str, DualTargetBaseline], RateModel]] = {}

    def features(
        self, samples: pd.DataFrame, cutoff: pd.Timestamp
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        key = (
            str(cutoff),
            stable_digest(samples[META].astype(str).to_dict("records")),
        )
        if key in self.cache:
            return self.cache[key]
        built = build_features(
            samples,
            operation=self.operation,
            burden=self.burden,
            history=self.history,
            fit_cutoff=cutoff,
            config=self.algorithm["features"],
        )
        full = add_process_change_features(
            built.X,
            self.selection["process_change"],
            baseline_value_columns=self.algorithm["features"]["operation"][
                "value_columns"
            ],
        )
        component_frames = {
            name: stable_features(
                select_candidate_features(
                    full, self.candidates[name], self.selection
                ),
                samples,
                self.history,
                cutoff,
                "R2",
            )
            for name in COMPONENTS
        }
        self.cache[key] = (full, component_frames)
        return full, component_frames

    def fit(self, month: int) -> None:
        cutoff = stamp(month)
        train = eligible(self.labels, cutoff)
        _, matrices = self.features(train, cutoff)
        models: dict[str, DualTargetBaseline] = {}
        for name in COMPONENTS:
            model = DualTargetBaseline(
                self.algorithm["baseline"]["parameters"], ("spout_no",)
            )
            model.fit(matrices[name], train[TARGETS])
            models[name] = model
        rate = RateModel().fit(
            matrices["E09_PROCESS_CHANGE_E02"],
            train.tap_iron,
            train.tap_time_len,
            self.algorithm["baseline"]["parameters"],
        )
        self.models[month] = (models, rate)

    def predict(
        self, month: int, samples: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cutoff = stamp(month)
        full, matrices = self.features(samples, cutoff)
        models, rate = self.models[month]
        component_predictions: dict[str, pd.DataFrame] = {}
        for name in COMPONENTS:
            values = models[name].predict_raw(matrices[name]).clip(lower=0.0)
            values.insert(0, "sample_id", samples.sample_id.astype(str).to_numpy())
            component_predictions[name] = values
        base = blend(
            component_predictions["E09_PROCESS_CHANGE_E02"],
            component_predictions["E04"],
            0.8,
        ).set_index("sample_id").loc[samples.sample_id.astype(str)].reset_index()
        predicted_rate = rate.predict(matrices["E09_PROCESS_CHANGE_E02"])
        base["pred_rate"] = predicted_rate
        states = self.state_frame(
            full,
            matrices["E09_PROCESS_CHANGE_E02"],
            base,
            samples,
            cutoff,
        ).reset_index(drop=True)
        return base[INPUT].reset_index(drop=True), states

    def state_frame(
        self,
        full: pd.DataFrame,
        stable_e09: pd.DataFrame,
        predictions: pd.DataFrame,
        samples: pd.DataFrame,
        cutoff: pd.Timestamp,
    ) -> pd.DataFrame:
        config = self.registration["state"]
        state = pd.DataFrame(index=samples.index)
        state["state__spout_2"] = (
            samples.spout_no.astype(str).to_numpy() == "2"
        ).astype(float)
        for name in ("hour_sin", "hour_cos", "weekday"):
            state[f"state__{name}"] = full[name].to_numpy(dtype=float)
        for value in config["operation_values"]:
            names = {
                "latest": f"operation__{value}__latest",
                "6h_mean": f"operation__{value}__6h__mean",
                "24h_mean": f"operation__{value}__24h__mean",
                "latest_minus_24h_mean": f"process_change__{value}__latest_minus_24h_mean",
            }
            for representation in config["operation_representations"]:
                source = names[representation]
                state[f"state__{source}"] = full[source].to_numpy(dtype=float)
        for value in config["burden_values"]:
            source = f"burden__{value}__latest"
            state[f"state__{source}"] = full[source].to_numpy(dtype=float)
        for group in ("all", "spout"):
            for target in TARGETS:
                for window in config["history_windows"]:
                    source = f"history__{group}__{target}__last{window}_median"
                    state[f"state__{source}"] = stable_e09[source].to_numpy(dtype=float)
        for column in ("pred_tap_iron", "pred_tap_time_len", "pred_rate"):
            state[f"state__{column}"] = predictions[column].to_numpy(dtype=float)
        state["state__horizon_days"] = (
            samples.reference_time - cutoff
        ).dt.total_seconds().to_numpy() / 86400.0
        return state


def prediction_metrics(actual: pd.DataFrame, predicted: pd.DataFrame) -> dict:
    overall = score_predictions(actual, predicted)
    monthly = {}
    months = actual.reference_time.dt.strftime("%Y-%m")
    for month in sorted(months.unique()):
        part = actual.loc[months == month]
        pred = predicted.loc[predicted.sample_id.isin(part.sample_id)]
        monthly[month] = score_predictions(part, pred)
    return {"overall": overall, "by_month": monthly}


def oof_row(
    ctx: DevelopmentContext,
    month: int,
    samples: pd.DataFrame,
    predictions: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    cutoff = stamp(month)
    train = eligible(ctx.labels, cutoff)
    history = ctx.history.loc[ctx.history.available_at <= cutoff]
    row = samples[
        ["sample_id", "reference_time", "spout_no", *TARGETS, "label_available_at"]
    ].copy()
    row = row.merge(predictions, on="sample_id", validate="one_to_one")
    state = states.copy()
    state.insert(0, "sample_id", samples.sample_id.astype(str).to_numpy())
    row = row.merge(state, on="sample_id", validate="one_to_one")
    row["fold_cutoff"] = cutoff
    row["train_reference_max"] = train.reference_time.max()
    row["train_available_max"] = train.label_available_at.max()
    row["history_available_max"] = history.available_at.max()
    return row


def run(data_config: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    ctx = DevelopmentContext(data_config)
    registration = ctx.registration
    state_columns: tuple[str, ...] | None = None
    oof_parts = []
    fit_count = 0
    for month in registration["development_origins"]:
        ctx.fit(month)
        fit_count += 5
        evaluation = ctx.labels.loc[
            (ctx.labels.reference_time >= stamp(month))
            & (ctx.labels.reference_time < stamp(month + 1))
        ].copy()
        predictions, states = ctx.predict(month, evaluation)
        if state_columns is None:
            state_columns = tuple(states.columns)
        elif tuple(states.columns) != state_columns:
            raise ContractError("state feature schema changed between origins")
        oof_parts.append(oof_row(ctx, month, evaluation, predictions, states))
        print(
            f"v0.16 origin {month}: {len(evaluation)} causal OOF rows; "
            f"target fits={fit_count}",
            flush=True,
        )
    if state_columns is None:
        raise ContractError("v0.16 produced no state columns")
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(output / "causal_oof.csv", index=False)

    results: dict[str, dict] = {}
    diagnostic_rows = []
    prediction_files = output / "predictions"
    prediction_files.mkdir()
    for fold, spec in registration["evaluation_folds"].items():
        origin, end_month = int(spec["origin"]), int(spec["end_month"])
        evaluation = ctx.labels.loc[
            (ctx.labels.reference_time >= stamp(origin))
            & (ctx.labels.reference_time < stamp(end_month))
        ].copy()
        base, states = ctx.predict(origin, evaluation)
        alpha, _ = fit_correction(oof, stamp(origin), minimum=100)
        global_prediction = apply_correction(base, alpha)
        candidates = {
            "R2_REPLAY": base[["sample_id", *PRED]],
            "V1_GLOBAL_STRUCTURAL_REPLAY": global_prediction,
        }
        fold_diagnostics = {"global_alpha": alpha, "analogs": {}}
        for settings in registration["calibrators"]:
            calibrator = StateAnalogCalibrator(
                state_columns,
                neighbors=int(settings["neighbors"]),
                prior_strength=float(settings["prior_strength"]),
                ood_quantile=float(settings["ood_quantile"]),
            ).fit(oof, stamp(origin))
            prediction, diagnostics = calibrator.predict(base, states)
            candidate = f"V9_{settings['id']}"
            candidates[candidate] = prediction
            fold_diagnostics["analogs"][candidate] = {
                "global_alpha": calibrator.global_alpha_.tolist(),
                "reference_radius": calibrator.reference_radius_,
                "reliability_quantiles": dict(
                    zip(
                        ("p05", "p25", "median", "p75", "p95"),
                        np.quantile(
                            diagnostics.reliability, [0.05, 0.25, 0.5, 0.75, 0.95]
                        ).tolist(),
                    )
                ),
                "alpha_iron_quantiles": np.quantile(
                    diagnostics.alpha[:, 0], [0.05, 0.5, 0.95]
                ).tolist(),
                "alpha_time_quantiles": np.quantile(
                    diagnostics.alpha[:, 1], [0.05, 0.5, 0.95]
                ).tolist(),
            }
            for index, sample_id in enumerate(evaluation.sample_id.astype(str)):
                diagnostic_rows.append(
                    {
                        "fold": fold,
                        "candidate": candidate,
                        "sample_id": sample_id,
                        "reference_time": evaluation.iloc[index].reference_time,
                        "reliability": diagnostics.reliability[index],
                        "neighbor_radius": diagnostics.radius[index],
                        "effective_neighbors": diagnostics.effective_neighbors[index],
                        "alpha_iron": diagnostics.alpha[index, 0],
                        "alpha_time": diagnostics.alpha[index, 1],
                    }
                )
        alpha_settings = registration["state_alpha"]
        alpha_calibrator = StateAlphaCalibrator(
            state_columns, parameters=alpha_settings["parameters"]
        ).fit(oof, stamp(origin))
        alpha_prediction, alpha_diagnostics = alpha_calibrator.predict(base, states)
        alpha_candidate = f"V9_{alpha_settings['id']}"
        candidates[alpha_candidate] = alpha_prediction
        fold_diagnostics["state_alpha"] = {
            "candidate": alpha_candidate,
            "global_alpha": alpha_calibrator.global_alpha_.tolist(),
            "gate_validation": alpha_calibrator.gate_validation_,
            "alpha_iron_quantiles": np.quantile(
                alpha_diagnostics.alpha[:, 0], [0.05, 0.5, 0.95]
            ).tolist(),
            "alpha_time_quantiles": np.quantile(
                alpha_diagnostics.alpha[:, 1], [0.05, 0.5, 0.95]
            ).tolist(),
        }
        results[fold] = {
            "rows": len(evaluation),
            "origin": str(stamp(origin)),
            "diagnostics": fold_diagnostics,
            "candidates": {
                name: prediction_metrics(evaluation, prediction)
                for name, prediction in candidates.items()
            },
        }
        for name, prediction in candidates.items():
            prediction.to_csv(prediction_files / f"{fold}_{name}.csv", index=False)

    reference = "V1_GLOBAL_STRUCTURAL_REPLAY"
    selection = {}
    candidate_names = [
        *(f"V9_{settings['id']}" for settings in registration["calibrators"]),
        f"V9_{registration['state_alpha']['id']}",
    ]
    for candidate in candidate_names:
        deltas = {
            fold: results[fold]["candidates"][candidate]["overall"]["loss"]
            - results[fold]["candidates"][reference]["overall"]["loss"]
            for fold in results
        }
        mean_delta = float(np.mean(list(deltas.values())))
        candidate_diagnostics = pd.DataFrame(diagnostic_rows)
        median_reliability = (
            float(
                candidate_diagnostics.loc[
                    candidate_diagnostics.candidate == candidate, "reliability"
                ].median()
            )
            if candidate != f"V9_{registration['state_alpha']['id']}"
            else 1.0
        )
        policy = registration["selection"]
        passed = (
            mean_delta <= float(policy["mean_development_delta_max"])
            and max(deltas.values()) <= float(policy["single_fold_delta_max"])
            and median_reliability >= float(policy["minimum_reliability_median"])
        )
        selection[candidate] = {
            "fold_deltas": deltas,
            "mean_delta": mean_delta,
            "median_reliability": median_reliability,
            "passed": bool(passed),
        }
    accepted = [name for name, item in selection.items() if item["passed"]]
    selected = min(accepted, key=lambda name: selection[name]["mean_delta"]) if accepted else None
    summary = {
        "G0": "PASS_DEVELOPMENT_CAUSAL_REPLAY",
        "G1": "PASS" if selected else "FAIL_RETAIN_V1",
        "selected": selected,
        "fit_count": fit_count,
        "protected_labels_read": False,
        "state_columns": list(state_columns),
        "selection": selection,
        "environment": runtime_environment(),
        "data_consistency": ctx.consistency,
    }
    pd.DataFrame(diagnostic_rows).to_csv(output / "analog_diagnostics.csv", index=False)
    atomic_write_json(output / "metrics.json", results)
    atomic_write_json(output / "summary.json", summary)
    print(summary, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_config, args.output)


if __name__ == "__main__":
    main()
