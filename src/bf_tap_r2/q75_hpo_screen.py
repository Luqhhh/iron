"""Joint TabM hyper-parameter screen on the frozen folds.

Every arm is a single-factor change from the incumbent V12 recipe, except the
three combined cosine+L1 arms.  The control arm reuses the parent training path
unchanged so it reproduces recorded predictions bit-exactly.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from .q75_schedule_screen import ScheduledJointRegressor

TARGETS = ("tap_iron", "tap_time_len")
RECORDED = "local/runs/round2-v12-joint-tabm/development-r1"

#: name -> (settings overrides, recipe overrides)
CONFIGS: dict[str, tuple[dict, dict]] = {
    "CONTROL": ({}, {}),
    "DROP0": ({"dropout": 0.0}, {}),
    "LR3": ({"learning_rate": 0.003}, {}),
    "WD1E3": ({"weight_decay": 0.001}, {}),
    "BATCH128": ({"batch_size": 128}, {}),
    "FREQ001": ({}, {"frequency": 0.001}),
    "FREQ01": ({}, {"frequency": 0.1}),
    "NFREQ32": ({"n_frequencies": 32}, {}),
    "EMB32": ({"embedding_dim": 32}, {}),
    "K8": ({"tabm_k": 8}, {}),
    "K32": ({"tabm_k": 32}, {}),
    "BLOCKS3": ({"blocks": 3}, {}),
    "WIDTH384": ({"width": 384}, {}),
    "COS_L1_LR3": ({"lr_schedule": "cosine", "loss": "mae", "learning_rate": 0.003}, {}),
    "COS_L1_DROP0": ({"lr_schedule": "cosine", "loss": "mae", "dropout": 0.0}, {}),
    "COS_L1_LR3_DROP0": ({"lr_schedule": "cosine", "loss": "mae", "learning_rate": 0.003,
                          "dropout": 0.0}, {}),
    # stage 1b: combinations on top of the confirmed k=32 arm
    "K32": ({"tabm_k": 32}, {}),
    "K32_COS_MAE": ({"tabm_k": 32, "lr_schedule": "cosine", "loss": "mae"}, {}),
    "K32_FREQ001": ({"tabm_k": 32}, {"frequency": 0.001}),
    "K32_COS_MAE_FREQ001": ({"tabm_k": 32, "lr_schedule": "cosine", "loss": "mae"},
                            {"frequency": 0.001}),
    # stage 1c: pure internal-ensemble width sweep
    "K48": ({"tabm_k": 48}, {}),
    "K64": ({"tabm_k": 64}, {}),
    "K96": ({"tabm_k": 96}, {}),
}


def wmape(y, p):
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def run_unit(root, settings, seed, fold, configs):
    import torch

    from .v3_run import load_training_frame, load_fold_vector

    torch.set_num_threads(1)
    root = Path(root)
    frame = load_training_frame(root)
    folds = load_fold_vector(root, frame, seed)
    mask = folds == fold
    training = frame.loc[~mask].reset_index(drop=True)
    query = frame.loc[mask].drop(columns=list(TARGETS)).reset_index(drop=True)
    records = {}
    for name in configs:
        overrides, recipe_overrides = CONFIGS[name]
        unit = dict(settings)
        unit.update(overrides)
        recipe = {"backbone": "tabm", "frequency": 0.01}
        recipe.update(recipe_overrides)
        model = ScheduledJointRegressor(recipe, unit).fit(
            training, training[list(TARGETS)].to_numpy())
        prediction = model.predict(query)
        record = {"selected_epoch": int(model.metadata_["selected_epoch"]),
                  "wmape": {t: wmape(frame[t].to_numpy()[mask], prediction[:, i])
                            for i, t in enumerate(TARGETS)}}
        if name == "CONTROL":
            reference = np.load(root / RECORDED / f"joint-joint_plr001-s{seed}-f{fold}.npy")
            record["control_max_abs_diff"] = float(np.max(np.abs(prediction - reference)))
        records[name] = record
    return seed, fold, records


def run_stage1(root, output, configs=None, units=((42, 0), (42, 1)), workers=6):
    root = Path(root).resolve()
    out = (root / output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    configs = list(configs or CONFIGS)
    frame_spec = yaml.safe_load((root / "configs/round2_v12/SPEC.yaml").read_text())
    settings = dict(frame_spec["training"])
    results, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(run_unit, str(root), settings, seed, fold, configs): (seed, fold)
                for seed, fold in units}
        for future in as_completed(jobs):
            seed, fold = jobs[future]
            key = f"s{seed}-f{fold}"
            try:
                _, _, records = future.result()
                results[key] = records
                print(json.dumps({"unit": key,
                                  "wmape": {n: r["wmape"] for n, r in records.items()}}),
                      flush=True)
            except Exception as exc:
                failures.append({"unit": key, "error": repr(exc)})
                print(json.dumps({"unit": key, "error": repr(exc)}), flush=True)
    payload = {"stage": "stage1", "units": results, "failures": failures,
               "configs": {n: {"settings": CONFIGS[n][0], "recipe": CONFIGS[n][1]}
                           for n in configs}}
    (out / "stage1.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    if failures:
        return 1
    differences = [r["control_max_abs_diff"] for unit in results.values() for r in unit.values()
                   if "control_max_abs_diff" in r]
    control = max(differences) if differences else None
    print(json.dumps({"control_max_abs_diff": control,
                      "control": ("passed" if control <= 1e-9 else "FAILED") if differences
                      else "not_in_arm_set"}))
    ordered = sorted(results)
    baseline = "CONTROL" if "CONTROL" in configs else configs[0]
    for target in TARGETS:
        base = {unit: results[unit][baseline]["wmape"][target] for unit in ordered}
        print(f"== {target} baseline {baseline} {np.mean(list(base.values())):.6f}")
        ranked = []
        for name in configs:
            values = np.array([results[unit][name]["wmape"][target] for unit in ordered])
            reference = np.array([base[unit] for unit in ordered])
            ranked.append((values.mean(), name, bool((values < reference).all()),
                           values.mean() - reference.mean()))
        for mean, name, same_direction, delta in sorted(ranked):
            print(f"   {name:18s} mean {mean:.6f} delta {delta:+.6f} both {same_direction}")
    return 0


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", default="")
    args = parser.parse_args(argv)
    configs = [c for c in args.configs.split(",") if c] or None
    return run_stage1(args.root, args.output, configs)


if __name__ == "__main__":
    raise SystemExit(main())


def run_stage2(root, output, configs, units=None, workers=6):
    """Confirm selected hyper-parameter arms on complete two-seed coverage."""
    root = Path(root).resolve()
    out = (root / output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    configs = list(configs)
    for name in configs:
        if name not in CONFIGS:
            raise ValueError(f"Unknown config: {name}")
    spec = yaml.safe_load((root / "configs/round2_v12/SPEC.yaml").read_text())
    settings = dict(spec["training"])
    units = tuple(units or [(seed, fold) for seed in (42, 3407) for fold in range(5)])

    from .v3_run import load_training_frame, load_fold_vector
    frame = load_training_frame(root)
    folds = {seed: load_fold_vector(root, frame, seed) for seed, _ in units}
    y = {t: frame[t].to_numpy(float) for t in TARGETS}
    results, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(run_unit, str(root), settings, seed, fold, configs): (seed, fold)
                for seed, fold in units}
        for future in as_completed(jobs):
            seed, fold = jobs[future]
            key = f"s{seed}-f{fold}"
            try:
                _, _, records = future.result()
                results[key] = records
                with (out / f"{key}.json").open("x", encoding="utf-8") as stream:
                    json.dump(records, stream, indent=1)
                print(json.dumps({"unit": key, "wmape": {n: r["wmape"] for n, r in records.items()}}),
                      flush=True)
            except Exception as exc:
                failures.append({"unit": key, "error": repr(exc)})
                print(json.dumps({"unit": key, "error": repr(exc)}), flush=True)
    if failures:
        (out / "stage2.json").write_text(json.dumps({"units": results, "failures": failures}, indent=1),
                                         encoding="utf-8")
        return 1
    differences = [r["control_max_abs_diff"] for unit in results.values() for r in unit.values()
                   if "control_max_abs_diff" in r]
    control = max(differences) if differences else None
    print(json.dumps({"control_max_abs_diff": control,
                      "control": ("passed" if control <= 1e-9 else "FAILED") if differences
                      else "not_in_arm_set"}))
    report = {"units": results, "failures": [], "configs": configs,
              "control_max_abs_diff": control, "per_seed": {}}
    review = root / "local/runs/q75-combination-review-20261002/review-r1"
    for name in configs:
        print(f"== {name}")
        for index, target in enumerate(TARGETS):
            per_seed = {}
            for seed in sorted({s for s, _ in units}):
                rows = [f"s{seed}-f{fold}" for _, fold in units if _ == seed]
                base, cand, incumbent = [], [], []
                z = np.load(review / f"combination-s{seed}.npz")
                reference = z["iron"] if target == "tap_iron" else z["q75"]
                for key, fold in [(k, int(k.split("-f")[1])) for k in rows]:
                    mask = folds[seed] == fold
                    recorded = np.load(root / RECORDED / f"joint-joint_plr001-s{seed}-f{fold}.npy")[:, index]
                    base.append(wmape(y[target][mask], recorded))
                    cand.append(results[key][name]["wmape"][target])
                    incumbent.append(wmape(y[target][mask], reference[mask]))
                per_seed[str(seed)] = {
                    "single_recipe": float(np.mean(base)), "candidate": float(np.mean(cand)),
                    "incumbent": float(np.mean(incumbent)),
                    "delta_vs_single_recipe": float(np.mean(base) - np.mean(cand)),
                    "delta_vs_incumbent": float(np.mean(incumbent) - np.mean(cand))}
                print(f"   {target:13s} seed{seed} " + json.dumps(per_seed[str(seed)]))
            report["per_seed"].setdefault(name, {})[target] = per_seed
    (out / "stage2.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


#: k=32 neighbourhood arms: name -> settings overrides on top of K32
NEIGHBOURHOOD: dict[str, dict] = {
    "K32": {},
    "K32_DROP0": {"dropout": 0.0},
    "K32_DROP2": {"dropout": 0.2},
    "K32_DROP3": {"dropout": 0.3},
    "K32_WD0": {"weight_decay": 0.0},
    "K32_WD1E3": {"weight_decay": 0.001},
    "K32_BATCH128": {"batch_size": 128},
    "K32_BATCH512": {"batch_size": 512},
    "K32_W128": {"width": 128},
    "K32_W384": {"width": 384},
    "K32_B1": {"blocks": 1},
    "K32_B3": {"blocks": 3},
    "K32_NFREQ32": {"n_frequencies": 32},
    "K32_EMB32": {"embedding_dim": 32},
    "K32_LONG": {"max_epochs": 400, "patience": 60},
    "K32_LR3": {"learning_rate": 0.003},
}
for _name, _overrides in NEIGHBOURHOOD.items():
    CONFIGS.setdefault(_name, ({"tabm_k": 32, **_overrides}, {}))


def run_parallel(root, output, arms=None, units=((42, 0), (42, 1)), workers=6):
    """One job per (unit, arm) so long arms do not serialise inside a worker."""
    root = Path(root).resolve()
    out = (root / output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    arms = list(arms or NEIGHBOURHOOD)
    spec = yaml.safe_load((root / "configs/round2_v12/SPEC.yaml").read_text())
    settings = dict(spec["training"])
    results, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(run_unit, str(root), settings, seed, fold, [arm]): (seed, fold, arm)
                for seed, fold in units for arm in arms}
        for future in as_completed(jobs):
            seed, fold, arm = jobs[future]
            key = f"s{seed}-f{fold}"
            try:
                _, _, records = future.result()
                results.setdefault(key, {}).update(records)
                print(json.dumps({"unit": key, "arm": arm, "wmape": records[arm]["wmape"]}),
                      flush=True)
            except Exception as exc:
                failures.append({"unit": key, "arm": arm, "error": repr(exc)})
                print(json.dumps({"unit": key, "arm": arm, "error": repr(exc)}), flush=True)
    payload = {"stage": "neighbourhood", "units": results, "failures": failures, "arms": arms}
    (out / "stage.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    if failures:
        return 1
    ordered = sorted(results)
    for target in TARGETS:
        base = {unit: results[unit]["K32"]["wmape"][target] for unit in ordered}
        print(f"== {target} K32 baseline {np.mean(list(base.values())):.6f}")
        ranked = []
        for arm in arms:
            values = np.array([results[unit][arm]["wmape"][target] for unit in ordered])
            reference = np.array([base[unit] for unit in ordered])
            ranked.append((values.mean() - reference.mean(), arm, bool((values < reference).all()),
                           values.mean()))
        for delta, arm, same, mean in sorted(ranked):
            print(f"   {arm:16s} mean {mean:.6f} delta {delta:+.6f} both {same}")
    return 0


#: stage 2b: combinations of the same-direction iron winners at k=32
IRON_COMBINATIONS: dict[str, dict] = {
    "K32": {},
    "K32_LR3_DROP0": {"learning_rate": 0.003, "dropout": 0.0},
    "K32_LR3_COS_MAE": {"learning_rate": 0.003, "lr_schedule": "cosine", "loss": "mae"},
    "K32_DROP0_COS_MAE": {"dropout": 0.0, "lr_schedule": "cosine", "loss": "mae"},
    "K32_LR3_DROP0_COS_MAE": {"learning_rate": 0.003, "dropout": 0.0,
                              "lr_schedule": "cosine", "loss": "mae"},
}
for _name, _overrides in IRON_COMBINATIONS.items():
    CONFIGS.setdefault(_name, ({"tabm_k": 32, **_overrides}, {}))
