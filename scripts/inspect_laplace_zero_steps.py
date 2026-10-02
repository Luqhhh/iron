"""Read-only mathematical probes on our completed synthetic G0 tree states."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle

import numpy as np

from bf_tap_r2.joint_support_audit import local_output, sha, write_new
from bf_tap_r2.laplace_noise import advance


def mean_nll(y, params):
    return float((math.log(2.)+params[:, 1]+np.abs(y-params[:, 0])/np.exp(params[:, 1])).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    source = root/"local/runs/laplace-epoch-20261002/engineering-r1"
    result = json.loads((source/"result.json").read_text(encoding="utf-8"))
    receipt = json.loads((source/"cold-readback.json").read_text(encoding="utf-8"))
    if (result["stage"] != "synthetic_complete_epoch_engineering"
            or receipt["result_sha256"] != sha(source/"result.json") or receipt["checked_models"] != 4):
        raise ValueError("Only our completed synthetic engineering archive is permitted")
    for name, digest in result["output_sha256"].items():
        if sha(source/name) != digest:
            raise ValueError("Own engineering artifact identity changed")
    with np.load(source/"data.npz", allow_pickle=False) as data:
        x, y = data["x"][:108], data["y"][:108]
    probes = {}
    for recipe in ("FIXED", "ADAPTIVE"):
        with (source/f"{recipe}-complete.pkl").open("rb") as handle:
            model = pickle.load(handle)
        zero = next((i for i, entry in enumerate(model.history_) if entry["step"] == 0.), None)
        if zero is None:
            probes[recipe] = {"zero_step_found": False}
            continue
        params = model.predict_params(x, epoch=zero)
        direction = np.zeros_like(params)
        z = (x-model.x_mean_)/model.x_scale_
        for dim, tree in enumerate(model.trees_[zero]):
            direction[:, dim] = tree.predict(z)
        sy = (y-model.y_median_)/model.y_scale_
        original = mean_nll(sy, params)
        values = [{"step": 2.**(-power), "training_nll_change": mean_nll(sy, advance(params, direction, 2.**(-power)))-original}
                  for power in range(21)]
        probes[recipe] = {"zero_step_found": True, "first_zero_epoch": zero+1,
                          "all_later_steps_zero": all(h["step"] == 0. for h in model.history_[zero:]),
                          "direction_mean_norm": float(np.linalg.norm(direction, axis=1).mean()),
                          "original_training_nll": original, "probes": values,
                          "strict_decrease_below_original_grid": any(p["step"] < .03125 and p["training_nll_change"] < 0 for p in values)}
    out = local_output(root, args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = {"stage": "synthetic_G0_math_probe_not_scientific_quality", "model_fits": 0, "tree_fits": 0,
              "official_data_reads": 0, "changed_models": 0, "packages": 0, "formal_promoted": False,
              "source_result_sha256": sha(source/"result.json"), "script_sha256": sha(Path(__file__)), "arms": probes}
    write_new(out/"result.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
