"""Cold-process verification of all 60 screening model bundles, without refits."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, stable_digest, verify_file_identities
from ..exceptions import ContractError
from .search_models import SingleTargetModel, search_configurations
from .v3_run import DevelopmentContext, TARGETS
from .validation import select_partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        ctx = DevelopmentContext(args.data_config, args.output)
        source_config = json.loads((args.screening_run / "resolved_config.json").read_text())
        if stable_digest(ctx.inputs) != stable_digest(source_config["inputs"]):
            raise ContractError("bundle verification source data identities differ")
        checks = []
        for unit in ctx.screening:
            _, actual, _ = select_partitions(ctx.labels, unit)
            # Exercise real feature schema and categories without another full validation run.
            sample = actual.groupby("spout_no", sort=True).head(2)
            X = ctx.X(sample, unit.fit_cutoff)
            for config in search_configurations("configs/optimization_v0_3/models"):
                for target in TARGETS:
                    model = SingleTargetModel.load(args.screening_run / "bundles" / unit.id / config["id"] / target)
                    if (model.parameters != config["parameters"] or model.target != target
                            or model.model_type != config["model_type"] or model.identity["fit_cutoff"] != str(unit.fit_cutoff)):
                        raise ContractError("restored target, parameters or cutoff identity mismatch")
                    old = pd.read_csv(args.screening_run / f"{unit.id}_{config['id']}_{target}.csv",
                                      dtype={"sample_id": "string"}, float_precision="round_trip")
                    expected = pd.DataFrame({"sample_id": sample.sample_id.astype(str)}).merge(old, on="sample_id", validate="one_to_one")
                    if len(expected) != len(sample):
                        raise ContractError("bundle verification prediction sample IDs differ")
                    difference = float(np.max(np.abs(np.maximum(0, model.predict(X)) - expected[f"pred_{target}"].to_numpy())))
                    if difference > 1e-12:
                        raise ContractError(f"restored prediction differs: {unit.id}/{config['id']}/{target}: {difference}")
                    checks.append({"fold": unit.id, "candidate": config["id"], "target": target,
                                   "samples": len(sample), "max_absolute_difference": difference})
        verify_file_identities(ctx.inputs)
        atomic_write_json(args.output / "checks.json", checks)
        atomic_write_json(args.output / "final_status.json", {
            "engineering_status": "G0_60_BUNDLE_COLD_PROCESS_PASS", "model_quality_status": "G1_NOT_EVALUATED",
            "bundles_checked": len(checks), "protected_labels_read": False,
            "max_absolute_difference": max(c["max_absolute_difference"] for c in checks),
            "scope": "two_samples_per_observed_spout_per_fold_not_full_composite_release"})
    except Exception as exc:
        atomic_write_json(args.output / "final_status.json", {"status": "FAILED", "error": str(exc), "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
