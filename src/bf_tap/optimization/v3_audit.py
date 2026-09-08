"""Read metadata only to audit existing E12/E16 component stage alignment."""
import argparse
import json
from pathlib import Path

from ..artifacts import atomic_write_json, file_sha256, stable_digest
from ..config import load_yaml
from ..exceptions import ContractError
from ..io import read_csv
from .alignment import stage_alignment
from .config import load_validation
from .validation import units_for_origin
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = load_yaml(args.data_config)
    validation, _, origins = load_validation("configs/optimization_v0_2/validation.yaml", expected_timezone="Asia/Shanghai")
    cells = {}
    for origin in origins:
        for unit in units_for_origin(origin, pd.Timestamp(validation["train_start"])):
            cells.setdefault(unit.horizon, []).append(unit.id)
    components = {
        "E09_PROCESS_CHANGE_E02": Path("local/runs/optimization-v0.2-opt05-e09-test-a-r1/bundle/bundle.json"),
        "E04": Path("local/runs/optimization-v0.2-opt05-e04-test-a-r1/bundle/bundle.json"),
    }
    records = []
    for name, path in components.items():
        bundle = json.loads(path.read_text())
        if bundle["training"]["candidate_id"] != name or bundle["training"]["protection_access"]["protected_access"]:
            raise ContractError("unexpected historical development bundle identity")
        cutoff = bundle["training"]["fit_cutoff"]
        for stage in ("test_a", "test_b", "test_c"):
            source = data["paths"][stage + "_samples"]
            samples = read_csv(source, usecols=["sample_id", "reference_time"],
                               required_columns=["sample_id", "reference_time"], time_columns=["reference_time"])
            record = stage_alignment(component=name, stage=stage, fit_cutoff=cutoff,
                                     history_cutoff=cutoff, label_available_cutoff=cutoff,
                                     reference_times=samples.reference_time,
                                     identity={"source": bundle["code_identity"]["source_snapshot_sha256"],
                                               "data": stable_digest({"training": bundle["training"]["input_identities"],
                                                                      "stage_samples_sha256": file_sha256(source)}),
                                               "candidate": bundle["optimization"]["sha256"], "component": file_sha256(path)},
                                     development_cells=cells)
            record["cutoff_evidence"] = "legacy_training_fit_cutoff_and_origin_freeze_contract"
            record["status"] = "existing_test_a_component" if stage == "test_a" else "prospective_mapping_only_no_stage_prediction"
            records.append(record)
    atomic_write_json(args.output / "stage_alignment.json", {"schema_version": "stage-alignment-v1", "records": records,
                                                              "protected_labels_read": False})
    atomic_write_json(args.output / "calibration_provenance.json", {
        "schema_version": "legacy-calibration-audit-v1", "candidate": "E16_TIMECAL_E12",
        "residual": {"tap_iron": 0, "tap_time_len": 1.68610975},
        "reported_fit_origins": ["O202406", "O202407", "O202408"],
        "reported_check_origins": ["O202409", "O202410"],
        "exact_fit_sample_provenance": "UNRESOLVED", "strict_full_pipeline_oof": False,
        "source_log_sha256": file_sha256("docs/submission_log.md"),
        "sample_ids_sha256": None, "fitted_at": None,
        "warning": "Global fixed residual reused across earlier origins and fit origins; historical development selection only."})
    atomic_write_json(args.output / "final_status.json", {"engineering_status": "G0_METADATA_AUDIT_PASS",
                                                          "model_quality_status": "G1_NOT_EVALUATED", "protected_labels_read": False})


if __name__ == "__main__":
    main()
