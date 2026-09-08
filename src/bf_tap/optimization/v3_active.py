"""Predict with the explicitly activated release, without training or label reads."""
import argparse
import hashlib
from io import BytesIO
from pathlib import Path
import shutil
from zipfile import ZipFile

import pandas as pd

from ..artifacts import atomic_write_json, file_sha256
from ..config import load_yaml
from ..exceptions import ContractError
from ..submission import validate_submission
from .v3_followup import read
from .v3_release import predict_stage


def active_bundle(config_path):
    cfg = load_yaml(config_path)
    if (cfg.get("release_protocol") != "development-fc-catboost-v1"
            or cfg.get("active_candidate") != "CB-FC-CVcal"
            or cfg.get("promotion") != "USER_OVERRIDE_G1_FAIL"):
        raise ContractError("unsupported active release pointer")
    bundle = Path(cfg["bundle"])
    if file_sha256(bundle / "bundle.json") != cfg["bundle_sha256"]:
        raise ContractError("active release bundle identity mismatch")
    if read(bundle / "bundle.json")["candidate"] != cfg["active_candidate"]:
        raise ContractError("active release candidate mismatch")
    return bundle


def export_archived_submission(config_path, data_config, stage, output):
    """Export the exact retained E16 artifact, not a new model prediction."""
    cfg = load_yaml(config_path)
    if (cfg.get("release_protocol") != "archived-e16-submission-v1"
            or cfg.get("active_candidate") != "E16_TIMECAL_E12" or stage != "test_a"):
        raise ContractError("archived release export supports declared E16 test_a only")
    source = Path(cfg["test_a_zip"])
    if file_sha256(source) != cfg["test_a_zip_sha256"]:
        raise ContractError("archived submission identity mismatch")
    with ZipFile(source) as archive:
        if archive.namelist() != ["result.csv"]:
            raise ContractError("archived submission ZIP structure mismatch")
        payload = archive.read("result.csv")
    if hashlib.sha256(payload).hexdigest() != cfg["test_a_result_sha256"]:
        raise ContractError("archived CSV identity mismatch")
    data = load_yaml(data_config)
    expected = pd.read_csv(data["paths"]["test_a_samples"], usecols=["sample_id"], dtype={"sample_id": "string"})
    validate_submission(pd.read_csv(BytesIO(payload), dtype={"sample_id": "string"}), expected.sample_id)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    destination = output / source.name
    shutil.copyfile(source, destination)
    if file_sha256(destination) != cfg["test_a_zip_sha256"]:
        raise ContractError("archived export copy mismatch")
    atomic_write_json(output / "export_manifest.json", {
        "candidate": "E16_TIMECAL_E12", "stage": stage, "mode": "exact_archived_submission_export_not_retraining",
        "zip_sha256": file_sha256(destination), "protected_labels_read": False})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-config", default="configs/optimization_v0_3/active_release.yaml")
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    parser.add_argument("--stage", choices=["test_a", "test_b", "test_c"], default="test_a")
    parser.add_argument("--sample-index", nargs="+")
    parser.add_argument("--export-submission", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.export_submission:
        export_archived_submission(args.release_config, args.data_config, args.stage, args.output)
        return
    if load_yaml(args.release_config).get("release_protocol") == "archived-e16-submission-v1":
        raise ContractError("E16 restored as an archived release; use --export-submission, not raw-input prediction")
    if not args.sample_index:
        raise ContractError("raw-input F-C prediction requires explicit sample-index sources")
    bundle = active_bundle(args.release_config)
    predict_stage(bundle, args.data_config, args.stage, args.sample_index, args.output)


if __name__ == "__main__":
    main()
