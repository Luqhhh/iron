"""Read-only V22 certificate and fresh-inference audit; fitting is forbidden."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from bf_tap.artifacts import atomic_write_json, file_sha256, verify_file_identities
from bf_tap.exceptions import ContractError
from bf_tap.optimization.qrf_support_shrink import (
    apply_shrink,
    verify_h2_bank,
    verify_lambda_certificate,
    verify_median_certificate,
    verify_prediction_directions,
)
from bf_tap.optimization.v13_common import zero_fit


REPO = Path(__file__).resolve().parents[1]


def read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_sources(root: Path, manifest: dict) -> None:
    repair_path = root / "engineering_repair/manifest.json"
    if not repair_path.exists():
        verify_file_identities(manifest["sources"])
        return
    repair = read(repair_path)
    if (
        repair.get("kind") != "V22_ENGINEERING_RESUME_v1"
        or repair.get("original_manifest_sha256") != file_sha256(root / "manifest.json")
        or repair.get("algorithm_or_parameter_change") is not False
        or repair.get("additional_forest_or_preprocessor_fits_authorized") != 0
    ):
        raise ContractError("invalid V22 engineering repair registration")
    original_failure = repair["original_failure"]
    if file_sha256(original_failure["path"]) != original_failure["sha256"]:
        raise ContractError("preserved original V22 failure identity changed")
    verify_file_identities(repair["preserved_artifacts"])
    cold_resume_path = root / "engineering_repair/cold_resume_manifest.json"
    active_sources = repair["repaired_sources"]
    if cold_resume_path.exists():
        cold_resume = read(cold_resume_path)
        if (
            cold_resume.get("kind") != "V22_COLD_SCORE_RESUME_v1"
            or cold_resume.get("prior_repair_manifest_sha256") != file_sha256(repair_path)
            or cold_resume.get("algorithm_or_prediction_change") is not False
            or cold_resume.get("additional_fit_attempts_authorized") != 0
        ):
            raise ContractError("invalid V22 cold/score resume registration")
        verify_file_identities(cold_resume["preserved_artifacts"])
        active_sources = cold_resume["repaired_sources"]
    certificate_resume_path = root / "engineering_repair/certificate_resume_manifest.json"
    if certificate_resume_path.exists():
        certificate_resume = read(certificate_resume_path)
        if (
            certificate_resume.get("kind") != "V22_CERTIFICATE_SCORE_RESUME_v1"
            or certificate_resume.get("prior_cold_resume_manifest_sha256") != file_sha256(cold_resume_path)
            or certificate_resume.get("algorithm_or_prediction_change") is not False
            or certificate_resume.get("additional_fit_attempts_authorized") != 0
        ):
            raise ContractError("invalid V22 certificate/score resume registration")
        verify_file_identities(certificate_resume["preserved_artifacts"])
        active_sources = certificate_resume["repaired_sources"]
    label_resume_path = root / "engineering_repair/label_certificate_resume_manifest.json"
    if label_resume_path.exists():
        label_resume = read(label_resume_path)
        if (
            label_resume.get("kind") != "V22_LABEL_CERTIFICATE_SCORE_RESUME_v1"
            or label_resume.get("prior_certificate_resume_manifest_sha256") != file_sha256(certificate_resume_path)
            or label_resume.get("algorithm_or_prediction_change") is not False
            or label_resume.get("additional_fit_attempts_authorized") != 0
        ):
            raise ContractError("invalid V22 label-certificate resume registration")
        verify_file_identities(label_resume["preserved_artifacts"])
        active_sources = label_resume["repaired_sources"]
    verify_file_identities(active_sources)
    repaired = {
        Path(value["path"]).resolve()
        for sources in (repair["repaired_sources"], active_sources)
        for value in sources.values()
    }
    unchanged = {
        key: value for key, value in manifest["sources"].items()
        if Path(value["path"]).resolve() not in repaired
    }
    verify_file_identities(unchanged)


def _prediction_invariance(v10: pd.DataFrame, bank: pd.DataFrame, lambdas: dict[str, float]) -> None:
    expected = apply_shrink(v10, bank, lambdas).set_index("sample_id")
    indices = [
        np.arange(len(v10))[::-1],
        np.arange(0, len(v10), max(1, len(v10) // 7)),
        np.asarray([len(v10) // 2]),
    ]
    indices.extend(np.arange(i, min(i + 127, len(v10))) for i in range(0, len(v10), 127))
    for selected in indices:
        ids = v10.iloc[selected].sample_id.astype(str).tolist()
        part = apply_shrink(
            v10.iloc[selected].reset_index(drop=True),
            bank.set_index("sample_id").loc[ids].reset_index(),
            lambdas,
        ).set_index("sample_id")
        if not part.equals(expected.loc[ids]):
            raise ContractError("V22 composition changes with order/subset/chunk/single inference")


def check(run: Path, *, resume_certificates: bool = False) -> dict:
    root = run.resolve()
    manifest = read(root / "manifest.json")
    _verify_sources(root, manifest)
    p0 = read(root / "p0_original_replay.json")
    if p0.get("status") != "PASS":
        raise ContractError("cold validation requires a passed P0; blocked runs remain NOT_RUN")
    complete = read(root / "predictions_complete.json")
    if complete.get("status") != "PASS":
        raise ContractError("completed frozen predictions are required")
    verify_file_identities(complete["identities"])
    cold_root = root / "cold_worker"
    if cold_root.exists() and not resume_certificates:
        raise FileExistsError(cold_root)
    cold_root.mkdir(exist_ok=resume_certificates)
    plan = read(root / "cold_plan.json")
    worker = REPO / "workers/qrf_v015"
    worker_reports = []
    with zero_fit() as root_counts:
        for task in plan["qrf_tasks"]:
            output = cold_root / f"{task['id']}.npz"
            if not resume_certificates:
                subprocess.run(
                    [
                        str(worker / ".venv/bin/python"), str(worker / "inference_v022.py"),
                        "--model-manifest", task["model_manifest"],
                        "--model-manifest-sha256", task["model_manifest_sha256"],
                        "--input", task["input"], "--input-sha256", task["input_sha256"],
                        "--expected-output", task["expected_output"], "--output", str(output),
                    ],
                    check=True,
                    cwd=REPO,
                )
            else:
                if not output.is_file() or not Path(str(output) + ".json").is_file():
                    raise ContractError("preserved cold worker output is incomplete")
                with np.load(output, allow_pickle=False) as got, np.load(task["expected_output"], allow_pickle=False) as expected:
                    if set(got.files) != set(expected.files) or not all(np.array_equal(got[name], expected[name]) for name in got.files):
                        raise ContractError("preserved cold worker output differs from frozen prediction")
            report = read(str(output) + ".json")
            if any(report["zero_fit"].values()) or not all(report["checks"].values()):
                raise ContractError("worker cold audit attempted a fit or failed consistency")
            worker_reports.append(report)
        for item in plan["median_tasks"]:
            parameters = read(item["parameters"])
            history = pd.read_csv(item["history"], dtype={"sample_id": str, "spout_no": str})
            for certificate in parameters["certificates"].values():
                verify_median_certificate(history, certificate)
        for item in plan["lambda_tasks"]:
            parameters = read(item["parameters"])
            for spout, value in parameters["spouts"].items():
                selected = pd.read_csv(item["selected"][spout], dtype={"sample_id": str}, float_precision="round_trip")
                verify_lambda_certificate(selected, value["certificate"])
        for item in plan["composition_tasks"]:
            v10 = pd.read_csv(item["v10"], dtype={"sample_id": str}, float_precision="round_trip")
            bank = pd.read_csv(item["bank"], dtype={"sample_id": str, "spout_no": str}, float_precision="round_trip")
            verify_h2_bank(bank) if "label_available_at" in bank else verify_prediction_directions(bank)
            parameters = read(item["parameters"])
            lambdas = {s: row["lambda"] for s, row in parameters["spouts"].items()}
            _prediction_invariance(v10, bank, lambdas)
            actual = pd.read_csv(item["prediction"], dtype={"sample_id": str}, float_precision="round_trip")
            if not actual.equals(apply_shrink(v10, bank, lambdas)):
                raise ContractError("cold V22 composition differs from stored full-precision prediction")
    result = {
        "status": "PASS",
        "root_zero_fit": root_counts,
        "worker_tasks": len(worker_reports),
        "worker_zero_fit": True,
        "median_certificates_verified": sum(len(read(x["parameters"])["certificates"]) for x in plan["median_tasks"]),
        "lambda_certificates_verified": sum(len(read(x["parameters"])["spouts"]) for x in plan["lambda_tasks"]),
        "full_reverse_chunk_subset_single_exact": True,
        "quality_evaluated": False,
        "platform_verified": False,
        "preserved_worker_outputs_reused": bool(resume_certificates),
    }
    atomic_write_json(root / "cold_validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--resume-certificates", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.run, resume_certificates=args.resume_certificates), ensure_ascii=False, indent=2))
