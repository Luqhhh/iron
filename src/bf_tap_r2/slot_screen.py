"""Zero-fit review of the platform response to recorded single-column candidate receipts.

Reads only our own frozen submission packages plus registered platform deltas; no model,
no training labels, no test labels, no new package and no upload.  See
``docs/slot_screen/PREREGISTRATION.md`` and ``configs/slot_screen/SPEC.json``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ZIP_NAME = "Luqhhh_bf_tap_predict_round2.zip"
TARGET_KEYS = ("pred_tap_iron", "pred_tap_time_len")
TEMPLATE = ROOT / "复赛_test/result_template.csv"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def template_ids() -> list[str]:
    with TEMPLATE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row["sample_id"] for row in rows]


def read_package(directory: Path, expected_ids: list[str]) -> dict:
    """Return the frozen string fields and float values of one submission package."""
    archive = directory / ZIP_NAME
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if names != ["result.csv"]:
            raise ValueError(f"{archive}: unexpected zip members {names}")
        payload = handle.read("result.csv")
        if handle.testzip() is not None:
            raise ValueError(f"{archive}: CRC failure")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    if len(rows) != 322 or len({row["sample_id"] for row in rows}) != 322:
        raise ValueError(f"{archive}: expected 322 unique rows")
    if [row["sample_id"] for row in rows] != expected_ids:
        raise ValueError(f"{archive}: submission order differs from the official template")
    fields = {key: [row[key] for row in rows] for key in TARGET_KEYS}
    values = {key: np.array([float(v) for v in fields[key]]) for key in TARGET_KEYS}
    for key in TARGET_KEYS:
        if not np.isfinite(values[key]).all() or (values[key] < 0).any():
            raise ValueError(f"{archive}: {key} must be finite and non-negative")
    return {
        "zip": str(archive.relative_to(ROOT)),
        "zip_sha256": file_sha256(archive),
        "fields": fields,
        "values": values,
    }


def slope_on_parent(delta: np.ndarray, parent: np.ndarray) -> float:
    centred = parent - parent.mean()
    denominator = float((centred ** 2).sum())
    if denominator <= 0:
        raise ValueError("Parent column has no variance")
    return float((centred * (delta - delta.mean())).sum() / denominator)


def geometry(candidate: dict, parent: dict) -> dict:
    """Verify the single-column identity gate and measure the change geometry."""
    changed = [key for key in TARGET_KEYS if candidate["fields"][key] != parent["fields"][key]]
    unchanged = [key for key in TARGET_KEYS if candidate["fields"][key] == parent["fields"][key]]
    if len(changed) != 1 or len(unchanged) != 1:
        raise ValueError("A receipt must change exactly one target column with the other byte-identical")
    key = changed[0]
    delta = candidate["values"][key] - parent["values"][key]
    base = parent["values"][key]
    scale = float(np.abs(base).mean())
    return {
        "changed_target": key,
        "unchanged_target": unchanged[0],
        "scale": scale,
        "rho": float(math.sqrt(float((delta ** 2).mean())) / scale),
        "bias_pct": float(100.0 * delta.mean() / scale),
        "slope": slope_on_parent(delta, base),
        "max_abs_pct": float(100.0 * float(np.abs(delta).max()) / scale),
        "changed_fraction": float(np.mean([a != b for a, b in
                                           zip(candidate["fields"][key], parent["fields"][key])])),
        "delta": delta,
        "parent_column": base,
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = average_ranks(x), average_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / denominator)


def permutation_p(x: np.ndarray, y: np.ndarray, permutations: int, seed: int) -> float:
    observed = abs(spearman(x, y))
    generator = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        if abs(spearman(x, generator.permutation(y))) >= observed:
            exceed += 1
    return float((exceed + 1) / (permutations + 1))


def hypergeometric_tail(successes_in_group: int, group: int, positives: int, total: int) -> float:
    """P(X <= successes_in_group) for a hypergeometric draw, exact rational arithmetic."""
    def choose(n: int, k: int) -> int:
        return math.comb(n, k) if 0 <= k <= n else 0
    denominator = choose(total, group)
    return float(sum(choose(positives, k) * choose(total - positives, group - k)
                     for k in range(0, successes_in_group + 1)) / denominator)


def natural_experiments(records: list[dict]) -> list[dict]:
    found = []
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            if left["changed_target"] != right["changed_target"]:
                continue
            if left["candidate_zip_sha256"] == right["candidate_zip_sha256"]:
                continue
            if float(np.abs(left["delta"] - right["delta"]).max()) <= 1e-12:
                found.append({
                    "target": left["changed_target"],
                    "first": left["id"], "first_parent": left["parent"], "first_platform_delta": left["platform_delta"],
                    "second": right["id"], "second_parent": right["parent"], "second_platform_delta": right["platform_delta"],
                    "max_abs_delta_difference": float(np.abs(left["delta"] - right["delta"]).max()),
                    "platform_delta_difference": float(left["platform_delta"] - right["platform_delta"]),
                })
    return found


def classify(records: list[dict], rule: dict) -> dict:
    """Descriptive application of the prospective rule to the receipts that produced it."""
    passing, failing = [], []
    for record in records:
        conditions = {
            "slope_less_or_equal_zero": record["slope"] <= 0.0,
            "combined_rho_less_or_equal_0.01": record["rho"] <= 0.01,
        }
        target = passing if all(conditions.values()) else failing
        target.append({
            "id": record["id"], "platform_delta": record["platform_delta"],
            "slope": record["slope"], "rho": record["rho"], "conditions": conditions,
        })
    return {
        "rule_status": rule["status"],
        "conditions": rule["conditions"],
        "note": rule["note"],
        "descriptive_only_not_validation": True,
        "would_pass": passing,
        "would_fail": failing,
        "pass_platform_deltas": [item["platform_delta"] for item in passing],
        "fail_platform_deltas": [item["platform_delta"] for item in failing],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/slot_screen/SPEC.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = read_json(ROOT / args.spec)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    expected = template_ids()
    cache: dict[str, dict] = {}

    def package(relative: str) -> dict:
        if relative not in cache:
            cache[relative] = read_package(ROOT / relative, expected)
        return cache[relative]

    records, rejected = [], []
    for receipt in spec["receipts"]:
        try:
            candidate = package(receipt["dir"])
            parent = package(receipt["parent_dir"])
            measured = geometry(candidate, parent)
        except Exception as error:  # keep every rejection in the evidence
            rejected.append({"id": receipt["id"], "reason": f"{type(error).__name__}: {error}"})
            continue
        registered = spec["parent_scores"][receipt["parent"]]
        records.append({
            **{k: v for k, v in measured.items() if k not in ("delta", "parent_column")},
            "delta": measured["delta"],
            "id": receipt["id"],
            "parent": receipt["parent"],
            "platform_delta": float(receipt["platform_delta"]),
            "local_gain": receipt["local_gain"],
            "local_gain_source": receipt["local_gain_source"],
            "candidate_zip": candidate["zip"],
            "candidate_zip_sha256": candidate["zip_sha256"],
            "parent_zip": parent["zip"],
            "parent_zip_sha256": parent["zip_sha256"],
            "registered_parent_score": registered,
        })

    platform = np.array([record["platform_delta"] for record in records])
    tests = {}
    for statistic in spec["association_tests"]["statistics"]:
        subset = [record for record in records if record.get(statistic) is not None]
        if len(subset) < 4 or len({record[statistic] for record in subset}) < 2:
            tests[statistic] = {"n": len(subset), "status": "insufficient_variation"}
            continue
        x = np.array([float(record[statistic]) for record in subset])
        y = np.array([record["platform_delta"] for record in subset])
        tests[statistic] = {
            "n": len(subset),
            "spearman": spearman(x, y),
            "permutation_p_two_sided": permutation_p(
                x, y, spec["association_tests"]["permutations"], spec["association_tests"]["seed"]),
            "ids": [record["id"] for record in subset],
        }

    expansion = [record for record in records if record["slope"] > 0]
    compression = [record for record in records if record["slope"] <= 0]
    positives = sum(1 for record in records if record["platform_delta"] > 0)
    contingency = {
        "expansion_positive": sum(1 for record in expansion if record["platform_delta"] > 0),
        "expansion_total": len(expansion),
        "compression_positive": sum(1 for record in compression if record["platform_delta"] > 0),
        "compression_total": len(compression),
        "fisher_exact_one_sided_p": hypergeometric_tail(
            sum(1 for record in expansion if record["platform_delta"] > 0), len(expansion), positives, len(records)),
    }

    positive_deltas = [record["platform_delta"] for record in records if record["platform_delta"] > 0]
    negative_deltas = [record["platform_delta"] for record in records if record["platform_delta"] <= 0]
    report = {
        "stage": spec["stage"],
        "kind": spec["kind"],
        "objective": spec["objective"],
        "reference": spec["reference"],
        "budget_actual": {
            "new_fits": 0, "training_label_reads": 0, "new_packages": 0,
            "desktop_writes": 0, "agent_uploads": 0,
        },
        "receipts_used": len(records),
        "receipts_rejected": rejected,
        "records": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in record.items() if k != "parent_column"}
            for record in records
        ],
        "association_tests": tests,
        "expansion_contingency": contingency,
        "natural_experiments": natural_experiments(records),
        "platform_delta_summary": {
            "n": len(records),
            "positive": len(positive_deltas),
            "non_positive": len(negative_deltas),
            "positive_mean": float(np.mean(positive_deltas)) if positive_deltas else None,
            "positive_max": float(np.max(positive_deltas)) if positive_deltas else None,
            "non_positive_mean": float(np.mean(negative_deltas)) if negative_deltas else None,
            "non_positive_min": float(np.min(negative_deltas)) if negative_deltas else None,
            "all_mean": float(platform.mean()),
            "target_gap_from_reference": 96.45 - spec["reference"]["score"],
        },
        "classification": classify(records, spec["slot_admission_rule"]),
        "limits": [
            "Every receipt shares the same frozen 322 query rows and most share the incumbent, so the receipts are not independent samples.",
            "Five statistics were screened without family-wise correction; a single nominally small p-value is descriptive only.",
            "No platform score may be forecast from any statistic here; the historical local/platform relation is not a transfer rule.",
            "The prospective rule is derived from these same receipts and has no independent validation yet.",
        ],
    }
    write_json(output / "report.json", report)
    write_json(output / "classification.json", report["classification"])
    print(json.dumps({
        "records": len(records), "rejected": len(rejected),
        "positive": len(positive_deltas), "non_positive": len(negative_deltas),
        "natural_experiments": len(report["natural_experiments"]),
        "association_tests": {k: {kk: vv for kk, vv in v.items() if kk != "ids"}
                              for k, v in tests.items()},
        "expansion_contingency": contingency,
        "classification_pass": len(report["classification"]["would_pass"]),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
