"""Round2 V5 single-target replacement packaging on the frozen V36 parent.

This generalises :mod:`bf_tap_r2.v4_2_package` from an iron-only helper to a
target-agnostic one, keeping its contracts:

* the ZIP holds a single ``result.csv`` with ``sample_id, pred_tap_iron,
  pred_tap_time_len`` in template order, ``.17g`` numbers and ``\\n`` endings;
* the **unchanged** target keeps the parent package's exact CSV field text, so
  a single-column package cannot silently move the other column;
* the changed target is ``(1 - alpha) * parent + alpha * member`` in original
  units, clipped only by the frozen non-negative handling;
* the unchanged column is re-read from the written file and compared to the
  parent strings before the package is returned.

An optional full-data reproduction check and an optional cold-inference audit
are accepted as callbacks so the builder stays independent of the member's
model family.  This module never uploads anything and refuses to write outside
``local/``.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .audit import digest
from .data import SUBMISSION_COLUMNS, TARGETS
from .submission import ZIP_NAME, package, validate_result
from .v2_release import load_v2

__all__ = [
    "DEFAULT_PARENT_CSV",
    "DEFAULT_V34A_PACKAGE",
    "read_parent_columns",
    "payload_with_parent_other_column",
    "build_single_target_replacement_package",
    "main",
]

ROOT_DEFAULT = Path("/home/lux1/iron")

#: The current user-reported platform best, V36_USER_REQUESTED_OUTER_FAILED.
DEFAULT_PARENT_CSV = (
    "local/runs/round2-v3.6-loss-training-and-numeric-encoding/release-user-requested-r1/"
    "V36_USER_REQUESTED_OUTER_FAILED/result.csv"
)

#: The V34_A release package supplies the A endpoint for full-data builds.
DEFAULT_V34A_PACKAGE = (
    "local/runs/round2-v3.4-ebm-and-constrained-composition/release-candidates-r1/"
    "01_V34_A_MECHANICAL_CONSTRAINED/result.csv"
)

MemberPredictor = Callable[[Any, Any], np.ndarray]
ReproductionCheck = Callable[[Any, Mapping[str, float]], Mapping[str, Any]]
ColdCheck = Callable[[Any], Mapping[str, Any]]


def _require_private(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to((root / "local").resolve()):
        raise ValueError("V5 packages are private and must stay under local/")
    return resolved


def read_parent_columns(parent_path: Path) -> tuple[list[str], dict[str, tuple[np.ndarray, list[str]]]]:
    """Read a parent package keeping the exact field text of every column.

    ``pd.read_csv`` + ``str()`` re-formats values, which previously drifted the
    "byte-for-byte preserved" column.  The unchanged column is therefore taken
    straight from the CSV field text.
    """
    text = Path(parent_path).read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:  # pragma: no cover - empty file is a hard error
        raise ValueError("Parent package is empty") from exc
    if list(header) != list(SUBMISSION_COLUMNS):
        raise ValueError("Parent package column mismatch")
    indexes = {name: header.index(name) for name in SUBMISSION_COLUMNS}
    ids: list[str] = []
    values: dict[str, tuple[list[float], list[str]]] = {
        target: ([], []) for target in TARGETS
    }
    for row in reader:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError("Parent package row width mismatch")
        ids.append(row[indexes["sample_id"]])
        for target in TARGETS:
            column = f"pred_{target}"
            text_value = row[indexes[column]]
            values[target][0].append(float(text_value))
            values[target][1].append(text_value)
    out = {target: (np.asarray(values[target][0], dtype=float), values[target][1]) for target in TARGETS}
    return ids, out


def _align_parent(ids: Sequence[str], parent_ids: Sequence[str],
                  columns: Mapping[str, tuple[np.ndarray, Sequence[str]]]
                  ) -> dict[str, tuple[np.ndarray, list[str]]]:
    ids = [str(v) for v in ids]
    parent_ids = [str(v) for v in parent_ids]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Parent package has duplicate sample_id values")
    if set(parent_ids) != set(ids):
        raise ValueError("Parent package ID set mismatch")
    order = {value: position for position, value in enumerate(parent_ids)}
    aligned: dict[str, tuple[np.ndarray, list[str]]] = {}
    for target, (values, strings) in columns.items():
        array = np.asarray([values[order[sid]] for sid in ids], dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"Parent {target} column is not finite")
        aligned[target] = (array, [strings[order[sid]] for sid in ids])
    return aligned


def payload_with_parent_other_column(ids: Sequence[str], target: str, values: np.ndarray,
                                     other_strings: Sequence[str]) -> bytes:
    """Changed column formatted from floats; the other column keeps parent text."""
    ids = list(ids)
    values = np.asarray(values, dtype=float)
    if target not in TARGETS:
        raise ValueError(f"Unknown V5 packaging target: {target!r}")
    if len(values) != len(ids) or len(other_strings) != len(ids):
        raise ValueError("V5 packaging column length mismatch")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("V5 predictions must be finite and non-negative")
    other = next(t for t in TARGETS if t != target)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(list(SUBMISSION_COLUMNS))
    for sid, value, other_text in zip(ids, values, other_strings):
        row = {target: format(float(value), ".17g"), other: other_text}
        writer.writerow([sid, row["tap_iron"], row["tap_time_len"]])
    payload = stream.getvalue().encode("utf-8")
    validate_result(payload, ids)
    return payload


def _verify_unchanged_column(result_path: Path, target: str,
                             expected_strings: Sequence[str]) -> dict[str, Any]:
    reader = csv.reader(io.StringIO(result_path.read_text(encoding="utf-8"), newline=""))
    header = next(reader)
    other = next(t for t in TARGETS if t != target)
    index = header.index(f"pred_{other}")
    observed = [row[index] for row in reader if row]
    if len(observed) != len(expected_strings):
        raise ValueError("V5 unchanged-column length mismatch")
    mismatches = [position for position, (left, right) in enumerate(zip(observed, expected_strings))
                  if left != right]
    return {
        "target": other,
        "rows": len(observed),
        "mismatches": len(mismatches),
        "first_mismatch": (int(mismatches[0]) if mismatches else None),
        "byte_identical": not mismatches,
    }


def build_single_target_replacement_package(
    root: Path | str = ROOT_DEFAULT,
    output: Path | str | None = None,
    *,
    target: str,
    alpha: float,
    member_predictor: MemberPredictor,
    name: str,
    status: str,
    parent_csv: Path | str = DEFAULT_PARENT_CSV,
    alpha_basis: str = "V5_NESTED_OOF_SELECTED_WEIGHT",
    member_meta: Mapping[str, Any] | None = None,
    reproduction_check: ReproductionCheck | None = None,
    cold_check: ColdCheck | None = None,
    gate_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one isolated single-column replacement package under ``local/``."""
    root = Path(root).resolve()
    if target not in TARGETS:
        raise ValueError(f"Unknown V5 packaging target: {target!r}")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    out = _require_private(root, Path(output) if output is not None
                           else root / "local/runs/round2-v5-error-covariance/release-r1" / str(name))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing package: {out}")
    parent_path = Path(parent_csv)
    if not parent_path.is_absolute():
        parent_path = root / parent_path
    if not parent_path.is_file():
        raise FileNotFoundError(f"Parent package result.csv not found: {parent_path}")

    train = load_v2(root / "复赛_train", "train", 2754)
    test = load_v2(root / "复赛_test", "test", 322)
    ids = test.sample_id.tolist()
    parent_ids, parent_columns = read_parent_columns(parent_path)
    aligned = _align_parent(ids, parent_ids, parent_columns)
    parent_values, parent_strings = aligned[target]
    other = next(t for t in TARGETS if t != target)
    other_strings = aligned[other][1]

    member_prediction = np.asarray(member_predictor(train, test), dtype=float)
    if member_prediction.shape != (len(ids),) or not np.isfinite(member_prediction).all():
        raise ValueError("V5 member predictor returned an invalid column")

    blended = (1.0 - float(alpha)) * parent_values + float(alpha) * member_prediction
    clip_record = {
        "raw_min": float(blended.min()),
        "clip_applied": bool(float(blended.min()) < 0.0),
        "negative_rows_clipped": int((blended < 0.0).sum()),
    }
    changed = np.maximum(blended, 0.0)

    reproduction = dict(reproduction_check(train, test) if reproduction_check else {})
    cold = dict(cold_check(member_prediction) if cold_check else {})

    out.mkdir(parents=True, exist_ok=False)
    payload = payload_with_parent_other_column(ids, target, changed, other_strings)
    package(out, payload, ids)
    result_sha = digest(out / "result.csv")
    zip_sha = digest(out / ZIP_NAME)
    unchanged = _verify_unchanged_column(out / "result.csv", target, other_strings)
    if not unchanged["byte_identical"]:
        raise AssertionError("Unchanged column is not byte-identical; package must not be released")

    manifest: dict[str, Any] = {
        "name": str(name),
        "candidate": str(name),
        "status": str(status),
        "recipe": (
            f"parent V36 test column + {alpha:g} * full-data V5 member on {target}; "
            f"{other} keeps the parent strings byte-for-byte"
        ),
        "changed_target": target,
        "unchanged_targets": [other],
        "alpha": float(alpha),
        "alpha_basis": str(alpha_basis),
        "parent": {
            "result_csv": str(parent_path.relative_to(root)),
            "result_sha256": digest(parent_path),
            "changed_column_used_as_blend_endpoint": True,
            "unchanged_column_preserved_byte_for_byte": True,
        },
        "unchanged_column_verification": unchanged,
        "member": dict(member_meta or {}),
        "reproduction_check": reproduction,
        "cold_check": cold,
        "blend": {"clip": clip_record},
        "rows": int(len(ids)),
        "columns": list(SUBMISSION_COLUMNS),
        "result_sha256": result_sha,
        "zip_sha256": zip_sha,
        "agent_uploads": 0,
        "platform_score_forecast": None,
        "gate_summary": dict(gate_summary or {}),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    readme = "\n".join([
        f"V5 single-target replacement package: {name}",
        "=" * (36 + len(str(name))),
        "",
        f"Status: {status}",
        "Gate provenance is recorded verbatim in manifest.json -> gate_summary;",
        "no gate outcome is asserted by this README.",
        "",
        f"Changed column : {target}   (alpha = {alpha:g})",
        f"Unchanged      : {other} (parent strings preserved byte-for-byte)",
        f"Parent         : {parent_path}",
        f"Cold check     : {cold.get('passed')}",
        "",
        f"ZIP: {ZIP_NAME}",
        f"ZIP SHA-256: {zip_sha}",
        f"result.csv SHA-256: {result_sha}",
        "",
        "Upload is performed by the user.",
        "",
    ])
    (out / "README.txt").write_text(readme, encoding="utf-8")
    with (out / "SHA256SUMS.txt").open("x", encoding="utf-8") as handle:
        handle.write(f"{result_sha}  result.csv\n{zip_sha}  {ZIP_NAME}\n")
    if cold and cold.get("passed") is False:
        raise AssertionError("Cold-inference audit failed; package must not be released")
    return {**manifest, "output": str(out)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--parent-csv", default=DEFAULT_PARENT_CSV)
    args = parser.parse_args(argv)
    train = load_v2(Path(args.root) / "复赛_train", "train", 2754)
    test = load_v2(Path(args.root) / "复赛_test", "test", 322)
    ids = test.sample_id.tolist()
    parent_ids, columns = read_parent_columns(Path(args.root) / args.parent_csv)
    aligned = _align_parent(ids, parent_ids, columns)
    print(json.dumps({
        "rows": len(ids),
        "train_rows": int(len(train)),
        "parent": str(args.parent_csv),
        "changed_column_endpoints": {t: float(aligned[t][0].mean()) for t in TARGETS},
        "agent_uploads": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
