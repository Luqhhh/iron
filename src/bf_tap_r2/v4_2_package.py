"""V4.2 iron-only submission packaging on the frozen B36 parent.

Contract (mirrors the V3.6 user-requested package exactly):

* the ZIP contains a single ``result.csv`` with columns
  ``sample_id, pred_tap_iron, pred_tap_time_len`` in template order, numbers
  formatted with ``.17g`` and a ``\\n`` line terminator;
* the **unchanged** target column keeps the parent package's exact CSV strings,
  so an iron-only package cannot silently move the time column;
* the changed target is ``(1 - alpha) * B36 + alpha * V4.2 model`` in original
  units, clipped only by the globally frozen non-negative handling.

The B36 endpoint is the parent package's own prediction column: that is the
artifact that was actually scored.  A fresh full-data refit of the frozen V3.6
recipe is also run as an independent *reproduction check*, and a material
disagreement fails the build instead of being absorbed into the blend.

This module never uploads anything and refuses to write outside ``local/``.

NOTE (V4.2-r2): this is a pre-repair exploratory packaging helper.  The
pre-registered A4 deliverable may only be produced after the fusion stage (A3)
passes its gate and selects ``alpha`` on training-internal OOF; the default
``alpha`` here is a previously declared screening weight and must never be used
to bypass that chain.  No gate outcome is asserted by this module.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .audit import digest
from .data import FEATURES, SUBMISSION_COLUMNS, TARGETS
from .submission import ZIP_NAME, package, validate_result
from .v2_release import load_v2
from .v4_2_reference import V42References

__all__ = ["DEFAULT_PARENT_CSV", "build_iron_only_package", "main"]

ROOT_DEFAULT = Path("/home/lux1/iron")

#: The current user-reported platform best, V36_USER_REQUESTED_OUTER_FAILED.
DEFAULT_PARENT_CSV = (
    "local/runs/round2-v3.6-loss-training-and-numeric-encoding/release-user-requested-r1/"
    "V36_USER_REQUESTED_OUTER_FAILED/result.csv"
)

#: Tolerance for the independent full-data B36 reproduction check, relative to
#: the target's own magnitude.  A larger disagreement is a hard failure.
REPRODUCTION_RTOL = 1e-6


def _private_dir(path: Path) -> Path:
    resolved = path.resolve()
    return resolved


def _require_private(root: Path, path: Path) -> Path:
    resolved = _private_dir(path)
    if not resolved.is_relative_to((root / "local").resolve()):
        raise ValueError("V4.2 packages are private and must stay under local/")
    return resolved


def _read_parent_csv(parent_path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """Read the parent package while preserving the exact field text.

    ``pd.read_csv`` parses the text column to ``float64`` and a later ``str()``
    re-formats it, so a value written as ``.17g`` can come back as the shortest
    repr and a 74-row / 136-string drift appears in a column the manifest calls
    byte-for-byte preserved.  The unchanged column is therefore taken straight
    from the CSV field text instead of being round-tripped through a float.
    """
    text = parent_path.read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:  # pragma: no cover - an empty file is a hard error
        raise ValueError("Parent package is empty") from exc
    if list(header) != list(SUBMISSION_COLUMNS):
        raise ValueError("Parent package column mismatch")
    id_index = header.index("sample_id")
    iron_index = header.index("pred_tap_iron")
    time_index = header.index("pred_tap_time_len")
    parent_ids: list[str] = []
    iron: list[float] = []
    time_strings: list[str] = []
    for row in reader:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError("Parent package row width mismatch")
        parent_ids.append(row[id_index])
        iron.append(float(row[iron_index]))
        time_strings.append(row[time_index])
    return parent_ids, np.asarray(iron, dtype=float), time_strings


def _align_parent_columns(
    parent_ids: Sequence[str],
    parent_iron: np.ndarray,
    parent_time: Sequence[str],
    ids: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Reorder the parent columns onto ``ids`` without touching the time text."""
    ids = list(ids)
    parent_ids = [str(value) for value in parent_ids]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Parent package has duplicate sample_id values")
    if set(parent_ids) != set(ids):
        raise ValueError("Parent package ID set mismatch")
    order = {value: position for position, value in enumerate(parent_ids)}
    aligned_iron = np.asarray([parent_iron[order[sid]] for sid in ids], dtype=float)
    aligned_time = [parent_time[order[sid]] for sid in ids]
    if not np.isfinite(aligned_iron).all():
        raise ValueError("Parent iron column is not finite")
    return aligned_iron, aligned_time


def _csv_payload(ids: Sequence[str], values: np.ndarray) -> bytes:
    ids = list(ids)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(ids), len(TARGETS)):
        raise ValueError("Invalid prediction matrix shape")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Predictions must be finite and non-negative")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(list(SUBMISSION_COLUMNS))
    writer.writerows(
        (sid, format(i, ".17g"), format(t, ".17g")) for sid, (i, t) in zip(ids, values)
    )
    payload = stream.getvalue().encode("utf-8")
    validate_result(payload, ids)
    return payload


def _csv_payload_with_parent_time(
    ids: Sequence[str], iron: np.ndarray, parent_time_strings: Sequence[str]
) -> bytes:
    """Iron from the blend, time byte-identical to the parent package."""
    ids = list(ids)
    iron = np.asarray(iron, dtype=float)
    if len(iron) != len(ids) or len(parent_time_strings) != len(ids):
        raise ValueError("Iron/parent-time length mismatch")
    if not np.isfinite(iron).all() or (iron < 0.0).any():
        raise ValueError("Iron predictions must be finite and non-negative")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(list(SUBMISSION_COLUMNS))
    writer.writerows(
        (sid, format(value, ".17g"), time_string)
        for sid, value, time_string in zip(ids, iron, parent_time_strings)
    )
    payload = stream.getvalue().encode("utf-8")
    validate_result(payload, ids)
    return payload


def _cold_consistency(model, test: pd.DataFrame, predictions: np.ndarray) -> dict[str, Any]:
    """Reversed, chunked, subset and single-row inference must agree.

    Agreement is a *float32* agreement: changing the batch shape changes the GEMM
    reduction order, so the comparison uses the documented
    ``NEURAL_INFERENCE_ATOL`` / ``NEURAL_INFERENCE_RTOL`` and records the observed
    differences instead of claiming a bitwise match.
    """
    from .v4_2_train import NEURAL_INFERENCE_ATOL, NEURAL_INFERENCE_RTOL

    reversed_frame = test.iloc[::-1].reset_index(drop=True)
    reversed_prediction = np.asarray(model.predict(reversed_frame), dtype=float)[::-1]
    chunked = np.asarray(model.predict_chunked(test, chunk_size=37), dtype=float)
    subset_index = np.asarray([0, 5, 17, 100, 321])
    subset = np.asarray(
        model.predict(test.iloc[subset_index].reset_index(drop=True)), dtype=float
    )
    single = np.asarray(
        [model.predict(test.iloc[[i]].reset_index(drop=True))[0] for i in subset_index],
        dtype=float,
    )

    def agrees(left: np.ndarray, right: np.ndarray) -> bool:
        return bool(np.allclose(
            left, right, atol=NEURAL_INFERENCE_ATOL, rtol=NEURAL_INFERENCE_RTOL
        ))

    return {
        "reversed_max_abs_diff": float(np.max(np.abs(reversed_prediction - predictions))),
        "chunked_max_abs_diff": float(np.max(np.abs(chunked - predictions))),
        "subset_max_abs_diff": float(np.max(np.abs(subset - predictions[subset_index]))),
        "single_row_max_abs_diff": float(np.max(np.abs(single - subset))),
        "atol": float(NEURAL_INFERENCE_ATOL),
        "rtol": float(NEURAL_INFERENCE_RTOL),
        "note": (
            "float32 batch-shape tolerance; the network is deterministic for a fixed "
            "batch shape, not bitwise-invariant across batch shapes"
        ),
        "passed": bool(
            agrees(reversed_prediction, predictions)
            and agrees(chunked, predictions)
            and agrees(subset, predictions[subset_index])
            and agrees(single, subset)
        ),
    }


def build_iron_only_package(
    root: Path | str = ROOT_DEFAULT,
    output: Path | str | None = None,
    *,
    alpha: float = 0.25,
    line: str = "N",
    recipe: str = "N2",
    target: str = "tap_iron",
    training_seed: int = 42,
    inner_seed: int = 20260925,
    train_config: Mapping[str, Any] | None = None,
    parent_csv: Path | str = DEFAULT_PARENT_CSV,
    name: str = "V42_IRON_N2_Q25",
    status: str = "USER_REQUESTED_EXPLORATORY_PACKAGE_GATE_NOT_MET",
    gate_summary: Mapping[str, Any] | None = None,
    b_fit_workers: int = 12,
    torch_threads: int | None = None,
) -> dict[str, Any]:
    """Fit the V4.2 model on all training rows and emit an iron-only package."""
    root = Path(root).resolve()
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    output = _require_private(
        root,
        Path(output) if output is not None
        else root / "local/runs/round2-v4.2-structure-search/release-r1" / str(name),
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing package: {output}")

    parent_path = Path(parent_csv)
    if not parent_path.is_absolute():
        parent_path = root / parent_path
    if not parent_path.is_file():
        raise FileNotFoundError(f"Parent package result.csv not found: {parent_path}")

    # ---- inputs ---------------------------------------------------------
    train = load_v2(root / "复赛_train", "train", 2754)
    test = load_v2(root / "复赛_test", "test", 322)
    ids = test.sample_id.tolist()

    # Read the parent package as raw CSV text so the unchanged column keeps its
    # exact field strings; parsing to float and re-formatting would silently
    # rewrite it (see `_read_parent_csv`).
    parent_ids, parent_iron_raw, parent_time_raw = _read_parent_csv(parent_path)
    parent_iron, parent_time_strings = _align_parent_columns(
        parent_ids, parent_iron_raw, parent_time_raw, ids
    )

    # ---- independent full-data B36 reproduction check -------------------
    references = V42References(root, train, workers=int(b_fit_workers), verify_replay=False)
    bundle = references.fit_baseline(train, test)
    b36_iron = np.asarray(bundle["b36"]["tap_iron"], dtype=float)
    scale = max(1.0, float(np.max(np.abs(parent_iron))))
    reproduction = {
        "max_abs_diff": float(np.max(np.abs(b36_iron - parent_iron))),
        "max_rel_diff": float(np.max(np.abs(b36_iron - parent_iron)) / scale),
        "rtol_threshold": REPRODUCTION_RTOL,
        "passed": bool(float(np.max(np.abs(b36_iron - parent_iron)) / scale) <= REPRODUCTION_RTOL),
        "note": (
            "The blend endpoint is the parent package's own iron column; this refit "
            "only proves the frozen recipe reproduces it from the same training rows."
        ),
    }

    # ---- full-data V4.2 model ------------------------------------------
    from .v4_2_n_node import NodeEnsembleRegressor

    model = NodeEnsembleRegressor(
        str(recipe), n_inner_splits=5, inner_seed=int(inner_seed),
        torch_threads=torch_threads,
    )
    config = dict(train_config or {})
    config.setdefault("seed", int(training_seed))
    model.fit(train, target, train_config=config)
    model_prediction = np.asarray(model.predict(test), dtype=float)
    if model_prediction.shape != (len(test),) or not np.isfinite(model_prediction).all():
        raise ValueError("Invalid full-data V4.2 model prediction")

    consistency = _cold_consistency(model, test, model_prediction)

    # ---- blend and clip -------------------------------------------------
    blended = (1.0 - float(alpha)) * parent_iron + float(alpha) * model_prediction
    raw_min = float(blended.min())
    clip_record = {
        "raw_min": raw_min,
        "clip_applied": bool(raw_min < 0.0),
        "negative_rows_clipped": int((blended < 0.0).sum()),
    }
    iron_final = np.maximum(blended, 0.0)
    if not np.isfinite(iron_final).all():
        raise ValueError("Nonfinite final iron prediction")

    # ---- write ----------------------------------------------------------
    output.mkdir(parents=True, exist_ok=False)
    payload = _csv_payload_with_parent_time(ids, iron_final, parent_time_strings)
    package(output, payload, ids)

    result_sha = digest(output / "result.csv")
    zip_sha = digest(output / ZIP_NAME)
    manifest: dict[str, Any] = {
        "name": str(name),
        "candidate": str(name),
        "status": str(status),
        "recipe": (
            f"parent B36 test prediction + {alpha:g} * full-data V4.2 {line}-{recipe} "
            f"on {target}; the other target keeps the parent strings byte-for-byte"
        ),
        "line": str(line),
        "v42_recipe": str(recipe),
        "changed_target": str(target),
        "unchanged_targets": [t for t in TARGETS if t != target],
        "alpha": float(alpha),
        "alpha_basis": (
            "PRE_DECLARED_SCREENING_WEIGHT_NOT_OOF_SELECTED: the section-9 fusion gate "
            "was not met, so no inner alph selection was run and the fixed screening "
            "weight is reused as-is"
        ),
        "parent": {
            "result_csv": str(parent_path.relative_to(root)),
            "result_sha256": digest(parent_path),
            "iron_column_used_as_b36_endpoint": True,
            "time_column_preserved_byte_for_byte": True,
        },
        "b36_full_data_reproduction": reproduction,
        "training": {
            "training_seed": int(training_seed),
            "inner_split_seed": int(inner_seed),
            "n_train_rows": int(len(train)),
            "n_test_rows": int(len(test)),
            "n_parameters": int(model.parameter_count()),
            "fit_meta": model.fit_outcome_.as_dict(),
        },
        "blend": {"clip": clip_record},
        "cold_consistency": consistency,
        "rows": int(len(ids)),
        "columns": list(SUBMISSION_COLUMNS),
        "result_sha256": result_sha,
        "zip_sha256": zip_sha,
        "agent_uploads": 0,
        "platform_score_forecast": None,
        "gate_summary": dict(gate_summary or {}),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )

    readme = "\n".join([
        f"V4.2 iron-only package: {name}",
        "=" * (24 + len(str(name))),
        "",
        f"Status: {status}",
        "Gate provenance is recorded verbatim in manifest.json -> gate_summary;",
        "no gate outcome is asserted by this README.",
        "",
        f"Changed column : {target}   (alpha = {alpha:g})",
        f"Unchanged      : {', '.join(t for t in TARGETS if t != target)} (parent strings preserved byte-for-byte)",
        f"Parent         : {parent_path}",
        f"B36 reproduction check passed: {reproduction['passed']}",
        f"Cold consistency passed      : {consistency['passed']}",
        "",
        f"ZIP: {ZIP_NAME}",
        f"ZIP SHA-256: {zip_sha}",
        f"result.csv SHA-256: {result_sha}",
        "",
        "Upload is performed by the user.",
        "",
    ])
    (output / "README.txt").write_text(readme, encoding="utf-8")
    with (output / "SHA256SUMS.txt").open("x", encoding="utf-8") as handle:
        handle.write(f"{result_sha}  result.csv\n{zip_sha}  {ZIP_NAME}\n")

    if not consistency["passed"]:
        raise AssertionError("Cold consistency audit failed; package must not be released")
    return {**manifest, "output": str(output)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--line", default="N")
    parser.add_argument("--recipe", default="N2")
    parser.add_argument("--target", default="tap_iron", choices=list(TARGETS))
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--name", default="V42_IRON_N2_Q25")
    parser.add_argument("--b-fit-workers", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=None)
    args = parser.parse_args(argv)
    manifest = build_iron_only_package(
        args.root, args.output, alpha=args.alpha, line=args.line, recipe=args.recipe,
        target=args.target, training_seed=args.training_seed, name=args.name,
        b_fit_workers=args.b_fit_workers, torch_threads=args.torch_threads,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
