"""Round2 V5 unified zero-fit OOF candidate library.

The library is built only from already-recorded out-of-fold predictions under
``local/``.  It performs no model fitting.  Every entry is keyed by
``(source, seed, trial id)`` because the same trial id appears in more than one
run directory with a different training convention (the V3 coarse batches used
early stopping; the V3 refine batches did not).

Nothing here selects, promotes, packages or uploads anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import TARGETS
from .splits import make_folds
from .v5_spec import V5Spec, load_v5_spec

__all__ = [
    "TRAIN_ROWS",
    "TEST_ROWS",
    "LibraryEntry",
    "CandidateLibrary",
    "ColumnReference",
    "load_v5_training_frame",
    "fold_vector",
    "build_candidate_library",
    "load_column_reference",
    "library_audit",
    "main",
]

TRAIN_ROWS = 2754
TEST_ROWS = 322

TARGET_SET = tuple(TARGETS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v5_training_frame(root: Path | str) -> pd.DataFrame:
    """Load the frozen round-two training frame in its recorded row order."""
    from .v3_run import load_training_frame

    train = load_training_frame(Path(root))
    if len(train) != TRAIN_ROWS:
        raise ValueError(f"V5 training frame has {len(train)} rows, expected {TRAIN_ROWS}")
    return train


def fold_vector(root: Path | str, train: pd.DataFrame, seed: int,
                spec: V5Spec | None = None) -> np.ndarray:
    """Return the fold assignment for ``seed`` aligned to ``train`` row order.

    A frozen fold file is used when one is declared by the specification;
    otherwise the assignment is derived with the same deterministic
    group-safe stratified routine used for the original folds.  Derived seeds
    are declared in the pre-registration before use.
    """
    root = Path(root)
    declared = dict((spec.raw["reference"]["frozen_folds"] if spec else {}))
    # YAML may parse the numeric seed keys as integers; normalise both spellings
    # so a frozen fold file is never skipped in favour of a derived assignment.
    declared_by_seed = {int(key): str(value) for key, value in declared.items()}
    path = (root / declared_by_seed[int(seed)]) if int(seed) in declared_by_seed else None
    if path is not None and path.is_file():
        frame = pd.read_csv(path, dtype={"sample_id": "string"})
        if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(train.sample_id):
            raise ValueError(f"V5 frozen fold identity mismatch: {path}")
        frame = frame.set_index("sample_id").loc[train.sample_id]
        folds = frame["fold"].to_numpy()
        if set(folds) != set(range(5)) or not (frame["seed"] == int(seed)).all():
            raise ValueError(f"V5 frozen fold vector is invalid: {path}")
        return folds
    frame = make_folds(train, int(seed)).set_index("sample_id").loc[train.sample_id]
    folds = frame["fold"].to_numpy()
    if set(folds) != set(range(5)):
        raise ValueError("V5 derived fold vector is incomplete")
    return folds


@dataclass
class LibraryEntry:
    """One candidate prediction column over one or more split seeds."""

    key: str
    name: str
    source: str
    target: str
    family: str
    convention: str
    coverage: str
    folds: tuple[int, ...]
    vectors: dict[int, np.ndarray]
    hashes: dict[int, str] = field(default_factory=dict)
    packagability: str = "P2"
    packagability_note: str = ""

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted(self.vectors))

    def coverage_rows(self) -> int:
        return int(np.isfinite(next(iter(self.vectors.values()))).sum())


@dataclass
class CandidateLibrary:
    """All candidate columns per target, plus the audit trail."""

    feed_entries: dict[str, dict[str, LibraryEntry]]
    audit: dict[str, Any]

    def entries(self, target: str) -> dict[str, LibraryEntry]:
        return dict(self.feed_entries[str(target)])

    def complete_entries(self, target: str, min_seeds: int = 2) -> dict[str, LibraryEntry]:
        """Entries with full row coverage on at least ``min_seeds`` split seeds."""
        out: dict[str, LibraryEntry] = {}
        for key, entry in self.feed_entries[str(target)].items():
            if len(entry.vectors) < int(min_seeds):
                continue
            if all(np.isfinite(v).all() for v in entry.vectors.values()):
                out[key] = entry
        return out


@dataclass
class ColumnReference:
    """The frozen V36 column per target and seed, with its member vectors."""

    weights: dict[str, list[float]]
    members: dict[str, list[str]]
    a_dev: dict[int, dict[str, np.ndarray]]
    base: dict[int, dict[str, np.ndarray]]
    expert_vectors: dict[int, dict[str, dict[str, np.ndarray]]]
    summary_path: Path
    summary_sha256: str

    def base_for(self, target: str, seed: int) -> np.ndarray:
        return np.asarray(self.base[int(seed)][str(target)], dtype=float)

    def member_vector(self, target: str, seed: int, name: str) -> np.ndarray:
        if name == "A_dev":
            return np.asarray(self.a_dev[int(seed)][str(target)], dtype=float)
        return np.asarray(self.expert_vectors[int(seed)][str(target)][name], dtype=float)


def _family_of(trial_id: str) -> str:
    """Recipe family of a trial id.

    ``v3-catboost-tap_iron-0034`` -> ``catboost``; ``v36-s1-N-0047`` -> ``N``.
    """
    parts = str(trial_id).split("-")
    if not parts:
        return "unknown"
    if parts[0] == "v36":
        return parts[2] if len(parts) > 2 else "unknown"
    if parts[0] == "v3":
        return parts[1] if len(parts) > 1 else "unknown"
    return parts[1] if len(parts) > 1 else "unknown"


def _target_of(trial_id: str) -> str:
    for target in TARGET_SET:
        if f"-{target}-" in f"-{trial_id}-" or str(trial_id) == target:
            return target
    return "unknown"


def _coverage_kind(folds: Sequence[int]) -> str:
    return "full" if tuple(sorted(int(f) for f in folds)) == (0, 1, 2, 3, 4) else "folds01"


def _load_v36_ledger(root: Path, spec: V5Spec) -> dict[str, dict[str, Any]]:
    path = root / str(spec.raw["reference"]["v36_coarse_ledger"])
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"V5 requires the V3.6 fixed-batch ledger: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "complete":
            records[str(event["trial_id"])] = event
    return records


def _load_v3_coarse_ledgers(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ledger in sorted((root / "local/runs/round2-v3-local-search").glob("coarse-*-r1/fit_ledger.jsonl")):
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "complete":
                out[str(event["trial_id"])] = event
    return out


def _npy_entries(root: Path, source: Mapping[str, Any], spec: V5Spec,
                 train: pd.DataFrame, v36_ledger: Mapping[str, dict[str, Any]]) -> list[LibraryEntry]:
    """Load a ``pred-*.npy`` source declared in the specification."""
    name = str(source["name"])
    base = root / str(source["path"])
    seeds = tuple(int(v) for v in source["seeds"])
    folds = tuple(int(v) for v in source["folds"])
    pattern = str(source["pattern"])
    convention = str(source.get("early_stopping", "recorded"))
    out: list[LibraryEntry] = []
    if not base.is_dir():
        raise FileNotFoundError(f"V5 library source missing: {base}")

    grouped: dict[str, dict[int, Path]] = {}
    if name == "v36_dev_experts":
        for path in sorted(base.glob("seed-*/pred-*.npy")):
            seed = int(path.parent.name.split("-")[1])
            trial_id = path.name[len("pred-"):-len(".npy")]
            grouped.setdefault(trial_id, {})[seed] = path
        packagability, note = "P1", "recipe recoverable from the V3.6 fixed-batch ledger"
    elif name == "v36_coarse_diagnostic":
        for path in sorted(base.glob("pred-*.npy")):
            trial_id = path.name[len("pred-"):-len(".npy")]
            grouped.setdefault(trial_id, {})[42] = path
        packagability, note = "P1", "recipe recoverable from the V3.6 fixed-batch ledger"
    else:
        for path in sorted(base.glob("pred-*.npy")):
            stem = path.name[len("pred-"):-len(".npy")]
            parts = stem.split("-", 1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            seed, trial_id = int(parts[0]), parts[1]
            grouped.setdefault(trial_id, {})[seed] = path
        packagability, note = "P1", "recipe recoverable from the V3 coarse ledger"

    for trial_id, by_seed in sorted(grouped.items()):
        if not set(seeds).issubset(by_seed):
            continue
        record = v36_ledger.get(trial_id)
        target = str(record["target"]) if record else _target_of(trial_id)
        if target not in TARGET_SET:
            continue
        vectors: dict[int, np.ndarray] = {}
        hashes: dict[int, str] = {}
        for seed in seeds:
            path = by_seed[seed]
            values = np.load(path).astype(float, copy=False)
            if values.shape != (len(train),):
                raise ValueError(f"V5 candidate shape mismatch: {path}")
            mask = np.isin(fold_vector(root, train, seed, spec), list(folds))
            if not np.isfinite(values[mask]).all():
                raise ValueError(f"V5 candidate coverage failure: {path}")
            if np.isfinite(values[~mask]).any():
                raise ValueError(f"V5 candidate carries out-of-fold values: {path}")
            vectors[seed] = values
            hashes[seed] = _sha256_file(path)
        out.append(LibraryEntry(
            key=f"{name}:{trial_id}",
            name=trial_id,
            source=name,
            target=target,
            family=_family_of(trial_id),
            convention=convention,
            coverage=_coverage_kind(folds),
            folds=folds,
            vectors=vectors,
            hashes=hashes,
            packagability=packagability,
            packagability_note=note,
        ))
    return out


def _v2_entries(root: Path, spec: V5Spec, train: pd.DataFrame,
                fold_lookup: Mapping[int, np.ndarray]) -> list[LibraryEntry]:
    """Load every numeric OOF column of every recorded ``round2-v2*`` run."""
    source = next(s for s in spec.library_sources if str(s["name"]) == "v2_oof_columns")
    seeds = tuple(int(v) for v in source["seeds"])
    folds = tuple(int(v) for v in source["folds"])
    root_runs = root / str(source["path"])
    grouped: dict[tuple[str, str], dict[int, Path]] = {}
    for path in sorted(root_runs.glob("round2-v2*/**/oof/*.csv")):
        stem = path.stem
        parts = stem.split("-", 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        seed, target = int(parts[0]), parts[1]
        if target not in TARGET_SET or seed not in seeds:
            continue
        tag = str(path.parent.parent.relative_to(root_runs))
        grouped.setdefault((tag, target), {})[seed] = path

    out: list[LibraryEntry] = []
    for (tag, target), by_seed in sorted(grouped.items()):
        columns: dict[str, dict[int, np.ndarray]] = {}
        hashes: dict[str, dict[int, str]] = {}
        for seed, path in sorted(by_seed.items()):
            frame = pd.read_csv(path, dtype={"sample_id": "string"})
            if len(frame) != len(train) or set(frame.sample_id) != set(train.sample_id):
                raise ValueError(f"V5 OOF identity mismatch: {path}")
            frame = frame.set_index("sample_id").loc[train.sample_id]
            if "fold" in frame.columns:
                observed = frame["fold"].to_numpy()
                if not np.array_equal(observed, fold_lookup[int(seed)]):
                    raise ValueError(f"V5 OOF fold alignment failure: {path}")
            digest = _sha256_file(path)
            for column in frame.columns:
                if column in {"spout_no", "fold", target}:
                    continue
                if not pd.api.types.is_numeric_dtype(frame[column]):
                    continue
                values = frame[column].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    continue
                columns.setdefault(str(column), {})[seed] = values
                hashes.setdefault(str(column), {})[seed] = digest
        for column, vectors in sorted(columns.items()):
            if not set(seeds).issubset(vectors):
                continue
            out.append(LibraryEntry(
                key=f"v2:{tag}:{column}",
                name=f"{tag}:{column}",
                source="v2_oof_columns",
                target=target,
                family="v2",
                convention="recorded",
                coverage=_coverage_kind(folds),
                folds=folds,
                vectors=vectors,
                hashes=hashes[column],
                packagability="P1",
                packagability_note="recipe recorded in the owning round2-v2 run",
            ))
    return out


def build_candidate_library(root: Path | str, spec: V5Spec,
                            train: pd.DataFrame | None = None) -> CandidateLibrary:
    """Build the zero-fit candidate library and its audit record."""
    root = Path(root).resolve()
    train = load_v5_training_frame(root) if train is None else train
    folds = {int(seed): fold_vector(root, train, int(seed), spec)
             for seed in spec.raw["resolution"]["new_seed_derivation"]["frozen_seeds"]}
    v36_ledger = _load_v36_ledger(root, spec)
    v3_ledger = _load_v3_coarse_ledgers(root)
    merged_ledger = {**v3_ledger, **v36_ledger}

    entries: list[LibraryEntry] = []
    source_counts: dict[str, int] = {}
    for source in spec.library_sources:
        name = str(source["name"])
        if name == "v2_oof_columns":
            loaded = _v2_entries(root, spec, train, folds)
        else:
            loaded = _npy_entries(root, source, spec, train, merged_ledger)
        source_counts[name] = len(loaded)
        entries.extend(loaded)

    per_target: dict[str, dict[str, LibraryEntry]] = {target: {} for target in TARGET_SET}
    duplicates: list[str] = []
    for entry in entries:
        bucket = per_target.setdefault(entry.target, {})
        if entry.key in bucket:
            duplicates.append(entry.key)
            continue
        bucket[entry.key] = entry
    if duplicates:
        raise ValueError(f"V5 library contains duplicate keys: {duplicates[:5]}")

    audit = {
        "version": spec.version,
        "train_rows": int(len(train)),
        "frozen_seeds": sorted(folds),
        "source_counts": source_counts,
        "entries_total": len(entries),
        "entries_per_target": {t: len(per_target[t]) for t in TARGET_SET},
        "full_coverage_per_target": {
            t: sum(1 for e in per_target[t].values()
                   if e.coverage == "full" and all(np.isfinite(v).all() for v in e.vectors.values()))
            for t in TARGET_SET
        },
        "two_seed_full_per_target": {
            t: sum(1 for e in per_target[t].values()
                   if e.coverage == "full" and len(e.vectors) >= 2
                   and all(np.isfinite(v).all() for v in e.vectors.values()))
            for t in TARGET_SET
        },
        "agents_uploads": 0,
    }
    return CandidateLibrary(feed_entries=per_target, audit=audit)


def load_column_reference(root: Path | str, train: pd.DataFrame,
                          spec: V5Spec) -> ColumnReference:
    """Reconstruct the frozen V36 column (zero fits) with all member vectors."""
    root = Path(root).resolve()
    summary_path = root / str(spec.raw["reference"]["v36_summary"])
    if not summary_path.is_file():
        raise FileNotFoundError(f"V5 requires the released V36 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    composition = summary.get("complete_development_composition")
    if not isinstance(composition, dict) or composition.get("reference") != "A_development_replay":
        raise ValueError("V5 requires the released A-development V36 composition")

    from .v4_run import _load_a_dev_reference

    a_dev = _load_a_dev_reference(root, train)
    cache_root = root / str(spec.raw["reference"]["v36_development_cache"])
    weights: dict[str, list[float]] = {}
    members: dict[str, list[str]] = {}
    base: dict[int, dict[str, np.ndarray]] = {}
    expert_vectors: dict[int, dict[str, dict[str, np.ndarray]]] = {}

    target_records = composition.get("targets", {})
    if set(map(str, target_records)) != set(TARGET_SET):
        raise ValueError("V5 requires both targets in the V36 composition")
    for target in TARGET_SET:
        record = target_records[target]
        selected = [str(v) for v in record["selected_experts"]]
        target_weights = [float(v) for v in record["weights"]]
        if len(target_weights) != 1 + len(selected):
            raise ValueError(f"V5 V36 weight/member mismatch for {target}")
        weights[target] = target_weights
        members[target] = selected

    for seed in sorted(a_dev):
        base[seed] = {}
        expert_vectors[seed] = {}
        for target in TARGET_SET:
            target_weights = weights[target]
            values = target_weights[0] * np.asarray(a_dev[seed][target], dtype=float)
            store: dict[str, np.ndarray] = {}
            for offset, name in enumerate(members[target], start=1):
                path = cache_root / f"seed-{int(seed)}" / f"pred-{name}.npy"
                if not path.is_file():
                    raise FileNotFoundError(path)
                vector = np.load(path).astype(float, copy=False)
                if vector.shape != (len(train),) or not np.isfinite(vector).all():
                    raise ValueError(f"V5 V36 expert cache is invalid: {path}")
                store[name] = vector
                values = values + target_weights[offset] * vector
            expert_vectors[seed][target] = store
            base[seed][target] = np.maximum(values, 0.0)

    return ColumnReference(
        weights=weights,
        members=members,
        a_dev=a_dev,
        base=base,
        expert_vectors=expert_vectors,
        summary_path=summary_path,
        summary_sha256=_sha256_file(summary_path),
    )


def library_audit(root: Path | str, spec: V5Spec | None = None) -> dict[str, Any]:
    """Return the audit payload for the current library state (no fitting)."""
    root = Path(root).resolve()
    spec = spec or load_v5_spec(root)
    library = build_candidate_library(root, spec)
    train = load_v5_training_frame(root)
    reference = load_column_reference(root, train, spec)
    payload = dict(library.audit)
    payload["column_reference"] = {
        "summary": str(reference.summary_path.relative_to(root)),
        "summary_sha256": reference.summary_sha256,
        "weights": {t: [float(v) for v in reference.weights[t]] for t in TARGET_SET},
        "selected_experts": {t: list(reference.members[t]) for t in TARGET_SET},
        "seeds": sorted(reference.base),
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None,
                        help="optional private path for the audit JSON")
    args = parser.parse_args(argv)
    payload = library_audit(Path(args.root))
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
