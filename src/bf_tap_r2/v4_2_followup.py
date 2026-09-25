"""Round2 V4.2-r2 follow-up: N2 seed-initialisation repair and full coverage.

This module is the *new* runner for the N2 repair.  It deliberately does not
reuse the historical 96-slot cache identity:

* every slot carries a full candidate identity (data hash, exact training and
  evaluation IDs and order, group hash, fold vector, model recipe, encoder and
  target transform, initialisation and batch-order seeds, a model/training source
  digest and the dependency identity);
* the ledger key is that identity digest plus the slot, so a source or seed
  change can never hit the old N cache;
* ``B_fit`` baselines are reused only when their *own* identity can be verified;
  otherwise they are refitted.  ``B_replay`` is never substituted for ``B_fit``;
* the historical runs (``coarse-r1``, ``full-r1``, ``repl-t3407-r1``) stay
  untouched and are preserved as diagnostics of the pre-repair protocol.

The stage machine is explicit about incomplete coverage: with folds 0/1 only,
the full-coverage gate is ``NOT_EVALUATED_INCOMPLETE_COVERAGE``, never reported
as a failure or a pass.

Nothing here reads test labels, packages a submission or uploads anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import v4_2_screen as screen
from .data import TARGETS
from .v3_run import load_fold_vector, load_training_frame
from .v4_2_prep import exclusion_group_keys
from .v4_2_reference import V42References, baseline_cache_identity
from .v4_2_rng import dependency_identity, source_digest, source_digest_payload
from .v4_2_spec import DEFAULT_SPEC_PATH, SearchSpec, load_search_spec

__all__ = [
    "COARSE_FOLDS",
    "FOLLOWUP_REPAIR_VERSION",
    "FULL_FOLDS",
    "FUSION_ALPHA_GRID",
    "REPAIR_TRIAL_ID_SUFFIX",
    "b_fit_identity",
    "candidate_identity",
    "candidate_identity_digest",
    "evaluate_stages",
    "fusion_protocol_plan",
    "load_followup_declarations",
    "run_followup",
    "select_units",
]

#: Repair identity of this round.  It appears in the trial id, the run
#: directory, the ledger key and every manifest, so pre- and post-repair
#: evidence can never be silently merged.
FOLLOWUP_REPAIR_VERSION = "SEED_INIT_V2"
REPAIR_TRIAL_ID_SUFFIX = f"-{FOLLOWUP_REPAIR_VERSION}"


def trial_id_suffix(spec: SearchSpec) -> str:
    """The candidate-id suffix for a spec's schema version.

    The V4.2 repair keeps ``-SEED_INIT_V2`` verbatim so its recorded identities
    are unchanged; a later schema gets its own suffix instead of inheriting a
    repair name it did not perform.
    """
    schema = str(spec.raw.get("schema_version"))
    if schema == "v4.2":
        return REPAIR_TRIAL_ID_SUFFIX
    return "-" + schema.upper().replace(".", "_")

#: The coarse screen is folds 0/1; full coverage adds 2/3/4.
COARSE_FOLDS: tuple[int, ...] = (0, 1)
FULL_FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: The pre-registered fusion alpha grid; ties go to the smaller alpha.
FUSION_ALPHA_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)

DEFAULT_FOLLOWUP_OUTPUT = "local/runs/round2-v4.2-structure-search/n2-seed-init-v2"
#: Declarations for the repair identity and the design-only next hypothesis.
FOLLOWUP_SPEC_PATH = "configs/round2_v4_2/FOLLOWUP_SPEC.yaml"
#: Pre-repair evidence whose frozen ``B_fit`` vectors may be reused once their
#: stored identity is verified field by field.  ``coarse-r1`` covers folds 0/1,
#: ``full-r1`` covers folds 0-4.
DEFAULT_SOURCE_OUTPUT = "local/runs/round2-v4.2-structure-search/coarse-r1"
DEFAULT_FULL_SOURCE_OUTPUT = "local/runs/round2-v4.2-structure-search/full-r1"

#: The inner U/H split seed used by the shared screen worker for every line.
SCREEN_INNER_SEED = 20260925


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def load_followup_declarations(
    path: Path | str = FOLLOWUP_SPEC_PATH, *, root: Path | str = "."
) -> dict[str, Any]:
    """Read the r2 declarations and refuse to start a disabled design.

    The next structural hypothesis (per-depth feature selection) is declared
    ``enabled: false``: this loader rejects a spec that tries to turn it on, so
    path B cannot be started implicitly by editing the file.
    """
    import yaml

    base = Path(root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = base / target
    if not target.is_file():
        raise FileNotFoundError(f"V4.2-r2 FOLLOWUP_SPEC not found: {target}")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("FOLLOWUP_SPEC must be a YAML mapping")
    schema = str(raw.get("schema_version"))
    if schema not in ("v4.2-r2", "v4.4-r1"):
        raise ValueError(
            f"Unsupported FOLLOWUP_SPEC schema_version: {schema!r}; "
            "accepted: ['v4.2-r2', 'v4.4-r1']"
        )
    if schema == "v4.2-r2":
        followup = raw.get("followup")
        if not isinstance(followup, dict):
            raise ValueError("FOLLOWUP_SPEC is missing the followup section")
        if str(followup.get("repair_version")) != FOLLOWUP_REPAIR_VERSION:
            raise ValueError(f"Repair version must be {FOLLOWUP_REPAIR_VERSION}")
        if str(followup.get("path")) != "fixed_quarter":
            raise ValueError("The repair keeps the pre-declared fixed_quarter path")
        if [int(v) for v in followup.get("coarse_folds", [])] != list(COARSE_FOLDS):
            raise ValueError("The repair coarse folds must be 0/1")
        if [int(v) for v in followup.get("full_folds", [])] != list(FULL_FOLDS):
            raise ValueError("Full coverage must be folds 0-4")
        path_b = raw.get("path_b")
        if not isinstance(path_b, dict):
            raise ValueError("FOLLOWUP_SPEC is missing the path_b design section")
        if bool(path_b.get("enabled", False)):
            raise ValueError(
                "path_b is declared design-only (enabled: false) in this plan and must "
                "not be started automatically"
            )
        return raw

    # v4.4-r1: the per-depth and fusion mechanisms are the round, so the same
    # design must be declared *enabled*; the stage gates are re-checked against
    # the frozen values rather than trusted from the file.
    if str(raw.get("seed_init_protocol")) != FOLLOWUP_REPAIR_VERSION:
        raise ValueError(f"V4.4 must run the {FOLLOWUP_REPAIR_VERSION} seed protocol")
    mechanisms = raw.get("mechanisms")
    if not isinstance(mechanisms, dict) or set(mechanisms) != {"N", "R"}:
        raise ValueError("V4.4 FOLLOWUP_SPEC must declare exactly the N and R mechanisms")
    for name, section in mechanisms.items():
        if not bool(section.get("enabled", False)):
            raise ValueError(f"V4.4 disables the {name} mechanism it is meant to test")
    coarse = raw["stage_machine"]["coarse"]
    if [int(v) for v in coarse["folds"]] != list(COARSE_FOLDS):
        raise ValueError("The coarse stage must stay on folds 0/1")
    if [int(v) for v in raw["stage_machine"]["full_coverage"]["folds"]] != list(FULL_FOLDS):
        raise ValueError("Full coverage must be folds 0-4")
    if float(coarse["gate"]["mean_full_package_gain_min"]) != 0.005:
        raise ValueError("The coarse gate threshold is 0.005")
    if float(raw["stage_machine"]["full_coverage"]["gate"]["mean_gain_min"]) != 0.02:
        raise ValueError("The full-coverage fusion gate threshold is 0.02")
    return raw


def select_units(
    spec: SearchSpec,
    *,
    lines: Sequence[str] | None = None,
    recipes: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Exact unit selection, so a follow-up run never leans on ``--limit``.

    An unknown line, recipe or target is an error rather than an empty result:
    silently selecting nothing would look like a completed run.
    """
    requested_lines = None if lines is None else {str(v).upper() for v in lines}
    requested_recipes = None if recipes is None else {str(v).upper() for v in recipes}
    requested_targets = None if targets is None else {str(v) for v in targets}
    known_lines = {str(name) for name in spec.lines}
    known_recipes = {str(recipe) for name in spec.lines for recipe in spec.recipes(name)}
    known_targets = {str(value) for value in TARGETS}
    for name, requested, known in (
        ("line", requested_lines, known_lines),
        ("recipe", requested_recipes, known_recipes),
        ("target", requested_targets, known_targets),
    ):
        unknown = sorted((requested or set()) - known)
        if unknown:
            raise ValueError(f"Unknown V4.2 {name}(s): {unknown}")
    return [
        (line, recipe, target)
        for line, recipe, target in spec.units
        if (requested_lines is None or line in requested_lines)
        and (requested_recipes is None or recipe in requested_recipes)
        and (requested_targets is None or target in requested_targets)
    ]


def _ordered_id_hash(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def fold_vector_digest(fold_vector: Sequence[int]) -> str:
    array = np.ascontiguousarray(np.asarray(fold_vector, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def candidate_identity(
    train: pd.DataFrame,
    *,
    spec: SearchSpec,
    line: str,
    recipe: str,
    target: str,
    seeds: Sequence[int],
    folds: Sequence[int],
    fold_vectors: Mapping[int, np.ndarray],
    training_seed: int,
    inner_seed: int,
    torch_threads: int | None,
) -> dict[str, Any]:
    """The complete identity of the repaired candidate.

    Two runs share a cache entry only when this payload hashes identically; any
    change to the data, the split, the recipe, the encoders, the seeds, the model
    source or the dependency set produces a different digest.
    """
    from .v4_2_n_node import NodeSpec

    spec_model = NodeSpec(str(recipe)).as_dict()
    return {
        "kind": "V42_FOLLOWUP_CANDIDATE_IDENTITY_V2",
        "repair_version": str(FOLLOWUP_REPAIR_VERSION),
        "candidate_trial_id": f"{line}-{recipe}-{target}{trial_id_suffix(spec)}",
        "line": str(line),
        "recipe": str(recipe),
        "target": str(target),
        "data": {
            "snapshot": str(spec.raw["data_contract"]["snapshot"]),
            "n_rows": int(len(train)),
            "train_row_ids_sha256": _ordered_id_hash(train["sample_id"].astype(str).tolist()),
            "train_row_order_sha256": _ordered_id_hash(
                [f"{i}:{sid}" for i, sid in enumerate(train["sample_id"].astype(str).tolist())]
            ),
        },
        "groups": {
            "group_key_sha256": _ordered_id_hash(
                exclusion_group_keys(train).tolist()
            ),
        },
        "fold_vectors": {str(int(seed)): fold_vector_digest(fold_vectors[int(seed)]) for seed in seeds},
        "folds": [int(v) for v in folds],
        "split_seeds": [int(v) for v in seeds],
        "seeds": {
            "training_seed": int(training_seed),
            "initialisation_seed": int(training_seed),
            "init_seed_shared_across_stages": True,
            "inner_split_seed": int(inner_seed),
            "batch_order_seed_offset": 1_000_003,
            "seed_derivation": "arithmetic_offset_never_process_local",
        },
        "model_recipe": spec_model,
        "numeric_encoding": str(spec_model["numeric_encoding"]),
        "target_transform": "train_subset_mean_abs",
        "torch_threads": None if torch_threads is None else int(torch_threads),
        "source_digest": source_digest_payload(),
        "dependency_identity": dependency_identity(),
    }


def candidate_identity_digest(identity: Mapping[str, Any]) -> str:
    return _sha256_json(identity)


def b_fit_source_files() -> tuple[Path, ...]:
    """The frozen V3.4/V3.6 reference implementation set.

    Deliberately conservative: if any file of the frozen reference chain changes,
    a cached ``B_fit`` vector must not be trusted.  This is independent of the
    N-line source digest, so repairing N never invalidates ``B_fit``.
    """
    here = Path(__file__).resolve().parent
    files = sorted(here.glob("v3_4_*.py")) + sorted(here.glob("v3_6_*.py"))
    files.append(here / "v4_1_reference.py")
    return tuple(files)


def b_fit_identity(
    train_subset: pd.DataFrame,
    *,
    seed: int,
    fold: int,
    source_hash: str,
    fold_vector_sha256: str,
    dependency: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Identity of one ``B_fit`` outer fit, keyed by its *own* dependencies."""
    payload = dict(
        baseline_cache_identity(train_subset, seed=int(seed), fold=int(fold), source_hash=str(source_hash))
    )
    payload.update({
        "kind": "V42_FOLLOWUP_B_FIT_IDENTITY_V2",
        "fold_vector_sha256": str(fold_vector_sha256),
        "dependency_identity_sha256": _sha256_json(
            dependency if dependency is not None else dependency_identity()
        ),
        "reference_source_digest": source_digest(b_fit_source_files()),
    })
    return payload


def _legacy_b_fit_verified(
    source_cache: Mapping[str, Any],
    source_identity: Mapping[str, Any] | None,
    *,
    key: str,
    train_subset: pd.DataFrame,
    seed: int,
    fold: int,
    source_hash: str,
) -> bool:
    """Can the pre-repair B_fit cache entry be trusted and reused?

    Every field of the recorded legacy identity must reproduce exactly and the
    recorded source hash must still match.  ``B_fit`` does not depend on the N
    seeds, so a matching identity is sufficient to reuse the vector; anything
    else is refitted instead of guessed.
    """
    if f"{key}-pred" not in source_cache or f"{key}-id" not in source_cache:
        return False
    stored = str(source_cache[f"{key}-id"])
    try:
        recorded = json.loads(stored)
    except json.JSONDecodeError:
        return False
    expected = baseline_cache_identity(
        train_subset, seed=int(seed), fold=int(fold), source_hash=str(source_hash)
    )
    if not isinstance(recorded, dict) or recorded != expected:
        return False
    if source_identity is not None:
        recorded_source = source_identity.get("source_hash")
        if recorded_source is not None and str(recorded_source) != str(source_hash):
            return False
    return True


# ---------------------------------------------------------------------------
# stage machine
# ---------------------------------------------------------------------------

def _coarse_stage(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"status": "NOT_EVALUATED", "reason": "coarse row not available"}
    gate = dict(row.get("gate", {}))
    per_path = gate.get("per_path", {})
    path = gate.get("path")
    folds = sorted({
        int(fold)
        for seed_entry in row.get("per_seed", {}).values()
        for fold in seed_entry.get("fold_level", {})
    })
    return {
        "status": "PASSED" if gate.get("passes") else "FAILED",
        "covered_folds": folds,
        "path": path,
        "per_path_mean_gain": {
            name: float(values.get("mean_gain"))
            for name, values in per_path.items()
            if values.get("mean_gain") is not None
        },
        "per_path_both_seeds_positive": {
            name: bool(values.get("both_seeds_positive"))
            for name, values in per_path.items()
        },
    }


def _full_stage(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "status": "NOT_EVALUATED",
            "reason": "full-coverage aggregation has not been run",
        }
    coverage = dict(row.get("coverage_gate", {}))
    status = str(coverage.get("status", "NOT_EVALUATED"))
    return {
        "status": status,
        "evaluated": bool(coverage.get("evaluated", False)),
        "covered_folds": coverage.get("covered_folds"),
        "expected_folds": coverage.get("expected_folds"),
        "mean_gain_min": coverage.get("mean_gain_min"),
        "positive_folds_min": coverage.get("positive_folds_min"),
        "path": coverage.get("path"),
        "positive_folds": dict(row.get("positive_folds", {})),
        "per_path_mean_gain": {
            name: values.get("mean_gain")
            for name, values in coverage.get("per_path", {}).items()
        },
    }


def evaluate_stages(
    *,
    coarse_row: Mapping[str, Any] | None,
    full_row: Mapping[str, Any] | None,
    repair_version: str = FOLLOWUP_REPAIR_VERSION,
) -> dict[str, Any]:
    """Turn the measured rows into an explicit, non-overstated stage table.

    Unrun stages are ``NOT_EVALUATED``.  An incomplete full coverage is
    ``NOT_EVALUATED_INCOMPLETE_COVERAGE``; it is never a failure and never a pass.
    """
    coarse = _coarse_stage(coarse_row)
    full = _full_stage(full_row)

    if coarse["status"] == "NOT_EVALUATED":
        next_action = "STOP_NO_COARSE_EVIDENCE"
        fusion = {"status": "NOT_EVALUATED", "reason": "coarse stage has no result"}
    elif coarse["status"] == "FAILED":
        next_action = "STOP_COARSE_GATE_FAILED"
        fusion = {
            "status": "NOT_EVALUATED",
            "reason": "the repaired coarse gate did not pass; no alpha scan is permitted",
        }
    elif full["status"] == "NOT_EVALUATED":
        next_action = "RUN_FULL_COVERAGE"
        fusion = {"status": "NOT_EVALUATED", "reason": "full coverage not yet run"}
    elif full["status"] == "NOT_EVALUATED_INCOMPLETE_COVERAGE":
        next_action = "RUN_FULL_COVERAGE"
        fusion = {
            "status": "NOT_EVALUATED_INCOMPLETE_COVERAGE",
            "reason": "the full-coverage gate cannot be judged on partial coverage",
        }
    elif full["status"] == "FAILED":
        next_action = "STOP_FULL_GATE_FAILED"
        fusion = {
            "status": "NOT_EVALUATED",
            "reason": "the full-coverage gate failed; the line stops with no alpha rescue",
        }
    else:
        next_action = "RUN_FUSION"
        fusion = {
            "status": "NOT_EVALUATED",
            "reason": "gate passed; inner-OOF alpha selection has not been executed",
        }

    return {
        "repair_version": str(repair_version),
        "stages": {
            "coarse": coarse,
            "full_coverage": full,
            "fusion": fusion,
            "replication": {
                "status": "NOT_EVALUATED",
                "reason": "replication requires a completed fusion selection",
            },
            "submission": {
                "status": "NOT_EVALUATED",
                "reason": "no package is generated or uploaded by this runner",
            },
        },
        "next_action": next_action,
    }


def fusion_protocol_plan(spec: SearchSpec | None = None, *, inner_folds: int = 3) -> dict[str, Any]:
    """The frozen A3 protocol, recorded but not executed by this runner."""
    return {
        "step": "A3_fusion_selection",
        "enabled": False,
        "precondition": "full_coverage gate PASSED (>=0.02 mean, both seeds positive, >=8/10 folds)",
        "inner_folds": int(inner_folds),
        "group_safe": True,
        "alpha_grid": list(FUSION_ALPHA_GRID),
        "tie_break": "smaller_alpha",
        "alpha_zero_fallback_allowed": True,
        "both_endpoints_refit_in_every_inner_U_H": True,
        "selection_data": "aligned inner OOF only",
        "forbidden": [
            "selecting alpha from outer validation residuals",
            "slicing the full-data outer OOF as an inner OOF",
        ],
        "evaluate_on_V_after_fixing_alpha": True,
        "note": "Only reachable when evaluate_stages reports next_action == RUN_FUSION.",
    }


def replication_protocol_plan(*, training_seed: int = 3407) -> dict[str, Any]:
    """The frozen A4 replication protocol, recorded but not executed here."""
    return {
        "step": "A4_training_seed_replication",
        "enabled": False,
        "precondition": "A3 fusion selection completed",
        "rerun_training_seed": int(training_seed),
        "alpha_reused_from_first_training_seed": True,
        "structure_reselection_on_second_seed": False,
        "gates": {"mean_gain_min": 0.02, "both_split_seeds_positive": True, "positive_folds_min": 8},
        "final_training_seed": 42,
        "max_preferred_packages": 1,
        "auto_upload": False,
    }


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def _require_private(root: Path, output: Path) -> Path:
    resolved = output.resolve()
    if not resolved.is_relative_to((root / "local/runs").resolve()):
        raise ValueError("V4.2 follow-up outputs must remain beneath local/runs")
    return resolved


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_ledger(path: Path, event: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _ledger_key(event: Mapping[str, Any], identity_digest: str) -> str:
    base = "|".join(
        str(event[k]) for k in ("line", "recipe", "target", "seed", "fold")
    )
    # The repair identity is part of the cache key, so pre-repair entries and a
    # changed source/seed/data digest can never collide.
    return f"{FOLLOWUP_REPAIR_VERSION}|{identity_digest}|{base}"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _ensure_baselines(
    root: Path,
    output: Path,
    *,
    train: pd.DataFrame,
    fold_vectors: Mapping[int, np.ndarray],
    seeds: Sequence[int],
    folds: Sequence[int],
    references: V42References,
    source_outputs: Sequence[Path],
    force_refit: bool,
) -> dict[str, Any]:
    """Reuse verified ``B_fit`` vectors, refit anything that cannot be verified.

    ``B_replay`` is never an acceptable substitute here, and reusing a baseline
    does not depend on the N candidate at all, so this stays independent of the
    repair version.  Any number of previously verified source caches may be
    offered (the pre-repair coarse and full runs computed the same frozen
    ``B_fit`` on the same ``T``/``V`` cells); every candidate is still checked
    field-by-field before its vector is accepted.
    """
    baseline_path = output / "baselines.npz"
    cache: dict[str, np.ndarray] = {}
    if baseline_path.is_file():
        cache = {k: v for k, v in np.load(baseline_path, allow_pickle=False).items()}

    sources: list[tuple[Path, dict[str, Any], dict[str, Any] | None]] = []
    for candidate in source_outputs:
        baseline_file = candidate / "baselines.npz"
        if not baseline_file.is_file():
            continue
        source_cache = {
            k: v for k, v in np.load(baseline_file, allow_pickle=False).items()
        }
        identity_path = candidate / "baseline_identity.json"
        identity_payload = (
            json.loads(identity_path.read_text(encoding="utf-8"))
            if identity_path.is_file() else None
        )
        sources.append((candidate, source_cache, identity_payload))

    dependency = dependency_identity()
    meta: dict[str, Any] = {
        "computed": 0,
        "reused_local": 0,
        "reused_from_source": 0,
        "identity": {},
        "source_outputs": [str(path) for path in source_outputs],
        "sources_considered": [str(path) for path, _, _ in sources],
        "force_refit": bool(force_refit),
        "b_replay_substitution": False,
    }
    for seed in seeds:
        folds_vector = np.asarray(fold_vectors[int(seed)], dtype=int)
        for fold in folds:
            train_mask = folds_vector != int(fold)
            valid_mask = folds_vector == int(fold)
            subset = train.loc[train_mask].reset_index(drop=True)
            key = f"s{seed}-f{fold}"
            identity = b_fit_identity(
                subset, seed=int(seed), fold=int(fold),
                source_hash=references.source_hash,
                fold_vector_sha256=fold_vector_digest(folds_vector),
                dependency=dependency,
            )
            meta["identity"][key] = identity
            vector: np.ndarray | None = None
            if not force_refit and f"{key}-pred" in cache and f"{key}-id" in cache:
                try:
                    stored = json.loads(str(cache[f"{key}-id"]))
                except json.JSONDecodeError:
                    stored = None
                if stored == identity:
                    vector = np.asarray(cache[f"{key}-pred"], dtype=float)
                    meta["reused_local"] += 1
            if vector is None and not force_refit:
                for candidate, source_cache, source_identity in sources:
                    if _legacy_b_fit_verified(
                        source_cache, source_identity, key=key, train_subset=subset,
                        seed=int(seed), fold=int(fold), source_hash=references.source_hash,
                    ):
                        vector = np.asarray(source_cache[f"{key}-pred"], dtype=float)
                        meta["reused_from_source"] += 1
                        meta.setdefault("reused_from", {})[key] = str(candidate)
                        break
            if vector is None:
                bundle = references.fit_baseline(
                    subset, train.loc[valid_mask].reset_index(drop=True)
                )
                vector = np.full((len(train), len(TARGETS)), np.nan, dtype=float)
                for column, target in enumerate(TARGETS):
                    vector[valid_mask, column] = np.asarray(bundle["b36"][target], dtype=float)
                # A refit has no reusable file provenance: the recomputed
                # identity is the only accepted key.
                meta["computed"] += 1
            cache[f"{key}-pred"] = vector
            cache[f"{key}-id"] = np.asarray(json.dumps(identity, sort_keys=True))
            np.savez_compressed(baseline_path, **cache)
    meta["file"] = str(baseline_path.relative_to(root))
    meta["source_hash"] = references.source_hash
    meta["dependency_identity_sha256"] = _sha256_json(dependency)
    _write_json(output / "baseline_identity.json", meta)
    return meta


def _schedule_and_fit(
    *,
    root: Path,
    output: Path,
    train: pd.DataFrame,
    fold_vectors: Mapping[int, np.ndarray],
    worker_spec: Mapping[str, Any],
    identity_digest: str,
    line: str,
    recipe: str,
    target: str,
    seeds: Sequence[int],
    folds: Sequence[int],
    jobs: int,
    torch_threads: int | None,
    start_method: str,
) -> dict[str, Any]:
    ledger = output / "fit_ledger.jsonl"
    known = {
        _ledger_key(event, identity_digest)
        for event in _read_ledger(ledger)
        if event.get("event") == "complete" and event.get("identity_digest") == identity_digest
    }
    tasks = [
        {
            "line": str(line), "recipe": str(recipe), "target": str(target),
            "seed": int(seed), "fold": int(fold),
        }
        for seed in seeds
        for fold in folds
    ]
    scheduled = [task for task in tasks if _ledger_key(task, identity_digest) not in known]
    results: list[dict[str, Any]] = []
    source_digest = source_digest_payload()
    dependency = dependency_identity()

    def persist(outcome: Mapping[str, Any]) -> dict[str, Any]:
        record = {key: value for key, value in outcome.items() if key != "prediction"}
        record["event"] = "complete" if outcome["status"] == "available" else "failed"
        record["repair_version"] = FOLLOWUP_REPAIR_VERSION
        record["identity_digest"] = str(identity_digest)
        record["source_digest"] = source_digest
        record["dependency_identity"] = dependency
        if outcome["status"] == "available":
            path = output / "predictions" / (
                f"{outcome['line']}-{outcome['recipe']}-{outcome['target']}"
                f"-s{outcome['seed']}-f{outcome['fold']}.npy"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.asarray(outcome["prediction"], dtype=float))
            record["prediction_file"] = str(path.relative_to(output))
            record["prediction_sha256"] = _file_sha256(path)
            finite = np.isfinite(np.asarray(outcome["prediction"], dtype=float))
            record["prediction_finite_rows"] = int(finite.sum())
        _append_ledger(ledger, record)
        results.append(dict(outcome))
        print(json.dumps({
            "line": outcome["line"], "recipe": outcome["recipe"], "target": outcome["target"],
            "seed": outcome.get("seed"), "fold": outcome.get("fold"),
            "status": outcome["status"],
            "seconds": round(float(outcome.get("seconds", 0.0)), 1),
            "best_epoch": (record.get("fit_meta") or {}).get("best_epoch"),
            "init_hash": ((record.get("fit_meta") or {}).get("randomness") or {}).get(
                "initialisation_hash_stage1", "")[:12],
            "prediction_sha256": record.get("prediction_sha256", "")[:12],
        }, ensure_ascii=False), flush=True)
        return record

    if jobs > 1 and len(scheduled) > 1:
        context = multiprocessing.get_context(str(start_method))
        with ProcessPoolExecutor(
            max_workers=int(jobs),
            mp_context=context,
            initializer=screen._initialise_worker,
            initargs=(train, fold_vectors, worker_spec, str(root), torch_threads),
        ) as executor:
            futures = {executor.submit(screen._fit_one, task): task for task in scheduled}
            for future in as_completed(futures):
                persist(future.result())
    else:
        screen._initialise_worker(train, fold_vectors, worker_spec, str(root), torch_threads)
        for task in scheduled:
            persist(screen._fit_one(task))

    return {
        "slots": len(tasks),
        "scheduled_this_run": len(scheduled),
        "already_complete": len(tasks) - len(scheduled),
        "completed": int(sum(1 for r in results if r["status"] == "available")),
        "failed": int(sum(1 for r in results if r["status"] != "available")),
    }


def _load_unit_row(output: Path, trial_id: str, *, filename: str = "coarse_summary.json") -> dict[str, Any] | None:
    path = output / filename
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("trial_id") == trial_id:
            return row
    return None


def _aggregate(
    *,
    root: Path,
    output: Path,
    spec: SearchSpec,
    train: pd.DataFrame,
    fold_vectors: Mapping[int, np.ndarray],
    line: str,
    recipe: str,
    target: str,
    seeds: Sequence[int],
    folds: Sequence[int],
    tag: str,
) -> dict[str, Any]:
    """Aggregate one stage and keep a stage-named copy of its evidence.

    ``aggregate_screen`` always writes ``coarse_summary.*``; A2 must not
    overwrite A1's coarse-only evidence, so the stage copy is preserved.
    """
    result = screen.aggregate_screen(
        root=root, output=output, spec=spec, train=train, fold_vectors=fold_vectors,
        units=[(str(line), str(recipe), str(target))], seeds=seeds, folds=folds,
        trial_id_suffix=trial_id_suffix(spec),
    )
    for suffix in ("summary.csv", "summary.json", "selection.json"):
        source = output / f"coarse_{suffix}"
        if source.is_file():
            (output / f"{tag}_stage_{suffix}").write_bytes(source.read_bytes())
    return result


def run_followup(
    root: Path | str = ".",
    output: Path | str | None = None,
    *,
    spec_path: Path | str = DEFAULT_SPEC_PATH,
    line: str = "N",
    recipe: str = "N2",
    target: str = "tap_iron",
    seeds: Sequence[int] = (42, 3407),
    training_seed: int = 42,
    inner_seed: int = 20260925,
    jobs: int = 1,
    torch_threads: int | None = None,
    b_fit_workers: int = 12,
    start_method: str = "spawn",
    source_outputs: Sequence[Path | str] | None = (DEFAULT_SOURCE_OUTPUT, DEFAULT_FULL_SOURCE_OUTPUT),
    force_refit_baselines: bool = False,
    skip_baseline: bool = False,
    aggregate_only: bool = False,
) -> dict[str, Any]:
    """Run A1 and, only if A1 passes, A2 for the repaired N2 candidate."""
    root = Path(root).resolve()
    spec = load_search_spec(spec_path, root=root)
    output = (
        Path(output).resolve() if output is not None
        else root / DEFAULT_FOLLOWUP_OUTPUT
    )
    output = _require_private(root, output)
    output.mkdir(parents=True, exist_ok=True)
    sources: list[Path] = []
    for candidate in (source_outputs or ()):
        candidate_source = Path(candidate)
        if not candidate_source.is_absolute():
            candidate_source = root / candidate_source
        if candidate_source.is_dir():
            sources.append(candidate_source)

    if str(line) not in spec.lines:
        raise ValueError(f"Unknown V4.2 line: {line!r}")
    if str(recipe) not in spec.recipes(str(line)):
        raise ValueError(f"Recipe {recipe!r} does not belong to line {line!r}")
    if str(target) not in TARGETS:
        raise ValueError(f"Unknown V4.2 target: {target!r}")
    selected = select_units(spec, lines=[line], recipes=[recipe], targets=[target])
    if (str(line), str(recipe), str(target)) not in selected:
        raise ValueError(
            f"Unit {(str(line), str(recipe), str(target))!r} is not a pre-registered V4.2 unit"
        )
    # The shared screen worker builds every model with one fixed inner split
    # seed; the identity records it, so a mismatch must fail loudly rather than
    # silently run a different split from the one that was hashed.
    if int(inner_seed) != SCREEN_INNER_SEED:
        raise ValueError(
            f"inner_seed must stay at the screen value {SCREEN_INNER_SEED}"
        )

    train = load_training_frame(root)
    fold_vectors = {int(seed): load_fold_vector(root, train, int(seed)) for seed in seeds}
    worker_spec = deepcopy(spec.raw)
    worker_spec["common_training"]["neural_starting_point"]["initialisation_seed"] = int(training_seed)

    identity = candidate_identity(
        train, spec=spec, line=str(line), recipe=str(recipe), target=str(target),
        seeds=seeds, folds=FULL_FOLDS, fold_vectors=fold_vectors,
        training_seed=int(training_seed), inner_seed=int(inner_seed),
        torch_threads=torch_threads,
    )
    identity_digest = candidate_identity_digest(identity)
    trial_id = str(identity["candidate_trial_id"])
    _write_json(output / "candidate_identity.json", identity)

    started = time.perf_counter()
    baseline_meta: dict[str, Any] = {"skipped": True}
    baseline_meta_full: dict[str, Any] = {"skipped": True}
    references: V42References | None = None
    if not aggregate_only and not skip_baseline:
        references = V42References(
            root, train, workers=int(b_fit_workers), verify_replay=False
        )
        # Only the baselines the current stage needs are materialised: A1 asks
        # for folds 0/1, and folds 2/3/4 are added only if A1 passes.
        baseline_meta = _ensure_baselines(
            root, output, train=train, fold_vectors=fold_vectors, seeds=seeds,
            folds=COARSE_FOLDS, references=references, source_outputs=sources,
            force_refit=bool(force_refit_baselines),
        )

    _write_json(output / "spec_snapshot.json", worker_spec)

    # ---- A1: repaired coarse units --------------------------------------
    coarse_run = {"slots": 0, "scheduled_this_run": 0, "already_complete": 0,
                  "completed": 0, "failed": 0}
    if not aggregate_only:
        coarse_run = _schedule_and_fit(
            root=root, output=output, train=train, fold_vectors=fold_vectors,
            worker_spec=worker_spec, identity_digest=identity_digest, line=str(line),
            recipe=str(recipe), target=str(target), seeds=seeds, folds=COARSE_FOLDS,
            jobs=int(jobs), torch_threads=torch_threads, start_method=str(start_method),
        )
    _aggregate(
        root=root, output=output, spec=spec, train=train, fold_vectors=fold_vectors,
        line=str(line), recipe=str(recipe), target=str(target), seeds=seeds,
        folds=COARSE_FOLDS, tag="coarse",
    )
    coarse_row = _load_unit_row(output, trial_id, filename="coarse_stage_summary.json")

    stages = evaluate_stages(coarse_row=coarse_row, full_row=None)
    full_run = {"slots": 0, "scheduled_this_run": 0, "already_complete": 0,
                "completed": 0, "failed": 0}
    if stages["stages"]["coarse"]["status"] == "PASSED":
        # ---- A2: fill the six missing folds ------------------------------
        if not aggregate_only and not skip_baseline and references is not None:
            baseline_meta_full = _ensure_baselines(
                root, output, train=train, fold_vectors=fold_vectors, seeds=seeds,
                folds=FULL_FOLDS, references=references, source_outputs=sources,
                force_refit=bool(force_refit_baselines),
            )
        if not aggregate_only:
            full_run = _schedule_and_fit(
                root=root, output=output, train=train, fold_vectors=fold_vectors,
                worker_spec=worker_spec, identity_digest=identity_digest, line=str(line),
                recipe=str(recipe), target=str(target), seeds=seeds, folds=FULL_FOLDS,
                jobs=int(jobs), torch_threads=torch_threads, start_method=str(start_method),
            )
        _aggregate(
            root=root, output=output, spec=spec, train=train, fold_vectors=fold_vectors,
            line=str(line), recipe=str(recipe), target=str(target), seeds=seeds,
            folds=FULL_FOLDS, tag="full",
        )
        full_row = _load_unit_row(output, trial_id, filename="full_stage_summary.json")
        stages = evaluate_stages(coarse_row=coarse_row, full_row=full_row)

    manifest = {
        "status": "V42_FOLLOWUP_PRIVATE_EVIDENCE_NOT_SUBMISSION",
        "repair_version": FOLLOWUP_REPAIR_VERSION,
        "candidate_trial_id": trial_id,
        "identity_digest": identity_digest,
        "line": str(line),
        "recipe": str(recipe),
        "target": str(target),
        "numeric_encoding": str(identity["numeric_encoding"]),
        "split_seeds": [int(v) for v in seeds],
        "training_seed": int(training_seed),
        "inner_split_seed": int(inner_seed),
        "torch_threads": None if torch_threads is None else int(torch_threads),
        "source_outputs": [str(path) for path in sources],
        "historical_runs_preserved": [
            "coarse-r1", "full-r1", "repl-t3407-r1",
        ],
        "baseline_coarse": baseline_meta,
        "baseline_full": baseline_meta_full,
        "coarse_run": coarse_run,
        "full_run": full_run,
        "stages": stages["stages"],
        "next_action": stages["next_action"],
        "fusion_protocol": fusion_protocol_plan(spec),
        "replication_protocol": replication_protocol_plan(),
        "platform_uploads": 0,
        "submission_packages": 0,
        "promotion": "none",
        "seconds": round(float(time.perf_counter() - started), 2),
    }
    _write_json(output / "followup_manifest.json", manifest)
    return {"output": str(output), **manifest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--spec", type=Path, default=Path(DEFAULT_SPEC_PATH))
    parser.add_argument("--line", default="N")
    parser.add_argument("--recipe", default="N2")
    parser.add_argument("--target", default="tap_iron", choices=list(TARGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 3407])
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--b-fit-workers", type=int, default=12)
    parser.add_argument("--start-method", choices=["spawn", "fork", "forkserver"], default="spawn")
    parser.add_argument("--source-outputs", type=Path, nargs="+",
                        default=[Path(DEFAULT_SOURCE_OUTPUT), Path(DEFAULT_FULL_SOURCE_OUTPUT)],
                        help="pre-repair runs whose verified B_fit vectors may be reused")
    parser.add_argument("--force-refit-baselines", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    result = run_followup(
        args.root, args.output, spec_path=args.spec, line=args.line, recipe=args.recipe,
        target=args.target, seeds=args.seeds, training_seed=args.training_seed,
        jobs=args.jobs, torch_threads=args.torch_threads, b_fit_workers=args.b_fit_workers,
        start_method=args.start_method, source_outputs=args.source_outputs,
        force_refit_baselines=args.force_refit_baselines, skip_baseline=args.skip_baseline,
        aggregate_only=args.aggregate_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
