"""Paired incumbent diagnostics on immutable saved OOF predictions; no fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def verify_files(root, files):
    root = Path(root).resolve()
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError(f"Frozen input identity changed: {name}")


def vectors(y, reference, candidate):
    result = tuple(np.asarray(x, dtype=float) for x in (y, reference, candidate))
    if (result[0].ndim != 1 or not len(result[0])
            or any(x.shape != result[0].shape or not np.isfinite(x).all() for x in result)
            or (result[0] < 0).any() or result[0].sum() <= 0):
        raise ValueError("Finite, aligned vectors and positive WMAPE denominator required")
    return result


def gain(y, reference, candidate):
    y, reference, candidate = vectors(y, reference, candidate)
    return float(50 * (np.abs(y-reference).sum()-np.abs(y-candidate).sum()) / y.sum())


def reduction_detail(y, reference, candidate):
    y, reference, candidate = vectors(y, reference, candidate)
    residual = y-reference
    movement = candidate-reference
    contribution = np.abs(residual)-np.abs(y-candidate)
    positive = np.maximum(contribution, 0)
    count = max(1, int(np.ceil(.01*len(y))))
    top = np.argsort(positive, kind="stable")[-count:]
    keep = np.ones(len(y), bool)
    keep[top] = False
    crossing = residual * (y-candidate) < 0
    return dict(rows=len(y), score_gain=gain(y, reference, candidate),
                mae_reduction=float(contribution.mean()),
                row_improvement_fraction=float(np.mean(contribution > 0)),
                movement_toward_truth_fraction=float(np.mean(residual*movement > 0)),
                crosses_truth_fraction=float(crossing.mean()),
                crosses_and_worsens_fraction=float(np.mean(crossing & (contribution < 0))),
                top_one_percent_positive_reduction_share=float(positive[top].sum()/positive.sum())
                    if positive.sum() else None,
                without_top_one_percent_positive_rows_gain=gain(y[keep], reference[keep], candidate[keep])
                    if keep.any() and y[keep].sum() > 0 else None,
                positive_error_reduction=float(positive.sum()),
                negative_error_reduction=float(np.minimum(contribution, 0).sum()))


def sample_indices(n, size, replicates, seed, groups=None, quotas=None):
    if not 0 < size <= n or replicates <= 0:
        raise ValueError("Invalid sample size/replicates")
    rng = np.random.default_rng(seed)
    if groups is None:
        if quotas is not None:
            raise ValueError("Quotas require groups")
        return np.stack([rng.choice(n, size, replace=False) for _ in range(replicates)])
    groups = np.asarray(groups)
    if groups.shape != (n,) or quotas is None or sum(quotas.values()) != size:
        raise ValueError("Aligned groups and exact test-size quotas required")
    pools = {int(k): np.flatnonzero(groups == int(k)) for k in quotas}
    if any(q < 0 or q > len(pools[k]) for k, q in quotas.items()):
        raise ValueError("Stratified sample has insufficient group coverage")
    return np.stack([np.concatenate([rng.choice(pools[k], q, replace=False)
                                    for k, q in sorted(quotas.items())])
                     for _ in range(replicates)])


def simulated_gains(y, reference, candidate, draws):
    y, reference, candidate = vectors(y, reference, candidate)
    draws = np.asarray(draws)
    if (draws.ndim != 2 or draws.dtype.kind not in "iu" or draws.min() < 0
            or draws.max() >= len(y) or (np.diff(np.sort(draws, axis=1), axis=1) == 0).any()):
        raise ValueError("Each simulated test must contain distinct valid rows")
    denominator = y[draws].sum(axis=1)
    if (denominator <= 0).any():
        raise ValueError("Simulated denominator must be positive")
    values = 50 * (np.abs(y-reference)-np.abs(y-candidate))[draws].sum(axis=1)/denominator
    return values, denominator


def distribution(values):
    return dict(mean=float(np.mean(values)), negative_fraction=float(np.mean(values < 0)),
                zero_fraction=float(np.mean(values == 0)),
                quantiles={str(q): float(np.quantile(values, q)) for q in (.025, .1, .5, .9, .975)})


def assemble_fold_columns(archive, folds, n_outputs=2):
    """Keep each seed's predictions separate and require exactly one fold per row."""
    result = {}
    for seed, assignment in folds.items():
        assignment = np.asarray(assignment)
        combined = np.full((len(assignment), n_outputs), np.nan)
        covered = np.zeros_like(combined, dtype=int)
        for fold in range(5):
            column = np.asarray(archive[f"s{seed}-f{fold}-pred"], float)
            if column.shape != combined.shape:
                raise ValueError("Fold prediction shape mismatch")
            present = np.isfinite(column)
            expected = np.repeat((assignment == fold)[:, None], n_outputs, axis=1)
            if not np.array_equal(present, expected):
                raise ValueError("Fold/NaN coverage mismatch")
            combined[present] = column[present]
            covered[present] += 1
        if not (covered == 1).all():
            raise ValueError("Expected exactly one prediction per seed/row/target")
        result[seed] = combined
    return result


def fold_median_regions(values, folds):
    values, folds = np.asarray(values, float), np.asarray(folds)
    if values.shape != folds.shape or not np.isfinite(values).all():
        raise ValueError("Finite aligned inputs required for regions")
    high = np.zeros(len(values), bool)
    thresholds = {}
    for fold in range(5):
        held = folds == fold
        if not held.any() or held.all():
            raise ValueError("Five complete outer folds required")
        threshold = float(np.median(values[~held]))
        high[held] = values[held] >= threshold
        thresholds[str(fold)] = threshold
    return high, thresholds


def residual_detail(y, prediction):
    residual = np.asarray(y, float)-np.asarray(prediction, float)
    return dict(rows=len(residual), mean=float(residual.mean()), median=float(np.median(residual)),
                positive_fraction=float(np.mean(residual > 0)), mae=float(np.abs(residual).mean()))


def load_saved(spec):
    from .component_regularization_run import collect
    from .v5_library import fold_vector, load_v5_training_frame, load_column_reference
    from .v5_spec import load_v5_spec
    from .v7_periodic import digest

    root, ptarl = Path(spec["main_root"]), Path(spec["ptarl_root"])
    verify_files(root, spec["inputs"])
    incumbent = json.loads((root/"EVIDENCE_STATUS.json").read_text())["round2_current_platform_best"]
    if any(incumbent[k] != v for k, v in spec["reference"].items()):
        raise ValueError("Current platform reference changed: prospectively refreeze")
    manifest = json.loads((ptarl/"manifest.json").read_text())
    audit = json.loads((ptarl/"development/audit.json").read_text())
    if audit["status"] != "passed" or audit["manifest_sha256"] != sha(ptarl/"manifest.json"):
        raise ValueError("PTaRL audit/manifest mismatch")
    verify_files(ptarl/"development", audit["artifact_hashes"])
    # Audit anchors bind original scientific sources, not this new diagnostic module.
    verify_files(Path(manifest["workspace"]), manifest["source_hashes"])
    frame = load_v5_training_frame(root)
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in spec["seeds"]}
    base, members = collect(root/"local/runs/strong-component-regularization/development-r2",
                            frame, folds, {"tap_time_len": ["EMA", "SAM"]})
    ptarl_columns, ptarl_parents = {}, {}
    native_reference = load_column_reference(root,frame,load_v5_spec(root))
    for seed, fv in folds.items():
        if digest(fv.tolist()) != manifest["reference"]["audit"]["fold_hashes"][str(seed)]:
            raise ValueError("PTaRL/EMA fold identities differ")
        old_v7 = np.empty(len(frame))
        for fold in range(5):
            name=f"local/runs/round2-v7-periodic-networks/development-r1/tap_time_len-tabm_plr001-s{seed}-f{fold}.npy"
            verify_files(root,{name:manifest["reference"]["audit"]["hashes"][name]})
            old_v7[fv==fold]=np.load(root/name,allow_pickle=False)
        name=f"local/runs/round2-v5-error-covariance/time-n-family-r1/seed-{seed}/pred-v36-s1-N-0048.npy"
        verify_files(root,{name:manifest["reference"]["audit"]["hashes"][name]})
        ptarl_parents[seed]=(.2*native_reference.base_for("tap_time_len",seed)
                            +.3*np.load(root/name,allow_pickle=False)+.5*old_v7)
        np.testing.assert_array_equal(old_v7,base[seed]["v7_time"])
        parent = np.ascontiguousarray(ptarl_parents[seed],dtype="<f8")
        parent_sha = hashlib.sha256(json.dumps(list(parent.shape),sort_keys=True,
                                    separators=(",",":"),allow_nan=False).encode()+parent.tobytes()).hexdigest()
        if parent_sha != manifest["reference"]["columns"]["current"][str(seed)]["tap_time_len"]:
            raise ValueError("Original PTaRL parent fails frozen prediction identity")
        vector = np.full(len(frame), np.nan)
        for fold in range(5):
            path = ptarl/f"development/units/tap_time_len-s{seed}-f{fold}"
            complete = json.loads((path/"complete.json").read_text())
            held = fv == fold
            if (complete["task"] != dict(target="tap_time_len", seed=seed, fold=fold)
                    or complete["metadata"]["PTARL_AUX_refit"]["fit_ids_digest"]
                    != digest(frame.loc[~held, "sample_id"].tolist())
                    or complete["prediction_sha256"] != sha(path/"predictions.npz")):
                raise ValueError("PTaRL task/training/prediction identity mismatch")
            with np.load(path/"predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["query_ids"], frame.loc[held,"sample_id"].to_numpy(str))
                vector[held] = saved["PTARL_AUX"]
        if not np.isfinite(vector).all():
            raise ValueError("Incomplete PTaRL coverage")
        ptarl_columns[seed] = vector
    return frame, folds, base, members, ptarl_columns, ptarl_parents


def historical_residual_audit(spec, frame, folds):
    from .data import TARGETS
    root = Path(spec["main_root"])
    folder = root/"local/runs/round2-v4.4-mechanism-completion/N2-tap_iron"
    with np.load(folder/"baselines.npz", allow_pickle=False) as saved:
        columns = assemble_fold_columns(saved, folds)
    historical = json.loads((root/"local/runs/round2-v4.5-residual-direction/diagnostic-r1/diagnostic.json").read_text())
    recorded_matches, proper_differences = [], []
    last = spec["seeds"][-1]
    for cell in historical["cells"]:
        target_index = list(TARGETS).index(cell["target"])
        held = folds[cell["seed"]] == cell["fold"]
        actual = frame[cell["target"]].to_numpy(float)
        legacy_mae = float(np.abs(actual[held]-columns[last][held, target_index]).mean())
        proper_mae = float(np.abs(actual[held]-columns[cell["seed"]][held,target_index]).mean())
        recorded_matches.append(abs(legacy_mae-cell["mae_delta0"]))
        proper_differences.append(dict(seed=cell["seed"], fold=cell["fold"], target=cell["target"],
                                      recorded_mae=cell["mae_delta0"], proper_seed_mae=proper_mae))
    if max(recorded_matches) > 1e-10:
        raise ValueError("Historical output is not explained by the audited seed-overwrite path")
    encoded = (frame.spout_no.to_numpy(float) > 0).astype(int)
    rows = []
    for seed in folds:
        for i, target in enumerate(TARGETS):
            y = frame[target].to_numpy(float)
            row = dict(seed=seed, target=target, full=residual_detail(y, columns[seed][:,i]),
                       by_spout={str(s): residual_detail(y[frame.spout_no == s], columns[seed][frame.spout_no == s,i])
                                 for s in sorted(frame.spout_no.unique())}, regions={})
            for feature in spec["conditional_features"]:
                high, thresholds = fold_median_regions(frame[feature], folds[seed])
                row["regions"][feature] = dict(thresholds=thresholds,
                    low=residual_detail(y[~high], columns[seed][~high,i]),
                    high=residual_detail(y[high], columns[seed][high,i]))
            rows.append(row)
    return dict(recorded_zero_correction_mae_reproduced_max_difference=max(recorded_matches),
                recorded_cells=len(recorded_matches), final_seed_overwrite=last,
                true_spout_values=sorted(int(s) for s in frame.spout_no.unique()),
                old_spout_indicator_unique_values=np.unique(encoded).tolist(),
                seed_prediction_max_difference=float(np.max(np.abs(columns[42]-columns[3407]))),
                proper_seed_cells=proper_differences, corrected_zero_fit_residuals=rows,
                original_execution_source_hash_available=False,
                historical_identity_scope="saved inputs and zero-correction output attribution only",
                learned_corrector_external_isolation="not_established_by_mixed_OOF",
                conditional_utility="not_tested_no_corrector_fits", historical_decision_unchanged=True)


def run(spec_path, workspace):
    import pandas as pd
    spec_path, workspace = Path(spec_path).resolve(), Path(workspace).resolve()
    spec = json.loads(spec_path.read_text())
    root = Path(spec["main_root"])
    out = (root/spec["output"]).resolve()
    if not out.is_relative_to(root/"local/runs") or spec["new_fits"] != 0:
        raise ValueError("Private zero-fit output required")
    out.mkdir(parents=True, exist_ok=False)
    module_path = Path(__file__).resolve()
    execution_files = {str(spec_path):sha(spec_path), str(module_path):sha(module_path),
                       str(workspace/"docs/ema_evaluation_diagnostics/PREREGISTRATION.md"):
                           sha(workspace/"docs/ema_evaluation_diagnostics/PREREGISTRATION.md")}
    write_new(out/"execution-manifest.json", dict(spec=spec, execution_files=execution_files))
    try:
        frame, folds, base, members, ptarl, ptarl_parents = load_saved(spec)
        y = frame.tap_time_len.to_numpy(float)
        test = pd.read_csv(root/"复赛_test/test_samples.csv", usecols=["sample_id","spout_no"])
        if len(test) != spec["test_rows"] or not test.sample_id.is_unique:
            raise ValueError("Official test input identity mismatch")
        quotas = {int(k): int(v) for k,v in test.spout_no.value_counts().items()}
        test_hash = sha(root/"复赛_test/test_samples.csv")
        comparisons, simulations = [], []
        ema_gains = []
        for seed, fv in folds.items():
            b, old = base[seed]["tap_time_len"], base[seed]["v7_time"]
            historical_ema = b+.5*(members[seed,"tap_time_len","EMA"]-old)
            incumbent = b+spec["incumbent_q"]*(members[seed,"tap_time_len","EMA"]-old)
            c = .8*b+.2*ptarl[seed]
            original_c = .8*ptarl_parents[seed]+.2*ptarl[seed]
            ema_gains.append(gain(y,b,historical_ema))
            np.testing.assert_allclose(gain(y,ptarl_parents[seed],original_c),
                                       spec["old_ptarl_gains"][str(seed)],atol=1e-10,rtol=0)
            endpoints = dict(PTARL_Q20=c, EMA_PTARL_ENDPOINT_HALF=.5*incumbent+.5*c,
                             EMA_PLUS_PTARL_INCREMENT_Q20=incumbent+.2*(ptarl[seed]-b),
                             SAM_Q50=b+.5*(members[seed,"tap_time_len","SAM"]-old),
                             EMA_Q50=historical_ema,EMA_Q75=incumbent,PTARL_ORIGINAL_Q20=original_c)
            draws = {"uniform":sample_indices(len(y),spec["test_rows"],spec["replicates"],spec["simulation_seed"]+seed),
                     "test_spout_mix":sample_indices(len(y),spec["test_rows"],spec["replicates"],spec["simulation_seed"]+seed,
                                                    frame.spout_no,quotas)}
            for scheme, indices in draws.items():
                with (out/f"draws-{seed}-{scheme}.npy").open("xb") as stream:
                    np.save(stream,indices,allow_pickle=False)
            for name,prediction in endpoints.items():
                if (prediction < 0).any():
                    raise ValueError("Negative diagnostic endpoint; no clipping authorized")
                references = {"EMA_Q75":incumbent,"EMA_Q50":historical_ema,"V32":b,
                              "PTARL_ORIGINAL_PARENT":ptarl_parents[seed]}
                for reference_name, reference in references.items():
                    detail = reduction_detail(y,reference,prediction)
                    regions = {}
                    for feature in spec["conditional_features"]:
                        high,thresholds = fold_median_regions(frame[feature],fv)
                        regions[feature] = dict(thresholds=thresholds,
                            low=reduction_detail(y[~high],reference[~high],prediction[~high]),
                            high=reduction_detail(y[high],reference[high],prediction[high]))
                    comparisons.append(dict(seed=seed,candidate=name,reference=reference_name,**detail,
                        fold_gains=[gain(y[fv==f],reference[fv==f],prediction[fv==f]) for f in range(5)],
                        by_spout={str(s):reduction_detail(y[frame.spout_no==s],reference[frame.spout_no==s],prediction[frame.spout_no==s])
                                  for s in sorted(frame.spout_no.unique())},regions=regions))
                    for scheme,indices in draws.items():
                        values, denominators = simulated_gains(y,reference,prediction,indices)
                        simulations.append(dict(seed=seed,candidate=name,reference=reference_name,scheme=scheme,
                            **distribution(values),denominator_min=float(denominators.min()),denominator_max=float(denominators.max()),
                            draws_sha256=sha(out/f"draws-{seed}-{scheme}.npy")))
            with (out/f"paired-oof-{seed}.npz").open("xb") as stream:
                np.savez_compressed(stream,query_ids=frame.sample_id.to_numpy(str),folds=fv,y=y,V32=b,EMA=incumbent,
                                    PTARL_ORIGINAL_PARENT=ptarl_parents[seed],PTARL_RAW=ptarl[seed],**endpoints)
        np.testing.assert_allclose(np.mean(ema_gains),spec["old_ema_mean_gain"],atol=1e-11,rtol=0)
        residual_audit = historical_residual_audit(spec,frame,folds)
        verify_files(root,spec["inputs"])
        if sha(root/"复赛_test/test_samples.csv") != test_hash:
            raise ValueError("Official test inputs changed")
        for path,expected in execution_files.items():
            if sha(path) != expected:
                raise ValueError("Diagnostic sources changed during evaluation")
        report = dict(G0="passed_identity_and_independent_arithmetic_audit_pending",G1="descriptive_only_no_promotion",
                      reference=spec["reference"],rows_per_seed=len(frame),cross_seed_prediction_averaging=False,
                      old_ema_mean_reproduced=float(np.mean(ema_gains)),comparisons=comparisons,simulations=simulations,
                      test_spout_quotas=quotas,test_input_sha256=test_hash,historical_residual_audit=residual_audit,
                      parent_prediction_identity_differences={str(s):float(np.max(np.abs(base[s]["tap_time_len"]-ptarl_parents[s])))
                                                             for s in folds},
                      new_fits=0,new_split_seeds=0,packages=0,desktop_writes=0,agent_uploads=0)
        write_new(out/"report.json",report)
        return report
    except BaseException as error:
        write_new(out/"failure.json",dict(error=repr(error)))
        raise
