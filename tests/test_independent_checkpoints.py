from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.component_regularization import ComponentRegressor
from bf_tap_r2.component_regularization_run import RECIPE, outputs
from bf_tap_r2.independent_checkpoints import (EpochSelector, clean_query, native_parts,
    verify_disjoint, select_epochs, fit_background, fresh_refit)
from bf_tap_r2.independent_checkpoints_audit import verify_epoch_unit, cold_check
from bf_tap_r2.independent_checkpoints_run import load_spec, select_finalists, private_path, finish_unit
from bf_tap_r2.v12_joint import JointRegressor
from bf_tap_r2.v7_periodic import PeriodicRegressor
from bf_tap_r2.v49_run import verified_unit, unit_id


def sample(n=80):
    rng = np.random.default_rng(19)
    f = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    f["sample_id"] = [f"synthetic-checkpoint-{i}" for i in range(n)]
    f["spout_no"] = rng.integers(1, 3, n)
    f["tap_iron"] = 500+10*f[FEATURES[0]]
    f["tap_time_len"] = 100+2*f[FEATURES[1]]
    return f


def tiny(seed=42):
    return dict(random_seed=seed, inner_seed=42, width=16, blocks=1, tabm_k=2,
                dropout=.1, embedding_dim=4, n_frequencies=2, lite=True,
                loss="mse", optimizer="adamw", learning_rate=.001, weight_decay=.0001,
                batch_size=32, max_epochs=4, patience=2, min_delta=1e-5)


@pytest.mark.parametrize("target", TARGETS)
def test_same_native_trajectory_selected_epoch_and_refit(tmp_path, target):
    f = sample(); training = f.iloc[:65].reset_index(drop=True); query = clean_query(f.iloc[65:])
    settings = tiny(); _, _, c = native_parts(training)
    directory = tmp_path/"select"; directory.mkdir()
    selector = EpochSelector(settings, target, directory)
    selection = selector.select(training, np.full(len(c), 250 if target == "tap_iron" else 50), query)
    native = JointRegressor(RECIPE, settings).fit(training, training[outputs(target)].to_numpy())
    if target == "tap_time_len":
        single = PeriodicRegressor(RECIPE, settings).fit(training, training[target].to_numpy())
        assert single.metadata_["selected_epoch"] == native.metadata_["selected_epoch"]
        np.testing.assert_array_equal(single.predict(query), native.predict(query)[:, 0])
    assert selection["epochs"]["E-NATIVE"] == native.metadata_["selected_epoch"]
    refit_dir = tmp_path/"refit"; refit_dir.mkdir()
    refit = fresh_refit(training, target, settings, selection["epochs"]["E-NATIVE"], refit_dir)
    np.testing.assert_array_equal(refit.predict(query), native.predict(query))
    for key, state in native.model_.state_dict().items(): assert torch.equal(state, refit.model_.state_dict()[key])
    verify_epoch_unit(directory, training, c, np.full(len(c), 250 if target == "tap_iron" else 50), target, settings)
    cold = ComponentRegressor.load(refit_dir/"refit.pt")
    assert cold_check(cold, query, refit.predict(query), 5e-4) <= 5e-4


def test_training_seed_never_changes_inner_partition():
    f = sample()
    masks = [native_parts(f, tiny(seed)["inner_seed"])[0] for seed in (42, 104729, 130363)]
    for mask in masks[1:]: np.testing.assert_array_equal(mask, masks[0])


def test_complete_network_seeds_have_distinct_parameters():
    f = sample(); y = f[list(TARGETS)].to_numpy(); states = []
    for seed in (42, 104729, 130363):
        model = ComponentRegressor(RECIPE, tiny(seed), "BASE", {})
        model._initialize(f, y)
        states.append(torch.cat([v.flatten() for v in model.model_.state_dict().values()]))
    assert all(not torch.equal(a, b) for i, a in enumerate(states) for b in states[i+1:])


def test_observers_use_same_visited_set_ties_earliest_and_native_tolerance():
    history = [dict(epoch=i+1, validation_mae=n, target_mae=t, compose_mae=c)
               for i, (n, t, c) in enumerate([(1., 3., 2.), (.9, 1., 2.), (.900001, 1., .5), (.91, 2., .5)])]
    assert select_epochs(history, tiny()) == {"E-NATIVE": 2, "E-TARGET": 2, "E-COMPOSE": 3}
    with pytest.raises(ValueError, match="beyond native patience"):
        select_epochs(history+[dict(history[-1], epoch=5)], dict(tiny(), max_epochs=5))
    with pytest.raises(ValueError, match="Incomplete"):
        select_epochs(history[1:], tiny())


@pytest.mark.parametrize("metric", ["validation_mae", "target_mae", "compose_mae"])
def test_nonfinite_selector_metrics_rejected(metric):
    row = dict(epoch=1, validation_mae=1., target_mae=1., compose_mae=1.)
    row[metric] = float("nan")
    with pytest.raises(ValueError, match="Invalid selector metric"): select_epochs([row], dict(tiny(), max_epochs=1))


def test_partition_leakage_and_query_labels_rejected():
    f = sample(); a, c = f.iloc[:40], f.iloc[40:60]
    verify_disjoint(a, c, clean_query(f.iloc[60:]))
    with pytest.raises(ValueError, match="overlap"): verify_disjoint(f, c)
    with pytest.raises(ValueError, match="contains labels"): verify_disjoint(a, c, f.iloc[60:])


def test_background_factory_only_receives_A_and_unlabelled_C(monkeypatch, tmp_path):
    import bf_tap_r2.v4_1_reference as reference
    import bf_tap_r2.v5_replicate as replicate
    import bf_tap_r2.v5_spec as spec_module
    f = sample(); _, a, c = native_parts(f); calls = []
    class Factory:
        def __init__(self, root, workers):
            assert workers == 4
            self.a_factory = SimpleNamespace(_recovery=SimpleNamespace(BASE_NAMES=list(range(20))),
                _expert_ids={"tap_iron": list(range(3)), "tap_time_len": list(range(2))})
            self.selected_experts = {"tap_iron": [1, 2], "tap_time_len": [3, 4]}
        def fit_predict(self, train, query):
            assert list(train.sample_id) == list(a.sample_id)
            assert list(query.sample_id) == list(c.sample_id)
            assert not any(t in query for t in TARGETS)
            calls.append("V36")
            return {"b36": {"tap_iron": np.full(len(query), 500.), "tap_time_len": np.full(len(query), 100.)}, "meta": {}}
    def n_fit(train, query, target):
        assert list(train.sample_id) == list(a.sample_id)
        assert not any(t in query for t in TARGETS)
        calls.append("N"); return np.full(len(query), 120.)
    monkeypatch.setattr(reference, "V36FixedRecipeFactory", Factory)
    monkeypatch.setattr(replicate, "load_candidate", lambda *args: (n_fit, {}))
    monkeypatch.setattr(spec_module, "load_v5_spec", lambda *args: None)
    spec = dict(budget=dict(reference_workers=4, inner_reference_pipeline_fits_per_factory=30), reference=dict(n_trial="v36-s1-N-0048"))
    values, metadata = fit_background(tmp_path, a, c, spec, tmp_path)
    assert calls == ["V36", "N"]
    np.testing.assert_array_equal(values["tap_iron"], np.full(len(c), 250.))
    np.testing.assert_array_equal(values["tap_time_len"], np.full(len(c), 56.))
    assert not metadata["query_labels_received"]
    with pytest.raises(ValueError, match="overlap"): fit_background(tmp_path, f, c, spec, tmp_path)


def test_epoch_predictions_tampering_fails_audit(tmp_path):
    f = sample(); _, _, c = native_parts(f); background = np.full(len(c), 50.)
    selector = EpochSelector(tiny(), "tap_time_len", tmp_path); selector.select(f, background)
    path = tmp_path/"epoch-predictions.npz"
    with np.load(path) as saved: data = {k: saved[k].copy() for k in saved.files}
    data["standardized"][0, 0, 0] += .01
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="metric arithmetic"):
        verify_epoch_unit(tmp_path, f, c, background, "tap_time_len", tiny())


def test_finalist_floor_and_two_positive_full_splits():
    spec = dict(promotion=dict(development_mean_gain_ge=.01), tie_preference_by_target={t: ["E-TARGET", "E-COMPOSE", "DE3"] for t in TARGETS})
    def row(arm, values): return dict(target="tap_iron", arm=arm, seeds={str(i): dict(gain=g) for i, g in enumerate(values)}, paired=dict(mean=np.mean(values)))
    assert select_finalists([row("DE3", [.009, .009])], spec, "DE3")["tap_iron"] is None
    assert select_finalists([row("DE3", [-.01, .04])], spec, "DE3")["tap_iron"] is None
    rows = [row("E-NATIVE", [0., 0.]), row("E-TARGET", [.01, .012]), row("E-COMPOSE", [.01, .012])]
    assert select_finalists(rows, spec, "E-COMPOSE")["tap_iron"] == "E-TARGET"


def test_append_only_run_and_nested_unit_identities(tmp_path):
    out = tmp_path/"out"; out.mkdir()
    manifest = dict(identity="frozen", output_directory=str(out))
    ids = []
    for name in ("iron", "time"):
        directory = out/name/"training-seed-42"; directory.mkdir(parents=True)
        (directory/"artifact.txt").write_text(name)
        finish_unit(directory, manifest)
        key = str(directory.relative_to(out))
        ids.append(unit_id(manifest, key))
        verified_unit(directory, ids[-1])
        with pytest.raises(FileExistsError): finish_unit(directory, manifest)
    assert ids[0] != ids[1]


def test_spec_preserves_frozen_gates_and_native_contract():
    spec = load_spec(Path.cwd())
    assert spec["training_seeds"] == [42, 104729, 130363]
    assert spec["promotion"]["local_working_gate"] == 96.25
    assert spec["reference"]["time_weights"] == dict(v36=.2, n0048=.3, v7_periodic=.5)
    with pytest.raises(ValueError, match="private"): private_path(Path.cwd(), Path("result"))


def test_old_reference_audit_uses_old_partition_contract(monkeypatch):
    import bf_tap_r2.independent_checkpoints_run as runner
    spec = load_spec(Path.cwd()); calls = []
    def observe(root, old):
        calls.append(old["calibration"]["split_seed"])
        assert old["calibration"]["split_seed"] == 27001
        assert old["reference"] == spec["reference"]
        return {"status": "original_cache_verified"}
    monkeypatch.setattr(runner, "verify_reference_cache", observe)
    assert runner.verify_original_reference(Path.cwd(), spec)["status"] == "original_cache_verified"
    assert calls == [27001] and spec["calibration"]["split_seed"] == 42


@pytest.mark.parametrize("queue,seeds", [("DE3", [42, 3407]), ("DE3", [271828, 314159]),
                                        ("E-COMPOSE", [42, 3407]), ("E-COMPOSE", [271828, 314159])])
def test_full_coverage_summary_and_confirmation_scope(tmp_path, queue, seeds):
    from bf_tap_r2.independent_checkpoints_run import summarize
    spec = load_spec(Path.cwd()); frame = sample(50)
    folds = {s: np.arange(len(frame)) % 5 for s in seeds}
    # A fabricated complete OOF input checks arithmetic/coverage only.
    for seed, fv in folds.items():
        for fold in range(5):
            mask = fv == fold; ids = frame.loc[mask, "sample_id"].to_numpy(dtype=str)
            ref = tmp_path/f"reference-s{seed}-f{fold}"; ref.mkdir()
            np.savez_compressed(ref/"predictions.npz", tap_iron=frame.loc[mask, "tap_iron"].to_numpy()+2,
                tap_time_len=frame.loc[mask, "tap_time_len"].to_numpy()+1,
                v12_iron=frame.loc[mask, "tap_iron"].to_numpy()+2,
                v7_time=frame.loc[mask, "tap_time_len"].to_numpy()+1, query_ids=ids)
            for target in TARGETS:
                if seeds[0] == 271828 and target == "tap_time_len": continue
                d = tmp_path/f"{target}-{queue}-s{seed}-f{fold}"; d.mkdir()
                y = frame.loc[mask, target].to_numpy(); current = y+(2 if target == "tap_iron" else 1)
                values = dict(query_ids=ids)
                if queue == "DE3":
                    values.update(DE3=(current+y+y)/3, seed_42=current, seed_104729=y, seed_130363=y)
                else:
                    values.update({"E-NATIVE": current, "E-TARGET": y, "E-COMPOSE": y+.1})
                np.savez_compressed(d/"predictions.npz", **values)
    (tmp_path/"fit_ledger.jsonl").write_text('{}\n')
    result = summarize(tmp_path, frame, folds, spec, queue)
    assert result["records"]
    if seeds[0] == 271828:
        assert result["tiers"] is None
        assert result["selected_for_confirmation"]["tap_time_len"] is None
    if queue == "DE3":
        assert not result["arithmetic_audit"]["provenance_verified"]
    else:
        record = next(r for r in result["records"] if r["target"] == "tap_iron" and r["arm"] == "E-COMPOSE")
        assert not record["composition_mechanism_supported"]


@pytest.mark.parametrize("queue,target", [(q, t) for q in ("DE3", "E-COMPOSE") for t in TARGETS])
def test_unit_model_artifacts_and_fit_accounting(tmp_path, queue, target):
    from bf_tap_r2.independent_checkpoints_run import candidate_unit
    f = sample(); training = f.iloc[:65].reset_index(drop=True); query = clean_query(f.iloc[65:])
    original_root = tmp_path/"original"; original_root.mkdir()
    original = dict(identity="old", output_directory=str(original_root))
    (original_root/"manifest.json").write_text(json.dumps(original))
    source = original_root/f"{target}-BASE-s42-f0"; source.mkdir()
    model = ComponentRegressor(RECIPE, tiny(), "BASE", {}, source).fit(training, training[outputs(target)].to_numpy())
    pred = model.predict(query)
    np.savez_compressed(source/"predictions.npz", prediction=pred, query_ids=query.sample_id.to_numpy(dtype=str))
    (source/"metadata.json").write_text(json.dumps({"model": model.metadata_}))
    finish_unit(source, original)
    out = tmp_path/"out"; out.mkdir()
    manifest = dict(identity="new", output_directory=str(out))
    rd = out/"reference-s42-f0"; rd.mkdir()
    np.savez_compressed(rd/"predictions.npz", **{"v12_iron" if target == "tap_iron" else "v7_time": pred[:, 0]})
    if queue == "E-COMPOSE":
        from bf_tap_r2.v7_periodic import digest
        _, a, c = native_parts(training)
        bd = out/"inner-reference-s42-f0"; bd.mkdir()
        np.savez_compressed(bd/"predictions.npz", query_ids=c.sample_id.to_numpy(dtype=str),
                            **{target: np.full(len(c), 250 if target == "tap_iron" else 50)})
        (bd/"metadata.json").write_text(json.dumps({"fit_ids_digest": digest(a.sample_id.tolist())}))
    spec = dict(training={target: tiny()}, training_seeds=[42, 104729, 130363])
    result = candidate_unit(tmp_path, out, training, query, 42, 0, target, queue, spec, manifest, source)
    verified_unit(out/result["key"], result["identity"])
    assert result["reused_native_units"] >= 1
    if queue == "DE3":
        assert result["selector_fits"] == result["refit_fits"] == 2
    else:
        assert result["selector_fits"] == 1
        metadata = json.loads((out/result["key"]/"metadata.json").read_text())
        assert metadata["native_selector_state_exact"]
        assert result["refit_fits"] <= 2
