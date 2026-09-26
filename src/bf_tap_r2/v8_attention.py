"""From-scratch feature attention and a uniform-attention attribution control."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import rtdl_revisiting_models as rtdl
import yaml

from .data import FEATURES, TARGETS
from .v3_6_networks import NumericPreprocessor
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import nested_blend, wmape
from .v5_spec import load_v5_spec
from .v7_periodic import PeriodicRegressor, digest, file_hash, references, score_detail, write_new


class UniformAttention(nn.Module):
    """Retain V/output projections and attention dropout; disable Q/K weights."""

    def __init__(self, source):
        super().__init__()
        if source.key_compression is not None or source.value_compression is not None:
            raise ValueError("Uniform control does not support token compression")
        self.source = source

    def forward(self, x_q, x_kv):
        layer = self.source
        v = layer.W_v(x_kv)
        batch, n_q, heads = len(x_q), x_q.shape[1], layer._n_heads
        probs = v.new_full((batch * heads, n_q, x_kv.shape[1]), 1.0 / x_kv.shape[1])
        if layer.dropout is not None:
            probs = layer.dropout(probs)
        x = probs @ layer._reshape(v)
        x = x.reshape(batch, heads, n_q, v.shape[-1] // heads).transpose(1, 2).reshape(batch, n_q, v.shape[-1])
        return layer.W_out(x) if layer.W_out is not None else x


class FeatureAttentionNetwork(nn.Module):
    def __init__(self, mode, settings, categories):
        super().__init__()
        if mode not in {"uniform", "learned"}:
            raise ValueError("Unknown attention mode")
        self.base = rtdl.FTTransformer(
            n_cont_features=len(FEATURES), cat_cardinalities=[categories], d_out=1,
            n_blocks=settings["blocks"], d_block=settings["width"], attention_n_heads=settings["heads"],
            attention_dropout=settings["attention_dropout"], ffn_d_hidden=None,
            ffn_d_hidden_multiplier=settings["ffn_hidden_multiplier"],
            ffn_dropout=settings["ffn_dropout"], residual_dropout=settings["residual_dropout"],
        )
        if mode == "uniform":
            for block in self.base.backbone.blocks:
                block["attention"] = UniformAttention(block["attention"])

    def forward(self, x_num, x_cat):
        return self.base(x_num, x_cat).unsqueeze(1)


class AttentionRegressor(PeriodicRegressor):
    def _initialize(self, frame, y):
        torch.set_num_threads(1)
        torch.manual_seed(self.settings["random_seed"])
        self.preprocessor_ = NumericPreprocessor(structure="raw_tabm").fit(frame)
        self.mean_, self.std_ = float(np.mean(y)), float(np.std(y))
        if not self.std_ > 0:
            raise ValueError("Constant target")
        self.model_ = FeatureAttentionNetwork(self.recipe["attention"], self.settings,
                                              self.preprocessor_.n_spout_categories_)
        self.optimizer_ = torch.optim.AdamW(self.model_.base.make_parameter_groups(),
                                            lr=self.settings["learning_rate"],
                                            weight_decay=self.settings["weight_decay"])

    def _train(self, frame, y, epochs, validation=None):
        result = super()._train(frame, y, epochs, validation)
        if validation is not None:
            steps = max(float(v["step"]) for v in self.optimizer_.state.values() if "step" in v)
            self.selection_stopped_epoch_ = int(round(steps / math.ceil(len(frame) / self.settings["batch_size"])))
        return result

    def fit(self, frame, y):
        super().fit(frame, y)
        self.metadata_.update(attention=self.recipe["attention"],
                              selection_stopped_epoch=self.selection_stopped_epoch_,
                              budget_limited=self.selection_stopped_epoch_ >= self.settings["max_epochs"],
                              parameter_count=sum(p.numel() for p in self.model_.parameters()))
        return self


def _job(frame, folds, target, recipe, settings, fold):
    model = AttentionRegressor(recipe, settings).fit(frame.loc[folds != fold].reset_index(drop=True),
                                                    frame.loc[folds != fold, target].values)
    return model.predict(frame.loc[folds == fold]), model.metadata_


def v7_reference(root, train, folds, spec, base):
    directory = root / spec["v7_reference"]["prediction_directory"]
    manifest = json.loads((directory / "manifest.json").read_text())
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(train, index=True).values.tobytes()).hexdigest()
    if data_hash != manifest["data_digest"]:
        raise ValueError("V7 reference data identity mismatch")
    events = [json.loads(x) for x in (directory / "fit_ledger.jsonl").read_text().splitlines()]
    events = {e["key"]: e for e in events if e["event"] == "complete"}
    result, hashes = {}, {}
    for seed in spec["split_seeds"]:
        if digest(folds[seed].tolist()) != manifest["fold_digests"][str(seed)]:
            raise ValueError("V7 reference fold identity mismatch")
        values = np.full(len(train), np.nan)
        for fold in range(5):
            key = f"tap_time_len-{spec['v7_reference']['recipe']}-s{seed}-f{fold}"
            path = directory / f"{key}.npy"
            actual_hash = file_hash(path)
            if actual_hash != events[key]["prediction_sha256"]:
                raise ValueError("V7 reference prediction hash mismatch")
            values[folds[seed] == fold] = np.load(path)
            hashes[str(path.relative_to(root))] = actual_hash
        if not np.isfinite(values).all():
            raise ValueError("V7 reference coverage incomplete")
        alpha = spec["v7_reference"]["blend_weight"]
        result[seed] = {"tap_iron": base[seed]["tap_iron"],
                        "tap_time_len": (1 - alpha) * base[seed]["tap_time_len"] + alpha * values}
    return result, hashes


def run(root, output):
    from .candidate_tiers import classify_candidates
    root = Path(root).resolve()
    path = root / "configs/round2_v8/SPEC.yaml"
    spec = yaml.safe_load(path.read_text())
    versions = {k: importlib.metadata.version(k) for k in spec["runtime_versions"]}
    if versions != spec["runtime_versions"]:
        raise ValueError("V8 runtime mismatch")
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v8-feature-attention"):
        raise ValueError("V8 output must stay private")
    out.mkdir(parents=True, exist_ok=False)
    train = load_v5_training_frame(root)
    folds = {s: fold_vector(root, train, s, load_v5_spec(root)) for s in spec["split_seeds"]}
    base, q20 = references(root, train, spec)
    v7, reference_hashes = v7_reference(root, train, folds, spec, base)
    write_new(out / "manifest.json", {"spec_sha256": file_hash(path), "code_sha256": file_hash(__file__),
              "versions": versions, "author_package_source_sha256": file_hash(rtdl.__file__),
              "data_digest": hashlib.sha256(pd.util.hash_pandas_object(train, index=True).values.tobytes()).hexdigest(),
              "fold_digests": {s: digest(f.tolist()) for s, f in folds.items()},
              "v7_reference_hashes": reference_hashes,
              "dependency_code_hashes": {p.name: file_hash(p) for p in sorted((root / "src/bf_tap_r2").glob("*.py"))}})
    failures = 0
    with ProcessPoolExecutor(max_workers=spec["budget"]["workers"]) as executor:
        jobs = {executor.submit(_job, train, folds[s], target, recipe, spec["training"], f): (target, name, s, f)
                for target in TARGETS for name, recipe in spec["recipes"].items()
                for s in folds for f in range(5)}
        for future in as_completed(jobs):
            target, name, seed, fold = jobs[future]
            key = f"{target}-{name}-s{seed}-f{fold}"
            try:
                pred, metadata = future.result()
                with (out / f"{key}.npy").open("xb") as stream:
                    np.save(stream, pred)
                event = {"event": "complete", "key": key, "metadata": metadata,
                         "prediction_sha256": file_hash(out / f"{key}.npy")}
            except Exception as exc:
                failures += 1
                event = {"event": "failed", "key": key, "error": repr(exc)}
            with (out / "fit_ledger.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            print(json.dumps({k: v for k, v in event.items() if k != "metadata"}), flush=True)
    if failures:
        raise RuntimeError(f"Failed V8 fits retained: {failures}")
    metrics, records = {}, []
    for target in TARGETS:
        y = train[target].values
        metrics[target] = {spec["reference_by_target"][target]: {
            str(s): score_detail(y, base[s][target], folds[s], train.spout_no.values) for s in folds}}
        for name in spec["recipes"]:
            pred = {s: np.full(len(train), np.nan) for s in folds}
            for s in folds:
                for f in range(5):
                    pred[s][folds[s] == f] = np.load(out / f"{target}-{name}-s{s}-f{f}.npy")
                if not np.isfinite(pred[s]).all():
                    raise ValueError("Incomplete V8 prediction coverage")
            comparisons = {label: nested_blend(y, folds, {s: ref[s][target] for s in folds}, pred, spec["blend_grid"])
                           for label, ref in [("A35", base), ("Q20", q20), ("V7_candidate_pool", v7)]}
            eligible = all(g > 0 for g in comparisons["A35"]["seed_gains"].values())
            if target == "tap_time_len":
                eligible &= all(g > 0 for g in comparisons["V7_candidate_pool"]["seed_gains"].values())
            records.append({"target": target, "recipe": name, "comparisons": comparisons,
                            "confirmation_eligible": bool(eligible),
                            "single_wmape": {s: wmape(y, pred[s]) for s in folds}})
            metrics[target][name] = {}
            for s in folds:
                alpha = comparisons["A35"]["alphas"][s]
                blend = (1 - alpha) * base[s][target] + alpha * pred[s]
                metrics[target][name][str(s)] = score_detail(y, blend, folds[s], train.spout_no.values)
    tiers = classify_candidates(metrics, spec, yaml.safe_load((root / "configs/candidate_tiers.yaml").read_text()))
    summary = {"status": "development_complete", "records": records, "candidate_tiers": tiers,
               "packages": 0, "uploads": 0, "release_authorized": False}
    write_new(out / "summary.json", summary)
    for row in records:
        print(json.dumps({"target": row["target"], "recipe": row["recipe"],
                          "a35": row["comparisons"]["A35"]["seed_gains"],
                          "v7": row["comparisons"]["V7_candidate_pool"]["seed_gains"],
                          "confirmation_eligible": row["confirmation_eligible"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if os.environ.get(key) != "1":
            parser.error(f"Set {key}=1")
    run(Path.cwd(), args.output)


if __name__ == "__main__":
    main()
