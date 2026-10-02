"""Hash-bound Q75 intake and paired zero-fit review; never load external models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time
import zipfile

import numpy as np

from bf_tap_r2.data import FEATURES
from bf_tap_r2.joint_support_audit import sha, write_new
from bf_tap_r2.laplace_epoch_experiment import inventory, folds_for
from bf_tap_r2.metrics import wmape
from bf_tap_r2.v2_release import load_v2

SPEC = "configs/laplace_q75_review/SPEC.json"
SOURCES = (SPEC, "docs/laplace_q75_review/PREREGISTRATION.md", __file__)


def safe_members(z):
    infos = z.infolist()
    names = [i.filename for i in infos]
    if len(names) != len(set(n.casefold() for n in names)):
        raise ValueError("Duplicate/case-colliding archive paths")
    if len(infos) > 1000 or sum(i.file_size for i in infos) > 64*1024**2:
        raise ValueError("Material exceeds frozen intake size limit")
    for i in infos:
        p = PurePosixPath(i.filename)
        if (not i.filename or p.is_absolute() or ".." in p.parts or
                "\\" in i.filename or ":" in i.filename or i.flag_bits & 1 or
                stat.S_ISLNK(i.external_attr >> 16) or i.is_dir() or
                i.file_size > 16*1024**2 or
                p.suffix.lower() in (".pkl", ".pickle", ".pt", ".cbm")):
            raise ValueError("Unsafe or unexpected archive member")
    return names


def aligned_data(q, own):
    np.testing.assert_array_equal(q["ids"], own["ids"])
    np.testing.assert_array_equal(q["numeric"], own["x"][:, :21])
    np.testing.assert_array_equal(q["targets"][:, 0], own["y"])
    np.testing.assert_array_equal(q["spout"] == 1, own["x"][:, 21])
    np.testing.assert_array_equal(q["spout"] == 2, own["x"][:, 22])


def paired_gain(y, parent, candidate, weight):
    p = (1-weight)*parent + weight*candidate
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid blend; no clipping is permitted")
    gain = 50*(wmape(y, parent)-wmape(y, p))
    scalar = 50*(math.fsum(abs(float(a)-float(b)) for a, b in zip(y, parent))-
                 math.fsum(abs(float(a)-float(b)) for a, b in zip(y, p)))/math.fsum(abs(float(a)) for a in y)
    if abs(gain-scalar) > 1e-12:
        raise ValueError("Independent arithmetic differs")
    return p, gain, abs(gain-scalar)


def intake(root, archive, out, spec):
    if sha(archive) != spec["material_sha256"]:
        raise ValueError("Material archive changed")
    material = out/"material"
    material.mkdir()
    with zipfile.ZipFile(archive) as z:
        names = safe_members(z)
        if z.testzip() is not None:
            raise ValueError("Archive CRC failure")
        m = json.loads(z.read("MANIFEST.json"))
        if set(names) != set(m["files"]) | {"MANIFEST.json", "SHA256SUMS.txt"}:
            raise ValueError("Unbound archive members")
        for name in names:
            value = z.read(name)
            if name in m["files"]:
                item = m["files"][name]
                if len(value) != item["bytes"] or hashlib.sha256(value).hexdigest() != item["sha256"]:
                    raise ValueError("Manifest file mismatch")
            target = material/name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as h:
                h.write(value)
    reader = material/"tools/verify_material.py"
    if sha(reader) != spec["reviewed_material_reader_sha256"]:
        raise ValueError("Reviewed material reader changed")
    items = json.loads((material/"data_identity/five_files.json").read_text())["files"]
    expected_paths = {"复赛_train/train_samples.csv", "复赛_train/train_features.csv",
                      "复赛_test/test_samples.csv", "复赛_test/test_features.csv", "复赛_test/result_template.csv"}
    if len(items) != 5 or {i["path"] for i in items} != expected_paths:
        raise ValueError("Wrong official data inventory")
    snapshot = out/"donor-lf-snapshot"
    data_identity = []
    for item in items:
        p = root/item["path"]
        raw = p.read_bytes()
        normalized = raw.replace(b"\r\n", b"\n")
        if len(normalized) != item["bytes"] or hashlib.sha256(normalized).hexdigest() != item["sha256"]:
            raise ValueError("Data difference is not exactly CRLF to LF")
        target = snapshot/item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as h:
            h.write(normalized)
        data_identity.append(dict(path=item["path"], raw_sha256=sha(p),
                                  donor_and_LF_sha256=sha(target), raw_bytes=len(raw),
                                  LF_bytes=len(normalized), CRLF_count=raw.count(b"\r\n")))
    cmd = [sys.executable, "-B", str(reader), str(material), "--data-root", str(snapshot),
           "--output", str(out/"material-readback.json")]
    with (out/"material-readback.stdout").open("xb") as stdout, (out/"material-readback.stderr").open("xb") as stderr:
        completed = subprocess.run(cmd, stdout=stdout, stderr=stderr)
    if completed.returncode:
        raise RuntimeError(f"Reviewed material reader exit {completed.returncode}; preserved stderr")
    receipt = json.loads((out/"material-readback.json").read_text())
    if receipt["status"] != "passed" or receipt["original_submission_sha256"] != spec["reference_zip_sha256"]:
        raise ValueError("Reference audit did not close")
    return dict(status="passed", material=str(material), archive_sha256=sha(archive),
                reviewed_reader_exit_code=completed.returncode, data=data_identity,
                material_reader_receipt_sha256=sha(out/"material-readback.json"),
                new_fits=0, model_deserializations=0, packages_created=0)


def review(root, candidate, intake_dir, out, spec):
    receipt = json.loads((intake_dir/"result.json").read_text())
    if receipt["status"] != "passed" or receipt["archive_sha256"] != spec["material_sha256"]:
        raise ValueError("Intake must pass first")
    material = Path(receipt["material"])
    manifest = json.loads((material/"MANIFEST.json").read_text())
    for name, item in manifest["files"].items():
        if sha(material/name) != item["sha256"]:
            raise ValueError("Intake material changed")
    if sha(candidate/"manifest.json") != spec["candidate_manifest_sha256"] or sha(candidate/"result.json") != spec["candidate_result_sha256"]:
        raise ValueError("Candidate identity changed")
    cm = json.loads((candidate/"manifest.json").read_text())
    if cm["source_sha256"] != inventory(root):
        raise ValueError("Frozen candidate source changed")
    if cm["input_sha256"] != {name:sha(root/name) for name in cm["input_sha256"]}:
        raise ValueError("Frozen candidate data changed")
    r = json.loads((candidate/"result.json").read_text())
    for name, h in r["output_sha256"].items():
        if sha(candidate/name) != h:
            raise ValueError("Candidate outputs changed")
    cold = json.loads((candidate/"cold-readback.json").read_text())
    if cold["checked_models"] != 40 or cold["status"] != "passed_source_data_partitions_cold_checkpoints_and_independent_arithmetic":
        raise ValueError("Original candidate cold audit missing")
    if cold["manifest_sha256"] != sha(candidate/"manifest.json") or cold["result_sha256"] != sha(candidate/"result.json"):
        raise ValueError("Cold candidate receipt identity differs")
    frame = load_v2(root/"复赛_train", "train", 2754)
    with np.load(material/"oof/original/q75-development-reference.npz", allow_pickle=False) as q, np.load(candidate/"data.npz", allow_pickle=False) as own:
        aligned_data(q, own)
        ids, y, spout = q["ids"], q["targets"], q["spout"]
        np.testing.assert_array_equal(ids, frame.sample_id.to_numpy(dtype=str))
        np.testing.assert_array_equal(y, frame[["tap_iron", "tap_time_len"]].to_numpy(dtype=float))
        roles = json.loads((material/"oof/partitions.json").read_text())["units"]
        partition_checks = []
        for seed in spec["split_seeds"]:
            folds = folds_for(frame, seed)
            np.testing.assert_array_equal(folds, q[f"fold-{seed}"])
            with np.load(candidate/f"folds-{seed}.npz", allow_pickle=False) as f:
                np.testing.assert_array_equal(folds, f["folds"])
            for fold in range(5):
                outer, query = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                role = next(a for a in roles if a["split_seed"] == seed and a["outer_fold"] == fold)
                if role["outer_training_ids"] != ids[outer].tolist() or role["outer_query_ids"] != ids[query].tolist():
                    raise ValueError("Outer partitions differ")
                inner_ref = folds_for(frame.iloc[outer], spec["reference_calibration_seed"])
                if role["selector_fitting_ids"] != ids[outer[inner_ref != 0]].tolist() or role["selector_calibration_ids"] != ids[outer[inner_ref == 0]].tolist():
                    raise ValueError("Reference inner partitions differ")
                inner_own = folds_for(frame.iloc[outer], spec["candidate_calibration_seed"])
                for recipe in spec["recipes"]:
                    with np.load(candidate/f"s{seed}-f{fold}-{recipe}-query.npz", allow_pickle=False) as a:
                        for k, v in dict(outer=outer, query=query, fit=outer[inner_own != 0], calibration=outer[inner_own == 0], inner_folds=inner_own).items():
                            np.testing.assert_array_equal(a[k], v)
                partition_checks.append(dict(seed=seed, fold=fold, outer_identity="exact", reference_inner_seed=42, candidate_inner_seed=27001))
        metrics, max_difference = {}, 0.
        for seed in spec["split_seeds"]:
            parent, folds = q[f"current-{seed}"], q[f"fold-{seed}"]
            base_score = 100-50*(wmape(y[:, 0], parent[:, 0])+wmape(y[:, 1], parent[:, 1]))
            arms = {}
            for recipe in spec["recipes"]:
                with np.load(candidate/f"oof-{seed}-{recipe}-3000.npz", allow_pickle=False) as a:
                    pred = a["prediction"].copy()
                p, gain, diff = paired_gain(y[:, 0], parent[:, 0], pred, spec["weight"])
                max_difference = max(max_difference, diff)
                arms[recipe] = dict(gain=gain, local_score=base_score+gain,
                    candidate_iron_wmape=wmape(y[:, 0], pred), blend_iron_wmape=wmape(y[:, 0], p),
                    folds={str(f):paired_gain(y[folds==f, 0], parent[folds==f, 0], pred[folds==f], spec["weight"])[1] for f in range(5)},
                    spouts={str(s):paired_gain(y[spout==s, 0], parent[spout==s, 0], pred[spout==s], spec["weight"])[1] for s in (1, 2)},
                    rows=len(ids))
            metrics[str(seed)] = dict(Q75_local_score=base_score, Q75_iron_wmape=wmape(y[:, 0], parent[:, 0]), arms=arms)
    eligible = [a for a in spec["candidate_tie_order"] if all(metrics[str(s)]["arms"][a]["gain"] > 0 for s in spec["split_seeds"])]
    mean = {a:float(np.mean([metrics[str(s)]["arms"][a]["gain"] for s in spec["split_seeds"]])) for a in spec["recipes"]}
    finalist = max(eligible, key=lambda a: mean[a]) if eligible else None
    return dict(status="passed_identity_and_independent_arithmetic", metrics=metrics, mean_gain=mean,
                eligible_for_separately_frozen_confirmation=eligible, finalist=finalist,
                partition_checks=partition_checks, scalar_max_difference=max_difference,
                candidate_completed_models=r["model_fits"], candidate_tree_fits=r["tree_fits"],
                data_CRLF_equivalence=True, exact_numerical_alignment=True,
                inner_protocols_identical=False, outer_protocols_identical=True,
                new_fits=0, confirmation_seeds_consumed=0, formal_promoted=False,
                full_fits=0, packages=0, agent_uploads=0, platform_score=None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("intake", "review"))
    p.add_argument("--archive", type=Path)
    p.add_argument("--intake", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    root = Path.cwd().resolve()
    out = args.output.resolve()
    if not out.is_relative_to(root/"local/runs"):
        raise ValueError("New private output required")
    out.mkdir(parents=True, exist_ok=False)
    spec = json.loads((root/SPEC).read_text())
    sources = {str(Path(s).resolve().relative_to(root)):sha(Path(s)) for s in SOURCES}
    write_new(out/"manifest.json", dict(spec=spec, sources=sources, started_ns=time.time_ns(),
                                      python=sys.version, numpy=np.__version__))
    try:
        result = intake(root, args.archive.resolve(), out, spec) if args.action == "intake" else review(root, args.candidate.resolve(), args.intake.resolve(), out, spec)
        if sources != {str(Path(s).resolve().relative_to(root)):sha(Path(s)) for s in SOURCES}:
            raise ValueError("Review source changed during execution")
        result.update(manifest_sha256=sha(out/"manifest.json"), completed_ns=time.time_ns())
        write_new(out/"result.json", result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        write_new(out/"FAILED.json", dict(type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == "__main__":
    main()
