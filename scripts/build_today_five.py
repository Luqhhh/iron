"""Compose the 2026-10-01 five-pack handoff from pinned, audited originals.

Standard library only. This never trains or uploads. Missing originals and
unfinished scientific releases refuse the entire handoff before output creation.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ZIP = "Luqhhh_bf_tap_predict_round2.zip"
HEADER = ("sample_id", "pred_tap_iron", "pred_tap_time_len")
CURRENT = {
    "candidate": "EMA_TIME_Q75", "score": 96.3920,
    "zip_sha256": "41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825",
}
ORIGINALS = {
    "DE3": ("81d9d1b12c3b4fa9f40770b2ac2fef0caef7b500c5ff0ab0d7cc2948aa679ee0",
            "local/runs/de3-user-requested-release-20260930/release-r2/" + ZIP),
    "Q75": (CURRENT["zip_sha256"],
            "local/runs/ema-time-followup-20261001/probes-r2/EMA_TIME_Q75/" + ZIP),
    "Q100": ("a102cbf123981ca3103b5ba24f78b8b36203bee29810c6f0b68c6aad8ca139cb",
             "local/runs/ema-time-followup-20261001/probes-r2/EMA_TIME_Q100/" + ZIP),
    "RESERVE": ("86bf20d8cbe938f06b7550a3cf50e28fd100ba94a672c666d0aee576f25bcee3",
                "local/runs/ema-time-followup-20261001/reserve-DE3-Q75-r1/DE3_IRON_EMA_TIME_Q75_RESERVE/" + ZIP),
}
NEW_NAMES = ("EMA_IRON_EMA_TIME", "PTARL_TIME_Q20")
OUTPUT_NAMES = (
    "01_PTARL_TIME_Q20",
    "02_EMA_IRON_EMA_Q75",
    "03_DE3_IRON_EMA_PTARL_T50_50",
    "04_DE3_IRON_EMA_Q875",
    "05_DE3_IRON_EMA_Q75_RESERVE",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse(payload, ids):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if tuple(reader.fieldnames or ()) != HEADER:
        raise ValueError("Expected exactly the official three columns")
    rows = list(reader)
    if len(ids) != 322 or len(set(ids)) != 322 or any(
        re.fullmatch(r"R2S2_TEST_[0-9A-F]{12}", sid) is None for sid in ids
    ):
        raise ValueError("Expected 322 unique official V2 template IDs")
    if [row["sample_id"] for row in rows] != ids:
        raise ValueError("Missing, duplicate, extra or reordered sample IDs")
    for row in rows:
        if set(row) != set(HEADER) or None in row.values():
            raise ValueError("Malformed CSV row")
        for field in HEADER[1:]:
            number = float(row[field])
            if not math.isfinite(number) or number < 0:
                raise ValueError("Predictions must be finite and nonnegative")
    return rows


def load_zip(path, ids, expected=None):
    if expected is not None and digest(path) != expected:
        raise ValueError(f"Original ZIP identity mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["result.csv"] or archive.testzip() is not None:
            raise ValueError(f"Invalid ZIP contents or CRC: {path}")
        payload = archive.read("result.csv")
    return payload, parse(payload, ids)


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def composites(frames):
    result = []
    for recipe in ("EMA_PTARL_T50_50", "Q875"):
        rows = []
        for de3, q75, q100, ptarl in zip(
            frames["DE3"], frames["Q75"], frames["Q100"], frames["PTARL_TIME_Q20"], strict=True
        ):
            if len({r["sample_id"] for r in (de3, q75, q100, ptarl)}) != 1:
                raise ValueError("Source row identity mismatch")
            if len({r[HEADER[1]] for r in (q75, q100, ptarl)}) != 1:
                raise ValueError("Time sources must share the original V32 iron strings")
            with localcontext() as ctx:
                ctx.prec = 80
                left = Decimal(q75[HEADER[2]])
                right = q100 if recipe == "Q875" else ptarl
                mixed = (left + Decimal(right[HEADER[2]])) / 2
                value = format(mixed, ".17g")
            rows.append(dict(zip(HEADER, (de3["sample_id"], de3[HEADER[1]], value))))
        result.append(csv_bytes(rows))
    return result


def collect(args):
    root = args.artifact_root.resolve()
    known = {name: (args.input_dir / (name + ".zip") if args.input_dir else root / rel)
             for name, (_, rel) in ORIGINALS.items()}
    release = args.release_dir or root / "local/runs/ptarl-ema-exploration-release-20261001/release-r1"
    status = args.status_file or root / "EVIDENCE_STATUS.json"
    template = root / "复赛_test/result_template.csv"
    required = [*known.values(), status, template, release / "manifest.json",
                release / "release-summary.json", release / "completion-event.json"]
    for name in NEW_NAMES:
        required += [release / name / "package" / ZIP]
        required += [release / name / f for f in
                     ("audit.json", "cold.json", "independent-package-audit.json")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required originals/evidence missing; no packages written:\n" + "\n".join(missing))
    current = read_json(status)["round2_current_platform_best"]
    if {k: current.get(k) for k in CURRENT} != CURRENT:
        raise ValueError("Current incumbent changed or status is stale; revisit the prospective plan")
    with template.open(encoding="utf-8-sig", newline="") as handle:
        ids = [r["sample_id"] for r in csv.DictReader(handle)]
    payloads, frames = {}, {}
    inventory = {str(p.resolve()): digest(p) for p in required}
    inventory[str(Path(__file__).resolve())] = digest(__file__)
    for name, path in known.items():
        payloads[name], frames[name] = load_zip(path, ids, ORIGINALS[name][0])
    for i, reserve in enumerate(frames["RESERVE"]):
        if (reserve[HEADER[1]] != frames["DE3"][i][HEADER[1]]
                or reserve[HEADER[2]] != frames["Q75"][i][HEADER[2]]):
            raise ValueError("Reserve must preserve original DE3/Q75 field strings")
    terminal = read_json(release / "completion-event.json")
    summary = read_json(release / "release-summary.json")
    manifest = read_json(release / "manifest.json")
    if terminal.get("status") != "completed" or terminal.get("packages") != 2:
        raise ValueError("The two scientific exploration releases are not completed")
    if summary.get("comparison") != CURRENT or manifest.get("best") != CURRENT:
        raise ValueError("Upstream scientific release was not bound to the scored Q75 incumbent")
    entries = summary.get("packages", [])
    if len(entries) != 2 or {e.get("candidate") for e in entries} != set(NEW_NAMES):
        raise ValueError("Wrong upstream candidate inventory")
    for name in NEW_NAMES:
        work = release / name
        entry = next(e for e in entries if e["candidate"] == name)
        audit, cold, replay = (read_json(work / f) for f in
                               ("audit.json", "cold.json", "independent-package-audit.json"))
        if (audit.get("G0") != "passed" or cold.get("G0") != "passed"
                or digest(work / "audit.json") != entry["audit_sha256"]
                or digest(work / "cold.json") != entry["cold_sha256"]
                or not cold.get("training_reads_prohibited")
                or replay.get("G0") != "passed" or not replay.get("cold_csv_byte_identical")
                or replay.get("rows") != 322 or replay.get("new_fits") != 0
                or replay.get("unchanged_field_mismatches") != 0
                or not replay.get("training_reads_prohibited")
                or replay.get("zip_sha256") != entry["zip_sha256"]):
            raise ValueError(f"Original saved-state/cold/package audit is incomplete: {name}")
        payloads[name], frames[name] = load_zip(work / "package" / ZIP, ids, entry["zip_sha256"])
        if hashlib.sha256(payloads[name]).hexdigest() != entry["csv_sha256"]:
            raise ValueError("Upstream CSV identity mismatch")
        unchanged = HEADER[2] if name == NEW_NAMES[0] else HEADER[1]
        if [r[unchanged] for r in frames[name]] != [r[unchanged] for r in frames["Q75"]]:
            raise ValueError("Upstream package did not preserve its designated Q75 column")
    return ids, frames, payloads, inventory, release


def verify(out, ids, frames, payloads, inventory):
    receipt = read_json(out / "receipt.json")
    if receipt["source_inventory"] != inventory:
        raise ValueError("Source inputs changed after the handoff freeze")
    seen = set()
    for index, name in enumerate(OUTPUT_NAMES):
        path = out / name / ZIP
        record = receipt["packages"][index]
        data, rows = load_zip(path, ids, record["sha256"])
        numeric_key = tuple((r[HEADER[1]], r[HEADER[2]]) for r in rows)
        numeric_key = tuple((float(a), float(b)) for a, b in numeric_key)
        if numeric_key in seen:
            raise ValueError("Two proposed submissions contain identical predictions")
        seen.add(numeric_key)
        if index in (0, 1, 4):
            source = {0: "PTARL_TIME_Q20", 1: "EMA_IRON_EMA_TIME", 4: "RESERVE"}[index]
            if data != payloads[source]:
                raise ValueError("Original exploration CSV changed during copying")
            continue
        for i, row in enumerate(rows):
            if row[HEADER[1]] != frames["DE3"][i][HEADER[1]]:
                raise ValueError("DE3 iron field string changed")
            a = float(frames["Q75"][i][HEADER[2]])
            source = "Q100" if index == 3 else "PTARL_TIME_Q20"
            expected = math.fsum((a / 2, float(frames[source][i][HEADER[2]]) / 2))
            if not math.isclose(float(row[HEADER[2]]), expected, rel_tol=0, abs_tol=1e-10):
                raise ValueError("Independent source-column arithmetic mismatch")
    return {"G0": "five_original_or_composed_packages_verified", "packages": 5,
            "rows_each": 322, "new_fits": 0, "agent_uploads": 0,
            "scientific_cold_audits": "reused upstream reports; no models replayed by this composer",
            "platform_scores": "unmeasured", "candidate_05": "reserve_not_in_current_upload_queue"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "build", "verify"))
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", type=Path, help="Optional directory containing DE3.zip, Q75.zip, Q100.zip, RESERVE.zip")
    parser.add_argument("--release-dir", type=Path, help="Original or transferred two-release directory, including audits")
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ids, frames, payloads, inventory, release = collect(args)
    if args.mode == "check":
        print(json.dumps({"status": "ready", "source_files": len(inventory), "platform_reference": CURRENT}))
        return
    out = (args.output or args.artifact_root / "local/deliveries/today-five-20261001-r1").resolve()
    if not out.is_relative_to(args.artifact_root.resolve() / "local"):
        raise ValueError("Output must stay below the artifact root's ignored local directory")
    if any(Path(p).is_relative_to(out) for p in inventory):
        raise ValueError("Output overlaps original evidence")
    if args.mode == "verify":
        print(json.dumps(verify(out, ids, frames, payloads, inventory)))
        return
    derived = composites(frames)
    planned = (payloads["PTARL_TIME_Q20"], payloads["EMA_IRON_EMA_TIME"], derived[0], derived[1], payloads["RESERVE"])
    for blob in planned:
        parse(blob, ids)
    if len({tuple((float(r[HEADER[1]]), float(r[HEADER[2]])) for r in parse(blob, ids))
            for blob in planned}) != 5:
        raise ValueError("Candidate predictions are duplicated; revise the five-pack plan")
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for index, (name, blob) in enumerate(zip(OUTPUT_NAMES, planned, strict=True)):
        dest = out / name
        dest.mkdir()
        path = dest / ZIP
        if index in (0, 1):
            source = "PTARL_TIME_Q20" if index == 0 else "EMA_IRON_EMA_TIME"
            shutil.copyfile(release / source / "package" / ZIP, path)
        elif index == 4:
            source = args.input_dir / "RESERVE.zip" if args.input_dir else args.artifact_root / ORIGINALS["RESERVE"][1]
            shutil.copyfile(source, path)
        else:
            info = zipfile.ZipInfo("result.csv", (2026, 10, 1, 0, 0, 0))
            with zipfile.ZipFile(path, "x") as archive:
                archive.writestr(info, blob)
        records.append({"candidate": name, "sha256": digest(path), "platform_score": None,
                        "role": "reserve_not_in_current_upload_queue" if index == 4 else "exploration"})
    with (out / "receipt.json").open("x", encoding="utf-8") as handle:
        json.dump({"comparison": CURRENT, "source_inventory": inventory, "packages": records,
                   "G1": "explicit_five_pack_exploration_not_formal_promotion"}, handle, indent=2)
    subprocess.run([sys.executable, __file__, "verify", *sys.argv[2:]], check=True)
    print(str(out))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
