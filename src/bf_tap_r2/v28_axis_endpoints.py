#!/usr/bin/env python3
"""V28: iron x time axis endpoint packages from the delivered ZIPs (zero fit).

Composes the missing iron ``w=1.0`` endpoint and two combined iron/time corners
field-exactly from the hash-pinned A35/A60/V7/V12 packages, then verifies and
audits them.  Never trains, never uploads, refuses to write outside
``local/runs/round2-v28``.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .submission import ZIP_NAME, package
from .v18_compose import (
    read_sources,
    recover_endpoints,
    build_design,
    verify_payload,
    sha256_bytes,
    sha256_file,
)

ROOT_DEFAULT = Path("/home/lux1/iron")
SPEC_DEFAULT = Path("configs/round2_v28/SPEC.yaml")
V18_SPEC = Path("configs/round2_v18/SPEC.yaml")
OUTPUT_DEFAULT = Path("local/runs/round2-v28/packages-r1")
IRON_COLUMN = "pred_tap_iron"


def _context(root: Path, spec: Mapping[str, Any]):
    spec18 = yaml.safe_load((root / spec["source_spec"]).read_text())
    tables = read_sources(root, spec18)
    endpoints, recovery = recover_endpoints(tables, spec18)
    return spec18, tables, endpoints, recovery


def run(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = SPEC_DEFAULT,
        output: Path | str = OUTPUT_DEFAULT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec_path = (root / spec_path).resolve()
    if not spec_path.is_relative_to(root / "configs/round2_v28"):
        raise ValueError("V28 specification required")
    spec = yaml.safe_load(spec_path.read_text())
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v28"):
        raise ValueError("V28 packages are private and must stay under local/runs/round2-v28")
    if out.exists():
        raise FileExistsError("V28 output already exists; refusing overwrite")
    _, tables, endpoints, recovery = _context(root, spec)
    out.mkdir(parents=True)
    tolerance = float(spec["verification"]["blend_relative_tolerance"])
    report: dict[str, Any] = {"recovery": recovery, "packages": {}, "new_fits": 0,
                              "desktop_writes": 0, "agent_uploads": 0}
    for name, design in spec["designs"].items():
        payload, meta = build_design(design, endpoints, tables)
        verification = verify_payload(payload, design, endpoints, tables, tolerance)
        package_dir = out / name
        package_dir.mkdir(exist_ok=False)
        package(package_dir, payload, list(tables["a35"].ids))
        import zipfile

        with zipfile.ZipFile(package_dir / ZIP_NAME) as archive:
            if archive.namelist() != ["result.csv"] or archive.read("result.csv") != payload:
                raise ValueError(f"{name}: ZIP read-back failed")
        record = {"design": json.loads(json.dumps(design)), "readback": verification,
                  "zip_sha256": sha256_file(package_dir / ZIP_NAME),
                  "result_csv_sha256": sha256_bytes((package_dir / "result.csv").read_bytes())}
        (package_dir / "manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        report["packages"][name] = record
        print(json.dumps({"built": name, "zip_sha256": record["zip_sha256"]}))
    (out / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return report


def audit(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = SPEC_DEFAULT,
          output: Path | str = OUTPUT_DEFAULT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v28"):
        raise ValueError("V28 audit only covers private local output")
    _, tables, endpoints, recovery = _context(root, spec)
    template = (root / "复赛_test/result_template.csv").read_text(encoding="utf-8-sig")
    template_ids = [row["sample_id"] for row in csv.DictReader(io.StringIO(template, newline=""))]
    if template_ids != list(tables["a35"].ids):
        raise ValueError("Package order differs from the official result template")
    tolerance = float(spec["verification"]["blend_relative_tolerance"])
    report: dict[str, Any] = {"template_order": True, "recovery": recovery, "packages": {}}
    for name, design in spec["designs"].items():
        package_dir = out / name
        payload = (package_dir / "result.csv").read_bytes()
        import zipfile

        with zipfile.ZipFile(package_dir / ZIP_NAME) as archive:
            if archive.namelist() != ["result.csv"] or archive.read("result.csv") != payload:
                raise ValueError(f"{name}: archive mismatch")
        report["packages"][name] = {
            "readback": verify_payload(payload, design, endpoints, tables, tolerance),
            "zip_sha256": sha256_file(package_dir / ZIP_NAME)}
    (out / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "audited", "packages": len(report["packages"])}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["build", "audit"], default="build")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--spec", type=Path, default=SPEC_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    if args.mode == "audit":
        audit(args.root, args.spec, args.output)
    else:
        run(args.root, args.spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
