"""Committed source/runtime admission, then immutable reference attachment."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from .t2g_model import canonical,file_hash
from .t2g_reference import runtime

ROOT=Path(__file__).resolve().parents[2]

def source_snapshot(root=ROOT):
    root=Path(root)
    paths=[]
    for directory in ("src","tests","scripts"):
        paths.extend((root/directory).rglob("*.py"))
    for suffix in ("*.yaml","*.yml","*.json"):
        paths.extend((root/"configs").rglob(suffix))
    paths.extend([root/"pyproject.toml",root/"uv.lock",root/"docs/t2g_graph/DESIGN.md"])
    return {p.relative_to(root).as_posix():file_hash(p) for p in sorted(set(paths))}

def write_new(path,value):
    path=Path(path)
    with path.open("x") as f:
        f.write(canonical(value)+"\n"); f.flush(); os.fsync(f.fileno())

def verify_engineering_fields(m,sources,environment):
    if (m.get("kind")!="t2g-engineering-v1" or m.get("identity")!="T2G_GRAPH_V1"
        or m.get("sources")!=sources or m.get("runtime")!=environment):
        raise ValueError("Frozen engineering source/runtime differs")

def read_bound(path,sha):
    if file_hash(path)!=sha: raise ValueError("Frozen external artifact differs")
    return json.loads(Path(path).read_text())

def private_directory(output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/"local") or output==ROOT/"local":
        raise ValueError("Unique ignored private directory required")
    return output

def freeze_engineering(output):
    sources=source_snapshot()
    environment=runtime()
    if sys.version_info[:2]!=(3,12) or any(v is None for v in environment["packages"].values()):
        raise ValueError("Complete neural Python3.12 runtime required")
    changed=subprocess.check_output(["git","status","--porcelain","--",*sources],cwd=ROOT,text=True)
    if changed.strip(): raise ValueError("Commit all frozen sources before admission")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    upstream=subprocess.check_output(["git","rev-parse","@{upstream}"],cwd=ROOT,text=True).strip()
    if head!=upstream: raise ValueError("Push validated sources before freeze")
    verification=ROOT/"local/t2g-graph-v1/verification"
    tests={}
    for name in ("neural","locked"):
        p=verification/(name+".json")
        evidence=json.loads(p.read_text())
        log=Path(evidence["log"])
        if (evidence["status"]!="passed" or evidence["exit_code"]!=0 or evidence["full_suite"] is not True
            or evidence["sources"]!=sources or file_hash(log)!=evidence["log_sha256"]
            or not evidence["runtime"]["python"].startswith("3.12.")):
            raise ValueError("Passed exact-source full Python3.12 tests required")
        if name=="neural" and evidence["runtime"]!=environment: raise ValueError("Neural test runtime differs")
        tests[name]={"path":str(p),"sha256":file_hash(p),"log_sha256":file_hash(log)}
    guard=verification/"private-guard.json"
    g=json.loads(guard.read_text())
    if g["status"]!="passed" or g["exit_code"]!=0 or g["sources"]!=sources:
        raise ValueError("Exact-source private artifact guard required")
    output=private_directory(output);output.mkdir(parents=True,exist_ok=False)
    m={"kind":"t2g-engineering-v1","identity":"T2G_GRAPH_V1","workspace":str(ROOT),
        "sources":sources,"runtime":environment,"commit":head,"tests":tests,
        "guard":{"path":str(guard),"sha256":file_hash(guard)},"output":str(output),
        "budgets":{"g0_paths":2,"g0_optimizers":4,"development_fits":40,"development_optimizers":80,
            "confirmation_max_fits":40,"confirmation_max_optimizers":80},
        "spec_sha256":file_hash(ROOT/"configs/t2g_graph/SPEC.json")}
    write_new(output/"engineering.json",m)
    write_new(output/"engineering.complete.json",{"sha256":file_hash(output/"engineering.json")})
    return output/"engineering.json"

def verify_manifest(path):
    path=Path(path).resolve(); m=json.loads(path.read_text())
    if m.get("workspace")!=str(ROOT) or path.parent!=Path(m.get("output","")).resolve():
        raise ValueError("Frozen workspace/output binding differs")
    if m.get("kind")=="t2g-formal-v1":
        anchor=read_bound(path.parent/"engineering.json",m["engineering_sha256"])
        verify_engineering_fields(anchor,source_snapshot(),runtime())
        formal_anchor=json.loads((path.parent/"formal.complete.json").read_text())
        read_bound(path,formal_anchor["sha256"])
        if m["sources"]!=anchor["sources"] or m["runtime"]!=anchor["runtime"] or m["g0_status"]!="passed":
            raise ValueError("Formal engineering identity changed")
        report=read_bound(m["g0_path"],m["g0_sha256"])
        if report["status"]!="passed": raise ValueError("Passed G0 required")
        return m
    complete=json.loads((path.parent/"engineering.complete.json").read_text())
    read_bound(path,complete["sha256"])
    verify_engineering_fields(m,source_snapshot(),runtime())
    if m["workspace"]!=str(ROOT): raise ValueError("Frozen workspace differs")
    for entry in m["tests"].values():
        evidence=read_bound(entry["path"],entry["sha256"])
        if evidence["sources"]!=m["sources"]: raise ValueError("Frozen test evidence differs")
        if file_hash(evidence["log"])!=entry["log_sha256"]: raise ValueError("Frozen test log differs")
    read_bound(m["guard"]["path"],m["guard"]["sha256"])
    return m

def attach_reference(engineering_manifest,reference,output):
    from .t2g_reference import verify_reference
    engineering_manifest=Path(engineering_manifest)
    m=verify_manifest(engineering_manifest)
    checked=verify_reference(reference.identity["native_root"],reference.identity["overlay_root"])
    if checked.identity!=reference.identity: raise ValueError("Reference attachment identity differs")
    g0=engineering_manifest.parent/"g0-r1"/"preflight.json"
    marker=json.loads((g0.parent/"preflight.complete.json").read_text())
    report=read_bound(g0,marker["sha256"])
    if report["status"]!="passed" or report["engineering_sha256"]!=file_hash(engineering_manifest):
        raise ValueError("Passed same-source G0 required")
    output=Path(output).resolve()
    if output!=engineering_manifest.parent/"formal.json": raise ValueError("One immutable formal attachment required")
    formal={**m,"kind":"t2g-formal-v1","engineering_sha256":file_hash(engineering_manifest),
        "reference":checked.identity,"development":str(engineering_manifest.parent/"development-r1"),"g0_status":"passed","g0_path":str(g0),"g0_sha256":file_hash(g0)}
    write_new(output,formal)
    write_new(output.parent/"formal.complete.json",{"sha256":file_hash(output)})
    return output
