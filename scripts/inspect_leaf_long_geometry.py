"""Completed-run convex direction and saved-trajectory diagnostics; no fit/search."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from bf_tap_r2.joint_support_audit import local_output, sha, write_new
from bf_tap_r2.laplace_leaf_long_experiment import inventory, original, SPEC


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    root=Path.cwd().resolve()
    run,out=local_output(root,args.run),local_output(root,args.output)
    manifest=json.loads((run/"manifest.json").read_text())
    result=json.loads((run/"result.json").read_text())
    cold=json.loads((run/"cold-readback.json").read_text())
    if (manifest["action"]!="development" or manifest["sources"]!=inventory(root)
            or cold["checked_models"]!=20 or cold["maximum_cold_difference"]!=0
            or cold["prefix_models_checked"]!=10
            or cold["manifest_sha256"]!=sha(run/"manifest.json")
            or cold["result_sha256"]!=sha(run/"result.json")):
        raise ValueError("Original complete twenty-model cold-audited long run required")
    for name,digest in result["output_sha256"].items():
        if sha(run/name)!=digest: raise ValueError("Original long output changed")
    _,q=original(root,json.loads((root/SPEC).read_text()))
    path=root/"configs/laplace_leaf_long/GEOMETRY.json"
    spec=json.loads(path.read_text())
    helper=root/"scripts/inspect_laplace_q75_geometry.py"
    module_spec=importlib.util.spec_from_file_location("trusted_native_convex_geometry",helper)
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    bindings={"script":sha(Path(__file__)),"spec":sha(path),"derivative_helper":sha(helper),
              "original_result":sha(run/"result.json"),"original_cold":sha(run/"cold-readback.json")}
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/"manifest.json",dict(identity=spec["identity"],spec=spec,bindings=bindings,started_ns=time.time_ns()))
    try:
        findings={}
        for seed in spec["seeds"]:
            with np.load(run/f"oof-{seed}.npz",allow_pickle=False) as file:
                prediction=file["prediction"]
            findings[str(seed)]=module.right_loss_derivative(q["targets"][:,0],q[f"current-{seed}"][:,0],prediction)
        profiles=[]
        for unit in result["units"]:
            trace=json.loads((run/f"{unit['key']}-selector-trace.json").read_text())
            history=trace["history"]
            if len(history)!=12000 or [h["epoch"] for h in history]!=list(range(1,12001)):
                raise ValueError("Incomplete saved trajectory")
            profiles.append(dict(key=unit["key"],selected_epoch=unit["selected_epoch"],
                selected_in_last_ten_percent=unit["selected_epoch"]>=spec["near_end_descriptive_fraction"]*12000,
                checkpoints={str(n):dict(train_mae_normalized=history[n-1]["train_mae_normalized"],
                                         calibration_mae=history[n-1]["calibration_mae"]) for n in spec["checkpoints"]},
                final_thousand_calibration_improved=history[-1]["calibration_mae"]<history[10999]["calibration_mae"]))
        if bindings["script"]!=sha(Path(__file__)) or bindings["spec"]!=sha(path) or bindings["derivative_helper"]!=sha(helper):
            raise ValueError("Diagnostic source changed")
        report=dict(status="completed_native_convex_geometry_and_saved_trajectory_description",findings=findings,
            profiles=profiles,new_fits=0,model_deserializations=0,alpha_search=False,
            selected_weight=None,selection_changed=False,packages=0,platform_claim=False,
            caution="Fixed observed OOF geometry only. Near-end fraction is descriptive, not a new stopping/selection gate.",
            manifest_sha256=sha(out/"manifest.json"))
        write_new(out/"result.json",report)
        print(json.dumps(report,allow_nan=False))
    except Exception as exc:
        write_new(out/"FAILED.json",dict(type=type(exc).__name__,error=str(exc)))
        raise


if __name__=="__main__": main()
