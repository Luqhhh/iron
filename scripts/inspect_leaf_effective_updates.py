"""Post-completion-only exact floating-parameter update audit, zero new fits."""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np

from bf_tap_r2.joint_support_audit import local_output, sha, write_new
from bf_tap_r2.laplace_leaf_experiment import inventory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    root=Path.cwd().resolve()
    run,out=local_output(root,args.run),local_output(root,args.output)
    r=json.loads((run/"result.json").read_text())
    m=json.loads((run/"manifest.json").read_text())
    c=json.loads((run/"cold-readback.json").read_text())
    if (m["identity"]!="XJN_LAD_LEAF_MEDIAN_IRON_V1" or m["action"]!="development" or
            c["checked_models"]!=40 or c["manifest_sha256"]!=sha(run/"manifest.json") or
            c["result_sha256"]!=sha(run/"result.json") or m["sources"]!=inventory(root)):
        raise ValueError("Completed native cold-audited forty-model run required")
    for name,h in r["output_sha256"].items():
        if sha(run/name)!=h:
            raise ValueError("Original run output changed")
    out.mkdir(parents=True,exist_ok=False)
    specpath=root/"configs/laplace_leaf_median/EFFECTIVE_UPDATES.json"
    spec=json.loads(specpath.read_text())
    bindings=dict(spec_sha256=sha(specpath),source_sha256=sha(Path(__file__)),
                  original_result_sha256=sha(run/"result.json"),cold_receipt_sha256=sha(run/"cold-readback.json"))
    write_new(out/"manifest.json",dict(spec=spec,bindings=bindings,started_ns=time.time_ns()))
    try:
        with np.load(run/"data.npz",allow_pickle=False) as a:
            x,y=a["x"],a["y"]
        units=[]
        for unit in r["units"]:
            key=unit["key"]
            with np.load(run/f"{key}-query.npz",allow_pickle=False) as a:
                fit=a["fit"]
            # Only this native, fully hash-bound run is allowed above.
            with (run/f"{key}-selector.pkl").open("rb") as h:
                model=pickle.load(h)
            z=(x[fit]-model.x_mean_)/model.x_scale_
            sy=(y[fit]-model.y_median_)/model.y_scale_
            mu=np.zeros(len(fit))
            exact_changed,material_changed,loss_changed,proposal_without_change=0,0,0,0
            selected_exact,selected_material=0,0
            previous=float(np.abs(sy).mean())
            for epoch,(tree,values,event) in enumerate(zip(model.trees_,model.leaf_values_,model.history_,strict=True),1):
                update=.05*values[tree.apply(z)]
                before=mu.copy()
                mu+=update
                delta=mu-before
                changed=int(np.count_nonzero(delta))
                material=int(np.count_nonzero(np.abs(delta)>spec["descriptive_material_change_threshold_normalized"]))
                proposal=int(np.count_nonzero(update))
                loss=float(np.abs(sy-mu).mean())
                if proposal!=event["nonzero_update_rows"] or loss!=event["train_mae_normalized"]:
                    raise ValueError("Independent native update history differs")
                exact_changed+=changed>0
                material_changed+=material>0
                loss_changed+=loss<previous
                proposal_without_change+=(proposal>0 and changed==0)
                if epoch<=model.selected_epoch_:
                    selected_exact+=changed>0
                    selected_material+=material>0
                previous=loss
            units.append(dict(key=key,seed=unit["seed"],fold=unit["fold"],arm=unit["arm"],
                              selected_epoch=model.selected_epoch_,complete_epochs=len(model.trees_),
                              exact_parameter_change_epochs=exact_changed,material_parameter_change_epochs=material_changed,
                              selected_exact_change_epochs=selected_exact,selected_material_change_epochs=selected_material,
                              strictly_decreasing_training_loss_epochs=loss_changed,
                              nonzero_proposal_but_no_float_parameter_change_epochs=proposal_without_change))
        if bindings["source_sha256"]!=sha(Path(__file__)) or bindings["spec_sha256"]!=sha(specpath):
            raise ValueError("Diagnostic specification changed")
        result=dict(status="completed_twenty_selector_effective_parameter_audit",units=units,new_fits=0,
                    checkpoints_changed=False,weights_changed=False,packages=0,
                    manifest_sha256=sha(out/"manifest.json"))
        write_new(out/"result.json",result)
        print(json.dumps(result,allow_nan=False))
    except Exception as exc:
        write_new(out/"FAILED.json",dict(type=type(exc).__name__,error=str(exc)))
        raise


if __name__=="__main__":
    main()
