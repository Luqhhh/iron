"""Apply existing prospective development tiers, never four-seed promotion."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from bf_tap_r2.candidate_tiers import classify_candidates
from bf_tap_r2.joint_support_audit import local_output, sha, write_new
from bf_tap_r2.laplace_leaf_experiment import inventory, reference, SPEC
from bf_tap_r2.metrics import wmape


def describe(y, p, folds, spout):
    return dict(wmape=wmape(y,p),by_fold={str(f):wmape(y[folds==f],p[folds==f]) for f in range(5)},
                by_spout={str(s):wmape(y[spout==s],p[spout==s]) for s in (1,2)})


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
    if (manifest["sources"]!=inventory(root) or manifest["action"]!="development" or cold["checked_models"]!=40 or
            cold["result_sha256"]!=sha(run/"result.json") or cold["manifest_sha256"]!=sha(run/"manifest.json")):
        raise ValueError("Native complete cold-audited run required")
    for name,h in result["output_sha256"].items():
        if sha(run/name)!=h:
            raise ValueError("Bound run changed")
    q,_=reference(root,json.loads((root/SPEC).read_text()))
    triage=yaml.safe_load((root/"configs/laplace_leaf_median/TRIAGE.yaml").read_text())
    policy=yaml.safe_load((root/"configs/candidate_tiers.yaml").read_text())
    y,spout=q["targets"][:,0],q["spout"]
    metrics={"tap_iron":{route:{} for route in ("EMA_TIME_Q75","MEDIAN_D3_A20","MEDIAN_D6_A20")}}
    for seed in (42,3407):
        folds,parent=q[f"fold-{seed}"],q[f"current-{seed}"][:,0]
        metrics["tap_iron"]["EMA_TIME_Q75"][str(seed)]=describe(y,parent,folds,spout)
        for arm in ("MEDIAN_D3","MEDIAN_D6"):
            with np.load(run/f"oof-{seed}-{arm}.npz",allow_pickle=False) as a:
                blend=.8*parent+.2*a["prediction"]
            metrics["tap_iron"][arm+"_A20"][str(seed)]=describe(y,blend,folds,spout)
    out.mkdir(parents=True,exist_ok=False)
    source_paths=("configs/laplace_leaf_median/TRIAGE.yaml","configs/candidate_tiers.yaml",
                  "src/bf_tap_r2/candidate_tiers.py","src/bf_tap_r2/weak_models.py",__file__)
    write_new(out/"manifest.json",dict(sources={str(Path(p)):sha(Path(p)) for p in source_paths},
              native_result_sha256=sha(run/"result.json"),cold_receipt_sha256=sha(run/"cold-readback.json")))
    report=classify_candidates(metrics,triage,policy)
    report.update(formal_scientific_promotion=False,confirmation_seeds_consumed=0,new_fits=0,
                  packages=0,agent_uploads=0,original_phase_finalist=result["finalist"],
                  interpretation="Existing two-seed development tiers only; original phase finalist unchanged; four-seed gate and release checks still required")
    write_new(out/"result.json",report)
    print(json.dumps(report,allow_nan=False))


if __name__=="__main__":
    main()
