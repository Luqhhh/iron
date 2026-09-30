"""Independent row arithmetic for frozen isolated endpoints and seed gates."""
from pathlib import Path
import json
import numpy as np
from .data import TARGETS
from .t2g_model import file_hash
from .t2g_run import validate_completion
from .t2g_protocol import SEEDS,ARMS

def score(frame,predictions,mask=None):
    mask=np.ones(len(frame),bool) if mask is None else np.asarray(mask,bool)
    errors=[]
    for target in TARGETS:
        y=frame[target].to_numpy(float)[mask]; p=np.asarray(predictions[target],float)[mask]
        if not len(y) or p.shape!=y.shape or not np.isfinite(p).all() or not np.isfinite(y).all() or np.abs(y).sum()<=0:
            raise ValueError("Invalid finite score input/denominator")
        errors.append(float(np.abs(p-y).sum()/np.abs(y).sum()))
    return 100-50*sum(errors)

def evaluate_target(reference,predictions,target,seeds):
    if target not in TARGETS or not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("Invalid target/seed selection")
    rows=[]; gains=[]; mechanisms=[]; scores=[]
    for s in seeds:
        current=reference.current[s]; base=score(reference.frame,current)
        historical_score=score(reference.frame,reference.historical[s])
        variants={}
        for arm in ARMS:
            member=np.asarray(predictions[s][arm],float)
            if member.shape!=(len(reference.frame),) or not np.isfinite(member).all():
                raise ValueError("Incomplete finite candidate rows")
            endpoint={t:np.asarray(current[t],float).copy() for t in TARGETS}
            endpoint[target]=.8*np.asarray(current[target],float)+.2*member
            variants[arm]=endpoint
        graph_score=score(reference.frame,variants["LEARNED_GRAPH"])
        dense_score=score(reference.frame,variants["DENSE_CONTROL"])
        gain=graph_score-base; mechanism=graph_score-dense_score
        gains.append(gain);mechanisms.append(mechanism);scores.append(graph_score)
        fold_gains=[]
        if s in reference.folds:
            for fold in range(5):
                mask=reference.folds[s]==fold
                fold_gains.append(score(reference.frame,variants["LEARNED_GRAPH"],mask)-score(reference.frame,current,mask))
        rows.append({"seed":s,"current_score":base,"graph_score":graph_score,"dense_score":dense_score,
            "current_gain":gain,"mechanism_gain":mechanism,"historical_B0_gain":graph_score-historical_score,
            "fold_gains_descriptive":fold_gains})
    lcb=float(np.mean(gains)-2.3533634348018264*np.std(gains,ddof=1)/2) if len(seeds)==4 else None
    return {"target":target,"seeds":seeds,"seed_gains":gains,"mean_gain":float(np.mean(gains)),
        "mean_mechanism_gain":float(np.mean(mechanisms)),"mean_score":float(np.mean(scores)),
        "lcb95":lcb,"per_seed":rows}

def qualifies_development(r,arm="LEARNED_GRAPH"):
    return (arm=="LEARNED_GRAPH" and r["seeds"]==[42,3407]
        and all(g>0 for g in r["seed_gains"]) and r["mean_gain"]>=.01
        and r["mean_mechanism_gain"]>0 and r["mean_score"]>=96.25)

def qualifies_confirmation(r):
    return (r["seeds"]==[42,3407,7777,12011] and all(g>0 for g in r["seed_gains"])
        and r["lcb95"] is not None and r["lcb95"]>0 and r["mean_mechanism_gain"]>0)

def read_predictions(output,reference,phase):
    output=Path(output)
    p=json.loads((output/"phase.json").read_text())
    c=json.loads((output/"complete.json").read_text())
    anchor=json.loads((output/"audit.complete.json").read_text())
    if file_hash(output/"audit.json")!=anchor["audit_sha256"]:
        raise ValueError("Audit external digest differs")
    a=json.loads((output/"audit.json").read_text())
    if (p["phase"]!=phase or a["status"]!="passed" or a["phase"]!=phase
        or a["complete_sha256"]!=file_hash(output/"complete.json")
        or a["manifest_digest"]!=p["manifest_digest"] or c["manifest_digest"]!=p["manifest_digest"]):
        raise ValueError("Passed cold audit bound to exact phase required")
    validate_completion(phase,p["selected_targets"],c["records"])
    seeds=SEEDS[phase]
    result={t:{s:{arm:np.full(len(reference.frame),np.nan) for arm in ARMS}
               for s in seeds} for t in p["selected_targets"]}
    seen=set()
    for r in c["records"]:
        key=r["key"]; directory=output/r["directory"]
        if not directory.resolve().is_relative_to(output.resolve()):
            raise ValueError("Unit escapes phase")
        if file_hash(directory/"complete.json")!=r["complete_sha256"]:
            raise ValueError("Unit complete external hash differs")
        unit=json.loads((directory/"complete.json").read_text())
        if unit["key"]!=key or unit["manifest_digest"]!=p["manifest_digest"]:
            raise ValueError("Unit identity differs")
        pred_path=directory/"predictions.npz"
        if file_hash(pred_path)!=unit["hashes"]["predictions.npz"]:
            raise ValueError("Raw prediction bytes changed")
        s,f,t,arm=key["seed"],key["fold"],key["target"],key["arm"]
        mask=reference.folds[s]==f
        ids=reference.frame.loc[mask,"sample_id"].astype(str).tolist()
        if unit["query_ids"]!=ids: raise ValueError("Held-out unit IDs differ")
        with np.load(pred_path,allow_pickle=False) as stored:
            if set(stored.files)!={"query_ids","prediction"} or stored["query_ids"].tolist()!=ids:
                raise ValueError("Raw query IDs differ")
            values=stored["prediction"]
            if values.shape!=(int(mask.sum()),) or not np.isfinite(values).all():
                raise ValueError("Invalid prediction coverage")
            result[t][s][arm][mask]=values
        identity=(t,s,f,arm)
        if identity in seen: raise ValueError("Repeated prediction cell")
        seen.add(identity)
    if any(not np.isfinite(values).all() for target in result.values() for seed in target.values() for values in seed.values()):
        raise ValueError("Missing prediction rows")
    return result

def decide(development,confirmation,reference):
    dev=read_predictions(development,reference,"development")
    metrics={t:evaluate_target(reference,dev[t],t,[42,3407]) for t in TARGETS}
    selected=[t for t in TARGETS if qualifies_development(metrics[t])]
    result={"selected_for_confirmation":selected,"development":metrics,"confirmed":[],
        "fold_criteria":"descriptive_only","reference_identity":reference.identity}
    if confirmation is not None:
        confirmed=read_predictions(confirmation,reference,"confirmation")
        if set(confirmed)!=set(selected) or not selected:
            raise ValueError("Confirmation targets differ from development qualification")
        combined={t:{**dev[t],**confirmed[t]} for t in selected}
        four={t:evaluate_target(reference,combined[t],t,[42,3407,7777,12011]) for t in selected}
        result["four_seed"]=four
        result["confirmed"]=[t for t in selected if qualifies_confirmation(four[t])]
    return result
