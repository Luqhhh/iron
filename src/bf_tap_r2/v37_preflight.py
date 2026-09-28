"""Append-only, synthetic-only full-shape equivalence and resource probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch
import yaml

from .v37_hard_tree import HardTreeEnsemble
from .v36_hard_tree_reference import HardTreeEnsemble as ReferenceEnsemble


def equivalent(settings, arm):
    results = []
    for training in (False, True):
        for nonzero in (False, True):
            reference = ReferenceEnsemble(24, arm, settings)
            if nonzero:
                with torch.no_grad():
                    reference.estimator_weights.normal_(0, .5)
            model = HardTreeEnsemble(24, arm, settings)
            model.load_state_dict(reference.state_dict())
            reference.train(training)
            model.train(training)
            torch.manual_seed(891)
            x = torch.randn(256, 24, requires_grad=True)
            z = x.detach().clone().requires_grad_(True)
            torch.manual_seed(901)
            a = reference(x)
            a.square().sum().backward()
            torch.manual_seed(901)
            b = model(z)
            b.square().sum().backward()
            pairs = {"output": (a, b), "input_gradient": (x.grad, z.grad)}
            pairs.update({name: (p.grad, dict(model.named_parameters())[name].grad)
                          for name, p in reference.named_parameters()})
            differences = {}
            for name, (left, right) in pairs.items():
                torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
                differences[name] = float((left-right).abs().max().detach())
            model.eval()
            with torch.no_grad():
                paths, _ = model.routing_weights(z.detach())
                assert torch.equal(paths.sum(-1), torch.ones_like(paths.sum(-1)))
                assert torch.all((paths == 0) | (paths == 1))
            results.append({"training": training, "nonzero_logits": nonzero,
                            "maximum_absolute_differences": differences})
    return {"mode": "equivalence", "arm": arm, "cases": results,
            "optimizer_runs": 0, "passed": True}


def probe(spec, arm):
    model = HardTreeEnsemble(24, arm, spec["model"])
    training = spec["training"]
    optimizer = torch.optim.Adam(
        [{"params": [getattr(model, name)], "lr": rate}
         for name, rate in training["learning_rates"].items()],
        betas=tuple(training["betas"]), eps=training["eps"],
        weight_decay=training["weight_decay"])
    torch.manual_seed(25001)
    x, y = torch.randn(256, 24), torch.randn(256)
    elapsed = []
    budget = spec["engineering_budget"]
    for step in range(budget["warmup_steps"]+budget["timed_steps"]):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x)-y).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        optimizer.step()
        duration = time.perf_counter()-start
        if step >= budget["warmup_steps"]:
            elapsed.append(duration)
    model.eval()
    evaluation = []
    with torch.no_grad():
        for _ in range(budget["evaluation_steps"]):
            start = time.perf_counter()
            output = model(x)
            assert torch.isfinite(output).all()
            evaluation.append(time.perf_counter()-start)
    return {"mode": "resource", "arm": arm, "optimizer_runs": 1,
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_seconds": elapsed, "evaluation_seconds": evaluation,
            "training_p95_seconds": float(np.quantile(elapsed, .95)),
            "evaluation_p95_seconds": float(np.quantile(evaluation, .95)),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}


def admission(results, available_mib):
    """Original V27 limits and projection, with no post-result adjustment."""
    if {r["arm"] for r in results} != {"GLOBAL", "INSTANCE"} or len(results) != 2:
        raise ValueError("Exactly one resource result per arm required")
    step = max(r["training_p95_seconds"] for r in results)
    evaluation = max(r["evaluation_p95_seconds"] for r in results)
    peak = max(r["peak_rss_mib"] for r in results)
    seconds = (40/4)*1.5*250*((math.ceil(2204/256)+math.ceil(1764/256))*step
                            + math.ceil(441/256)*evaluation)
    checks = {"time": seconds <= 21600, "worker_rss": peak <= 1536,
              "available_ram": 4*peak+1024 <= available_mib}
    return {"projected_seconds": seconds, "peak_worker_rss_mib": peak,
            "available_ram_mib": available_mib, "checks": checks,
            "passed": all(checks.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["equivalence", "resource"])
    parser.add_argument("arm", choices=["GLOBAL", "INSTANCE"])
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec_path = root/"configs/round2_v37/SPEC.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    run = Path(args.run_dir).resolve()
    if not run.is_relative_to(root/"local/runs/round2-v37"):
        raise ValueError("Run must be private and scoped to V37")
    run.mkdir(parents=True, exist_ok=True)
    prefix = run/f"{args.mode}-{args.arm}"
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [spec_path, Path(__file__),
                               root/"src/bf_tap_r2/v36_hard_tree.py",
                               root/"src/bf_tap_r2/v37_hard_tree.py",
                               root/"src/bf_tap_r2/v36_hard_tree_reference.py",
                               root/"uv.lock"]}
    with prefix.with_suffix(".started.json").open("x") as f:
        json.dump({"source_hashes": source_hashes, "torch": torch.__version__,
                   "numpy": np.__version__, "optimizer_budget_consumed": int(args.mode == "resource")}, f)
        f.flush()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        result = equivalent(spec["model"], args.arm) if args.mode == "equivalence" else probe(spec, args.arm)
        result["source_hashes"] = source_hashes
        with prefix.with_suffix(".json").open("x") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result))
    except BaseException as error:
        with prefix.with_suffix(".failed.json").open("x") as f:
            json.dump({"error": repr(error)}, f)
        raise


if __name__ == "__main__":
    main()
