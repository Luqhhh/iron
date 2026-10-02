"""Zero-fit error geometry, not an OOF weight optimizer or platform predictor."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from bf_tap_r2.joint_support_audit import sha, write_new


def right_loss_derivative(actual, parent, candidate):
    y, p, c = (np.asarray(a, dtype=float) for a in (actual, parent, candidate))
    if y.ndim != 1 or not len(y) or y.shape != p.shape or p.shape != c.shape or not all(np.isfinite(a).all() for a in (y, p, c)):
        raise ValueError("Finite aligned nonempty arrays required")
    denominator = math.fsum(abs(float(v)) for v in y)
    if denominator <= 0:
        raise ValueError("Positive denominator required")
    residual, direction = y-p, c-p
    term = np.where(residual == 0, np.abs(direction), -np.sign(residual)*direction)
    vector = float(term.sum()/denominator)
    scalar = math.fsum(abs(float(d)) if float(r) == 0 else (-1 if r > 0 else 1)*float(d)
                       for r, d in zip(residual, direction))/denominator
    if abs(vector-scalar) > 1e-12:
        raise ValueError("Derivative independent arithmetic differs")
    return dict(right_WMAPE_derivative=scalar, right_local_score_derivative=-50*scalar,
                convex_lower_bound_slope=scalar,
                every_positive_weight_nonimproving_on_this_fixed_OOF=scalar >= 0,
                exact_zero_residual_rows=int((residual == 0).sum()),
                scalar_difference=abs(vector-scalar))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--intake", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path.cwd().resolve()
    out = args.output.resolve()
    if not out.is_relative_to(root/"local/runs"):
        raise ValueError("Private output required")
    out.mkdir(parents=True, exist_ok=False)
    specpath = root/"configs/laplace_q75_review/GEOMETRY.json"
    spec = json.loads(specpath.read_text())
    review = json.loads((args.review/"result.json").read_text())
    if review["status"] != "passed_identity_and_independent_arithmetic":
        raise ValueError("Matched review must pass")
    material = Path(json.loads((args.intake/"result.json").read_text())["material"])
    reference = material/"oof/original/q75-development-reference.npz"
    cm = json.loads((material/"MANIFEST.json").read_text())
    if sha(reference) != cm["files"]["oof/original/q75-development-reference.npz"]["sha256"]:
        raise ValueError("Reference changed")
    cr = json.loads((args.candidate/"result.json").read_text())
    bindings = {"spec":sha(specpath), "source":sha(Path(__file__)), "review":sha(args.review/"result.json"),
                "reference":sha(reference), "candidate_result":sha(args.candidate/"result.json")}
    write_new(out/"manifest.json", dict(identity=spec["identity"], spec=spec, hashes=bindings, frozen_ns=time.time_ns()))
    try:
        findings = {}
        with np.load(reference, allow_pickle=False) as q:
            for seed in spec["seeds"]:
                y, parent = q["targets"][:, 0], q[f"current-{seed}"][:, 0]
                for recipe in spec["candidate_recipes"]:
                    name = f"oof-{seed}-{recipe}-3000.npz"
                    if sha(args.candidate/name) != cr["output_sha256"][name]:
                        raise ValueError("Candidate OOF changed")
                    with np.load(args.candidate/name, allow_pickle=False) as a:
                        candidate = a["prediction"]
                    findings[f"s{seed}-{recipe}"] = right_loss_derivative(y, parent, candidate)
        for k, v in bindings.items():
            if k == "source" and sha(Path(__file__)) != v:
                raise ValueError("Diagnostic source changed")
        result = dict(status="completed_independent_directional_derivatives", findings=findings,
                      new_fits=0, alpha_search=False, selected_weight=None, confirmation_seeds_consumed=0,
                      packages=0, platform_claim=False, model_deserializations=0,
                      caveat="Exact convex geometry only for these fixed observed OOF vectors, not unseen platform rows or a whole model family",
                      manifest_sha256=sha(out/"manifest.json"))
        write_new(out/"result.json", result)
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        write_new(out/"FAILED.json", dict(type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == "__main__":
    main()
