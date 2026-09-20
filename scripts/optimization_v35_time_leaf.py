#!/usr/bin/env python3
import argparse
from pathlib import Path
from bf_tap.optimization.time_leaf_v35_run import cold, develop, finalize, prepare, register, run, score

DEFAULT_RUN = Path("local/runs/optimization-v0.35-time-leaf-location-and-vote-r1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "register", "p0", "develop", "score", "finalize", "cold"), nargs="?", default="run")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    {"run": run, "register": register, "p0": prepare, "develop": develop, "score": score,
     "finalize": finalize, "cold": cold}[args.action](args.output.resolve())
