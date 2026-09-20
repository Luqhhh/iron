#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from bf_tap.optimization.single_slot_blend_v36_run import cold, develop, finalize, prepare, register, run, score


DEFAULT_RUN = Path("local/runs/optimization-v0.36-single-slot-time-blend-r1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "register", "p0", "develop", "score", "finalize", "cold"), nargs="?", default="run")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    arguments = parser.parse_args()
    {"run": run, "register": register, "p0": prepare, "develop": develop, "score": score,
     "finalize": finalize, "cold": cold}[arguments.action](arguments.output.resolve())
