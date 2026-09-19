#!/usr/bin/env python3
import argparse
from pathlib import Path

from bf_tap.optimization.same_spout_v32_run import cold, develop, finalize, prepare, register, run, score


DEFAULT_RUN = Path("local/runs/optimization-v0.32-same-spout-oob-responses-r1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "register", "p0", "develop", "score", "finalize", "cold"), nargs="?", default="run")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    arguments = parser.parse_args()
    actions = {"run": run, "register": register, "p0": prepare, "develop": develop,
               "score": score, "finalize": finalize, "cold": cold}
    actions[arguments.action](arguments.output.resolve())


if __name__ == "__main__":
    main()
