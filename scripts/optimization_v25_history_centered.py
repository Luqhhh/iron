#!/usr/bin/env python3
import argparse
from pathlib import Path

from bf_tap.optimization.history_centered_v25_run import cold, develop, finalize, prepare, register, run, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "prepare", "develop", "score", "finalize", "cold", "run"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    globals()[arguments.action](arguments.output.resolve())


if __name__ == "__main__":
    main()

