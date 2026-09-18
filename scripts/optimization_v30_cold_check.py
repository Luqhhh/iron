#!/usr/bin/env python3
import argparse
from pathlib import Path

from bf_tap.optimization.oob_compose_growth_v30_run import cold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    cold(arguments.run.resolve())
