#!/usr/bin/env python3
import argparse
from pathlib import Path

from bf_tap.optimization.qrf_iron_leaf_recency_v27_run import cold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    cold(arguments.run.resolve())
