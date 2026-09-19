#!/usr/bin/env python3
import argparse
from pathlib import Path

from bf_tap.optimization.qrf_sampling_v33_run import cold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    cold(args.run.resolve())
