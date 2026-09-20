#!/usr/bin/env python3
import argparse
from pathlib import Path
from bf_tap.optimization.time_leaf_v35_run import cold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, required=True)
    cold(parser.parse_args().run.resolve())
