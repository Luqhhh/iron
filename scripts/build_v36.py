#!/usr/bin/env python3
"""Build the one frozen v0.36 package; standard-library dependencies only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bf_tap.optimization.single_slot_blend_v36 import build_package


DEFAULT_V34 = Path("local/runs/optimization-v0.34-oob-leaf-median-bagging-r2/submissions/V34T_OOB_LEAF_MEDIAN_BAGGING_TIME")
DEFAULT_V30 = Path("local/runs/optimization-v0.30-oob-compose-and-time-growth-r6/submissions/V30A_OOB_BOTH_TARGETS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v34-result", type=Path, default=DEFAULT_V34 / "result.csv")
    parser.add_argument("--v34-zip", type=Path, default=DEFAULT_V34 / "Luqhhh_bf_tap_predict_prelim.zip")
    parser.add_argument("--v30-result", type=Path, default=DEFAULT_V30 / "result.csv")
    parser.add_argument("--v30-zip", type=Path, default=DEFAULT_V30 / "Luqhhh_bf_tap_predict_prelim.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=335)
    arguments = parser.parse_args()
    receipt = build_package(
        v34_result=arguments.v34_result,
        v34_zip=arguments.v34_zip,
        v30_result=arguments.v30_result,
        v30_zip=arguments.v30_zip,
        output_dir=arguments.output,
        expected_rows=arguments.expected_rows,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
