from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import catboost
import numpy
import pandas
import pytest
import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "catboost": catboost.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pytest": pytest.__version__,
        "pyyaml": yaml.__version__,
    }
    Path(args.output).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
