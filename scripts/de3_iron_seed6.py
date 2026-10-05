"""Locked entry point for the frozen DE3 iron six-seed extension."""
import os
import sys
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "src"))

from bf_tap_r2.de3_iron_seed6 import main  # noqa: E402


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Locked Python 3.12 environment required")
    main()
