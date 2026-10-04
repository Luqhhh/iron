"""Locked entry point for the frozen iron direction audit."""
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "src"))

from bf_tap_r2.iron_direction_audit import main  # noqa: E402


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Locked Python 3.12 environment required")
    main()
