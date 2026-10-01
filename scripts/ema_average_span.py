"""Entrypoint for the frozen EMA averaging span development."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from bf_tap_r2.ema_average_span import main

if __name__ == '__main__':
    main()
