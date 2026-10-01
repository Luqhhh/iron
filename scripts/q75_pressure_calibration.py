"""Entrypoint for the isolated pressure calibration development."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bf_tap_r2.q75_pressure_calibration import main
if __name__ == '__main__': main()
