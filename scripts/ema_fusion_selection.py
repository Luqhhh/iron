"""Run the isolated preregistered paired selector."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from bf_tap_r2.ema_fusion_selection import main

if __name__ == '__main__':
    main()
