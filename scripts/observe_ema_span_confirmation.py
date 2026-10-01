"""Reuse the original actual-exit/600-second observer for this controller."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bf_tap_r2.ema_span_confirmation_observer import main

if __name__=='__main__':main()
