"""Entrypoint for the independently frozen SHORT_SPAN confirmation."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bf_tap_r2.ema_span_confirmation import main

if __name__=='__main__':main()
