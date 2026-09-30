"""Entrypoint for isolated, append-only BASE/EMA/SAM scale diagnostics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
if "--workspace" not in sys.argv:
    sys.argv.extend(["--workspace",str(ROOT)])
from bf_tap_r2.ema_training_scale import main
if __name__=="__main__":
    main()
