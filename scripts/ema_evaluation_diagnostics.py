"""Run the frozen zero-fit diagnostics without loading a model trainer."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from bf_tap_r2.ema_evaluation_diagnostics import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec",type=Path,default=ROOT/"configs/ema_evaluation_diagnostics/SPEC.json")
    args = parser.parse_args()
    result = run(args.spec,ROOT)
    print(json.dumps({"output":json.loads(args.spec.read_text())["output"],"G0":result["G0"],
                      "new_fits":result["new_fits"]}))
