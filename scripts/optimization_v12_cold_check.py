"""Cold process, old bundle histories only, zero CatBoost and LAD fits."""
import argparse
from pathlib import Path
from bf_tap.optimization.recency_run import main

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True,type=Path)
    main(parser.parse_args().source,'cold')
