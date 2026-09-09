"""Score already generated four-arm CSVs; does not establish temporal legality."""
import argparse
from pathlib import Path
import pandas as pd
from bf_tap.artifacts import atomic_write_json
from bf_tap.optimization.refresh_factorial import score_factorial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame = pd.read_csv(args.input, dtype={'sample_id':str})
    report = ({c:score_factorial(p) for c,p in frame.groupby('component')}
              if 'component' in frame else score_factorial(frame))
    atomic_write_json(args.output, report)

if __name__ == '__main__':
    main()
