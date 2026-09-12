"""One zero-fit V11 package, or certified V10 stage inference without packaging."""
import argparse
from pathlib import Path
from bf_tap.optimization.platform_probe import stage_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--output', required=True, type=Path)
    predict = sub.add_parser('predict')
    predict.add_argument('--bundle', required=True, type=Path)
    predict.add_argument('--bundle-sha256', required=True)
    predict.add_argument('--data-config', required=True, type=Path)
    predict.add_argument('--stage', required=True, choices=['test_a','test_b','test_c'])
    predict.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'predict':
        stage_predict(args.bundle, args.bundle_sha256, args.data_config, args.stage, args.output)
    else:
        from bf_tap.optimization.platform_probe import prepare_probe
        prepare_probe(args.output)
