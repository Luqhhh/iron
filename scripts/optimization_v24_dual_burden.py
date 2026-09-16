#!/usr/bin/env python3
import argparse
from pathlib import Path
from bf_tap.optimization.dual_burden_v24_run import cold, develop, finalize, prepare, register, run, score


def main():
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("register","prepare","develop","score","finalize","cold","run"))
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();root=args.output.resolve()
    globals()[args.action](root)


if __name__=="__main__":main()

