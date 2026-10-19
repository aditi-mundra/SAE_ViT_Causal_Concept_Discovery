"""CLI: python extract.py --mode demo [--layer 9] [--force]"""
import argparse
from cli import add_common_args, config_from_args
from activations.extract import run_extraction

if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(run_extraction(config_from_args(a), force=a.force))
