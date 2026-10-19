"""CLI: python train.py --mode demo --layer 9 --sae-width 4096 --sae-type topk --k 32"""
import argparse
from cli import add_common_args, config_from_args
from sae.train import train_sae

if __name__ == "__main__":
    a = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    print(train_sae(config_from_args(a)))
