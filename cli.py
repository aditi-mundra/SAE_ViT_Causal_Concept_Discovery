"""cli.py [CORE]: shared argparse options -> ExperimentConfig."""
from __future__ import annotations

import argparse

from config import PRESETS, ExperimentConfig, get_config


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--mode", default="demo", choices=list(PRESETS), help="config preset")
    p.add_argument("--exp-name", default=None)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--sae-width", type=int, default=None)
    p.add_argument("--sae-type", default=None, choices=["standard", "l1", "topk"])
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--l1-lambda", type=float, default=None)
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--normalization", default=None, choices=["standardize", "l2", "none"])
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    cfg = get_config(args.mode)
    if args.exp_name: cfg.name = args.exp_name
    if args.seed is not None: cfg.seed = args.seed
    if args.device: cfg.device = args.device
    if args.sae_width: cfg.sae.width = args.sae_width
    if args.sae_type: cfg.sae.sae_type = args.sae_type
    if args.k: cfg.sae.k = args.k
    if args.epochs: cfg.sae.epochs = args.epochs
    if args.lr: cfg.sae.lr = args.lr
    if args.l1_lambda is not None: cfg.sae.l1_lambda = args.l1_lambda
    if args.num_images: cfg.data.num_images = args.num_images
    if args.dataset: cfg.data.dataset = args.dataset
    if args.model: cfg.model.name = args.model
    if args.normalization: cfg.extract.normalization = args.normalization
    if args.layer is not None:
        cfg.layer = args.layer
        if args.layer not in cfg.extract.layers:
            cfg.extract.layers = sorted(set(cfg.extract.layers + [args.layer]))
    cfg.validate()
    return cfg
