"""CLI: python discover.py --mode demo            (full discovery)
      python discover.py --mode demo --feature 172  (report for one feature)"""
import argparse
from cli import add_common_args, config_from_args
from context import RunContext
from features.discover import run_discovery
from features.report import feature_report

if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--feature", type=int, default=None)
    a = ap.parse_args()
    cfg = config_from_args(a)
    if a.feature is None:
        print(run_discovery(cfg))
    else:
        ctx = RunContext(cfg)
        s = ctx.summary().loc[a.feature]
        print(s.to_string())
        print(feature_report(ctx, a.feature))
