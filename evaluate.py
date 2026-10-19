"""CLI: python evaluate.py --mode demo   (causal evaluation + layer summary + comparison + hypotheses)"""
import argparse
from cli import add_common_args, config_from_args
from causal.evaluate import run_causal_evaluation

if __name__ == "__main__":
    a = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(a)
    print(run_causal_evaluation(cfg))
    try:
        from causal.compare import build_comparison
        from causal.hypotheses import evaluate_hypotheses
        df = build_comparison(cfg)
        evaluate_hypotheses(cfg, df)
    except Exception as e:
        print("[evaluate] comparison/hypotheses skipped:", e)
