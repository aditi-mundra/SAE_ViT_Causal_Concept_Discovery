"""
run_pipeline.py [CORE]
  python run_pipeline.py --mode selftest   # offline tiny ViT + synthetic data (tests the whole code path)
  python run_pipeline.py --mode smoke      # 10 images, real ViT
  python run_pipeline.py --mode demo       # CIFAR-100, 500 images, layer 9, 1024 features, Top-K=16
  python run_pipeline.py --mode full [--sweep]   # research mode; --sweep adds layers/width/type/k comparisons
Stages: extract -> train -> discover -> evaluate -> compare/hypotheses/report
"""
import argparse
import time

from cli import add_common_args, config_from_args
from utils import get_logger


def sweep_specs(cfg):
    specs = [cfg.variant(layer=l) for l in cfg.extract.layers]
    specs += [cfg.variant(width=w) for w in cfg.sweep.widths]
    specs += [cfg.variant(sae_type=t) for t in cfg.sweep.types]
    if cfg.sae.sae_type == "topk":
        specs += [cfg.variant(k=k) for k in cfg.sweep.topks]
    seen, out = set(), []
    for s in specs:
        if s.run_tag() not in seen:
            seen.add(s.run_tag()); out.append(s)
    return out


def run_one(cfg, log, stages):
    from sae.train import train_sae
    from features.discover import run_discovery
    from causal.evaluate import run_causal_evaluation
    if "train" in stages: train_sae(cfg, logger=log)
    if "discover" in stages: run_discovery(cfg, logger=log)
    if "evaluate" in stages: run_causal_evaluation(cfg, logger=log)


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--sweep", action="store_true", help="also train/evaluate layer, width, type and Top-K variants")
    ap.add_argument("--stages", default="extract,train,discover,evaluate,report")
    ap.add_argument("--force-extract", action="store_true")
    a = ap.parse_args()
    cfg, log = config_from_args(a), get_logger()
    stages = set(a.stages.split(","))
    t0 = time.time()
    if "extract" in stages:
        from activations.extract import run_extraction
        run_extraction(cfg, force=a.force_extract, logger=log)
    runs = sweep_specs(cfg) if a.sweep else [cfg]
    for c in runs:
        log.info(f"===== run {c.run_tag()} =====")
        try:
            run_one(c, log, stages)
        except Exception as e:
            log.error(f"run {c.run_tag()} failed: {type(e).__name__}: {e}")
            if not a.sweep:
                raise
    if "report" in stages:
        from causal.compare import build_comparison
        from causal.hypotheses import evaluate_hypotheses
        df = build_comparison(cfg)
        evaluate_hypotheses(cfg, df)
        try:
            from reports.generate_report import generate_report
            log.info(f"report -> {generate_report(cfg)}")
        except Exception as e:
            log.warning(f"report generation skipped: {e}")
    log.info(f"pipeline finished in {time.time()-t0:.1f}s; outputs in {cfg.exp_dir()}")
