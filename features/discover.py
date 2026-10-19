"""
features/discover.py [CORE]
Feature discovery on the FEATURE-VALIDATION split (disjoint from SAE-train and causal-eval):
top-100 patches per feature, class purity/entropy/enrichment, spatial statistics, raw-neuron baseline,
optional CLIP labels, decoder-vector clusters, reports for the best features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from context import RunContext
from features.cluster import cluster_decoder_vectors
from features.report import feature_report
from features.stats import build_summary, compute_feature_stats
from utils import get_logger, save_json


def run_discovery(cfg, ctx=None, logger=None):
    log = logger or get_logger()
    ctx = ctx or RunContext(cfg)
    dc = cfg.discovery
    out = ctx.disc_dir; out.mkdir(parents=True, exist_ok=True)
    val = ctx.splits["val"]
    log.info(f"[discover] {cfg.run_tag()} on {len(val)} validation images")
    stats = compute_feature_stats(ctx.sae.encode, ctx.sae.width, ctx.store, ctx.layer, val, ctx.normalizer,
                                  ctx.device, dc.top_n_patches, dc.chunk_images)
    classes = ctx.store.class_names
    summary = build_summary(stats, ctx.store, cfg.model.grid, dc, classes)
    summary = summary.set_index("feature_id", drop=False)
    # ---- raw neuron baseline (positive part of normalised activation) for the SAE-vs-neuron comparison
    nstats = compute_feature_stats(lambda x: torch.relu(x), ctx.store.d_model, ctx.store, ctx.layer, val,
                                   ctx.normalizer, ctx.device, min(dc.top_n_patches, 50), dc.chunk_images)
    nsum = build_summary(nstats, ctx.store, cfg.model.grid, dc, classes)
    nsum.to_csv(out / "neuron_summary.csv", index=False)
    if dc.use_clip:
        try:
            from semantics.clip_labels import label_features_with_clip
            summary = label_features_with_clip(ctx, summary, stats, dc, log)
        except Exception as e:
            log.warning(f"[discover] CLIP labelling skipped: {e}")
    summary.to_csv(out / "feature_summary.csv", index=False)
    np.savez_compressed(out / "features.npz", top_vals=stats["top_vals"], top_idx=stats["top_idx"],
                        class_mass=stats["class_mass"].astype(np.float32), spatial=stats["spatial"].astype(np.float32),
                        positions=stats["positions"], freq=stats["freq"])
    ctx._disc, ctx._summary = None, None
    # ---- searchable feature database: top-N patches per feature
    P = ctx.store.P
    rows = []
    for j in np.where(~summary["dead"].values)[0]:
        for r, gi in enumerate(stats["top_idx"][:, j]):
            if gi < 0:
                break
            pos = int(gi // P)
            rows.append((j, r, int(ctx.store.image_ids[pos]), pos, classes[ctx.store.labels[pos]], int(gi % P),
                         float(stats["top_vals"][r, j])))
    pd.DataFrame(rows, columns=["feature_id", "rank", "image_id", "position", "class", "patch_id", "activation"]
                 ).to_csv(out / "feature_top_patches.csv.gz", index=False, compression="gzip")
    # ---- clustering
    try:
        res = cluster_decoder_vectors(ctx.sae, summary, dc, cfg.seed, logger=log)
        if res is not None:
            coords, cdesc = res
            coords.to_csv(out / "clusters.csv", index=False); cdesc.to_csv(out / "cluster_descriptions.csv", index=False)
            from visualization.plots import plot_clusters, plot_feature_distributions, plot_sae_vs_neuron
            plot_clusters(coords, cdesc, ctx.run_dir / "plots" / "feature_clusters.png")
    except Exception as e:
        log.warning(f"[discover] clustering skipped: {e}")
    try:
        from visualization.plots import plot_feature_distributions, plot_sae_vs_neuron
        plot_feature_distributions(summary, ctx.run_dir / "plots" / "feature_quality.png")
        plot_sae_vs_neuron(summary, nsum, ctx.run_dir / "plots" / "sae_vs_neurons.png")
    except Exception as e:
        log.warning(f"[discover] plots skipped: {e}")
    # ---- reports for best features
    best = summary[~summary["dead"]].sort_values("interp_score", ascending=False).index[: dc.n_feature_reports]
    for fid in best:
        try:
            feature_report(ctx, int(fid))
        except Exception as e:
            log.warning(f"[discover] report for feature {fid} failed: {e}")
    alive = summary[~summary["dead"]]
    save_json({"n_alive": int(len(alive)), "width": int(ctx.sae.width),
               "median_purity": float(alive["purity"].median()),
               "median_class_entropy_norm": float(alive["class_entropy_norm"].median()),
               "median_spatial_conc10": float(alive["spatial_conc10"].median()),
               "median_top_patch_purity": float(alive["top_patch_purity"].median()),
               "median_interp_score": float(alive["interp_score"].median()),
               "pct_highly_specific": float(100 * (alive["specificity"] == "highly_specific").mean()),
               "pct_moderately_specific": float(100 * (alive["specificity"] == "moderately_specific").mean()),
               "pct_polysemantic": float(100 * (alive["specificity"] == "polysemantic").mean()),
               "neuron_median_purity": float(nsum["purity"].median()),
               "neuron_median_class_entropy_norm": float(nsum["class_entropy_norm"].median()),
               "neuron_median_spatial_conc10": float(nsum["spatial_conc10"].median()),
               "neuron_median_interp_score": float(nsum["interp_score"].median()),
               "clip_mean_score": float(summary["clip_score"].mean()) if "clip_score" in summary else None},
              out / "discovery_summary.json")
    log.info(f"[discover] done; alive features={len(alive)}/{ctx.sae.width}")
    return out
