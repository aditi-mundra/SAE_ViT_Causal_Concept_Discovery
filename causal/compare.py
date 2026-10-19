"""causal/compare.py [CORE]: cross-run comparison tables/figures (layers, SAE width, SAE type, Top-K)."""
from __future__ import annotations

import pandas as pd

from utils import load_json

COLS = ["run_tag", "layer", "sae_type", "width", "k", "mse", "explained_variance", "l0", "dead_pct_eval",
        "median_purity", "median_class_entropy_norm", "median_spatial_conc10", "median_top_patch_purity",
        "median_interp_score", "pct_highly_specific", "pct_polysemantic", "neuron_median_interp_score",
        "clip_mean_score", "n_candidates", "mean_ablation_logit_effect", "median_ablation_logit_effect",
        "mean_random_effect", "mean_amp_delta_logit"]


def build_comparison(cfg, save: bool = True) -> pd.DataFrame:
    rows = []
    for d in sorted(cfg.exp_dir().iterdir()):
        p = d / "layer_summary.json"
        if d.is_dir() and p.exists():
            rows.append(load_json(p))
    if not rows:
        raise FileNotFoundError("no completed runs (layer_summary.json) in " + str(cfg.exp_dir()))
    df = pd.DataFrame(rows)
    df = df.reindex(columns=[c for c in COLS if c in df.columns]).sort_values(["layer", "sae_type", "width"])
    df["causal_minus_random"] = df["mean_ablation_logit_effect"] - df["mean_random_effect"]
    if save:
        df.to_csv(cfg.exp_dir() / "comparison.csv", index=False)
        try:
            from visualization import plots as V
            pl = cfg.exp_dir() / "plots"; pl.mkdir(exist_ok=True)
            base_type, base_w, base_k = cfg.sae.sae_type, cfg.sae.width, cfg.sae.k
            lay = df[(df.sae_type == base_type) & (df.width == base_w)]
            if base_type == "topk":
                lay = lay[lay.k == base_k]
            if len(lay) > 1: V.plot_layer_comparison(lay, pl / "layer_comparison.png")
            wid = df[(df.layer == cfg.layer) & (df.sae_type == base_type)]
            if base_type == "topk": wid = wid[wid.k == base_k]
            if len(wid) > 1: V.plot_width_comparison(wid, pl / "sae_width_comparison.png")
            typ = df[(df.layer == cfg.layer) & (df.width == base_w)]
            if len(typ) > 1: V.plot_sae_type_comparison(typ, pl / "sae_type_comparison.png")
        except Exception as e:
            print("[compare] figures skipped:", e)
    return df
