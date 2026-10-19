"""
causal/hypotheses.py [CORE]
Data-driven verdicts for H1-H6 with the exact decision rules stated below. Nothing is assumed: if an input
is missing the verdict is 'Not tested'.
 H1  SAE features more interpretable than neurons: Mann-Whitney one-sided on class_entropy_norm (lower), purity,
     spatial_conc10, interp_score (higher). Supported if >=3/4 metrics p<0.05; Partially if 1-2; else Not supported.
 H2  Later layers more semantic: Spearman(layer, median_interp_score) over runs with same SAE settings (>=3 layers).
     Supported if rho>=0.8 and last>first; Partially if rho>0; else Not supported.
 H3  Ablation > random: a candidate is 'separated' if Holm-adjusted paired Wilcoxon p<0.05, effect>random mean, and
     (empirical p<0.05 or fewer than 19 random controls). Supported if >=70% of candidates, Partially if >=1, else Not.
 H4  Amplification increases target logit: same rule on delta_logit at the primary alpha.
 H5  Local stronger than global: per candidate, any local percentile with mean paired diff>0 and Holm p<0.05.
     Supported if majority; Partially if >=1 (or only per-patch effect larger); else Not supported.
 H6  Redundant representation: pairs with category 'redundant' (bootstrap CI of interaction < 0).
     Supported if >=1 such pair; Partially if only decoder-cosine>0.5 pairs; else Not supported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from causal.stats import holm_correction
from utils import save_json


def _verdict(frac, n, majority=0.7):
    if n == 0: return "Not tested"
    return "Supported" if frac >= majority else ("Partially supported" if frac > 0 else "Not supported")


def evaluate_hypotheses(cfg, comp_df: pd.DataFrame):
    rd = cfg.run_dir()
    ev, out = {}, {}
    # H1
    try:
        s = pd.read_csv(rd / "discovery" / "feature_summary.csv"); n = pd.read_csv(rd / "discovery" / "neuron_summary.csv")
        s, n = s[~s["dead"]], n[~n["dead"]]
        wins, det = 0, {}
        for col, alt in [("class_entropy_norm", "less"), ("purity", "greater"), ("spatial_conc10", "greater"), ("interp_score", "greater")]:
            a, b = s[col].dropna(), n[col].dropna()
            p = mannwhitneyu(a, b, alternative=alt).pvalue if len(a) > 3 and len(b) > 3 else np.nan
            det[col] = {"sae_median": float(a.median()), "neuron_median": float(b.median()), "p": float(p)}
            wins += int(p < 0.05)
        out["H1"] = {"verdict": "Supported" if wins >= 3 else ("Partially supported" if wins >= 1 else "Not supported"),
                     "metrics_significantly_better": wins, "details": det,
                     "caveat": "neurons use the positive part of the normalised activation; metrics are class/spatial proxies, not human ratings"}
    except Exception as e:
        out["H1"] = {"verdict": "Not tested", "reason": str(e)}
    # H2
    try:
        b = comp_df[(comp_df.sae_type == cfg.sae.sae_type) & (comp_df.width == cfg.sae.width)]
        if cfg.sae.sae_type == "topk": b = b[b.k == cfg.sae.k]
        b = b.sort_values("layer")
        if len(b) >= 3:
            rho = float(spearmanr(b["layer"], b["median_interp_score"])[0])
            v = "Supported" if (rho >= 0.8 and b["median_interp_score"].iloc[-1] > b["median_interp_score"].iloc[0]) else ("Partially supported" if rho > 0 else "Not supported")
            out["H2"] = {"verdict": v, "spearman_layer_vs_interp": rho, "layers": b["layer"].tolist(),
                         "median_interp_score": b["median_interp_score"].tolist(), "median_purity": b["median_purity"].tolist()}
        else:
            out["H2"] = {"verdict": "Not tested", "reason": "needs >=3 layers trained with identical SAE settings"}
    except Exception as e:
        out["H2"] = {"verdict": "Not tested", "reason": str(e)}
    # H3 / H4 / H5
    try:
        cd = pd.read_csv(rd / "causal" / "candidate_causal_summary.csv")
        def sep(prefix, pholm, mean_col, rand_col):
            if pholm not in cd: return None
            R = cd[f"{prefix}_rand_n_random"]
            ok = (cd[pholm] < 0.05) & (cd[mean_col] > cd[rand_col]) & ((cd[f"{prefix}_rand_p_empirical"] < 0.05) | (R < 19))
            return ok
        s3 = sep("abl", "abl_p_holm_paired", "abl_rand_cand_mean", "abl_rand_rand_mean")
        if s3 is None: out["H3"] = {"verdict": "Not tested", "reason": "no random controls"}
        else: out["H3"] = {"verdict": _verdict(s3.mean(), len(s3)), "n_separated": int(s3.sum()), "n_candidates": int(len(s3)),
                           "features_separated": cd.loc[s3, "feature_id"].tolist()}
        s4 = sep("amp", "amp_p_holm_paired", "amp_rand_cand_mean", "amp_rand_rand_mean")
        if s4 is None: out["H4"] = {"verdict": "Not tested", "reason": "no random controls"}
        else: out["H4"] = {"verdict": _verdict(s4.mean(), len(s4)), "n_separated": int(s4.sum()), "n_candidates": int(len(s4)),
                           "features_separated": cd.loc[s4, "feature_id"].tolist(),
                           "mean_delta_logit_primary_alpha": float(cd["amp_mean"].mean())}
        lg = pd.read_csv(rd / "causal" / "local_vs_global.csv")
        loc = lg[lg.scope != "global"].copy()
        loc["p_holm"] = holm_correction(loc["p_vs_global"].fillna(1.0).values) if len(loc) else []
        win_total = loc[(loc.mean_diff_vs_global > 0) & (loc.p_holm < 0.05)].feature_id.unique()
        g = lg[lg.scope == "global"].set_index("feature_id")["effect_per_patch"]
        win_pp = loc[loc.apply(lambda r: r["effect_per_patch"] > g.get(r["feature_id"], np.inf), axis=1)].feature_id.unique()
        nf = lg.feature_id.nunique()
        if len(win_total):
            v = "Supported" if len(win_total) / nf > 0.5 else "Partially supported"
        elif len(win_pp):
            v = "Partially supported (per-modified-patch effect only)"
        else:
            v = "Not supported"
        out["H5"] = {"verdict": v, "features_local_total_stronger": [int(x) for x in win_total],
                     "features_local_per_patch_stronger": [int(x) for x in win_pp], "n_features": int(nf)}
    except Exception as e:
        for h in ("H3", "H4", "H5"):
            out.setdefault(h, {"verdict": "Not tested", "reason": str(e)})
    # H6
    try:
        it = pd.read_csv(rd / "causal" / "interactions.csv")
        red = it[it["category"] == "redundant"]
        cosy = it[(it["decoder_cosine"] > 0.5)] if "decoder_cosine" in it else it.iloc[0:0]
        v = "Supported" if len(red) else ("Partially supported" if len(cosy) else "Not supported")
        out["H6"] = {"verdict": v, "n_pairs": int(len(it)), "n_redundant_pairs": int(len(red)),
                     "n_synergistic_pairs": int((it["category"] == "synergistic").sum()),
                     "n_high_decoder_cosine_pairs": int(len(cosy)),
                     "n_insufficient_pairs": int((it["category"] == "insufficient_data").sum())}
    except Exception as e:
        out["H6"] = {"verdict": "Not tested", "reason": str(e)}
    save_json(out, cfg.exp_dir() / "hypotheses.json")
    md = ["# Hypothesis verdicts (computed from experiment outputs)\n"]
    for h, d in out.items():
        md.append(f"## {h}: **{d['verdict']}**\n")
        for k, v in d.items():
            if k != "verdict": md.append(f"- {k}: {v}")
        md.append("")
    (cfg.exp_dir() / "hypotheses.md").write_text("\n".join(md))
    return out
