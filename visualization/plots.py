"""visualization/plots.py [CORE]: publication-style figures (matplotlib, Agg backend)."""
from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features.heatmaps import normalize_map, overlay

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.25})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return str(path)


def plot_training_curves(df, path):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    for a, (c, t) in zip(ax, [("val_mse", "val MSE"), ("val_explained_variance", "val explained variance"),
                              ("val_l0", "val L0"), ("val_dead_pct", "val dead features (%)")]):
        a.plot(df["epoch"], df[c], marker="o"); a.set_title(t); a.set_xlabel("epoch")
    return _save(fig, path)


def plot_feature_distributions(summ, path):
    s = summ[~summ["dead"]]
    fig, ax = plt.subplots(1, 5, figsize=(17, 3))
    ax[0].hist(np.log10(summ["freq"].clip(lower=1e-8)), bins=40); ax[0].set_title("log10 feature frequency")
    ax[1].bar(["alive", "dead"], [len(s), int(summ["dead"].sum())]); ax[1].set_title("dead features")
    ax[2].hist(s["purity"].dropna(), bins=30); ax[2].set_title("class purity")
    ax[3].hist(s["spatial_conc10"].dropna(), bins=30); ax[3].set_title("spatial concentration (top-10% share)")
    ax[4].scatter(s["class_entropy_norm"], s["spatial_conc10"], s=6, alpha=0.5)
    ax[4].set_xlabel("normalised class entropy"); ax[4].set_ylabel("SCS"); ax[4].set_title("monosemanticity scatter")
    return _save(fig, path)


def plot_sae_vs_neuron(summ, nsum, path):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    label_kw = "tick_labels" if "tick_labels" in inspect.signature(ax[0].boxplot).parameters else "labels"
    for a, (c, t) in zip(ax, [("purity", "class purity"), ("class_entropy_norm", "norm. class entropy"),
                              ("spatial_conc10", "spatial concentration"), ("interp_score", "interp. score")]):
        a.boxplot([summ.loc[~summ["dead"], c].dropna(), nsum.loc[~nsum["dead"], c].dropna()],
                  showfliers=False, **{label_kw: ["SAE features", "raw neurons"]}); a.set_title(t)
    return _save(fig, path)


def plot_clusters(coords, cdesc, path):
    xcol = "umap1" if "umap1" in coords else "tsne1"
    ycol = xcol.replace("1", "2")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(coords[xcol], coords[ycol], c=coords["cluster"], cmap="tab10", s=8)
    for _, r in cdesc.iterrows():
        m = coords[coords.cluster == r["cluster"]]
        ax.annotate(f"{int(r['cluster'])}:{r['dominant_class']}", (m[xcol].mean(), m[ycol].mean()), fontsize=7)
    ax.set_title(f"SAE decoder-vector clusters ({xcol[:-1]})"); ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    return _save(fig, path)


def plot_dose_response(dose, cd, path, max_panels=8):
    keys = list(dose)[:max_panels]
    n = len(keys); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3 * rows), squeeze=False)
    for ax, k in zip(axes.ravel(), keys):
        d = dose[k]; m = d["multipliers"]
        ax.errorbar(m, d["mean_logit"], yerr=[np.array(d["mean_logit"]) - np.array(d["ci_lo"]),
                    np.array(d["ci_hi"]) - np.array(d["mean_logit"])], marker="o", label="feature")
        if d.get("random_mean_curve"):
            ax.plot(m, d["random_mean_curve"], "--", color="gray", label="random mean")
        ax.set_title(f"feature {k} (rho={d['spearman_rho']:.2f})", fontsize=8); ax.set_xlabel("multiplier"); ax.set_ylabel("target logit")
    axes.ravel()[0].legend(fontsize=6)
    for ax in axes.ravel()[n:]: ax.axis("off")
    return _save(fig, path)


def plot_random_controls(rc, cd, path, max_panels=8):
    keys = [k for k in rc if k.endswith("_ablation")][:max_panels]
    if not keys: return None
    n = len(keys); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3 * rows), squeeze=False)
    for ax, k in zip(axes.ravel(), keys):
        d = rc[k]; ax.boxplot(d["own_means"], showfliers=True, widths=0.5)
        ax.scatter([1], [d["cand_mean"]], color="red", zorder=5, label="candidate")
        ax.set_title(f"feature {k.split('_')[0]} p={d['p_empirical']:.3f}", fontsize=8)
        ax.set_xticks([]); ax.set_ylabel("mean logit drop")
    axes.ravel()[0].legend(fontsize=6)
    for ax in axes.ravel()[n:]: ax.axis("off")
    return _save(fig, path)


def plot_causal_distribution(rows, path):
    r = rows[rows["condition"] == "ablation_x0"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(r["logit_effect"].dropna(), bins=40); ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("target-logit drop under ablation (per feature x image)"); ax.set_title("causal effect distribution")
    return _save(fig, path)


def plot_interaction_matrix(it, path):
    d = it[it["category"] != "insufficient_data"] if len(it) else it
    if not len(d) or "interaction" not in d: return None
    f = sorted(set(it["feature_a"]) | set(it["feature_b"]))
    M = pd.DataFrame(np.nan, index=f, columns=f)
    for _, r in d.iterrows():
        M.loc[r["feature_a"], r["feature_b"]] = M.loc[r["feature_b"], r["feature_a"]] = r["interaction"]
    fig, ax = plt.subplots(figsize=(5, 4.3))
    v = np.nanmax(np.abs(M.values)) if np.isfinite(M.values).any() else 1
    im = ax.imshow(M.values.astype(float), cmap="coolwarm", vmin=-v, vmax=v)
    ax.set_xticks(range(len(f))); ax.set_xticklabels(f, rotation=90, fontsize=7); ax.set_yticks(range(len(f))); ax.set_yticklabels(f, fontsize=7)
    plt.colorbar(im, label="interaction = E_AB - E_A - E_B"); ax.set_title("feature interaction (logit drop)"); ax.grid(False)
    return _save(fig, path)


def plot_steering(st, path):
    s = st[st["type"] == "single"]
    if not len(s): return None
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for fid, g in s.groupby("feature_id"):
        g = g.sort_values("alpha"); ax.plot(g["alpha"], g["mean_delta_logit"], marker="o", label=str(fid))
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("steering alpha"); ax.set_ylabel("mean change in target logit")
    ax.legend(fontsize=6, ncol=2, title="feature"); ax.set_title("feature steering")
    return _save(fig, path)


def plot_local_global(lg, path):
    p = lg.pivot_table(index="feature_id", columns="scope", values="mean_logit_effect")
    cols = ["global"] + [c for c in p.columns if c != "global"]
    ax = p[cols].plot.bar(figsize=(8, 3.5)); ax.set_ylabel("mean logit drop"); ax.set_title("global vs local ablation")
    ax.legend(fontsize=6); return _save(ax.figure, path)


def plot_ablation_amplification(cd, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(cd))
    ax[0].bar(x, cd["abl_mean"], yerr=[cd["abl_mean"] - cd["abl_ci_lo"], cd["abl_ci_hi"] - cd["abl_mean"]], capsize=2)
    if "abl_rand_rand_mean" in cd: ax[0].scatter(x, cd["abl_rand_rand_mean"], color="red", zorder=5, label="random mean")
    ax[0].set_xticks(x); ax[0].set_xticklabels(cd["feature_id"], rotation=90, fontsize=7); ax[0].set_title("ablation: target-logit drop"); ax[0].legend(fontsize=6)
    ax[1].bar(x, cd["amp_mean"], yerr=[cd["amp_mean"] - cd["amp_ci_lo"], cd["amp_ci_hi"] - cd["amp_mean"]], capsize=2, color="C2")
    if "amp_rand_rand_mean" in cd: ax[1].scatter(x, cd["amp_rand_rand_mean"], color="red", zorder=5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(cd["feature_id"], rotation=90, fontsize=7); ax[1].set_title("amplification: target-logit change")
    return _save(fig, path)


def plot_gradcam_comparison(pil, cam, fmap, emap, path, title=""):
    S = 224
    img = np.asarray(pil.convert("RGB").resize((S, S)))
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    ax[0].imshow(img); ax[0].set_title("original")
    ax[1].imshow(overlay(img, normalize_map(cam))); ax[1].set_title("Grad-CAM\n(where is the model looking?)")
    ax[2].imshow(overlay(img, normalize_map(fmap))); ax[2].set_title("SAE feature map\n(what is active?)")
    ax[3].imshow(overlay(img, normalize_map(np.clip(emap, 0, None)))); ax[3].set_title("ablation effect map\n(does it matter?)")
    for a in ax: a.axis("off")
    fig.suptitle(title, fontsize=9)
    return _save(fig, path)


def _line(df, xcol, cols, titles, path, xlabel):
    fig, ax = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 3))
    for a, c, t in zip(np.atleast_1d(ax), cols, titles):
        if c in df: a.plot(df[xcol], df[c], marker="o")
        a.set_title(t, fontsize=8); a.set_xlabel(xlabel)
    return _save(fig, path)


_C = ["explained_variance", "l0", "median_purity", "median_spatial_conc10", "median_interp_score", "mean_ablation_logit_effect"]
_T = ["explained variance", "L0", "median purity", "median SCS", "median interp. score", "mean ablation logit drop"]


def plot_layer_comparison(df, path): return _line(df.sort_values("layer"), "layer", _C, _T, path, "ViT block")
def plot_width_comparison(df, path): return _line(df.sort_values("width"), "width", _C + ["dead_pct_eval"], _T + ["dead %"], path, "SAE width")


def plot_sae_type_comparison(df, path):
    cols = ["mse", "explained_variance", "l0", "dead_pct_eval", "median_purity", "mean_ablation_logit_effect"]
    fig, ax = plt.subplots(1, len(cols), figsize=(3 * len(cols), 3))
    lab = df["sae_type"] + np.where(df["k"].notna(), "-k" + df["k"].fillna(0).astype(int).astype(str), "")
    for a, c in zip(ax, cols):
        if c in df: a.bar(lab, df[c])
        a.set_title(c, fontsize=8); a.tick_params(axis="x", rotation=45, labelsize=7)
    return _save(fig, path)
