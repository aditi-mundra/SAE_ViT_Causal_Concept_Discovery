"""features/cluster.py [CORE + OPTIONAL UMAP]: cluster SAE decoder vectors (KMeans) and embed with PCA / t-SNE / UMAP."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def cluster_decoder_vectors(sae, summary: pd.DataFrame, dcfg, seed: int = 0, max_features: int = 4000, logger=None):
    alive = summary.index[(~summary["dead"]).values].to_numpy(dtype=np.int64, copy=True)
    if len(alive) < 15:
        if logger: logger.warning("[cluster] too few alive features to cluster")
        return None
    if len(alive) > max_features:
        alive = np.random.RandomState(seed).choice(alive, max_features, replace=False)
    dec = sae.W_dec[alive].detach().cpu().numpy().copy()
    dec = dec / np.maximum(np.linalg.norm(dec, axis=1, keepdims=True), 1e-8)
    Z = PCA(n_components=min(50, dec.shape[0] - 1, dec.shape[1]), random_state=seed).fit_transform(dec)
    k = min(dcfg.n_clusters, len(alive) // 3)
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    out = pd.DataFrame({"feature_id": alive, "cluster": lab, "pca1": Z[:, 0], "pca2": Z[:, 1]})
    perp = float(min(30, max(5, (len(alive) - 1) / 3)))
    ts = TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(Z)
    out["tsne1"], out["tsne2"] = ts[:, 0], ts[:, 1]
    if dcfg.use_umap:
        try:
            import umap
            u = umap.UMAP(n_components=2, random_state=seed).fit_transform(Z)
            out["umap1"], out["umap2"] = u[:, 0], u[:, 1]
        except Exception as e:
            if logger: logger.warning(f"[cluster] UMAP skipped: {e}")
    # interpret clusters AFTER clustering (no manual forcing)
    rows = []
    for c in range(k):
        fs = out.loc[out.cluster == c, "feature_id"].values
        sub = summary.loc[fs]
        tc = sub["top_class_name"].value_counts()
        row = {"cluster": c, "n_features": len(fs), "dominant_class": tc.index[0] if len(tc) else "",
               "dominant_class_share": float(tc.iloc[0] / len(fs)) if len(tc) else np.nan,
               "median_purity": float(sub["purity"].median()), "median_spatial_conc10": float(sub["spatial_conc10"].median())}
        if "clip_concept" in sub:
            cc = sub["clip_concept"].replace("", np.nan).dropna().value_counts()
            row["dominant_clip_concept"] = cc.index[0] if len(cc) else ""
        rows.append(row)
    return out, pd.DataFrame(rows)
