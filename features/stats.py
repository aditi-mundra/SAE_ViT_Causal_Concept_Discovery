"""
features/stats.py [CORE]
Streaming per-feature statistics over a set of image positions, then a summary table.

Definitions (activation mass = sum of feature activations):
  P(c|f)            = mass of f on images of class c / total mass of f
  purity(f)         = max_c P(c|f)
  H(C|f)            = -sum_c P(c|f) log2 P(c|f);  normalised by log2(#classes)
  enrichment(c,f)   = P(c|f) / P(c)
  Spatial Concentration Score (SCS) = share of the feature's mean spatial activation map that lies in its
                      top 10% patches (chance = 0.10, max = 1.0)
  spatial entropy   = H(m)/log2(P), m = normalised mean spatial map
  act_entropy_norm  = entropy of the distribution of the feature's mass over images / log2(n_images)
  interp_score      = mean(1 - H_norm, top-patch class purity, rescaled SCS)  [uses NO causal information and NO CLIP]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from utils import entropy_bits


@torch.no_grad()
def compute_feature_stats(encode_fn, width, store, layer, positions, normalizer, device,
                          top_n: int = 100, chunk_images: int = 16) -> dict:
    P, D = store.P, store.d_model
    positions = np.asarray(positions, dtype=np.int64)
    n, C = len(positions), len(store.class_names)
    labels_t = torch.from_numpy(store.labels[positions]).long().to(device)
    pos_t = torch.from_numpy(positions).to(device)
    top_vals = torch.zeros(top_n, width, device=device)
    top_idx = torch.full((top_n, width), -1, dtype=torch.long, device=device)
    count = torch.zeros(width, device=device)
    sum_act = torch.zeros(width, device=device, dtype=torch.float64)
    class_mass = torch.zeros(C, width, device=device)
    class_active = torch.zeros(C, width, device=device)
    spatial = torch.zeros(P, width, device=device)
    img_mass = torch.zeros(n, width, device=device)
    img_active = torch.zeros(width, device=device)
    for s in range(0, n, chunk_images):
        pos = positions[s:s + chunk_images]
        m = len(pos)
        x = normalizer(store.read_patches(layer, pos).to(device)).reshape(-1, D)
        f = encode_fn(x)
        f3 = f.reshape(m, P, width)
        count += (f > 0).sum(0)
        sum_act += f.sum(0).double()
        imass = f3.sum(1)
        img_mass[s:s + m] = imass
        lab = labels_t[s:s + m]
        class_mass.index_add_(0, lab, imass)
        act_img = (f3.amax(1) > 0).float()
        img_active += act_img.sum(0)
        class_active.index_add_(0, lab, act_img)
        spatial += f3.sum(0)
        kk = min(top_n, f.shape[0])
        v, i = torch.topk(f, kk, dim=0)
        gi = pos_t[s:s + m][i // P] * P + (i % P)
        cv = torch.cat([top_vals, v], 0)
        ci = torch.cat([top_idx, gi], 0)
        tv, ti = torch.topk(cv, top_n, dim=0)
        top_vals, top_idx = tv, torch.gather(ci, 0, ti)
    top_idx = torch.where(top_vals > 0, top_idx, torch.full_like(top_idx, -1))
    cpu = lambda t: t.cpu().numpy()
    return {"positions": positions, "top_vals": cpu(top_vals), "top_idx": cpu(top_idx),
            "freq": cpu(count) / (n * P), "mean_act": cpu(sum_act).astype(np.float32) / (n * P),
            "class_mass": cpu(class_mass), "class_active": cpu(class_active), "spatial": cpu(spatial),
            "img_mass": cpu(img_mass), "img_active": cpu(img_active)}


def spatial_stats(spatial: np.ndarray, grid: int) -> dict:
    P, W = spatial.shape
    tot = spatial.sum(0)
    m = spatial / np.maximum(tot, 1e-12)
    k = max(1, int(np.ceil(0.1 * P)))
    conc = np.sort(m, axis=0)[-k:].sum(0)
    ent = entropy_bits(m, 0) / np.log2(P)
    rr, cc = np.divmod(np.arange(P), grid)
    com_r, com_c = (m * rr[:, None]).sum(0), (m * cc[:, None]).sum(0)
    var = (m * ((rr[:, None] - com_r) ** 2 + (cc[:, None] - com_c) ** 2)).sum(0)
    bad = tot <= 0
    out = {"spatial_conc10": conc, "spatial_entropy_norm": ent, "com_row": com_r, "com_col": com_c, "spatial_var": var}
    for k_, v in out.items():
        v[bad] = np.nan
    return out


def build_summary(stats: dict, store, grid: int, dcfg, class_names) -> pd.DataFrame:
    P = store.P
    mass, pos = stats["class_mass"], stats["positions"]
    C, W = mass.shape
    n = len(pos)
    tot = mass.sum(0)
    alive_mass = tot > 0
    pc = mass / np.maximum(tot, 1e-12)
    purity = np.where(alive_mass, pc.max(0), np.nan)
    top_class = pc.argmax(0)
    H = entropy_bits(pc, 0)
    H_norm = np.where(alive_mass, H / np.log2(max(C, 2)), np.nan)
    prior = np.bincount(store.labels[pos], minlength=C) / n
    enrich = pc / np.maximum(prior[:, None], 1e-12)
    top_enrich = np.where(alive_mass, enrich.max(0), np.nan)
    pimg = stats["img_mass"] / np.maximum(tot[None, :], 1e-12)
    act_ent = np.where(alive_mass, entropy_bits(pimg, 0) / np.log2(max(n, 2)), np.nan)
    sp = spatial_stats(stats["spatial"], grid)
    top_idx = stats["top_idx"]
    tpp, tun = np.full(W, np.nan), np.zeros(W)
    for j in np.where(alive_mass)[0]:
        valid = top_idx[:, j][top_idx[:, j] >= 0]
        if len(valid) == 0:
            continue
        imgs = valid // P
        tpp[j] = np.bincount(store.labels[imgs], minlength=C).max() / len(valid)
        tun[j] = len(np.unique(imgs)) / len(valid)
    conc_rescaled = np.clip((sp["spatial_conc10"] - 0.1) / 0.9, 0, 1)
    interp = np.nanmean(np.vstack([1 - H_norm, tpp, conc_rescaled]), axis=0)
    lo, hi = dcfg.specific_thresholds
    spec = np.where(np.isnan(H_norm), "dead", np.where(H_norm < lo, "highly_specific",
                    np.where(H_norm < hi, "moderately_specific", "polysemantic")))
    df = pd.DataFrame({
        "feature_id": np.arange(W), "freq": stats["freq"], "mean_act": stats["mean_act"],
        "max_act": stats["top_vals"][0], "n_active_images": stats["img_active"].astype(int),
        "img_coverage": stats["img_active"] / n, "class_coverage": (stats["class_active"] > 0).sum(0) / C,
        "act_entropy_norm": act_ent, "purity": purity, "top_class": top_class,
        "top_class_name": [class_names[i] if a else "" for i, a in zip(top_class, alive_mass)],
        "class_entropy_bits": np.where(alive_mass, H, np.nan), "class_entropy_norm": H_norm,
        "top_enrichment": top_enrich, "top_patch_purity": tpp, "top_unique_images": tun,
        "spatial_conc10": sp["spatial_conc10"], "spatial_entropy_norm": sp["spatial_entropy_norm"],
        "com_row": sp["com_row"], "com_col": sp["com_col"], "spatial_var": sp["spatial_var"],
        "specificity": spec, "dead": stats["freq"] <= 0, "interp_score": interp})
    return df
