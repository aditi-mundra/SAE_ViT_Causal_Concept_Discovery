"""sae/metrics.py [CORE]: reconstruction / sparsity metrics evaluated on held-out positions."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def evaluate_sae(sae, normalizer, store, layer, positions, device, img_chunk: int = 32, thr: float = 1e-8) -> dict:
    sae.eval()
    D, W = sae.d_in, sae.width
    sse = sum_x2 = l0 = 0.0
    sum_x = torch.zeros(D, dtype=torch.float64, device=device)
    counts = torch.zeros(W, device=device)
    n = 0
    positions = np.asarray(positions)
    for s in range(0, len(positions), img_chunk):
        x = normalizer(store.read_patches(layer, positions[s:s + img_chunk]).to(device)).reshape(-1, D)
        xh, f = sae(x)
        sse += ((x - xh) ** 2).sum().item()
        sum_x2 += (x ** 2).sum().item()
        sum_x += x.sum(0).double()
        act = f > thr
        l0 += act.sum().item()
        counts += act.sum(0)
        n += x.shape[0]
    mean = sum_x / n
    sst = sum_x2 - n * (mean ** 2).sum().item()
    freq = (counts / n).cpu().numpy()
    return {"mse": sse / (n * D), "explained_variance": 1.0 - sse / max(sst, 1e-12), "l0": l0 / n,
            "dead_pct_eval": 100.0 * float((counts == 0).float().mean().item()),
            "mean_feature_density": float(freq.mean()), "n_vectors": n, "frequency": freq}


@torch.no_grad()
def compute_image_max(sae, normalizer, store, layer, positions, device, img_chunk: int = 32) -> np.ndarray:
    """Max activation of every feature over the patches of each image -> [n_images, width]."""
    out = []
    positions = np.asarray(positions)
    for s in range(0, len(positions), img_chunk):
        pos = positions[s:s + img_chunk]
        x = normalizer(store.read_patches(layer, pos).to(device)).reshape(-1, sae.d_in)
        f = sae.encode(x).reshape(len(pos), store.P, sae.width)
        out.append(f.amax(1).cpu().numpy())
    return np.concatenate(out, 0)
