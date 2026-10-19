"""
causal/gradcam.py [CORE]
Grad-CAM for ViTs (target layer = last block's norm1; weights = token-mean of gradients), and the 4-panel comparison:
original | Grad-CAM ("where is the model looking?") | SAE feature map ("what is active?") | ablation map ("what matters?").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from interventions.spatial import patch_ablation_map


def vit_gradcam(model, image: torch.Tensor, grid: int, num_prefix: int = 1, target=None) -> np.ndarray:
    """image [1,3,H,W] on the model device -> [grid, grid] CAM (ReLU, un-normalised)."""
    store = {}

    def hook(m, i, o):
        o.retain_grad()
        store["a"] = o

    h = model.blocks[-1].norm1.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            img = image.clone().requires_grad_(True)
            model.zero_grad()
            logits = model(img)
            t = int(logits.argmax(1).item()) if target is None else int(target)
            logits[0, t].backward()
    finally:
        h.remove()
    a, g = store["a"][0, num_prefix:], store["a"].grad[0, num_prefix:]
    cam = F.relu((a * g.mean(0, keepdim=True)).sum(-1)).reshape(grid, grid)
    return cam.detach().cpu().numpy()


def make_gradcam_figures(ctx, hid, target, active, causal_pos, fids, n: int, out_dir, logger=None) -> pd.DataFrame:
    from visualization.plots import plot_gradcam_comparison
    g, rows = ctx.cfg.model.grid, []
    eng = ctx.engine()
    for fid in fids[:n]:
        cand = np.where(active[:, fid])[0]
        if len(cand) == 0:
            continue
        i = int(cand[0]); pos = int(causal_pos[i])
        pil = ctx.image_pil(pos)
        x = ctx.image_tensor(pil).unsqueeze(0).to(ctx.device)
        cam = vit_gradcam(ctx.model, x, g, ctx.store.num_prefix, target=int(target[i]))
        fmap = ctx.feature_activations(fid, [pos])[0].reshape(g, g)
        _, eff = patch_ablation_map(eng, hid[i:i + 1], fid, int(target[i]), ctx.cfg.intervention.batch_size)
        emap = eff.reshape(g, g)
        plot_gradcam_comparison(pil, cam, fmap, emap, out_dir / f"gradcam_vs_sae_feature{fid}.png",
                                f"feature {fid} | target: {ctx.vit_class_names[int(target[i])]}")
        def sp(a, b):
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                return np.nan
            return float(spearmanr(a.ravel(), b.ravel())[0])
        rows.append({"feature_id": fid, "position": pos, "spearman_gradcam_vs_feature": sp(cam, fmap),
                     "spearman_gradcam_vs_ablation": sp(cam, emap), "spearman_feature_vs_ablation": sp(fmap, emap)})
    return pd.DataFrame(rows)
