"""features/report.py [CORE]: per-feature report (top images, histogram, class distribution, heatmaps)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from features.heatmaps import normalize_map, patch_map, save_feature_heatmaps


def feature_report(ctx, fid: int, out_dir=None, n_top: int = 12) -> dict:
    disc, summ = ctx.disc, ctx.summary()
    if disc is None:
        raise FileNotFoundError("run discover.py first")
    out_dir = Path(out_dir or ctx.run_dir / "plots" / "features"); out_dir.mkdir(parents=True, exist_ok=True)
    g, ps, P = ctx.cfg.model.grid, ctx.cfg.model.patch_size, ctx.store.P
    row = summ.loc[fid]
    idx = disc["top_idx"][:, fid]; vals = disc["top_vals"][:, fid]
    ok = idx >= 0
    idx, vals = idx[ok], vals[ok]
    S = g * ps
    fig = plt.figure(figsize=(14, 9), dpi=130)
    gs = fig.add_gridspec(3, 6)
    for k in range(min(n_top, len(idx))):
        ax = fig.add_subplot(gs[k // 6, k % 6])
        pos, patch = int(idx[k] // P), int(idx[k] % P)
        ax.imshow(ctx.image_pil(pos).convert("RGB").resize((S, S)))
        r, c = divmod(patch, g)
        ax.add_patch(plt.Rectangle((c * ps, r * ps), ps, ps, fill=False, ec="red", lw=1.5))
        ax.set_title(f"{ctx.store.class_names[ctx.store.labels[pos]]}\n{vals[k]:.2f}", fontsize=6); ax.axis("off")
    val_pos = ctx.splits["val"]
    acts = ctx.feature_activations(fid, val_pos)
    ax = fig.add_subplot(gs[2, 0:2])
    nz = acts[acts > 0]
    if nz.size: ax.hist(nz, bins=40, color="C0")
    ax.set_title("non-zero activation histogram", fontsize=8); ax.set_yscale("log")
    ax = fig.add_subplot(gs[2, 2:4])
    mass = disc["class_mass"][:, fid]
    pc = mass / max(mass.sum(), 1e-12)
    top = np.argsort(-pc)[:8]
    ax.barh([ctx.store.class_names[i] for i in top][::-1], pc[top][::-1], color="C1")
    ax.set_title("P(class | feature)", fontsize=8); ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[2, 4:6])
    sp = patch_map(disc["spatial"][:, fid], g)
    im = ax.imshow(normalize_map(sp), cmap="viridis"); ax.axis("off"); ax.set_title("mean spatial map", fontsize=8)
    title = f"Feature {fid} | layer {ctx.layer} | purity {row['purity']:.2f} | SCS {row['spatial_conc10']:.2f}"
    if "clip_concept" in row and isinstance(row["clip_concept"], str) and row["clip_concept"]:
        title += f" | CLIP: {row['clip_concept']}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = out_dir / f"feature_{fid}_report.png"
    fig.savefig(path); plt.close(fig)
    hm = {}
    if len(idx):
        pos = int(idx[0] // P)
        a = ctx.feature_activations(fid, [pos])[0]
        hm = save_feature_heatmaps(ctx.image_pil(pos), a, g, ps, out_dir / f"feature_{fid}_heatmaps", "top1",
                                   ctx.cfg.discovery.heatmap_threshold)
    return {"report": str(path), "heatmaps": hm}
