"""features/heatmaps.py [CORE]: 14x14 patch maps -> raw / normalised / overlay / mask / bounding box."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


def patch_map(act_patches, grid: int) -> np.ndarray:
    a = np.asarray(act_patches, dtype=np.float64)
    if a.size != grid * grid:
        raise ValueError(f"expected {grid*grid} patch values, got {a.size}")
    return a.reshape(grid, grid)


def normalize_map(m: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(m)), float(np.max(m))
    return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m, dtype=np.float64)


def upsample(m: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(m.astype(np.float32), mode="F").resize((size, size), Image.BILINEAR))


def overlay(img: np.ndarray, norm_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h = upsample(norm_map, img.shape[0])
    rgb = plt.get_cmap("jet")(np.clip(h, 0, 1))[..., :3]
    return np.clip((1 - alpha) * img / 255.0 + alpha * rgb, 0, 1)


def threshold_mask(norm_map: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return norm_map >= thr


def strongest_bbox(norm_map: np.ndarray, thr: float, patch_size: int):
    """(x0, y0, x1, y1) in pixels around the connected above-threshold region containing the peak."""
    peak = np.unravel_index(int(np.argmax(norm_map)), norm_map.shape)
    mask = norm_map >= thr
    if not mask[peak]:
        mask = np.zeros_like(mask); mask[peak] = True
    lab, _ = ndimage.label(mask)
    rows, cols = np.where(lab == lab[peak])
    return (int(cols.min() * patch_size), int(rows.min() * patch_size),
            int((cols.max() + 1) * patch_size), int((rows.max() + 1) * patch_size))


def save_feature_heatmaps(pil_img: Image.Image, act_patches, grid: int, patch_size: int, out_dir, stem: str,
                          thr: float = 0.5) -> dict:
    """Writes the five required heatmap files. Returns {name: path}."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    S = grid * patch_size
    img = np.asarray(pil_img.convert("RGB").resize((S, S), Image.BICUBIC))
    raw = patch_map(act_patches, grid)
    nm = normalize_map(raw)
    paths = {}

    def _fig(name, draw, title):
        fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=150)
        draw(ax); ax.set_title(title, fontsize=8); ax.axis("off")
        p = out_dir / f"{stem}_{name}.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths[name] = str(p)

    def d_raw(ax):
        im = ax.imshow(raw, cmap="viridis"); plt.colorbar(im, ax=ax, fraction=0.046)
    _fig("raw", d_raw, "1. raw activation (14x14)")
    _fig("normalized", lambda ax: ax.imshow(nm, cmap="viridis", vmin=0, vmax=1), "2. normalised [0,1]")
    _fig("overlay", lambda ax: ax.imshow(overlay(img, nm)), "3. overlay")
    _fig("mask", lambda ax: ax.imshow(threshold_mask(nm, thr), cmap="gray"), f"4. mask (>= {thr:.2f} of max)")

    def d_box(ax):
        ax.imshow(img)
        x0, y0, x1, y1 = strongest_bbox(nm, thr, patch_size)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=2))
    _fig("bbox", d_box, "5. strongest-region bounding box")
    return paths
