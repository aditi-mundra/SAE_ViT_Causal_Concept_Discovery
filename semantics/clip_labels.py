"""
semantics/clip_labels.py [ADVANCED OPTIONAL]
CLIP-based concept labels for SAE features. Each top-activating patch is cropped (3x3-patch neighbourhood),
embedded with CLIP, and compared with text prompts. CLIP is used for INTERPRETATION only, never as causal evidence.
clip_score = mean cosine similarity of the best concept; clip_entropy = entropy of the mean softmax over concepts
(normalised; low = one clear concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image

from utils import entropy_bits


def label_features_with_clip(ctx, summary: pd.DataFrame, disc: dict, dcfg, logger=None) -> pd.DataFrame:
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        if logger: logger.warning("[clip] transformers not installed; skipping CLIP labels")
        return summary
    dev = ctx.device
    model = CLIPModel.from_pretrained(dcfg.clip_model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(dcfg.clip_model)
    prompts = dcfg.clip_prompts
    with torch.no_grad():
        t = proc(text=[f"a photo of {p}" for p in prompts], return_tensors="pt", padding=True).to(dev)
        txt = model.get_text_features(**t)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    cand = summary[(~summary["dead"]) & summary["interp_score"].notna()].sort_values("interp_score", ascending=False)
    fids = cand.index[: dcfg.clip_max_features].tolist()
    g, ps, P = ctx.cfg.model.grid, ctx.cfg.model.patch_size, ctx.store.P
    S = g * ps
    cols = {"clip_concept": "", "clip_score": np.nan, "clip_entropy": np.nan, "clip_top3": ""}
    for c, v in cols.items():
        summary[c] = v
    cache = {}
    for fid in fids:
        idx = disc["top_idx"][: dcfg.clip_top_m, fid]
        idx = idx[idx >= 0]
        if len(idx) == 0:
            continue
        crops = []
        for gi in idx:
            pos, patch = int(gi // P), int(gi % P)
            if pos not in cache:
                if len(cache) > 256: cache.clear()
                cache[pos] = ctx.image_pil(pos).convert("RGB").resize((S, S), Image.BICUBIC)
            r, c = divmod(patch, g)
            cy, cx, h = r * ps + ps // 2, c * ps + ps // 2, int(1.5 * ps)
            crops.append(cache[pos].crop((max(0, cx - h), max(0, cy - h), min(S, cx + h), min(S, cy + h))))
        with torch.no_grad():
            inp = proc(images=crops, return_tensors="pt").to(dev)
            emb = model.get_image_features(**inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            sim = (emb @ txt.T).float()
            probs = torch.softmax(100.0 * sim, -1).mean(0).cpu().numpy()
        mcos = sim.mean(0).cpu().numpy()
        top = int(np.argmax(mcos))
        summary.at[fid, "clip_concept"] = prompts[top]
        summary.at[fid, "clip_score"] = float(mcos[top])
        summary.at[fid, "clip_entropy"] = float(entropy_bits(probs) / np.log2(len(prompts)))
        summary.at[fid, "clip_top3"] = ",".join(prompts[i] for i in np.argsort(-mcos)[:3])
    return summary
