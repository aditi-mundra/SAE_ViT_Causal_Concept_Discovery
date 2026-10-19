"""features/retrieval.py [CORE]: analyse an uploaded image -> its active SAE features and matching concept examples."""
from __future__ import annotations

import numpy as np
import torch

from models.vit import capture


@torch.no_grad()
def analyze_image(ctx, pil_img, top_n: int = 8) -> dict:
    x = ctx.image_tensor(pil_img).unsqueeze(0).to(ctx.device)
    logits, hid = capture(ctx.model, x, [ctx.layer])
    hidden = hid[ctx.layer]
    patches = hidden[:, ctx.store.num_prefix:]
    D = ctx.store.d_model
    f = ctx.sae.encode(ctx.normalizer(patches).reshape(-1, D))        # [P, W]
    mx, arg = f.max(0)
    order = torch.argsort(mx, descending=True)[:top_n]
    summ = ctx.summary() if (ctx.disc_dir / "feature_summary.csv").exists() else None
    g = ctx.cfg.model.grid
    feats = []
    for fid in order.tolist():
        if mx[fid] <= 0:
            continue
        p = int(arg[fid]); r, c = divmod(p, g)
        d = {"feature_id": fid, "max_activation": float(mx[fid]), "patch": p, "row": r, "col": c,
             "n_active_patches": int((f[:, fid] > 0).sum())}
        if summ is not None and fid in summ.index:
            s = summ.loc[fid]
            d.update({"purity": float(s["purity"]), "top_class": s["top_class_name"],
                      "spatial_conc10": float(s["spatial_conc10"]),
                      "concept": s["clip_concept"] if "clip_concept" in s and isinstance(s["clip_concept"], str) and s["clip_concept"] else s["top_class_name"]})
        feats.append(d)
    return {"logits": logits[0].cpu().numpy(), "hidden": hidden.cpu(), "features": feats,
            "feature_map": f.cpu().numpy()}
