"""
activations/extract.py [CORE]
Streaming multi-layer extraction: DataLoader -> frozen ViT -> hooks -> memmap on disk.
Per layer: patches_L{l}.npy [N,P,D] (CLS removed), prefix_L{l}.npy [N,npre,D] (CLS kept separately).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import build_imageset
from models.vit import capture, forward_from_hidden, get_norm_stats, load_vit, model_class_names
from utils import ensure_dir, get_device, get_logger, load_json, make_splits, save_json, set_seed


def run_extraction(cfg, force: bool = False, logger=None):
    log = logger or get_logger()
    store_dir = ensure_dir(cfg.store_dir())
    layers = sorted(set(cfg.extract.layers))
    meta_path = store_dir / "meta.json"
    if meta_path.exists() and not force:
        m = load_json(meta_path)
        if m.get("complete") and m["n"] <= cfg.data.num_images and all(l in m["layers"] for l in layers):
            log.info(f"[extract] store exists at {store_dir} (use --force to redo)")
            return store_dir
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    model = load_vit(cfg.model, device)
    if max(layers) >= len(model.blocks):
        raise ValueError(f"layer {max(layers)} >= number of blocks {len(model.blocks)}")
    mean, std = get_norm_stats(model)
    imgset, raw = build_imageset(cfg, mean, std)
    N, P, D, npre = len(imgset), cfg.model.num_patches, model.embed_dim, model.num_prefix_tokens
    dt = np.float16 if cfg.extract.dtype == "float16" else np.float32
    log.info(f"[extract] N={N} layers={layers} P={P} D={D} dtype={dt.__name__} device={device}")
    mm = {l: np.lib.format.open_memmap(store_dir / f"patches_L{l}.npy", "w+", dt, (N, P, D)) for l in layers}
    pm = {l: np.lib.format.open_memmap(store_dir / f"prefix_L{l}.npy", "w+", dt, (N, npre, D)) for l in layers}
    loader = DataLoader(imgset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    labels = np.zeros(N, dtype=np.int64)
    logits_mm, checked = None, False
    for imgs, lab, pos in tqdm(loader, desc="extract"):
        imgs = imgs.to(device)
        logits, hid = capture(model, imgs, layers)
        if not checked:
            l0 = layers[0]
            diff = (forward_from_hidden(model, hid[l0], l0) - logits).abs().max().item()
            if diff > 1e-3:
                log.warning(f"[extract] forward_from_hidden mismatch {diff:.2e}: intervention results may be unreliable")
            log.info(f"[extract] forward_from_hidden consistency check: max|diff|={diff:.2e}")
            checked = True
        if logits_mm is None:
            logits_mm = np.lib.format.open_memmap(store_dir / "logits.npy", "w+", np.float32, (N, logits.shape[1]))
        p = pos.numpy()
        logits_mm[p] = logits.cpu().numpy()
        labels[p] = lab.numpy()
        for l in layers:
            h = hid[l]
            if h.shape[1] - npre != P:
                raise ValueError(f"patch count {h.shape[1]-npre} != cfg.model.num_patches {P}; fix model.img_size/patch_size")
            a, b = h[:, npre:].cpu().numpy(), h[:, :npre].cpu().numpy()
            if dt == np.float16 and (not np.isfinite(a.astype(np.float16)).all()):
                raise OverflowError("activations overflow float16; set extract.dtype='float32'")
            mm[l][p] = a.astype(dt); pm[l][p] = b.astype(dt)
    for l in layers:
        mm[l].flush(); pm[l].flush()
    logits_mm.flush()
    np.save(store_dir / "labels.npy", labels)
    np.save(store_dir / "image_ids.npy", imgset.ids)
    np.save(store_dir / "preds.npy", np.asarray(logits_mm).argmax(1))
    save_json({"n": N, "num_patches": P, "d_model": D, "layers": layers, "num_prefix_tokens": npre,
               "dtype": dt.__name__, "model": cfg.model.name, "dataset": cfg.data.dataset, "seed": cfg.seed,
               "class_names": list(raw.classes), "model_class_names": model_class_names(model),
               "grid": cfg.model.grid, "patch_size": cfg.model.patch_size, "complete": True}, meta_path)
    tr, va, ca = make_splits(N, cfg.data.split_fractions, cfg.seed)
    split = np.empty(N, dtype=object); split[tr], split[va], split[ca] = "sae_train", "feature_val", "causal_eval"
    cn = raw.classes
    pd.DataFrame({"image_id": imgset.ids, "position": np.arange(N), "class_id": labels,
                  "class_name": [cn[i] for i in labels], "split": split}).to_csv(store_dir / "image_metadata.csv", index=False)
    save_json({"sae_train": tr, "feature_val": va, "causal_eval": ca}, store_dir / "splits.json")
    if cfg.extract.write_patch_metadata:
        g = cfg.model.grid
        rows = [(int(imgset.ids[i]), int(labels[i]), cn[labels[i]], l, p, p // g, p % g)
                for l in layers for i in range(N) for p in range(P)]
        pd.DataFrame(rows, columns=["image_id", "class_id", "class_name", "layer", "patch_id", "row", "column"]
                     ).to_csv(store_dir / "activation_metadata.csv", index=False)
    log.info(f"[extract] done -> {store_dir}")
    return store_dir
