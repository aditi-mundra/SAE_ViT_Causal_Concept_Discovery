"""
sae/train.py [CORE]
Train one SAE on the SAE-train split of a layer's patch activations (streamed from disk).
Outputs in run_dir: checkpoint.pt, metrics.csv, results.json, config.json, experiment_config.json, plots/, warnings.json
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from activations.normalize import fit_normalizer
from activations.store import ActivationStore
from sae.health import check_sae
from sae.metrics import evaluate_sae
from sae.model import SparseAutoencoder, sae_loss, save_sae_checkpoint
from utils import ensure_dir, get_device, get_logger, make_splits, save_json, set_seed


def train_sae(cfg, store=None, logger=None):
    log = logger or get_logger()
    sc, layer = cfg.sae, cfg.layer
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    store = store or ActivationStore(cfg.store_dir())
    if layer not in store.layers:
        raise ValueError(f"layer {layer} not in store layers {store.layers}; run extract.py --layer {layer}")
    tr, va, _ = make_splits(store.n, cfg.data.split_fractions, cfg.seed)
    normalizer = fit_normalizer(store, layer, tr, cfg.extract.normalization).to(device)
    D = store.d_model
    sae = SparseAutoencoder(D, sc.width, sc.sae_type, sc.k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=sc.lr)
    run_dir = cfg.run_dir()
    ensure_dir(run_dir / "plots")
    rng = np.random.RandomState(cfg.seed)
    rows, t0 = [], time.time()
    val_sub = va[: sc.val_images_per_epoch]
    log.info(f"[train] {cfg.run_tag()} train_imgs={len(tr)} val_imgs={len(va)} device={device}")
    for ep in range(sc.epochs):
        sae.train()
        perm = rng.permutation(tr)
        fired = torch.zeros(sc.width, dtype=torch.bool, device=device)
        agg = {"loss": 0.0, "recon": 0.0, "l0": 0.0, "steps": 0}
        for s in tqdm(range(0, len(perm), sc.chunk_images), desc=f"epoch {ep+1}/{sc.epochs}", leave=False):
            x = normalizer(store.read_patches(layer, perm[s:s + sc.chunk_images]).to(device)).reshape(-1, D)
            x = x[torch.randperm(x.shape[0], device=device)]
            for b in range(0, x.shape[0], sc.batch_size):
                xb = x[b:b + sc.batch_size]
                loss, parts = sae_loss(sae, xb, sc.l1_lambda)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite SAE loss; lower lr or check activations")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if sc.grad_clip:
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), sc.grad_clip)
                opt.step()
                if sc.decoder_unit_norm:
                    sae.normalize_decoder()
                with torch.no_grad():
                    fired |= (sae.encode(xb) > sc.dead_threshold).any(0)
                agg["loss"] += loss.item(); agg["recon"] += parts["recon"].item(); agg["l0"] += parts["l0"].item()
                agg["steps"] += 1
        vm = evaluate_sae(sae, normalizer, store, layer, val_sub, device, thr=sc.dead_threshold)
        st = max(agg["steps"], 1)
        row = {"epoch": ep + 1, "train_loss": agg["loss"] / st, "train_mse": agg["recon"] / st,
               "train_l0": agg["l0"] / st, "train_dead_pct": 100.0 * float((~fired).float().mean().item()),
               "val_mse": vm["mse"], "val_explained_variance": vm["explained_variance"], "val_l0": vm["l0"],
               "val_dead_pct": vm["dead_pct_eval"], "val_density": vm["mean_feature_density"],
               "seconds": time.time() - t0}
        rows.append(row)
        log.info("[train] " + " ".join(f"{k}={v:.4g}" for k, v in row.items()))
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    final = evaluate_sae(sae, normalizer, store, layer, va, device, thr=sc.dead_threshold)
    freq = final.pop("frequency")
    np.save(run_dir / "val_feature_frequency.npy", freq)
    final["dead_pct_train_last_epoch"] = rows[-1]["train_dead_pct"]
    warns = check_sae(final, sc.width)
    for w in warns:
        log.warning("[health] " + w)
    cfg.save(run_dir / "config.json"); cfg.save(run_dir / "experiment_config.json")
    save_json({"metrics": final, "warnings": warns, "train_seconds": time.time() - t0,
               "n_train_images": len(tr), "n_val_images": len(va)}, run_dir / "results.json")
    save_json(warns, run_dir / "warnings.json")
    save_sae_checkpoint(run_dir / "checkpoint.pt", sae, normalizer, cfg.to_dict(), final)
    try:
        from visualization.plots import plot_training_curves
        plot_training_curves(pd.DataFrame(rows), run_dir / "plots" / "sae_training.png")
    except Exception as e:  # plotting must never kill training
        log.warning(f"[train] plot skipped: {e}")
    log.info(f"[train] done: EV={final['explained_variance']:.3f} L0={final['l0']:.1f} dead={final['dead_pct_eval']:.1f}%")
    return run_dir
