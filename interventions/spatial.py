"""interventions/spatial.py [CORE]: per-patch ablation effect maps (spatial causal analysis)."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def patch_ablation_map(engine, hidden_1: torch.Tensor, fid: int, target: int, batch_size: int = 32):
    """
    hidden_1: [1, tokens, D]. Ablate feature `fid` at ONE patch at a time (only patches where it is active).
    Returns (activation [P], effect [P]) with effect = z_target(base) - z_target(ablated at that patch).
    """
    dev = engine.device
    h = hidden_1.to(dev)
    f = engine.encode_features(h, [fid])[0, :, 0].cpu().numpy()
    P = f.shape[0]
    base = engine.baseline(h)[0, target].item()
    active = np.where(f > 0)[0]
    eff = np.zeros(P)
    for s in range(0, len(active), batch_size):
        idx = active[s:s + batch_size]
        mask = torch.zeros(len(idx), P, dtype=torch.bool)
        mask[torch.arange(len(idx)), torch.from_numpy(idx)] = True
        res = engine.run_many(h.repeat(len(idx), 1, 1).cpu(), [fid], [0.0], "scale", explicit_mask=mask,
                              batch_size=batch_size)
        eff[idx] = base - res["logits"][:, target]
    return f, eff
