"""
causal/metrics.py [CORE] (numpy only)
Logit-space causal effect:  CE = z_y(x) - z_y(x_do)  (positive = target logit DROPPED).
`delta_logit = z_new - z_base` is the signed change (positive = target logit ROSE; use for amplification/steering).
"""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def compute_effects(base: np.ndarray, new: np.ndarray, target: np.ndarray) -> dict:
    n = np.arange(len(target))
    zb, zn = base[n, target], new[n, target]
    pb, pn = softmax(base)[n, target], softmax(new)[n, target]
    rank_b, rank_n = (base > zb[:, None]).sum(1), (new > zn[:, None]).sum(1)
    top5 = np.argsort(-new, axis=1)[:, :5]
    return {"base_logit": zb, "new_logit": zn, "logit_effect": zb - zn, "delta_logit": zn - zb,
            "base_prob": pb, "new_prob": pn, "prob_effect": pb - pn,
            "flip": new.argmax(1) != target, "rank_change": rank_n - rank_b, "top5_new": top5}
