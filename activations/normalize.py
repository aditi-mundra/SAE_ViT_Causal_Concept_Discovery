"""activations/normalize.py [CORE]: standardisation / L2 normalisation shared by training and intervention."""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import torch


class Normalizer:
    """
    standardize: x' = (x - mu) / sigma         (per-dimension, fit on SAE-train patches)
    l2:          x' = sqrt(D) * x / ||x||_2    (constant sqrt(D) keeps the loss scale; direction identical to x/||x||)
    none:        x' = x
    `delta_to_raw` maps a change in normalised space back to raw space (exact for
    standardize/none; first-order for l2).
    """

    def __init__(self, mode: str = "standardize", mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None):
        self.mode, self.mean, self.std = mode, mean, std

    def to(self, device):
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "standardize":
            return (x - self.mean) / self.std
        if self.mode == "l2":
            return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8) * (x.shape[-1] ** 0.5)
        return x

    def delta_to_raw(self, delta: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        if self.mode == "standardize":
            return delta * self.std
        if self.mode == "l2":
            return delta * (x_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8) / (x_raw.shape[-1] ** 0.5))
        return delta

    def state(self) -> dict:
        return {"mode": self.mode,
                "mean": None if self.mean is None else self.mean.detach().cpu(),
                "std": None if self.std is None else self.std.detach().cpu()}

    @classmethod
    def from_state(cls, st: dict) -> "Normalizer":
        return cls(st["mode"], st.get("mean"), st.get("std"))


def fit_normalizer(store, layer: int, train_positions, mode: str, chunk: int = 128, cache: bool = True) -> Normalizer:
    if mode != "standardize":
        return Normalizer(mode)
    pos = np.asarray(train_positions)
    key = hashlib.md5(pos.tobytes()).hexdigest()[:10]
    path = store.root / f"norm_L{layer}_{key}.npz"
    if cache and path.exists():
        d = np.load(path)
        mean, std = d["mean"], d["std"]
    else:
        arr = store.patches(layer)
        D = arr.shape[-1]
        s, s2, n = np.zeros(D), np.zeros(D), 0
        for st in range(0, len(pos), chunk):
            x = np.asarray(arr[np.sort(pos[st:st + chunk])], dtype=np.float64).reshape(-1, D)
            s += x.sum(0); s2 += (x ** 2).sum(0); n += len(x)
        mean = s / n
        std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
        std = np.maximum(std, 1e-6)
        mean, std = mean.astype(np.float32), std.astype(np.float32)
        np.savez(path, mean=mean, std=std)
    return Normalizer("standardize", torch.from_numpy(mean).float(), torch.from_numpy(std).float())
