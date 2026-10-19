"""activations/store.py [CORE]: read-only access to the on-disk activation store."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from utils import load_json


def _read(arr, positions) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64)
    order = np.argsort(pos)
    out = np.empty((len(pos),) + arr.shape[1:], dtype=np.float32)
    out[order] = arr[pos[order]]
    return out


class ActivationStore:
    def __init__(self, root):
        self.root = Path(root)
        mp = self.root / "meta.json"
        if not mp.exists():
            raise FileNotFoundError(f"No activation store at {self.root}. Run `python extract.py` first.")
        self.meta = load_json(mp)
        self.n, self.P, self.d_model = self.meta["n"], self.meta["num_patches"], self.meta["d_model"]
        self.layers = self.meta["layers"]
        self.num_prefix = self.meta["num_prefix_tokens"]
        self.labels = np.load(self.root / "labels.npy")
        self.image_ids = np.load(self.root / "image_ids.npy")
        self.class_names = self.meta["class_names"]
        self.model_class_names = self.meta["model_class_names"]
        self._logits = None

    def patches(self, layer: int):
        return np.load(self.root / f"patches_L{layer}.npy", mmap_mode="r")

    def prefix(self, layer: int):
        return np.load(self.root / f"prefix_L{layer}.npy", mmap_mode="r")

    @property
    def logits(self) -> np.ndarray:
        if self._logits is None:
            self._logits = np.load(self.root / "logits.npy")
        return self._logits

    def read_patches(self, layer: int, positions) -> torch.Tensor:
        return torch.from_numpy(_read(self.patches(layer), positions))

    def read_hidden(self, layer: int, positions) -> torch.Tensor:
        """[n, num_prefix + P, D] float32 (prefix tokens first)."""
        pre = torch.from_numpy(_read(self.prefix(layer), positions))
        return torch.cat([pre, self.read_patches(layer, positions)], dim=1)
