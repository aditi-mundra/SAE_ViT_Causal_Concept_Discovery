"""context.py [CORE]: RunContext bundles everything needed to analyse one trained SAE run."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from activations.store import ActivationStore
from config import ExperimentConfig
from data.datasets import get_raw, make_transform
from interventions.engine import InterventionEngine
from models.vit import get_norm_stats, load_vit, model_class_names
from sae.model import load_sae_checkpoint
from utils import get_device, make_splits


class RunContext:
    def __init__(self, cfg: ExperimentConfig, device=None):
        self.cfg, self.layer = cfg, cfg.layer
        self.device = device or get_device(cfg.device)
        self.run_dir = Path(cfg.run_dir())
        ck = self.run_dir / "checkpoint.pt"
        if not ck.exists():
            raise FileNotFoundError(f"{ck} missing: run `python train.py` for this configuration first")
        self.sae, self.normalizer, self.ckpt = load_sae_checkpoint(ck, self.device)
        self.store = ActivationStore(cfg.store_dir())
        tr, va, ca = make_splits(self.store.n, cfg.data.split_fractions, cfg.seed)
        self.splits = {"train": tr, "val": va, "causal": ca}
        self._model = self._raw = self._disc = self._summary = None

    @classmethod
    def from_run_dir(cls, run_dir, device=None) -> "RunContext":
        return cls(ExperimentConfig.load(Path(run_dir) / "config.json"), device)

    # ---- lazily loaded heavy objects
    @property
    def model(self):
        if self._model is None:
            self._model = load_vit(self.cfg.model, self.device)
        return self._model

    @property
    def raw(self):
        if self._raw is None:
            self._raw = get_raw(self.cfg.data, self.cfg.seed, self.cfg.model.img_size)
        return self._raw

    @property
    def vit_class_names(self):
        return self.store.model_class_names

    def transform(self):
        mean, std = get_norm_stats(self.model)
        return make_transform(self.cfg.model.img_size, mean, std)

    def image_pil(self, position: int) -> Image.Image:
        return self.raw.pil(int(self.store.image_ids[int(position)]))

    def image_tensor(self, pil: Image.Image) -> torch.Tensor:
        return self.transform()(pil.convert("RGB"))

    # ---- discovery artifacts
    @property
    def disc_dir(self) -> Path:
        return self.run_dir / "discovery"

    @property
    def disc(self) -> Optional[dict]:
        if self._disc is None and (self.disc_dir / "features.npz").exists():
            self._disc = dict(np.load(self.disc_dir / "features.npz"))
        return self._disc

    def summary(self) -> pd.DataFrame:
        if self._summary is None:
            p = self.disc_dir / "feature_summary.csv"
            if not p.exists():
                raise FileNotFoundError(f"{p} missing: run `python discover.py` first")
            self._summary = pd.read_csv(p).set_index("feature_id", drop=False)
        return self._summary

    # ---- helpers
    def hidden(self, positions) -> torch.Tensor:
        return self.store.read_hidden(self.layer, positions)

    def engine(self) -> InterventionEngine:
        return InterventionEngine(self.model, self.sae, self.normalizer, self.layer,
                                  self.store.num_prefix, self.cfg.intervention)

    @torch.no_grad()
    def feature_activations(self, fid: int, positions, chunk: int = 32) -> np.ndarray:
        """Activation of one feature at every patch -> [n_images, P]."""
        out, pos = [], np.asarray(positions)
        for s in range(0, len(pos), chunk):
            x = self.normalizer(self.store.read_patches(self.layer, pos[s:s + chunk]).to(self.device))
            f = self.sae.encode(x.reshape(-1, self.store.d_model))[:, fid]
            out.append(f.reshape(len(pos[s:s + chunk]), self.store.P).cpu().numpy())
        return np.concatenate(out, 0)
