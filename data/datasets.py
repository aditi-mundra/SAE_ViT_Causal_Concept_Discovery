"""
data/datasets.py [CORE]
Dataset access: CIFAR-100, ImageFolder (ImageNet val / Places365 / any), synthetic.
`ImageSet` yields (tensor, label, position); `position` indexes the extracted store.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class SyntheticRaw:
    """Deterministic striped/blob images, 10 classes (offline tests only)."""
    n_total = 2000

    def __init__(self, seed: int = 0, size: int = 224):
        self.seed, self.size = seed, size
        self.classes = [f"synthetic_{i}" for i in range(10)]

    def __len__(self):
        return self.n_total

    def label(self, i: int) -> int:
        return int(i % 10)

    def pil(self, i: int) -> Image.Image:
        s, c = self.size, int(i % 10)
        rng = np.random.RandomState(self.seed * 100003 + int(i))
        yy, xx = np.mgrid[0:s, 0:s] / float(s)
        ang, freq = c * np.pi / 10, 4 + c
        stripes = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (xx * np.cos(ang) + yy * np.sin(ang)) + rng.rand() * 6.28)
        color = np.array([(c * 37) % 255, (c * 91) % 255, (c * 53) % 255]) / 255.0
        img = stripes[..., None] * color[None, None, :] + 0.1 * rng.rand(s, s, 3)
        r0, c0 = (c % 3) * (s // 3) + 8, ((c // 3) % 3) * (s // 3) + 8
        img[r0:r0 + s // 5, c0:c0 + s // 5] = 1.0 - color
        return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


class CifarRaw:
    def __init__(self, root: str):
        import torchvision
        Path(root).mkdir(parents=True, exist_ok=True)
        try:
            self.ds = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
        except Exception as e:
            raise RuntimeError(f"Could not load/download CIFAR-100 into {root}: {e}") from e
        self.classes = list(self.ds.classes)

    def __len__(self):
        return len(self.ds)

    def label(self, i):
        return int(self.ds.targets[i])

    def pil(self, i):
        return Image.fromarray(self.ds.data[i]).convert("RGB")


class FolderRaw:
    def __init__(self, folder: str):
        import torchvision
        if not folder or not Path(folder).exists():
            raise FileNotFoundError(f"ImageFolder path not found: '{folder}' (set data.folder)")
        self.ds = torchvision.datasets.ImageFolder(folder)
        self.classes = list(self.ds.classes)

    def __len__(self):
        return len(self.ds.samples)

    def label(self, i):
        return int(self.ds.samples[i][1])

    def pil(self, i):
        return Image.open(self.ds.samples[i][0]).convert("RGB")


def get_raw(data_cfg, seed: int = 0, size: int = 224):
    d = data_cfg.dataset
    if d == "cifar100":
        return CifarRaw(data_cfg.root)
    if d == "synthetic":
        return SyntheticRaw(seed, size)
    if d == "imagenet_val":
        return FolderRaw(data_cfg.folder or str(Path(data_cfg.root) / "imagenet_val"))
    if d == "imagefolder":
        return FolderRaw(data_cfg.folder)
    raise ValueError(f"unknown dataset {d}")


def make_transform(img_size: int, mean, std):
    return T.Compose([T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
                      T.ToTensor(), T.Normalize(mean, std)])


class ImageSet(Dataset):
    def __init__(self, raw, ids: np.ndarray, transform):
        self.raw, self.ids, self.transform = raw, np.asarray(ids), transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, pos: int):
        i = int(self.ids[pos])
        return self.transform(self.raw.pil(i)), self.raw.label(i), pos


def build_imageset(cfg, mean, std) -> Tuple[ImageSet, object]:
    raw = get_raw(cfg.data, cfg.seed, cfg.model.img_size)
    n = min(cfg.data.num_images, len(raw))
    if n < cfg.data.num_images:
        print(f"[data] only {len(raw)} images available; using {n}")
    ids = np.random.RandomState(cfg.seed).choice(len(raw), n, replace=False)
    return ImageSet(raw, ids, make_transform(cfg.model.img_size, mean, std)), raw
