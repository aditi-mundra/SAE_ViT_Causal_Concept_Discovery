"""utils.py [CORE]: seeding, device, logging, JSON, splits, entropy helpers."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(pref: str = "auto"):
    import torch
    if pref != "auto":
        if pref == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if pref == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str = "saevit", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return json.loads(path.read_text())


def make_splits(n: int, fractions: Sequence[float] = (0.6, 0.2, 0.2),
                seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic disjoint position splits: (sae_train, feature_val, causal_eval)."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    if n < 3:
        raise ValueError("need at least 3 images to make 3 splits")
    idx = np.random.RandomState(seed).permutation(n)
    a = max(1, int(round(fractions[0] * n)))
    b = max(1, int(round(fractions[1] * n)))
    b = min(b, n - a - 1)
    return idx[:a], idx[a:a + b], idx[a + b:]


def entropy_bits(p: np.ndarray, axis: int = 0) -> np.ndarray:
    """Shannon entropy in bits of (already normalised) distributions along `axis`."""
    p = np.asarray(p, dtype=np.float64)
    safe = np.where(p > 0, p, 1.0)
    return -(p * np.log2(safe)).sum(axis=axis)
