"""
config.py [CORE]
Central serialisable configuration. Standard library only.

Layer convention: layer i = output of timm `model.blocks[i]` (0-indexed).
Presets: selftest (tiny random ViT + synthetic data, offline), smoke, demo, full.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class ModelConfig:
    name: str = "vit_base_patch16_224"   # any timm ViT, or "tiny_vit" (offline self-test model)
    pretrained: bool = True
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768                  # overwritten from the loaded model
    num_prefix_tokens: int = 1            # overwritten from the loaded model

    @property
    def grid(self) -> int:
        return self.img_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid * self.grid


@dataclass
class DataConfig:
    dataset: str = "cifar100"            # cifar100 | imagenet_val | imagefolder | synthetic
    root: str = str(PROJECT_ROOT / "data" / "raw")
    folder: str = ""                      # ImageFolder path (imagefolder / imagenet_val / places365)
    num_images: int = 10000
    batch_size: int = 64
    num_workers: int = 2
    split_fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2)  # SAE-train / feature-val / causal-eval


@dataclass
class ExtractConfig:
    layers: List[int] = field(default_factory=lambda: [3, 6, 9, 11])
    out_dir: str = str(PROJECT_ROOT / "activations" / "store")
    dtype: str = "float16"
    normalization: str = "standardize"   # standardize | l2 | none
    write_patch_metadata: bool = False    # patch-level activation_metadata.csv (large!)


@dataclass
class SAEConfig:
    sae_type: str = "topk"               # standard (ReLU, lambda=0) | l1 | topk
    width: int = 4096
    k: int = 32
    l1_lambda: float = 3e-3
    lr: float = 1e-3
    batch_size: int = 4096
    epochs: int = 5
    dead_threshold: float = 1e-8
    decoder_unit_norm: bool = True
    grad_clip: float = 1.0
    chunk_images: int = 64
    val_images_per_epoch: int = 300


@dataclass
class DiscoveryConfig:
    top_n_patches: int = 100
    heatmap_threshold: float = 0.5
    clip_prompts: List[str] = field(default_factory=lambda: [
        "wheel", "road", "grass", "sky", "building", "window", "face", "fur",
        "water", "text", "leaf", "snow", "sand", "metal", "wood"])
    use_clip: bool = True                # ADVANCED OPTIONAL
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_top_m: int = 10
    clip_max_features: int = 256
    chunk_images: int = 16
    n_feature_reports: int = 12
    specific_thresholds: Tuple[float, float] = (0.4, 0.7)  # normalised class entropy cut-offs
    min_freq: float = 1e-4
    max_freq: float = 0.2
    n_clusters: int = 10
    use_umap: bool = True                # ADVANCED OPTIONAL


@dataclass
class InterventionConfig:
    suppress_alphas: Tuple[float, ...] = (0.75, 0.5, 0.25, 0.0)
    amplify_alphas: Tuple[float, ...] = (1.5, 2.0, 3.0, 5.0)
    dose_multipliers: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    steer_alphas: Tuple[float, ...] = (-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0)
    local_percentiles: Tuple[int, ...] = (50, 75, 90, 95)
    n_random_controls: int = 30
    n_bootstrap: int = 1000
    n_eval_images: int = 100
    n_candidates: int = 10
    min_active_images: int = 5
    control_freq_ratio: float = 3.0
    n_interaction_features: int = 6
    n_spatial_features: int = 3
    n_spatial_images: int = 5
    n_multi_steer: int = 3
    random_dose_controls: int = 10
    amp_alpha_primary: float = 2.0
    n_gradcam_figs: int = 3
    batch_size: int = 32
    max_norm_ratio: float = 3.0
    max_abs_alpha: float = 10.0


@dataclass
class SweepConfig:
    widths: Tuple[int, ...] = (2048, 4096, 8192)
    types: Tuple[str, ...] = ("standard", "l1", "topk")
    topks: Tuple[int, ...] = (16, 32, 64)


@dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 42
    layer: int = 9
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)

    # ---- paths -----------------------------------------------------------
    def store_dir(self) -> Path:
        return Path(self.extract.out_dir) / self.name

    def exp_dir(self) -> Path:
        p = PROJECT_ROOT / "experiments" / self.name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def run_tag(self) -> str:
        t = f"layer_{self.layer}__{self.sae.sae_type}_{self.sae.width}"
        if self.sae.sae_type == "topk":
            t += f"_k{self.sae.k}"
        return t

    def run_dir(self) -> Path:
        p = self.exp_dir() / self.run_tag()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def variant(self, **kw) -> "ExperimentConfig":
        """Copy with overrides: layer=, sae_type=, width=, k="""
        c = copy.deepcopy(self)
        if "layer" in kw:
            c.layer = kw["layer"]
        if "sae_type" in kw:
            c.sae.sae_type = kw["sae_type"]
        if "width" in kw:
            c.sae.width = kw["width"]
        if "k" in kw:
            c.sae.k = kw["k"]
        return c

    # ---- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        d["run_tag"] = self.run_tag()
        return d

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text())
        return _build(cls, raw)

    def validate(self) -> None:
        if abs(sum(self.data.split_fractions) - 1.0) > 1e-6:
            raise ValueError("split_fractions must sum to 1")
        if self.sae.sae_type not in {"standard", "l1", "topk"}:
            raise ValueError(f"unknown sae_type {self.sae.sae_type}")
        if self.sae.sae_type == "topk" and not (0 < self.sae.k <= self.sae.width):
            raise ValueError("Top-K requires 0 < k <= width")
        if self.extract.normalization not in {"standardize", "l2", "none"}:
            raise ValueError("normalization must be standardize | l2 | none")
        if self.data.dataset not in {"cifar100", "imagenet_val", "imagefolder", "synthetic"}:
            raise ValueError("dataset must be cifar100 | imagenet_val | imagefolder | synthetic")
        if any(l < 0 for l in self.extract.layers):
            raise ValueError("layers must be >= 0")
        if self.layer not in self.extract.layers:
            raise ValueError(f"layer {self.layer} not in extract.layers {self.extract.layers}")
        if not (3 <= self.intervention.n_random_controls <= 50):
            raise ValueError("n_random_controls should be in [3, 50] (recommended 10-50)")


def _build(klass, d: dict):
    kw = {}
    for f in fields(klass):
        if f.name not in d:
            continue
        v = d[f.name]
        if f.default is not MISSING:
            default = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            default = f.default_factory()  # type: ignore[misc]
        else:
            default = None
        if is_dataclass(default):
            kw[f.name] = _build(type(default), v)
        elif isinstance(default, tuple):
            kw[f.name] = tuple(v)
        else:
            kw[f.name] = v
    return klass(**kw)


# ----------------------------------------------------------------- presets
def selftest_config() -> ExperimentConfig:
    """Offline end-to-end test: tiny random ViT + synthetic images."""
    c = ExperimentConfig(name="selftest", layer=2)
    c.model.name, c.model.pretrained, c.model.embed_dim = "tiny_vit", False, 64
    c.data.dataset, c.data.num_images, c.data.batch_size, c.data.num_workers = "synthetic", 80, 20, 0
    c.extract.layers = [2]
    c.sae.width, c.sae.k, c.sae.epochs, c.sae.batch_size, c.sae.chunk_images = 256, 8, 2, 1024, 16
    c.discovery.use_clip, c.discovery.top_n_patches, c.discovery.n_feature_reports = False, 20, 2
    c.discovery.min_freq, c.discovery.use_umap = 1e-3, False
    i = c.intervention
    i.n_random_controls, i.n_bootstrap, i.n_eval_images, i.n_candidates = 5, 100, 8, 4
    i.min_active_images, i.n_interaction_features, i.n_spatial_images = 2, 3, 2
    i.n_spatial_features, i.random_dose_controls, i.batch_size, i.n_gradcam_figs = 1, 3, 16, 1
    c.sweep.widths, c.sweep.types, c.sweep.topks = (256,), ("topk",), (8,)
    return c


def smoke_config() -> ExperimentConfig:
    """10 images, 1 layer, 256 SAE features, 1 epoch (real ViT)."""
    c = ExperimentConfig(name="smoke")
    c.data.num_images, c.data.batch_size, c.data.num_workers = 10, 5, 0
    c.extract.layers = [9]
    c.sae.width, c.sae.k, c.sae.epochs, c.sae.batch_size, c.sae.chunk_images = 256, 8, 1, 512, 5
    c.discovery.use_clip, c.discovery.top_n_patches, c.discovery.n_feature_reports = False, 20, 1
    c.discovery.min_freq, c.discovery.use_umap = 1e-3, False
    i = c.intervention
    i.n_random_controls, i.n_bootstrap, i.n_eval_images, i.n_candidates = 5, 100, 4, 3
    i.min_active_images, i.n_interaction_features, i.n_spatial_images = 1, 3, 1
    i.n_spatial_features, i.random_dose_controls, i.n_gradcam_figs = 1, 3, 1
    c.sweep.widths, c.sweep.types, c.sweep.topks = (256,), ("topk",), (8,)
    return c


def demo_config() -> ExperimentConfig:
    """CIFAR-100, 500 images, layer 9, 1024 features, Top-K = 16."""
    c = ExperimentConfig(name="demo")
    c.data.num_images = 500
    c.extract.layers = [9]
    c.sae.sae_type, c.sae.width, c.sae.k, c.sae.epochs = "topk", 1024, 16, 3
    c.discovery.top_n_patches, c.discovery.clip_max_features = 50, 64
    i = c.intervention
    i.n_eval_images, i.n_candidates, i.n_random_controls, i.min_active_images = 40, 8, 20, 3
    c.sweep.widths, c.sweep.topks = (512, 1024), (16,)
    return c


def full_config() -> ExperimentConfig:
    """Research mode: 10k images, layers 3/6/9/11, 4096 features (scale to hardware)."""
    c = ExperimentConfig(name="full")
    c.data.num_images = 10000
    c.sae.width, c.sae.k, c.sae.epochs = 4096, 32, 10
    c.layer = 9
    return c


PRESETS = {"selftest": selftest_config, "smoke": smoke_config, "demo": demo_config, "full": full_config}


def get_config(mode: str = "demo") -> ExperimentConfig:
    if mode not in PRESETS:
        raise KeyError(f"mode must be one of {list(PRESETS)}")
    cfg = PRESETS[mode]()
    cfg.validate()
    return cfg


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    cfg = get_config(mode)
    print("saved ->", cfg.save(PROJECT_ROOT / "configs" / f"{mode}.json"))
