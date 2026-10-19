"""
models/vit.py [CORE]
Frozen ViT loading (timm), hook-based activation capture, and the key helper
`forward_from_hidden` that resumes the forward pass from the output of block L.
`TinyViT` is a small randomly-initialised ViT used only for offline self-tests.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    """Minimal timm-like ViT (blocks / norm / forward_head / embed_dim) for tests."""

    def __init__(self, img_size=224, patch_size=16, embed_dim=64, depth=4, heads=4,
                 num_classes=10, seed=0):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.embed_dim, self.num_prefix_tokens = embed_dim, 1
            g = img_size // patch_size
            self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.pos_embed = nn.Parameter(torch.randn(1, g * g + 1, embed_dim) * 0.02)
            self.blocks = nn.ModuleList([_Block(embed_dim, heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.class_names = [f"synthetic_{i}" for i in range(num_classes)]
        self.pretrained_cfg = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}

    def forward_features(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + self.pos_embed
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def forward_head(self, x, pre_logits: bool = False):
        x = x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def load_vit(model_cfg, device) -> nn.Module:
    """Load and freeze the ViT. Updates model_cfg.embed_dim / num_prefix_tokens in place."""
    if model_cfg.name == "tiny_vit":
        model = TinyViT(model_cfg.img_size, model_cfg.patch_size, model_cfg.embed_dim or 64)
    else:
        try:
            import timm
        except ImportError as e:
            raise ImportError("timm is required: pip install timm") from e
        model = timm.create_model(model_cfg.name, pretrained=model_cfg.pretrained)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model_cfg.embed_dim = int(model.embed_dim)
    model_cfg.num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    return model


def get_norm_stats(model) -> Tuple[tuple, tuple]:
    pc = getattr(model, "pretrained_cfg", None) or {}
    return (tuple(pc.get("mean", (0.485, 0.456, 0.406))),
            tuple(pc.get("std", (0.229, 0.224, 0.225))))


def model_class_names(model) -> List[str]:
    if hasattr(model, "class_names"):
        return list(model.class_names)
    n = int(getattr(model, "num_classes", 1000))
    if n == 1000:
        try:
            from timm.data import ImageNetInfo
            info = ImageNetInfo()
            return [info.index_to_description(i) for i in range(n)]
        except Exception:
            pass
    return [f"class_{i}" for i in range(n)]


@torch.no_grad()
def capture(model, images: torch.Tensor, layers: Sequence[int]):
    """Forward pass; returns (logits, {layer: block-output [B, tokens, D]})."""
    store: Dict[int, torch.Tensor] = {}
    hooks = [model.blocks[l].register_forward_hook(
        lambda m, i, o, l=l: store.__setitem__(l, o.detach())) for l in layers]
    try:
        logits = model(images)
    finally:
        for h in hooks:
            h.remove()
    return logits, store


@torch.no_grad()
def forward_from_hidden(model, hidden: torch.Tensor, layer: int) -> torch.Tensor:
    """Resume the ViT after block `layer` from hidden states [B, tokens, D] -> logits."""
    x = hidden
    for blk in list(model.blocks)[layer + 1:]:
        x = blk(x)
    x = model.norm(x)
    return model.forward_head(x)
