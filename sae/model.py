"""
sae/model.py [CORE]
Sparse autoencoder: standard ReLU (lambda=0), L1-penalised ReLU, or Top-K.

  pre = (x - b_dec) W_enc + b_enc ;  f = ReLU(pre) ; Top-K keeps the K largest f, zeroes the rest
  x_hat = f W_dec + b_dec
W_dec has shape [width, d]: row i is the decoder direction of feature i (the document's W_dec[:, i]).
Loss: per-element MSE  ||x - x_hat||^2 / d  (+ lambda * ||f||_1 for the l1 type).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, width: int, sae_type: str = "topk", k: int = 32):
        super().__init__()
        if sae_type not in {"standard", "l1", "topk"}:
            raise ValueError(sae_type)
        if sae_type == "topk" and not (0 < k <= width):
            raise ValueError("need 0 < k <= width")
        self.d_in, self.width, self.sae_type, self.k = d_in, width, sae_type, k
        self.W_enc = nn.Parameter(torch.empty(d_in, width))
        self.b_enc = nn.Parameter(torch.zeros(width))
        self.W_dec = nn.Parameter(torch.empty(width, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        with torch.no_grad():
            dec = torch.randn(width, d_in)
            dec /= dec.norm(dim=1, keepdim=True)
            self.W_dec.copy_(dec)
            self.W_enc.copy_(dec.T.clone())

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        f = F.relu(self.pre_acts(x))
        if self.sae_type == "topk":
            vals, idx = torch.topk(f, self.k, dim=-1)
            f = torch.zeros_like(f).scatter(-1, idx, vals)
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def decoder_direction(self, i: int) -> torch.Tensor:
        return self.W_dec[i]


def sae_loss(sae: SparseAutoencoder, x: torch.Tensor, l1_lambda: float = 0.0):
    x_hat, f = sae(x)
    recon = ((x - x_hat) ** 2).sum(-1).mean() / x.shape[-1]
    sparse = f.abs().sum(-1).mean()
    lam = l1_lambda if sae.sae_type == "l1" else 0.0
    total = recon + lam * sparse
    return total, {"recon": recon.detach(), "l1": sparse.detach(), "l0": (f > 0).float().sum(-1).mean().detach()}


def save_sae_checkpoint(path, sae, normalizer, cfg_dict: dict, metrics: dict) -> None:
    torch.save({"state": sae.state_dict(), "d_in": sae.d_in, "width": sae.width, "sae_type": sae.sae_type,
                "k": sae.k, "normalizer": normalizer.state(), "config": cfg_dict, "metrics": metrics}, path)


def load_sae_checkpoint(path, device):
    from activations.normalize import Normalizer
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(ck["d_in"], ck["width"], ck["sae_type"], ck["k"])
    sae.load_state_dict(ck["state"])
    sae.eval().to(device)
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae, Normalizer.from_state(ck["normalizer"]).to(device), ck
