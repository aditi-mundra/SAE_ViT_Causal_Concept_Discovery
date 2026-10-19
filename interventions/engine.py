"""
interventions/engine.py [CORE] -- the causal intervention engine.

For hidden states h (block-`layer` output) split into prefix tokens and patch activations x [B,P,D]:
  x_n = Normalizer(x);  f = SAE.encode(x_n)                      (features of interest only)
  mode "scale": f'_i = alpha_i * f_i  (alpha=0 ablation, <1 suppression, >1 amplification)
                delta_n = sum_i (f'_i - f_i) * W_dec[i]
  mode "steer": delta_n = sum_i alpha_i * W_dec[i]                (added at every selected patch, active or not)
  x' = x + Normalizer.delta_to_raw(delta_n, x)                    (SAE reconstruction error is PRESERVED)
Then the forward pass resumes from block layer+1. Patch selection: global (all patches), local
(patches whose f_i >= the q-th percentile of that feature's *active* patches in the image), or an explicit mask.
Safety: alpha clamp, activation-norm ratio, NaN/Inf and logit/probability ranges are reported in `info`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from models.vit import forward_from_hidden


class InterventionEngine:
    def __init__(self, model, sae, normalizer, layer: int, num_prefix: int, icfg):
        self.model, self.sae, self.normalizer = model, sae, normalizer
        self.layer, self.npre = layer, num_prefix
        self.max_ratio, self.max_alpha = icfg.max_norm_ratio, icfg.max_abs_alpha

    @property
    def device(self):
        return self.sae.W_dec.device

    # ---------------------------------------------------------------- helpers
    def _alphas(self, alphas, k):
        a = [float(alphas)] * k if np.isscalar(alphas) else [float(v) for v in alphas]
        if len(a) != k:
            raise ValueError("need one alpha per feature")
        if any(not np.isfinite(v) for v in a):
            raise ValueError("alpha must be finite")
        clamped = any(abs(v) > self.max_alpha for v in a)
        return [float(np.clip(v, -self.max_alpha, self.max_alpha)) for v in a], clamped

    @torch.no_grad()
    def encode_features(self, hidden: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
        """SAE activations of selected features at every patch -> [B,P,len(ids)]."""
        x = hidden[:, self.npre:]
        B, P, D = x.shape
        f = self.sae.encode(self.normalizer(x).reshape(-1, D)).reshape(B, P, -1)
        return f[:, :, list(ids)]

    @staticmethod
    def _local_mask(f: torch.Tensor, q: float) -> torch.Tensor:
        vals = torch.where(f > 0, f, torch.full_like(f, float("nan")))
        thr = torch.nanquantile(vals, q / 100.0, dim=1, keepdim=True)
        return (f >= thr) & (f > 0)

    @torch.no_grad()
    def baseline(self, hidden: torch.Tensor) -> torch.Tensor:
        return forward_from_hidden(self.model, hidden, self.layer)

    @torch.no_grad()
    def baseline_many(self, hidden: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        out = [self.baseline(hidden[s:s + batch_size].to(self.device)).cpu() for s in range(0, len(hidden), batch_size)]
        return torch.cat(out).numpy()

    # -------------------------------------------------------------------- run
    @torch.no_grad()
    def run(self, hidden: torch.Tensor, ids: Sequence[int], alphas, mode: str = "scale",
            local_percentile: Optional[float] = None, explicit_mask: Optional[torch.Tensor] = None):
        ids = [int(i) for i in ids]
        alphas, clamped = self._alphas(alphas, len(ids))
        prefix, x = hidden[:, :self.npre], hidden[:, self.npre:]
        f = self.encode_features(hidden, ids)
        mask = torch.ones_like(f, dtype=torch.bool)
        if local_percentile is not None:
            mask = self._local_mask(f, local_percentile)
        if explicit_mask is not None:
            mask = mask & explicit_mask.bool().unsqueeze(-1)
        a = torch.tensor(alphas, device=f.device, dtype=f.dtype).view(1, 1, -1)
        if mode == "scale":
            df = torch.where(mask, f * a - f, torch.zeros_like(f))
            n_mod = (df != 0).any(-1).sum(1)
        elif mode == "steer":
            df = a * mask.to(f.dtype)
            n_mod = mask.any(-1).sum(1)
        else:
            raise ValueError("mode must be 'scale' or 'steer'")
        delta_n = torch.einsum("bpk,kd->bpd", df, self.sae.W_dec[ids])
        x_new = x + self.normalizer.delta_to_raw(delta_n, x)
        logits = forward_from_hidden(self.model, torch.cat([prefix, x_new], 1), self.layer)
        n0, n1 = x.norm(dim=-1), x_new.norm(dim=-1)
        ratio = (n1 / n0.clamp_min(1e-8)).max().item()
        bad = (not torch.isfinite(logits).all().item()) or (not torch.isfinite(x_new).all().item())
        probs = torch.softmax(logits.float(), -1)
        info = {"n_modified": n_mod.cpu().numpy(), "max_norm_ratio": ratio, "norm_before": n0.mean().item(),
                "norm_after": n1.mean().item(), "nan": bad, "clamped": clamped,
                "flagged": bad or ratio > self.max_ratio,
                "logit_min": logits.min().item(), "logit_max": logits.max().item(),
                "prob_min": probs.min().item(), "prob_max": probs.max().item()}
        return logits, info

    def run_many(self, hidden: torch.Tensor, ids, alphas, mode: str = "scale", local_percentile=None,
                 explicit_mask: Optional[torch.Tensor] = None, batch_size: int = 32) -> Dict:
        lg, infos = [], []
        for s in range(0, hidden.shape[0], batch_size):
            em = None if explicit_mask is None else explicit_mask[s:s + batch_size].to(self.device)
            l, i = self.run(hidden[s:s + batch_size].to(self.device), ids, alphas, mode, local_percentile, em)
            lg.append(l.cpu()); infos.append(i)
        return {"logits": torch.cat(lg).numpy(),
                "n_modified": np.concatenate([i["n_modified"] for i in infos]),
                "max_norm_ratio": max(i["max_norm_ratio"] for i in infos),
                "norm_before": float(np.mean([i["norm_before"] for i in infos])),
                "norm_after": float(np.mean([i["norm_after"] for i in infos])),
                "nan": any(i["nan"] for i in infos), "clamped": any(i["clamped"] for i in infos),
                "flagged": any(i["flagged"] for i in infos),
                "logit_min": min(i["logit_min"] for i in infos), "logit_max": max(i["logit_max"] for i in infos),
                "prob_min": min(i["prob_min"] for i in infos), "prob_max": max(i["prob_max"] for i in infos)}
