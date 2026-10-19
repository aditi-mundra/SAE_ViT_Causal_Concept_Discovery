"""reports/generate_report.py [CORE]: builds a research-style report; every number is read from experiment outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df is None or len(df) == 0:
        return "_(no data)_"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values)


def _read(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def generate_report(cfg) -> Path:
    ed, rd = cfg.exp_dir(), cfg.run_dir()
    comp = _read(ed / "comparison.csv")
    cand = _read(rd / "causal" / "candidate_causal_summary.csv")
    inter = _read(rd / "causal" / "interactions.csv")
    gc = _read(rd / "causal" / "gradcam_comparison.csv")
    sp = _read(rd / "causal" / "spatial_summary.csv")
    hyp = json.loads((ed / "hypotheses.json").read_text()) if (ed / "hypotheses.json").exists() else {}
    res = json.loads((rd / "results.json").read_text())["metrics"] if (rd / "results.json").exists() else {}
    warn = json.loads((rd / "warnings.json").read_text()) if (rd / "warnings.json").exists() else []
    plots = rd / "plots"
    figs = "\n".join(f"- `{p.relative_to(PROJECT_ROOT)}`" for p in sorted(plots.glob("*.png"))) if plots.exists() else ""
    abstract = (f"We train sparse autoencoders ({cfg.sae.sae_type}, width {cfg.sae.width}) on patch activations of "
                f"{cfg.model.name} at block {cfg.layer} using {cfg.data.num_images} {cfg.data.dataset} images "
                f"(seed {cfg.seed}). On held-out validation patches the SAE reaches explained variance "
                f"{res.get('explained_variance', float('nan')):.3f} with mean L0 {res.get('l0', float('nan')):.1f} "
                f"and {res.get('dead_pct_eval', float('nan')):.1f}% dead features. Concept candidates are chosen from "
                f"validation data only and tested by ablation, amplification, steering and frequency-matched random "
                f"controls on a disjoint causal-evaluation split.")
    sections = [
        f"# SAE-Based Causal Concept Discovery in Vision Transformers\n\n_Experiment `{cfg.name}`, run `{cfg.run_tag()}`. "
        "All numbers below were produced by this code; nothing is hand-entered._\n",
        "## Abstract\n" + abstract,
        "## Introduction & Motivation\nViT activations are polysemantic: single neurons mix many visual concepts (superposition). "
        "Sparse autoencoders re-express activations in an overcomplete sparse basis whose directions may be interpretable. "
        "Heatmap methods show where the model looks, not what is represented or whether it matters.",
        "## Problem Statement & Research Questions\nCan SAEs decompose ViT representations into interpretable concepts, and do controlled "
        "interventions show that these concepts causally influence predictions? (RQ1-RQ10 are answered below with measured values.)",
        "## Hypotheses (declared before running)\nH1 SAE features more interpretable than neurons; H2 later layers more semantic; "
        "H3 ablation > random ablation; H4 amplification raises the aligned logit for some features; H5 local sometimes > global; "
        "H6 some concepts are redundantly represented. Verdict rules are documented in `causal/hypotheses.py`.",
        "## Related Work\nViT (Dosovitskiy et al.), Grad-CAM (Selvaraju et al.), superposition/toy models (Elhage et al.), "
        "dictionary learning for LLM interpretability (Bricken et al.; Cunningham et al.), Top-K SAEs (Gao et al.), "
        "activation steering and causal scrubbing, CLIP (Radford et al.). See References.",
        "## Methodology\nViT -> block-L patch activations (CLS kept separately) -> normalisation -> SAE -> feature discovery "
        "(top-100 patches, class purity/entropy, spatial concentration, CLIP labels) -> candidate selection (validation only) -> "
        "intervention x' = x + (f'_i - f_i) W_dec[i] preserving SAE error -> logit-space effects -> random controls, bootstrap, "
        "paired tests, Holm correction.",
        f"## Experimental Setup\nModel `{cfg.model.name}`, dataset `{cfg.data.dataset}` ({cfg.data.num_images} images), split "
        f"{cfg.data.split_fractions} (SAE-train / feature-val / causal-eval), normalisation `{cfg.extract.normalization}`, "
        f"SAE `{cfg.sae.sae_type}` width {cfg.sae.width} k={cfg.sae.k} lambda={cfg.sae.l1_lambda} lr={cfg.sae.lr} epochs={cfg.sae.epochs}.",
        "## Results: SAE quality (RQ1, RQ2)\n" + "\n".join(f"- {k}: {v:.4g}" for k, v in res.items() if isinstance(v, (int, float))),
        "## Layer / Width / SAE-type Comparison (RQ4)\n" + df_to_md(comp),
        "## Causal Analysis (RQ5-RQ8)\n" + (df_to_md(cand[[c for c in ["feature_id", "concept", "n_images", "abl_mean", "abl_ci_lo", "abl_ci_hi",
              "abl_dz", "abl_rand_rand_mean", "abl_rand_p_empirical", "abl_p_holm_paired", "amp_mean", "dose_spearman_rho",
              "causal_rank", "interp_rank", "combined_rank"] if c in cand.columns]]) if cand is not None else "_(not run)_"),
        "## Spatial causal effects (RQ9)\n" + df_to_md(sp),
        "## Feature Interaction (RQ10)\n" + df_to_md(inter),
        "## Grad-CAM Comparison\nGrad-CAM: where is the model focusing? SAE feature: what internal feature is active? Intervention: does changing it "
        "change the prediction?\n" + df_to_md(gc),
        "## Hypotheses: verdicts\n" + "\n".join(f"- **{h}**: {d['verdict']}" for h, d in hyp.items()),
        "## Automatic warnings\n" + ("\n".join(f"- {w}" for w in warn) if warn else "_none_"),
        "## Figures\n" + figs,
        "## Limitations\nSAE reconstruction error is not zero; feature splitting and absorption can blur concepts; candidate selection uses "
        "proxy metrics; effects are measured for the model's own predicted class (ImageNet head) on CIFAR-100 images (distribution shift); "
        "single seed unless repeated; CLIP labels are interpretation, not evidence; ablation at one layer ignores compensating pathways.",
        "## Future Work\nDINOv2, ViT-L, ImageNet-1K, CLIP vision encoder, video/multimodal transformers, automated concept discovery, "
        "causal feature circuits, feature graphs, cross-layer feature tracking.",
        "## Conclusion\nConclusions must rest on all three levels: representation (feature exists), interpretation (independent evidence for what it "
        "encodes), and causality (controlled intervention beats random controls). See hypothesis verdicts above.",
        "## References\nDosovitskiy 2021 (ViT); Selvaraju 2017 (Grad-CAM); Elhage 2022 (Toy Models of Superposition); Bricken 2023 (Towards Monosemanticity); "
        "Cunningham 2023 (Sparse Autoencoders Find Highly Interpretable Features); Gao 2024 (Scaling and evaluating sparse autoencoders); "
        "Radford 2021 (CLIP); Geiger 2021 / Chan 2022 (causal abstraction / scrubbing).",
    ]
    out = PROJECT_ROOT / "reports" / f"report_{cfg.name}.md"
    out.write_text("\n\n".join(sections))
    return out
