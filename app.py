"""
app.py [CORE] -- SAE-ViT Research Dashboard  (streamlit run app.py)
Requires a finished run: python run_pipeline.py --mode demo  (or selftest / smoke)
UI/UX branch: ui/research-dashboard-polish
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PROJECT_ROOT
from context import RunContext
from features.report import feature_report
from features.retrieval import analyze_image
from utils import load_json

# ─────────────────────────── page config ─────────────────────────────────────
st.set_page_config(
    page_title="SAE-ViT · Causal Concept Discovery",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "SAE-Based Causal Concept Discovery in Vision Transformers"},
)

# ─────────────────────────── global CSS ──────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp, [data-testid="stAppViewContainer"] {
    background: #0f1120;
    color: #e8ecff;
}
[data-testid="stHeader"] { background: rgba(15, 17, 32, .96); }
.block-container { padding-top: 1.4rem; padding-bottom: 2.4rem; max-width: 1400px; }

[data-testid="metric-container"] {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 12px 16px;
    min-height: 96px;
}
[data-testid="stMetricLabel"] { font-size: 0.72rem; color: #8b9ac4; letter-spacing: .05em; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-size: 1.28rem; font-weight: 700; color: #e8ecff; line-height: 1.25; word-break: break-word; }
[data-testid="stMetricDelta"] { font-size: 0.78rem; }

.section-header {
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    border-bottom: 1px solid #2d3250;
    padding-bottom: 6px;
    margin: 28px 0 14px;
}
.page-kicker {
    color: #6b7db3;
    font-size: 0.88rem;
    line-height: 1.55;
    margin: 0 0 18px;
    overflow-wrap: anywhere;
}
.page-title {
    color: #e8ecff;
    font-size: clamp(1.6rem, 2.4vw, 2.15rem);
    font-weight: 800;
    letter-spacing: -.03em;
    line-height: 1.3;
    margin: 0 0 4px;
    overflow-wrap: anywhere;
    padding: .08em 0 .12em;
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
}
.badge-info    { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb33; }
.badge-success { background: #1a3a2a; color: #4ade80; border: 1px solid #16a34a33; }
.badge-warn    { background: #3b2a00; color: #fbbf24; border: 1px solid #d9770033; }
.badge-danger  { background: #3b0000; color: #f87171; border: 1px solid #dc262633; }

.info-card {
    background: #1a1d2e;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #15182a;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 8px 10px 6px;
    height: 100%;
}
.empty-state {
    background: #15182a;
    border: 1px dashed #3a4268;
    border-radius: 12px;
    padding: 28px 22px;
    text-align: center;
    margin: 8px 0 16px;
}
.empty-title { color: #e8ecff; font-weight: 650; font-size: 0.95rem; margin-bottom: 6px; }
.empty-detail { color: #6b7db3; font-size: 0.82rem; line-height: 1.55; }

.plot-caption {
    font-size: 0.75rem;
    color: #7b8dc4;
    margin: 4px 0 14px;
    line-height: 1.45;
    font-style: italic;
}

[data-testid="stSidebarContent"] { background: #0f1120; }
.sidebar-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #a5b4fc;
    letter-spacing: -.01em;
    margin-bottom: 4px;
}
.sidebar-sub {
    font-size: 0.7rem;
    color: #6b7db3;
    margin-bottom: 16px;
    line-height: 1.5;
}

.stTabs [data-baseweb="tab-list"] { gap: 2px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.8rem;
    font-weight: 500;
    padding: 8px 12px;
    height: auto;
    white-space: nowrap;
}
[data-testid="stExpander"] summary { font-size: 0.82rem; color: #8b9ac4; }
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
[data-testid="stImage"] img {
    width: 100%;
    height: auto;
    object-fit: contain;
    border-radius: 8px;
    border: 1px solid #2d3250;
}
[data-testid="caption"] { font-size: 0.72rem; color: #6b7db3; text-align: center; line-height: 1.35; }
hr { border-color: #2d3250 !important; margin: 20px 0; }
[data-testid="stAlert"] { border-radius: 8px; font-size: 0.82rem; }
div[data-testid="stDownloadButton"] button {
    width: 100%;
    white-space: normal;
    height: auto;
    min-height: 2.5rem;
    font-size: 0.78rem;
    line-height: 1.3;
    text-align: left;
    padding: 0.45rem 0.7rem;
}
</style>
""", unsafe_allow_html=True)

EXPS = PROJECT_ROOT / "experiments"
_PLOT_H = 400
_PLOT_H_LG = 480
_PLOT_COLORWAY = ["#6366f1", "#f472b6", "#34d399", "#fbbf24", "#60a5fa", "#a78bfa", "#fb7185", "#2dd4bf"]
_PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "sae_vit_chart",
        "scale": 2,
    },
}


# ─────────────────────────── helpers ─────────────────────────────────────────
def read_csv(p) -> Optional[pd.DataFrame]:
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def section(label: str):
    st.markdown(f'<p class="section-header">{label}</p>', unsafe_allow_html=True)


def caption(text: str):
    st.markdown(f'<p class="plot-caption">{text}</p>', unsafe_allow_html=True)


def badge(text: str, kind: str = "info"):
    st.markdown(f'<span class="badge badge-{kind}">{text}</span>', unsafe_allow_html=True)


def empty_state(title: str, detail: str = ""):
    extra = f'<div class="empty-detail">{detail}</div>' if detail else ""
    st.markdown(
        f'<div class="empty-state"><div class="empty-title">{title}</div>{extra}</div>',
        unsafe_allow_html=True,
    )


def safe_json(path) -> dict:
    try:
        return load_json(path)
    except Exception:
        return {}


def pct_bar(val: float, lo: float = 0, hi: float = 1, color: str = "#6366f1") -> str:
    w = int(100 * max(0, min(1, (val - lo) / max(hi - lo, 1e-9))))
    return (f'<div style="background:#1e2130;border-radius:4px;height:6px;width:100%">'
            f'<div style="background:{color};width:{w}%;height:6px;border-radius:4px"></div></div>')


def plotly_defaults(fig: go.Figure, height: int = _PLOT_H) -> go.Figure:
    axis = dict(
        gridcolor="#1e2130",
        linecolor="#2d3250",
        zerolinecolor="#2d3250",
        title_font=dict(size=12, color="#c7d0ef"),
        tickfont=dict(size=11, color="#9aa8d6"),
        automargin=True,
    )
    fig.update_layout(
        height=height,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f1120",
        font=dict(color="#c7d0ef", size=12, family="Inter,Segoe UI,sans-serif"),
        title=dict(font=dict(size=14, color="#e8ecff"), x=0, xanchor="left", y=0.98),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=11),
        ),
        margin=dict(l=56, r=28, t=84, b=64),
        colorway=_PLOT_COLORWAY,
        hoverlabel=dict(bgcolor="#1e2130", font_size=12, font_family="Inter,Segoe UI,sans-serif"),
        bargap=0.22,
        xaxis=axis,
        yaxis=axis,
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def show_chart(fig: go.Figure, height: int = _PLOT_H):
    st.plotly_chart(
        plotly_defaults(fig, height),
        use_container_width=True,
        config=_PLOTLY_CONFIG,
    )


def show_table(df: Optional[pd.DataFrame], height: int = 360, empty_title: str = "No rows to display"):
    if df is None or df.empty:
        empty_state(empty_title, "This artifact is missing or empty for the selected run.")
        return
    column_config = {}
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            column_config[col] = st.column_config.NumberColumn(col.replace("_", " "), format="%.4f")
        else:
            column_config[col] = st.column_config.Column(str(col).replace("_", " "))
    n = len(df)
    table_h = min(max(140, 38 + 32 * min(n, 12)), height)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=table_h,
        column_config=column_config,
    )


def show_png_grid(paths, ncols: int = 3, meta: Optional[dict] = None):
    paths = list(paths)
    if not paths:
        empty_state("No saved plots found", "Pipeline plot PNGs will appear here after a completed run.")
        return
    meta = meta or {}
    for i in range(0, len(paths), ncols):
        row = paths[i:i + ncols]
        cols = st.columns(ncols)
        for col, p in zip(cols, row):
            title, tip = meta.get(p.name, (p.stem.replace("_", " ").title(), ""))
            with col:
                with st.container(border=True):
                    st.image(str(p), caption=title, width="stretch")
                    if tip:
                        caption(tip)


# ─────────────────────────── sidebar ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">⚗️ SAE-ViT Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">Causal Concept Discovery in Vision Transformers · Mechanistic Interpretability</p>', unsafe_allow_html=True)

    exps = sorted(
        (d.name for d in EXPS.iterdir() if d.is_dir()),
        key=lambda name: ("selftest" in name.lower(), name.lower()),
    ) if EXPS.exists() else []
    if not exps:
        st.error("No experiments found. Run `python run_pipeline.py --mode demo` first.")
        st.stop()

    exp = st.selectbox("📁 Experiment", exps, help="Select an experiment run")
    runs = sorted(
        d.name for d in (EXPS / exp).iterdir()
        if d.is_dir() and (d / "checkpoint.pt").exists()
    )
    if not runs:
        st.error("This experiment has no trained SAE runs.")
        st.stop()

    run = st.selectbox("🔬 Run  (layer / SAE type)", runs,
                        help="Each run = one layer × SAE architecture combination")
    run_dir = EXPS / exp / run
    causal_dir = run_dir / "causal"
    disc_dir = run_dir / "discovery"

    st.divider()

    st.caption("Branch: `ui/research-dashboard-polish`")
    st.caption("Never retrained — all results from saved checkpoints.")


# ─────────────────────────── cached context load ─────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading run context (ViT + SAE + activation store)…")
def get_ctx(run_dir: str) -> RunContext:
    return RunContext.from_run_dir(run_dir)


ctx = get_ctx(str(run_dir))
cfg = ctx.cfg
summ_ok = (disc_dir / "feature_summary.csv").exists()
cand = read_csv(causal_dir / "candidate_causal_summary.csv")
default_feat = int(cand["feature_id"].iloc[0]) if cand is not None and len(cand) else 0
res = safe_json(run_dir / "results.json")
m = res.get("metrics", {})
layer_sum = safe_json(run_dir / "layer_summary.json")
# ─────────────────────────── intervention helpers ────────────────────────────
def need_causal_images(fid, limit=300):
    pos = ctx.splits["causal"][:limit]
    act = ctx.feature_activations(fid, pos).max(1) > 0
    return pos[act]


def run_intervention(hidden, ids, alphas, mode, local):
    eng = ctx.engine()
    h = hidden.to(ctx.device)
    base = eng.baseline(h).cpu().numpy()[0]
    lg, info = eng.run(h, ids, alphas, mode, local)
    new = lg.cpu().numpy()[0]
    return base, new, info


def show_result(base, new, info):
    names, y = ctx.vit_class_names, int(base.argmax())
    sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    col = st.columns(5)
    col[0].metric("Target class", names[y])
    col[1].metric("Logit  before → after", f"{base[y]:.2f} → {new[y]:.2f}",
                  delta=f"{new[y]-base[y]:+.3f}")
    col[2].metric("Probability  before → after",
                  f"{sm(base)[y]:.3f} → {sm(new)[y]:.3f}",
                  delta=f"{sm(new)[y]-sm(base)[y]:+.3f}")
    col[3].metric("Prediction flip", "✅ YES" if new.argmax() != y else "—  no")
    col[4].metric("Max norm ratio", f"{info['max_norm_ratio']:.2f}",
                  help="Ratio of intervened to baseline activation norm. >5 may be unreliable.")

    if info["flagged"]:
        st.warning("⚠️ Safety flag: extreme activation norm or NaN/Inf after intervention.", icon="⚠️")
    if info["clamped"]:
        st.info("Strength was clamped to the configured safety limit.", icon="ℹ️")

    with st.expander("📋 Top-5 class logits before / after"):
        t = pd.DataFrame({
            "class (before)": [names[i] for i in np.argsort(-base)[:5]],
            "logit before": np.sort(base)[::-1][:5].round(3),
            "class (after)": [names[i] for i in np.argsort(-new)[:5]],
            "logit after": np.sort(new)[::-1][:5].round(3),
        })
        show_table(t, height=220)


def pick_image(key, fid):
    sources = ["Held-out causal-eval"]
    if ctx.supports_uploaded_image_analysis:
        sources.append("Upload your own")
    src = st.radio("Image source", sources, key=key + "src", horizontal=True)
    if src == "Upload your own":
        up = st.file_uploader("Image (PNG / JPG)", type=["png", "jpg", "jpeg"], key=key + "up")
        if up is None:
            return None, None
        from PIL import Image
        pil = Image.open(up).convert("RGB")
        return pil, analyze_image(ctx, pil)["hidden"]
    pos_list = need_causal_images(fid)
    if len(pos_list) == 0:
        st.info("This feature has no active images in the held-out causal-eval split.", icon="ℹ️")
        return None, None
    pos = int(st.selectbox("Store position", pos_list, key=key + "pos",
                            help="Images in the held-out causal-eval split where this feature is active"))
    return ctx.image_pil(pos), ctx.hidden([pos])


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER  (above tabs)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-title">SAE-ViT · Causal Concept Discovery</div>
<div class="page-kicker">
  Mechanistic interpretability: Sparse Autoencoder feature discovery + causal interventions in Vision Transformers
  &nbsp;|&nbsp; <b style="color:#a5b4fc">{exp}</b> &nbsp;·&nbsp; <b style="color:#a5b4fc">{run}</b>
</div>
""".format(exp=exp, run=run), unsafe_allow_html=True)

tabs = st.tabs([
    "📊 Overview", "🖼 Image Explorer", "🔬 Feature Explorer",
    "⚡ Causal Intervention", "🎲 Random Controls",
    "🎯 Feature Steering", "🔗 Feature Interaction",
    "📐 Layer Comparison", "🗂 SAE Comparison", "📤 Export",
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:

    # ── top-level key metrics row ─────────────────────────────────────────────
    section("Experiment Configuration")
    c = st.columns(4)
    c[0].metric("Model", cfg.model.name, help="timm Vision Transformer backbone")
    c[1].metric("Dataset", f"{cfg.data.dataset}", f"{cfg.data.num_images:,} imgs",
                help="Number of images processed by the full pipeline")
    c[2].metric("Layer (block index)", str(cfg.layer),
                help="ViT encoder block whose patch activations are decomposed")
    sae_label = f"{cfg.sae.sae_type.upper()}  width={cfg.sae.width}"
    if cfg.sae.sae_type == "topk":
        sae_label += f"  k={cfg.sae.k}"
    c[3].metric("SAE Architecture", sae_label)

    section("SAE Reconstruction Quality")
    ev = m.get("explained_variance", float("nan"))
    c = st.columns(4)
    c[0].metric("Explained Variance (EV)", f"{ev:.3f}",
                help="Fraction of activation variance explained by the SAE reconstruction. Target ≥ 0.70")
    c[1].metric("Reconstruction MSE", f"{m.get('mse', float('nan')):.4f}",
                help="Mean squared error between original and reconstructed patch activations")
    c[2].metric("L0  (avg active features)", f"{m.get('l0', float('nan')):.1f}",
                help="Average number of non-zero SAE features per token (sparsity)")
    c[3].metric("Dead features (val %)", f"{m.get('dead_pct_eval', float('nan')):.1f}%",
                help="Percentage of features that never activate on the validation split")

    st.markdown(
        f'<div style="margin-top:6px">{pct_bar(ev, 0, 1, "#6366f1")}'
        f'<span style="font-size:0.68rem;color:#6b7db3">&nbsp;Explained Variance: {ev:.1%}'
        f"</span></div>",
        unsafe_allow_html=True,
    )

    # ── feature discovery summary ─────────────────────────────────────────────
    if layer_sum:
        section("Feature Discovery Summary")
        c = st.columns(5)
        c[0].metric("Alive features", f"{layer_sum.get('n_alive', '?')} / {cfg.sae.width}")
        c[1].metric("Median purity", f"{layer_sum.get('median_purity', float('nan')):.3f}",
                    help="Fraction of top activating images from the dominant class. Higher = more monosemantic.")
        c[2].metric("Highly specific %", f"{layer_sum.get('pct_highly_specific', float('nan')):.1f}%",
                    help="Features with interpretability score ≥ threshold")
        c[3].metric("Polysemantic %", f"{layer_sum.get('pct_polysemantic', float('nan')):.1f}%",
                    help="Features associated with many unrelated classes")
        c[4].metric("Median SCS", f"{layer_sum.get('median_spatial_conc10', float('nan')):.3f}",
                    help="Spatial Concentration Score: fraction of feature mass in the top-10% of patches")

        c = st.columns(4)
        c[0].metric("SAE median purity", f"{layer_sum.get('median_purity', float('nan')):.3f}")
        c[1].metric("Neuron median purity", f"{layer_sum.get('neuron_median_purity', float('nan')):.3f}",
                    help="Baseline: purity of raw ViT neurons without SAE decomposition")
        c[2].metric("SAE median interp. score", f"{layer_sum.get('median_interp_score', float('nan')):.3f}")
        c[3].metric("Neuron median interp. score", f"{layer_sum.get('neuron_median_interp_score', float('nan')):.3f}")
        caption("SAE features should show higher purity and interpretability scores than raw neurons — "
                "this validates the SAE decomposition.")

    # ── causal summary ────────────────────────────────────────────────────────
    if layer_sum and layer_sum.get("n_candidates"):
        section("Causal Evaluation Summary")
        c = st.columns(4)
        c[0].metric("Candidate features", layer_sum.get("n_candidates", "?"),
                    help="Features selected from the validation split using non-causal criteria")
        c[1].metric("Mean ablation logit drop", f"{layer_sum.get('mean_ablation_logit_effect', float('nan')):.5f}")
        c[2].metric("Mean random control effect", f"{layer_sum.get('mean_random_effect', float('nan')):.5f}")
        c[3].metric("Mean amp. Δlogit", f"{layer_sum.get('mean_amp_delta_logit', float('nan')):.5f}")
        caption("Ablation logit drop measures the causal influence of each feature on model predictions. "
                "Candidates should exceed random-control baselines.")

    # ── saved plots grid ──────────────────────────────────────────────────────
    plots_dir = run_dir / "plots"
    all_plots = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []
    if all_plots:
        section("Saved Pipeline Plots")

        PLOT_META = {
            "sae_training.png":             ("SAE Training Curves", "Loss, MSE, and L0 sparsity across training epochs."),
            "feature_quality.png":          ("Feature Quality Overview", "Distribution of purity, spatial concentration, and interpretability scores across all features."),
            "feature_clusters.png":         ("Feature Clusters", "UMAP or PCA projection of SAE decoder directions, coloured by dominant concept."),
            "sae_vs_neurons.png":           ("SAE vs. Neurons", "Comparison of SAE features against raw ViT neurons: purity, entropy, and interpretability."),
            "causal_effect_distribution.png": ("Causal Effect Distribution", "Distribution of ablation logit drops for all candidate features vs. random controls."),
            "random_controls.png":          ("Random Controls", "Candidate feature effects plotted against frequency-matched random feature baselines."),
            "ablation_amplification.png":   ("Ablation vs. Amplification", "Side-by-side comparison of ablation and amplification effects for each candidate feature."),
            "dose_response.png":            ("Dose-Response Curves", "Logit effect as a function of intervention strength α for each feature."),
            "steering_curves.png":          ("Steering Curves", "Multi-step feature steering: how prediction changes as α is swept from suppression to amplification."),
            "local_vs_global.png":          ("Local vs. Global Intervention", "Effect of spatially-masked (local) vs. full-image (global) ablation for each feature."),
            "feature_interaction.png":      ("Feature Interaction", "Synergy and redundancy between pairs of candidate features."),
        }

        gcam_plots = [p for p in all_plots if "gradcam" in p.name]
        regular_plots = [p for p in all_plots if "gradcam" not in p.name]
        show_png_grid(regular_plots, ncols=3, meta=PLOT_META)

        if gcam_plots:
            section("Grad-CAM vs. SAE Feature Comparison")
            caption("Side-by-side comparison of gradient-based (Grad-CAM) and SAE-based concept localisation.")
            show_png_grid(gcam_plots, ncols=3)
    else:
        empty_state(
            "No saved pipeline plots yet",
            "PNG figures from `run_pipeline.py` will appear in a 3-column gallery here.",
        )

    # ── config expander ───────────────────────────────────────────────────────
    with st.expander("⚙️ Experiment configuration (click to expand)"):
        st.json(cfg.to_dict())

    with st.expander("📄 Layer summary JSON"):
        st.json(layer_sum)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — IMAGE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section("Upload an Image — Inspect Active SAE Features")
    st.markdown(
        "Upload an image to inspect its ImageNet prediction and active SAE features, annotated with "
        "information from the selected run's saved discovery artifacts.",
    )

    if not summ_ok:
        st.warning("Feature discovery data not found. Run `python discover.py` first.", icon="⚠️")
        empty_state("Image explorer unavailable", "Feature discovery artifacts are required to inspect active SAE features.")
    else:
        up = None
        if ctx.supports_uploaded_image_analysis:
            up = st.file_uploader("Upload image (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"], key="explore")
        if not ctx.supports_uploaded_image_analysis:
            st.caption(
                "Uploaded-image analysis is available for runs with a pretrained ImageNet classifier "
                "and a non-synthetic SAE training dataset."
            )
        elif up is None:
            empty_state("No image uploaded", "Choose a PNG or JPG above to inspect ViT predictions and active SAE features.")
        else:
            from PIL import Image as PilImage
            pil = PilImage.open(up).convert("RGB")
            out = analyze_image(ctx, pil)
            z = out["logits"]
            prob = np.exp(z - z.max()); prob /= prob.sum()
            top_idx = np.argsort(-z)[:5]

            col_img, col_pred = st.columns([1, 2])
            with col_img:
                st.image(pil, caption="Uploaded image", width="stretch")

            with col_pred:
                section("Model Prediction (ImageNet head)")
                pred_df = pd.DataFrame({
                    "Rank": range(1, 6),
                    "Class": [ctx.vit_class_names[i] for i in top_idx],
                    "Logit": z[top_idx].round(3),
                    "Confidence": (prob[top_idx] * 100).round(2),
                })
                show_table(pred_df, height=240)

                top_class = ctx.vit_class_names[top_idx[0]]
                top_conf = prob[top_idx[0]] * 100
                st.metric("Top-1 prediction", top_class, f"{top_conf:.1f}% confidence")

            st.divider()
            section("Active SAE Features — Layer " + str(cfg.layer))
            caption(
                f"Features with highest maximum activation in the SAE at block {cfg.layer}. "
                "Feature metrics and any labels shown here come from saved discovery artifacts."
            )

            feat_df = pd.DataFrame(out["features"])
            if not feat_df.empty:
                show_table(feat_df, height=320)
            else:
                empty_state("No active SAE features", "This image did not produce tabulated feature activations.")

            disc_db = read_csv(disc_dir / "feature_top_patches.csv.gz")
            feats_to_show = out["features"][:4]
            if feats_to_show and disc_db is not None:
                section("Top Activating Example Images (per Feature)")
                caption("For each active feature, the most strongly activating training images from the dataset.")
                for f in feats_to_show:
                    fid_local = f["feature_id"]
                    concept = f.get("concept") or f.get("top_class", f"Feature {fid_local}")
                    purity_str = f" · purity {f['purity']:.2f}" if "purity" in f else ""
                    scs_str = f" · SCS {f['spatial_conc10']:.2f}" if "spatial_conc10" in f else ""
                    st.markdown(
                        f"**Feature {fid_local}** — *{concept}*{purity_str}{scs_str}  "
                        f"· max activation: `{f['max_activation']:.3f}`"
                    )
                    ex = disc_db[disc_db.feature_id == fid_local].head(6)
                    if len(ex):
                        img_cols = st.columns(6)
                        for icol, (_, r) in zip(img_cols, ex.iterrows()):
                            try:
                                img = ctx.image_pil(int(r["position"]))
                                icol.image(
                                    img,
                                    caption=f"{r['class']} · {r['activation']:.2f}",
                                    width="stretch",
                                )
                            except Exception:
                                icol.caption("(unavailable)")
                    else:
                        empty_state("No example patches", f"No stored top patches for feature {fid_local}.")
                    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — FEATURE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    if not summ_ok:
        st.warning("Feature discovery data not found. Run `python discover.py` first.", icon="⚠️")
        empty_state("Feature explorer unavailable", "Run discovery so feature summaries, reports, and heatmaps can load.")
    else:
        S = ctx.summary()
        alive_feats = S.index[~S["dead"]].tolist()
    
        col_ctrl, col_info = st.columns([1, 2])
        with col_ctrl:
            section("Feature Selector")
            fid = int(st.number_input(
                "Feature ID", 0, int(ctx.sae.width) - 1, default_feat, key="fx",
                help=f"Select feature 0–{ctx.sae.width - 1}. Dead features show a warning."
            ))
            if alive_feats:
                st.caption(f"{len(alive_feats)} / {cfg.sae.width} features alive in this run.")
    
        row = S.loc[fid]
    
        with col_info:
            section("Feature Status")
            if row["dead"]:
                st.error("❌ This feature is **dead** — never active on the validation set.", icon="❌")
            elif row["freq"] < 1e-4:
                st.warning("⚠️ This feature activates almost nowhere (very sparse).", icon="⚠️")
            elif row["freq"] > 0.2:
                st.warning("⚠️ This feature is very dense — may be polysemantic.", icon="⚠️")
            else:
                st.success(f"✅ Feature {fid} is alive and healthy.", icon="✅")

            if row["dead"]:
                empty_state("No live metrics for this feature", "Dead features have no quality metrics, heatmaps, or class-mass plots.")
            if not row["dead"]:
                # ── core feature metrics ───────────────────────────────────────────────
                section("Feature Quality Metrics")
                c = st.columns(5)
                c[0].metric("Purity", f"{row['purity']:.3f}",
                            help="Fraction of top activating patches from the dominant class. Closer to 1 = monosemantic.")
                c[1].metric("Top class", str(row["top_class_name"]))
                c[2].metric("SCS  (spatial conc.)", f"{row['spatial_conc10']:.3f}",
                            help="Fraction of feature activation mass in the top-10% of spatial patches.")
                c[3].metric("Frequency", f"{row['freq']:.4f}",
                            help="Fraction of validation images where this feature is non-zero.")
                c[4].metric("Specificity", str(row.get("specificity", "—")))
        
                c2 = st.columns(4)
                c2[0].metric("Mean activation", f"{row['mean_act']:.3f}")
                c2[1].metric("Max activation", f"{row['max_act']:.3f}")
                c2[2].metric("Top-patch purity", f"{row['top_patch_purity']:.3f}",
                             help="Purity measured over the single most strongly activating patches.")
                c2[3].metric("Class coverage", f"{row['class_coverage']:.3f}",
                             help="Fraction of all classes that this feature is active on.")
        
                # ── CLIP label ─────────────────────────────────────────────────────────
                if "clip_concept" in row and isinstance(row["clip_concept"], str) and row["clip_concept"]:
                    st.markdown(
                        f'<div class="info-card">🤖 <b>CLIP concept</b>: '
                        f'<span style="color:#a5b4fc;font-size:1.05rem">{row["clip_concept"]}</span>'
                        f' &nbsp;·&nbsp; score <code>{row["clip_score"]:.3f}</code>'
                        f' &nbsp;·&nbsp; entropy <code>{row["clip_entropy"]:.2f}</code>'
                        f'<br><span style="color:#6b7db3;font-size:0.75rem">Top-3: {row["clip_top3"]}</span></div>',
                        unsafe_allow_html=True,
                    )
        
                # ── causal stats if candidate ──────────────────────────────────────────
                if cand is not None and fid in set(cand.feature_id):
                    cr = cand[cand.feature_id == fid].iloc[0]
                    section("Causal Evidence")
                    c = st.columns(4)
                    c[0].metric("Mean ablation logit drop", f"{cr['abl_mean']:.4f}")
                    c[1].metric("95% CI", f"[{cr['abl_ci_lo']:.4f}, {cr['abl_ci_hi']:.4f}]")
                    c[2].metric("Flip rate", f"{cr['abl_flip_rate']:.3f}",
                                help="Fraction of images where ablation changes the top-1 prediction.")
                    c[3].metric("Causal rank", str(int(cr["causal_rank"])))
        
                # ── report image (top activating patches + hist + spatial map) ─────────
                section("Feature Report")
                caption("Top 12 maximally activating image patches (red box), activation histogram, "
                        "class distribution, and mean spatial activation map.")
                with st.spinner("Generating feature report…"):
                    rep = feature_report(ctx, fid)
                st.image(rep["report"], width="stretch")
        
                # ── heatmaps row ───────────────────────────────────────────────────────
                hm = rep.get("heatmaps", {})
                if hm:
                    section("Patch Activation Heatmaps (Top-1 Image)")
                    caption("Raw activation map, normalized activation, image overlay, binary mask, and bounding box "
                            "for the top-activating image of this feature.")
                    HM_LABELS = {
                        "raw": ("Raw activation map", "Unthresholded per-patch activation values interpolated to image resolution."),
                        "normalized": ("Normalized map", "Activation map min-max normalized to [0,1]."),
                        "overlay": ("Overlay", "Normalized activation map blended on the original image."),
                        "mask": ("Binary mask", "Patches above the threshold (default 90th percentile)."),
                        "bbox": ("Bounding box", "Tight bounding box around the thresholded activation region."),
                    }
                hm_cols = st.columns(5, gap="small")
                for col, (k, (title, tip)) in zip(hm_cols, HM_LABELS.items()):
                    if k in hm:
                        with col:
                            st.image(hm[k], caption=title, width="stretch")
                            st.caption(tip)
        
                # ── class distribution bar chart ───────────────────────────────────────
                section("Class Activation Distribution")
                caption("Probability that each ImageNet class is the most active when this feature fires. "
                        "A peaked distribution indicates a monosemantic, class-specific feature.")
                cls = ctx.disc["class_mass"][:, fid]
                pc = cls / max(cls.sum(), 1e-12)
                top10_idx = np.argsort(-pc)[:10]
                fig = go.Figure(go.Bar(
                    x=[ctx.store.class_names[i] for i in top10_idx],
                    y=pc[top10_idx],
                    marker_color="#6366f1",
                    text=[f"{v:.3f}" for v in pc[top10_idx]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title=f"P(class | Feature {fid} active) — top 10",
                    xaxis_title="Class", yaxis_title="Probability",
                )
                show_chart(fig, _PLOT_H)
        
                # ── saved feature-level plots ──────────────────────────────────────────
                feats_plots_dir = run_dir / "plots" / "features"
                feat_report_png = feats_plots_dir / f"feature_{fid}_report.png"
                if feat_report_png.exists():
                    with st.expander(f"📁 Cached feature {fid} report (saved during pipeline run)"):
                        st.image(str(feat_report_png), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — CAUSAL INTERVENTION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    if not summ_ok:
        st.warning("Run `python discover.py` first to enable causal interventions.", icon="⚠️")
        empty_state("Causal intervention unavailable", "Discovery outputs are required before live interventions can run.")
    else:
        section("Intervention Controls")
        col_f, col_k, col_s, col_sc, col_p = st.columns(5)
        fid = int(col_f.number_input("Feature ID", 0, int(ctx.sae.width) - 1,
                                      default_feat, key="ci_f"))
        kind = col_k.selectbox("Intervention type",
                                ["ablation (set to 0)", "suppression (×α)", "amplification (×α)", "steering (add α·dec)"],
                                help="Ablation removes the feature entirely; steering shifts activations along its decoder direction.")
        default_alpha = {"ablation (set to 0)": 0.0, "suppression (×α)": 0.5,
                         "amplification (×α)": 2.0, "steering (add α·dec)": 2.0}[kind]
        strength = col_s.number_input("Strength α", -10.0, 10.0, default_alpha,
                                       key="ci_s", step=0.5,
                                       help="Multiplier or steering coefficient. α=0 = ablation.")
        scope = col_sc.selectbox("Scope", ["global", "local"],
                                  help="Global: modify all patches. Local: modify only the most active spatial region.")
        pct = col_p.selectbox("Local %ile", [50, 75, 90, 95], index=2) if scope == "local" else None

        st.divider()
        pil, hidden = pick_image("ci", fid)
        if hidden is not None:
            col_img, col_res = st.columns([1, 2])
            with col_img:
                st.image(pil, caption="Selected image", width="stretch")

            with col_res:
                mode = "steer" if kind.startswith("steer") else "scale"
                alpha_val = 0.0 if kind.startswith("abl") else strength
                with st.spinner("Running intervention…"):
                    base, new, info = run_intervention(hidden, [fid], [alpha_val], mode, pct)
                show_result(base, new, info)

            # logit bar chart for visual comparison
            section("Logit Comparison — Before vs After")
            caption("Top-10 class logits before and after the intervention. "
                    "A meaningful causal feature should produce a visible shift for its associated class.")
            names = ctx.vit_class_names
            top_before = np.argsort(-base)[:10]
            top_all = np.unique(np.concatenate([top_before, np.argsort(-new)[:10]]))
            classes = [names[i] for i in top_all]
            fig = go.Figure()
            fig.add_bar(name="Before", x=classes, y=base[top_all], marker_color="#6366f1")
            fig.add_bar(name="After", x=classes, y=new[top_all], marker_color="#f472b6")
            fig.update_layout(barmode="group", xaxis_title="Class", yaxis_title="Logit",
                              title=f"Feature {fid} {kind} — Logit shift")
            show_chart(fig, _PLOT_H_LG)
        else:
            empty_state("Select an image to intervene", "Use a held-out causal-eval image or upload one to see before/after logits.")

        # ── dose-response from saved data ─────────────────────────────────────
        dr_path = causal_dir / "dose_response.json"
        if dr_path.exists():
            st.divider()
            section("Pre-computed Dose-Response Curves")
            caption("Logit effect as a function of intervention strength α, evaluated on held-out images. "
                    "These are cached results from the pipeline evaluation — no recomputation required.")
            dr = safe_json(dr_path)
            feats_with_dr = [int(k) for k in dr.keys()]
            if feats_with_dr:
                dr_sel = st.multiselect("Show dose-response for features:", feats_with_dr,
                                         default=feats_with_dr[:min(3, len(feats_with_dr))],
                                         key="dr_sel")
                if dr_sel:
                    fig = go.Figure()
                    colors = px.colors.qualitative.Plotly
                    for i, f_dr in enumerate(dr_sel):
                        d = dr.get(str(f_dr), {})
                        if d and "alphas" in d and "mean_effects" in d:
                            concept = d.get("concept", f"Feature {f_dr}")
                            fig.add_scatter(
                                x=d["alphas"], y=d["mean_effects"],
                                mode="lines+markers", name=f"F{f_dr}: {concept}",
                                line=dict(color=colors[i % len(colors)], width=2),
                                marker=dict(size=6),
                            )
                    fig.add_hline(y=0, line_dash="dot", line_color="#6b7db3",
                                  annotation_text="no effect", annotation_position="bottom right")
                    fig.update_layout(
                        title="Dose-Response: Mean Logit Effect vs. Intervention Strength α",
                        xaxis_title="α (intervention strength)",
                        yaxis_title="Mean logit effect",
                    )
                    show_chart(fig, _PLOT_H_LG)
                else:
                    empty_state("No dose-response series selected", "Choose one or more features above to plot cached α curves.")

            # show saved dose-response plot
            dr_plot = run_dir / "plots" / "dose_response.png"
            if dr_plot.exists():
                with st.expander("📊 Saved dose-response plot (pipeline output)"):
                    st.image(str(dr_plot), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — RANDOM CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    rc_p = causal_dir / "random_controls.json"
    if not rc_p.exists() or cand is None:
        st.warning("Run `python evaluate.py` first to generate random control results.", icon="⚠️")
        empty_state("Random control results missing", "This tab needs `random_controls.json` and the candidate causal summary.")
    else:
        section("Statistical Controls — Candidate vs. Random Features")
        st.markdown(
            "Each candidate feature is compared against frequency-matched **random** SAE features "
            "to assess whether its causal effect is statistically distinguishable from noise. "
            "A causal feature should lie well outside the random control distribution."
        )

        rc = safe_json(rc_p)
        feats_rc = sorted({int(k.split("_")[0]) for k in rc})
        f_rc = st.selectbox("Feature", feats_rc, key="rc_f",
                             help="Select a candidate feature to inspect its control comparison.")

        for cond in ("ablation", "amplification"):
            d = rc.get(f"{f_rc}_{cond}")
            if not d:
                continue

            section(f"{cond.title()} — Effect vs. Random Controls")

            # statistical summary row
            lo, hi = d["rand_ci95"]
            p_emp = d["p_empirical"]
            p_wilcox = d.get("paired_p_wilcoxon", float("nan"))
            dz = d.get("paired_dz", float("nan"))
            sig = p_emp < 0.05
            c = st.columns(5)
            c[0].metric("Candidate effect", f"{d['cand_mean']:.4f}")
            c[1].metric("Random mean", f"{d['rand_mean']:.4f}")
            c[2].metric("Random 95% CI", f"[{lo:.4f}, {hi:.4f}]")
            c[3].metric("Empirical p", f"{p_emp:.4f}",
                        delta="significant" if sig else "not significant",
                        delta_color="normal" if sig else "inverse")
            c[4].metric("Cohen's d_z", f"{dz:.2f}",
                        help="Paired effect size. |d_z| > 0.5 = medium, > 0.8 = large")

            if not sig:
                st.warning(f"⚠️ Feature {f_rc} {cond}: causal effect is **not** distinguishable from "
                           f"random controls (p_emp = {p_emp:.3f} ≥ 0.05).", icon="⚠️")
            else:
                st.success(f"✅ Feature {f_rc} {cond}: causal effect is statistically significant "
                           f"(p_emp = {p_emp:.4f} < 0.05).", icon="✅")

            # box plot
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=d["own_means"], name="Random features (freq-matched)",
                boxpoints="all", jitter=0.4, pointpos=-1.5,
                marker=dict(color="#6b7db3", size=5, opacity=0.7),
                line=dict(color="#4b5563"),
                fillcolor="#1e2130",
            ))
            fig.add_trace(go.Scatter(
                x=["Random features (freq-matched)"], y=[d["cand_mean"]],
                mode="markers", name="Candidate feature",
                marker=dict(color="#f472b6" if cond == "ablation" else "#34d399",
                            size=16, symbol="star", line=dict(color="white", width=1)),
            ))
            fig.add_shape(type="line", y0=lo, y1=lo, x0=-0.5, x1=0.5,
                          line=dict(color="#fbbf24", dash="dash", width=1))
            fig.add_shape(type="line", y0=hi, y1=hi, x0=-0.5, x1=0.5,
                          line=dict(color="#fbbf24", dash="dash", width=1))
            fig.update_layout(
                title=f"Feature {f_rc} · {cond.title()} · Effect vs. Frequency-Matched Random Controls",
                yaxis_title="Mean logit effect",
            )
            show_chart(fig, _PLOT_H)
            caption("Yellow dashed lines = 95% CI of the random-control distribution. "
                    "The candidate star should lie outside these bounds to be considered causally significant.")

            with st.expander(f"📋 Statistical details — {cond}"):
                st.markdown(f"""
| Statistic | Value |
|---|---|
| Candidate mean effect | `{d['cand_mean']:.6f}` |
| Random mean effect | `{d['rand_mean']:.6f}` |
| Random 95% CI | `[{lo:.6f}, {hi:.6f}]` |
| Empirical p-value | `{p_emp:.6f}` |
| Paired Wilcoxon p | `{p_wilcox:.6g}` |
| Cohen's d_z | `{dz:.4f}` |
""")

        st.divider()
        section("All Candidate Features Summary")
        caption("All features selected as causal candidates (selection from validation split only, "
                "non-causal criteria), with their statistical test results.")
        show_table(cand, height=420, empty_title="No candidate causal summary rows")

        # show saved random controls plot
        rc_plot = run_dir / "plots" / "random_controls.png"
        if rc_plot.exists():
            with st.expander("📊 Saved random controls plot (pipeline output)"):
                st.image(str(rc_plot), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — FEATURE STEERING
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    if not summ_ok:
        st.warning("Run `python discover.py` first.", icon="⚠️")
        empty_state("Steering unavailable", "Feature discovery is required to list alive features for steering.")
    else:
        section("Interactive Feature Steering")
        st.markdown(
            "Steer the model by amplifying or suppressing multiple SAE features simultaneously. "
            "This demonstrates directional activation patching along the SAE decoder direction."
        )
        alive = S.index[~S["dead"]].tolist()
        sel = st.multiselect(
            "Features to steer (multi-select)",
            alive, default=[default_feat] if default_feat in alive else alive[:1],
            help="Select one or more alive features. Their decoder directions will be combined.",
        )
        alpha = st.slider("Steering α (units of normalised decoder direction)", -10.0, 10.0, 2.0, 0.5,
                           help="Positive = amplify; negative = suppress. α=0 = no intervention.")

        if sel:
            pil, hidden = pick_image("st", sel[0])
            if hidden is not None:
                col_img, col_res = st.columns([1, 2])
                with col_img:
                    st.image(pil, caption="Selected image", width="stretch")
                with col_res:
                    with st.spinner("Steering…"):
                        base, new, info = run_intervention(hidden, sel, [alpha] * len(sel), "steer", None)
                    show_result(base, new, info)
            else:
                empty_state("Select an image to steer", "Pick a held-out image or upload one to apply the selected decoder directions.")
        else:
            empty_state("No features selected", "Choose at least one alive feature to steer.")

        # ── saved steering curves ──────────────────────────────────────────────
        stf = read_csv(causal_dir / "steering.csv")
        if stf is not None:
            st.divider()
            section("Pre-computed Steering Curves (Held-out Images)")
            caption("Mean logit delta as α is swept from suppression to amplification, "
                    "evaluated on held-out images from the causal-eval split.")
            single = stf[stf.type == "single"]
            if not single.empty:
                fig = px.line(
                    single, x="alpha", y="mean_delta_logit",
                    color=single["feature_id"].astype(str),
                    markers=True,
                    labels={"color": "Feature", "mean_delta_logit": "Mean Δlogit", "alpha": "α"},
                    title="Single-feature steering curves",
                )
                show_chart(fig, _PLOT_H_LG)
            else:
                empty_state("No single-feature steering curves", "The steering CSV has no `type == single` rows for this run.")

            multi = stf[stf.type == "multi"] if (stf.type == "multi").any() else pd.DataFrame()
            if not multi.empty:
                section("Multi-Feature Steering")
                caption("Effect of steering multiple features simultaneously.")
                show_table(multi, height=320)

            sc_plot = run_dir / "plots" / "steering_curves.png"
            if sc_plot.exists():
                with st.expander("📊 Saved steering curves plot"):
                    st.image(str(sc_plot), width="stretch")

        # ── local vs global ────────────────────────────────────────────────────
        lvg = read_csv(causal_dir / "local_vs_global.csv")
        if lvg is not None:
            st.divider()
            section("Local vs. Global Scope Comparison")
            caption("Ablation effect when restricted to the most active spatial region (local) "
                    "vs. applied to all patches (global). A spatially-concentrated feature "
                    "should show a similar effect under both conditions.")
            show_table(lvg, height=320)
            lvg_plot = run_dir / "plots" / "local_vs_global.png"
            if lvg_plot.exists():
                with st.expander("📊 Saved local vs. global plot"):
                    st.image(str(lvg_plot), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — FEATURE INTERACTION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    section("Feature Interaction Analysis")
    st.markdown(
        "Measures synergy and redundancy between pairs of candidate features. "
        "Synergy: ablating both features together has a larger effect than the sum of individual ablations. "
        "Redundancy: the combined effect is smaller than the sum."
    )
    it = read_csv(causal_dir / "interactions.csv")
    if it is not None:
        cols_show = [c for c in it.columns if c in
                     ["feature_a", "feature_b", "interaction_type", "interaction_effect",
                      "individual_a", "individual_b", "combined", "synergy_ratio", "n_images"]]
        show_table(it[cols_show] if cols_show else it, height=360, empty_title="Interaction table is empty")

        fi_plot = run_dir / "plots" / "feature_interaction.png"
        if fi_plot.exists():
            st.image(str(fi_plot), caption="Feature interaction plot (saved pipeline output)",
                     width="stretch")
    else:
        empty_state("No interaction CSV found for this run", "Pairwise synergy/redundancy results will appear after causal evaluation.")

    if not summ_ok:
        st.warning("Run `python discover.py` first for live interaction computation.", icon="⚠️")
    else:
        st.divider()
        section("Live Interaction Computation")
        caption("Select two alive features to compute their interaction live on held-out causal-eval images. "
                "This uses saved hidden states — no model re-inference needed.")
        alive = ctx.summary().index[~ctx.summary()["dead"]].tolist()
        c = st.columns(2)
        a = c[0].selectbox("Feature A", alive,
                            index=alive.index(default_feat) if default_feat in alive else 0, key="ia")
        b = c[1].selectbox("Feature B", alive, index=min(1, len(alive) - 1), key="ib")
        if a == b:
            st.warning("Feature A and B must be different.", icon="⚠️")
        elif st.button("⚡ Compute interaction (causal-eval images)", type="primary"):
            from causal.metrics import compute_effects
            from causal.stats import interaction_stats
            pos = ctx.splits["causal"]
            both = (ctx.feature_activations(a, pos).max(1) > 0) & (ctx.feature_activations(b, pos).max(1) > 0)
            if both.sum() < 3:
                st.warning(f"Only {both.sum()} images have both features active — too few for reliable statistics.", icon="⚠️")
            else:
                with st.spinner(f"Computing interaction on {both.sum()} images…"):
                    eng = ctx.engine()
                    h = ctx.hidden(pos[both][:100])
                    bsln = eng.baseline_many(h)
                    y = bsln.argmax(1)
                    eff_a = compute_effects(bsln, eng.run_many(h, [a], [0.0])["logits"], y)["logit_effect"]
                    eff_b = compute_effects(bsln, eng.run_many(h, [b], [0.0])["logits"], y)["logit_effect"]
                    eff_ab = compute_effects(bsln, eng.run_many(h, [a, b], [0.0, 0.0])["logits"], y)["logit_effect"]
                    result = interaction_stats(eff_a, eff_b, eff_ab, 500)
                st.json(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 8 — LAYER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    comp = read_csv(EXPS / exp / "comparison.csv")
    section("Layer Comparison")
    st.markdown(
        "Compare the same SAE architecture across different ViT encoder layers. "
        "Run `python run_pipeline.py --mode full --sweep` to populate multiple layers."
    )
    if comp is None:
        empty_state("No comparison data found", "Run `--sweep` to generate data across multiple layers.")
    else:
        lay = comp[(comp.sae_type == cfg.sae.sae_type) & (comp.width == cfg.sae.width)]
        numeric_cols = [c for c in comp.columns if comp[c].dtype.kind == "f"]
        if not numeric_cols:
            st.warning("No numeric columns found in comparison.csv.")
        else:
            metric = st.selectbox("Metric to compare", numeric_cols, key="lm",
                                   index=numeric_cols.index("explained_variance") if "explained_variance" in numeric_cols else 0)
            if not lay.empty:
                fig = px.line(
                    lay.sort_values("layer"), x="layer", y=metric, markers=True,
                    labels={metric: metric.replace("_", " ").title(), "layer": "ViT Block Layer"},
                    title=f"{metric.replace('_', ' ').title()} across layers — {cfg.sae.sae_type.upper()} SAE",
                )
                show_chart(fig, _PLOT_H_LG)
                caption(f"Each point is one trained SAE run at the corresponding ViT encoder block. "
                        f"Showing: {cfg.sae.sae_type.upper()} SAE, width={cfg.sae.width}.")
            else:
                empty_state(
                    "No matching layer runs",
                    f"comparison.csv has no rows for {cfg.sae.sae_type.upper()} SAE with width={cfg.sae.width}.",
                )

        with st.expander("📋 Full comparison table"):
            show_table(lay, height=420, empty_title="No matching rows in comparison.csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 9 — SAE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    section("SAE Architecture Comparison")
    st.markdown(
        "Compare different SAE types (Standard ReLU / L1 / Top-K) and widths at the same layer. "
        "Run `--sweep` to generate results across multiple architectures."
    )
    comp_sae = read_csv(EXPS / exp / "comparison.csv")
    if comp_sae is None:
        empty_state("No comparison data found", "Run `--sweep` to generate data across multiple SAE types.")
    else:
        sub = comp_sae[comp_sae.layer == cfg.layer]
        numeric_cols = [c for c in comp_sae.columns if comp_sae[c].dtype.kind == "f"]
        if not numeric_cols:
            st.warning("No numeric columns found in comparison.csv.")
        else:
            metric = st.selectbox("Metric to compare", numeric_cols, key="sm",
                                   index=numeric_cols.index("explained_variance") if "explained_variance" in numeric_cols else 0)
            sub = sub.copy()
            sub["label"] = (
                sub["sae_type"] + "-W" + sub["width"].astype(str)
                + sub["k"].fillna(0).astype(int).map(lambda k: f"-k{k}" if k else "")
            )
            if not sub.empty:
                fig = px.bar(
                    sub, x="label", y=metric,
                    color="sae_type",
                    labels={metric: metric.replace("_", " ").title(), "label": "SAE configuration"},
                    title=f"{metric.replace('_', ' ').title()} across SAE architectures — Layer {cfg.layer}",
                )
                show_chart(fig, _PLOT_H_LG)
                caption("Each bar is one SAE architecture at the same ViT layer. "
                        "Compare reconstruction quality, sparsity, and feature quality across SAE designs.")
            else:
                empty_state("No SAE runs at this layer", f"comparison.csv has no rows for layer {cfg.layer}.")

        with st.expander("📋 Full architecture comparison table"):
            show_table(sub if not sub.empty else comp_sae, height=420)

    # ── spatial summary (Grad-CAM correlation) ─────────────────────────────────
    sp_sum = read_csv(causal_dir / "spatial_summary.csv")
    if sp_sum is not None:
        st.divider()
        section("Spatial Specificity & Grad-CAM Correlation")
        caption("Spearman correlation between patch activation strength and patch causal effect. "
                "A high correlation indicates that the feature's spatial structure predicts its causal influence.")
        show_table(sp_sum, height=320)
    else:
        empty_state("No spatial summary CSV", "Grad-CAM correlation tables appear here after causal evaluation.")

    gcam_sum = read_csv(causal_dir / "gradcam_comparison.csv")
    if gcam_sum is not None:
        section("Grad-CAM vs. SAE Feature Overlap")
        caption("Comparison of SAE feature heatmaps against gradient-based Grad-CAM saliency maps.")
        show_table(gcam_sum, height=320)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 10 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[9]:
    section("Export Research Artifacts")
    st.markdown(
        "Download all saved pipeline outputs: CSVs, JSONs, and PNGs. "
        "These files contain the complete experimental record for this run."
    )

    DATA_FILES = {
        "📊 Results (all conditions) — CSV": causal_dir / "results_rows.csv",
        "📊 Candidate causal summary — CSV": causal_dir / "candidate_causal_summary.csv",
        "📊 Feature summary — CSV": disc_dir / "feature_summary.csv",
        "📊 Neuron summary — CSV": disc_dir / "neuron_summary.csv",
        "📊 Clusters — CSV": disc_dir / "clusters.csv",
        "📊 Steering curves — CSV": causal_dir / "steering.csv",
        "📊 Interactions — CSV": causal_dir / "interactions.csv",
        "📊 Spatial summary — CSV": causal_dir / "spatial_summary.csv",
        "📊 Grad-CAM comparison — CSV": causal_dir / "gradcam_comparison.csv",
        "📊 Local vs. global — CSV": causal_dir / "local_vs_global.csv",
        "📊 Comparison — CSV": EXPS / exp / "comparison.csv",
        "📊 Training metrics — CSV": run_dir / "metrics.csv",
        "🔧 Run results — JSON": run_dir / "results.json",
        "🔧 Layer summary — JSON": run_dir / "layer_summary.json",
        "🔧 Dose-response — JSON": causal_dir / "dose_response.json",
        "🔧 Random controls — JSON": causal_dir / "random_controls.json",
        "🔧 Hypotheses — JSON": EXPS / exp / "hypotheses.json",
        "🔧 Discovery summary — JSON": disc_dir / "discovery_summary.json",
        "🔧 Config — JSON": run_dir / "config.json",
    }

    section("Data & Results")
    inventory = pd.DataFrame([
        {
            "artifact": label,
            "file": Path(p).name,
            "status": "available" if Path(p).exists() else "missing",
            "size_kb": round(Path(p).stat().st_size / 1024, 1) if Path(p).exists() else None,
        }
        for label, p in DATA_FILES.items()
    ])
    show_table(inventory, height=420)
    caption("Download buttons below are shown only for files that exist in this run. Missing artifacts are listed, not invented.")

    present = [(label, p) for label, p in DATA_FILES.items() if Path(p).exists()]
    if not present:
        empty_state("No downloadable data files", "This run directory does not yet contain the listed CSVs or JSONs.")
    else:
        data_cols = st.columns(3)
        for i, (label, p) in enumerate(present):
            data_cols[i % 3].download_button(
                label, Path(p).read_bytes(), file_name=Path(p).name, key="dl_" + label
            )

    # ── plots ─────────────────────────────────────────────────────────────────
    all_pngs = list((run_dir / "plots").rglob("*.png")) if (run_dir / "plots").exists() else []
    st.divider()
    section("Plots & Visualisations")
    if all_pngs:
        caption(f"{len(all_pngs)} PNG files under `plots/`. Buttons wrap to four columns; hover for the relative path.")
        png_cols = st.columns(4)
        for i, p in enumerate(sorted(all_pngs)):
            rel = str(p).replace(str(run_dir), "").lstrip("/\\")
            png_cols[i % 4].download_button(
                f"🖼 {p.name}", p.read_bytes(), file_name=p.name, mime="image/png",
                key="png_" + str(p), help=rel,
            )
    else:
        empty_state("No plot PNGs found", "Saved matplotlib figures from the pipeline will be listed here.")

    # ── full results.csv ───────────────────────────────────────────────────────
    global_results = EXPS / "results.csv"
    if global_results.exists():
        st.divider()
        section("Global Results Table (all experiments)")
        gr = read_csv(global_results)
        show_table(gr, height=480, empty_title="Global results.csv is empty")
        st.download_button("📥 Download global results.csv",
                           global_results.read_bytes(), file_name="results.csv", key="dl_global")
    else:
        empty_state("No global results.csv", "A combined experiment table will appear at `experiments/results.csv`.")
