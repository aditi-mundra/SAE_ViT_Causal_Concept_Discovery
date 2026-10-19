"""
app.py [CORE] -- Streamlit dashboard.   Run:  streamlit run app.py
Needs a finished run (extract -> train -> discover -> evaluate), e.g. `python run_pipeline.py --mode demo`.
"""
from pathlib import Path

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

st.set_page_config(page_title="SAE-ViT Causal Concept Discovery", layout="wide")
EXPS = PROJECT_ROOT / "experiments"


@st.cache_resource(show_spinner="Loading run (ViT, SAE, activation store)...")
def get_ctx(run_dir: str):
    return RunContext.from_run_dir(run_dir)


def read_csv(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


# ------------------------------------------------------------------ sidebar
exps = sorted(d.name for d in EXPS.iterdir() if d.is_dir()) if EXPS.exists() else []
if not exps:
    st.error("No experiments found. Run `python run_pipeline.py --mode demo` first.")
    st.stop()
exp = st.sidebar.selectbox("Experiment", exps)
runs = sorted(d.name for d in (EXPS / exp).iterdir() if d.is_dir() and (d / "checkpoint.pt").exists())
if not runs:
    st.error("This experiment has no trained SAE runs.")
    st.stop()
run = st.sidebar.selectbox("Run (layer / SAE)", runs)
run_dir = EXPS / exp / run
ctx = get_ctx(str(run_dir))
cfg, summ_ok = ctx.cfg, (run_dir / "discovery" / "feature_summary.csv").exists()
causal_dir = run_dir / "causal"
cand = read_csv(causal_dir / "candidate_causal_summary.csv")
default_feat = int(cand["feature_id"].iloc[0]) if cand is not None and len(cand) else 0
st.title("SAE-Based Causal Concept Discovery in Vision Transformers")
tabs = st.tabs(["1 Overview", "2 Image Explorer", "3 Feature Explorer", "4 Causal Intervention", "5 Random Controls",
                "6 Feature Steering", "7 Feature Interaction", "8 Layer Comparison", "9 SAE Comparison", "10 Export"])


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
    c = st.columns(5)
    c[0].metric("Target class", names[y])
    c[1].metric("Logit before / after", f"{base[y]:.2f} / {new[y]:.2f}", f"{new[y]-base[y]:+.3f}")
    c[2].metric("Prob before / after", f"{sm(base)[y]:.3f} / {sm(new)[y]:.3f}", f"{sm(new)[y]-sm(base)[y]:+.3f}")
    c[3].metric("Prediction flip", "YES" if new.argmax() != y else "no")
    c[4].metric("Max norm ratio", f"{info['max_norm_ratio']:.2f}")
    if info["flagged"]:
        st.warning("Safety flag: extreme activation norm or NaN/Inf after intervention.")
    if info["clamped"]:
        st.warning("Strength was clamped to the configured safety limit.")
    t = pd.DataFrame({"class (before)": [names[i] for i in np.argsort(-base)[:5]], "logit before": np.sort(base)[::-1][:5],
                      "class (after)": [names[i] for i in np.argsort(-new)[:5]], "logit after": np.sort(new)[::-1][:5]})
    st.dataframe(t, use_container_width=True)


def pick_image(key, fid):
    src = st.radio("Image source", ["Held-out causal-eval image", "Upload"], key=key + "src", horizontal=True)
    if src == "Upload":
        up = st.file_uploader("Image", type=["png", "jpg", "jpeg"], key=key + "up")
        if up is None:
            return None, None
        from PIL import Image
        pil = Image.open(up).convert("RGB")
        return pil, analyze_image(ctx, pil)["hidden"]
    pos_list = need_causal_images(fid)
    if len(pos_list) == 0:
        st.info("This feature is not active on the first causal images.")
        return None, None
    pos = int(st.selectbox("Store position", pos_list, key=key + "pos"))
    return ctx.image_pil(pos), ctx.hidden([pos])


# ------------------------------------------------------------------ 1 overview
with tabs[0]:
    res = load_json(run_dir / "results.json") if (run_dir / "results.json").exists() else {}
    m = res.get("metrics", {})
    c = st.columns(4)
    c[0].metric("Model", cfg.model.name); c[1].metric("Dataset", f"{cfg.data.dataset} ({cfg.data.num_images})")
    c[2].metric("Layer (block)", cfg.layer); c[3].metric("SAE", f"{cfg.sae.sae_type} {cfg.sae.width}" + (f" k={cfg.sae.k}" if cfg.sae.sae_type == 'topk' else ""))
    c = st.columns(4)
    c[0].metric("Reconstruction MSE", f"{m.get('mse', float('nan')):.4f}"); c[1].metric("Explained variance", f"{m.get('explained_variance', float('nan')):.3f}")
    c[2].metric("L0", f"{m.get('l0', float('nan')):.1f}"); c[3].metric("Dead features (val)", f"{m.get('dead_pct_eval', float('nan')):.1f}%")
    warns = load_json(run_dir / "warnings.json") if (run_dir / "warnings.json").exists() else []
    st.subheader("Automatic warnings")
    for w in warns:
        st.warning(w)
    if not warns:
        st.success("No warnings.")
    for p in sorted((run_dir / "plots").glob("*.png"))[:6] if (run_dir / "plots").exists() else []:
        st.image(str(p), caption=p.name)
    st.json(cfg.to_dict())

# ------------------------------------------------------------------ 2 image explorer
with tabs[1]:
    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="explore")
    if up is not None and summ_ok:
        from PIL import Image
        pil = Image.open(up).convert("RGB")
        out = analyze_image(ctx, pil)
        z = out["logits"]; p = np.exp(z - z.max()); p /= p.sum()
        c1, c2 = st.columns(2)
        c1.image(pil, caption="input")
        top = np.argsort(-z)[:5]
        c2.dataframe(pd.DataFrame({"class": [ctx.vit_class_names[i] for i in top], "logit": z[top], "confidence": p[top]}))
        st.subheader("Feature retrieval: strongly active SAE features")
        st.dataframe(pd.DataFrame(out["features"]), use_container_width=True)
        disc_db = read_csv(run_dir / "discovery" / "feature_top_patches.csv.gz")
        for f in out["features"][:4]:
            st.markdown(f"**Feature {f['feature_id']}** - {f.get('concept', '')}")
            ex = disc_db[(disc_db.feature_id == f["feature_id"])].head(4) if disc_db is not None else None
            if ex is not None and len(ex):
                cols = st.columns(len(ex))
                for col, (_, r) in zip(cols, ex.iterrows()):
                    col.image(ctx.image_pil(int(r["position"])).resize((160, 160)), caption=f"{r['class']} ({r['activation']:.2f})")
    elif not summ_ok:
        st.info("Run discover.py to enable feature retrieval.")

# ------------------------------------------------------------------ 3 feature explorer
with tabs[2]:
    if not summ_ok:
        st.info("Run `python discover.py` first.")
    else:
        S = ctx.summary()
        fid = int(st.number_input("Feature id", 0, int(ctx.sae.width) - 1, default_feat, key="fx"))
        row = S.loc[fid]
        if row["dead"]:
            st.warning("This feature is dead (never active on the validation set).")
        else:
            if row["freq"] < 1e-4:
                st.warning("Feature activates almost nowhere.")
            if row["freq"] > 0.2:
                st.warning("Feature is very dense.")
            c = st.columns(5)
            c[0].metric("Purity", f"{row['purity']:.2f}"); c[1].metric("Top class", row["top_class_name"])
            c[2].metric("Spatial conc. (SCS)", f"{row['spatial_conc10']:.2f}"); c[3].metric("Frequency", f"{row['freq']:.4f}")
            c[4].metric("Specificity", row["specificity"])
            if "clip_concept" in row and isinstance(row["clip_concept"], str) and row["clip_concept"]:
                st.write(f"CLIP concept: **{row['clip_concept']}** (score {row['clip_score']:.3f}, entropy {row['clip_entropy']:.2f}); top-3: {row['clip_top3']}")
            if cand is not None and fid in set(cand.feature_id):
                cr = cand[cand.feature_id == fid].iloc[0]
                st.write(f"Causal score: mean ablation logit drop **{cr['abl_mean']:.3f}** [{cr['abl_ci_lo']:.3f}, {cr['abl_ci_hi']:.3f}], causal rank {int(cr['causal_rank'])}")
            rep = feature_report(ctx, fid)
            st.image(rep["report"])
            hm = rep["heatmaps"]
            if hm:
                cols = st.columns(5)
                for col, k in zip(cols, ["raw", "normalized", "overlay", "mask", "bbox"]):
                    col.image(hm[k], caption=k)
            cls = ctx.disc["class_mass"][:, fid]; pc = cls / max(cls.sum(), 1e-12)
            top = np.argsort(-pc)[:10]
            st.plotly_chart(px.bar(x=[ctx.store.class_names[i] for i in top], y=pc[top], labels={"x": "class", "y": "P(class|feature)"}))

# ------------------------------------------------------------------ 4 causal intervention
with tabs[3]:
    if not summ_ok:
        st.info("Run discover.py first.")
    else:
        fid = int(st.number_input("Feature", 0, int(ctx.sae.width) - 1, default_feat, key="ci_f"))
        c = st.columns(4)
        kind = c[0].selectbox("Intervention", ["ablation (x0)", "suppression", "amplification", "steering"])
        default = {"ablation (x0)": 0.0, "suppression": 0.5, "amplification": 2.0, "steering": 2.0}[kind]
        strength = c[1].number_input("Strength (multiplier / steering alpha)", -10.0, 10.0, default, key="ci_s")
        scope = c[2].selectbox("Scope", ["global", "local"])
        pct = c[3].selectbox("Local percentile", [50, 75, 90, 95], index=2) if scope == "local" else None
        pil, hidden = pick_image("ci", fid)
        if hidden is not None:
            st.image(pil, width=240)
            base, new, info = run_intervention(hidden, [fid], [0.0 if kind.startswith("abl") else strength],
                                               "steer" if kind == "steering" else "scale", pct)
            show_result(base, new, info)

# ------------------------------------------------------------------ 5 random controls
with tabs[4]:
    rc_p = causal_dir / "random_controls.json"
    if not rc_p.exists() or cand is None:
        st.info("Run evaluate.py first.")
    else:
        rc = load_json(rc_p)
        feats = sorted({int(k.split("_")[0]) for k in rc})
        f = st.selectbox("Feature", feats, key="rc_f")
        for cond in ("ablation", "amplification"):
            d = rc.get(f"{f}_{cond}")
            if not d:
                continue
            fig = go.Figure()
            fig.add_trace(go.Box(y=d["own_means"], name="random features (own active images)", boxpoints="all"))
            fig.add_trace(go.Scatter(x=["random features (own active images)"], y=[d["cand_mean"]], mode="markers",
                                     marker=dict(color="red", size=14), name="candidate"))
            fig.update_layout(title=f"{cond}: mean logit effect", yaxis_title="effect")
            st.plotly_chart(fig, use_container_width=True)
            lo, hi = d["rand_ci95"]
            st.write(f"candidate {d['cand_mean']:.3f} | random mean {d['rand_mean']:.3f} (95% CI [{lo:.3f}, {hi:.3f}]) | "
                     f"empirical p = {d['p_empirical']:.4f} | paired Wilcoxon p = {d['paired_p_wilcoxon']:.4g} | paired d_z = {d['paired_dz']:.2f}")
            if d["p_empirical"] >= 0.05:
                st.warning("Causal effect is indistinguishable from random controls.")
        st.dataframe(cand, use_container_width=True)

# ------------------------------------------------------------------ 6 steering
with tabs[5]:
    if not summ_ok:
        st.info("Run discover.py first.")
    else:
        S = ctx.summary()
        alive = S.index[~S["dead"]].tolist()
        sel = st.multiselect("Features", alive, default=[default_feat] if default_feat in alive else alive[:1])
        alpha = st.slider("Steering alpha (units of the normalised decoder direction)", -10.0, 10.0, 2.0, 0.5)
        if sel:
            pil, hidden = pick_image("st", sel[0])
            if hidden is not None:
                st.image(pil, width=240)
                base, new, info = run_intervention(hidden, sel, [alpha] * len(sel), "steer", None)
                show_result(base, new, info)
        stf = read_csv(causal_dir / "steering.csv")
        if stf is not None:
            st.subheader("Evaluated steering curves (held-out images)")
            st.plotly_chart(px.line(stf[stf.type == "single"], x="alpha", y="mean_delta_logit", color=stf[stf.type == "single"]["feature_id"].astype(str), markers=True))
            st.dataframe(stf[stf.type == "multi"] if (stf.type == "multi").any() else stf.head(0))

# ------------------------------------------------------------------ 7 interaction
with tabs[6]:
    it = read_csv(causal_dir / "interactions.csv")
    if it is not None:
        st.dataframe(it, use_container_width=True)
    if summ_ok:
        S = ctx.summary(); alive = S.index[~S["dead"]].tolist()
        a = st.selectbox("Feature A", alive, index=alive.index(default_feat) if default_feat in alive else 0, key="ia")
        b = st.selectbox("Feature B", alive, index=min(1, len(alive) - 1), key="ib")
        if a != b and st.button("Compute live (causal-eval images where both are active)"):
            from causal.metrics import compute_effects
            from causal.stats import interaction_stats
            pos = ctx.splits["causal"]
            both = (ctx.feature_activations(a, pos).max(1) > 0) & (ctx.feature_activations(b, pos).max(1) > 0)
            if both.sum() < 3:
                st.warning("Fewer than 3 images where both features are active.")
            else:
                eng = ctx.engine(); h = ctx.hidden(pos[both][:100])
                base = eng.baseline_many(h); y = base.argmax(1)
                eff = lambda ids: compute_effects(base, eng.run_many(h, ids, [0.0] * len(ids))["logits"], y)["logit_effect"]
                r = interaction_stats(eff([a]), eff([b]), eff([a, b]), 500)
                st.json(r)

# ------------------------------------------------------------------ 8 / 9 comparison
comp = read_csv(EXPS / exp / "comparison.csv")
with tabs[7]:
    if comp is None:
        st.info("Run run_pipeline.py with --sweep (or several layers) to populate comparisons.")
    else:
        lay = comp[(comp.sae_type == cfg.sae.sae_type) & (comp.width == cfg.sae.width)]
        st.dataframe(lay, use_container_width=True)
        metric = st.selectbox("Metric", [c for c in comp.columns if comp[c].dtype.kind == "f"], key="lm")
        st.plotly_chart(px.line(lay.sort_values("layer"), x="layer", y=metric, markers=True))
with tabs[8]:
    if comp is None:
        st.info("No comparison data yet.")
    else:
        sub = comp[comp.layer == cfg.layer]
        st.dataframe(sub, use_container_width=True)
        metric = st.selectbox("Metric", [c for c in comp.columns if comp[c].dtype.kind == "f"], key="sm")
        sub = sub.assign(label=sub.sae_type + "-" + sub.width.astype(str) + sub.k.fillna(0).astype(int).map(lambda k: f"-k{k}" if k else ""))
        st.plotly_chart(px.bar(sub, x="label", y=metric))

# ------------------------------------------------------------------ 10 export
with tabs[9]:
    files = {"results (all conditions) CSV": causal_dir / "results_rows.csv", "candidate summary CSV": causal_dir / "candidate_causal_summary.csv",
             "comparison CSV": EXPS / exp / "comparison.csv", "feature summary CSV": run_dir / "discovery" / "feature_summary.csv",
             "hypotheses JSON": EXPS / exp / "hypotheses.json", "run results JSON": run_dir / "results.json",
             "layer summary JSON": run_dir / "layer_summary.json", "config JSON": run_dir / "config.json"}
    for label, p in files.items():
        if p.exists():
            st.download_button(label, p.read_bytes(), file_name=p.name, key="dl" + label)
    for p in sorted((run_dir / "plots").glob("*.png")) if (run_dir / "plots").exists() else []:
        st.download_button(f"PNG: {p.name}", p.read_bytes(), file_name=p.name, mime="image/png", key="png" + p.name)
