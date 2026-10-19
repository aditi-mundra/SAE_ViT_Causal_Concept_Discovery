"""
causal/evaluate.py [CORE]
Causal evaluation on the held-out CAUSAL-EVAL split.

Design (leakage / selection-bias control):
  * Candidate features are chosen ONLY from validation-split, non-causal criteria (interp_score; no CLIP, no causal outcome).
  * Each candidate is evaluated on causal-split images where it is active (random subsample, no outcome-based selection).
  * Random controls are frequency-matched features: (a) evaluated on the SAME images (paired), (b) evaluated on their OWN
    active images (activity-matched null; p_empirical).
  * Interaction / multi-feature steering use the top candidates by interp_score (not by causal effect).
Effects: logit_effect = z_y(base) - z_y(intervened) for the model's baseline top-1 class y (positive = logit dropped);
         delta_logit = z_y(new) - z_y(base) (positive = logit rose) is used for amplification / steering.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from causal.metrics import compute_effects
from causal.results_db import write_results
from causal.stats import (bootstrap_ci, compare_to_random, dose_response_stats, holm_correction,
                          interaction_stats, paired_tests)
from context import RunContext
from interventions.spatial import patch_ablation_map
from sae.health import check_intervention
from sae.metrics import compute_image_max
from utils import ensure_dir, get_logger, load_json, save_json


def select_candidates(summary, active, icfg, dcfg):
    s = summary[(~summary["dead"]) & summary["interp_score"].notna()]
    ok = s[(s["freq"] >= dcfg.min_freq) & (s["freq"] <= dcfg.max_freq)]
    notes = []
    if len(ok) == 0:
        ok, notes = s, ["no feature within [min_freq,max_freq]; frequency filter relaxed"]
    ok = ok.sort_values("interp_score", ascending=False)
    out = [int(f) for f in ok.index if active[:, int(f)].sum() >= icfg.min_active_images][: icfg.n_candidates]
    return out, notes


def matched_pool(summary, active, fid, exclude, ratio, R, min_active, rng):
    freq = summary["freq"].values
    ok = (~summary["dead"].values) & (active.sum(0) >= min_active)
    ok[list(exclude)] = False
    cand = np.where(ok)[0]
    if len(cand) == 0:
        return np.array([], dtype=int)
    r, fc = ratio, max(freq[fid], 1e-12)
    while True:
        band = cand[(freq[cand] >= fc / r) & (freq[cand] <= fc * r)]
        if len(band) >= R or r > 1e4:
            break
        r *= 2
    return rng.choice(band, min(R, len(band)), replace=False) if len(band) else np.array([], dtype=int)


class _Eval:
    def __init__(self, ctx, hid, base, target, active, rng, warnings):
        self.ctx, self.hid, self.base, self.target, self.active = ctx, hid, base, target, active
        self.rng, self.warnings, self.icfg = rng, warnings, ctx.cfg.intervention
        self.engine = ctx.engine()

    def sample_active(self, fid, mask=None):
        a = self.active[:, fid] if mask is None else mask
        idx = np.where(a)[0]
        if len(idx) > self.icfg.n_eval_images:
            idx = np.sort(self.rng.choice(idx, self.icfg.n_eval_images, replace=False))
        return idx

    def run(self, idx, ids, alphas, mode="scale", local_percentile=None, explicit_mask=None):
        res = self.engine.run_many(self.hid[idx], ids, alphas, mode, local_percentile, explicit_mask, self.icfg.batch_size)
        for w in check_intervention(res):
            if w not in self.warnings:
                self.warnings.append(w)
        return compute_effects(self.base[idx], res["logits"], self.target[idx]), res


def _summ(x, n_boot, seed):
    m, lo, hi = bootstrap_ci(x, n_boot, seed=seed)
    t = paired_tests(x)
    return {"mean": m, "ci_lo": lo, "ci_hi": hi, "p_t": t["p_t"], "p_w": t["p_wilcoxon"], "dz": t["dz"]}


def _random_control(E, summ, fid, idx, alpha, measure, exclude, R, rng):
    icfg = E.icfg
    rids = matched_pool(summ, E.active, fid, exclude, icfg.control_freq_ratio, R, icfg.min_active_images, rng)
    if len(rids) < 3:
        return None
    same, own = np.zeros((len(rids), len(idx))), np.zeros(len(rids))
    for j, r in enumerate(rids):
        same[j] = E.run(idx, [int(r)], [alpha])[0][measure]
        own[j] = E.run(E.sample_active(int(r)), [int(r)], [alpha])[0][measure].mean()
    return same, own, rids


def _cond_rows(E, base_row, cond, idx, eff, fid_extra):
    rows = []
    for k, i in enumerate(idx):
        r = dict(base_row)
        r.update({"image_id": int(E.ctx.store.image_ids[E.causal_pos[i]]),
                  "class": E.ctx.vit_class_names[int(E.target[i])],
                  "dataset_class": E.ctx.store.class_names[int(E.ctx.store.labels[E.causal_pos[i]])],
                  "condition": cond, "baseline_logit": eff["base_logit"][k], "ablated_logit": eff["new_logit"][k],
                  "logit_effect": eff["logit_effect"][k], "baseline_probability": eff["base_prob"][k],
                  "ablated_probability": eff["new_prob"][k], "probability_effect": eff["prob_effect"][k],
                  "flip": bool(eff["flip"][k]), "rank_change": int(eff["rank_change"][k]),
                  "activation": float(E.img_max[i, base_row["feature_id"]])})
        r.update(fid_extra(k))
        rows.append(r)
    return rows


def run_causal_evaluation(cfg, ctx=None, logger=None):
    log = logger or get_logger()
    ctx = ctx or RunContext(cfg)
    icfg, dcfg, nb = cfg.intervention, cfg.discovery, cfg.intervention.n_bootstrap
    out = ensure_dir(ctx.run_dir / "causal"); plots = ensure_dir(ctx.run_dir / "plots")
    rng = np.random.RandomState(cfg.seed + 7)
    summ = ctx.summary()
    causal_pos = ctx.splits["causal"]
    log.info(f"[causal] {cfg.run_tag()}: {len(causal_pos)} held-out causal-eval images")
    hid = ctx.hidden(causal_pos)
    engine = ctx.engine()
    base = engine.baseline_many(hid, icfg.batch_size)
    target = base.argmax(1)
    img_max = compute_image_max(ctx.sae, ctx.normalizer, ctx.store, ctx.layer, causal_pos, ctx.device)
    active = img_max > 0
    warnings = list(load_json(ctx.run_dir / "warnings.json")) if (ctx.run_dir / "warnings.json").exists() else []
    cands, notes = select_candidates(summ, active, icfg, dcfg)
    warnings += notes
    if not cands:
        raise RuntimeError("no candidate feature is active on enough causal images; increase num_images or lower min_active_images")
    E = _Eval(ctx, hid, base, target, active, rng, warnings)
    E.causal_pos, E.img_max = causal_pos, img_max
    # sanity: alpha=1 must reproduce the baseline exactly
    i0 = np.arange(min(8, len(causal_pos)))
    err = np.abs(engine.run_many(hid[i0], [cands[0]], [1.0], "scale")["logits"] - base[i0]).max()
    if err > 1e-3 * max(1.0, np.abs(base).max()):
        warnings.append(f"identity intervention (alpha=1) changed logits by {err:.2e}")
    log.info(f"[causal] candidates (validation-only selection): {cands}")

    all_rows, recs, rc_store, dose_store, lg_rows, steer_rows = [], [], {}, {}, [], []
    exp_id = f"{cfg.name}/{cfg.run_tag()}"
    for fid in cands:
        s = summ.loc[fid]
        idx = E.sample_active(fid)
        concept = s["clip_concept"] if ("clip_concept" in s and isinstance(s["clip_concept"], str) and s["clip_concept"]) else s["top_class_name"]
        rec = {"feature_id": fid, "n_images": len(idx), "concept": concept, "purity": s["purity"],
               "clip_score": s.get("clip_score", np.nan), "spatial_score": s["spatial_conc10"],
               "interp_score": s["interp_score"], "freq": s["freq"], "class_entropy_norm": s["class_entropy_norm"]}
        base_row = {"experiment_id": exp_id, "layer": cfg.layer, "sae_type": cfg.sae.sae_type, "sae_width": cfg.sae.width,
                    "feature_id": fid, "concept": concept, "purity": s["purity"], "clip_score": rec["clip_score"],
                    "spatial_score": s["spatial_conc10"], "flagged": False}
        no_extra = lambda k: {}
        # ---- global ablation (alpha = 0)
        ab, res = E.run(idx, [fid], [0.0])
        base_row["flagged"] = bool(res["flagged"])
        sa = _summ(ab["logit_effect"], nb, fid)
        rec.update({f"abl_{k}": v for k, v in sa.items()})
        rec.update({"abl_flip_rate": float(ab["flip"].mean()), "abl_mean_prob_effect": float(ab["prob_effect"].mean()),
                    "abl_mean_rank_change": float(ab["rank_change"].mean()),
                    "abl_mean_patches_modified": float(res["n_modified"].mean()),
                    "abl_max_norm_ratio": res["max_norm_ratio"], "abl_flagged": bool(res["flagged"])})
        # ---- random control for ablation
        rc = _random_control(E, summ, fid, idx, 0.0, "logit_effect", set(cands) | {fid}, icfg.n_random_controls, rng)
        rids = []
        extra_abl = no_extra
        if rc is None:
            warnings.append(f"feature {fid}: too few matched random controls")
        else:
            same, own, rids = rc
            cmp_ = compare_to_random(ab["logit_effect"], same, own, nb, fid)
            rec.update({f"abl_rand_{k}": v for k, v in cmp_.items() if not isinstance(v, list)})
            rec["abl_rand_ci95"] = cmp_["rand_ci95"]
            rc_store[f"{fid}_ablation"] = {"own_means": own.tolist(), "same_means": same.mean(1).tolist(), **{k: v for k, v in cmp_.items()}}
            rm = same.mean(0)
            ci_str = f"[{sa['ci_lo']:.3f}, {sa['ci_hi']:.3f}]"
            extra_abl = lambda k, rm=rm, p=cmp_["p_empirical"], c=ci_str: {"random_effect": float(rm[k]), "p_value": p, "confidence_interval": c}
            if cmp_["p_empirical"] >= 0.05:
                warnings.append(f"feature {fid}: ablation effect not distinguishable from random (p_emp={cmp_['p_empirical']:.3f})")
        all_rows += _cond_rows(E, base_row, "ablation_x0", idx, ab, extra_abl)
        if len(idx) < icfg.min_active_images:
            warnings.append(f"feature {fid}: activates on only {len(idx)} causal images")
        # ---- suppression
        for a in icfg.suppress_alphas:
            if a == 0.0:
                continue
            e, r_ = E.run(idx, [fid], [a])
            rec[f"sup_{a}_mean_logit_effect"] = float(e["logit_effect"].mean())
            all_rows += _cond_rows(E, base_row, f"suppress_x{a}", idx, e, no_extra)
        # ---- amplification (+ random control at the primary alpha)
        amps = sorted(set(icfg.amplify_alphas) | {icfg.amp_alpha_primary})
        for a in amps:
            e, r_ = E.run(idx, [fid], [a])
            sm = _summ(e["delta_logit"], nb, fid + 1)
            rec[f"amp_{a}_mean_delta_logit"] = sm["mean"]; rec[f"amp_{a}_flip_rate"] = float(e["flip"].mean())
            extra = no_extra
            if a == icfg.amp_alpha_primary:
                rec.update({f"amp_{k}": v for k, v in sm.items()})
                rcA = _random_control(E, summ, fid, idx, a, "delta_logit", set(cands) | {fid}, icfg.n_random_controls, rng)
                if rcA is not None:
                    sameA, ownA, _ = rcA
                    cA = compare_to_random(e["delta_logit"], sameA, ownA, nb, fid + 2)
                    rec.update({f"amp_rand_{k}": v for k, v in cA.items() if not isinstance(v, list)})
                    rc_store[f"{fid}_amplification"] = {"own_means": ownA.tolist(), "same_means": sameA.mean(1).tolist(), **cA}
                    rmA, ciA = sameA.mean(0), f"[{sm['ci_lo']:.3f}, {sm['ci_hi']:.3f}]"
                    extra = lambda k, rmA=rmA, p=cA["p_empirical"], c=ciA: {"random_effect": float(rmA[k]), "p_value": p, "confidence_interval": c}
            all_rows += _cond_rows(E, base_row, f"amplify_x{a}", idx, e, extra)
        # ---- dose-response (candidate + random mean curve)
        mults = list(icfg.dose_multipliers)
        L = np.stack([E.run(idx, [fid], [m])[0]["new_logit"] for m in mults])
        dr = dose_response_stats(mults, L)
        rand_curves = []
        for r in rids[: icfg.random_dose_controls]:
            rand_curves.append([E.run(idx, [int(r)], [m])[0]["new_logit"].mean() for m in mults])
        dr["random_mean_curve"] = np.mean(rand_curves, 0).tolist() if rand_curves else None
        dr["random_curves"] = [list(map(float, c)) for c in rand_curves]
        dose_store[str(fid)] = dr
        rec.update({"dose_spearman_rho": dr["spearman_rho"], "dose_monotone_inc": dr["monotone_increasing"]})
        # ---- local vs global ablation
        gm = ab["logit_effect"]
        lg_rows.append({"feature_id": fid, "scope": "global", "mean_logit_effect": float(gm.mean()),
                        "ci_lo": sa["ci_lo"], "ci_hi": sa["ci_hi"], "mean_patches_modified": float(res["n_modified"].mean()),
                        "effect_per_patch": float(gm.mean() / max(res["n_modified"].mean(), 1e-9)),
                        "p_vs_global": np.nan, "mean_diff_vs_global": 0.0})
        for q in icfg.local_percentiles:
            e, r_ = E.run(idx, [fid], [0.0], local_percentile=q)
            m, lo, hi = bootstrap_ci(e["logit_effect"], nb, seed=fid + q)
            pt = paired_tests(e["logit_effect"] - gm)
            lg_rows.append({"feature_id": fid, "scope": f"local_p{q}", "mean_logit_effect": m, "ci_lo": lo, "ci_hi": hi,
                            "mean_patches_modified": float(r_["n_modified"].mean()),
                            "effect_per_patch": m / max(float(r_["n_modified"].mean()), 1e-9),
                            "p_vs_global": pt["p_wilcoxon"], "mean_diff_vs_global": pt["mean"]})
            all_rows += _cond_rows(E, base_row, f"local_ablation_p{q}", idx, e, no_extra)
        # ---- steering (single feature, global)
        for a in icfg.steer_alphas:
            e, r_ = E.run(idx, [fid], [a], "steer")
            m, lo, hi = bootstrap_ci(e["delta_logit"], nb, seed=fid + 99)
            steer_rows.append({"feature_id": fid, "type": "single", "alpha": a, "mean_delta_logit": m, "ci_lo": lo, "ci_hi": hi,
                               "top1_change_rate": float(e["flip"].mean()),
                               "mean_prob_change": float((e["new_prob"] - e["base_prob"]).mean()),
                               "max_norm_ratio": r_["max_norm_ratio"], "flagged": bool(r_["flagged"]), "n": len(idx)})
            all_rows += _cond_rows(E, base_row, f"steer_a{a}", idx, e, no_extra)
        recs.append(rec)
        log.info(f"[causal] feature {fid} ({concept}): ablation logit drop {sa['mean']:.3f} "
                 f"[{sa['ci_lo']:.3f},{sa['ci_hi']:.3f}] n={len(idx)}")

    # ---------------------------------------------------- multi-feature steering
    ms = cands[: icfg.n_multi_steer]
    if len(ms) >= 2:
        idx_u = E.sample_active(ms[0], mask=active[:, ms].any(1))
        for a in icfg.steer_alphas:
            if a == 0.0:
                continue
            comb, r_ = E.run(idx_u, ms, [a] * len(ms), "steer")
            singles = [E.run(idx_u, [f], [a], "steer")[0]["delta_logit"].mean() for f in ms]
            steer_rows.append({"feature_id": "+".join(map(str, ms)), "type": "multi", "alpha": a,
                               "mean_delta_logit": float(comb["delta_logit"].mean()), "sum_of_singles": float(np.sum(singles)),
                               "superadditivity": float(comb["delta_logit"].mean() - np.sum(singles)),
                               "top1_change_rate": float(comb["flip"].mean()), "max_norm_ratio": r_["max_norm_ratio"],
                               "flagged": bool(r_["flagged"]), "n": len(idx_u)})
    # ---------------------------------------------------- feature interactions
    inter_rows = []
    for a_, b_ in itertools.combinations(cands[: icfg.n_interaction_features], 2):
        both = active[:, a_] & active[:, b_]
        cos = float(torch.nn.functional.cosine_similarity(ctx.sae.W_dec[a_], ctx.sae.W_dec[b_], dim=0).item())
        if both.sum() < max(icfg.min_active_images, 2):
            inter_rows.append({"feature_a": a_, "feature_b": b_, "category": "insufficient_data", "n": int(both.sum()),
                               "decoder_cosine": cos})
            continue
        idx = E.sample_active(a_, mask=both)
        ea = E.run(idx, [a_], [0.0])[0]["logit_effect"]
        eb = E.run(idx, [b_], [0.0])[0]["logit_effect"]
        eab = E.run(idx, [a_, b_], [0.0, 0.0])[0]["logit_effect"]
        d = interaction_stats(ea, eb, eab, nb, a_ * 1000 + b_)
        d.update({"feature_a": a_, "feature_b": b_, "decoder_cosine": cos})
        inter_rows.append(d)
    # ---------------------------------------------------- spatial causal maps
    sp_rows, sp_store = [], {}
    for fid in cands[: icfg.n_spatial_features]:
        ids_ = E.sample_active(fid)[: icfg.n_spatial_images]
        for i in ids_:
            act, eff = patch_ablation_map(engine, hid[i:i + 1], fid, int(target[i]), icfg.batch_size)
            a_idx = act > 0
            rho = float(spearmanr(act[a_idx], eff[a_idx])[0]) if a_idx.sum() >= 5 and np.ptp(eff[a_idx]) > 0 else np.nan
            glob = E.run(np.array([i]), [fid], [0.0])[0]["logit_effect"][0]
            sp_rows.append({"feature_id": fid, "position": int(causal_pos[i]), "n_active_patches": int(a_idx.sum()),
                            "spearman_activation_vs_effect": rho, "sum_single_patch_effects": float(eff.sum()),
                            "global_effect": float(glob), "max_patch_effect": float(eff.max()),
                            "top_effect_patch": int(eff.argmax())})
            sp_store[f"{fid}_{int(causal_pos[i])}_act"] = act; sp_store[f"{fid}_{int(causal_pos[i])}_eff"] = eff

    # ---------------------------------------------------- tables, rankings, hypotheses inputs
    cd = pd.DataFrame(recs)
    if "abl_rand_p_empirical" in cd:
        cd["abl_p_holm_paired"] = holm_correction(cd["abl_rand_paired_p_wilcoxon"].fillna(1.0).values)
    if "amp_rand_p_empirical" in cd:
        cd["amp_p_holm_paired"] = holm_correction(cd["amp_rand_paired_p_wilcoxon"].fillna(1.0).values)
    cd["causal_rank"] = cd["abl_mean"].rank(ascending=False, method="min")
    cd["interp_rank"] = cd["interp_score"].rank(ascending=False, method="min")
    pct = lambda s: s.rank(pct=True)
    comps = [pct(cd["abl_mean"]), pct(cd["interp_score"])]
    if "abl_rand_z_vs_random" in cd and cd["abl_rand_z_vs_random"].notna().any():
        comps.append(pct(cd["abl_rand_z_vs_random"].fillna(cd["abl_rand_z_vs_random"].min())))
    cd["combined_score"] = np.mean(comps, axis=0)       # equal weights on percentile ranks (no tuned weights)
    cd["combined_rank"] = cd["combined_score"].rank(ascending=False, method="min")
    cd.to_csv(out / "candidate_causal_summary.csv", index=False)
    pd.DataFrame(lg_rows).to_csv(out / "local_vs_global.csv", index=False)
    pd.DataFrame(steer_rows).to_csv(out / "steering.csv", index=False)
    pd.DataFrame(inter_rows).to_csv(out / "interactions.csv", index=False)
    pd.DataFrame(sp_rows).to_csv(out / "spatial_summary.csv", index=False)
    np.savez_compressed(out / "spatial_maps.npz", **sp_store)
    save_json(rc_store, out / "random_controls.json"); save_json(dose_store, out / "dose_response.json")
    write_results(pd.DataFrame(all_rows), exp_id)
    pd.DataFrame(all_rows).to_csv(out / "results_rows.csv", index=False)
    # ---- Grad-CAM comparison
    try:
        from causal.gradcam import make_gradcam_figures
        g_df = make_gradcam_figures(ctx, hid, target, active, causal_pos, cands, icfg.n_gradcam_figs, plots, log)
        g_df.to_csv(out / "gradcam_comparison.csv", index=False)
    except Exception as e:
        log.warning(f"[causal] Grad-CAM comparison skipped: {e}")
    # ---- figures
    try:
        from visualization import plots as V
        V.plot_dose_response(dose_store, cd, plots / "dose_response.png")
        V.plot_random_controls(rc_store, cd, plots / "random_controls.png")
        V.plot_causal_distribution(pd.DataFrame(all_rows), plots / "causal_effect_distribution.png")
        V.plot_interaction_matrix(pd.DataFrame(inter_rows), plots / "feature_interaction.png")
        V.plot_steering(pd.DataFrame(steer_rows), plots / "steering_curves.png")
        V.plot_local_global(pd.DataFrame(lg_rows), plots / "local_vs_global.png")
        V.plot_ablation_amplification(cd, plots / "ablation_amplification.png")
    except Exception as e:
        log.warning(f"[causal] some figures skipped: {e}")
    # ---- layer-level summary for cross-run comparison
    res_sae = load_json(ctx.run_dir / "results.json")["metrics"]
    dsum = load_json(ctx.disc_dir / "discovery_summary.json") if (ctx.disc_dir / "discovery_summary.json").exists() else {}
    summary = {"run_tag": cfg.run_tag(), "layer": cfg.layer, "sae_type": cfg.sae.sae_type, "width": cfg.sae.width,
               "k": cfg.sae.k if cfg.sae.sae_type == "topk" else None, **res_sae, **dsum,
               "n_candidates": len(cands), "mean_ablation_logit_effect": float(cd["abl_mean"].mean()),
               "median_ablation_logit_effect": float(cd["abl_mean"].median()),
               "mean_random_effect": float(cd["abl_rand_rand_mean"].mean()) if "abl_rand_rand_mean" in cd else None,
               "mean_amp_delta_logit": float(cd["amp_mean"].mean()),
               "selection_note": "candidates selected from validation-split, non-causal criteria only",
               "warnings": warnings}
    save_json(summary, ctx.run_dir / "layer_summary.json")
    save_json(sorted(set(warnings)), ctx.run_dir / "warnings.json")
    log.info(f"[causal] done -> {out}  ({len(set(warnings))} warnings)")
    return out
