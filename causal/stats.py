"""
causal/stats.py [CORE] (numpy/scipy only)
Bootstrap CIs, paired tests, effect sizes, random-control comparison, dose-response, interaction, Holm correction.
Effect size = Cohen's d_z = mean(x) / sd(x) for paired/one-sample differences.
Interaction  I = E_AB - E_A - E_B  (E = logit drop):  I<0 sub-additive (redundant), I>0 super-additive (synergistic).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def bootstrap_ci(x, n_boot: int = 1000, ci: float = 0.95, seed: int = 0, stat=np.mean):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    boots = stat(x[rng.randint(0, len(x), size=(n_boot, len(x)))], axis=1)
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(stat(x)), float(lo), float(hi)


def paired_tests(d) -> dict:
    """Tests that the paired differences d have zero centre (paired t-test and Wilcoxon signed-rank)."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    out = {"n": n, "mean": float(d.mean()) if n else np.nan, "dz": np.nan, "p_t": np.nan, "p_wilcoxon": np.nan}
    if n < 2:
        return out
    sd = d.std(ddof=1)
    out["dz"] = float(d.mean() / sd) if sd > 0 else (0.0 if d.mean() == 0 else float("inf"))
    out["p_t"] = float(sps.ttest_1samp(d, 0.0).pvalue) if sd > 0 else (1.0 if d.mean() == 0 else 0.0)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pw = float(sps.wilcoxon(d).pvalue)
        out["p_wilcoxon"] = 1.0 if np.isnan(pw) else pw
    except ValueError:  # all differences zero
        out["p_wilcoxon"] = 1.0
    return out


def compare_to_random(cand_eff, rand_same, rand_own_means, n_boot: int = 1000, seed: int = 0) -> dict:
    """
    cand_eff        [n_img]    candidate per-image effects
    rand_same       [R, n_img] random features evaluated on the SAME images (paired comparison)
    rand_own_means  [R]        mean effect of each random feature on its OWN active images (activity-matched null)
    Empirical two-sided p = (1 + #{|random| >= |candidate|}) / (R + 1).
    """
    cand_eff = np.asarray(cand_eff, dtype=np.float64)
    rand_same = np.asarray(rand_same, dtype=np.float64).reshape(-1, len(cand_eff))
    own = np.asarray(rand_own_means, dtype=np.float64)
    R = len(own)
    cm = float(np.nanmean(cand_eff))
    _, lo, hi = bootstrap_ci(own, n_boot, seed=seed)
    diff = cand_eff - np.nanmean(rand_same, axis=0)
    pt = paired_tests(diff)
    _, dlo, dhi = bootstrap_ci(diff, n_boot, seed=seed + 1)
    sd = float(np.nanstd(own, ddof=1)) if R > 1 else np.nan
    return {"cand_mean": cm, "rand_mean": float(np.nanmean(own)), "rand_std": sd, "rand_median": float(np.nanmedian(own)),
            "rand_ci95": [lo, hi], "n_random": R,
            "z_vs_random": float((cm - np.nanmean(own)) / sd) if sd and sd > 0 else np.nan,
            "p_empirical": float((1 + np.sum(np.abs(own) >= abs(cm))) / (R + 1)),
            "p_empirical_same_images": float((1 + np.sum(np.abs(np.nanmean(rand_same, axis=1)) >= abs(cm))) / (len(rand_same) + 1)),
            "paired_p_t": pt["p_t"], "paired_p_wilcoxon": pt["p_wilcoxon"], "paired_dz": pt["dz"],
            "paired_diff_mean": pt["mean"], "paired_diff_ci95": [dlo, dhi]}


def holm_correction(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def dose_response_stats(multipliers, logits_matrix) -> dict:
    """logits_matrix [n_mult, n_img] target logits. Reports mean, bootstrap CI, Spearman rho, monotonicity."""
    m = np.asarray(multipliers, dtype=np.float64)
    L = np.asarray(logits_matrix, dtype=np.float64)
    means = L.mean(1)
    cis = [bootstrap_ci(L[i], 500, seed=i)[1:] for i in range(len(m))]
    rho, p = sps.spearmanr(m, means) if len(m) > 2 and np.ptp(means) > 0 else (np.nan, np.nan)
    tol = 1e-9
    d = np.diff(means[np.argsort(m)])
    return {"multipliers": m.tolist(), "mean_logit": means.tolist(), "ci_lo": [c[0] for c in cis],
            "ci_hi": [c[1] for c in cis], "spearman_rho": float(rho), "spearman_p": float(p),
            "monotone_increasing": bool(np.all(d >= -tol)), "monotone_decreasing": bool(np.all(d <= tol))}


def interaction_stats(e_a, e_b, e_ab, n_boot: int = 1000, seed: int = 0) -> dict:
    e_a, e_b, e_ab = (np.asarray(v, dtype=np.float64) for v in (e_a, e_b, e_ab))
    inter = e_ab - e_a - e_b
    mean, lo, hi = bootstrap_ci(inter, n_boot, seed=seed)
    pt = paired_tests(inter)
    if np.isnan(lo):
        cat = "insufficient_data"
    elif lo > 0:
        cat = "synergistic"
    elif hi < 0:
        cat = "redundant"
    else:
        cat = "independent_or_additive"
    return {"effect_a": float(e_a.mean()), "effect_b": float(e_b.mean()), "effect_ab": float(e_ab.mean()),
            "interaction": mean, "ci_lo": lo, "ci_hi": hi, "p_t": pt["p_t"], "p_wilcoxon": pt["p_wilcoxon"],
            "category": cat, "n": int(len(inter))}
