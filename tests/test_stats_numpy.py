"""Numpy-only unit tests: causal metric calculation, random control, statistics, heatmaps."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from causal.metrics import compute_effects
from causal.stats import (bootstrap_ci, compare_to_random, dose_response_stats, holm_correction,
                          interaction_stats, paired_tests)
from features.heatmaps import normalize_map, patch_map, save_feature_heatmaps, strongest_bbox, threshold_mask
from utils import entropy_bits


def test_causal_metric_calculation():
    rng = np.random.RandomState(0)
    base = rng.randn(8, 10); t = base.argmax(1); new = base.copy(); new[np.arange(8), t] -= 3.0
    e = compute_effects(base, new, t)
    assert np.allclose(e["logit_effect"], 3.0) and np.allclose(e["delta_logit"], -3.0)
    assert (e["prob_effect"] > 0).all() and (e["rank_change"] >= 0).all()


def test_random_control_separates_strong_effect():
    rng = np.random.RandomState(1)
    cand = rng.randn(40) * 0.2 + 2.0
    rs = rng.randn(25, 40) * 0.2
    r = compare_to_random(cand, rs, rs.mean(1), 200)
    assert r["p_empirical"] < 0.06 and r["z_vs_random"] > 5 and r["paired_p_wilcoxon"] < 1e-3


def test_random_control_null_not_significant():
    rng = np.random.RandomState(2)
    rs = rng.randn(25, 40) * 0.2
    r = compare_to_random(rng.randn(40) * 0.2, rs, rs.mean(1), 200)
    assert r["p_empirical"] > 0.05


def test_bootstrap_and_tests():
    x = np.random.RandomState(3).randn(200) + 1
    m, lo, hi = bootstrap_ci(x, 500)
    assert lo < m < hi and lo > 0.7
    assert paired_tests(x)["p_t"] < 1e-6 and paired_tests(np.zeros(5))["p_t"] == 1.0
    assert np.allclose(holm_correction([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_dose_response_and_interaction():
    d = dose_response_stats([0, 1, 2], np.array([[0.] * 4, [1.] * 4, [2.] * 4]))
    assert d["monotone_increasing"] and d["spearman_rho"] == 1.0
    rng = np.random.RandomState(4)
    a, b = np.full(30, 2.0) + rng.randn(30) * .1, np.full(30, 2.0) + rng.randn(30) * .1
    assert interaction_stats(a, b, np.full(30, 2.5) + rng.randn(30) * .1)["category"] == "redundant"
    assert interaction_stats(a, b, a + b + 1.0)["category"] == "synergistic"


def test_entropy():
    assert np.isclose(entropy_bits(np.array([.5, .5])), 1.0) and np.isclose(entropy_bits(np.array([1., 0.])), 0.0)


def test_heatmap_generation(tmp_path):
    act = np.zeros(196); act[30] = 5; act[31] = 4
    m = patch_map(act, 14)
    assert m.shape == (14, 14) and normalize_map(m).max() == 1.0
    assert threshold_mask(normalize_map(m), 0.5).sum() == 2
    x0, y0, x1, y1 = strongest_bbox(normalize_map(m), 0.5, 16)
    assert (x1 - x0, y1 - y0) == (32, 16)
    paths = save_feature_heatmaps(Image.new("RGB", (64, 64), "gray"), act, 14, 16, tmp_path, "t")
    assert len(paths) == 5 and all(Path(p).exists() for p in paths.values())
