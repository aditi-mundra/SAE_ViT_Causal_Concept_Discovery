"""Torch unit tests + end-to-end smoke test (offline: tiny random ViT + synthetic data). Requires torch."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from activations.normalize import Normalizer
from config import get_config
from interventions.engine import InterventionEngine
from models.vit import TinyViT, capture, forward_from_hidden
from sae.model import SparseAutoencoder, sae_loss


def test_sae_encoder_decoder_shapes():
    sae = SparseAutoencoder(16, 64, "l1")
    x = torch.randn(10, 16)
    xh, f = sae(x)
    assert f.shape == (10, 64) and xh.shape == x.shape and (f >= 0).all()
    assert torch.allclose(sae.decode(f), f @ sae.W_dec + sae.b_dec)
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-5)


@pytest.mark.parametrize("k", [4, 16])
def test_topk_operation(k):
    sae = SparseAutoencoder(16, 64, "topk", k=k)
    f = sae.encode(torch.randn(20, 16))
    assert ((f > 0).sum(-1) <= k).all()
    pre = torch.relu(sae.pre_acts(torch.randn(1, 16)))
    assert sae.k == k and pre.shape == (1, 64)


def test_loss_decreases():
    torch.manual_seed(0)
    sae, x = SparseAutoencoder(16, 64, "topk", k=8), torch.randn(256, 16)
    opt = torch.optim.Adam(sae.parameters(), 1e-2)
    l0 = sae_loss(sae, x)[0].item()
    for _ in range(100):
        loss, _ = sae_loss(sae, x); opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < l0


def test_normalizer_delta_roundtrip():
    x = torch.randn(5, 7, 16) * 3 + 1
    n = Normalizer("standardize", x.mean((0, 1)), x.std((0, 1)))
    d = torch.randn_like(x)
    assert torch.allclose(n(x + n.delta_to_raw(d, x)), n(x) + d, atol=1e-4)


def test_forward_from_hidden_matches_full_forward():
    m = TinyViT(depth=4).eval()
    img = torch.randn(2, 3, 224, 224)
    logits, hid = capture(m, img, [1])
    assert torch.allclose(forward_from_hidden(m, hid[1], 1), logits, atol=1e-5)
    assert hid[1].shape == (2, 197, 64)


def test_intervention_equation_and_error_preservation():
    torch.manual_seed(0)
    m = TinyViT(depth=4).eval()
    sae = SparseAutoencoder(64, 128, "topk", k=8).eval()
    norm = Normalizer("standardize", torch.randn(64) * .1, torch.rand(64) + .5)
    cfg = get_config("selftest")
    eng = InterventionEngine(m, sae, norm, 1, 1, cfg.intervention)
    _, hid = capture(m, torch.randn(3, 3, 224, 224), [1]); h = hid[1]
    fid, alpha = 5, 0.0
    f = eng.encode_features(h, [fid])[..., 0]
    x = h[:, 1:]
    manual_delta_n = ((alpha * f - f)[..., None]) * sae.W_dec[fid]
    x_new = x + norm.delta_to_raw(manual_delta_n, x)
    expect = forward_from_hidden(m, torch.cat([h[:, :1], x_new], 1), 1)
    got, info = eng.run(h, [fid], [alpha], "scale")
    assert torch.allclose(got, expect, atol=1e-5)
    # alpha=1 leaves the model untouched (SAE reconstruction error is preserved, not substituted)
    same, _ = eng.run(h, [fid], [1.0], "scale")
    assert torch.allclose(same, forward_from_hidden(m, h, 1), atol=1e-6)
    # steering adds alpha * W_dec to every patch
    st, _ = eng.run(h, [fid], [2.0], "steer")
    x_s = x + norm.delta_to_raw((2.0 * sae.W_dec[fid]).expand_as(x), x)
    assert torch.allclose(st, forward_from_hidden(m, torch.cat([h[:, :1], x_s], 1), 1), atol=1e-5)
    # clamping and safety info
    _, info = eng.run(h, [fid], [1e6], "steer")
    assert info["clamped"]


def test_local_mask_subset_of_global():
    f = torch.rand(2, 50, 1) * (torch.rand(2, 50, 1) > .5)
    loc = InterventionEngine._local_mask(f, 90)
    assert (loc <= (f > 0)).all() and loc.sum() <= (f > 0).sum()


def test_end_to_end_selftest(tmp_path):
    """Whole pipeline on 80 synthetic images: extract -> train -> discover -> causal evaluation."""
    from activations.extract import run_extraction
    from causal.compare import build_comparison
    from causal.evaluate import run_causal_evaluation
    from causal.hypotheses import evaluate_hypotheses
    from features.discover import run_discovery
    from sae.train import train_sae
    cfg = get_config("selftest")
    cfg.name, cfg.extract.out_dir, cfg.device = "pytest_selftest", str(tmp_path / "store"), "cpu"
    run_extraction(cfg, force=True)
    rd = train_sae(cfg)
    assert (rd / "checkpoint.pt").exists() and (rd / "metrics.csv").exists() and (rd / "experiment_config.json").exists()
    run_discovery(cfg)
    assert (rd / "discovery" / "feature_summary.csv").exists()
    run_causal_evaluation(cfg)
    assert (rd / "causal" / "candidate_causal_summary.csv").exists() and (rd / "layer_summary.json").exists()
    evaluate_hypotheses(cfg, build_comparison(cfg))
