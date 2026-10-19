import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from config import get_config, ExperimentConfig
from utils import make_splits


@pytest.mark.parametrize("mode", ["smoke", "demo", "full"])
def test_presets_validate(mode):
    get_config(mode)


def test_roundtrip(tmp_path):
    cfg = get_config("demo")
    p = cfg.save(tmp_path / "c.json")
    cfg2 = ExperimentConfig.load(p)
    assert cfg2.sae.k == cfg.sae.k
    assert cfg2.data.split_fractions == cfg.data.split_fractions
    assert cfg2.intervention.dose_multipliers == cfg.intervention.dose_multipliers


def test_bad_topk():
    cfg = get_config("demo")
    cfg.sae.k = 10 ** 6
    with pytest.raises(ValueError):
        cfg.validate()


def test_splits_disjoint_and_complete():
    a, b, c = make_splits(1000, (0.6, 0.2, 0.2), seed=1)
    assert not (set(a) & set(b)) and not (set(a) & set(c)) and not (set(b) & set(c))
    assert len(a) + len(b) + len(c) == 1000
    a2, _, _ = make_splits(1000, (0.6, 0.2, 0.2), seed=1)
    assert np.array_equal(a, a2)


def test_grid():
    cfg = get_config("demo")
    assert cfg.model.grid == 14 and cfg.model.num_patches == 196
