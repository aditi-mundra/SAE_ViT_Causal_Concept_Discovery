"""causal/results_db.py [CORE]: centralised experiments/results.csv (one row per feature x image x condition)."""
from __future__ import annotations

import pandas as pd

from config import PROJECT_ROOT

COLUMNS = ["experiment_id", "layer", "sae_type", "sae_width", "feature_id", "concept", "image_id", "class",
           "dataset_class", "activation", "purity", "clip_score", "spatial_score", "condition", "baseline_logit",
           "ablated_logit", "logit_effect", "baseline_probability", "ablated_probability", "probability_effect",
           "flip", "rank_change", "random_effect", "p_value", "confidence_interval", "flagged"]


def write_results(df: pd.DataFrame, experiment_id: str, path=None):
    path = path or PROJECT_ROOT / "experiments" / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.reindex(columns=COLUMNS)
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["experiment_id"] != experiment_id]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    return path
