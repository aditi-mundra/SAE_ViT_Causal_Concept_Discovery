"""sae/health.py [CORE]: automatic failure detection (warnings shown in the dashboard)."""
from __future__ import annotations

from typing import Dict, List


def check_sae(m: Dict, width: int) -> List[str]:
    w = []
    ev = m.get("explained_variance", 1.0)
    if ev < 0.7:
        w.append(f"SAE reconstruction is poor (explained variance {ev:.3f} < 0.70)")
    if m.get("dead_pct_eval", 0) > 30:
        w.append(f"Too many dead features ({m['dead_pct_eval']:.1f}% never active on the validation set)")
    if m.get("l0", 0) > 0.1 * width:
        w.append(f"Features are too dense (L0={m['l0']:.1f} > 10% of width {width})")
    return w


def check_intervention(info: Dict) -> List[str]:
    w = []
    if info.get("nan"):
        w.append("Intervention produced NaN/Inf")
    if info.get("flagged") and not info.get("nan"):
        w.append(f"Extreme activation after intervention (max norm ratio {info.get('max_norm_ratio', 0):.2f})")
    if info.get("clamped"):
        w.append("Intervention strength was clamped to the safety limit")
    return w
