"""CLI: python intervene.py --mode demo --feature 172 --strength 0 [--steer] [--local 90] [--image-position N]
Runs one intervention on one held-out causal-split image and prints before/after logits (top-5)."""
import argparse
import numpy as np
from cli import add_common_args, config_from_args
from context import RunContext

if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--feature", type=int, required=True)
    ap.add_argument("--strength", type=float, default=0.0, help="scale multiplier (or steering alpha with --steer)")
    ap.add_argument("--steer", action="store_true")
    ap.add_argument("--local", type=float, default=None, help="local percentile (50/75/90/95)")
    ap.add_argument("--image-position", type=int, default=None, help="store position (default: first causal image where feature is active)")
    a = ap.parse_args()
    ctx = RunContext(config_from_args(a)); eng = ctx.engine()
    pos = a.image_position
    if pos is None:
        for p in ctx.splits["causal"]:
            if (ctx.feature_activations(a.feature, [p]) > 0).any():
                pos = int(p); break
        if pos is None:
            raise SystemExit(f"feature {a.feature} is not active on any causal-eval image")
    h = ctx.hidden([pos])
    base = eng.baseline(h.to(ctx.device)).cpu().numpy()[0]
    lg, info = eng.run(h.to(ctx.device), [a.feature], [a.strength], "steer" if a.steer else "scale", a.local)
    new = lg.cpu().numpy()[0]
    names, y = ctx.vit_class_names, int(base.argmax())
    print(f"image position {pos} | target class: {names[y]}")
    print(f"target logit  before {base[y]:.4f}  after {new[y]:.4f}  effect(before-after) {base[y]-new[y]:.4f}")
    print("top-5 before:", [(names[i], round(float(base[i]), 3)) for i in np.argsort(-base)[:5]])
    print("top-5 after :", [(names[i], round(float(new[i]), 3)) for i in np.argsort(-new)[:5]])
    print("safety:", {k: (round(v, 4) if isinstance(v, float) else (v.tolist() if hasattr(v, 'tolist') else v)) for k, v in info.items()})
