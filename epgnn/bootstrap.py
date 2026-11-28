"""Paired weekly block bootstrap of the MAE difference between two models (or two groups of seeds).

    python -m epgnn.bootstrap c3s_gat_interconnects_curves_rolling c3s_gat_self_curves_rolling
    python -m epgnn.bootstrap A_s0,A_s1,A_s2 B_s0,B_s1,B_s2 --split test --static --n 5000

Both sides are scored on the same (zone, hour) cells: those valid in the split's samples and predicted
by every run. With several runs per side, a cell's error is the mean of the runs' absolute errors (so
the result is the difference of the seed-averaged MAEs, not of an ensemble).

Errors within a week are strongly correlated (weather regimes, outages, gas prices), so the resampling
unit is a block of `block` consecutive delivery days, all zones together. The statistic is
MAE_A - MAE_B over the resampled cells (a ratio of sums, so days with more valid hours weigh more,
exactly as in the headline MAE). Negative = A is better.

Reads runs/<name>/pred_<split>.npy (or pred_<split>_static.npy with --static, which the rolling runs
of gat_scope save). Writes nothing unless --out is given.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from epgnn.data import samples as S
from epgnn.metrics import slices
from epgnn.paths import ROOT


def day_sums(err_a: np.ndarray, err_b: np.ndarray, ok: np.ndarray, day_idx: np.ndarray, n_days: int):
    """Per-day sums of |e_A|, |e_B| and the cell count. err_*: (n, H) absolute errors; ok: (n, H)."""
    w = ok.sum(1)
    sa = np.bincount(day_idx, np.where(ok, err_a, 0).sum(1), n_days)
    sb = np.bincount(day_idx, np.where(ok, err_b, 0).sum(1), n_days)
    cnt = np.bincount(day_idx, w, n_days)
    return sa, sb, cnt


def block_bootstrap(sa: np.ndarray, sb: np.ndarray, cnt: np.ndarray, block: int = 7, n: int = 2000,
                    seed: int = 0) -> dict:
    """Moving-block bootstrap over days of (sum sa - sum sb) / sum cnt."""
    d = len(cnt)
    delta = float((sa.sum() - sb.sum()) / cnt.sum())
    n_blocks = int(np.ceil(d / block))
    starts = np.arange(d - block + 1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(starts, size=(n, n_blocks))
    idx = (pick[..., None] + np.arange(block)).reshape(n, -1)[:, :d]           # (n, d) resampled days
    boot = (sa[idx].sum(1) - sb[idx].sum(1)) / cnt[idx].sum(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"delta": delta, "ci95": [float(lo), float(hi)], "se": float(boot.std(ddof=1)),
            "p_a_not_better": float((boot >= 0).mean()), "mae_a": float(sa.sum() / cnt.sum()),
            "mae_b": float(sb.sum() / cnt.sum()), "days": d, "block": block, "n": n}


def load_preds(names: list[str], split: str, static: bool) -> list[np.ndarray]:
    suffix = "_static" if static else ""
    return [np.load(ROOT / "runs" / nm / f"pred_{split}{suffix}.npy") for nm in names]


def compare(a: list[str], b: list[str], split: str = "test", static: bool = False, block: int = 7,
            n: int = 2000, slice_names: list[str] | None = None) -> dict:
    s = S.build(split)
    pa, pb = load_preds(a, split, static), load_preds(b, split, static)
    for p in pa + pb:
        assert p.shape == s.y.shape, f"prediction shape {p.shape} != samples {s.y.shape} (zone subset?)"
    ok = s.mask & np.isfinite(s.y)
    for p in pa + pb:
        ok &= np.isfinite(p)
    ea = np.mean([np.abs(p - s.y) for p in pa], axis=0)
    eb = np.mean([np.abs(p - s.y) for p in pb], axis=0)
    days, day_idx = np.unique(s.day, return_inverse=True)
    out = {"a": a, "b": b, "split": split, "static": static, "slices": {}}
    for name, rows in slices(s).items():
        if slice_names and name not in slice_names:
            continue
        m = ok & rows[:, None]
        if not m.any():
            continue
        sa, sb, cnt = day_sums(ea, eb, m, day_idx, len(days))
        keep = cnt > 0
        out["slices"][name] = block_bootstrap(sa[keep], sb[keep], cnt[keep], block, n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a", help="run name(s) of model A, comma-separated for seeds")
    ap.add_argument("b", help="run name(s) of model B")
    ap.add_argument("--split", default="test")
    ap.add_argument("--static", action="store_true", help="use pred_<split>_static.npy")
    ap.add_argument("--block", type=int, default=7)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--slices", default=None, help="comma-separated slice names (default all)")
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    args = ap.parse_args()
    r = compare(args.a.split(","), args.b.split(","), args.split, args.static, args.block, args.n,
                args.slices.split(",") if args.slices else None)
    print(f"A = {r['a']}\nB = {r['b']}\nsplit={r['split']} static={r['static']}  (negative delta = A better)")
    for name, x in r["slices"].items():
        print(f"  {name:13s} MAE A {x['mae_a']:6.2f}  B {x['mae_b']:6.2f}  delta {x['delta']:+.3f}  "
              f"95% CI [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]  P(A not better) {x['p_a_not_better']:.3f}  "
              f"({x['days']} days)")
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
