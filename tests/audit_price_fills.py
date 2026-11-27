"""Where do gap-filled price lags come from, the past or the future? Run from the repo root with PYTHONPATH=."""
import numpy as np, pandas as pd
from epgnn.data import tables
n = tables.nodes(); H = pd.Timedelta(hours=1)
idx = n.set_index(["zone", "ts_utc"])
def at(col, shift):                       # value of `col` `shift` hours before each row
    return idx[col].reindex(pd.MultiIndex.from_arrays([n.zone, n.ts_utc - shift * H])).to_numpy(dtype="float64")
a, b = n.price_lag24.to_numpy(dtype="float64"), at("price", 24)
fill = np.isfinite(a) & ~np.isfinite(b)
past = {f"same hour {k} days earlier": at("price", 24 + 24 * k) for k in range(1, 15)}
future = {f"same hour {k} days LATER": at("price", 24 - 24 * k) for k in range(1, 8)}
near_past = {f"{k} h earlier": at("price", 24 + k) for k in range(1, 7)}
near_future = {f"{k} h LATER": at("price", 24 - k) for k in range(1, 7)}
left = fill.copy(); tally = {}
for group, cands in [("past day", past), ("past hour", near_past), ("FUTURE day", future), ("FUTURE hour", near_future)]:
    for name, c in cands.items():
        m = left & np.isfinite(c) & (np.abs(a - c) < 1e-3)
        if m.any(): tally[f"{group}: {name}"] = int(m.sum()); left &= ~m
print("filled rows:", int(fill.sum())); [print("  ", k, v) for k, v in tally.items()]
print("still unexplained:", int(left.sum()))
u = n[left]
print(u.groupby("zone").agg(rows=("ts_utc", "size"), first=("ts_utc", "min"), last=("ts_utc", "max")).to_string())
# for the unexplained rows: is the value a blend of earlier and later days (interpolation)?
p_prev, p_next = at("price", 48), at("price", 0)
z = left & np.isfinite(p_prev) & np.isfinite(p_next)
print("unexplained rows where both neighbours (t-48h and t) exist:", int(z.sum()))
if z.any():
    w = (a[z] - p_prev[z]) / np.where(np.abs(p_next[z] - p_prev[z]) < 1e-6, np.nan, p_next[z] - p_prev[z])
    print("  weight on the LATER price if it were a blend: median", np.nanmedian(w).round(3), "| share within [0.05,0.95]:", float(np.nanmean((w > 0.05) & (w < 0.95))).__round__(3))
# in the splits we train and score on, how many filled lags are there at all?
print("filled price_lag24 rows by split:", n[fill].split.value_counts().to_dict(), "| unexplained by split:", u.split.value_counts().to_dict())
