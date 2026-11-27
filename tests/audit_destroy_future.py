"""Destroy the future: overwrite everything that cannot be known at the cutoff of delivery day D0 with junk,
rebuild the model inputs for D0 and count what changed. Run from the repo root with PYTHONPATH=."""
import numpy as np, pandas as pd, torch
from epgnn.data import features as F, samples as S, tables, graph as G
from epgnn.models import gru_mlp as M
from epgnn.models.gat import ZoneGAT
from epgnn.splits import splits

FULL = tables.nodes(); ZONES = sorted(FULL.zone.unique()); H = pd.Timedelta(hours=1)
REAL = ["net_import", "temp_pw", "temp_anom", "wind_cf", "low_wind", "solar_cf"]
WLAG = ["temp_pw_lag24", "temp_anom_lag24", "wind_cf_lag24", "solar_cf_lag24"]
OWN = F.FEATURE_SETS["own"]; KNOWN = [i for i, c in enumerate(OWN) if c in F.KNOWN_AT_CUTOFF]

def split_of(day):
    return next(k for k, (a, b) in splits().items() if a <= day <= b)

def corrupt(sub, day, strict):
    """strict: nothing after the cutoff survives. Not strict: keep the two documented assumptions
    (lag-24 weather for the hours of D0; realised values of the hour that starts at the cutoff)."""
    s2 = sub.copy(); t, dd = s2.ts_utc, s2.delivery_date
    cut = pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=1) + pd.Timedelta(hours=S.CUTOFF_UTC)
    start = t[dd == day].min(); rng = np.random.default_rng(1)
    def junk(mask, cols):
        for c in cols:
            s2[c] = s2[c].astype("float32")
            s2.loc[mask, c] = (9000 + 100 * rng.random(int(mask.sum()))).astype("float32")
    junk(dd >= day, ["price"])                                   # prices of D0 and later: not auctioned yet
    junk(t - 24 * H >= start, ["price_lag24"])                   # stored lags whose source hour is in D0 or later
    junk(t - 168 * H >= start, ["price_lag168"])
    junk(dd > day, ["load_fc"])                                  # forecasts for days after D0
    junk((t >= cut) if strict else (t > cut), REAL)              # observed values of hours not finished at the cutoff
    junk((t - 24 * H >= cut) if strict else (dd > day), WLAG)
    junk(t - 48 * H >= cut, ["net_import_lag48"])
    junk(t >= cut, ["n_presolved", "min_ram_ratio"])
    return s2

def inputs(table, day):
    tables.nodes = lambda: table                                 # samples.py reads the table through this
    s = S.build(split_of(day), ZONES); k = s.day == day
    s = S.Samples(s.zone[k], s.day[k], s.y[k], s.mask[k], s.hours[k])
    x = S.day_inputs(s, OWN)
    keys = pd.MultiIndex.from_arrays([np.repeat(s.zone, S.H)[s.hours.ravel() >= 0], pd.to_datetime(s.hours.ravel()[s.hours.ravel() >= 0], utc=True)])
    unguarded = np.full(s.hours.size, np.nan, "float32"); unguarded[s.hours.ravel() >= 0] = table.set_index(["zone", "ts_utc"]).price_lag24.reindex(keys).to_numpy("float32")
    return {"hist": S.history(s, F.HISTORY, 336), "xday": x, "base": M._base(s), "unguarded_lag24": unguarded.reshape(len(s), S.H)}, s

def changed(a, b):
    return ~((a == b) | (np.isnan(a) & np.isnan(b)))

cnt = FULL[FULL.zone == "FR"].groupby("delivery_date").size()
dst = [d for d in cnt[cnt != 24].index if "2016-03-01" <= d <= "2026-09-01"]
after = [(pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") for d in dst]
rnd = list(np.random.default_rng(7).choice([d for d in cnt.index if "2016-03-01" <= d <= "2026-09-01"], 40, replace=False))
days = sorted(set(str(d) for d in dst + after + rnd)); long_days = set(cnt[cnt == 25].index)
print(f"{len(days)} delivery days x {len(ZONES)} zones: {len([d for d in dst if d in long_days])} 25-hour days, {len(dst) - len([d for d in dst if d in long_days])} 23-hour days, {len(after)} days after a clock change, 40 random")

tot = {m: {"xday": np.zeros(len(OWN), int), "hist_last": np.zeros(len(F.HISTORY), int), "hist_older": np.zeros(len(F.HISTORY), int), "base": 0, "unguarded": 0, "pred": 0.0} for m in ("lenient", "strict")}
torch.manual_seed(0); model = ZoneGAT(len(F.HISTORY), len(OWN), mix_cols=KNOWN).eval()
zz = torch.as_tensor(G.zone_graph(ZONES))
def predict(d):
    hist = torch.as_tensor(np.nan_to_num(d["hist"][:, -168:]) / 100)[None]; x = torch.as_tensor(np.nan_to_num(d["xday"]) / 100)[None]
    with torch.no_grad(): return model(hist, x, {"zz": zz}).numpy()
n_elem = 0
for day in days:
    lo, hi = pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=25), pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=9)
    sub = FULL[(FULL.ts_utc >= lo) & (FULL.ts_utc <= hi)]
    clean, s = inputs(sub, day); p0 = predict(clean); n_elem += clean["xday"].size + clean["hist"][:, -168:].size
    for mode in ("lenient", "strict"):
        bad, _ = inputs(corrupt(sub, day, mode == "strict"), day); T = tot[mode]
        T["xday"] += changed(clean["xday"], bad["xday"]).sum(axis=(0, 1))
        hc = changed(clean["hist"], bad["hist"])
        T["hist_last"] += hc[:, -1].sum(axis=0); T["hist_older"] += hc[:, :-1].sum(axis=(0, 1))
        T["base"] += int(changed(clean["base"], bad["base"]).sum())
        T["unguarded"] += int(changed(clean["unguarded_lag24"], bad["unguarded_lag24"]).sum())
        T["pred"] = max(T["pred"], float(np.abs(predict(bad) - p0).max()))
tables.nodes = lambda: FULL
print(f"input values checked: {n_elem:,}")
for mode, T in tot.items():
    print(f"\n== {mode}: input values that changed when the future was destroyed")
    print("  day-D inputs :", {c: int(v) for c, v in zip(OWN, T["xday"])})
    print("  -> columns the cross-hour layer reads:", int(T["xday"][KNOWN].sum()))
    print("  history, hour starting at the cutoff:", {c: int(v) for c, v in zip(F.HISTORY, T["hist_last"])})
    print("  history, all earlier hours          :", int(T["hist_older"].sum()))
    print("  base price of the target            :", T["base"])
    print("  largest change in any forecast (model with the cross-hour layer, random weights):", T["pred"])
    print("  [control] price_lag24 read straight from the table, without the clock-change fix:", T["unguarded"])
