"""Read-only audit of the release label tables: USDM county-week, NASS state-week pasture condition, hurricane tracks
and landfalls. Completeness of the weekly series, internal consistency, and the major Florida landfalls of the
record compared with the NHC best tracks as summarised in the Tropical Cyclone Reports."""
import sys

import numpy as np
import pandas as pd

from fpcube import config

L = config.RELEASE / (sys.argv[1] if len(sys.argv) > 1 else "v1.0") / "labels"
issues = []
def issue(m): issues.append(m); print("  !!", m, flush=True)
def say(*a): print(*a, flush=True)

# ------------------------------------------------------------------ USDM
say("== USDM county-week")
u = pd.read_parquet(L / "usdm_county_weekly.parquet")
weeks = pd.DatetimeIndex(sorted(u.valid_start.unique()))
say(f"{u.county_fips.nunique()} counties, {len(weeks)} weeks, {weeks[0].date()} .. {weeks[-1].date()}")
if u.county_fips.nunique() != 67: issue(f"{u.county_fips.nunique()} counties, Florida has 67")
step = np.diff(weeks.values).astype("timedelta64[D]").astype(int)
if (step != 7).any(): issue(f"USDM weeks not every 7 days: gaps at {weeks[1:][step != 7].date.tolist()[:5]}")
full = pd.MultiIndex.from_product([u.county_fips.unique(), weeks])
miss = full.difference(pd.MultiIndex.from_arrays([u.county_fips, u.valid_start]))
if len(miss): issue(f"{len(miss)} county-weeks missing")
if u.duplicated(["county_fips", "valid_start"]).any(): issue("duplicate county-weeks")
if ((u.valid_end - u.valid_start).dt.days != 6).any(): issue("valid_end is not valid_start + 6 days")
if weeks[-1] + pd.Timedelta(days=6) < pd.Timestamp(config.END) - pd.Timedelta(days=7):
    issue(f"USDM ends {weeks[-1].date()}, before END {config.END}")
D = ["D0", "D1", "D2", "D3", "D4"]
c = u[["None"] + D]
if ((c < -1e-6) | (c > 100 + 1e-6)).any().any(): issue("USDM percentages outside [0, 100]")
s = (u["None"] + u["D0"] - 100).abs()
say(f"None + D0 - 100 (cumulative statistics): max |diff| {s.max():.3f}")
if s.max() > 0.05: issue("None + D0 != 100: statistics are not cumulative as assumed")
mono = (u[D].diff(axis=1).iloc[:, 1:] > 1e-6).any(axis=1)
if mono.any(): issue(f"{int(mono.sum())} rows where a worse class covers more area than a milder one")
if (abs(u.dsci - u[D].sum(axis=1)) > 1e-6).any(): issue("dsci != D0+..+D4")
say("dominant class shares:", u.dominant_class.value_counts(normalize=True).sort_index().round(3).to_dict())
yr = u.groupby(u.valid_start.dt.year).dsci.mean().round(0).to_dict()
say("mean DSCI by year:", yr)

# ------------------------------------------------------------------ NASS
say("== NASS pasture condition, Florida, weekly")
n = pd.read_parquet(L / "nass_fl_pasture_condition_weekly.parquet").sort_values("week_ending")
w = pd.DatetimeIndex(n.week_ending)
say(f"{len(n)} weeks, {w[0].date()} .. {w[-1].date()}; weekdays {sorted(set(w.dayofweek))}")
if n.week_ending.duplicated().any(): issue("duplicate NASS weeks")
expected = pd.date_range(w[0], w[-1], freq="7D")
gaps = expected.difference(w)
say(f"missing weeks {len(gaps)}:", pd.Series(gaps).groupby(gaps.year).size().to_dict())
runs = []
if len(gaps):
    g = pd.Series(gaps); brk = (g.diff().dt.days != 7).cumsum()
    runs = [(x.iloc[0].date(), x.iloc[-1].date(), len(x)) for _, x in g.groupby(brk)]
    say("  gap runs (first, last, n):", runs)
C = ["very_poor", "poor", "fair", "good", "excellent"]
tot = n[C].sum(axis=1)
say(f"category sum: min {tot.min()}, max {tot.max()}")
if ((tot < 98) | (tot > 102)).any(): issue(f"{int(((tot < 98) | (tot > 102)).sum())} NASS weeks whose categories do not sum to 100 +- 2")
ci = sum(n[c_] * (i + 1) for i, c_ in enumerate(C)) / 100
if (abs(ci - n.condition_index) > 1e-6).any(): issue("condition_index inconsistent")
say("condition index by year (mean):", n.groupby(w.year).condition_index.mean().round(2).to_dict())

# ------------------------------------------------------------------ hurricanes
say("== hurricane tracks and landfalls")
t = pd.read_parquet(L / "hurricane_tracks_florida.parquet")
lf = pd.read_parquet(L / "hurricane_landfalls.parquet")
say(f"tracks: {t.storm_id.nunique()} storms, {len(t)} points, sources {t.source.value_counts().to_dict()}, "
    f"{t.time.min().date()} .. {t.time.max().date()}")
if t.duplicated(["storm_id", "time"]).any(): issue("duplicate track points")
ex = t[t.source != "HURDAT2"]
if len(ex):
    say("storms not yet in HURDAT2 (working best tracks):", ex.groupby(["storm_id", "name"]).vmax_kt.max().to_dict())
fl = lf[lf.lon.between(-87.7, -79.9) & lf.lat.between(24.4, 31.1)]
say(f"landfalls: {len(lf)} in table, {len(fl)} inside the Florida box; storms with a landfall elsewhere only: "
    f"{lf[~lf.storm_id.isin(fl.storm_id)].name.nunique()}")
say(fl[["name", "time", "status", "lat", "lon", "vmax_kt"]].sort_values("time").to_string(index=False))
# Major Florida landfalls, NHC Tropical Cyclone Reports (date UTC, approximate place, intensity in kt)
known = [("HERMINE", "2016-09-02", 70), ("IRMA", "2017-09-10", 115), ("MICHAEL", "2018-10-10", 140),
         ("IAN", "2022-09-28", 130), ("NICOLE", "2022-11-10", 65), ("IDALIA", "2023-08-30", 90),
         ("DEBBY", "2024-08-05", 70), ("HELENE", "2024-09-27", 120), ("MILTON", "2024-10-10", 105)]
say("== known Florida landfalls (TCR) vs table")
for name, day, kt in known:
    m = fl[(fl.name.str.upper() == name) & (fl.time.dt.date.astype(str) == day)]
    if not len(m):
        issue(f"{name} {day}: no Florida landfall in the table")
        continue
    v = m.vmax_kt.max()
    say(f"  {name:8s} {day}: table {len(m)} landfall(s), max {v:.0f} kt (TCR about {kt} kt)")
    if abs(v - kt) > 10: issue(f"{name}: landfall intensity {v} kt far from the TCR value {kt} kt")
y26 = t[t.time.dt.year == 2026]
if len(y26):
    near = y26[y26.lon.between(-87.7, -79.9) & y26.lat.between(24.4, 31.1)]
    say("2026 storms with points over the Florida box:", near.groupby("name").vmax_kt.max().to_dict(),
        "| 2026 landfall rows:", int((lf.time.dt.year == 2026).sum()))
    if len(near) and not ((lf.time.dt.year == 2026) & lf.storm_id.isin(near[near.vmax_kt >= 64].storm_id)).any():
        issue("a 2026 hurricane crossed the Florida box but has no landfall record")

say("\n== ISSUES"); [say("-", i) for i in issues]
