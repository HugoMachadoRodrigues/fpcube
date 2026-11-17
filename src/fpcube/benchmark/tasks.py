"""Benchmark tasks, fixed splits and evaluation for the Florida Pasture Cube.

Task 1  Greenness forecasting: from the 60 days before an issue date (12 composites of NDVI, NDMI, VV, VH,
        rainfall, temperature, VPD) plus static soil/terrain attributes, predict observed (not gap-filled) NDVI
        15, 30, 45 and 60 days ahead. Splits: issue year 2016-2022 train, 2023-2024 validation, 2025-2026 test;
        a held-out county group (spatial test) is removed from training and validation.
Task 2  Event exposure detection: from the composites 30 days before to 30 days after an event date, decide
        whether a field was exposed to the event. Reference labels come only from sources the inputs do not use
        directly as labels: hurricane tracks (HURDAT2/NHC) with PRISM 3-day rainfall, PRISM extreme rainfall,
        PRISM minimum temperature (freeze) and the US Drought Monitor (drought onset). Positives and negatives
        share the event date. Split: leave-one-event-out.
Task 3  Weakly supervised condition: from the composites of all fields in a county over the 4 weeks before
        a US Drought Monitor map date, predict the county's drought class (none, D0, D1, D2 or worse).
        Labels exist only per county and week. Splits as Task 1 (by year) plus a county-group test.
"""
import numpy as np
import pandas as pd
import xarray as xr

from fpcube import config

HORIZONS = (3, 6, 9, 12)              # composites ahead: 15, 30, 45, 60 days
LOOKBACK = 12                          # 60 days
DYN = ["ndvi", "ndmi", "evi", "vv_db", "vh_db", "ppt_sum", "tmean_mean", "vpdmax_mean"]
STATIC = ["drainage_code", "flooding_code", "aws0_100_mm", "hand_mean_m", "twi_mean", "elev_mean_m",
          "wetland_frac", "area_ha"]
SEED = 20261116


def county_groups(fields, n_groups=5):
    """Deterministic assignment of counties to n groups, balanced by field count; group 0 = spatial test."""
    c = fields.groupby("county").size().sort_values(ascending=False)
    rng = np.random.default_rng(SEED)
    order = list(c.index)
    rng.shuffle(order)
    load = np.zeros(n_groups)
    grp = {}
    for name in sorted(order, key=lambda k: -c[k]):
        g = int(np.argmin(load))
        grp[name] = g
        load[g] += c[name]
    return grp


def year_split(year):
    return np.where(year <= 2022, "train", np.where(year <= 2024, "val", "test"))


# ---------------------------------------------------------------- Task 1
def task1_samples(cube, fields, stride=6, max_per_field=None):
    """Issue dates every `stride` composites; one sample per field x issue date with an observed target."""
    nd = cube["ndvi"].values
    filled = cube["gap_filled"].values
    nobs = cube["n_optical_obs"].values
    observed = (nobs > 0) & ~filled
    T = nd.shape[1]
    times = pd.DatetimeIndex(cube.time.values)
    issues = np.arange(LOOKBACK, T - max(HORIZONS), stride)
    rows = []
    for t in issues:
        ok_in = np.isfinite(nd[:, t - LOOKBACK:t]).mean(axis=1) >= 0.5
        for h in HORIZONS:
            ok = ok_in & observed[:, t + h - 1]
            idx = np.nonzero(ok)[0]
            rows.append(pd.DataFrame({"f": idx, "t": t, "h": h}))
    s = pd.concat(rows, ignore_index=True)
    s["issue_date"] = times[s.t]
    s["split"] = year_split(s.issue_date.dt.year.to_numpy())
    grp = county_groups(fields)
    s["county_group"] = fields.county.map(grp).to_numpy()[s.f]
    s.loc[s.county_group == 0, "split"] = s.loc[s.county_group == 0, "split"].map(
        {"train": "drop", "val": "drop", "test": "test_spatial"})
    return s[s.split != "drop"].reset_index(drop=True)


def task1_features(cube, fields, s):
    X = {}
    for v in DYN:
        a = cube[v].values
        for lag in range(1, LOOKBACK + 1, 2):
            X[f"{v}_l{lag}"] = a[s.f.to_numpy(), (s.t - lag).to_numpy()]
    nd = cube["ndvi"].values
    X["ndvi_mean_l12"] = np.nanmean(np.stack([nd[s.f, s.t - k] for k in range(1, LOOKBACK + 1)]), axis=0)
    X["doy_sin"] = np.sin(2 * np.pi * s.issue_date.dt.day_of_year / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * s.issue_date.dt.day_of_year / 365.25)
    X["horizon"] = s.h.to_numpy()
    for c in STATIC:
        if c in fields:
            X[c] = fields[c].to_numpy()[s.f]
    y = nd[s.f.to_numpy(), (s.t + s.h - 1).to_numpy()]
    return pd.DataFrame(X), y


def task1_climatology(cube, s):
    """Simple baseline: the field's mean observed NDVI for the same 5-day period of the year, train years only."""
    nd = cube["ndvi"].values
    times = pd.DatetimeIndex(cube.time.values)
    poy = np.asarray((times.day_of_year - 1) // config.COMPOSITE_DAYS)
    train = np.asarray(times.year <= 2022)
    nper = poy.max() + 1
    clim = np.full((nd.shape[0], nper), np.nan, np.float32)
    for p in range(nper):
        cols = np.nonzero(train & (poy == p))[0]
        if len(cols):
            clim[:, p] = np.nanmean(nd[:, cols], axis=1)
    target_t = (s.t + s.h - 1).to_numpy()
    return clim[s.f.to_numpy(), poy[target_t]]


# ---------------------------------------------------------------- Task 2
def _prism_daily(var):
    import json
    import rasterio
    out = {}
    for y in config.YEARS:
        p = config.RAW / "prism" / var / f"prism_{var}_{y}.tif"
        if p.exists():
            out[y] = p
    return out


def field_daily(var, fields):
    """Daily PRISM series at each field's 4 km cell: DataFrame (date x field index)."""
    import json
    import rasterio
    pts = fields.geometry.representative_point()
    frames = []
    for y, p in _prism_daily(var).items():
        with rasterio.open(p) as src:
            xy = pts.to_crs(src.crs)
            r, c = rasterio.transform.rowcol(src.transform, xy.x.values, xy.y.values)
            r = np.clip(r, 0, src.height - 1)
            c = np.clip(c, 0, src.width - 1)
            a = src.read()[:, r, c]
            frames.append(pd.DataFrame(a, index=pd.to_datetime(json.loads(src.tags()["dates"]))))
    return pd.concat(frames)


def task2_events(fields, tracks, landfalls, usdm):
    """Event table: event_id, kind, date, positive field indices, negative field indices."""
    from shapely.geometry import LineString, Point
    ppt = field_daily("ppt", fields)
    ppt3 = ppt.rolling(3).sum()
    tmin = field_daily("tmin", fields)
    lonlat = np.c_[fields.lon.to_numpy(), fields.lat.to_numpy()]
    ev = []
    # hurricanes: storms reaching >= 64 kt with track points within 150 km of any field
    for sid, g in tracks.groupby("storm_id"):
        if g.vmax_kt.max() < 64:
            continue
        g = g.sort_values("time")
        d = np.min(np.hypot((lonlat[:, None, 0] - g.lon.to_numpy()[None]) * 98.0,
                            (lonlat[:, None, 1] - g.lat.to_numpy()[None]) * 111.0), axis=1)
        if d.min() > 150:
            continue
        t0 = g.time.iloc[np.argmin(np.min(np.hypot((lonlat[:, None, 0] - g.lon.to_numpy()[None]) * 98.0,
                                                     (lonlat[:, None, 1] - g.lat.to_numpy()[None]) * 111.0), axis=0))]
        day = pd.Timestamp(t0).normalize()
        win = ppt3.loc[day: day + pd.Timedelta(days=3)].max()
        pos = np.nonzero((d <= 150) & (win.to_numpy() >= 100))[0]
        neg = np.nonzero((d >= 300) & (win.to_numpy() < 25))[0]
        if len(pos) >= 20 and len(neg) >= 20:
            ev.append({"event_id": f"hurricane_{sid}", "kind": "hurricane", "date": day,
                       "name": g.name.iloc[0], "pos": pos, "neg": neg})
    # extreme rainfall outside hurricane windows: days with >= 200 fields at >= 150 mm in 3 days
    hur_days = {e["date"] for e in ev}
    hits = (ppt3 >= 150).sum(axis=1)
    cand = hits[hits >= 200].index
    taken = []
    for day in cand:
        if any(abs((day - h).days) <= 10 for h in list(hur_days) + taken):
            continue
        v = ppt3.loc[day].to_numpy()
        pos, neg = np.nonzero(v >= 150)[0], np.nonzero(v < 25)[0]
        if len(neg) >= 20:
            ev.append({"event_id": f"rain_{day:%Y%m%d}", "kind": "extreme_rain", "date": day, "name": "",
                       "pos": pos, "neg": neg})
            taken.append(day)
    # freezes: days with >= 200 fields at tmin <= -2 C, at least 10 days apart
    hits = (tmin <= -2).sum(axis=1)
    taken = []
    for day in hits[hits >= 200].index:
        if any(abs((day - h).days) <= 10 for h in taken):
            continue
        v = tmin.loc[day].to_numpy()
        pos, neg = np.nonzero(v <= -2)[0], np.nonzero(v >= 4)[0]
        if len(neg) >= 20:
            ev.append({"event_id": f"freeze_{day:%Y%m%d}", "kind": "freeze", "date": day, "name": "",
                       "pos": pos, "neg": neg})
            taken.append(day)
    # drought onset: first week a county reaches D2+ after >= 8 weeks below D1
    fips = fields.county_fips.to_numpy() if "county_fips" in fields else None
    if fips is not None:
        u = usdm.sort_values(["county_fips", "valid_start"])
        u["d2"] = u.D2 >= 50
        for cf, g in u.groupby("county_fips"):
            prev = g.d2.shift(1, fill_value=False).rolling(8, min_periods=8).max().fillna(1).astype(bool)
            for day in g.loc[g.d2 & ~prev, "valid_start"]:
                same = u[(u.valid_start == day)]
                calm = set(same.loc[same.D0 < 10, "county_fips"])
                pos = np.nonzero(fips == cf)[0]
                neg = np.nonzero(np.isin(fips, list(calm)))[0]
                if len(pos) >= 20 and len(neg) >= 20:
                    ev.append({"event_id": f"drought_{cf}_{day:%Y%m%d}", "kind": "drought", "date": day,
                               "name": cf, "pos": pos, "neg": neg})
    return ev


def task2_samples(events, cube, neg_ratio=1.0, seed=SEED):
    rng = np.random.default_rng(seed)
    times = pd.DatetimeIndex(cube.time.values)
    rows = []
    for e in events:
        t = int(np.searchsorted(times, e["date"]))
        if t < 6 or t + 6 >= len(times):
            continue
        neg = e["neg"]
        if len(neg) > neg_ratio * len(e["pos"]):
            neg = rng.choice(neg, int(neg_ratio * len(e["pos"])), replace=False)
        for lab, idx in ((1, e["pos"]), (0, neg)):
            rows.append(pd.DataFrame({"f": idx, "t": t, "y": lab, "event_id": e["event_id"], "kind": e["kind"]}))
    return pd.concat(rows, ignore_index=True)


def task2_features(cube, s, pre=6, post=6):
    X = {}
    for v in ("ndvi", "ndmi", "mndwi", "evi", "vv_db", "vh_db"):
        a = cube[v].values
        f, t = s.f.to_numpy(), s.t.to_numpy()
        before = np.nanmean(np.stack([a[f, t - k] for k in range(1, pre + 1)]), axis=0)
        after = np.nanmean(np.stack([a[f, t + k] for k in range(1, post + 1)]), axis=0)
        mn = np.nanmin(np.stack([a[f, t + k] for k in range(0, post + 1)]), axis=0)
        X[f"{v}_pre"], X[f"{v}_post"], X[f"{v}_delta"] = before, after, after - before
        X[f"{v}_min_post_delta"] = mn - before
    X["doy_sin"] = np.sin(2 * np.pi * pd.DatetimeIndex(cube.time.values[s.t]).day_of_year / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * pd.DatetimeIndex(cube.time.values[s.t]).day_of_year / 365.25)
    return pd.DataFrame(X)


# ---------------------------------------------------------------- Task 3
def drought_class(row):
    """USDM percent-area (cumulative) to 4 classes by the >= 50% area rule: 0 none, 1 D0, 2 D1, 3 D2+."""
    if row.D2 >= 50:
        return 3
    if row.D1 >= 50:
        return 2
    if row.D0 >= 50:
        return 1
    return 0


def task3_samples(usdm, fields, cube):
    times = pd.DatetimeIndex(cube.time.values)
    u = usdm.copy()
    u["y"] = u.apply(drought_class, axis=1)
    u["t"] = np.searchsorted(times, u.valid_start)
    u = u[(u.t >= 6) & (u.t < len(times))]
    have = set(fields.county_fips) if "county_fips" in fields else set()
    u = u[u.county_fips.isin(have)]
    u["split"] = year_split(u.valid_start.dt.year.to_numpy())
    grp = county_groups(fields)
    name_of = fields.drop_duplicates("county_fips").set_index("county_fips").county
    u["county_group"] = u.county_fips.map(name_of).map(grp)
    u.loc[u.county_group == 0, "split"] = u.loc[u.county_group == 0, "split"].map(
        {"train": "drop", "val": "drop", "test": "test_spatial"})
    return u[u.split != "drop"].reset_index(drop=True)


def ndvi_anomaly(cube):
    """NDVI minus the field's 2016-2022 mean for the same period of year (z-score per field and period)."""
    nd = cube["ndvi"].values
    times = pd.DatetimeIndex(cube.time.values)
    poy = np.asarray((times.day_of_year - 1) // config.COMPOSITE_DAYS)
    train = np.asarray(times.year <= 2022)
    an = np.full_like(nd, np.nan)
    for p in np.unique(poy):
        cols = poy == p
        ref = nd[:, cols & train]
        mu, sd = np.nanmean(ref, axis=1), np.nanstd(ref, axis=1)
        an[:, cols] = (nd[:, cols] - mu[:, None]) / np.where(sd > 0.01, sd, np.nan)[:, None]
    return an


def task3_bags(u, fields, cube, window=6):
    """For each county-week: matrix of field features (fields x features) over the 30 days before the map."""
    an = ndvi_anomaly(cube)
    ndmi = cube["ndmi"].values
    vv = cube["vv_db"].values
    ppt = cube["ppt_sum"].values if "ppt_sum" in cube else None
    fips = fields.county_fips.to_numpy()
    bags = []
    for r in u.itertuples():
        idx = np.nonzero(fips == r.county_fips)[0]
        sl = slice(r.t - window, r.t)
        feats = [np.nanmean(an[idx, sl], axis=1), np.nanmean(ndmi[idx, sl], axis=1), np.nanmean(vv[idx, sl], axis=1)]
        if ppt is not None:
            feats.append(np.nansum(ppt[idx, r.t - 12:r.t], axis=1))
        bags.append(np.c_[tuple(feats)].astype(np.float32))
    return bags
