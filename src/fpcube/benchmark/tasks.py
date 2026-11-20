"""Benchmark tasks, fixed splits and evaluation for the Florida Pasture Cube.

Causal inputs: the cube fills optical gaps of up to 30 days by interpolating between the observations on either side,
so a filled value before an issue date can depend on an observation after it. Benchmark inputs therefore never use
gap-filled values: every optical variable is set to missing where the period has no observation
(observed_only), and models receive the missing-value pattern (LightGBM natively, the deep models as a mask) plus
the last observed NDVI and its age. Radar and PRISM values are never interpolated.

Task 1  Greenness forecasting: from the 12 periods (60 days) before an issue date (observed optical composites of
        NDVI, NDMI, EVI, radar VV and VH, PRISM rainfall, temperature and VPD) plus static soil/terrain attributes,
        predict the observed NDVI composite 15, 30, 45 and 60 days ahead (the period starting 5h - 5 days after the
        issue date, h = 3, 6, 9, 12). A sample needs at least MIN_OBS_IN observed NDVI composites in the 60 days.
        Splits by year: 2016-2022 train, 2023-2024 validation, 2025-2026 test, applied to BOTH the issue date and
        the end of the target period, so no target crosses a split boundary. A held-out county group (spatial test)
        is removed from training and validation. Baselines: persistence (last observed NDVI before the issue date,
        within the 60 days) and climatology (mean of the field's observed NDVI composites in the same 5-day period
        of the year, 2016-2022). All models are scored on the same samples (those where climatology exists).
        Skill = 1 - MSE(model) / MSE(climatology) (mean squared error skill score).
Task 2  Event exposure detection: from the composites 30 days before to 30 days after an event date, decide
        whether a field was exposed to the event. Reference labels come from sources the inputs do not contain:
        hurricane tracks (HURDAT2/NHC) with PRISM 3-day rainfall, PRISM extreme rainfall, PRISM minimum temperature
        (freeze) and the US Drought Monitor (drought onset); PRISM is not an input. Positives and negatives share
        the event date. Two experiments, each with every model under the same protocol and validation events
        held out as whole events: leave-one-event-out (rule, LightGBM) and leave-one-kind-out (rule, LightGBM,
        transformer). Scores: AUROC per held-out event (median per kind) and pooled per kind.
Task 3  Weakly supervised condition: from the observed composites of all fields in a county over the 6 periods
        (30 days) that END on or before a US Drought Monitor map date, plus 30-day rainfall, predict the county's
        drought class (none, D0, D1, D2 or worse). Labels exist only per county and week. Splits by year of the map
        plus a county-group test. Baselines: majority class, last week's USDM class, NDVI-anomaly cut points with
        class frequencies from training, LightGBM on bag statistics, attention MIL.
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
MIN_OBS_IN = 2                         # observed NDVI composites needed in the 60 days before a Task 1 issue date
OPTICAL = ("blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "evi", "ndmi", "mndwi", "nbr", "ndvi_sd",
           "water_frac")


def observed_only(cube):
    """Optical variables set to missing where the period has no observation (gap-filled values removed)."""
    obs = cube["n_optical_obs"] > 0
    for v in OPTICAL:
        if v in cube:
            cube[v] = cube[v].where(obs)
    return cube


def period_ends(times):
    """Exclusive end of each 5-day period."""
    return pd.DatetimeIndex(times) + pd.Timedelta(days=config.COMPOSITE_DAYS)


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
def task1_samples(cube, fields, stride=6):
    """Issue dates every `stride` periods; one sample per field x issue date x horizon with an observed target and
    at least MIN_OBS_IN observed NDVI composites in the lookback. Expects observed_only(cube)."""
    nd = cube["ndvi"].values
    obs = np.isfinite(nd)
    T = nd.shape[1]
    times = pd.DatetimeIndex(cube.time.values)
    issues = np.arange(LOOKBACK, T - max(HORIZONS), stride)
    rows = []
    for t in issues:
        ok_in = obs[:, t - LOOKBACK:t].sum(axis=1) >= MIN_OBS_IN
        for h in HORIZONS:
            idx = np.nonzero(ok_in & obs[:, t + h - 1])[0]
            rows.append(pd.DataFrame({"f": idx, "t": t, "h": h}))
    s = pd.concat(rows, ignore_index=True)
    s["issue_date"] = times[s.t]
    s["target_end"] = period_ends(times)[s.t + s.h - 1] - pd.Timedelta(days=1)       # last day of the target period
    s["split"] = year_split(s.issue_date.dt.year.to_numpy())
    crosses = s.split.to_numpy() != year_split(s.target_end.dt.year.to_numpy())
    s = s[~crosses].copy()
    grp = county_groups(fields)
    s["county_group"] = fields.county.map(grp).to_numpy()[s.f]
    s.loc[s.county_group == 0, "split"] = s.loc[s.county_group == 0, "split"].map(
        {"train": "drop", "val": "drop", "test": "test_spatial"})
    s = s[s.split != "drop"].reset_index(drop=True)
    s.attrs["n_dropped_crossing_split"] = int(crosses.sum())
    return s


def last_observed(nd, f, t, lookback=LOOKBACK):
    """Last observed value of nd before period t (within lookback periods) and its age in days (issue date minus
    the start of that period)."""
    val = np.full(len(f), np.nan, np.float32)
    age = np.full(len(f), np.nan, np.float32)
    for k in range(1, lookback + 1):
        x = nd[f, t - k]
        m = np.isnan(val) & np.isfinite(x)
        val[m], age[m] = x[m], k * config.COMPOSITE_DAYS
    return val, age


def task1_features(cube, fields, s):
    X = {}
    f, t = s.f.to_numpy(), s.t.to_numpy()
    for v in DYN:
        a = cube[v].values
        for lag in range(1, LOOKBACK + 1, 2):
            X[f"{v}_l{lag}"] = a[f, t - lag]
    nd = cube["ndvi"].values
    lb = np.stack([nd[f, t - k] for k in range(1, LOOKBACK + 1)])
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        X["ndvi_mean_l12"] = np.nanmean(lb, axis=0)
    X["n_ndvi_obs_l12"] = np.isfinite(lb).sum(axis=0)
    X["ndvi_last"], X["ndvi_last_age_days"] = last_observed(nd, f, t)
    X["doy_sin"] = np.sin(2 * np.pi * s.issue_date.dt.day_of_year / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * s.issue_date.dt.day_of_year / 365.25)
    X["horizon"] = s.h.to_numpy()
    for c in STATIC:
        if c in fields:
            X[c] = fields[c].to_numpy()[f]
    y = nd[f, t + s.h.to_numpy() - 1]
    return pd.DataFrame(X), y


def task1_climatology(cube, s):
    """The field's mean observed NDVI composite for the same 5-day period of the year, 2016-2022 (expects
    observed_only(cube), so gap-filled values do not enter)."""
    nd = cube["ndvi"].values
    times = pd.DatetimeIndex(cube.time.values)
    poy = np.asarray((times.day_of_year - 1) // config.COMPOSITE_DAYS)
    train = np.asarray(times.year <= 2022)
    nper = poy.max() + 1
    clim = np.full((nd.shape[0], nper), np.nan, np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
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
        g = g[g.vmax_kt >= 64].sort_values("time")          # hurricane-strength part of the track only
        if not len(g):
            continue
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
    # drought onset: first week a county reaches D2+ (>= 50 % of area) after >= 8 weeks without; onsets of the same
    # week form one event (positives: fields in those counties; negatives: fields in counties with < 10 % in D0+)
    fips = fields.county_fips.to_numpy() if "county_fips" in fields else None
    if fips is not None:
        u = usdm.sort_values(["county_fips", "valid_start"]).copy()
        u["d2"] = u.D2 >= 50
        onsets = []
        for cf, g in u.groupby("county_fips"):
            prev = g.d2.shift(1, fill_value=False).rolling(8, min_periods=8).max().fillna(1).astype(bool)
            onsets += [(cf, d) for d in g.loc[g.d2 & ~prev, "valid_start"]]
        on = pd.DataFrame(onsets, columns=["county_fips", "day"])
        for day, g in on.groupby("day"):
            same = u[u.valid_start == day]
            calm = set(same.loc[same.D0 < 10, "county_fips"])
            pos = np.nonzero(np.isin(fips, g.county_fips.to_numpy()))[0]
            neg = np.nonzero(np.isin(fips, list(calm)))[0]
            if len(pos) >= 20 and len(neg) >= 20:
                ev.append({"event_id": f"drought_{day:%Y%m%d}", "kind": "drought", "date": day,
                           "name": ",".join(sorted(g.county_fips)), "pos": pos, "neg": neg})
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
    """Change features around the event date from observed composites only (expects observed_only(cube))."""
    import warnings
    X = {}
    f, t = s.f.to_numpy(), s.t.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for v in ("ndvi", "ndmi", "mndwi", "evi", "vv_db", "vh_db"):
            a = cube[v].values
            B = np.stack([a[f, t - k] for k in range(1, pre + 1)])
            A = np.stack([a[f, t + k] for k in range(1, post + 1)])
            before, after = np.nanmean(B, axis=0), np.nanmean(A, axis=0)
            mn = np.nanmin(np.stack([a[f, t + k] for k in range(0, post + 1)]), axis=0)
            X[f"{v}_pre"], X[f"{v}_post"], X[f"{v}_delta"] = before, after, after - before
            X[f"{v}_min_post_delta"] = mn - before
            if v == "ndvi":
                X["n_opt_pre"], X["n_opt_post"] = np.isfinite(B).sum(0), np.isfinite(A).sum(0)
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


def drought_classes(u):
    """Vectorised drought_class for a USDM table."""
    return np.select([u.D2 >= 50, u.D1 >= 50, u.D0 >= 50], [3, 2, 1], 0)


def task3_samples(usdm, fields, cube, window=6, horizon=4):
    """Task 3, drought class change: county-weeks issued at a USDM map date t0 (valid_start). Known at issue: the
    county class y0, its change over the previous `horizon` weeks ('momentum', y0 minus the class `horizon` weeks
    earlier) and the field evidence of the `window` periods [t - window, t) whose ends are all on or before t0.
    Target: the direction of the class change from t0 to the map `horizon` weeks later, 0 = better (lower class),
    1 = same, 2 = worse. Both maps around t0 must be exactly 7*horizon days away. The year split applies to the
    issue and the target date; county-weeks whose target falls in a later split are dropped (n in attrs)."""
    times = pd.DatetimeIndex(cube.time.values)
    u = usdm.sort_values(["county_fips", "valid_start"]).copy()
    u["y"] = drought_classes(u)
    g = u.groupby("county_fips")
    days = 7 * horizon
    fut, past = g.y.shift(-horizon), g.y.shift(horizon)
    fut_ok = (g.valid_start.shift(-horizon) - u.valid_start).dt.days == days
    past_ok = (u.valid_start - g.valid_start.shift(horizon)).dt.days == days
    u = u[fut_ok & past_ok].copy()
    u["y_future"] = fut[u.index].astype(int)
    u["momentum"] = (u.y - past[u.index]).astype(int)
    u["direction"] = np.sign(u.y_future - u.y).astype(int) + 1
    u["target_date"] = u.valid_start + pd.Timedelta(days=days)
    u["t"] = np.searchsorted(period_ends(times), u.valid_start, side="right")
    u = u[(u.t >= window) & (u.t <= len(times))]
    have = set(fields.county_fips) if "county_fips" in fields else set()
    u = u[u.county_fips.isin(have)]
    s_issue = year_split(u.valid_start.dt.year.to_numpy())
    s_target = year_split(u.target_date.dt.year.to_numpy())
    crossing = s_issue != s_target
    u["split"] = s_issue
    u = u[~crossing].copy()
    grp = county_groups(fields)
    name_of = fields.drop_duplicates("county_fips").set_index("county_fips").county
    u["county_group"] = u.county_fips.map(name_of).map(grp)
    u.loc[u.county_group == 0, "split"] = u.loc[u.county_group == 0, "split"].map(
        {"train": "drop", "val": "drop", "test": "test_spatial"})
    u = u[u.split != "drop"].reset_index(drop=True)
    u.attrs["n_dropped_crossing_split"] = int(crossing.sum())
    return u


def county_covariates(u):
    """County-level inputs known at issue: current class (one-hot), momentum, season (sin and cos of day of year)."""
    doy = u.valid_start.dt.day_of_year.to_numpy()
    return np.c_[np.eye(4)[u.y.to_numpy()], u.momentum.to_numpy(), np.sin(2 * np.pi * doy / 365.25),
                 np.cos(2 * np.pi * doy / 365.25)].astype(np.float32)


def transition_climatology(u_train, n_class=3):
    """Probability of each change direction given the current class and calendar month, from the training years
    (falls back to the current class alone, then to all training samples)."""
    key = lambda d: d.y.astype(str).to_numpy() + "_" + d.valid_start.dt.month.astype(str).to_numpy()
    dirs = u_train.direction.to_numpy()
    tab = pd.crosstab(key(u_train), dirs).reindex(columns=range(n_class), fill_value=0)
    by_y = pd.crosstab(u_train.y.to_numpy(), dirs).reindex(columns=range(n_class), fill_value=0)
    allp = np.bincount(dirs, minlength=n_class) / len(u_train)

    def prob(d):
        out = np.tile(allp, (len(d), 1))
        for i, (k, y) in enumerate(zip(key(d), d.y.to_numpy())):
            row = tab.loc[k] if k in tab.index else (by_y.loc[y] if y in by_y.index else None)
            if row is not None:
                v = row.to_numpy().astype(float) + 0.5
                out[i] = v / v.sum()
        return out
    return prob


def ndvi_anomaly(cube):
    """NDVI minus the field's 2016-2022 mean for the same period of year, divided by its standard deviation
    (z-score per field and period; with observed_only(cube) only observed composites enter)."""
    import warnings
    nd = cube["ndvi"].values
    times = pd.DatetimeIndex(cube.time.values)
    poy = np.asarray((times.day_of_year - 1) // config.COMPOSITE_DAYS)
    train = np.asarray(times.year <= 2022)
    an = np.full_like(nd, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for p in np.unique(poy):
            cols = poy == p
            ref = nd[:, cols & train]
            mu, sd = np.nanmean(ref, axis=1), np.nanstd(ref, axis=1)
            an[:, cols] = (nd[:, cols] - mu[:, None]) / np.where(sd > 0.01, sd, np.nan)[:, None]
    return an


def task3_bags(u, fields, cube, window=6):
    """For each county-week: matrix of field features (fields x features) over the `window` periods before the map
    (mean NDVI anomaly, mean NDMI, mean VV, rainfall sum over the same periods)."""
    import warnings
    an = ndvi_anomaly(cube)
    ndmi = cube["ndmi"].values
    vv = cube["vv_db"].values
    ppt = cube["ppt_sum"].values if "ppt_sum" in cube else None
    fips = fields.county_fips.to_numpy()
    bags = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r in u.itertuples():
            idx = np.nonzero(fips == r.county_fips)[0]
            sl = slice(r.t - window, r.t)
            feats = [np.nanmean(an[idx, sl], axis=1), np.nanmean(ndmi[idx, sl], axis=1),
                     np.nanmean(vv[idx, sl], axis=1)]
            if ppt is not None:
                feats.append(np.nansum(ppt[idx, sl], axis=1))
            bags.append(np.c_[tuple(feats)].astype(np.float32))
    return bags


def anomaly_cut_rule(m_train, y_train, n_class=3):
    """Ordinal cut points on the county mean NDVI anomaly: the lowest anomalies go to the highest class (for Task 3,
    'worse'), each class receiving its training frequency."""
    freq = np.bincount(y_train, minlength=n_class) / len(y_train)
    q = np.cumsum(freq[::-1])[:-1]                    # p(D2+), p(D2+)+p(D1), p(D2+)+p(D1)+p(D0)
    cuts = np.nanquantile(m_train, q)
    return lambda x: (n_class - 1) - np.searchsorted(cuts, x, side="right")
