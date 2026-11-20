"""Smoke tests of the benchmark code on a small synthetic cube (no downloads)."""
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from fpcube import cube as C
from fpcube.benchmark import models as M
from fpcube.benchmark import run as R
from fpcube.benchmark import tasks as T


def synthetic(n_fields=60, seed=0):
    rng = np.random.default_rng(seed)
    times = C.calendar()
    doy = np.asarray(times.day_of_year)
    seas = 0.55 + 0.2 * np.sin(2 * np.pi * (doy - 120) / 365.25)
    nd = seas[None] + 0.05 * rng.standard_normal((n_fields, len(times)))
    nd[rng.random(nd.shape) < 0.4] = np.nan
    ds = xr.Dataset(coords={"field": [f"FPC{i:06d}" for i in range(n_fields)], "time": times})
    for v in T.DYN + ["mndwi", "evi"]:
        ds[v] = (("field", "time"), (nd + 0.01 * rng.standard_normal(nd.shape)).astype(np.float32))
    ds["ndvi"] = (("field", "time"), nd.astype(np.float32))
    ds["n_optical_obs"] = (("field", "time"), np.isfinite(nd).astype(np.uint8))
    ds["gap_filled"] = (("field", "time"), np.zeros(nd.shape, bool))
    counties = ["Polk", "Hardee", "DeSoto", "Highlands", "Okeechobee", "Osceola"]
    fips = ["12105", "12049", "12027", "12055", "12093", "12097"]
    f = gpd.GeoDataFrame({"fid": np.arange(1, n_fields + 1), "field_id": ds.field.values,
                          "county": [counties[i % 6] for i in range(n_fields)],
                          "county_fips": [fips[i % 6] for i in range(n_fields)],
                          "area_ha": rng.uniform(2, 50, n_fields), "lon": -81.9, "lat": 27.4},
                         geometry=[Point(0, 0)] * n_fields, crs="EPSG:5070")
    return ds, f


def test_calendar_and_period_index():
    t = C.calendar()
    assert t[0] == pd.Timestamp("2016-01-01")
    assert (np.diff(t.values).astype("timedelta64[D]").astype(int) == 5).all()
    k = C.period_index(pd.Series(pd.to_datetime(["2016-01-01", "2016-01-05", "2016-01-06"])))
    assert list(k) == [0, 0, 1]


def test_run_lengths():
    m = np.array([[True, True, False, True, True, True, False]])
    assert C._run_lengths(m).tolist() == [[2, 2, 0, 3, 3, 3, 0]]


def test_task1_splits_and_features():
    ds, f = synthetic()
    s = T.task1_samples(ds, f)
    assert set(s.split) <= {"train", "val", "test", "test_spatial"}
    grp = T.county_groups(f)
    spatial = {c for c, g in grp.items() if g == 0}
    assert not s[s.split.isin(["train", "val"])].f.map(f.county).isin(spatial).any()
    X, y = T.task1_features(ds, f, s.head(500))
    assert len(X) == len(y) == 500 and np.isfinite(y).all()
    clim = T.task1_climatology(ds, s.head(500))
    assert np.nanmean(np.abs(clim - y)) < 0.2


def weekly_usdm(f, start="2018-01-02", n=120, seed=0):
    """Weekly maps (Tuesdays) with a class that drifts up and down, for every county of the synthetic fields."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in f.county_fips.unique():
        cls = np.clip(np.cumsum(rng.choice([-1, 0, 0, 0, 1], n)), 0, 3)
        for d, k in zip(pd.date_range(start, periods=n, freq="7D"), cls):
            rows.append({"county_fips": c, "valid_start": d, "D0": 60.0 * (k >= 1), "D1": 60.0 * (k >= 2),
                         "D2": 60.0 * (k >= 3), "D3": 0.0, "D4": 0.0})
    return pd.DataFrame(rows)


def test_task3_change_target_and_mil():
    ds, f = synthetic()
    usdm = weekly_usdm(f)
    u = T.task3_samples(usdm, f, ds, horizon=4)
    assert len(u) > 0 and set(u.direction) <= {0, 1, 2}
    # the target is the class 4 weeks later; momentum is the change over the previous 4 weeks
    full = usdm.assign(y=T.drought_classes(usdm)).set_index(["county_fips", "valid_start"]).y
    for r in u.sample(20, random_state=0).itertuples():
        fut = full[(r.county_fips, r.valid_start + pd.Timedelta(days=28))]
        past = full[(r.county_fips, r.valid_start - pd.Timedelta(days=28))]
        assert r.direction == np.sign(fut - r.y) + 1 and r.momentum == r.y - past
    # no target crosses a split boundary
    assert (T.year_split(u.target_date.dt.year.to_numpy())[u.split.isin(["train", "val", "test"])]
            == u.split[u.split.isin(["train", "val", "test"])]).all()
    bags = T.task3_bags(u, f, ds)
    assert all(b.shape[1] == 4 for b in bags)   # NDVI anomaly, NDMI, VV, 30-day rain
    cov = T.county_covariates(u)
    assert cov.shape == (len(u), 7)
    prob = T.transition_climatology(u)(u)
    assert np.allclose(prob.sum(1), 1)
    mb = [np.c_[b, np.repeat(c[None], len(b), 0)] for b, c in zip(bags, cov)]
    y = u.direction.to_numpy()
    mil = M.train_mil(mb[:30], y[:30], mb[30:40], y[30:40], n_class=3, epochs=2)
    pr = M.predict_mil_proba(mil, mb[40:60])
    assert pr.shape == (len(mb[40:60]), 3) and np.allclose(pr.sum(1), 1, atol=1e-5)
    r = M.direction_metrics(y[40:60], pr.argmax(1), pr)
    assert 0 <= r["macro_f1"] <= 1 and "auroc_worse" in r


def test_seq_model_runs():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 12, 5)).astype(np.float32)
    X[rng.random(X.shape) < 0.3] = np.nan
    S = rng.standard_normal((400, 3)).astype(np.float32)
    y = np.nan_to_num(X[:, -1, 0]).astype(np.float32)
    net = M.train_seq(X[:300], S[:300], y[:300], X[300:], S[300:], y[300:], epochs=3)
    p = M.predict_seq(net, X[300:], S[300:])
    assert p.shape == (100, 1) and np.isfinite(p).all()
    # raw inputs are standardized with the training statistics: shifting the inputs' units must not matter
    X2, S2 = X * 10 + 3, S * 5 - 1
    net2 = M.train_seq(X2[:300], S2[:300], y[:300], X2[300:], S2[300:], y[300:], epochs=3)
    p2 = M.predict_seq(net2, X2[300:], S2[300:])
    assert np.allclose(p, p2, atol=1e-3)


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert M.binary_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["auroc"] == 1.0
    assert M.ordinal_metrics(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))["qwk"] == 1.0
    r = M.regression_metrics(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]), ref=np.array([2.0, 2, 2]))
    assert r["rmse"] == 0 and r["skill_vs_ref"] == 1


def test_task1_no_target_crosses_split():
    ds, f = synthetic()
    s = T.task1_samples(T.observed_only(ds), f)
    assert (T.year_split(s.target_end.dt.year.to_numpy())[s.split != "test_spatial"]
            == s.split[s.split != "test_spatial"]).all()
    assert s.attrs["n_dropped_crossing_split"] > 0


def _gap_fill(ds):
    """Fill optical gaps by linear interpolation in time, as the cube does (n_optical_obs keeps the observed cells)."""
    out = ds.copy(deep=True)
    for v in T.OPTICAL:
        if v in out:
            out[v] = out[v].interpolate_na("time", max_gap=None, limit=6, use_coordinate=False)
    return out


def test_inputs_are_causal():
    """Changing observations after the issue date must not change any Task 1 input of that issue date, although
    the gap filling of the cube spreads them backwards."""
    ds, f = synthetic()
    base = T.observed_only(_gap_fill(ds))
    s = T.task1_samples(base, f)
    s = s[s.t == s.t.iloc[len(s) // 2]].head(50)
    t0 = int(s.t.iloc[0])
    alt = ds.copy(deep=True)
    for v in ["ndvi", "ndmi", "evi"]:
        alt[v][:, t0:] = alt[v][:, t0:] + 0.3
    leaky = _gap_fill(alt)
    assert not np.allclose(np.nan_to_num(leaky.ndvi.values[:, t0 - 3:t0]),
                           np.nan_to_num(_gap_fill(ds).ndvi.values[:, t0 - 3:t0]))     # the cube itself leaks
    X0, _ = T.task1_features(base, f, s)
    X1, _ = T.task1_features(T.observed_only(leaky), f, s)
    pd.testing.assert_frame_equal(X0, X1)


def test_observed_only_removes_filled_values():
    ds, f = synthetic()
    ds["n_optical_obs"][:, 5] = 0
    assert np.isnan(T.observed_only(ds.copy(deep=True)).ndvi.values[:, 5]).all()


def test_anomaly_cut_rule_gives_lowest_to_worse():
    y = np.r_[np.zeros(20, int), np.ones(60, int), np.full(20, 2)]       # better, same, worse
    rule = T.anomaly_cut_rule(np.arange(100.0), y)
    p = rule(np.arange(100.0))
    assert np.bincount(p, minlength=3).tolist() == [20, 60, 20]
    assert (p[:20] == 2).all() and (p[-20:] == 0).all()


def test_task3_window_ends_before_map():
    ds, f = synthetic()
    times = pd.DatetimeIndex(ds.time.values)
    u = T.task3_samples(weekly_usdm(f, n=60), f, ds)
    last_end = T.period_ends(times)[u.t - 1]
    assert (last_end <= u.valid_start).all()
    assert (T.period_ends(times)[np.minimum(u.t, len(times) - 1)] > u.valid_start).all()   # the latest such window


def test_cluster_bootstrap_interval():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(4000)
    p = y + 0.1 * rng.standard_normal(4000)
    ci = M.regression_ci(y, p, np.zeros(4000), rng.integers(0, 40, 4000), rng.integers(0, 30, 4000), B=200)
    assert ci["rmse_lo"] < 0.1 < ci["rmse_hi"] + 0.01 and ci["skill_lo"] > 0.95
