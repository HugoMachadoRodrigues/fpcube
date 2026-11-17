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


def test_task3_and_mil():
    ds, f = synthetic()
    times = pd.DatetimeIndex(ds.time.values)
    rows = []
    for c in f.county_fips.unique():
        for d in times[10::40]:
            rows.append({"county_fips": c, "valid_start": d, "D0": 60.0, "D1": 0.0, "D2": 0.0, "D3": 0.0, "D4": 0.0})
    u = T.task3_samples(pd.DataFrame(rows), f, ds)
    assert (u.y == 1).all()
    bags = T.task3_bags(u, f, ds)
    assert all(b.shape[1] == 4 for b in bags)   # NDVI anomaly, NDMI, VV, 60-day rain
    y = np.random.default_rng(0).integers(0, 4, len(bags))
    mil = M.train_mil(bags[:30], y[:30], bags[30:40], y[30:40], epochs=2)
    p = M.predict_mil(mil, bags[40:])
    assert ((p >= 0) & (p <= 3)).all()


def test_seq_model_runs():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 12, 5)).astype(np.float32)
    X[rng.random(X.shape) < 0.3] = np.nan
    S = rng.standard_normal((400, 3)).astype(np.float32)
    y = np.nan_to_num(X[:, -1, 0]).astype(np.float32)
    net = M.train_seq(X[:300], S[:300], y[:300], X[300:], S[300:], y[300:], epochs=3)
    p = M.predict_seq(net, X[300:], S[300:])
    assert p.shape == (100, 1) and np.isfinite(p).all()


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert M.binary_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["auroc"] == 1.0
    assert M.ordinal_metrics(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))["qwk"] == 1.0
    r = M.regression_metrics(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]), ref=np.array([2.0, 2, 2]))
    assert r["rmse"] == 0 and r["skill_vs_ref"] == 1
