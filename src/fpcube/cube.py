"""Assemble the Florida Pasture Cube from the per-source interim tables.

Calendar: 5-day periods starting config.COMPOSITE_ANCHOR (period k covers [anchor + 5k, anchor + 5k + 5) days).

Optical composite (per field and period): observations with at least max(3 px, 30% of the field's pixels)
clear and a geometric shift within 1 pixel are combined (median across observations of a period; when a field
falls in two overlapping MGRS tiles on the same date the records are first merged weighted by clear pixels).
Gaps up to 30 days are filled by linear interpolation in time (flag 'gap_filled'); longer gaps stay empty.

Radar composite: mean of linear gamma0 over the period's acquisitions (all orbits), converted to dB;
per-acquisition values stay in the long table with orbit metadata.

Weather: PRISM daily values at the field's 4 km cell, summed (ppt) or averaged (temperature, VPD) per period.

Release products (config.RELEASE / version):
  fields.parquet                 GeoParquet, one row per field with static attributes and quality flags
  cube.zarr                      dims (field, time): optical, radar, weather variables + counts and flags
  optical_observations/*.parquet long table, one row per field x acquisition
  radar_observations/*.parquet   long table, one row per field x Sentinel-1 acquisition
  labels/*.parquet               USDM (county-week), NASS (state-week), hurricane tracks and landfalls
"""
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from fpcube import config

OPT_VARS = ["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "evi", "ndmi", "mndwi", "nbr", "ndvi_sd",
            "water_frac"]
MAX_GAP_DAYS = 30
MAX_SHIFT_PX = 1.0
CHUNK = 8000                          # fields per composite chunk


def calendar():
    a = pd.Timestamp(config.COMPOSITE_ANCHOR)
    n = int((pd.Timestamp(config.END) - a).days // config.COMPOSITE_DAYS) + 1
    return a + pd.to_timedelta(np.arange(n) * config.COMPOSITE_DAYS, unit="D")


def period_index(dates):
    a = pd.Timestamp(config.COMPOSITE_ANCHOR)
    return ((pd.to_datetime(dates) - a).dt.days // config.COMPOSITE_DAYS).astype(int)


def _files(kind):
    if kind == "optical":
        fs = [f for f in (config.INTERIM / "optical").glob("T*/*.parquet")]
    else:
        fs = list((config.INTERIM / "radar" / "s1").glob("2*/*.parquet"))
    return sorted(fs)


def _load(kind, lo=None, hi=None, start=None, end=None):
    """Observations from the interim tables, optionally only fids in [lo, hi) and dates in [start, end)."""
    import pyarrow.dataset as ds
    files = _files(kind)
    if not files:
        return pd.DataFrame()
    dset = ds.dataset([str(f) for f in files], format="parquet")
    flt = None
    for cond in ((ds.field("fid") >= lo) if lo is not None else None,
                 (ds.field("fid") < hi) if hi is not None else None,
                 (ds.field("date") >= pd.Timestamp(start)) if start is not None else None,
                 (ds.field("date") < pd.Timestamp(end)) if end is not None else None):
        if cond is not None:
            flt = cond if flt is None else flt & cond
    d = dset.to_table(filter=flt).to_pandas()
    if len(d):
        d["date"] = pd.to_datetime(d.date)
    return d


def load_optical(lo=None, hi=None, start=None, end=None):
    return _load("optical", lo, hi, start, end)


def screen_optical(d):
    ok_clear = (d.n_clear_px >= np.maximum(3, 0.3 * d.n_field_px))
    sh = np.hypot(d.shift_row.fillna(0), d.shift_col.fillna(0))
    ok_geo = sh <= MAX_SHIFT_PX
    d = d.assign(qa_clear=ok_clear, qa_geometry=ok_geo, shift_px=sh)
    return d


def merge_tiles(d):
    """Same field, same day, several MGRS tiles: clear-pixel weighted mean."""
    d = d.assign(day=d.date.dt.floor("D"))
    w = d.n_clear_px.astype(float)
    num = d[OPT_VARS].mul(w, axis=0)
    g = pd.concat([d[["fid", "day", "sensor"]], num, w.rename("w")], axis=1).groupby(["fid", "day", "sensor"])
    s = g.sum()
    out = s[OPT_VARS].div(s.w, axis=0)
    out["n_clear_px"] = s.w
    return out.reset_index().rename(columns={"day": "date"})


def optical_composites(obs, fids, times):
    obs = obs[obs.qa_clear & obs.qa_geometry]
    obs = merge_tiles(obs)
    obs["k"] = period_index(obs.date)
    obs = obs[(obs.k >= 0) & (obs.k < len(times))]
    g = obs.groupby(["fid", "k"])
    med = g[OPT_VARS].median()
    cnt = g.size().rename("n_obs")
    sens = g.sensor.agg(lambda s: "+".join(sorted(set(s)))).rename("sensors")
    comp = pd.concat([med, cnt, sens], axis=1).reset_index()
    fi = pd.Index(fids)
    arr = {v: np.full((len(fids), len(times)), np.nan, np.float32) for v in OPT_VARS}
    nobs = np.zeros((len(fids), len(times)), np.uint8)
    r = fi.get_indexer(comp.fid)
    keep = r >= 0
    for v in OPT_VARS:
        arr[v][r[keep], comp.k[keep]] = comp[v][keep]
    nobs[r[keep], comp.k[keep]] = np.minimum(comp.n_obs[keep], 255)
    filled = np.zeros_like(nobs, dtype=bool)
    lim = MAX_GAP_DAYS // config.COMPOSITE_DAYS
    for v in OPT_VARS:
        a = arr[v]
        df = pd.DataFrame(a.T)
        f = df.interpolate(limit=lim, limit_area="inside", axis=0).to_numpy().T
        # do not fill across gaps longer than lim: pandas fills the first lim steps of longer gaps, undo that
        isn = np.isnan(a)
        runs = _run_lengths(isn)
        f[isn & (runs > lim)] = np.nan
        if v == "ndvi":
            filled = isn & np.isfinite(f)
        arr[v] = f.astype(np.float32)
    return arr, nobs, filled


def _run_lengths(mask):
    """Length of the run of True values each True cell belongs to (along axis 1)."""
    out = np.zeros(mask.shape, np.int32)
    for i in range(mask.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        for s, e in zip(starts, ends):
            out[i, s:e] = e - s
    return out


def load_radar(lo=None, hi=None, start=None, end=None):
    return _load("radar", lo, hi, start, end)


def radar_composites(obs, fids, times):
    obs = obs.assign(k=period_index(obs.date))
    obs = obs[(obs.k >= 0) & (obs.k < len(times))]
    w = obs.n_px.astype(float)
    g = pd.concat([obs[["fid", "k"]], obs[["vv_lin"]].mul(w, axis=0), obs[["vh_lin"]].mul(w, axis=0), w.rename("w")],
                  axis=1).groupby(["fid", "k"]).sum()
    g["n"] = obs.groupby(["fid", "k"]).size()
    fi = pd.Index(fids)
    r = fi.get_indexer(g.index.get_level_values(0))
    k = g.index.get_level_values(1)
    keep = r >= 0
    out = {}
    for v in ("vv", "vh"):
        a = np.full((len(fids), len(times)), np.nan, np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            a[r[keep], k[keep]] = 10 * np.log10((g[f"{v}_lin"] / g.w).to_numpy()[keep])
        out[f"{v}_db"] = a
    n = np.zeros((len(fids), len(times)), np.uint8)
    n[r[keep], k[keep]] = np.minimum(g.n.to_numpy()[keep], 255)
    out["vh_vv_db"] = out["vh_db"] - out["vv_db"]
    return out, n


def weather_composites(fields, times):
    import rasterio
    out = {}
    pts = fields.geometry.representative_point()
    for var, how in (("ppt", "sum"), ("tmean", "mean"), ("tmax", "mean"), ("tmin", "mean"), ("vpdmax", "mean")):
        series = []
        for y in config.YEARS:
            p = config.RAW / "prism" / var / f"prism_{var}_{y}.tif"
            if not p.exists():
                continue
            with rasterio.open(p) as s:
                xy = pts.to_crs(s.crs)
                rows, cols = rasterio.transform.rowcol(s.transform, xy.x.values, xy.y.values)
                rows = np.clip(rows, 0, s.height - 1)
                cols = np.clip(cols, 0, s.width - 1)
                a = s.read()[:, rows, cols]               # (days, fields)
                dates = pd.to_datetime(json.loads(s.tags()["dates"]))
            series.append(pd.DataFrame(a, index=dates))
        if not series:
            continue
        df = pd.concat(series)
        k = ((df.index - pd.Timestamp(config.COMPOSITE_ANCHOR)).days // config.COMPOSITE_DAYS)
        agg = df.groupby(k).agg(how)
        a = np.full((len(fields), len(times)), np.nan, np.float32)
        kk = agg.index.to_numpy()
        ok = (kk >= 0) & (kk < len(times))
        a[:, kk[ok]] = agg.to_numpy()[ok].T
        out[f"{var}_{how}"] = a
    return out


def static_table():
    f = gpd.read_parquet(config.INTERIM / "fields" / "fields.parquet")
    sdir = config.INTERIM / "static"
    for p in (sdir / "soil" / "soil_by_field.parquet", sdir / "terrain" / "terrain_by_field.parquet",
              sdir / "wetlands" / "wetlands_by_field.parquet"):
        if p.exists():
            f = f.merge(pd.read_parquet(p), on="fid", how="left")
    return f


def build(version="v1.0"):
    rel = config.RELEASE / version
    if rel.exists():
        shutil.rmtree(rel)
    rel.mkdir(parents=True)
    fields = static_table().sort_values("fid").reset_index(drop=True)
    fids = fields.fid.to_numpy()
    times = calendar()

    # composites field chunk by field chunk, so the observation tables never sit in memory whole
    opt = {v: np.full((len(fids), len(times)), np.nan, np.float32) for v in OPT_VARS}
    nobs = np.zeros((len(fids), len(times)), np.uint8)
    filled = np.zeros((len(fids), len(times)), bool)
    rad = {v: np.full((len(fids), len(times)), np.nan, np.float32) for v in ("vv_db", "vh_db", "vh_vv_db")}
    nrad = np.zeros((len(fids), len(times)), np.uint8)
    for k0 in range(0, len(fids), CHUNK):
        sl = slice(k0, k0 + CHUNK)
        lo, hi = int(fids[sl][0]), int(fids[sl][-1]) + 1
        o = load_optical(lo, hi)
        if len(o):
            a, n, f_ = optical_composites(screen_optical(o), fids[sl], times)
            for v in OPT_VARS:
                opt[v][sl] = a[v]
            nobs[sl], filled[sl] = n, f_
        r = load_radar(lo, hi)
        if len(r):
            a, n = radar_composites(r, fids[sl], times)
            for v in rad:
                rad[v][sl] = a[v]
            nrad[sl] = n
        print(f"cube fields {k0 + CHUNK}/{len(fids)}", flush=True)
    wx = weather_composites(fields, times)

    ds = xr.Dataset(coords={"field": fields.field_id.to_numpy(), "time": times})
    for v, a in opt.items():
        ds[v] = (("field", "time"), a)
    ds["n_optical_obs"] = (("field", "time"), nobs)
    ds["gap_filled"] = (("field", "time"), filled)
    for v, a in rad.items():
        ds[v] = (("field", "time"), a)
    ds["n_radar_obs"] = (("field", "time"), nrad)
    for v, a in wx.items():
        ds[v] = (("field", "time"), a)
    ds.attrs.update(title="Florida Pasture Cube", version=version, composite_days=config.COMPOSITE_DAYS,
                    anchor=config.COMPOSITE_ANCHOR, license="CC-BY-4.0",
                    max_gap_days_filled=MAX_GAP_DAYS, max_shift_px=MAX_SHIFT_PX)
    enc = {v: {"compressors": None} for v in ()}
    ds.chunk({"field": 2048, "time": 365}).to_zarr(rel / "cube.zarr", mode="w", consolidated=True)

    fields.to_parquet(rel / "fields.parquet")
    for kind, sub in (("optical", "optical_observations"), ("radar", "radar_observations")):
        (rel / sub).mkdir()
        for y in config.YEARS:
            d = _load(kind, start=f"{y}-01-01", end=f"{y + 1}-01-01")
            if not len(d):
                continue
            if kind == "optical":
                d = screen_optical(d).drop(columns=["item"], errors="ignore")
            d.to_parquet(rel / sub / f"{y}.parquet", index=False)
    (rel / "labels").mkdir()
    for p in (config.RAW / "labels").glob("*.parquet"):
        shutil.copy(p, rel / "labels" / p.name)
    return rel


if __name__ == "__main__":
    print(build())
