"""Assemble the Florida Pasture Cube from the per-source interim tables.

Calendar: 5-day periods starting config.COMPOSITE_ANCHOR (period k covers [anchor + 5k, anchor + 5k + 5) days).

Optical composite (per field and period): observations are used (flag 'qa_use') when they have at least
max(3 px, 30% of the field's pixels) clear ('qa_clear'), a geometric shift within 1 pixel or not assessed ('qa_geometry'; 'qa_geometry_status' tells pass, fail and not_assessed apart), all six
field-mean reflectances within [-0.01, 1] ('qa_range'; outside it are missed clouds above 1 and over-corrected
atmosphere below -0.01, whose NDVI sits well above the field's neighbouring dates), and field-mean blue reflectance
at most 0.25 ('qa_haze'; above it NDVI mostly drops below neighbouring dates: residual haze or thin cloud); the
validation measures both effects (stats 'evidence' in validation/summary.json). Used
observations are combined by the median across observations of a period; when a field falls in two overlapping MGRS
tiles on the same date the records are first merged weighted by clear pixels. All observations stay in the long
table with their flags.
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
REFL_RANGE = (-0.01, 1.0)             # field-mean surface reflectance outside this is not a valid observation
MAX_BLUE = 0.25                       # field-mean blue above this is residual haze or thin cloud
REFL_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
CHUNK = 8000                          # fields per composite chunk

_REFL = "median over the period of the field-mean HLS v2.0 surface reflectance, {}"
_IDX = "median over the period of the field-mean of per-pixel {}"
VAR_ATTRS = {
    "field": {"long_name": "field identifier", "comment": "joins fields.parquet on field_id"},
    "time": {"long_name": "start of the 5-day composite period", "comment": "period covers [time, time + 5 days)"},
    "blue": {"units": "1", "long_name": _REFL.format("blue (S30 B02, L30 B02)")},
    "green": {"units": "1", "long_name": _REFL.format("green (S30 B03, L30 B03)")},
    "red": {"units": "1", "long_name": _REFL.format("red (S30 B04, L30 B04)")},
    "nir": {"units": "1", "long_name": _REFL.format("near infrared (S30 B8A, L30 B05)")},
    "swir1": {"units": "1", "long_name": _REFL.format("shortwave infrared 1.6 um (S30 B11, L30 B06)")},
    "swir2": {"units": "1", "long_name": _REFL.format("shortwave infrared 2.2 um (S30 B12, L30 B07)")},
    "ndvi": {"units": "1", "long_name": _IDX.format("NDVI")},
    "evi": {"units": "1", "long_name": _IDX.format("EVI, clipped to [-1, 1]")},
    "ndmi": {"units": "1", "long_name": _IDX.format("NDMI (nir - swir1) / (nir + swir1)")},
    "mndwi": {"units": "1", "long_name": _IDX.format("MNDWI (green - swir1) / (green + swir1)")},
    "nbr": {"units": "1", "long_name": _IDX.format("NBR (nir - swir2) / (nir + swir2)")},
    "ndvi_sd": {"units": "1", "long_name": "median over the period of the within-field standard deviation of NDVI"},
    "water_frac": {"units": "1", "long_name": "median over the period of the fraction of clear field pixels "
                                              "flagged as water by Fmask"},
    "n_optical_obs": {"units": "1", "long_name": "number of HLS acquisition days in the period that passed the "
                                                 "clear-sky, geometry, reflectance-range and haze screens"},
    "gap_filled": {"long_name": "optical values linearly interpolated in time (no observation in the period, gap "
                                f"of at most {MAX_GAP_DAYS} days)"},
    "vv_db": {"units": "dB", "long_name": "Sentinel-1 RTC gamma0 VV, mean in linear power over the period's "
                                          "acquisitions, converted to dB"},
    "vh_db": {"units": "dB", "long_name": "Sentinel-1 RTC gamma0 VH, mean in linear power over the period's "
                                          "dual-polarisation acquisitions, converted to dB"},
    "vh_vv_db": {"units": "dB", "long_name": "VH minus VV backscatter"},
    "n_radar_obs": {"units": "1", "long_name": "number of Sentinel-1 acquisitions in the period"},
    "ppt_sum": {"units": "mm", "long_name": "PRISM precipitation, sum over the period"},
    "tmean_mean": {"units": "degC", "long_name": "PRISM mean daily temperature, mean over the period"},
    "tmax_mean": {"units": "degC", "long_name": "PRISM maximum daily temperature, mean over the period"},
    "tmin_mean": {"units": "degC", "long_name": "PRISM minimum daily temperature, mean over the period"},
    "vpdmax_mean": {"units": "hPa", "long_name": "PRISM maximum daily vapour pressure deficit, mean over the period"},
}


def calendar():
    """Start dates of the complete periods between COMPOSITE_ANCHOR and END (a trailing partial period is left out,
    so every period sums or averages the same number of days)."""
    a = pd.Timestamp(config.COMPOSITE_ANCHOR)
    n = ((pd.Timestamp(config.END) - a).days + 1) // config.COMPOSITE_DAYS
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


GEOMETRY_STATES = ("pass", "fail", "not_assessed")
MIN_SCENE_PEAK = 0.1      # scene-level estimates with a weaker phase-correlation peak are noise (72 of 74 'fails')


def load_scene_shifts():
    """Scene-level shift estimates (fpcube.sources.scene_shift) for granules whose field window was too small."""
    parts = [pd.read_parquet(p) for p in (config.INTERIM / "optical" / "scene_shift").glob("T*.parquet")]
    if not parts:
        return None
    s = pd.concat(parts, ignore_index=True).drop_duplicates("item")
    s = s[s.shift_peak >= MIN_SCENE_PEAK]                   # weaker peaks: the granule stays not assessed
    return s[["item", "shift_row", "shift_col", "shift_peak"]].dropna(subset=["shift_row"])


def screen_optical(d, scene=None):
    """Quality flags of each field x acquisition record. The geometric shift is estimated during extraction over the
    window of clear field pixels when it is at least 64 x 64 pixels ('fields' scope); for the other granules it is
    taken from a scene-level estimate over a 512 x 512 clear block of the tile ('scene' scope) when one exists with a
    phase-correlation peak of at least MIN_SCENE_PEAK.
    Records with neither stay 'not_assessed': kept (qa_geometry True) but with 'shift_px' missing, so users can
    exclude them."""
    d = d.drop(columns=[c for c in ("qa_geometry_status", "shift_scope", "shift_px") if c in d])
    scope = np.where(d.shift_row.notna(), "fields", "none").astype(object)
    if scene is not None and len(scene):
        m = d[["item"]].merge(scene, on="item", how="left")
        fill = d.shift_row.isna().to_numpy() & m.shift_row.notna().to_numpy()
        for c in ("shift_row", "shift_col", "shift_peak"):
            d[c] = np.where(fill, m[c].to_numpy(), d[c].to_numpy())
        scope[fill] = "scene"
    ok_clear = (d.n_clear_px >= np.maximum(3, 0.3 * d.n_field_px))
    sh = np.hypot(d.shift_row, d.shift_col)                     # NaN when the shift was not estimated
    assessed = sh.notna()
    status = np.where(~assessed, "not_assessed", np.where(sh <= MAX_SHIFT_PX, "pass", "fail"))
    ok_geo = pd.Series(status != "fail", index=d.index)
    r = d[REFL_BANDS]
    ok_range = ((r >= REFL_RANGE[0]) & (r <= REFL_RANGE[1])).all(axis=1)
    ok_haze = d.blue <= MAX_BLUE
    d = d.assign(qa_clear=ok_clear, qa_geometry=ok_geo,
                 qa_geometry_status=pd.Categorical(status, categories=GEOMETRY_STATES),
                 shift_scope=pd.Categorical(scope, categories=("fields", "scene", "none")),
                 qa_range=ok_range, qa_haze=ok_haze, qa_use=ok_clear & ok_geo & ok_range & ok_haze, shift_px=sh)
    return d


def apply_scene_shift(version="v1.0"):
    """Bring the scene-level shifts into a built release: re-screen the optical tables, then recompute the optical
    composites of the fields whose usable records changed (newly failed granules) and write them into cube.zarr and
    fields.parquet in place. Returns a summary."""
    import zarr
    rel = config.RELEASE / version
    scene = load_scene_shifts()
    changed, stats = set(), {}
    for p in sorted((rel / "optical_observations").glob("*.parquet")):
        d0 = pd.read_parquet(p)
        if "shift_scope" in d0:                                  # undo an earlier fill so this is repeatable
            back = d0.shift_scope == "scene"
            d0.loc[back, ["shift_row", "shift_col", "shift_peak"]] = np.nan
        d = screen_optical(d0, scene)
        diff = d0.qa_use.to_numpy() != d.qa_use.to_numpy()
        changed |= set(d.fid[diff].tolist())
        stats[p.stem] = {"records": int(len(d)), "scene_assessed": int((d.shift_scope == "scene").sum()),
                         "still_not_assessed": int((d.qa_geometry_status == "not_assessed").sum()),
                         "newly_failed": int(diff.sum())}
        tmp = p.with_suffix(".tmp")
        d.to_parquet(tmp, index=False)
        tmp.replace(p)
        print(p.stem, stats[p.stem], flush=True)
    if changed:
        fields = pd.read_parquet(rel / "fields.parquet")
        fids_all = fields.fid.to_numpy()
        fids = np.array(sorted(changed))
        times = calendar()
        obs = pd.concat([pd.read_parquet(p, filters=[("fid", "in", fids.tolist())])
                         for p in sorted((rel / "optical_observations").glob("*.parquet"))], ignore_index=True)
        a, n, f_ = optical_composites(obs, fids, times)
        idx = np.searchsorted(fids_all, fids)
        z = zarr.open(str(rel / "cube.zarr"), mode="r+")
        for v in OPT_VARS:
            z[v].set_orthogonal_selection((idx, slice(None)), a[v])
        z["n_optical_obs"].set_orthogonal_selection((idx, slice(None)), n)
        z["gap_filled"].set_orthogonal_selection((idx, slice(None)), f_)
        nobs_all = z["n_optical_obs"][:]
        nrad_all = z["n_radar_obs"][:]
        fields["n_optical_periods"] = (nobs_all > 0).sum(1).astype(np.int16)
        fields["flag_no_optical"] = fields.n_optical_periods == 0
        fields["flag_no_observations"] = fields.flag_no_optical & ((nrad_all > 0).sum(1) == 0)
        g = __import__("geopandas").read_parquet(rel / "fields.parquet")
        for c in ("n_optical_periods", "flag_no_optical", "flag_no_observations"):
            g[c] = fields[c].to_numpy()
        tmp = rel / "fields.tmp.parquet"
        g.to_parquet(tmp)
        tmp.replace(rel / "fields.parquet")
    stats["fields_recomposited"] = len(changed)
    (rel / "scene_shift_applied.json").write_text(__import__("json").dumps(stats, indent=1))
    return stats


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
    obs = obs[obs.qa_use]
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
        f = df.interpolate(limit=lim, limit_area="inside", axis=0).to_numpy(copy=True).T   # writable
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
    n, t = mask.shape
    m = np.concatenate([mask, np.zeros((n, 1), bool)], axis=1).ravel()   # a False column keeps rows apart
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    lengths = np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]
    out = np.zeros(m.shape, np.int32)
    out[m] = np.repeat(lengths, lengths)                                  # True cells come run by run, in order
    return out.reshape(n, t + 1)[:, :t]


def load_radar(lo=None, hi=None, start=None, end=None):
    return _load("radar", lo, hi, start, end)


def radar_composites(obs, fids, times):
    obs = obs.assign(k=period_index(obs.date))
    obs = obs[(obs.k >= 0) & (obs.k < len(times))]
    w = obs.n_px.astype(float)
    wh = w.where(np.isfinite(obs.vh_lin), 0.0)            # VV-only scenes carry no VH: no weight in the VH mean
    g = pd.concat([obs[["fid", "k"]], (obs.vv_lin * w).rename("vv_lin"), (obs.vh_lin.fillna(0) * wh).rename("vh_lin"),
                   w.rename("w_vv"), wh.rename("w_vh")], axis=1).groupby(["fid", "k"]).sum()
    g["n"] = obs.groupby(["fid", "k"]).size()
    fi = pd.Index(fids)
    r = fi.get_indexer(g.index.get_level_values(0))
    k = g.index.get_level_values(1)
    keep = r >= 0
    out = {}
    for v in ("vv", "vh"):
        a = np.full((len(fids), len(times)), np.nan, np.float32)
        wv = g[f"w_{v}"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(wv > 0, g[f"{v}_lin"].to_numpy() / wv, np.nan)
            a[r[keep], k[keep]] = 10 * np.log10(m[keep])
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
    scene = load_scene_shifts()

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
            a, n, f_ = optical_composites(screen_optical(o, scene), fids[sl], times)
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
    # observation coverage per field: narrow fields can have no pixel left after the inward buffers
    n_opt_p, n_rad_p = (nobs > 0).sum(1), (nrad > 0).sum(1)
    fields["n_optical_periods"] = n_opt_p.astype(np.int16)
    fields["n_radar_periods"] = n_rad_p.astype(np.int16)
    fields["flag_no_optical"] = n_opt_p == 0
    fields["flag_no_observations"] = (n_opt_p == 0) & (n_rad_p == 0)

    ids = fields.field_id.to_numpy().astype(str)          # fixed-width text: readable by any Zarr v2 reader
    ds = xr.Dataset(coords={"field": ids, "time": times})
    for v, a in opt.items():
        ds[v] = (("field", "time"), a)
    ds["n_optical_obs"] = (("field", "time"), nobs)
    ds["gap_filled"] = (("field", "time"), filled)
    for v, a in rad.items():
        ds[v] = (("field", "time"), a)
    ds["n_radar_obs"] = (("field", "time"), nrad)
    for v, a in wx.items():
        ds[v] = (("field", "time"), a)
    for v in ds.variables:
        ds[v].attrs.update(VAR_ATTRS.get(v, {}))
    end = times[-1] + pd.Timedelta(days=config.COMPOSITE_DAYS - 1)
    ds.attrs.update(title="Florida Pasture Cube", version=version, license="CC-BY-4.0", Conventions="CF-1.10",
                    summary="Per-field 5-day composites of HLS v2.0 optical reflectance and indices, Sentinel-1 RTC "
                            "backscatter and PRISM daily weather for Florida pasture fields; static field attributes "
                            "are in fields.parquet (join on field_id).",
                    source="NASA HLS v2.0 S30/L30 (LP DAAC, Microsoft Planetary Computer); Sentinel-1 RTC "
                           "(Microsoft Planetary Computer); PRISM AN81d daily",
                    time_coverage_start=str(times[0].date()), time_coverage_end=str(end.date()),
                    composite_days=config.COMPOSITE_DAYS, anchor=config.COMPOSITE_ANCHOR,
                    max_gap_days_filled=MAX_GAP_DAYS, max_shift_px=MAX_SHIFT_PX,
                    reflectance_valid_min=REFL_RANGE[0], reflectance_valid_max=REFL_RANGE[1], max_blue_haze=MAX_BLUE)
    ds.chunk({"field": 2048, "time": 365}).to_zarr(rel / "cube.zarr", mode="w", consolidated=True, zarr_format=2)

    fields.to_parquet(rel / "fields.parquet")
    for kind, sub in (("optical", "optical_observations"), ("radar", "radar_observations")):
        (rel / sub).mkdir()
        for y in config.YEARS:
            d = _load(kind, start=f"{y}-01-01", end=f"{y + 1}-01-01")
            if not len(d):
                continue
            if kind == "optical":
                d = screen_optical(d, scene)              # keeps 'item', the HLS granule id
            d.to_parquet(rel / sub / f"{y}.parquet", index=False)
    (rel / "labels").mkdir()
    for p in (config.RAW / "labels").glob("*.parquet"):
        shutil.copy(p, rel / "labels" / p.name)
    return rel


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--apply-scene-shift"]:
        print(apply_scene_shift(*sys.argv[2:3]))
    else:
        print(build())
