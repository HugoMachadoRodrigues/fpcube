"""Read-only audit of release/v1.0/cube.zarr: structure, coverage, value ranges, internal consistency, and an
independent recomputation of the composites for a random sample of fields from the interim tables."""
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import xarray as xr
import zarr

from fpcube import config

R = config.RELEASE / "v1.0"
ds = xr.open_zarr(R / "cube.zarr")
F = pd.read_parquet(R / "fields.parquet", columns=["field_id", "fid", "tier", "area_ha", "lat", "lon", "county",
                                                   "n_interior_pixels"])
times = pd.DatetimeIndex(ds.time.values)
issues = []


def say(*a):
    print(*a, flush=True)


def issue(msg):
    issues.append(msg)
    say("  !!", msg)


# ------------------------------------------------------------------ structure
say("== structure")
say(f"fields {ds.sizes['field']:,}, periods {ds.sizes['time']}, {times[0].date()} .. {times[-1].date()}")
if not (np.diff(times.values).astype("timedelta64[D]").astype(int) == 5).all():
    issue("time step is not 5 days everywhere")
if not np.array_equal(np.array(ds.field.values.tolist(), dtype=object), F.sort_values("fid").field_id.to_numpy(dtype=object)):
    issue("cube field order differs from fields.parquet sorted by fid")
if pd.Index(np.array(ds.field.values.tolist(), dtype=object)).duplicated().any():
    issue("duplicate field ids")
last_end = times[-1] + pd.Timedelta(days=5)
if last_end > pd.Timestamp(config.END) + pd.Timedelta(days=1):
    issue(f"last period {times[-1].date()}..{(last_end - pd.Timedelta(days=1)).date()} runs past END {config.END}: "
          f"{(last_end - pd.Timestamp(config.END)).days - 1} of 5 days have no data")
z = zarr.open_group(str(R / "cube.zarr"), mode="r")
say("zarr format", z.metadata.zarr_format, "| field coord dtype", z["field"].dtype, "| ndvi codecs",
    [type(c).__name__ for c in getattr(z["ndvi"].metadata, "codecs", [])])
if z.metadata.zarr_format != 2 or z["field"].dtype.kind not in "UO":
    issue(f"expected Zarr v2 with fixed-width string ids, got format {z.metadata.zarr_format}, dtype {z['field'].dtype}")
if ds.attrs.get("Conventions") != "CF-1.10":
    issue("global attribute Conventions is not CF-1.10")
say("global attrs:", sorted(ds.attrs))
missing_attrs = [v for v in list(ds.data_vars) + ["field", "time"] if not ds[v].attrs.get("long_name")]
if missing_attrs:
    issue(f"{len(missing_attrs)} of {len(ds.data_vars)} variables have no units/long_name attributes")


def load(v):
    return ds[v].values


# ------------------------------------------------------------------ optical
say("== optical")
OPT = ["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "evi", "ndmi", "mndwi", "nbr", "ndvi_sd", "water_frac"]
n = load("n_optical_obs")
gf = load("gap_filled")
nd = load("ndvi")
obs = n > 0
fin = np.isfinite(nd)
say(f"cells observed {obs.mean():.3f}, gap-filled {gf.mean():.3f}, empty {(~fin).mean():.3f}; "
    f"share of finite values that are interpolated {gf.sum() / fin.sum():.3f}")
if (gf & obs).any():
    issue(f"{int((gf & obs).sum()):,} cells flagged gap_filled although observed")
if (fin != (obs | gf)).any():
    issue(f"{int((fin != (obs | gf)).sum()):,} cells where finite ndvi != observed or gap_filled")
never = ~obs.any(1)
say(f"fields never observed {never.sum()} (tier {F.sort_values('fid').tier.values[never].tolist().count('A')} A)")
yr = times.year
say("observed share by year:", {int(y): round(float(obs[:, yr == y].mean()), 3) for y in np.unique(yr)})
mon = times.month
say("median observed ndvi by month:", {int(m): round(float(np.nanmedian(np.where(obs, nd, np.nan)[:, mon == m])), 3)
                                       for m in range(1, 13)})
nan_pattern = {}
for v in OPT:
    a = load(v)
    f = np.isfinite(a)
    d = int((f != fin).sum())
    if d:
        nan_pattern[v] = d
    lo, hi = np.nanpercentile(a, [0.01, 99.99])
    say(f"  {v:10s} min {np.nanmin(a):8.3f} p0.01 {lo:8.3f} median {np.nanmedian(a):7.3f} p99.99 {hi:7.3f} "
        f"max {np.nanmax(a):7.3f}")
    if v in ("ndvi", "evi", "ndmi", "mndwi", "nbr") and (np.nanmin(a) < -1 or np.nanmax(a) > 1):
        issue(f"{v} outside [-1, 1]")
    if v in ("water_frac",) and (np.nanmin(a) < 0 or np.nanmax(a) > 1):
        issue("water_frac outside [0, 1]")
    if v in ("blue", "green", "red", "nir", "swir1", "swir2") and (np.nanmin(a) < -0.01 - 1e-6 or np.nanmax(a) > 1 + 1e-6):
        issue(f"{v} outside the [-0.01, 1] screen")
    if v == "blue" and np.nanmax(a) > 0.25 + 1e-6:
        issue("blue above the 0.25 haze screen")
    if v in ("blue", "green", "red", "nir", "swir1", "swir2"):
        neg = float((a < 0).sum() / np.isfinite(a).sum())
        if neg > 0:
            say(f"             share < 0: {neg:.5f}")
    del a
if nan_pattern:
    issue(f"optical variables with a NaN pattern different from ndvi: {nan_pattern}")

# ------------------------------------------------------------------ radar
say("== radar")
vv, vh, ratio, nr = load("vv_db"), load("vh_db"), load("vh_vv_db"), load("n_radar_obs")
ro = nr > 0
say(f"cells observed {ro.mean():.3f}; first period with data {times[ro.any(0)][0].date()}")
say("observed share by year:", {int(y): round(float(ro[:, yr == y].mean()), 3) for y in np.unique(yr)})
if (np.isfinite(vv) != ro).any():
    issue(f"{int((np.isfinite(vv) != ro).sum()):,} cells where finite vv != n_radar_obs > 0")
d = np.nanmax(np.abs(ratio - (vh - vv)))
say(f"vv median {np.nanmedian(vv):.2f} dB [{np.nanpercentile(vv, 0.01):.1f}, {np.nanpercentile(vv, 99.99):.1f}], "
    f"vh median {np.nanmedian(vh):.2f}; max |ratio - (vh - vv)| {d:.2e}")
say(f"vh missing where vv present {int((np.isfinite(vv) & ~np.isfinite(vh)).sum()):,}; "
    f"infinite values {int(np.isinf(vv).sum() + np.isinf(vh).sum())}")
say("vv median by year:", {int(y): round(float(np.nanmedian(vv[:, yr == y])), 2) for y in np.unique(yr)})
say("share of observed vv below -25 dB by year:",
    {int(y): round(float((vv[:, yr == y] < -25).sum() / ro[:, yr == y].sum()), 5) for y in np.unique(yr)})
if np.nanmin(vv) < -30 - 1e-4:
    issue(f"vv below the -30 dB pixel floor: min {np.nanmin(vv):.2f}")
if np.isinf(vv).any() or np.isinf(vh).any() or np.isinf(ratio).any():
    issue("infinite radar values")
del vv, vh, ratio

# ------------------------------------------------------------------ weather
say("== weather")
for v in ("ppt_sum", "tmean_mean", "tmax_mean", "tmin_mean", "vpdmax_mean"):
    a = load(v)
    miss = np.isnan(a).mean(0)
    say(f"  {v:12s} missing {np.isnan(a).mean():.4f}; median {np.nanmedian(a):7.2f}; min {np.nanmin(a):7.2f} "
        f"max {np.nanmax(a):7.2f}; last 3 periods mean {np.nanmean(a[:, -3:], 0).round(2).tolist()}")
    if miss.any():
        say(f"     periods with any missing: {times[miss > 0].date.tolist()[:5]}")
    if v == "ppt_sum":
        say(f"     statewide mean annual ppt (sum of period sums / years) "
            f"{np.nanmean(a[:, yr < 2026].sum(1)) / 10:.0f} mm/yr")
    del a
tx, tn, tm = load("tmax_mean"), load("tmin_mean"), load("tmean_mean")
bad = int((tx < tn - 1e-3).sum())
if bad:
    issue(f"{bad} cells with tmax < tmin")
del tx, tn, tm

# ------------------------------------------------------------------ independent recomputation
say("== recomputation for 300 random fields from the interim tables")
rng = np.random.default_rng(3)
Fs = F.sort_values("fid").reset_index(drop=True)
pick = np.sort(rng.choice(len(Fs), 300, replace=False))
fids = Fs.fid.values[pick]
flt = pads.field("fid").isin(fids.tolist())
o = pads.dataset([str(p) for p in sorted((config.INTERIM / "optical").glob("T*/*.parquet"))]).to_table(filter=flt) \
    .to_pandas()
o["date"] = pd.to_datetime(o.date)
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
ok = (o.n_clear_px >= np.maximum(3, 0.3 * o.n_field_px)) & \
     (np.hypot(o.shift_row.fillna(0), o.shift_col.fillna(0)) <= 1.0) & \
     ((o[BANDS] >= -0.01) & (o[BANDS] <= 1.0)).all(axis=1) & (o.blue <= 0.25)
o = o[ok]
o["day"] = o.date.dt.floor("D")
o["w"] = o.n_clear_px.astype(float)
o["wnd"] = o.ndvi * o.w
g = o.groupby(["fid", "day", "sensor"])[["wnd", "w"]].sum()
g["ndvi"] = g.wnd / g.w
g = g.reset_index()
g["k"] = ((g.day - pd.Timestamp(config.COMPOSITE_ANCHOR)).dt.days // 5)
g = g[(g.k >= 0) & (g.k < len(times))]
med = g.groupby(["fid", "k"]).ndvi.median()
cnt = g.groupby(["fid", "k"]).size()
row = pd.Series(pick, index=fids)
cube_nd = nd[row.loc[med.index.get_level_values(0)].values, med.index.get_level_values(1)]
diff = np.abs(cube_nd - med.values)
say(f"optical cells compared {len(med):,}; max |diff| {np.nanmax(diff):.2e}; cube NaN where recomputed has data "
    f"{int(np.isnan(cube_nd).sum())}")
cube_n = n[row.loc[cnt.index.get_level_values(0)].values, cnt.index.get_level_values(1)]
say(f"n_optical_obs mismatches {int((cube_n != np.minimum(cnt.values, 255)).sum())}")
sub_obs = obs[pick]
rec = np.zeros_like(sub_obs)
rec[row.loc[med.index.get_level_values(0)].map(dict(zip(pick, range(300)))).values, med.index.get_level_values(1)] = 1
say(f"observed cells in cube but not recomputed {int((sub_obs & ~rec).sum())}")
if np.nanmax(diff) > 1e-5 or np.isnan(cube_nd).any():
    issue("optical composites differ from the independent recomputation")

r = pads.dataset([str(p) for p in sorted((config.INTERIM / "radar" / "s1").glob("2*/*.parquet"))]) \
    .to_table(filter=flt).to_pandas()
r["date"] = pd.to_datetime(r.date)
r["k"] = ((r.date - pd.Timestamp(config.COMPOSITE_ANCHOR)).dt.days // 5)
r = r[(r.k >= 0) & (r.k < len(times))]
r["w"] = r.n_px.astype(float)
r["wv"] = r.vv_lin * r.w
gr = r.groupby(["fid", "k"])[["wv", "w"]].sum()
rv = 10 * np.log10(gr.wv / gr.w)
vv = load("vv_db")
cv = vv[row.loc[gr.index.get_level_values(0)].values, gr.index.get_level_values(1)]
say(f"radar cells compared {len(rv):,}; max |diff| {np.nanmax(np.abs(cv - rv.values)):.2e} dB")
same_day_orbits = r.groupby(["fid", "k"]).relative_orbit.nunique()
say(f"radar periods mixing relative orbits {float((same_day_orbits > 1).mean()):.3f}; "
    f"mixing ascending+descending {float((r.groupby(['fid', 'k']).orbit_state.nunique() > 1).mean()):.3f}")

# ------------------------------------------------------------------ release observation tables
say("== release observation tables")
for sub, src in (("optical_observations", sorted((config.INTERIM / "optical").glob("T*/*.parquet"))),
                 ("radar_observations", sorted((config.INTERIM / "radar" / "s1").glob("2*/*.parquet")))):
    nrel = pads.dataset(str(R / sub)).count_rows()
    nint = pads.dataset([str(p) for p in src]).count_rows()
    say(f"{sub}: release {nrel:,} rows, interim {nint:,}")
    if nrel != nint:
        issue(f"{sub}: {nint - nrel:,} interim rows missing from the release")
names = pads.dataset(str(R / "optical_observations")).schema.names
miss = [c for c in ("qa_clear", "qa_geometry", "qa_range", "qa_haze", "qa_use") if c not in names]
if miss:
    issue(f"optical_observations lacks flags {miss}")
else:
    q = pads.dataset(str(R / "optical_observations")).to_table(columns=["qa_clear", "qa_geometry", "qa_range", "qa_haze", "qa_use"]).to_pandas()
    say("optical flags, share false:", {c: round(float((~q[c]).mean()), 5) for c in q.columns},
        "| used", f"{int(q.qa_use.sum()):,}")
    if (q.qa_use != (q.qa_clear & q.qa_geometry & q.qa_range & q.qa_haze)).any():
        issue("qa_use is not the AND of the four flags")
    del q
if "item" not in names:
    issue("optical_observations has no HLS granule id ('item' dropped), radar_observations keeps its scene id")

say("\n== ISSUES")
for i in issues:
    say("-", i)
sys.exit(0)
