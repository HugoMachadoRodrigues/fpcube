"""End-to-end dry run of the downstream stages (validation, benchmark, paper numbers, figures) on a synthetic
release that uses real field geometries and labels but simulated time series. Catches code errors before the
multi-day extraction finishes.

python tests/e2e_synthetic.py <out_dir> [n_fields]
"""
import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from fpcube import config
from fpcube import cube as C

out = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
rel = out / "v1.0"
rel.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)

f = gpd.read_parquet(config.INTERIM / "fields" / "fields.parquet")
ona = gpd.GeoSeries(gpd.points_from_xy([-81.94], [27.40]), crs=4326).to_crs(f.crs).iloc[0]
near = f.assign(d=f.distance(ona)).sort_values("d").head(n // 2)
pan = f[f.lat > 30.3].sample(min(n // 2, (f.lat > 30.3).sum()), random_state=1)
f = pd.concat([near, pan]).drop(columns="d").drop_duplicates("fid").sort_values("fid").reset_index(drop=True)
for c, lo, hi in (("hand_mean_m", 0, 8), ("twi_mean", 5, 15), ("elev_mean_m", 2, 60), ("aws0_100_mm", 50, 250),
                  ("drainage_code", 1, 7), ("flooding_code", 0, 4), ("wetland_frac", 0, 0.3)):
    f[c] = rng.uniform(lo, hi, len(f))
f.to_parquet(rel / "fields.parquet")

times = C.calendar()
T = len(times)
doy = np.asarray(times.day_of_year)
seas = 0.6 + 0.18 * np.sin(2 * np.pi * (doy - 110) / 365.25)
ppt = rng.gamma(0.6, 12, (len(f), T)).astype(np.float32)
nd = seas[None] + 0.04 * rng.standard_normal((len(f), T))
ian = int(np.searchsorted(times, pd.Timestamp("2022-09-28")))
ppt[: len(f) // 3, ian - 1:ian + 1] += 120
nd[: len(f) // 3, ian:ian + 3] -= 0.15
obs = rng.random(nd.shape) < 0.55
filled = (~obs) & (rng.random(nd.shape) < 0.4)
nd = np.where(obs | filled, nd, np.nan).astype(np.float32)
ds = xr.Dataset(coords={"field": f.field_id.values, "time": times})
for v in C.OPT_VARS:
    ds[v] = (("field", "time"), (nd + 0.02 * rng.standard_normal(nd.shape)).astype(np.float32))
ds["ndvi"] = (("field", "time"), nd)
ds["n_optical_obs"] = (("field", "time"), obs.astype(np.uint8))
ds["gap_filled"] = (("field", "time"), filled)
vv = (-11 + 0.04 * np.minimum(ppt, 60) + 0.5 * rng.standard_normal(nd.shape)).astype(np.float32)
ds["vv_db"] = (("field", "time"), vv)
ds["vh_db"] = (("field", "time"), vv - 5)
ds["vh_vv_db"] = (("field", "time"), np.full_like(vv, -5))
ds["n_radar_obs"] = (("field", "time"), np.ones_like(obs, np.uint8))
ds["ppt_sum"] = (("field", "time"), ppt)
for v, m in (("tmean_mean", 22), ("tmax_mean", 28), ("tmin_mean", 16), ("vpdmax_mean", 15)):
    ds[v] = (("field", "time"), (m + 6 * np.sin(2 * np.pi * (doy - 110) / 365.25)[None] +
                                 rng.standard_normal((len(f), T))).astype(np.float32))
ds.to_zarr(rel / "cube.zarr", mode="w")
for name, nobs in (("optical_observations", 40), ("radar_observations", 20)):
    (rel / name).mkdir(exist_ok=True)
    d = pd.DataFrame({"fid": np.repeat(f.fid.values, nobs),
                      "date": np.tile(pd.date_range("2016-01-01", "2026-10-01", periods=nobs), len(f))})
    if name.startswith("optical"):
        d["sensor"] = rng.choice(["S30", "L30"], len(d)); d["qa_clear"] = rng.random(len(d)) < 0.6
        d["qa_geometry"] = rng.random(len(d)) < 0.98
    else:
        d["platform"] = "SENTINEL-1A"
    for y, g in d.groupby(d.date.dt.year):
        g.to_parquet(rel / name / f"{y}.parquet", index=False)
(rel / "labels").mkdir(exist_ok=True)

# point the package at the synthetic release
config.RELEASE = out
from fpcube import validation as V                         # noqa: E402
from fpcube.benchmark import run as B                      # noqa: E402
V.config.RELEASE = out
B.config.RELEASE = out
wc = pd.DataFrame({"fid": f.fid, "grassland": 0.7, "cropland": 0.1, "tree": 0.1, "built": 0.01,
                   "herbaceous_wetland": 0.05, "water": 0.04})
(rel / "validation").mkdir(exist_ok=True)
wc.to_csv(rel / "validation" / "worldcover_by_field.csv", index=False)
s = V.main("v1.0")
print("validation ok:", s["ValidationPhysical"][:200])

sys.argv = ["run", "task1", "task3", "--release", "v1.0", "--max-train", "40000"]
B.main()
print(pd.read_csv(rel / "benchmark" / "task1_results.csv").query("horizon_days == 30 and split == 'test'"))
print(pd.read_csv(rel / "benchmark" / "task3_results.csv"))

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fg", root / "scripts" / "figures.py")
fg = importlib.util.module_from_spec(spec); spec.loader.exec_module(fg)
fg.OUT = out / "figures"
for name in ("fig2", "fig3", "fig4", "fig5"):
    getattr(fg, name)(rel)
print("figures ok")
