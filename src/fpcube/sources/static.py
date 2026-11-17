"""Static per-field layers: soil (gNATSGO), terrain (USGS 3DEP 1 arc-second) and wetlands (NOAA C-CAP).

Soil: majority map unit (mukey) per field from the gNATSGO 10 m raster, joined to the muaggatt table
(drainage class, flooding frequency, ponding frequency, hydrologic group, water-table depth); area-weighted
means of available water storage 0-100 cm and soil organic carbon 0-100 cm.
Terrain: 3DEP seamless 1 arc-second mosaic reprojected to the CDL grid (EPSG:5070, 30 m); WhiteboxTools
least-cost depression breaching, D8 flow accumulation, streams at 1 km2 contributing area, height above
nearest drainage (HAND, ElevationAboveStream) and topographic wetness index. Per field: mean elevation,
slope, HAND and TWI.
Wetlands: NOAA C-CAP wetland classes, fraction inside the field and inside a 500 m ring around it.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling

from fpcube import config

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
OUT = config.INTERIM / "static"
DRAIN = {"Excessively drained": 1, "Somewhat excessively drained": 2, "Well drained": 3,
         "Moderately well drained": 4, "Somewhat poorly drained": 5, "Poorly drained": 6,
         "Very poorly drained": 7, "Subaqueous": 8}
FLOOD = {"None": 0, "Very rare": 1, "Rare": 2, "Occasional": 3, "Frequent": 4, "Very frequent": 5}
CCAP_WET = (13, 14, 15, 16, 17, 18, 22, 23)   # palustrine and estuarine forested, scrub-shrub, emergent, aquatic bed


def _cat():
    import planetary_computer as pc
    import pystac_client
    return pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)


def _fields():
    return gpd.read_parquet(config.INTERIM / "fields" / "fields.parquet")[["fid", "geometry"]]


def clip_to(gdf, b, inset=1.0):
    """Clip geometries to raster bounds (slightly inset); drops empty results."""
    from shapely.geometry import box as _box
    g = gdf.copy()
    g["geometry"] = g.geometry.intersection(_box(b.left + inset, b.bottom + inset, b.right - inset, b.top - inset))
    return g[~g.geometry.is_empty & (g.geometry.area > 0)]


def _zonal(raster_path_or_ds, fields, ops):
    from exactextract import exact_extract
    df = exact_extract(raster_path_or_ds, fields, ops, include_cols=["fid"], output="pandas")
    return df


def soil():
    cat = _cat()
    f = _fields()
    bbox = list(f.to_crs(4326).total_bounds)
    items = list(cat.search(collections=["gnatsgo-rasters"], bbox=bbox).items())
    out = OUT / "soil"
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for k, it in enumerate(items):
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            if "mukey" not in it.assets:
                continue
            src = {a: rasterio.open(it.assets[a].href) for a in ("mukey", "aws0_100", "soc0_100") if a in it.assets}
            b = src["mukey"].bounds
            sub = f.cx[b.left:b.right, b.bottom:b.top]
            if not len(sub):
                continue
            sub = clip_to(sub, b)                      # exactextract 0.3 segfaults on polygons leaving the raster
            d = _zonal(src["mukey"], sub, ["majority", "count", "unique", "frac"])
            d["majority_frac"] = [float(np.max(fr)) if len(fr) else np.nan for fr in d["frac"]]
            d = d.drop(columns=["unique", "frac"])
            for a, col in (("aws0_100", "aws0_100_mm"), ("soc0_100", "soc0_100_gm2")):
                d[col] = _zonal(src[a], sub, ["mean"])["mean"].values if a in src else np.nan
            d["tile"] = it.id
            res.append(d)
        print("soil tile", k + 1, len(items), flush=True)
    d = pd.concat(res, ignore_index=True).sort_values("count").drop_duplicates("fid", keep="last").drop(columns="count")
    d = d.rename(columns={"majority": "mukey", "majority_frac": "mukey_frac"})
    t = next(cat.search(collections=["gnatsgo-tables"], ids=["muaggatt"]).items())
    mua = pd.read_parquet(t.assets["data"].href, storage_options=t.assets["data"].extra_fields.get(
        "table:storage_options", {}))
    mua["mukey"] = mua.mukey.astype(np.int64)
    cols = {"drclassdcd": "drainage_class", "flodfreqdcd": "flooding_frequency", "pondfreqprs": "ponding_pct",
            "hydgrpdcd": "hydrologic_group", "wtdepannmin": "water_table_min_cm", "musym": "musym",
            "muname": "muname"}
    mua = mua[["mukey"] + [c for c in cols if c in mua]].rename(columns=cols)
    d["mukey"] = d.mukey.astype("Int64")
    d = d.merge(mua, on="mukey", how="left")
    d["drainage_code"] = d.drainage_class.map(DRAIN)
    d["flooding_code"] = d.flooding_frequency.map(FLOOD)
    d.drop(columns=["tile"]).to_parquet(out / "soil_by_field.parquet", index=False)
    return d


def dem_mosaic():
    out = OUT / "terrain"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "dem30_5070.tif"
    if target.exists():
        return target
    cat = _cat()
    items = [it for it in cat.search(collections=["3dep-seamless"], bbox=[-87.7, 24.4, -79.9, 31.1]).items()
             if it.properties.get("gsd") == 30]
    srcs = [rasterio.open(it.assets["data"].href) for it in items]
    mos, tr = merge(srcs)
    with rasterio.open(config.INTERIM / "fields" / "field_labels.tif") as ref:
        dst = np.full((ref.height, ref.width), np.nan, np.float32)
        reproject(mos[0], dst, src_transform=tr, src_crs=srcs[0].crs, dst_transform=ref.transform,
                  dst_crs=ref.crs, resampling=Resampling.bilinear, src_nodata=srcs[0].nodata, dst_nodata=np.nan)
        prof = ref.profile | dict(dtype="float32", nodata=-9999.0, compress="deflate", predictor=1)
    dst = np.where(np.isfinite(dst), dst, -9999.0)
    with rasterio.open(target, "w", **prof) as d:
        d.write(dst.astype(np.float32), 1)
    return target


def _wbt(tool, wd, **kw):
    """Run a WhiteboxTools tool through the binary and fail loudly (the Python wrapper only returns a code)."""
    import subprocess
    import whitebox
    exe = Path(whitebox.__file__).parent / "WBT" / "whitebox_tools"
    args = [str(exe), f"--wd={wd}", f"--run={tool}"]
    for k, v in kw.items():
        args.append(f"--{k}" if v is True else f"--{k}={v}")
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0 or "panicked" in r.stderr + r.stdout:
        raise RuntimeError(f"{tool} failed: {(r.stderr or r.stdout)[-800:]}")


def terrain_rasters():
    out = OUT / "terrain"
    dem = dem_mosaic()
    p = {k: out / f"{k}.tif" for k in ("breached", "d8", "acc", "streams", "hand", "slope", "sca", "twi")}
    steps = [("breached", "BreachDepressionsLeastCost", dict(dem=dem.name, output="breached.tif", dist=100, fill=True)),
             ("d8", "D8Pointer", dict(dem="breached.tif", output="d8.tif")),
             ("acc", "D8FlowAccumulation", dict(input="breached.tif", output="acc.tif", out_type="cells")),
             ("streams", "ExtractStreams", dict(flow_accum="acc.tif", output="streams.tif", threshold=1111)),
             ("hand", "ElevationAboveStream", dict(dem="breached.tif", streams="streams.tif", output="hand.tif")),
             ("slope", "Slope", dict(dem=dem.name, output="slope.tif", units="degrees")),
             ("sca", "D8FlowAccumulation", dict(input="breached.tif", output="sca.tif",
                                                out_type="specific contributing area")),
             ("twi", "WetnessIndex", dict(sca="sca.tif", slope="slope.tif", output="twi.tif"))]
    for key, tool, kw in steps:                       # 1111 cells of 30 m = 1 km2 stream initiation
        if not p[key].exists():
            print("whitebox", tool, flush=True)
            _wbt(tool, out, **kw)
    return dem, {k: str(v) for k, v in p.items()}


def terrain():
    out = OUT / "terrain"
    dem, p = terrain_rasters()
    f = _fields()
    d = _zonal(str(dem), f, ["mean", "min", "max"]).rename(columns={"mean": "elev_mean_m", "min": "elev_min_m",
                                                                     "max": "elev_max_m"})
    for k, col in (("slope", "slope_mean_deg"), ("hand", "hand_mean_m"), ("twi", "twi_mean")):
        d[col] = _zonal(p[k], f, ["mean"])["mean"].values
    d["hand_min_m"] = _zonal(p["hand"], f, ["min"])["min"].values
    d.to_parquet(out / "terrain_by_field.parquet", index=False)
    return d


def wetlands():
    cat = _cat()
    f = _fields()
    items = sorted(cat.search(collections=["noaa-c-cap"], bbox=list(f.to_crs(4326).total_bounds)).items(),
                   key=lambda i: i.properties["start_datetime"])     # items carry start/end, not datetime
    it = items[-1]
    href = it.assets["data"].href
    ring = f.copy()
    ring["geometry"] = f.buffer(500).difference(f.geometry)
    with rasterio.open(href) as s:
        fr = _zonal(s, f.to_crs(s.crs), ["frac"])
        rr = _zonal(s, ring.to_crs(s.crs), ["frac"])

    def wet(df):
        cols = [c for c in df.columns if c.startswith("frac_")]
        return df[[c for c in cols if int(c.split("_")[1]) in CCAP_WET]].sum(axis=1)
    d = pd.DataFrame({"fid": f.fid.values, "wetland_frac": wet(fr).values, "wetland_frac_500m": wet(rr).values})
    d["ccap_year"] = int(it.properties["start_datetime"][:4])
    (OUT / "wetlands").mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "wetlands" / "wetlands_by_field.parquet", index=False)
    return d


if __name__ == "__main__":
    import sys
    for step in sys.argv[1:] or ["soil", "terrain", "wetlands"]:
        print(step, len({"soil": soil, "terrain": terrain, "wetlands": wetlands}[step]()))
