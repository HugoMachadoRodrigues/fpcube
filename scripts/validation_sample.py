"""Stratified sample of points for checking the fields against high-resolution imagery (e.g. Google Earth, NAIP).

Writes data/release/<v>/validation/photo_sample.csv and .kml with empty columns to fill by an interpreter:

  commission (inside fields), stratified by tier x region x field size, N_PER_STRATUM fields per stratum, one random
  point in the field's largest part:
      herbaceous   y / n            herbaceous cover at the point
      pasture      y / n / unsure   managed pasture or hay (fences, water points, cattle, hay bales, regular mowing)
      boundary     ok / partial / wrong   field outline compared with the visible land use
  omission (outside fields), stratified by why the pixel is not in a field:
      cdl_rare     CDL pasture in 1-5 of the 10 years
      screened     CDL pasture in >= 6 years but removed (woody screen, outside parcels, part < 2 ha)
      other_land   any other land pixel of Florida (estimates pasture never mapped as such by CDL)
      pasture      y / n / unsure

Strata weights (field counts or pixel counts) are in the CSV, so accuracies can be estimated with stratified
estimators. python scripts/validation_sample.py [--release v1.0]
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from fpcube import config

REGION = config.REGION

SEED = 20261120
N_PER_STRATUM = 10
N_OMISSION = {"cdl_rare": 150, "screened": 150, "other_land": 300}
SIZES = [2, 5, 20, 100, np.inf]


def commission(fields, rng):
    f = fields.assign(region=fields.county.map(REGION).fillna("Other"),
                      size=pd.cut(fields.area_ha, SIZES, right=False, labels=["2-5", "5-20", "20-100", ">=100"]))
    rows = []
    for (tier, reg, sz), g in f.groupby(["tier", "region", "size"], observed=True):
        pick = g.sample(min(N_PER_STRATUM, len(g)), random_state=int(rng.integers(1 << 31)))
        for r in pick.itertuples():
            geom = r.geometry
            if geom.geom_type == "MultiPolygon":
                geom = max(geom.geoms, key=lambda p: p.area)
            pt = geom.representative_point() if geom.area < 2e4 else _random_point(geom, rng)
            rows.append({"set": "commission", "stratum": f"{tier}|{reg}|{sz}", "stratum_size": len(g),
                         "field_id": r.field_id, "x": pt.x, "y": pt.y})
    return rows


def _random_point(poly, rng):
    from shapely.geometry import Point
    x0, y0, x1, y1 = poly.bounds
    for _ in range(1000):
        p = Point(rng.uniform(x0, x1), rng.uniform(y0, y1))
        if poly.contains(p):
            return p
    return poly.representative_point()


def omission(rng):
    fd = config.FIELDS_DIR
    with rasterio.open(fd / "field_labels.tif") as s:
        lab, tr, crs = s.read(1), s.transform, s.crs
    with rasterio.open(fd / "pasture_frequency.tif") as s:
        freq = s.read(1)
    with rasterio.open(config.RAW / "cdl" / f"CDL_2025_{config.FLORIDA_FIPS}.tif") as s:
        cdl = s.read(1)
    land = (cdl > 0) & ~np.isin(cdl, config.CDL_WATER)
    out = lab == 0
    need = int(np.ceil(config.PASTURE_MIN_FRACTION * len(config.PASTURE_YEARS)))
    strata = {"cdl_rare": out & (freq >= 1) & (freq < need), "screened": out & (freq >= need),
              "other_land": out & land & (freq == 0)}
    rows = []
    for name, m in strata.items():
        idx = np.flatnonzero(m)
        pick = rng.choice(idx, N_OMISSION[name], replace=False)
        r, c = np.unravel_index(pick, m.shape)
        xs, ys = rasterio.transform.xy(tr, r, c)
        rows += [{"set": "omission", "stratum": name, "stratum_size": int(m.sum()), "field_id": "", "x": x, "y": y}
                 for x, y in zip(xs, ys)]
    return rows, crs


def main(version):
    rel = config.RELEASE / version
    rng = np.random.default_rng(SEED)
    fields = gpd.read_parquet(rel / "fields.parquet", columns=["field_id", "tier", "county", "area_ha", "geometry"])
    rows = commission(fields, rng)
    om, crs = omission(rng)
    d = pd.DataFrame(rows + om)
    pts = gpd.GeoSeries(gpd.points_from_xy(d.x, d.y), crs=fields.crs).to_crs(4326)
    d["lon"], d["lat"] = pts.x.round(6), pts.y.round(6)
    d = d.drop(columns=["x", "y"]).sample(frac=1, random_state=SEED).reset_index(drop=True)   # blind order
    d.insert(0, "point_id", [f"P{i:04d}" for i in range(len(d))])
    for c in ("image_date", "herbaceous", "pasture", "boundary", "notes"):
        d[c] = ""
    out = rel / "validation"
    d.to_csv(out / "photo_sample.csv", index=False)
    g = gpd.GeoDataFrame(d[["point_id"]], geometry=gpd.points_from_xy(d.lon, d.lat), crs=4326)
    g.rename(columns={"point_id": "Name"}).to_file(out / "photo_sample.kml", driver="KML")
    print(d.groupby(["set", "stratum"]).size().to_string(), f"\n{len(d)} points ->", out / "photo_sample.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="v1.0")
    main(ap.parse_args().release)
