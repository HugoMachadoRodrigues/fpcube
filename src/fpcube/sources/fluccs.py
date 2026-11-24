"""Official Florida land use and land cover (FLUCCS level 3) of the five water management districts, the independent
reference for the field delineation.

Source: University of Florida GeoPlan Center, Florida Geographic Data Library (FGDL), "Statewide Land Use FLUCCS",
the update of LU_L3_STATE_JUL23 merging LU_NWFWMD_2022, LU_SRWMD_2020, LU_SWFWMD_2023, LU_SFWMD_2023 and
LU_SJRWMD_2022 (photo-interpreted from aerial imagery of 2019-2023; FLUCCS, Florida DOT 1999), served as a public
ArcGIS feature service. The polygons are downloaded in pages, kept as GeoParquet in RAW/fluccs and rasterized on the
30 m grid of the fields (pixel centres) as fluccs_l3.tif (uint16 FLUCCS level-3 code, 0 = no data) and
fluccs_source.tif (uint8 district, see SOURCES).

  python -m fpcube.sources.fluccs            # download (resumable) and rasterize
"""
import json
import shutil
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import rasterize
from shapely.geometry import MultiPolygon, Polygon

from fpcube import config

SERVICE = ("https://services.arcgis.com/rD2ylXRs80UroD90/ArcGIS/rest/services/Statewide_Land_Use_FLUCCS/"
           "FeatureServer/24")
OUT = config.RAW / "fluccs"
PAGE = 2000
TMP = Path("/private/tmp/fpcube_fluccs")
TMP.mkdir(exist_ok=True)
SOURCES = {"NWFWMD": 1, "SRWMD": 2, "SJRWMD": 3, "SWFWMD": 4, "SFWMD": 5}
VINTAGE = {"NWFWMD": "2022", "SRWMD": "2019-2020", "SJRWMD": "2020-2022", "SWFWMD": "2023", "SFWMD": "2021-2023"}

# reference classes (FLUCCS level 3, every code from 1000 to 8999 in exactly one class); 2100 is "cropland and
# pastureland", the only pasture class SWFWMD uses
CLASSES = {
    "pasture": list(range(2100, 2140)),                              # 2100, 2110 improved, 2120 unimproved, 2130 woodland
    "crops": list(range(2000, 2100)) + list(range(2140, 2300)) + list(range(2400, 2500)),  # agriculture n.e.c.; row, field and mixed crops; groves; nurseries
    "specialty_farms": list(range(2300, 2400)) + list(range(2500, 2600)),   # feeding operations, horse farms, dairies
    "rural_open": list(range(2600, 3000)),                           # other open rural land (2610 fallow cropland)
    "rangeland": list(range(3000, 4000)),
    "forest": list(range(4000, 5000)),
    "wetland": list(range(6000, 7000)),
    "rural_residential": list(range(1100, 1200)),                    # low-density and rural residential
    "mining_reclaimed": list(range(1600, 1700)),                     # extractive and reclaimed (phosphate) land
    "urban": [c for c in range(1000, 2000) if not 1100 <= c < 1200 and not 1600 <= c < 1700] + list(range(8000, 9000)),
    "water": list(range(5000, 6000)),
    "barren": list(range(7000, 8000)),
}


def classify(code):
    """FLUCCS level-3 code array -> class index (order of CLASSES, 0-based; -1 = no data or unlisted)."""
    out = np.full(code.shape, -1, np.int8)
    for i, cs in enumerate(CLASSES.values()):
        out[np.isin(code, cs)] = i
    return out


def _get(params, tries=6):
    for k in range(tries):
        try:
            r = requests.get(SERVICE + "/query", params=params, timeout=120)
            r.raise_for_status()
            d = r.json()
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def _rings_to_geom(rings):
    """Esri polygon rings -> shapely (outer rings clockwise, holes counter-clockwise)."""
    polys, cur = [], None
    for ring in rings:
        if len(ring) < 4:
            continue
        p = Polygon(ring)
        a = sum((x1 - x0) * (y1 + y0) for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:]))   # > 0: clockwise
        if a > 0 or cur is None:
            cur = [ring, []]
            polys.append(cur)
        else:
            cur[1].append(ring)
    ps = [Polygon(o, h) for o, h in polys]
    return ps[0] if len(ps) == 1 else MultiPolygon(ps)


def _page(lo, hi):
    p = OUT / "pages" / f"{lo:08d}.parquet"
    if p.exists():
        return p
    d = _get({"where": f"OBJECTID>={lo} AND OBJECTID<{hi}", "outFields": "OBJECTID,FLUCCS_L3,SOURCE",
              "outSR": 5070, "geometryPrecision": 1, "f": "json"})
    rows = [(f["attributes"]["OBJECTID"], f["attributes"]["FLUCCS_L3"], f["attributes"]["SOURCE"],
             _rings_to_geom(f["geometry"]["rings"])) for f in d["features"] if f.get("geometry")]
    g = gpd.GeoDataFrame(pd.DataFrame(rows, columns=["objectid", "fluccs_l3", "source", "geometry"]),
                         geometry="geometry", crs=5070)
    tmp = TMP / p.name                          # write outside OneDrive, then move into place
    g.to_parquet(tmp)
    shutil.move(str(tmp), str(p))
    return p


def download(workers=8):
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    st = _get({"where": "1=1", "outStatistics": json.dumps([
        {"statisticType": "min", "onStatisticField": "OBJECTID", "outStatisticFieldName": "lo"},
        {"statisticType": "max", "onStatisticField": "OBJECTID", "outStatisticFieldName": "hi"}]), "f": "json"})
    a = st["features"][0]["attributes"]
    starts = list(range(a["lo"], a["hi"] + 1, PAGE))
    with ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(lambda s: _page(s, s + PAGE), starts)):
            if (k + 1) % 50 == 0:
                print(f"fluccs pages {k + 1}/{len(starts)}", flush=True)
    n = _get({"where": "1=1", "returnCountOnly": "true", "f": "json"})["count"]
    got = sum(len(pd.read_parquet(p, columns=["objectid"])) for p in (OUT / "pages").glob("*.parquet"))
    if n - got > 10:                            # features without geometry are skipped
        raise RuntimeError(f"downloaded {got} of {n} features")
    (OUT / "download.json").write_text(json.dumps({"service": SERVICE, "n_features": n, "n_without_geometry": n - got,
                                                   "retrieved": pd.Timestamp.now().strftime("%Y-%m-%d")}))
    print(f"fluccs: {n} features", flush=True)


def rasterize_grid():
    with rasterio.open(config.vdir("fields", "v1.1") / "field_labels.tif") as s:
        prof, shape, tr = s.profile, s.shape, s.transform
    code = np.zeros(shape, np.uint16)
    src = np.zeros(shape, np.uint8)
    for p in sorted((OUT / "pages").glob("*.parquet")):
        g = gpd.read_parquet(p)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        shapes = list(zip(g.geometry, g.fluccs_l3.fillna(0).astype(int)))
        if shapes:
            rasterize(shapes, out=code, transform=tr, dtype="uint16")
            rasterize(zip(g.geometry, g.source.map(SOURCES).fillna(0).astype(int)), out=src, transform=tr,
                      dtype="uint8")
    prof.update(count=1, compress="deflate", nodata=0)
    for name, arr in (("fluccs_l3.tif", code), ("fluccs_source.tif", src)):
        prof.update(dtype=arr.dtype.name)
        tmp = TMP / name
        with rasterio.open(tmp, "w", **prof) as d:
            d.write(arr, 1)
        shutil.move(str(tmp), str(OUT / name))
    print("rasterized", int((code > 0).sum()), "pixels with a FLUCCS code", flush=True)


if __name__ == "__main__":
    if "rasterize" not in sys.argv[1:]:
        download()
    rasterize_grid()
