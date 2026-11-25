"""Download Florida Department of Revenue grazing-land parcels (DOR use codes 060-065).

Source: FDACS GIS service 'GIO_Parcels_for_OAWP_Optimized' (statewide parcels exported from the FDOR
August 2025 submissions). Only parcel geometry, county number, parcel id, use code and agricultural
exemption flag are requested; owner names are never downloaded.
"""
import json
import time

import geopandas as gpd
import pandas as pd
import requests

from fpcube import config


def _query(params, retries=5):
    url = config.PARCEL_SERVICE + "/query"
    for i in range(retries):
        try:
            r = requests.post(url, data=params, timeout=300)
            r.raise_for_status()
            d = r.json()
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(5 * (i + 1))


def grazing_where():
    codes = ",".join(f"'{c}'" for c in config.DOR_GRAZING)
    return f"DOR_UC IN ({codes})"


def download(out=None, page=2000):
    out = out or config.RAW / "parcels" / "fl_grazing_parcels_2025.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ids = _query({"where": grazing_where(), "returnIdsOnly": "true", "f": "json"})["objectIds"]
    ids = sorted(ids)
    frames = []
    for k in range(0, len(ids), page):
        chunk = ids[k:k + page]
        d = _query({"objectIds": ",".join(map(str, chunk)), "outFields": config.PARCEL_FIELDS + ",OBJECTID",
                    "outSR": "5070", "f": "geojson", "returnGeometry": "true"})
        frames.append(gpd.GeoDataFrame.from_features(d["features"], crs=config.CRS))
        print(f"parcels {min(k + page, len(ids))}/{len(ids)}", flush=True)
    gdf = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=config.CRS)
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf.to_parquet(out)
    meta = {"source": config.PARCEL_SERVICE, "where": grazing_where(), "n": len(gdf),
            "fields": config.PARCEL_FIELDS, "downloaded": pd.Timestamp.utcnow().isoformat()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out


def download_all_near_pasture(cell_m=10000, min_ha=5.0, out=None, page=2000):
    """All parcels (any use code) that intersect persistent-pasture patches lying outside the grazing-coded
    parcels; patches are grouped by 10 km cell only to batch the spatial queries. Needs fpcube.fields to have written persistent_pasture.tif and field_labels.tif."""
    import numpy as np
    import rasterio
    from rasterio import features
    from concurrent.futures import ThreadPoolExecutor

    out = out or config.RAW / "parcels" / "fl_parcels_near_pasture_2025.parquet"
    fdir = config.INTERIM / "fields"
    with rasterio.open(fdir / "persistent_pasture.tif") as s:
        keep = s.read(1) == 1
        tr, crs = s.transform, s.crs
    graz = gpd.read_parquet(config.RAW / "parcels" / "fl_grazing_parcels_2025.parquet").to_crs(crs)
    gmask = features.rasterize(((g, 1) for g in graz.geometry), out_shape=keep.shape, transform=tr, fill=0,
                               dtype="uint8").astype(bool)
    rest = keep & ~gmask
    rule = ("all parcels (any use code) intersecting persistent-pasture patches of at least "
            f"{config.MIN_FIELD_HA} ha that lie outside grazing-coded parcels (patches buffered 15 m)")
    return download_intersecting(rest, tr, crs, out, rule, cell_m=cell_m, page=page)


def download_intersecting(rest, tr, crs, out, rule, cell_m=10000, page=2000):
    """All parcels (any use code) intersecting the 4-connected patches of mask `rest` that reach MIN_FIELD_HA
    (patches buffered 15 m, grouped by cell only to batch the spatial queries). Owner names are never requested."""
    import numpy as np
    from rasterio import features
    from concurrent.futures import ThreadPoolExecutor
    from scipy import ndimage
    from shapely.geometry import shape
    lab, n = ndimage.label(rest, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    size = np.bincount(lab.ravel())
    min_px = int(np.ceil(config.MIN_FIELD_HA / (abs(tr.a * tr.e) / 1e4)))
    big = size >= min_px
    big[0] = False
    lab = np.where(big[lab], lab, 0).astype(np.int32)
    polys = [shape(g).buffer(15).simplify(30) for g, v in features.shapes(lab, mask=lab > 0, transform=tr)]
    patches = gpd.GeoDataFrame(geometry=polys, crs=crs)
    patches["cx"] = (patches.centroid.x // cell_m).astype(int)
    patches["cy"] = (patches.centroid.y // cell_m).astype(int)
    groups = []
    for _, g in patches.groupby(["cx", "cy"]):
        geoms = list(g.geometry)
        for k in range(0, len(geoms), 40):
            groups.append(geoms[k:k + 40])
    print(f"{len(patches)} pasture patches outside grazing parcels, {len(groups)} queries", flush=True)

    def esri_polygon(geoms):
        rings = []
        for gg in geoms:
            for p in getattr(gg, "geoms", [gg]):
                rings.append([list(map(float, xy)) for xy in p.exterior.coords])
        return json.dumps({"rings": rings, "spatialReference": {"wkid": 5070}})

    def group_ids(geoms):
        d = _query({"geometry": esri_polygon(geoms), "geometryType": "esriGeometryPolygon", "inSR": "5070",
                    "spatialRel": "esriSpatialRelIntersects", "where": "1=1", "returnIdsOnly": "true", "f": "json"})
        return d.get("objectIds") or []

    cells = groups
    with ThreadPoolExecutor(6) as ex:
        ids = set()
        for k, lst in enumerate(ex.map(group_ids, groups)):
            ids.update(lst)
            if (k + 1) % 200 == 0:
                print(f"queries {k + 1}/{len(groups)}, parcels {len(ids)}", flush=True)
    ids = sorted(ids)
    print(f"{len(ids)} parcels to fetch", flush=True)

    def fetch(chunk):
        d = _query({"objectIds": ",".join(map(str, chunk)), "outFields": config.PARCEL_FIELDS + ",OBJECTID",
                    "outSR": "5070", "f": "geojson", "returnGeometry": "true"})
        return gpd.GeoDataFrame.from_features(d["features"], crs=config.CRS)

    chunks = [ids[k:k + page] for k in range(0, len(ids), page)]
    frames = []
    with ThreadPoolExecutor(4) as ex:
        for k, f in enumerate(ex.map(fetch, chunks)):
            frames.append(f)
            print(f"fetched {min((k + 1) * page, len(ids))}/{len(ids)}", flush=True)
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=config.CRS)
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf.to_parquet(out)
    meta = {"source": config.PARCEL_SERVICE, "rule": rule, "n": len(gdf), "n_cells": len(cells),
            "fields": config.PARCEL_FIELDS, "downloaded": pd.Timestamp.now("UTC").isoformat()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out


if __name__ == "__main__":
    import sys
    print(download_all_near_pasture() if "all" in sys.argv else download())
