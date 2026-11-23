"""Field delineation: persistent pasture (CDL) x grazing-land parcels (FDOR) -> field polygons.

A field is the part of a 4-connected patch of persistent-pasture pixels that lies inside one parcel, at least
MIN_FIELD_HA in area. Patches are labelled before the split by parcel, so when a patch crosses a parcel boundary
the part inside one parcel can itself be disconnected (connected only through the neighbouring parcel); such fields
have n_parts > 1 in v1.0. From v1.1 (python -m fpcube.fields v1.1) a field is one 4-connected part inside one
parcel, and woody wetland (CDL 190) joins the woody screen. Persistent pasture: CDL Grass/Pasture or Other Hay in at
least PASTURE_MIN_FRACTION of the PASTURE_YEARS; pixels woody in more than WOODY_MAX_YEARS of the WOODY_SCREEN_YEARS
are removed (young pine stands are often mapped as pasture).

Field identifiers stay stable across versions (lineage): a later version's field that has exactly the pixels of a
v1.0 field keeps its id and reuses its observations; when a v1.0 field changes, its largest remaining part keeps the
id and is re-extracted, and further parts get new ids above the v1.0 maximum. lineage.parquet records parent and
status for every field; fields_extract.parquet holds the fields to extract.

Outputs (config.INTERIM / "fields"):
  field_labels.tif            field id per 30 m CDL pixel (0 = none), EPSG:5070
  persistent_pasture.tif      uint8: 1 = persistent pasture kept, 2 = removed by the woody screen
  pasture_frequency.tif       uint8: number of PASTURE_YEARS mapped as pasture
  fields.parquet              GeoParquet of field polygons with attributes
  delineation_stats.json      area accounting for the paper
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage

from fpcube import config

COUNTIES = {  # FDOR county numbers -> names
    11: "Alachua", 12: "Baker", 13: "Bay", 14: "Bradford", 15: "Brevard", 16: "Broward", 17: "Calhoun",
    18: "Charlotte", 19: "Citrus", 20: "Clay", 21: "Collier", 22: "Columbia", 23: "Miami-Dade", 24: "DeSoto",
    25: "Dixie", 26: "Duval", 27: "Escambia", 28: "Flagler", 29: "Franklin", 30: "Gadsden", 31: "Gilchrist",
    32: "Glades", 33: "Gulf", 34: "Hamilton", 35: "Hardee", 36: "Hendry", 37: "Hernando", 38: "Highlands",
    39: "Hillsborough", 40: "Holmes", 41: "Indian River", 42: "Jackson", 43: "Jefferson", 44: "Lafayette",
    45: "Lake", 46: "Lee", 47: "Leon", 48: "Levy", 49: "Liberty", 50: "Madison", 51: "Manatee", 52: "Marion",
    53: "Martin", 54: "Monroe", 55: "Nassau", 56: "Okaloosa", 57: "Okeechobee", 58: "Orange", 59: "Osceola",
    60: "Palm Beach", 61: "Pasco", 62: "Pinellas", 63: "Polk", 64: "Putnam", 65: "St. Johns", 66: "St. Lucie",
    67: "Santa Rosa", 68: "Sarasota", 69: "Seminole", 70: "Sumter", 71: "Suwannee", 72: "Taylor", 73: "Union",
    74: "Volusia", 75: "Wakulla", 76: "Walton", 77: "Washington",
}


def cdl_path(year):
    return config.RAW / "cdl" / f"CDL_{year}_{config.FLORIDA_FIPS}.tif"


def _check_grids(years):
    ref = None
    for y in years:
        with rasterio.open(cdl_path(y)) as s:
            g = (s.crs.to_epsg(), s.transform, s.width, s.height)
        if ref is None:
            ref = g
        elif g != ref:
            raise SystemExit(f"CDL {y} grid differs from {years[0]}: {g} vs {ref}")
    return ref


def _read(year):
    with rasterio.open(cdl_path(year)) as s:
        return s.read(1)


FIPS = {"Alachua": "12001", "Baker": "12003", "Bay": "12005", "Bradford": "12007", "Brevard": "12009",
        "Broward": "12011", "Calhoun": "12013", "Charlotte": "12015", "Citrus": "12017", "Clay": "12019",
        "Collier": "12021", "Columbia": "12023", "DeSoto": "12027", "Dixie": "12029", "Duval": "12031",
        "Escambia": "12033", "Flagler": "12035", "Franklin": "12037", "Gadsden": "12039", "Gilchrist": "12041",
        "Glades": "12043", "Gulf": "12045", "Hamilton": "12047", "Hardee": "12049", "Hendry": "12051",
        "Hernando": "12053", "Highlands": "12055", "Hillsborough": "12057", "Holmes": "12059", "Indian River":
        "12061", "Jackson": "12063", "Jefferson": "12065", "Lafayette": "12067", "Lake": "12069", "Lee":
        "12071", "Leon": "12073", "Levy": "12075", "Liberty": "12077", "Madison": "12079", "Manatee": "12081",
        "Marion": "12083", "Martin": "12085", "Miami-Dade": "12086", "Monroe": "12087", "Nassau": "12089",
        "Okaloosa": "12091", "Okeechobee": "12093", "Orange": "12095", "Osceola": "12097", "Palm Beach":
        "12099", "Pasco": "12101", "Pinellas": "12103", "Polk": "12105", "Putnam": "12107", "St. Johns":
        "12109", "St. Lucie": "12111", "Santa Rosa": "12113", "Sarasota": "12115", "Seminole": "12117",
        "Sumter": "12119", "Suwannee": "12121", "Taylor": "12123", "Union": "12125", "Volusia": "12127",
        "Wakulla": "12129", "Walton": "12131", "Washington": "12133"}  # US Census FIPS, Florida


def assign_county(gdf, fips=FIPS):
    """County holding the largest share of each field's area (US Census 2023 cartographic boundaries,
    raw/boundaries/fl_counties.parquet). The tax-roll county (FDOR CO_NO) is kept only where the boundary file
    gives none: in 76 fields the parcel's roll county holds less than half of the field (22 lie wholly outside it),
    which would put them in the wrong county for county-level labels."""
    p = config.RAW / "boundaries" / "fl_counties.parquet"
    if not p.exists():
        return gdf
    co = gpd.read_parquet(p).to_crs(gdf.crs)[["GEOID", "geometry"]]
    ov = gpd.overlay(gdf[["fid", "geometry"]], co, how="intersection", keep_geom_type=True)
    ov["a"] = ov.geometry.area
    best = ov.sort_values("a").drop_duplicates("fid", keep="last").set_index("fid").GEOID
    name = {v: k for k, v in fips.items()}
    g = gdf.fid.map(best)
    has = g.notna()
    gdf.loc[has, "county_fips"] = g[has].values
    gdf.loc[has, "county"] = g[has].map(name).values
    return gdf


def part_attributes(gdf):
    """Number of 4-connected parts of each field polygon and the share of its area in the largest part."""
    def one(g):
        if g.geom_type != "MultiPolygon":
            return 1, 1.0
        a = [p.area for p in g.geoms]
        return len(a), max(a) / sum(a)
    v = gdf.geometry.apply(one)
    gdf["n_parts"] = v.map(lambda x: x[0]).astype("int16")
    gdf["largest_part_frac"] = v.map(lambda x: round(x[1], 4)).astype("float32")
    return gdf


def add_part_attributes(paths):
    """Add n_parts and largest_part_frac to existing fields tables (v1.0 was built before these columns)."""
    import os
    import shutil
    import tempfile
    for p in map(Path, paths):
        g = gpd.read_parquet(p)
        if "n_parts" in g:
            continue
        g = part_attributes(g)
        cols = list(g.columns)
        i = cols.index("lat") + 1
        cols = cols[:i] + ["n_parts", "largest_part_frac"] + [c for c in cols[i:] if c not in ("n_parts", "largest_part_frac")]
        tmp = Path(tempfile.mkdtemp(dir="/private/tmp")) / p.name
        g[cols].to_parquet(tmp)
        shutil.move(str(tmp), str(p))
        print(p, int((g.n_parts > 1).sum()), "multipart fields", flush=True)


def connected_within_parcel(cand, pidx):
    """Label 4-connected groups of candidate pixels that share a parcel (a field never crosses a parcel boundary
    and never consists of disconnected pieces)."""
    from skimage.measure import label
    key = np.where(cand, pidx, 0)
    return label(key, background=0, connectivity=1).astype(np.int64)


def stable_ids(lab, base_labels_path):
    """Renumber a new label raster so that fields keep their v1.0 id where possible. Returns the relabelled raster
    and the lineage table (fid, parent_fid, status)."""
    with rasterio.open(base_labels_path) as s:
        old = s.read(1)
    m = lab > 0
    new_ids, old_ids = lab[m], old[m].astype(np.int64)
    n_new = int(lab.max())
    size_new = np.bincount(new_ids, minlength=n_new + 1)
    size_old = np.bincount(old.ravel(), minlength=int(old.max()) + 1)
    pair = pd.DataFrame({"new": new_ids, "old": old_ids}).value_counts().rename("n").reset_index()
    parent = pair.sort_values("n").drop_duplicates("new", keep="last").set_index("new")
    lin = pd.DataFrame({"new": np.arange(1, n_new + 1)})
    lin = lin[size_new[1:] > 0]
    lin["parent_fid"] = parent.old.reindex(lin.new).fillna(0).astype(np.int64).values
    lin["n_pixels"] = size_new[lin.new]
    lin["same"] = (lin.parent_fid > 0) & (size_new[lin.new] == size_old[lin.parent_fid]) & \
        (parent.n.reindex(lin.new).values == size_new[lin.new])
    lin = lin.sort_values(["parent_fid", "n_pixels"], ascending=[True, False])
    lin["rank"] = lin.groupby("parent_fid").cumcount()
    keep_old = (lin.parent_fid > 0) & (lin["rank"] == 0)
    lin["fid"] = np.where(keep_old, lin.parent_fid, 0)
    nxt = int(old.max()) + 1
    k = ~keep_old
    lin.loc[k, "fid"] = np.arange(nxt, nxt + int(k.sum()))
    lin["status"] = np.where(lin.same, "unchanged", np.where(keep_old, "reshaped", "new"))
    remap = np.zeros(n_new + 1, np.int64)
    remap[lin.new.to_numpy()] = lin.fid.to_numpy()
    present = np.nonzero(size_old)[0]
    removed = np.setdiff1d(present[present > 0], lin.parent_fid.to_numpy())
    out = lin[["fid", "parent_fid", "status", "n_pixels"]].sort_values("fid").reset_index(drop=True)
    gone = pd.DataFrame({"fid": removed, "parent_fid": removed, "status": "removed", "n_pixels": 0})
    return remap[lab], pd.concat([out, gone], ignore_index=True)


def build(out_dir=None, version="v1.0"):
    """v1.0: fields as released in v1.0. v1.1: one connected part per field, woody wetland screened, ids kept
    stable against v1.0 (see the module docstring)."""
    v11 = version != "v1.0"
    out = Path(out_dir or config.vdir("fields", version))
    out.mkdir(parents=True, exist_ok=True)
    years = config.PASTURE_YEARS
    epsg, transform, width, height = _check_grids(sorted(set(years + config.WOODY_SCREEN_YEARS)))
    px_ha = abs(transform.a * transform.e) / 1e4

    freq = np.zeros((height, width), np.uint8)
    annual_pasture = {}
    for y in years:
        a = _read(y)
        p = np.isin(a, config.CDL_PASTURE)
        freq += p
        annual_pasture[y] = p
        print("cdl", y, flush=True)
    woody = np.zeros((height, width), np.uint8)
    for y in config.WOODY_SCREEN_YEARS:
        woody += np.isin(_read(y), config.CDL_WOODY + (config.CDL_WETLAND_WOODY if v11 else ()))
    persistent = freq >= int(np.ceil(config.PASTURE_MIN_FRACTION * len(years)))
    removed_woody = persistent & (woody > config.WOODY_MAX_YEARS)
    keep = persistent & ~removed_woody

    prof = dict(driver="GTiff", width=width, height=height, count=1, crs=f"EPSG:{epsg}", transform=transform,
                compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(out / "pasture_frequency.tif", "w", dtype="uint8", **prof) as d:
        d.write(freq, 1)
    pp = np.where(keep, 1, np.where(removed_woody, 2, 0)).astype(np.uint8)
    with rasterio.open(out / "persistent_pasture.tif", "w", dtype="uint8", nodata=0, **prof) as d:
        d.write(pp, 1)

    pdir = config.RAW / "parcels"
    parts = [gpd.read_parquet(pdir / "fl_grazing_parcels_2025.parquet")]
    if (pdir / "fl_parcels_near_pasture_2025.parquet").exists():
        parts.append(gpd.read_parquet(pdir / "fl_parcels_near_pasture_2025.parquet"))
    parcels = pd.concat(parts, ignore_index=True).drop_duplicates("OBJECTID")
    parcels = gpd.GeoDataFrame(parcels, geometry="geometry", crs=parts[0].crs).to_crs(f"EPSG:{epsg}")
    parcels = parcels[parcels.geometry.notna() & ~parcels.geometry.is_empty]
    # rasterize grazing-coded parcels last so that they win where parcel layers overlap
    parcels["is_grazing"] = parcels.DOR_UC.isin(config.DOR_GRAZING)
    parcels = parcels.sort_values(["is_grazing", "OBJECTID"]).reset_index(drop=True)
    parcels["pidx"] = np.arange(1, len(parcels) + 1, dtype=np.int32)
    pidx = features.rasterize(zip(parcels.geometry, parcels.pidx), out_shape=(height, width), transform=transform,
                              fill=0, dtype="int32")
    print("parcels rasterized", flush=True)

    cand = keep & (pidx > 0)
    if v11:
        lab = connected_within_parcel(cand, pidx)
    else:
        comp, ncomp = ndimage.label(cand, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        # split components that span several parcels: unique (component, parcel) pairs
        key = comp.astype(np.int64) * (len(parcels) + 1) + pidx
        key[~cand] = 0
        uniq, inv = np.unique(key, return_inverse=True)
        lab = inv.reshape(key.shape).astype(np.int64)  # 0 stays 0 because uniq[0] == 0
        del key, inv
    counts = np.bincount(lab.ravel())
    min_px = int(np.ceil(config.MIN_FIELD_HA / px_ha))
    small = counts < min_px
    small[0] = True
    lab[small[lab]] = 0
    # renumber 1..N in raster order
    kept = np.flatnonzero(~small)
    remap = np.zeros(len(counts), np.int32)
    remap[kept] = np.arange(1, len(kept) + 1, dtype=np.int32)
    lab = remap[lab]
    nfield = len(kept)
    print("fields", nfield, flush=True)
    lineage = None
    if v11:
        lab, lineage = stable_ids(lab, config.vdir("fields", "v1.0") / "field_labels.tif")
        nfield = int(lab.max())                        # ids are no longer contiguous
        print(lineage.status.value_counts().to_dict(), flush=True)
    with rasterio.open(out / "field_labels.tif", "w", dtype="int32", nodata=0, **prof) as d:
        d.write(lab, 1)

    # vectorize
    geoms, ids = [], []
    for g, v in features.shapes(lab, mask=lab > 0, transform=transform, connectivity=4):
        geoms.append(g)
        ids.append(int(v))
    from shapely.geometry import shape
    gdf = gpd.GeoDataFrame({"fid": ids}, geometry=[shape(g) for g in geoms], crs=f"EPSG:{epsg}")
    gdf = gdf.dissolve("fid").reset_index()
    gdf = part_attributes(gdf)

    # attributes from rasters
    n_px = np.bincount(lab.ravel(), minlength=nfield + 1)
    parcel_of = np.zeros(nfield + 1, np.int64)
    parcel_of[lab[lab > 0]] = pidx[lab > 0]
    freq_mean = np.bincount(lab.ravel(), weights=freq.ravel(), minlength=nfield + 1) / np.maximum(n_px, 1)
    annual = {f"pasture_frac_{y}": np.bincount(lab.ravel(), weights=annual_pasture[y].ravel(),
                                                minlength=nfield + 1) / np.maximum(n_px, 1) for y in years}
    interior = ndimage.binary_erosion(lab > 0, structure=np.ones((3, 3)))
    same = ndimage.minimum_filter(lab, size=3) == ndimage.maximum_filter(lab, size=3)
    n_int = np.bincount(lab[interior & same], minlength=nfield + 1)

    a = pd.DataFrame({"fid": np.arange(nfield + 1), "n_pixels": n_px, "n_interior_pixels": n_int,
                      "pasture_years_mean": freq_mean, "pidx": parcel_of, **annual}).iloc[1:]
    gdf = gdf.merge(a, on="fid").merge(parcels[["pidx", "CO_NO", "DOR_UC", "AgTaxExempt"]], on="pidx", how="left")
    gdf["county"] = gdf.CO_NO.map(lambda c: COUNTIES.get(int(c)) if pd.notna(c) else None)
    gdf["county_fips"] = gdf.county.map(FIPS)
    gdf = assign_county(gdf)
    gdf["area_ha"] = gdf.n_pixels * px_ha
    gdf["perimeter_m"] = gdf.length
    gdf["compactness"] = 4 * np.pi * gdf.area / gdf.length ** 2
    cen = gdf.geometry.representative_point().to_crs(4326)
    gdf["lon"], gdf["lat"] = cen.x.round(6), cen.y.round(6)
    gdf["field_id"] = gdf.fid.map(lambda i: f"FPC{i:06d}")
    gdf["tier"] = np.where(gdf.DOR_UC.isin(config.DOR_GRAZING), "A", "B")
    gdf["flag_few_interior"] = gdf.n_interior_pixels < 5
    gdf["flag_landuse_change"] = gdf[[f"pasture_frac_{y}" for y in years[-3:]]].mean(axis=1) < 0.5
    gdf = gdf.rename(columns={"DOR_UC": "dor_use_code", "AgTaxExempt": "ag_exempt"}).drop(columns=["pidx", "CO_NO"])
    cols = ["field_id", "fid", "tier", "county", "county_fips", "dor_use_code", "ag_exempt", "area_ha", "perimeter_m", "compactness",
            "n_pixels", "n_interior_pixels", "pasture_years_mean", "lon", "lat", "n_parts", "largest_part_frac",
            "flag_few_interior", "flag_landuse_change"] + list(annual) + ["geometry"]
    gdf = gdf[cols].sort_values("fid")
    gdf.to_parquet(out / "fields.parquet")
    if lineage is not None:
        lineage.to_parquet(out / "lineage.parquet", index=False)
        ext = lineage[lineage.status.isin(["reshaped", "new"])].fid
        gdf[gdf.fid.isin(ext)].to_parquet(out / "fields_extract.parquet")

    stats = {
        "cdl_years": years, "woody_screen_years": config.WOODY_SCREEN_YEARS,
        "pixel_ha": px_ha,
        "persistent_pasture_ha": float(persistent.sum() * px_ha),
        "removed_by_woody_screen_ha": float(removed_woody.sum() * px_ha),
        "kept_pasture_ha": float(keep.sum() * px_ha),
        "kept_in_grazing_parcels_ha": float(cand.sum() * px_ha),
        "in_fields_ha": float((lab > 0).sum() * px_ha),
        "n_parcels": int(len(parcels)),
        "n_grazing_parcels": int(parcels.is_grazing.sum()),
        "n_fields": int(len(gdf)),
        "version": version,
        **({"lineage": lineage.status.value_counts().to_dict()} if lineage is not None else {}),
        "n_fields_tier_A": int((gdf.tier == "A").sum()),
        "area_ha_tier_A": float(gdf.loc[gdf.tier == "A", "area_ha"].sum()),
        "area_ha_tier_B": float(gdf.loc[gdf.tier == "B", "area_ha"].sum()),
        "field_area_ha_quantiles": gdf.area_ha.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(1).to_dict(),
        "any_year_pasture_ha": float((freq > 0).sum() * px_ha),
    }
    (out / "delineation_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return gdf


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--add-part-attributes"]:
        add_part_attributes(sys.argv[2:])
    else:
        build(version=(sys.argv[1:2] or ["v1.0"])[0])
