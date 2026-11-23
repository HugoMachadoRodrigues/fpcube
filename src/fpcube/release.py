"""STAC catalog, checksums and datasheet for a release directory, and the Zenodo deposit built from it.

python -m fpcube.release v1.0             # DATASHEET.md, stac/, SHA256SUMS inside the release
python -m fpcube.release v1.0 --deposit   # data/deposit/v1.0: the files to upload to Zenodo

Zenodo stores a flat list of files, so each directory of the release becomes one zip (stored, not recompressed:
Parquet and Zarr chunks are already compressed). Unzipping them in one folder gives back the release layout;
cube.zarr.zip holds the store contents at its root, so it opens directly with zarr's ZipStore.
"""
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pystac

from fpcube import config

LICENSE = "CC-BY-4.0"
DESCRIPTION = ("Field-level, 5-day time series of persistent pasture fields in Florida, 2016-2026: HLS optical "
               "reflectance and indices, Sentinel-1 backscatter, PRISM weather, gNATSGO soil, 3DEP terrain (HAND, TWI), "
               "C-CAP wetlands, US Drought Monitor, NASS pasture condition and hurricane tracks, with benchmark tasks.")


def sha256(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def _files(rel, zarr=False):
    for p in sorted(rel.rglob("*")):
        if p.is_file() and (zarr or not any(part.endswith(".zarr") for part in p.relative_to(rel).parts)):
            yield p


def build_catalog(rel):
    fields = gpd.read_parquet(rel / "fields.parquet", columns=["fid", "geometry"])
    bbox = list(fields.to_crs(4326).total_bounds)
    geom = {"type": "Polygon", "coordinates": [[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]],
                                                [bbox[0], bbox[3]], [bbox[0], bbox[1]]]]}
    start, end = pd.Timestamp(config.START).to_pydatetime(), pd.Timestamp(config.END).to_pydatetime()
    cat = pystac.Catalog(id="florida-pasture-cube", description=DESCRIPTION, title="Florida Pasture Cube")
    col = pystac.Collection(
        id=f"florida-pasture-cube-{rel.name}", description=DESCRIPTION, license=LICENSE,
        extent=pystac.Extent(pystac.SpatialExtent([bbox]), pystac.TemporalExtent([[start, end]])),
        title=f"Florida Pasture Cube {rel.name}", keywords=["pasture", "grazing", "Florida", "HLS", "Sentinel-1",
                                                            "data cube", "machine learning", "benchmark"])
    cat.add_child(col)
    groups = {"fields": ["fields.parquet"] + (["lineage.parquet"] if (rel / "lineage.parquet").exists() else []), "cube": ["cube.zarr"],
              "optical-observations": sorted(p.name for p in (rel / "optical_observations").glob("*.parquet")),
              "radar-observations": sorted(p.name for p in (rel / "radar_observations").glob("*.parquet")),
              "labels": sorted(p.name for p in (rel / "labels").glob("*.parquet")),
              "benchmark": sorted(p.name for p in (rel / "benchmark").glob("*")) if (rel / "benchmark").exists() else []}
    sub = {"optical-observations": "optical_observations", "radar-observations": "radar_observations",
           "labels": "labels", "benchmark": "benchmark"}
    for gid, names in groups.items():
        if not names:
            continue
        it = pystac.Item(id=gid, geometry=geom, bbox=bbox, datetime=None, start_datetime=start, end_datetime=end,
                         properties={})
        for nm in names:
            p = rel / sub.get(gid, "") / nm
            if nm.endswith(".zarr"):
                mt, extra = "application/vnd+zarr", {}
            else:
                mt = {"parquet": "application/vnd.apache.parquet", "csv": "text/csv"}.get(nm.rsplit(".", 1)[-1],
                                                                                        "application/octet-stream")
                extra = {"file:checksum": "1220" + sha256(p), "file:size": p.stat().st_size}
            it.add_asset(nm, pystac.Asset(href=str(p.resolve()), media_type=mt, roles=["data"], extra_fields=extra))
        col.add_item(it)
    stac = rel / "stac"
    shutil.rmtree(stac, ignore_errors=True)
    cat.normalize_hrefs(str(stac))
    cat.make_all_asset_hrefs_relative()
    cat.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    check_catalog(stac / "catalog.json")
    return stac / "catalog.json"


def check_catalog(path):
    """Every asset href must resolve to an existing file or directory, and no href may be absolute."""
    bad = []
    for item_file in path.parent.rglob("*.json"):
        d = json.loads(item_file.read_text())
        for a in d.get("assets", {}).values():
            if Path(a["href"]).is_absolute() or not (item_file.parent / a["href"]).exists():
                bad.append(f"{item_file.relative_to(path.parent)}: {a['href']}")
    if bad:
        raise RuntimeError("broken STAC asset links: " + "; ".join(bad[:5]))


DATASHEET = """# Datasheet: Florida Pasture Cube {version}

Following Gebru et al. (2021), *Datasheets for datasets*.

## Motivation
- **Purpose.** Provide a consistent, field-level record of Florida pastures for studying responses to drought,
  extreme rain, freezes and hurricanes, and a benchmark for machine learning on irregular multi-sensor time series.
- **Creators.** Watershed Hydrology Lab, UF/IFAS Range Cattle Research and Education Center, Ona, Florida.

## Composition
- **Instances.** {n_fields} persistent pasture fields (tier A: {n_a}; tier B: {n_b}), {n_periods} five-day
  periods ({start} to {end}). The fields are a retrospective cohort: pasture persistent in the CDL of 2016-2025
  inside 2025 parcels, at least 2 ha. They do not cover every pasture in Florida (see the area accounting in
  `validation/delineation_accounting.csv`).
- **Per field:** geometry, county, parcel use code (no owner information), area and shape, CDL pasture fraction
  per year, quality flags, soil, terrain and wetland attributes.
- **Per field and period:** HLS reflectance and indices (with observation counts and gap-filling flag),
  Sentinel-1 VV/VH, PRISM weather.
- **Per field and acquisition:** long tables with clear-pixel counts, geometric shift, sensor, orbit; optical
  records carry quality flags (`qa_clear`, `qa_geometry`, `qa_range`, `qa_haze`) and `qa_use`, the records that
  enter the composites. `qa_geometry_status` is `pass`, `fail` or `not_assessed`: the shift is estimated over the
  window of clear field pixels when it is at least 64 x 64 pixels (`shift_scope` = `fields`), otherwise over a clear
  512 x 512 block of the tile (`scene`, used when the correlation peak is at least 0.1); {geom_na} of the records
  were not assessed (`none`) and are kept (`qa_geometry` is True for them). Indices are computed per pixel and averaged over the field's clear
  pixels; in v1.0 a pixel whose index is undefined (zero denominator) counts as 0 and values are clipped to
  [-1, 1] before averaging (rare: see the paper).
{multipart}{lineage}- **Labels:** US Drought Monitor (county-week), NASS pasture condition (state-week), hurricane tracks.
- **Missing data.** Optical gaps longer than 30 days are left empty on purpose. Radar coverage is thinner in
  2022-2024 after the Sentinel-1B failure. `water_table_min_cm` is empty where gNATSGO records no water table in
  the soil profile (mostly excessively and well-drained soils): read it as deeper than about 2 m, not as unknown.
  All {n_wt_missing} fields without a value have a matched map unit (no missing survey or failed join); the value
  is that of the field's majority map unit. `twi_mean` uses tan(slope) >= 0.001, so flat fields have a value.
  Optical composites: `n_optical_obs` = 0 and `gap_filled` = True mark interpolated values. **Do not use
  gap-filled values as inputs of a forecast**: the interpolation uses the next observation, which lies after the
  forecast date (the benchmark removes them, `fpcube.benchmark.tasks.observed_only`).
  NASS pasture condition is absent for weeks NASS did not publish (shutdowns of Dec 2018-Jan 2019 and
  Oct-Nov 2025; Dec 2025-Mar 2026). Landfalls of 2026 storms are derived from working best tracks (preliminary).
- **Sensitive data.** None. Parcel owner names were never downloaded.

## Collection and preprocessing
See the paper, Sections 2-4. All sources are public; the processing code (`fpcube`) rebuilds the release.
Code that produced the validation and benchmark files of this release: commit {commit}, code hash {code_hash}
(`fpcube.provenance`; the same hash means the same code). CDL 2024 and 2025 are the 30 m versions distributed
by CropScape (NASS also publishes 10 m versions from 2024).

## Reading the data
```python
import geopandas as gpd, xarray as xr
fields = gpd.read_parquet("fields.parquet")                 # EPSG:5070
cube = xr.open_zarr("cube.zarr")                             # or zarr.storage.ZipStore("cube.zarr.zip", mode="r")
sel = fields[(fields.tier == "A") & ~fields.flag_few_interior & (fields.n_parts == 1)]
ndvi = cube.ndvi.sel(field=sel.field_id.values)
observed = ndvi.where(cube.n_optical_obs.sel(field=sel.field_id.values) > 0)   # drop interpolated values
```

## Variable dictionary
{dictionary}

## Uses
- **Intended:** pasture monitoring research, hazard impact analysis, forecasting, weakly supervised learning.
- **Not intended:** decisions about individual properties, insurance or regulatory compliance without field
  verification. Field boundaries follow parcels and 30 m pixels, not fences.

## Source data and attribution
- **Contains modified Copernicus Sentinel data (2016-2026)**: Sentinel-1 backscatter, and Sentinel-2 through HLS S30.
- HLS v2.0 (S30, L30): NASA LP DAAC; Landsat data courtesy of the U.S. Geological Survey.
- PRISM daily data: PRISM Group, Oregon State University, https://prism.oregonstate.edu, accessed {prism_date}.
- Cropland Data Layer and pasture condition: USDA National Agricultural Statistics Service.
- Field geometry: Florida Department of Revenue statewide parcel data (public record; no owner information kept).
- gNATSGO: USDA Natural Resources Conservation Service. 3DEP DEM: U.S. Geological Survey.
- C-CAP 2016 land cover: NOAA Office for Coastal Management.
- U.S. Drought Monitor: jointly produced by the National Drought Mitigation Center at the University of
  Nebraska-Lincoln, the U.S. Department of Agriculture and the National Oceanic and Atmospheric Administration.
- Hurricane tracks: NOAA National Hurricane Center (HURDAT2 and working best tracks).
- ESA WorldCover 2021 (validation table only): (c) ESA WorldCover project 2021 / Contains modified Copernicus
  Sentinel data (2021) processed by ESA WorldCover consortium, CC BY 4.0.

## Distribution and licence
CC BY 4.0. DOI in the README. STAC catalog: `stac/catalog.json`. Checksums of every file, including the Zarr
store, in `SHA256SUMS`. On Zenodo each directory is one zip: unzip them in one folder to recover this layout
(`unzip cube.zarr.zip -d cube.zarr`, or open `cube.zarr.zip` directly with zarr's `ZipStore`).

## Maintenance
Annual updates planned (new season, new CDL and parcel vintages); versions are immutable and numbered.
"""


def prism_retrieved(rel):
    """Retrieval date of the PRISM grids, as recorded by validation (summary.json)."""
    def find(d):
        if isinstance(d, dict):
            if "prism_retrieved" in d:
                return d["prism_retrieved"]
            for v in d.values():
                if (r := find(v)):
                    return r
    got = find(json.loads((rel / "validation" / "summary.json").read_text()))
    if not got:
        raise RuntimeError("prism_retrieved missing from validation/summary.json: run validation first")
    return got


MULTIPART = """- **Multipart fields.** {n} fields have more than one 4-connected part (`n_parts`, `largest_part_frac`): a
  pasture patch crossing a parcel boundary can leave a disconnected part inside one parcel. Their observations
  aggregate over all parts.
"""
LINEAGE = """- **Field identifiers across versions.** `lineage.parquet` links every field id to the version 1.0 field it
  comes from (`parent_fid`) with `status`: `unchanged` (same pixels, same id), `reshaped` (the largest part of a
  1.0 field keeps its id), `new` (a further part of a 1.0 field, id above the 1.0 maximum) or `removed` (1.0 id
  no longer used). Observations of unchanged fields are those of version 1.0.
"""


def copy_lineage(rel):
    """Copy the field lineage of the release's field version (none for v1.0)."""
    src = config.vdir("fields", rel.name) / "lineage.parquet"
    if rel.name != "v1.0" and src.exists():
        shutil.copy2(src, rel / "lineage.parquet")


FIELD_COLUMNS = {
    "field_id": ("", "field identifier FPCnnnnnn; joins the cube's field coordinate", "none"),
    "fid": ("", "integer field id; joins the observation tables", "none"),
    "tier": ("", "A: parcel coded as grazing land (FDOR use codes 060-065); B: other parcel codes", "none"),
    "county, county_fips": ("", "county holding the largest share of the field area (US Census 2023)", "none"),
    "dor_use_code, ag_exempt": ("", "FDOR parcel use code and agricultural exemption flag of the parcel", "may be empty"),
    "area_ha, perimeter_m, compactness": ("ha, m, 1", "area from 30 m pixels; perimeter and 4 pi A / P^2 of the polygon", "none"),
    "n_pixels, n_interior_pixels": ("1", "30 m CDL pixels; pixels at least one pixel from the edge", "none"),
    "n_parts, largest_part_frac": ("1", "4-connected parts of the field; area share of the largest part", "none"),
    "pasture_years_mean, pasture_frac_YYYY": ("years, 1", "mean number of years mapped as pasture; share of the field "
                                              "mapped as pasture in CDL year YYYY", "none"),
    "lon, lat": ("degrees", "representative point (EPSG:4326)", "none"),
    "flag_few_interior": ("bool", "fewer than 5 interior pixels (mixed-pixel risk)", "none"),
    "flag_landuse_change": ("bool", "mean pasture fraction 2023-2025 below 0.5", "none"),
    "mukey, mukey_frac, musym, muname": ("", "gNATSGO majority map unit (10 m raster) and its area share", "none"),
    "aws0_100_mm, soc0_100_gm2": ("mm, g m-2", "area-weighted field mean of gNATSGO AWS and SOC, 0-100 cm", "rare"),
    "drainage_class/_code, flooding_frequency/_code, hydrologic_group": ("", "muaggatt dominant condition of the "
                                                                          "majority map unit", "rare"),
    "water_table_min_cm": ("cm", "muaggatt minimum annual depth to water table, majority map unit", "empty: no water "
                                                                                                   "table within the profile"),
    "elev_mean/min/max_m, slope_mean_deg": ("m, degrees", "3DEP 1 arc-second at 30 m, field mean / min / max", "none"),
    "hand_mean_m, hand_min_m": ("m", "height above nearest drainage (streams at 1 km2), a relative terrain "
                                     "position, field mean and min", "none"),
    "twi_mean": ("1", "ln(SCA / max(tan slope, 0.001)), field mean", "none"),
    "wetland_frac, wetland_frac_500m, ccap_year": ("1", "share of C-CAP wetland classes in the field / 500 m ring",
                                                   "none"),
    "n_optical_periods, n_radar_periods": ("1", "periods with an optical / radar composite", "none"),
    "flag_no_optical, flag_no_observations": ("bool", "no optical composite / no observation at all", "none"),
}


def dictionary(rel):
    import xarray as xr
    ds = xr.open_zarr(rel / "cube.zarr")
    lines = ["### cube.zarr (dims field x time; NaN = missing)", "", "| variable | units | description |", "|---|---|---|"]
    for v in ds.data_vars:
        a = ds[v].attrs
        lines.append(f"| `{v}` | {a.get('units', '')} | {a.get('long_name', '')} |")
    lines += ["", "### fields.parquet", "", "| column | units | description | missing |", "|---|---|---|---|"]
    lines += [f"| `{k}` | {u} | {d} | {m} |" for k, (u, d, m) in FIELD_COLUMNS.items()]
    lines += ["", "### optical_observations/YYYY.parquet (one row per field and HLS granule)", "",
              "Field-mean surface reflectance of the clear pixels (`blue` ... `swir2`, unitless), field-mean of per-pixel "
              "indices (`ndvi`, `evi`, `ndmi`, `mndwi`, `nbr`), `ndvi_sd`, `water_frac` (Fmask water bit), "
              "`n_clear_px`, `n_field_px`, `clear_frac`, `date` (UTC), `sensor`, `tile`, `item` (granule id), "
              "`shift_row`, `shift_col` (30 m pixels, NaN = not estimated), `shift_peak`, `shift_px`, `shift_scope` "
              "(`fields`, `scene` or `none`: where the shift was estimated), the quality "
              "flags and `qa_use`.", "",
              "### radar_observations/YYYY.parquet (one row per field and Sentinel-1 acquisition)", "",
              "`vv_lin`, `vh_lin` (gamma0, linear, field mean over `n_px` pixels), `vv_db`, `vh_db`, `vh_vv_db`, "
              "`date`, `item`, `platform`, `orbit_state`, `relative_orbit`."]
    return "\n".join(lines)


def datasheet(rel):
    f = pd.read_parquet(rel / "fields.parquet", columns=["tier", "n_parts", "water_table_min_cm"])
    from fpcube.cube import calendar
    t = calendar()
    summ = json.loads((rel / "validation" / "summary.json").read_text())
    code = summ.get("code", {})
    gna = summ["stats"]["coverage"].get("geom_not_assessed_frac")
    n_multi = int((f.n_parts > 1).sum())
    txt = DATASHEET.format(version=rel.name, prism_date=prism_retrieved(rel), n_fields=len(f),
                           n_a=int((f.tier == "A").sum()),
                           n_b=int((f.tier == "B").sum()), n_periods=len(t), start=t[0].date(),
                           end=(t[-1] + pd.Timedelta(days=config.COMPOSITE_DAYS - 1)).date(),
                           geom_na=f"{100 * gna:.1f} %" if gna is not None else "some",
                           multipart=MULTIPART.format(n=n_multi) if n_multi else
                           "- **Fields are single parts.** Every field is one 4-connected part (`n_parts` = 1).\n",
                           lineage=LINEAGE if (rel / "lineage.parquet").exists() else "",
                           n_wt_missing=int(f.water_table_min_cm.isna().sum()),
                           commit=code.get("commit", "?") + (" (with local changes)" if code.get(
                               "code_modified_since_commit") else ""),
                           code_hash=code.get("code_sha256_16", "?"), dictionary=dictionary(rel))
    (rel / "DATASHEET.md").write_text(txt)


def checksums(rel):
    lines = [f"{sha256(p)}  {p.relative_to(rel)}" for p in _files(rel, zarr=True) if p.name != "SHA256SUMS"]
    (rel / "SHA256SUMS").write_text("\n".join(lines) + "\n")


ZIP_DATE = (2026, 1, 1, 0, 0, 0)          # fixed member dates: the same release gives byte-identical zips


def _zip(src, dest, prefix):
    """Zip every file under src, stored (no recompression), member names prefix + path relative to src."""
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in sorted(x for x in src.rglob("*") if x.is_file()):
            zi = zipfile.ZipInfo(prefix + p.relative_to(src).as_posix(), ZIP_DATE)
            zi.external_attr = 0o644 << 16
            with open(p, "rb") as f, z.open(zi, "w", force_zip64=True) as out:
                shutil.copyfileobj(f, out, 1 << 22)


def deposit(rel, out=None):
    """Flat folder for Zenodo: plain files as they are, each directory as a zip, and DEPOSIT_SHA256SUMS for the
    uploaded files. Built in a temporary folder and moved into place (new files written straight into a OneDrive
    folder can fail)."""
    out = out or config.DATA / "deposit" / rel.name
    tmp = Path(tempfile.mkdtemp(prefix="fpcube_deposit_", dir="/private/tmp"))
    tmp.chmod(0o755)                                        # mkdtemp makes it owner-only
    for p in sorted(rel.iterdir()):
        if p.is_file():
            shutil.copy2(p, tmp / p.name)
        elif p.name.endswith(".zarr"):
            _zip(p, tmp / f"{p.name}.zip", "")              # store contents at the root -> ZipStore opens it
        else:
            _zip(p, tmp / f"{p.name}.zip", p.name + "/")
        print("packed", p.name, flush=True)
    files = sorted(x for x in tmp.iterdir() if x.name != "DEPOSIT_SHA256SUMS")
    (tmp / "DEPOSIT_SHA256SUMS").write_text("".join(f"{sha256(x)}  {x.name}\n" for x in files))
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(out))
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rel = config.RELEASE / (args[0] if args else "v1.0")
    if "--deposit" in sys.argv:
        print(deposit(rel))
    else:
        copy_lineage(rel)
        datasheet(rel)
        print(build_catalog(rel))
        checksums(rel)
