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
DESCRIPTION = ("Field-level, 5-day time series of every persistent pasture field in Florida, 2016-2026: HLS optical "
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
    groups = {"fields": ["fields.parquet"], "cube": ["cube.zarr"],
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
- **Instances.** {n_fields} pasture fields (tier A: {n_a}; tier B: {n_b}), {n_periods} five-day periods
  ({start} to {end}).
- **Per field:** geometry, county, parcel use code (no owner information), area and shape, CDL pasture fraction
  per year, quality flags, soil, terrain and wetland attributes.
- **Per field and period:** HLS reflectance and indices (with observation counts and gap-filling flag),
  Sentinel-1 VV/VH, PRISM weather.
- **Per field and acquisition:** long tables with clear-pixel counts, geometric shift, sensor, orbit; optical
  records carry quality flags (`qa_clear`, `qa_geometry`, `qa_range`, `qa_haze`) and `qa_use`, the records that
  enter the composites.
- **Labels:** US Drought Monitor (county-week), NASS pasture condition (state-week), hurricane tracks.
- **Missing data.** Optical gaps longer than 30 days are left empty on purpose. Radar coverage is thinner in
  2022-2024 after the Sentinel-1B failure. `water_table_min_cm` is empty where gNATSGO records no water table in
  the soil profile (mostly excessively and well-drained soils): read it as deeper than about 2 m, not as unknown.
  `twi_mean` uses tan(slope) >= 0.001, so flat fields have a value.
  NASS pasture condition is absent for weeks NASS did not publish (shutdowns of Dec 2018-Jan 2019 and
  Oct-Nov 2025; Dec 2025-Mar 2026). Landfalls of 2026 storms are derived from working best tracks (preliminary).
- **Sensitive data.** None. Parcel owner names were never downloaded.

## Collection and preprocessing
See the paper, Sections 2-4. All sources are public; the processing code (`fpcube`) rebuilds the release.

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


def datasheet(rel):
    f = pd.read_parquet(rel / "fields.parquet", columns=["tier"])
    from fpcube.cube import calendar
    t = calendar()
    txt = DATASHEET.format(version=rel.name, prism_date=prism_retrieved(rel), n_fields=len(f),
                           n_a=int((f.tier == "A").sum()),
                           n_b=int((f.tier == "B").sum()), n_periods=len(t), start=t[0].date(),
                           end=(t[-1] + pd.Timedelta(days=config.COMPOSITE_DAYS - 1)).date())
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
        datasheet(rel)
        print(build_catalog(rel))
        checksums(rel)
