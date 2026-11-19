"""STAC catalog, checksums and datasheet for a release directory.

python -m fpcube.release v1.0
"""
import hashlib
import json
import sys
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


def _files(rel):
    for p in sorted(rel.rglob("*")):
        if p.is_file() and ".zarr" not in p.parts[-2] and not any(part.endswith(".zarr") for part in p.parts):
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
            href = str(p.relative_to(rel))
            it.add_asset(nm, pystac.Asset(href=href, media_type=mt, roles=["data"], extra_fields=extra))
        col.add_item(it)
    cat.normalize_hrefs(str(rel))
    cat.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return rel / "catalog.json"


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
- **Sensitive data.** None. Parcel owner names were never downloaded.

## Collection and preprocessing
See the paper, Sections 2-4. All sources are public; the processing code (`fpcube`) rebuilds the release.

## Uses
- **Intended:** pasture monitoring research, hazard impact analysis, forecasting, weakly supervised learning.
- **Not intended:** decisions about individual properties, insurance or regulatory compliance without field
  verification. Field boundaries follow parcels and 30 m pixels, not fences.

## Distribution and licence
CC BY 4.0. DOI in the README. STAC catalog: `catalog.json`.

## Maintenance
Annual updates planned (new season, new CDL and parcel vintages); versions are immutable and numbered.
"""


def datasheet(rel):
    f = pd.read_parquet(rel / "fields.parquet", columns=["tier"])
    from fpcube.cube import calendar
    t = calendar()
    txt = DATASHEET.format(version=rel.name, n_fields=len(f), n_a=int((f.tier == "A").sum()),
                           n_b=int((f.tier == "B").sum()), n_periods=len(t), start=t[0].date(),
                           end=(t[-1] + pd.Timedelta(days=config.COMPOSITE_DAYS - 1)).date())
    (rel / "DATASHEET.md").write_text(txt)


def checksums(rel):
    lines = [f"{sha256(p)}  {p.relative_to(rel)}" for p in _files(rel) if p.name != "SHA256SUMS"]
    (rel / "SHA256SUMS").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    rel = config.RELEASE / (sys.argv[1] if len(sys.argv) > 1 else "v1.0")
    datasheet(rel)
    print(build_catalog(rel))
    checksums(rel)
