# fpcube: the Florida Pasture Cube

`fpcube` builds, validates and documents the **Florida Pasture Cube**, a field-level data set of Florida grazing
lands. It covers 62,847 persistent pasture fields (0.98 Mha) with a 5-day time series from January 2016 to October
2026 (791 periods):

- **Optical:** HLS v2.0 reflectance and indices (NDVI, EVI, NDMI, MNDWI, NBR), with observation counts and a
  gap-filling flag.
- **Radar:** Sentinel-1 VV and VH backscatter.
- **Weather:** PRISM daily weather aggregated to 5 days.
- **Field attributes:** gNATSGO soil, 3DEP terrain (slope, TWI, HAND) and C-CAP wetland context.
- **Labels:** US Drought Monitor (county-week), NASS pasture condition (state-week), hurricane tracks and landfalls.
- **Benchmark:** three tasks with fixed splits and baseline results.

Per-acquisition long tables keep the quality information of every value, so every composite can be traced to the
observations it came from.

| | |
|---|---|
| Data (v1.3) | Zenodo, DOI to be added on publication (CC BY 4.0) |
| Code | this repository (MIT); archived on Zenodo with each release |
| Datasheet | `DATASHEET.md` in the data record |

## Fields

Fields are connected parts of persistent pasture inside one tax parcel, at least 2 ha:

- **Tier A:** fields on parcels coded as grazing land.
- **Tier B:** fields on other parcels, kept only when they were mapped as pasture in nearly every year.
- **Tier C (from v1.3):** open pasture taken from the official land-use maps of Florida's water management districts
  (FLUCCS) that the Cropland Data Layer maps as woody or wetland. It must be grassland or cropland in ESA WorldCover
  2021.

For a conservative set of grazing fields use tier A.

Only the land-use code and the agricultural-exemption flag of the public parcel roll are kept. Owner names and
addresses are never downloaded, and parcel identifiers are not distributed.

## Reading the data

The examples assume the Zenodo files are in the folder you work in, or a rebuilt release in `data/release/v1.3/`.
The cube opens directly from the zip:

```python
import geopandas as gpd
import xarray as xr
import zarr

cube = xr.open_zarr(zarr.storage.ZipStore("cube.zarr.zip", mode="r"))   # dims: field x time
fields = gpd.read_parquet("fields.parquet")                             # one row per field, EPSG:5070

ndvi = cube.ndvi.sel(field="FPC000123")
observed = ndvi.where(cube.n_optical_obs.sel(field="FPC000123") > 0)     # drop gap-filled values
```

Gap-filled optical values are interpolated towards the next observation. Do not use them as inputs to a forecast or
any other analysis that must be causal. The benchmark removes them with `fpcube.benchmark.tasks.observed_only`.

To recover the release layout, unzip every zip in one folder (`unzip cube.zarr.zip -d cube.zarr`, the others as they
are). Then `shasum -a 256 -c SHA256SUMS` checks every file, and `stac/catalog.json` describes them.

## Installation

```bash
uv venv --python 3.12
uv pip install -e .            # or: pip install -e .
pip install -e ".[test]" && pytest tests
```

## Rebuilding the data set

```bash
fpcube cdl && fpcube parcels && fpcube fields && fpcube parcels-all && fpcube fields
fpcube static && fpcube prism && fpcube labels         # NASS needs a free API key in FPCUBE_NASS_KEY
fpcube optical && fpcube radar                        # takes days; resumable
fpcube cube && fpcube benchmark && fpcube release
```

`scripts/pipeline.sh` runs the extraction steps unattended. It restarts any step that fails or stalls, then runs the
downstream steps in order.

Data are written to `$FPCUBE_DATA` (default `data/`). HLS before 2020 is read from NASA LP DAAC and needs an
Earthdata login in `~/.netrc`. All other sources are public: Microsoft Planetary Computer, PRISM, CropScape, the
Florida parcel roll, USDM, NASS and NHC.

### Field versions

Field versions are selected with `FPCUBE_FIELDS`. Each version reuses the observations of the fields it shares with
earlier versions, and `lineage.parquet` links every field identifier to v1.0.

| Version | What it does | How to build it |
|---|---|---|
| v1.0 | the build above | — |
| v1.1 | re-delineation into connected parts with a woody wetland screen; identifiers stay stable | `python -m fpcube.fields v1.1` |
| v1.2 | selection of the v1.1 fields: no land-use change, and tier B only with pasture in ≥ 9.5 of 10 CDL years | `python -m fpcube.fields v1.2` |
| v1.3 | v1.2 plus tier C; only tier C is extracted again | `python -m fpcube.fields v1.3` |

v1.3 needs `python -m fpcube.sources.fluccs` and `python -m fpcube.sources.worldcover` first.

After any version, run these with the same `FPCUBE_FIELDS`: `fpcube cube`, `python -m fpcube.validation <version>`,
`python -m fpcube.benchmark.run --release <version>` and `python -m fpcube.release <version>`.

PRISM values become stable about six months after the date. To refresh the provisional months, delete the latest
yearly PRISM files under `data/raw/prism/`, then run `fpcube prism`, `fpcube cube` and the validation again.

## Validation

`python -m fpcube.validation <version>` writes the validation tables and text:

- area accounting of the delineation;
- agreement of tiers A and B with the official FLUCCS land use and with ESA WorldCover 2021;
- observation density and quality by tier and field size;
- physical consistency: rainfall, the 2018 freeze, Hurricane Ian and NASS pasture condition.

Tier C was selected with FLUCCS and WorldCover, so these two sources cannot validate it. Its agreement with them is
reported as holding by construction.

`scripts/audit/` holds audits of the per-pixel indices and of the cube, static and label tables.

## Benchmark

| Task | Input | Target | Split |
|---|---|---|---|
| 1 Greenness forecast | 60 days of observed composites + static attributes | observed NDVI at +15/30/45/60 days | years 2016–22 / 23–24 / 25–26, plus held-out counties |
| 2 Event exposure | ±30 days of observed composites around an event | exposed or not (hurricane, extreme rain, freeze, drought onset) | leave one event out; leave one event type out |
| 3 County drought | fields of a county, 30 days ending before the map date | change of the USDM class four weeks ahead | years, plus held-out counties |

Run it with `python -m fpcube.benchmark.run task1 task2 task3`. Task 1 takes about 30 min on an M1 Max.

All models of a task are scored on the same samples. Intervals come from a two-way cluster bootstrap over counties
and dates in Tasks 1 and 3, and from resampling events in Task 2.

## Reproducibility

Every validation and benchmark file records the git commit and a content hash of the code (`fpcube.provenance`).
Release v1.3 was produced by commit `db2b57d`, with code hash `1a5eb14ada85067b`.

## Citation

Cite the data set through its Zenodo DOI and this code through the DOI of the release you used (see
`CITATION.cff`).

## Licence

Code: MIT. Data: CC BY 4.0. The data set contains modified Copernicus Sentinel data (2016–2026); the attributions of
all sources are listed in `DATASHEET.md`.
