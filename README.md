# fpcube — Florida Pasture Cube

Builds and reads the **Florida Pasture Cube**: every persistent pasture field in Florida (about 81,000 fields) with a
5-day time series from January 2016 to October 2026 of HLS optical reflectance and indices, Sentinel-1 backscatter,
and PRISM weather. Each field also carries gNATSGO soil, 3DEP terrain (HAND, TWI), C-CAP wetlands, US Drought Monitor
and NASS condition labels, hurricane tracks, and three benchmark tasks.

## Project layout

Everything for the project lives in this folder (OneDrive-Personal/Packages/FPCube):

| Path | What | In git |
|---|---|---|
| `src/fpcube/` | pipeline package (sources, fields, cube, validation, benchmark, release) | yes |
| `scripts/` | `pipeline.sh` (unattended build), `paper_numbers.py`, `figures.py`; `audit/` holds the cube and static-table audits | yes |
| `manuscript/` | ESSD data paper (`fpcube_essd.tex`); numbers, tables and figures are generated, never typed | yes |
| `tests/`, `notebooks/`, `config/` | tests and examples | yes |
| `data/` | raw downloads, interim tables, `release/v1.0`, logs and pipeline state (~30 GB) | no |
| `docs/` | proposal documents (AIIRI RFP, Track 2 proposal and concept) | no |

The Python environment is kept outside OneDrive (`~/fpcube_env`, about 1.5 GB of small files). Recreate it with
`uv venv ~/fpcube_env --python 3.12 && uv pip install --python ~/fpcube_env/bin/python -e .`

Keep `data/` set to "Always keep on this device" in OneDrive, so that the pipeline never reads cloud-only files.

## Reading the release

```python
import xarray as xr, geopandas as gpd
cube = xr.open_zarr("data/release/v1.0/cube.zarr")            # dims: field x time
fields = gpd.read_parquet("data/release/v1.0/fields.parquet")  # one row per field, EPSG:5070
ndvi = cube.ndvi.sel(field="FPC000123")
```

## Rebuilding

```bash
uv venv --python 3.12 && uv pip install -e .
fpcube cdl && fpcube parcels && fpcube fields && fpcube parcels-all && fpcube fields
fpcube static && fpcube prism && fpcube labels        # FPCUBE_NASS_KEY for NASS
fpcube optical && fpcube radar                       # days; resumable
fpcube cube && fpcube benchmark && fpcube release
```

Data go to `$FPCUBE_DATA` (default `data/` in this folder). HLS before 2020 is read from NASA LP DAAC and needs an Earthdata
login in `~/.netrc`; everything else is public (Microsoft Planetary Computer, PRISM, CropScape, FDACS, USDM, NHC).

Before submission, refresh the provisional PRISM months (values become stable about six months after the date):
delete `data/raw/prism/*/prism_*_2026.tif` (and 2025 if it ends within six months), then run `fpcube prism`, `fpcube cube`
and `python -m fpcube.validation`; the validation text states the retrieval date and the provisional window.

## Benchmark

| Task | Input | Target | Split |
|---|---|---|---|
| 1 Greenness forecast | 60 days of composites + static | NDVI at +15/30/45/60 days | years 2016–22 / 23–24 / 25–26, plus held-out counties |
| 2 Event exposure | ±30 days around an event | exposed or not (hurricane, extreme rain, freeze, drought onset) | leave one event out |
| 3 County drought | fields of a county, 30 days | USDM class (none, D0, D1, D2+) | as Task 1 |

`python -m fpcube.benchmark.run` reproduces the baselines.

## Licence

Code: MIT. Data: CC BY 4.0.
