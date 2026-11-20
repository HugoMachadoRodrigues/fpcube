# fpcube — Florida Pasture Cube

Builds and reads the **Florida Pasture Cube**: persistent pasture fields in Florida (about 81,000 fields) with a
5-day time series from January 2016 to October 2026 of HLS optical reflectance and indices, Sentinel-1 backscatter,
and PRISM weather. Each field also carries gNATSGO soil, 3DEP terrain (HAND, TWI), C-CAP wetlands, US Drought Monitor
and NASS condition labels, hurricane tracks, and three benchmark tasks.

## Project layout

Everything for the project lives in this folder (OneDrive-Personal/Packages/FPCube):

| Path | What | In git |
|---|---|---|
| `src/fpcube/` | pipeline package (sources, fields, cube, validation, benchmark, release) | yes |
| `scripts/` | `pipeline.sh` (unattended build), `figures.py`, `validation_sample.py`; `audit/` holds the cube and static-table audits | yes |
| `manuscript/` | Manuscript in Word (`fpcube_manuscript.docx`) and the figures written by `scripts/figures.py`; the numbers it quotes are in `validation/summary.json` and `benchmark/` of the release | yes |
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
observed = ndvi.where(cube.n_optical_obs.sel(field="FPC000123") > 0)   # without the interpolated values
```

Gap-filled optical values interpolate towards the next observation, so they must not be used as inputs of a forecast
(the benchmark drops them: `fpcube.benchmark.tasks.observed_only`).

From the Zenodo files, the cube opens without unzipping:

```python
import zarr
cube = xr.open_zarr(zarr.storage.ZipStore("cube.zarr.zip", mode="r"))
```

Unzipping every zip in one folder (`unzip cube.zarr.zip -d cube.zarr`, the others as they are) gives back the release
layout; `shasum -a 256 -c SHA256SUMS` then checks every file, and `stac/catalog.json` describes them.

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
| 1 Greenness forecast | 60 days of observed composites + static | observed NDVI at +15/30/45/60 days | years 2016–22 / 23–24 / 25–26 for issue date and target, plus held-out counties |
| 2 Event exposure | ±30 days of observed composites around an event | exposed or not (hurricane, extreme rain, freeze, drought onset) | leave one event out; leave one event type out |
| 3 County drought | fields of a county, 30 days ending before the map date | USDM class (none, D0, D1, D2+) | years, plus held-out counties |

`python -m fpcube.benchmark.run task1` (about 30 min on an M1 Max), then `python -m fpcube.benchmark.run task2 task3`.
Each task writes `<task>_run_info.json` with the code version (`fpcube.provenance`). All models of a task are scored
on the same samples; intervals come from two-way cluster bootstrap (county x date) in Tasks 1 and 3 and from
resampling events in Task 2.

## Checks and validation material

- `python -m fpcube.validation v1.0`: validation tables and text, including the area accounting of the delineation
  and quality by tier and field size.
- `python scripts/validation_sample.py`: stratified points inside and outside the fields
  (`validation/photo_sample.csv`, `.kml`) to label against high-resolution imagery.
- `python scripts/audit/index_audit.py 50`: re-reads a sample of HLS scenes to count undefined and out-of-range
  per-pixel indices (`data/audit/index_audit.json`).
- `python -m fpcube.sources.scene_shift` estimates the shift of granules whose field window was too small (scene level), and `python -m fpcube.cube --apply-scene-shift v1.0` brings it into the release (re-screened tables, recomposited fields).

## Publishing a version on Zenodo

1. `python -m fpcube.release v1.0` writes `DATASHEET.md`, `stac/` and `SHA256SUMS` in `data/release/v1.0`.
2. `python -m fpcube.release v1.0 --deposit` writes `data/deposit/v1.0`: the files to upload (one zip per directory,
   stored, plus `DEPOSIT_SHA256SUMS`). It can be deleted after the upload.
3. Data record: new Zenodo upload with the metadata in `zenodo/dataset_v1.0.json` (title, description, licence,
   keywords, notes with the Copernicus Sentinel data attribution); upload every file of `data/deposit/v1.0`.
4. Code record: make this repository public, switch it on at zenodo.org/account/settings/github, and publish a GitHub
   release (tag `v1.0.0`); Zenodo archives it with `.zenodo.json` and `CITATION.cff`.
5. Write both DOIs into the manuscript (`manuscript/fpcube_manuscript.docx`).

## Licence

Code: MIT. Data: CC BY 4.0.
