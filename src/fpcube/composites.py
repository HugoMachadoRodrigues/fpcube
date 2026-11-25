"""The 5-day composites of cube.zarr as one Parquet file per year, for readers without Zarr (R, SQL engines, Excel).

  python -m fpcube.composites v1.3        # writes <release>/composites/composites_YYYY.parquet

One row per field and 5-day period, one column per variable, the same values and types as in cube.zarr. Rows are
sorted by field and date, so a filter on `field_id` or `fid` reads only a few row groups. `date` is the first day
of the period; the period covers `date` to `date + 4 days`. The schema metadata holds the units and description of
every variable (key `fpcube`).
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fpcube import config, provenance

ROW_GROUP = 250_000
TMP = Path("/private/tmp")
OPTICAL = ("blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "evi", "ndmi", "mndwi", "nbr", "ndvi_sd",
           "water_frac")
RADAR = ("vv_db", "vh_db", "vh_vv_db")
WEATHER = ("ppt_sum", "tmean_mean", "tmin_mean", "tmax_mean", "vpdmax_mean")
COUNTS = ("n_optical_obs", "gap_filled", "n_radar_obs")


def group(v):
    return ("optical" if v in OPTICAL else "radar" if v in RADAR else "weather" if v in WEATHER else "count")


def variables(ds):
    """Ordered variable list (optical, counts, radar, weather) with units, description and group."""
    order = [v for g in (OPTICAL, COUNTS, RADAR, WEATHER) for v in g if v in ds.data_vars]
    extra = [v for v in ds.data_vars if v not in order]
    return [{"name": v, "group": group(v), "units": ds[v].attrs.get("units", ""),
             "description": ds[v].attrs.get("long_name", "")} for v in order + extra]


def export(rel, out=None):
    import xarray as xr
    ds = xr.open_zarr(rel / "cube.zarr")
    out = out or rel / "composites"
    out.mkdir(parents=True, exist_ok=True)
    fields = pd.read_parquet(rel / "fields.parquet", columns=["field_id", "fid"]).set_index("field_id").fid
    ids = ds.field.values.astype(str)
    fid = fields.reindex(ids).to_numpy()
    assert not np.isnan(fid.astype(float)).any(), "cube fields missing from fields.parquet"
    vs = variables(ds)
    meta = {"version": rel.name, "composite_days": int(ds.attrs.get("composite_days", config.COMPOSITE_DAYS)),
            "anchor": ds.attrs.get("anchor"), "variables": vs, "code": provenance.info(),
            "note": "date is the first day of the 5-day period. Optical values with n_optical_obs = 0 are gap-filled "
                    "(gap_filled = true): do not use them as inputs of a forecast or any causal analysis."}
    order = np.argsort(ids, kind="stable")
    years = pd.DatetimeIndex(ds.time.values).year
    written = []
    for y in sorted(set(years)):
        ti = np.flatnonzero(years == y)
        sub = ds.isel(time=ti).load()
        nf, nt = len(ids), len(ti)
        cols = {"field_id": pa.array(np.repeat(ids[order], nt)),
                "fid": pa.array(np.repeat(fid[order].astype("int32"), nt)),
                "date": pa.array(np.tile(pd.DatetimeIndex(sub.time.values).date, nf), pa.date32())}
        for v in vs:
            a = sub[v["name"]].transpose("field", "time").values[order].reshape(-1)
            cols[v["name"]] = pa.array(a)
        t = pa.table(cols).replace_schema_metadata({"fpcube": json.dumps(meta)})
        p = out / f"composites_{y}.parquet"
        tmp = TMP / p.name                        # write outside OneDrive, then move into place
        pq.write_table(t, tmp, compression="zstd", compression_level=6, row_group_size=ROW_GROUP,
                       write_statistics=True, use_dictionary=["field_id"])
        shutil.move(str(tmp), str(p))
        written.append(p)
        print(f"composites {y}: {t.num_rows} rows, {p.stat().st_size / 1e6:.0f} MB", flush=True)
    return written


def check(rel, n=200, seed=0):
    """Compare a random sample of field x period values of the Parquet files with cube.zarr."""
    import xarray as xr
    ds = xr.open_zarr(rel / "cube.zarr")
    rng = np.random.default_rng(seed)
    fi = rng.integers(0, ds.sizes["field"], n)
    ti = rng.integers(0, ds.sizes["time"], n)
    bad = 0
    for f, t in zip(fi, ti):
        fid, date = str(ds.field.values[f]), pd.Timestamp(ds.time.values[t])
        tab = pq.read_table(rel / "composites" / f"composites_{date.year}.parquet",
                            filters=[("field_id", "=", fid), ("date", "=", date.date())]).to_pandas()
        assert len(tab) == 1, (fid, date)
        cell = ds.isel(field=int(f), time=int(t)).load()
        for v in ds.data_vars:
            a, b = cell[v].values.item(), tab[v].iat[0]
            bad += not ((pd.isna(a) and pd.isna(b)) or a == b)
    if bad:
        raise RuntimeError(f"{bad} values differ between composites and cube.zarr")
    return n


if __name__ == "__main__":
    rel = config.RELEASE / (sys.argv[1] if len(sys.argv) > 1 else "v1.0")
    export(rel)
    print("checked", check(rel), "random field-periods against cube.zarr")
