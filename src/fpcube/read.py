"""Read a Florida Pasture Cube release (or the files downloaded from Zenodo) into pandas / geopandas.

The same functions as the R package fpcubeR (https://github.com/HugoMachadoRodrigues/fpcubeR):

    from fpcube import read
    f = read.fields("data/release/v1.3", tier="A", county="Hardee")
    x = read.composites("data/release/v1.3", fields=f, vars=["ndvi", "vv_db", "ppt_sum"], years=[2022])
    x = read.add_drought("data/release/v1.3", x)
    a = read.anomaly(x, "ndvi", ref_years=[2021, 2022])

`dir` is a folder with the layout of the release: fields.parquet, composites/composites_YYYY.parquet, labels/,
optical_observations/, radar_observations/. Optical values of gap-filled periods (n_optical_obs = 0) are
interpolated from the next observation; `composites(observed_only=True)`, the default, sets them to NaN.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

LABELS = {"usdm": "usdm_county_weekly.parquet", "nass": "nass_fl_pasture_condition_weekly.parquet",
          "tracks": "hurricane_tracks_florida.parquet", "landfalls": "hurricane_landfalls.parquet"}


def _need(p, part):
    if not Path(p).exists():
        raise FileNotFoundError(f"missing {p}; download the '{part}' part of the data set")
    return Path(p)


def _ids(fields):
    if fields is None:
        return None
    if isinstance(fields, pd.DataFrame):
        return [str(x) for x in fields["field_id"]]
    if isinstance(fields, str):
        return [fields]
    return [str(x) for x in fields]


def fields(dir, tier=None, county=None, fields=None, columns=None, geometry=True):
    """Fields as a GeoDataFrame (EPSG:5070) or a DataFrame, filtered by tier, county name or identifier."""
    import geopandas as gpd
    p = _need(Path(dir) / "fields.parquet", "fields")
    names = pq.read_schema(p).names
    cols = names if columns is None else list(dict.fromkeys(["field_id", "fid", "tier", "county", *columns]))
    if not geometry:
        cols = [c for c in cols if c != "geometry"]
    elif "geometry" not in cols:
        cols.append("geometry")
    bad = [c for c in cols if c not in names]
    if bad:
        raise ValueError(f"unknown column(s): {bad}")
    filt = []
    if tier is not None:
        filt.append(("tier", "in", [t.upper() for t in np.atleast_1d(tier)]))
    if fields is not None:
        filt.append(("field_id", "in", _ids(fields)))
    x = (gpd.read_parquet(p, columns=cols, filters=filt or None) if geometry
         else pd.read_parquet(p, columns=cols, filters=filt or None))
    if county is not None:
        norm = lambda s: re.sub(r"\s+county$", "", str(s).strip(), flags=re.I).lower()
        want = {norm(c) for c in np.atleast_1d(county)}
        x = x[x["county"].map(norm).isin(want)]
    return x.sort_values("field_id").reset_index(drop=True)


def _composite_files(dir, years=None):
    files = sorted((Path(dir) / "composites").glob("composites_[0-9][0-9][0-9][0-9].parquet"))
    if not files:
        raise FileNotFoundError(f"no composites in {Path(dir) / 'composites'}; download the 'composites' part")
    have = {int(f.stem[-4:]): f for f in files}
    if years is None:
        return list(have.values())
    miss = sorted(set(years) - set(have))
    if miss:
        raise FileNotFoundError(f"composites of {miss} are not in {dir}")
    return [have[y] for y in sorted(set(years))]


def variables(dir):
    """Name, group, units and description of the composite variables (from the Parquet metadata)."""
    meta = json.loads(pq.read_schema(_composite_files(dir)[0]).metadata[b"fpcube"])
    return pd.DataFrame(meta["variables"])


def composites(dir, fields=None, vars=None, years=None, start=None, end=None, observed_only=True, long=False):
    """5-day composites, one row per field and period (or per field, period and variable with long=True)."""
    if years is None and (start is not None or end is not None):
        all_y = [int(f.stem[-4:]) for f in _composite_files(dir)]
        lo = pd.Timestamp(start).year if start is not None else min(all_y)
        hi = pd.Timestamp(end).year if end is not None else max(all_y)
        years = [y for y in all_y if lo <= y <= hi]
    files = _composite_files(dir, years)
    v = variables(dir)
    allv = list(v["name"])
    vars = allv if vars is None else list(vars)
    bad = [x for x in vars if x not in allv]
    if bad:
        raise ValueError(f"unknown variable(s): {bad}")
    cols = list(dict.fromkeys(["field_id", "fid", "date", *vars, "n_optical_obs", "gap_filled", "n_radar_obs"]))
    expr = None
    ids = _ids(fields)
    if ids is not None:
        expr = ds.field("field_id").isin(ids)
    for op, val in ((">=", start), ("<=", end)):
        if val is not None:
            e = (ds.field("date") >= pd.Timestamp(val).date()) if op == ">=" else (ds.field("date") <= pd.Timestamp(val).date())
            expr = e if expr is None else expr & e
    x = ds.dataset([str(f) for f in files]).to_table(columns=cols, filter=expr).to_pandas()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values(["field_id", "date"]).reset_index(drop=True)
    if observed_only:
        opt = [c for c in vars if c in set(v.loc[v.group == "optical", "name"])]
        x.loc[x["n_optical_obs"] == 0, opt] = np.nan
    if long:
        keep = [c for c in vars if c not in ("n_optical_obs", "gap_filled", "n_radar_obs")]
        x = x.melt(id_vars=["field_id", "fid", "date", "n_optical_obs", "gap_filled", "n_radar_obs"],
                   value_vars=keep, var_name="variable", value_name="value")
        x = x.sort_values(["field_id", "date", "variable"]).reset_index(drop=True)
    return x


def observations(dir, type="optical", fields=None, years=None, usable_only=True):
    """Per-acquisition optical (HLS) or radar (Sentinel-1) records; optical keeps qa_use records by default."""
    sub = Path(dir) / f"{type}_observations"
    files = sorted(sub.glob("[0-9][0-9][0-9][0-9].parquet"))
    if years is not None:
        files = [f for f in files if int(f.stem) in set(years)]
    if not files:
        raise FileNotFoundError(f"no {type} observations for these years in {dir}")
    lut = globals()["fields"](dir, fields=fields, columns=[], geometry=False)
    expr = None
    if fields is not None:
        expr = ds.field("fid").isin([int(i) for i in lut.fid])
    if type == "optical" and usable_only:
        expr = ds.field("qa_use") if expr is None else expr & ds.field("qa_use")
    x = ds.dataset([str(f) for f in files]).to_table(filter=expr).to_pandas()
    x.insert(0, "field_id", x["fid"].map(dict(zip(lut.fid, lut.field_id))))
    return x.sort_values(["field_id", "date"]).reset_index(drop=True)


def labels(dir, which="usdm"):
    """usdm (county-week), nass (state-week), tracks or landfalls."""
    return pd.read_parquet(_need(Path(dir) / "labels" / LABELS[which], "labels"))


def add_drought(dir, x):
    """Add the US Drought Monitor class (0 none ... 5 D4), percent in D1+ and DSCI of the field's county."""
    f = fields(dir, fields=x["field_id"].unique(), columns=["county_fips"], geometry=False)
    u = labels(dir, "usdm")
    u["valid_start"] = pd.to_datetime(u["valid_start"]).astype("datetime64[ns]")
    u["valid_end"] = pd.to_datetime(u["valid_end"]).astype("datetime64[ns]")
    u["county_fips"] = u["county_fips"].astype(str)
    y = x.copy()
    y["date"] = pd.to_datetime(y["date"]).astype("datetime64[ns]")
    y["county_fips"] = y["field_id"].map(dict(zip(f.field_id, f.county_fips.astype(str))))
    y["_row"] = np.arange(len(y))
    m = pd.merge_asof(y.sort_values("date"), u.sort_values("valid_start")[
        ["county_fips", "valid_start", "valid_end", "dominant_class", "D1", "dsci"]],
        left_on="date", right_on="valid_start", by="county_fips", direction="backward")
    m.loc[m["date"] > m["valid_end"], ["dominant_class", "D1", "dsci"]] = np.nan
    m = m.sort_values("_row")
    out = x.copy()
    out["usdm_class"] = m["dominant_class"].to_numpy()
    out["usdm_d1_pct"] = m["D1"].to_numpy()
    out["dsci"] = m["dsci"].to_numpy()
    return out


def period_of_year(date):
    d = pd.to_datetime(pd.Series(date))
    return np.minimum((d.dt.day_of_year.to_numpy() - 1) // 5 + 1, 73)


def climatology(x, var, ref_years=None, min_years=2):
    """Mean of the values of each field and 5-day period of the year (1-73) over the reference years."""
    d = x[["field_id", "date", var]].copy()
    if ref_years is not None:
        d = d[d["date"].dt.year.isin(ref_years)]
    d["period"] = period_of_year(d["date"])
    d = d.dropna(subset=[var])
    c = d.groupby(["field_id", "period"])[var].agg(clim="mean", n_years="size").reset_index()
    c.loc[c["n_years"] < min_years, "clim"] = np.nan
    return c


def anomaly(x, var, ref_years=None, min_years=2):
    """x with `period`, `clim` and `anomaly` = value - climatology of the field and period."""
    c = climatology(x, var, ref_years, min_years)
    y = x.copy()
    y["period"] = period_of_year(y["date"])
    y = y.merge(c[["field_id", "period", "clim"]], on=["field_id", "period"], how="left")
    y["anomaly"] = y[var] - y["clim"]
    return y
