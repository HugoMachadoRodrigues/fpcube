"""Read a Florida Pasture Cube release (or the files downloaded from Zenodo) into pandas / geopandas.

The same functions as the R package fpcubeR (https://github.com/HugoMachadoRodrigues/fpcubeR):

    from fpcube import read
    f = read.fields("data/release/v1.3", tier="A", county="Hardee")
    x = read.composites("data/release/v1.3", fields=f, vars=["ndvi", "vv_db", "ppt_sum"], years=[2022])
    x = read.add_drought("data/release/v1.3", x)
    a = read.anomaly(x, "ndvi", ref_years=[2021, 2022])
    l = read.lineage("data/release/v1.3", parents="FPC057124")          # a version 1.0 field split in two
    s = read.benchmark("data/release/v1.3", "task1_samples", fields=f)  # NDVI forecast samples with their split

`dir` is a folder with the layout of the release: fields.parquet, lineage.parquet, composites/composites_YYYY.parquet,
labels/, optical_observations/, radar_observations/, validation/, benchmark/. Optical values of gap-filled periods
(n_optical_obs = 0) are interpolated from the next observation; `composites(observed_only=True)`, the default, sets
them to NaN.
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


# --- lineage, validation and benchmark tables ---------------------------------------------------------------------

STATUSES = ("unchanged", "reshaped", "new", "removed")


def _fpc_id(fid):
    return [f"FPC{int(i):06d}" for i in fid]


def lineage(dir, fields=None, parents=None, status=None):
    """Field identifiers across versions: the version 1.0 field each field of this version comes from.

    status is "unchanged" (same pixels and identifier as in version 1.0), "reshaped" (the largest part of a
    version 1.0 field, which keeps its identifier), "new" (a further part of a version 1.0 field, or a tier C field,
    which has no parent) or "removed" (a version 1.0 identifier that is no longer used). `fields` keeps fields of
    this version, `parents` the rows that come from these version 1.0 fields (removed ones included).

    Columns: field_id (NaN for removed identifiers), fid, status, parent_field_id and parent_fid (the version 1.0
    field; NaN / <NA> when there is none), n_pixels (30 m pixels in this version) and obs_version (the version whose
    observations the field's series use).
    """
    if status is not None:
        status = [str(s) for s in np.atleast_1d(status)]
        bad = [s for s in status if s not in STATUSES]
        if bad:
            raise ValueError(f"unknown status(es): {bad}; use {list(STATUSES)}")
    x = pd.read_parquet(_need(Path(dir) / "lineage.parquet", "lineage"))
    x["fid"] = x["fid"].astype("int64")
    x["n_pixels"] = x["n_pixels"].astype("int64")
    x["parent_fid"] = x["parent_fid"].astype("Int64").mask(x["parent_fid"] == 0)
    x["field_id"] = pd.Series(_fpc_id(x["fid"]), index=x.index).where(x["status"] != "removed")
    x["parent_field_id"] = pd.Series(_fpc_id(x["parent_fid"].fillna(0)), index=x.index).where(x["parent_fid"].notna())
    if fields is not None:
        x = x[x["field_id"].isin(_ids(fields))]
    if parents is not None:
        x = x[x["parent_field_id"].isin(_ids(parents))]
    if status is not None:
        x = x[x["status"].isin(status)]
    cols = ["field_id", "fid", "status", "parent_field_id", "parent_fid", "n_pixels", "obs_version"]
    return x.sort_values("fid")[cols].reset_index(drop=True)


VALIDATION_TABLES = pd.DataFrame([
    ("fluccs_by_field", "fluccs_by_field.csv",
     "30 m pixels of each field in each FLUCCS land-use group, and the water management district"),
    ("worldcover_by_field", "worldcover_by_field.csv", "share of each field in each ESA WorldCover 2021 class"),
    ("coverage_by_year", "coverage_by_year.csv",
     "median over fields of the usable optical days and the radar days, by year"),
    ("quality_by_stratum", "quality_by_stratum.csv",
     "observed, interpolated and empty shares and usable days by tier and field size"),
    ("delineation_accounting", "delineation_accounting.csv", "area at each step of the field delineation"),
    ("summary", "summary.json", "every number of the validation section (a dict)"),
], columns=["table", "file", "description"])


def _read_file(p):
    p = Path(p)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def validation(dir, table=None, fields=None):
    """The tables behind the validation of the data set (Sect. 5 of the paper); table=None lists them.

    fluccs_by_field and worldcover_by_field have one row per field, with field_id first; `fields` filters them.
    In coverage_by_year the year column is named `year` (`day` in the file), and the size classes of
    quality_by_stratum are plain text ("2-5 ha", ">=100 ha"). summary is a dict.
    """
    sub = _need(Path(dir) / "validation", "validation")
    tabs = VALIDATION_TABLES
    if table is None:
        return tabs.copy()
    if table not in set(tabs["table"]):
        raise ValueError(f"unknown validation table {table!r}; see read.validation(dir)")
    per_field = table in ("fluccs_by_field", "worldcover_by_field")
    if fields is not None and not per_field:
        raise ValueError("`fields` applies only to the per-field tables")
    x = _read_file(_need(sub / tabs.set_index("table").at[table, "file"], "validation"))
    if per_field:
        x["fid"] = x["fid"].astype("int64")
        x.insert(0, "field_id", _fpc_id(x["fid"]))
        x = x[["field_id", "fid", *[c for c in x.columns if c not in ("field_id", "fid")]]]
        if fields is not None:
            x = x[x["field_id"].isin(_ids(fields))]
        x = x.sort_values("fid").reset_index(drop=True)
    if table == "coverage_by_year":
        x = x.rename(columns={"day": "year"})
    if table == "quality_by_stratum":
        x["size"] = x["size"].str.replace("--", "-", regex=False).str.replace("$\\geq$", ">=", regex=False)
    return x


BENCHMARK_DESCRIPTIONS = {
    "task1_samples": "field, issue date and horizon of every NDVI forecast sample, with the split",
    "task1_results": "RMSE, MAE, bias, R2 and skill by split, horizon and model, with 95% intervals",
    "task1_lightgbm_importance": "gain of every LightGBM feature",
    "task1_run_info": "code version, dates and sample counts of the run (a dict)",
    "task2_events": "events with their kind, date, name and numbers of responding and other fields",
    "task2_results": "AUROC, AUPRC and F1 by experiment, event and model",
    "task2_run_info": "code version and dates of the run (a dict)",
    "task3_samples": "county, week, target and split of every drought-transition sample",
    "task3_predictions": "predictions of every model for the test samples",
    "task3_results": "macro F1, balanced accuracy and recall per class by split and model",
    "task3_field_evidence_gain": "AUROC gain from field evidence, with 95% interval",
    "task3_run_info": "code version, dates and dropped samples of the run (a dict)",
    "benchmark_text": "every number of the benchmark section (a dict)",
}


def _benchmark_description(table):
    if table in BENCHMARK_DESCRIPTIONS:
        return BENCHMARK_DESCRIPTIONS[table]
    if table.startswith("task3_confusion_"):
        return f"confusion matrix of {table[len('task3_confusion_'):]} (rows observed, columns predicted)"
    return ""


def benchmark(dir, table=None, fields=None):
    """The samples, scores, predictions and run information of the three benchmark tasks; table=None lists them.

    Task 1 forecasts the field NDVI 15 to 60 days ahead, task 2 detects the fields that responded to an extreme
    event, task 3 forecasts the change of the county drought class. `table` is a file name without extension.
    task1_samples stores each sample as positions in cube.zarr (f, t, h), which are kept; field_id, issue_date,
    horizon_days, target_start and target_end are added from fields.parquet, and `fields` filters the samples (12
    million rows in all). The task 3 confusion matrices have the observed class in rows (column `observed`) and the
    predicted class in columns. The run_info and benchmark_text files are dicts.
    """
    sub = _need(Path(dir) / "benchmark", "benchmark")
    files = sorted(p.name for p in sub.iterdir() if p.suffix in (".csv", ".parquet", ".json"))
    stems = [Path(f).stem for f in files]
    tabs = pd.DataFrame({
        "table": stems,
        "task": pd.array([int(m.group(1)) if (m := re.match(r"task(\d)", s)) else None for s in stems],
                         dtype="Int64"),
        "file": files,
        "description": [_benchmark_description(s) for s in stems]})
    if table is None:
        return tabs
    if table not in stems:
        raise ValueError(f"unknown benchmark table {table!r}; see read.benchmark(dir)")
    if fields is not None and table != "task1_samples":
        raise ValueError("`fields` applies only to task1_samples")
    path = sub / files[stems.index(table)]
    if table == "task1_samples":
        return _task1_samples(dir, path, fields)
    x = _read_file(path)
    if table.startswith("task3_confusion_"):
        x = x.rename(columns={x.columns[0]: "observed"})
    if table == "task1_lightgbm_importance":
        x.columns = ["feature", "gain"]
    if table == "task2_events":
        x["date"] = pd.to_datetime(x["date"])
    return x


ORIGIN = pd.Timestamp("2016-01-01")   # the 5-day periods of the cube are anchored on 1 January 2016


def period_start(t):
    """First day of cube period t (0-based): 1 January 2016 + 5 t days."""
    return ORIGIN + pd.to_timedelta(5 * np.asarray(t, dtype="int64"), unit="D")


def _task1_samples(dir, path, fields):
    # cube.zarr has the fields in the order of fields.parquet; f is the 0-based row
    ids = np.asarray(pq.read_table(_need(Path(dir) / "fields.parquet", "fields"), columns=["field_id"])
                     .column("field_id").to_pylist(), dtype=object)
    expr = None
    if fields is not None:
        pos = pd.Index(ids).get_indexer(_ids(fields))
        expr = ds.field("f").isin([int(i) for i in pos[pos >= 0]])
    x = ds.dataset(str(path)).to_table(filter=expr).to_pandas()
    f, t, h = (x[k].to_numpy(dtype="int64") for k in ("f", "t", "h"))
    if len(f) and f.max() >= len(ids):
        raise ValueError(f"task1_samples does not match fields.parquet in {dir}")
    target = period_start(t + h - 1)
    out = pd.DataFrame({"field_id": ids[f], "issue_date": period_start(t), "horizon_days": 5 * h,
                        "target_start": target, "target_end": target + pd.Timedelta(days=4),
                        "split": x["split"].to_numpy(), "f": f, "t": t, "h": h})
    return out
