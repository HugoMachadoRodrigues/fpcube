"""fpcube.read on a tiny synthetic release (two fields, two years)."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from fpcube import read


@pytest.fixture
def rel(tmp_path):
    f = gpd.GeoDataFrame({"field_id": ["FPC000001", "FPC000002"], "fid": [1, 2], "tier": ["A", "C"],
                          "county": ["Hardee County", "DeSoto County"], "county_fips": ["12049", "12027"],
                          "area_ha": [10.0, 20.0]},
                         geometry=[box(0, 0, 300, 300), box(1000, 0, 1500, 400)], crs=5070)
    f.to_parquet(tmp_path / "fields.parquet")
    meta = {"variables": [{"name": "ndvi", "group": "optical", "units": "1", "description": "NDVI"},
                          {"name": "n_optical_obs", "group": "count", "units": "1", "description": "n"},
                          {"name": "gap_filled", "group": "count", "units": "", "description": "gap"},
                          {"name": "n_radar_obs", "group": "count", "units": "1", "description": "n"},
                          {"name": "vv_db", "group": "radar", "units": "dB", "description": "VV"}]}
    (tmp_path / "composites").mkdir()
    for y in (2021, 2022):
        dates = pd.date_range(f"{y}-01-01", periods=73, freq="5D").date
        rows = []
        for fid, ident in ((1, "FPC000001"), (2, "FPC000002")):
            for i, dt in enumerate(dates):
                obs = i % 3 != 0
                rows.append({"field_id": ident, "fid": fid, "date": dt, "ndvi": 0.5 + 0.1 * np.sin(i / 10) + fid / 100,
                             "n_optical_obs": int(obs), "gap_filled": not obs, "n_radar_obs": 1,
                             "vv_db": -10.0 - fid})
        t = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
        t = t.replace_schema_metadata({"fpcube": json.dumps(meta)})
        pq.write_table(t, tmp_path / "composites" / f"composites_{y}.parquet")
    (tmp_path / "labels").mkdir()
    weeks = pd.date_range("2020-12-29", "2023-01-02", freq="7D")
    u = pd.concat([pd.DataFrame({"county_fips": c, "county_name": n, "valid_start": weeks.date,
                                 "valid_end": (weeks + pd.Timedelta(days=6)).date, "None": 100.0, "D0": 0.0,
                                 "D1": 0.0, "D2": 0.0, "D3": 0.0, "D4": 0.0, "dominant_class": k, "dsci": 0.0})
                   for c, n, k in (("12049", "Hardee County", 2), ("12027", "DeSoto County", 0))])
    u.to_parquet(tmp_path / "labels" / "usdm_county_weekly.parquet", index=False)
    return tmp_path


def test_fields(rel):
    f = read.fields(rel)
    assert f.crs.to_epsg() == 5070 and list(f.field_id) == ["FPC000001", "FPC000002"]
    assert list(read.fields(rel, tier="c", geometry=False).field_id) == ["FPC000002"]
    assert list(read.fields(rel, county="hardee", geometry=False).field_id) == ["FPC000001"]
    with pytest.raises(ValueError):
        read.fields(rel, columns=["nope"])


def test_composites_mask_and_filters(rel):
    raw = read.composites(rel, observed_only=False)
    x = read.composites(rel)
    assert len(x) == 2 * 2 * 73
    assert x.loc[x.n_optical_obs == 0, "ndvi"].isna().all()
    assert (x.vv_db == raw.vv_db).all()                     # radar untouched
    y = read.composites(rel, fields=["FPC000002"], vars=["ndvi"], start="2022-03-01", end="2022-03-31")
    assert y.field_id.unique().tolist() == ["FPC000002"]
    assert y.date.min() >= pd.Timestamp("2022-03-01") and y.date.max() <= pd.Timestamp("2022-03-31")
    long = read.composites(rel, fields="FPC000001", vars=["ndvi", "vv_db"], years=[2021], long=True)
    assert len(long) == 73 * 2 and set(long.variable) == {"ndvi", "vv_db"}
    with pytest.raises(ValueError):
        read.composites(rel, vars=["nope"])
    with pytest.raises(FileNotFoundError):
        read.composites(rel, years=[2016])


def test_variables(rel):
    v = read.variables(rel)
    assert v.set_index("name").loc["vv_db", "units"] == "dB"


def test_drought_and_anomaly(rel):
    x = read.add_drought(rel, read.composites(rel, vars=["ndvi"]))
    assert set(x.loc[x.field_id == "FPC000001", "usdm_class"]) == {2}
    assert set(x.loc[x.field_id == "FPC000002", "usdm_class"]) == {0}
    a = read.anomaly(read.composites(rel, vars=["ndvi"]), "ndvi", ref_years=[2021, 2022], min_years=1)
    ok = a.anomaly.notna()
    assert np.allclose(a.loc[ok, "anomaly"], a.loc[ok, "ndvi"] - a.loc[ok, "clim"])
    # identical years: the anomaly of every observed value is zero
    assert np.allclose(a.loc[ok, "anomaly"], 0)
    assert read.period_of_year(pd.to_datetime(["2022-01-01", "2022-01-06", "2022-12-31"])).tolist() == [1, 2, 73]


@pytest.fixture
def tables(rel):
    """Lineage, validation and benchmark files laid out as in a release, for the two fields of `rel`."""
    pd.DataFrame({"fid": [1, 2, 3, 5], "parent_fid": [1, 0, 3, 1],
                  "status": ["reshaped", "new", "removed", "new"], "n_pixels": [120, 230, 0, 40],
                  "obs_version": ["v1.1", "v1.3", None, "v1.1"]}).to_parquet(rel / "lineage.parquet", index=False)
    v = rel / "validation"
    v.mkdir()
    pd.DataFrame({"pasture": [100, 200], "crops": [20, 0], "no_reference": [0, 30], "fid": [2, 1],
                  "district": ["SWFWMD", "SWFWMD"]}).to_csv(v / "fluccs_by_field.csv", index=False)
    pd.DataFrame({"grassland": [0.9, 0.6], "cropland": [0.1, 0.4], "fid": [1, 2]}).to_csv(
        v / "worldcover_by_field.csv", index=False)
    pd.DataFrame({"day": [2021, 2022], "usable_optical_days_median_field": [30.0, 35.0],
                  "radar_days_median_field": [25.0, 26.0]}).to_csv(v / "coverage_by_year.csv", index=False)
    pd.DataFrame({"tier": ["A", "A"], "size": ["2--5 ha", "$\\geq$100 ha"], "observed": [0.4, 0.5]}).to_csv(
        v / "quality_by_stratum.csv", index=False)
    pd.DataFrame({"step": ["Persistent pasture", "Removed: woody"], "area_ha": [100.0, -10.0],
                  "n": [None, 3]}).to_csv(v / "delineation_accounting.csv", index=False)
    (v / "summary.json").write_text(json.dumps({"n_fields": 2}))
    b = rel / "benchmark"
    b.mkdir()
    pd.DataFrame({"f": [0, 0, 1, 1], "t": [12, 300, 12, 400], "h": [3, 6, 9, 12],
                  "split": ["train", "val", "test", "test_spatial"]}).to_parquet(b / "task1_samples.parquet", index=False)
    pd.DataFrame({"task": [1], "split": ["test"], "horizon_days": [30], "model": ["lightgbm"],
                  "rmse": [0.05]}).to_csv(b / "task1_results.csv", index=False)
    pd.Series({"ndvi_last": 5.0, "doy_cos": 2.0}).to_csv(b / "task1_lightgbm_importance.csv")
    pd.DataFrame([[3, 1, 0], [1, 5, 1], [0, 2, 4]], index=["better", "same", "worse"],
                 columns=["better", "same", "worse"]).to_csv(b / "task3_confusion_momentum.csv")
    pd.DataFrame({"event_id": ["hurricane_AL042024"], "kind": ["hurricane"], "date": ["2024-08-05"],
                  "name": ["DEBBY"], "pos": [10], "neg": [5]}).to_csv(b / "task2_events.csv", index=False)
    (b / "task1_run_info.json").write_text(json.dumps({"code": "test"}))
    (b / "benchmark_text.json").write_text(json.dumps({"n": 1}))
    return rel


def test_lineage(tables):
    l = read.lineage(tables)
    assert list(l.columns) == ["field_id", "fid", "status", "parent_field_id", "parent_fid", "n_pixels",
                               "obs_version"]
    assert l.fid.tolist() == [1, 2, 3, 5]
    assert l.field_id.isna().tolist() == [False, False, True, False]         # removed identifiers have none
    assert l.field_id[0] == "FPC000001" and l.parent_field_id[3] == "FPC000001"
    assert pd.isna(l.parent_fid[1]) and pd.isna(l.parent_field_id[1])          # tier C: no version 1.0 parent
    assert read.lineage(tables, fields=read.fields(tables, tier="C")).field_id.tolist() == ["FPC000002"]
    assert read.lineage(tables, parents="FPC000001").fid.tolist() == [1, 5]
    assert read.lineage(tables, status=["new", "removed"]).fid.tolist() == [2, 3, 5]
    with pytest.raises(ValueError, match="unknown status"):
        read.lineage(tables, status="gone")


def test_validation(tables):
    v = read.validation(tables)
    assert v.table.tolist()[-1] == "summary" and len(v) == 6
    fl = read.validation(tables, "fluccs_by_field")
    assert fl.columns[:2].tolist() == ["field_id", "fid"] and fl.field_id.tolist() == ["FPC000001", "FPC000002"]
    assert fl.loc[fl.field_id == "FPC000002", "pasture"].item() == 100        # rows follow fid, not file order
    w = read.validation(tables, "worldcover_by_field", fields=["FPC000002"])
    assert len(w) == 1 and w.grassland.item() == 0.6
    assert "year" in read.validation(tables, "coverage_by_year").columns
    assert read.validation(tables, "quality_by_stratum")["size"].tolist() == ["2-5 ha", ">=100 ha"]
    assert read.validation(tables, "summary") == {"n_fields": 2}
    with pytest.raises(ValueError, match="unknown validation table"):
        read.validation(tables, "nope")
    with pytest.raises(ValueError, match="per-field"):
        read.validation(tables, "summary", fields="FPC000001")


def test_benchmark(tables):
    b = read.benchmark(tables)
    assert b.set_index("table").at["task2_events", "task"] == 2
    assert pd.isna(b.set_index("table").at["benchmark_text", "task"])
    assert (b.description != "").all()
    s = read.benchmark(tables, "task1_samples")
    assert s.field_id.tolist() == ["FPC000001", "FPC000001", "FPC000002", "FPC000002"]
    assert s.issue_date[0] == pd.Timestamp("2016-03-01")                      # period 12 of the 2016 grid
    assert (s.horizon_days == 5 * s.h).all()
    assert ((s.target_start - s.issue_date).dt.days == s.horizon_days - 5).all()
    assert ((s.target_end - s.target_start).dt.days == 4).all()
    one = read.benchmark(tables, "task1_samples", fields=read.fields(tables, tier="C"))
    assert one.field_id.unique().tolist() == ["FPC000002"] and one.f.tolist() == [1, 1]
    assert read.benchmark(tables, "task3_confusion_momentum").columns.tolist() == ["observed", "better", "same",
                                                                                   "worse"]
    assert read.benchmark(tables, "task1_lightgbm_importance").columns.tolist() == ["feature", "gain"]
    assert read.benchmark(tables, "task2_events").date.dtype.kind == "M"
    assert read.benchmark(tables, "task1_run_info") == {"code": "test"}
    with pytest.raises(ValueError, match="unknown benchmark table"):
        read.benchmark(tables, "nope")
    with pytest.raises(ValueError, match="task1_samples"):
        read.benchmark(tables, "task1_results", fields="FPC000001")


def test_task1_samples_must_match_fields(tables):
    pd.DataFrame({"f": [7], "t": [12], "h": [3], "split": ["train"]}).to_parquet(
        tables / "benchmark" / "task1_samples.parquet", index=False)
    with pytest.raises(ValueError, match="does not match"):
        read.benchmark(tables, "task1_samples")


def test_missing_parts(rel):
    for fn, part in ((read.lineage, "lineage"), (read.validation, "validation"), (read.benchmark, "benchmark")):
        with pytest.raises(FileNotFoundError, match=part):
            fn(rel)
