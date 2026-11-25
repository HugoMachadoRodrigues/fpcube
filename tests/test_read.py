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
