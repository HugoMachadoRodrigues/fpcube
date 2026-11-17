"""Weak labels and event tables.

- US Drought Monitor, weekly county statistics (percent area in None, D0-D4), Florida, 2016-2026.
- USDA NASS Crop Progress, weekly Florida pasture condition (percent very poor ... excellent), 2016-2026.
  Needs a free Quick Stats API key in the environment variable FPCUBE_NASS_KEY.
- Hurricanes: NOAA HURDAT2 Atlantic best tracks (through 2025) plus NHC working best tracks (b-decks) for
  2026 storms that are not yet in HURDAT2. Track points within 300 km of Florida are kept.
"""
import io
import os
import re
import sys

import numpy as np
import pandas as pd
import requests

from fpcube import config

OUT = config.RAW / "labels"
USDM = ("https://usdmdataservices.unl.edu/api/CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent"
        "?aoi=FL&startdate={a}&enddate={b}&statisticsType=1")
HURDAT2 = "https://www.nhc.noaa.gov/data/hurdat/"
NHC_BTK = "https://ftp.nhc.noaa.gov/atcf/btk/"
FL_BOX = (-87.7, 24.4, -79.9, 31.1)


def usdm():
    frames = []
    for y in config.YEARS:
        b = min(pd.Timestamp(f"{y}-12-31"), pd.Timestamp(config.END))
        r = requests.get(USDM.format(a=f"1/1/{y}", b=f"{b.month}/{b.day}/{y}"), timeout=300)
        r.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(r.text), dtype={"FIPS": str}))
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["MapDate", "FIPS"])
    d["valid_start"] = pd.to_datetime(d.ValidStart)
    d["valid_end"] = pd.to_datetime(d.ValidEnd)
    cols = ["None", "D0", "D1", "D2", "D3", "D4"]
    d[cols] = d[cols].astype(float)
    # cumulative-to-category: USDM percent-area statistics are cumulative (D0 includes D1-D4)
    cat = d[["D0", "D1", "D2", "D3", "D4"]].to_numpy()
    d["dominant_class"] = np.select([cat[:, 4] >= 50, cat[:, 3] >= 50, cat[:, 2] >= 50, cat[:, 1] >= 50,
                                     cat[:, 0] >= 50], [5, 4, 3, 2, 1], 0)
    d["dsci"] = d[["D0", "D1", "D2", "D3", "D4"]].sum(axis=1)   # Drought Severity and Coverage Index (0-500)
    d = d.rename(columns={"FIPS": "county_fips", "County": "county_name"})
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["county_fips", "county_name", "valid_start", "valid_end", "None", "D0", "D1", "D2", "D3", "D4",
            "dominant_class", "dsci"]
    d[keep].sort_values(["county_fips", "valid_start"]).to_parquet(OUT / "usdm_county_weekly.parquet", index=False)
    return d


def nass_pasture_condition():
    key = os.environ.get("FPCUBE_NASS_KEY")
    if not key:
        raise SystemExit("set FPCUBE_NASS_KEY (free key: https://quickstats.nass.usda.gov/api)")
    q = {"key": key, "source_desc": "SURVEY", "sector_desc": "CROPS", "commodity_desc": "PASTURELAND",
         "statisticcat_desc": "CONDITION", "agg_level_desc": "STATE", "state_alpha": "FL",
         "year__GE": str(config.YEARS[0]), "format": "JSON"}
    r = requests.get("https://quickstats.nass.usda.gov/api/api_GET/", params=q, timeout=300)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["data"])
    d = d[d.freq_desc == "WEEKLY"]
    d["week_ending"] = pd.to_datetime(d.week_ending)
    d["value"] = pd.to_numeric(d.Value.str.replace(",", ""), errors="coerce")
    d["class"] = d.unit_desc.str.replace("PCT ", "").str.lower().str.replace(" ", "_")
    w = d.pivot_table(index="week_ending", columns="class", values="value", aggfunc="first").reset_index()
    order = ["very_poor", "poor", "fair", "good", "excellent"]
    w = w[["week_ending"] + [c for c in order if c in w]]
    w["condition_index"] = sum(w[c] * (i + 1) for i, c in enumerate(order)) / 100   # 1 (very poor) .. 5
    w.to_parquet(OUT / "nass_fl_pasture_condition_weekly.parquet", index=False)
    return w


def _hurdat2_url():
    html = requests.get(HURDAT2, timeout=120).text
    files = sorted(set(re.findall(r"hurdat2-1851-\d{4}-\d+\.txt", html)))
    return HURDAT2 + files[-1]


def _parse_hurdat2(text):
    rows, sid, name = [], None, None
    for line in text.splitlines():
        p = [x.strip() for x in line.split(",")]
        if p[0][:2] in ("AL", "EP", "CP") and len(p) >= 3 and p[0][2:4].isdigit():
            sid, name = p[0], p[1]
            continue
        t = pd.Timestamp(f"{p[0]} {p[1][:2]}:{p[1][2:]}")
        lat = float(p[4][:-1]) * (1 if p[4][-1] == "N" else -1)
        lon = float(p[5][:-1]) * (-1 if p[5][-1] == "W" else 1)
        rows.append({"storm_id": sid, "name": name, "time": t, "record": p[2], "status": p[3], "lat": lat,
                     "lon": lon, "vmax_kt": float(p[6]), "pmin_hpa": float(p[7]), "source": "HURDAT2"})
    return pd.DataFrame(rows)


def _btk_year(year):
    html = requests.get(NHC_BTK, timeout=120).text
    rows = []
    for f in sorted(set(re.findall(rf"bal\d\d{year}\.dat", html))):
        txt = requests.get(NHC_BTK + f, timeout=120).text
        for line in txt.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 9:
                continue
            lat = int(p[6][:-1]) / 10 * (1 if p[6][-1] == "N" else -1)
            lon = int(p[7][:-1]) / 10 * (-1 if p[7][-1] == "W" else 1)
            rows.append({"storm_id": f"AL{p[1]}{year}", "name": p[27] if len(p) > 27 else "", "time":
                         pd.Timestamp(p[2][:8] + " " + p[2][8:] + ":00"), "record": "", "status": p[10],
                         "lat": lat, "lon": lon, "vmax_kt": float(p[8]), "pmin_hpa": float(p[9] or "nan"),
                         "source": "NHC b-deck (preliminary)"})
    d = pd.DataFrame(rows).drop_duplicates(["storm_id", "time"])
    if len(d):
        real = d[~d.name.str.upper().str.match(r"^(GENESIS|INVEST|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)")
                 & (d.name != "")]
        final = real.sort_values("time").groupby("storm_id").name.last()
        d["name"] = d.storm_id.map(final).fillna(d.name)
    return d


def hurricanes():
    h = _parse_hurdat2(requests.get(_hurdat2_url(), timeout=300).text)
    last = h.time.dt.year.max()
    extra = [_btk_year(y) for y in config.YEARS if y > last]
    h = pd.concat([h] + extra, ignore_index=True)
    h = h[(h.time >= config.START) & (h.time <= pd.Timestamp(config.END) + pd.Timedelta(days=1))]
    lon0, lat0, lon1, lat1 = FL_BOX
    pad = 3.0   # ~300 km
    near = (h.lon.between(lon0 - pad, lon1 + pad)) & (h.lat.between(lat0 - pad, lat1 + pad))
    ids = h.loc[near, "storm_id"].unique()
    h = h[h.storm_id.isin(ids)].sort_values(["storm_id", "time"])
    OUT.mkdir(parents=True, exist_ok=True)
    h.to_parquet(OUT / "hurricane_tracks_florida.parquet", index=False)
    landfalls = h[h.record == "L"]
    landfalls.to_parquet(OUT / "hurricane_landfalls.parquet", index=False)
    return h


if __name__ == "__main__":
    which = sys.argv[1:] or ["usdm", "hurricanes"]
    for w in which:
        print(w, len({"usdm": usdm, "hurricanes": hurricanes, "nass": nass_pasture_condition}[w]()))
