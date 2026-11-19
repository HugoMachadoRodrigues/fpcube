"""PRISM daily 4 km climate (AN81d), 2016-2026, cropped to Florida.

Variables: ppt (mm), tmean, tmin, tmax (deg C), vpdmax (hPa). Files come from the PRISM web service
(services.nacse.org), one request at a time as PRISM asks; each day is cropped to Florida and stacked into
one compressed GeoTIFF per variable and year (band = day of year).
"""
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.windows import from_bounds

from fpcube import config

URL = "https://services.nacse.org/prism/data/get/us/4km/{var}/{date}"
VARS = ("ppt", "tmean", "tmin", "tmax", "vpdmax")
FL = (-87.7, 24.4, -79.9, 31.1)
OUT = config.RAW / "prism"


def fetch_day(var, d):
    for i in range(6):
        try:
            r = requests.get(URL.format(var=var, date=d.strftime("%Y%m%d")), timeout=180)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            name = next(n for n in z.namelist() if n.endswith(".tif"))
            with rasterio.MemoryFile(z.read(name)) as mf, mf.open() as s:
                w = from_bounds(*FL, transform=s.transform).round_offsets().round_lengths()
                a = s.read(1, window=w).astype(np.float32)
                if s.nodata is not None:
                    a[a == s.nodata] = np.nan
                return a, s.window_transform(w), s.crs
        except Exception:
            if i == 5:
                raise
            time.sleep(20 * (i + 1))


def year(var, y):
    out = OUT / var / f"prism_{var}_{y}.tif"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(f"{y}-01-01", min(pd.Timestamp(f"{y}-12-31"), pd.Timestamp(config.END)), freq="D")
    stack, tr, crs = [], None, None
    for d in days:
        a, tr, crs = fetch_day(var, d)
        stack.append(a)
        time.sleep(0.5)
    arr = np.stack(stack)
    with rasterio.open(out.with_suffix(".tmp.tif"), "w", driver="GTiff", count=arr.shape[0], height=arr.shape[1],
                       width=arr.shape[2], dtype="float32", crs=crs, transform=tr, nodata=np.nan,
                       compress="deflate", predictor=3) as dst:
        dst.write(arr)
        dst.update_tags(dates=json.dumps([str(d.date()) for d in days]), variable=var, source=URL,
                        retrieved=str(pd.Timestamp.now().date()))   # values < ~6 months old are provisional
    out.with_suffix(".tmp.tif").rename(out)
    print(var, y, arr.shape, flush=True)
    return out


def main(vars_=VARS):
    for y in config.YEARS:
        for v in vars_:
            year(v, y)


if __name__ == "__main__":
    main(sys.argv[1:] or VARS)
