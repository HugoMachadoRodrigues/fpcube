"""Per-field Sentinel-1 backscatter (Planetary Computer 'sentinel-1-rtc', gamma0 RTC, VV and VH), 2016-2026.

Each item is read at its first overview (20 m; fields are >= 2 ha, i.e. >= 50 pixels at 20 m). Fields are
buffered inward by 20 m and rasterized on the item's grid. Per field and acquisition: mean linear gamma0
(VV, VH) and its dB value, the VH/VV ratio in dB, pixel count, and acquisition geometry (orbit direction,
relative orbit, platform). Averaging is done in linear power before conversion to dB.

Output: config.INTERIM/radar/s1/<year>/<item>.parquet, a done-list and the item catalogue.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from shapely.geometry import box, shape

from fpcube import config

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
OUT = config.INTERIM / "radar" / "s1"
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                GDAL_HTTP_MAX_RETRY="6", GDAL_HTTP_RETRY_DELAY="4", GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                GDAL_CACHEMAX=512)
WORKERS = int(os.environ.get("FPCUBE_S1_WORKERS", "6"))


def search(bbox):
    import planetary_computer as pc
    import pystac_client
    cat = pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)
    rows = []
    for y in config.YEARS:
        a, b = f"{y}-01-01", min(f"{y}-12-31", config.END)
        for it in cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{a}/{b}", limit=500).items():
            p = it.properties
            rows.append({"id": it.id, "datetime": pd.Timestamp(it.datetime).tz_convert(None),
                         "platform": p.get("platform"), "orbit_state": p.get("sat:orbit_state"),
                         "relative_orbit": p.get("sat:relative_orbit"),
                         "footprint": json.dumps(it.geometry),
                         "vv": it.assets["vv"].href if "vv" in it.assets else None,
                         "vh": it.assets["vh"].href if "vh" in it.assets else None})
    return pd.DataFrame(rows).drop_duplicates("id").sort_values("datetime").reset_index(drop=True)


def _open(href):
    import planetary_computer as pc
    return rasterio.open(pc.sign(href), overview_level=0)


BLOCK_THREADS = int(os.environ.get("FPCUBE_S1_BLOCK_THREADS", "8"))


def process(it, fields):
    """Read only the 20 m blocks that contain field pixels (in parallel, one dataset handle per thread) and
    accumulate per-field sums of linear gamma0."""
    import threading
    fp = shape(json.loads(it.footprint))
    f = fields[fields.intersects(gpd.GeoSeries([fp], crs=4326).to_crs(fields.crs).iloc[0])]
    if not len(f) or it.vv is None:
        return None
    with rasterio.Env(**GDAL_ENV), _open(it.vv) as s0:
        crs, tr, shp, nod, (bh, bw) = s0.crs, s0.transform, s0.shape, s0.nodata, s0.block_shapes[0]
    g = f.to_crs(crs)
    inner = g.geometry.buffer(-20)
    inner = inner[~inner.is_empty]
    if not len(inner):
        return None
    lab = features.rasterize(zip(inner, g.loc[inner.index, "fid"]), out_shape=shp, transform=tr, fill=0,
                             dtype="int32")
    nmax = int(lab.max()) + 1
    blocks = [(r0, c0) for r0 in range(0, shp[0], bh) for c0 in range(0, shp[1], bw)
              if lab[r0:r0 + bh, c0:c0 + bw].any()]
    local = threading.local()

    def handles():
        if not hasattr(local, "h"):
            local.env = rasterio.Env(**GDAL_ENV)
            local.env.__enter__()
            local.h = (_open(it.vv), _open(it.vh) if it.vh else None)
        return local.h

    def one(rc):
        r0, c0 = rc
        lb = lab[r0:r0 + bh, c0:c0 + bw]
        w = rasterio.windows.Window(c0, r0, lb.shape[1], lb.shape[0])
        for attempt in range(4):
            try:
                svv, svh = handles()
                vv = svv.read(1, window=w).astype(np.float32)
                vh = svh.read(1, window=w).astype(np.float32) if svh is not None else np.full_like(vv, np.nan)
                break
            except Exception:
                if attempt == 3:
                    raise
                local.__dict__.pop("h", None)
                time.sleep(5 * (attempt + 1))
        ok = (lb > 0) & np.isfinite(vv) & (vv > 0)
        if nod is not None:
            ok &= vv != nod
        idx = lb[ok]
        vh = vh[ok]
        okh = np.isfinite(vh) & (vh > 0)
        return (np.bincount(idx, minlength=nmax), np.bincount(idx, weights=vv[ok], minlength=nmax),
                np.bincount(idx[okh], minlength=nmax), np.bincount(idx[okh], weights=vh[okh], minlength=nmax))

    s_n = np.zeros(nmax); s_vv = np.zeros(nmax); s_nh = np.zeros(nmax); s_vh = np.zeros(nmax)
    with ThreadPoolExecutor(BLOCK_THREADS) as ex:
        for a, b, c, d in ex.map(one, blocks):
            s_n += a; s_vv += b; s_nh += c; s_vh += d
    ids = np.nonzero(s_n)[0]
    if not len(ids):
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        mvh = np.where(s_nh[ids] > 0, s_vh[ids] / s_nh[ids], np.nan)
        df = pd.DataFrame({"fid": ids, "n_px": s_n[ids].astype(np.int32), "vv_lin": s_vv[ids] / s_n[ids],
                           "vh_lin": mvh})
        df["vv_db"] = 10 * np.log10(df.vv_lin)
        df["vh_db"] = 10 * np.log10(df.vh_lin)
        df["vh_vv_db"] = df.vh_db - df.vv_db
    df["date"], df["item"] = it.datetime, it.id
    df["platform"], df["orbit_state"], df["relative_orbit"] = it.platform, it.orbit_state, it.relative_orbit
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fields = gpd.read_parquet(config.INTERIM / "fields" / "fields.parquet")[["fid", "geometry"]]
    cat = OUT / "items.parquet"
    if cat.exists():
        items = pd.read_parquet(cat)
    else:
        items = search(list(fields.to_crs(4326).total_bounds))
        items.to_parquet(cat, index=False)
    done_p = OUT / "done.json"
    done = set(json.loads(done_p.read_text())) if done_p.exists() else set()
    todo = items[~items.id.isin(done)]
    print(f"{len(items)} S1 items, {len(todo)} to do", flush=True)
    t0 = time.time()

    def work(it):
        try:
            df = process(it, fields)
            if df is not None and len(df):
                d = OUT / str(it.datetime.year)
                d.mkdir(exist_ok=True)
                df.to_parquet(d / f"{it.id}.parquet", index=False)
            return it.id, None
        except Exception as e:
            return it.id, str(e)

    with ThreadPoolExecutor(WORKERS) as ex:
        for k, (iid, err) in enumerate(ex.map(work, todo.itertuples(index=False))):
            if err:
                print(f"{iid} ERROR {err}", flush=True)
            else:
                done.add(iid)
            if (k + 1) % 20 == 0:
                done_p.write_text(json.dumps(sorted(done)))
                print(f"{k + 1}/{len(todo)} items, {(time.time() - t0) / (k + 1):.1f} s/item", flush=True)
    done_p.write_text(json.dumps(sorted(done)))


if __name__ == "__main__":
    main()
