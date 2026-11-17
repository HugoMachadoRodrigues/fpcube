"""Per-field optical series from Harmonized Landsat Sentinel-2 (HLS v2.0, S30 and L30), 2016-2026.

Same product from two hosts: NASA LP DAAC (CMR granule search, Earthdata login via ~/.netrc) for the whole period and
Microsoft Planetary Computer (hls2-s30 / hls2-l30, identical HLS v2.0 files) from 2020.

For each MGRS tile that touches a field:
  1. fields are buffered inward by half a pixel and rasterized on the tile's 30 m grid (pixel centres inside);
  2. for each acquisition, Fmask is read first; a scene with no clear field pixel is logged and skipped;
  3. the six reflective bands are read only over the window that holds the clear field pixels;
  4. per field: mean reflectance and mean per-pixel indices over clear pixels, pixel counts, water fraction;
  5. geometric check: NIR of the window is cross-correlated with the tile reference (clearest 2021-2023 S30
     scene); the shift (rows, cols, in 30 m pixels) is stored with each record.

Clear pixel (HLS v2 Fmask bits): not cloud (1), not adjacent to cloud/shadow (2), not cloud shadow (3),
not snow/ice (4), not cirrus (0), aerosol level not high (6-7 != 3). Water (bit 5) is kept and reported.

Output: config.INTERIM/optical/<tile>/<year>.parquet (one row per field x acquisition) and a done-list.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import Window

from fpcube import config

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = {"S30": {"blue": "B02", "green": "B03", "red": "B04", "nir": "B8A", "swir1": "B11", "swir2": "B12"},
         "L30": {"blue": "B02", "green": "B03", "red": "B04", "nir": "B05", "swir1": "B06", "swir2": "B07"}}
PC_COL = {"S30": "hls2-s30", "L30": "hls2-l30"}
PC_START = "2020-01-01"
OUT = config.INTERIM / "optical"

GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MAX_RETRY="6", GDAL_HTTP_RETRY_DELAY="4", GDAL_HTTP_MULTIPLEX="YES",
                GDAL_HTTP_NETRC="YES",
                GDAL_HTTP_COOKIEFILE=str(config.DATA / ".gdal_cookies"),
                GDAL_HTTP_COOKIEJAR=str(config.DATA / ".gdal_cookies"))
THREADS = int(os.environ.get("FPCUBE_THREADS", "24"))


def tile_of(item_id):
    return item_id.split(".")[2]                     # HLS.S30.T17RML.2016105T155922.v2.0


CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
LP_SHORT = {"S30": "HLSS30", "L30": "HLSL30"}


def _get(url, params, headers=None, tries=10):
    import requests
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers or {}, timeout=120)
            r.raise_for_status()
            return r
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"retry {params.get('temporal', '')}: {str(e)[:80]}", flush=True)
            time.sleep(min(300, 15 * (i + 1)))


def cmr_month(sensor, bbox, a, b):
    """LP DAAC HLS granules from the CMR granule search (more robust than CMR-STAC for long harvests)."""
    params = {"short_name": LP_SHORT[sensor], "version": "2.0", "bounding_box": ",".join(map(str, bbox)),
              "temporal": f"{a}T00:00:00Z,{b}T23:59:59Z", "page_size": 2000}
    rows, after = [], None
    keep = set(BANDS[sensor].values()) | {"Fmask"}
    while True:
        r = _get(CMR, params, {"CMR-Search-After": after} if after else None)
        for g in r.json().get("items", []):
            u = g["umm"]
            gid = u["GranuleUR"]
            hrefs = {}
            for link in u.get("RelatedUrls", []):
                url = link.get("URL", "")
                if url.startswith("https://") and url.endswith(".tif"):
                    band = url.rsplit(".", 2)[-2]
                    if band in keep:
                        hrefs[band] = url
            cc = next((float(x["Values"][0]) for x in u.get("AdditionalAttributes", [])
                       if x.get("Name") == "CLOUD_COVERAGE"), None)
            t = u["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            rows.append({"id": gid, "tile": tile_of(gid), "sensor": sensor,
                         "datetime": pd.Timestamp(t).tz_convert(None), "host": "lpdaac", "cloud": cc, "hrefs": hrefs})
        after = r.headers.get("CMR-Search-After")
        if not after or not r.json().get("items"):
            return rows


def pc_month(sensor, bbox, a, b, cat):
    for i in range(10):
        try:
            its = list(cat.search(collections=[PC_COL[sensor]], bbox=bbox, datetime=f"{a}/{b}", limit=250).items())
            break
        except Exception as e:
            if i == 9:
                raise
            print(f"search retry {sensor} {a}: {str(e)[:80]}", flush=True)
            time.sleep(min(300, 15 * (i + 1)))
    return [{"id": it.id, "tile": tile_of(it.id), "sensor": sensor,
             "datetime": pd.Timestamp(it.datetime).tz_convert(None), "host": "pc",
             "cloud": it.properties.get("eo:cloud_cover"),
             "hrefs": {k: v.href for k, v in it.assets.items() if k in set(BANDS[sensor].values()) | {"Fmask"}}}
            for it in its]


def search_items(bbox, start=config.START, end=config.END):
    """All S30/L30 items over bbox: LP DAAC (CMR) before 2020 and Planetary Computer after.
    Each month is cached under OUT/catalog_months so an interrupted harvest resumes."""
    import planetary_computer as pc
    import pystac_client
    cat = pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)
    cache = OUT / "catalog_months"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    # LP DAAC (the HLS archive of record) for the whole period, Planetary Computer from 2020; the PC mirror misses
    # ~20 % of granules (and most of March 2026), so the catalog is their union, read from PC when PC has it
    for sensor in ("S30", "L30"):
        for host, a0, b0 in (("lpdaac", start, end), ("pc", PC_START, end)):
            for m0 in pd.date_range(a0, b0, freq="MS"):
                p = cache / f"{sensor}_{host}_{m0:%Y-%m}.parquet"
                if not p.exists():
                    m1 = min(m0 + pd.offsets.MonthEnd(0), pd.Timestamp(b0))
                    a, b = m0.strftime("%Y-%m-%d"), m1.strftime("%Y-%m-%d")
                    rows = cmr_month(sensor, bbox, a, b) if host == "lpdaac" else pc_month(sensor, bbox, a, b, cat)
                    df = pd.DataFrame(rows, columns=["id", "tile", "sensor", "datetime", "host", "cloud", "hrefs"])
                    df["hrefs"] = df.hrefs.map(json.dumps)
                    df.to_parquet(p, index=False)
                    print(f"catalog {sensor} {host} {m0:%Y-%m}: {len(df)} items", flush=True)
                frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    df["_pref"] = (df.host != "pc").astype(int)                       # prefer the PC copy (faster, same file)
    df = df.sort_values(["id", "_pref"]).drop_duplicates("id").drop(columns="_pref")
    df["hrefs"] = df.hrefs.map(json.loads)
    return df.sort_values(["tile", "datetime"]).reset_index(drop=True)


def _open(href, host):
    if host == "pc":
        import planetary_computer as pc
        href = pc.sign(href.split("?")[0])   # cached tokens expire; sign fresh
    return rasterio.open(href)


def _read(href, host, window=None, retries=4):
    for i in range(retries):
        try:
            with rasterio.Env(**GDAL_ENV), _open(href, host) as s:
                return s.read(1, window=window), s
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(10 * (i + 1))


def clear_mask(fm):
    fm = fm.astype(np.uint16)
    bad = ((fm >> 1) & 1) | ((fm >> 2) & 1) | ((fm >> 3) & 1) | ((fm >> 4) & 1) | (fm & 1)
    aero_high = ((fm >> 6) & 3) == 3
    return (bad == 0) & ~aero_high & (fm != 255)


def phase_shift(a, b):
    """Shift (drow, dcol) of b against a by phase correlation with sub-pixel parabolic refinement."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.mean() < 0.3:
        return np.nan, np.nan, np.nan
    a = np.where(m, a - np.nanmean(a[m]), 0.0)
    b = np.where(m, b - np.nanmean(b[m]), 0.0)
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.conj(np.fft.fft2(a * w)) * np.fft.fft2(b * w)
    R = np.fft.ifft2(F / (np.abs(F) + 1e-12)).real
    i, j = np.unravel_index(np.argmax(R), R.shape)
    peak = R[i, j]

    def sub(r, k, n):
        y0, y1, y2 = r[(k - 1) % n], r[k], r[(k + 1) % n]
        d = (y0 - 2 * y1 + y2)
        return 0.0 if d == 0 else 0.5 * (y0 - y2) / d
    di = i + sub(R[:, j], i, R.shape[0])
    dj = j + sub(R[i, :], j, R.shape[1])
    if di > R.shape[0] / 2:
        di -= R.shape[0]
    if dj > R.shape[1] / 2:
        dj -= R.shape[1]
    return float(di), float(dj), float(peak)


class Tile:
    def __init__(self, tile, items, fields):
        self.tile = tile
        self.items = items
        self.dir = OUT / tile
        self.dir.mkdir(parents=True, exist_ok=True)
        ref = items[(items.sensor == "S30")].copy()
        ref = ref[(ref.datetime >= "2021-01-01") & (ref.datetime < "2024-01-01")].sort_values("cloud")
        cands = ref.head(40) if len(ref) else items.sort_values("cloud").head(40)
        first = cands.iloc[0]
        fm, src = _read(first.hrefs["Fmask"], first.host)
        self.crs, self.transform, self.shape = src.crs, src.transform, fm.shape
        from shapely.geometry import box
        b = rasterio.transform.array_bounds(self.shape[0], self.shape[1], self.transform)
        f = fields.to_crs(self.crs)
        f = f[f.intersects(box(*b))]                    # array_bounds -> (west, south, east, north)
        inner = f.geometry.buffer(-15)
        inner = inner[~inner.is_empty]
        self.labels = features.rasterize(zip(inner, f.loc[inner.index, "fid"]), out_shape=self.shape,
                                         transform=self.transform, fill=0, dtype="int32")
        self.has = self.labels > 0
        ids = np.unique(self.labels[self.has])
        self.fids = ids
        self.lut = np.zeros(self.labels.max() + 1, np.int64)
        self.lut[ids] = np.arange(len(ids))
        self.idx = self.lut[self.labels]
        # reference: walk the 2021-2023 S30 scenes from least cloudy and take the first one whose clear pixels cover
        # >= 60 % of the field pixels (else the best of the first 40). Cloud percentages ignore swath edges, so the
        # least cloudy scenes are often thin slivers that miss the fields.
        best, best_n, n_has = None, -1, int(self.has.sum())
        for c in cands.itertuples(index=False):
            fmc = fm if c.id == first.id else _read(c.hrefs["Fmask"], c.host)[0]
            n_ok = int((clear_mask(fmc) & self.has).sum())
            if n_ok > best_n:
                best, best_n, best_fm = c, n_ok, fmc
            if n_ok >= 0.6 * n_has:
                break
        self.ref_clear_frac = best_n / max(n_has, 1)
        self.ref_item, fm = best, best_fm
        nir = BANDS[self.ref_item.sensor]["nir"]
        r, _ = _read(self.ref_item.hrefs[nir], self.ref_item.host)
        rc = clear_mask(fm)
        self.ref_nir = np.where(rc & (r > -9999), r * 1e-4, np.nan).astype(np.float32)
        self.done_path = self.dir / "done.json"
        self.done = set(json.loads(self.done_path.read_text())) if self.done_path.exists() else set()

    def process(self, it):
        fm, _ = _read(it.hrefs["Fmask"], it.host)
        cl = clear_mask(fm) & self.has
        n_tot_px = self.has & (fm != 255)
        if not cl.any():
            return None
        rows, cols = np.nonzero(cl)
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        win = Window(c0, r0, c1 - c0, r1 - r0)
        names = BANDS[it.sensor]
        with ThreadPoolExecutor(len(names)) as ex:
            arrs = dict(zip(names, ex.map(lambda b: _read(it.hrefs[b], it.host, window=win)[0], names.values())))
        m = cl[r0:r1, c0:c1]
        for k in arrs:
            arrs[k] = np.where(arrs[k] == -9999, np.nan, arrs[k] * 1e-4).astype(np.float32)
        valid = m & np.all([np.isfinite(a) for a in arrs.values()], axis=0)
        idx = self.idx[r0:r1, c0:c1][valid]
        n = len(self.fids)
        cnt = np.bincount(idx, minlength=n)
        if cnt.sum() == 0:
            return None
        b = {k: v[valid] for k, v in arrs.items()}
        with np.errstate(divide="ignore", invalid="ignore"):
            ind = {"ndvi": (b["nir"] - b["red"]) / (b["nir"] + b["red"]),
                   "evi": 2.5 * (b["nir"] - b["red"]) / (b["nir"] + 6 * b["red"] - 7.5 * b["blue"] + 1),
                   "ndmi": (b["nir"] - b["swir1"]) / (b["nir"] + b["swir1"]),
                   "mndwi": (b["green"] - b["swir1"]) / (b["green"] + b["swir1"]),
                   "nbr": (b["nir"] - b["swir2"]) / (b["nir"] + b["swir2"])}
        ind = {k: np.clip(np.nan_to_num(v, nan=0.0), -1, 1) for k, v in ind.items()}
        out = {"cnt": cnt}
        with np.errstate(invalid="ignore"):
            for k, v in {**b, **ind}.items():
                out[k] = np.bincount(idx, weights=v, minlength=n) / cnt
            s2 = np.bincount(idx, weights=ind["ndvi"] ** 2, minlength=n) / cnt
            out["ndvi_sd"] = np.sqrt(np.maximum(s2 - out["ndvi"] ** 2, 0))
        water = ((fm >> 5) & 1).astype(bool)[r0:r1, c0:c1][valid]
        out["water_frac"] = np.bincount(idx, weights=water, minlength=n) / np.maximum(cnt, 1)
        tot = np.bincount(self.idx[n_tot_px], minlength=n)
        nir_clear = np.where(clear_mask(fm)[r0:r1, c0:c1], arrs["nir"], np.nan)
        dy, dx, pk = phase_shift(self.ref_nir[r0:r1, c0:c1], nir_clear) if (r1 - r0) >= 64 and (c1 - c0) >= 64 \
            else (np.nan, np.nan, np.nan)
        keep = cnt > 0
        df = pd.DataFrame({k: v[keep] for k, v in out.items()})
        df.insert(0, "fid", self.fids[keep])
        df["n_field_px"] = tot[keep]
        df["clear_frac"] = df.cnt / np.maximum(df.n_field_px, 1)
        df["date"] = it.datetime
        df["sensor"], df["tile"], df["item"] = it.sensor, self.tile, it.id
        df["shift_row"], df["shift_col"], df["shift_peak"] = dy, dx, pk
        return df.rename(columns={"cnt": "n_clear_px"})

    def run(self):
        todo = self.items[~self.items.id.isin(self.done)]
        buf = {}
        t0 = time.time()
        for k, it in enumerate(todo.itertuples(index=False)):
            try:
                df = self.process(it)
            except Exception as e:
                print(f"{self.tile} {it.id} ERROR {e}", flush=True)
                continue
            if df is not None and len(df):
                buf.setdefault(it.datetime.year, []).append(df)
            self.done.add(it.id)
            if (k + 1) % 25 == 0 or k == len(todo) - 1:
                self.flush(buf)
                buf = {}
                el = time.time() - t0
                print(f"{self.tile} {k + 1}/{len(todo)} items, {el / (k + 1):.1f} s/item", flush=True)
        self.flush(buf)

    def flush(self, buf):
        for y, dfs in buf.items():
            p = self.dir / f"{y}.parquet"
            new = pd.concat(dfs, ignore_index=True)
            if p.exists():
                new = pd.concat([pd.read_parquet(p), new], ignore_index=True).drop_duplicates(["fid", "item"])
            new.to_parquet(p, index=False)
        self.done_path.write_text(json.dumps(sorted(self.done)))


def main(tiles=None):
    fields = gpd.read_parquet(config.INTERIM / "fields" / "fields.parquet")
    bbox = list(fields.to_crs(4326).total_bounds)
    cat_path = OUT / "items.parquet"
    OUT.mkdir(parents=True, exist_ok=True)
    if cat_path.exists():
        items = pd.read_parquet(cat_path)
        items["hrefs"] = items.hrefs.map(json.loads)
    else:
        items = search_items(bbox)
        tmp = items.copy()
        tmp["hrefs"] = tmp.hrefs.map(json.dumps)
        tmp.to_parquet(cat_path, index=False)
    print(f"{len(items)} items over {items.tile.nunique()} tiles", flush=True)
    n_with_fields = 0
    for tile, g in items.groupby("tile"):
        if tiles and tile not in tiles:
            continue
        T = Tile(tile, g, fields)
        if not len(T.fids):
            print(f"tile {tile}: no fields", flush=True)
            continue
        n_with_fields += 1
        print(f"tile {tile}: {len(T.fids)} fields, {len(g)} items, reference {T.ref_item.id} "
              f"({100 * T.ref_clear_frac:.0f}% of field pixels clear)", flush=True)
        T.run()
    if n_with_fields == 0:
        raise RuntimeError("no MGRS tile contains any field: check field/tile geometry")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
