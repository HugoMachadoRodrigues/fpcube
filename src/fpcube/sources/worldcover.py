"""ESA WorldCover 2021 (v200, 10 m) on the 30 m field grid: share of each 30 m cell that is open land (grassland or
cropland) and tree cover, in percent (255 = no data).

Used by the v1.3 recovery (fpcube.fields.recover) to keep only the open part of official pasture that the CDL maps as
woody or wetland. The share is computed by area: the 10 m classes are summed into the 30 m cells ("sum" resampling of
a 0/1 layer and of a 1 layer, then divided), so tile and chunk edges add up exactly.

  python -m fpcube.sources.worldcover
"""
import shutil
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
import rasterio.warp
import rasterio.windows
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from fpcube import config

OUT = config.RAW / "worldcover"
TMP = Path("/private/tmp/fpcube_worldcover")
OPEN = (30, 40)          # grassland, cropland
TREE = (10,)
CHUNK = 4000             # source rows per read


def _read(s, w, tries=6):
    for k in range(tries):
        try:
            return s.read(1, window=w)
        except rasterio.errors.RasterioIOError as e:
            if k == tries - 1 or "403" in str(e):
                raise
            time.sleep(30 * (k + 1))


def _tile(href, shape, dtr, dcrs):
    """Contributions of one WorldCover tile, as (layer, r1, r2, c0, c1, array); added only once the tile is complete."""
    parts = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="120",
                      GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="10"), rasterio.open(href) as s:
        for r0 in range(0, s.height, CHUNK):
            w = Window(0, r0, s.width, min(CHUNK, s.height - r0))
            a = _read(s, w)
            if not (a > 0).any():
                continue
            st = s.window_transform(w)
            cb = rasterio.warp.transform_bounds(s.crs, dcrs, *rasterio.windows.bounds(w, s.transform))
            dw = rasterio.windows.from_bounds(*cb, transform=dtr)
            c0, r1 = max(int(np.floor(dw.col_off)) - 2, 0), max(int(np.floor(dw.row_off)) - 2, 0)
            c1 = min(int(np.ceil(dw.col_off + dw.width)) + 2, shape[1])
            r2 = min(int(np.ceil(dw.row_off + dw.height)) + 2, shape[0])
            if c1 <= c0 or r2 <= r1:
                continue
            dwt = rasterio.windows.transform(Window(c0, r1, c1 - c0, r2 - r1), dtr)
            for k, cls in (("all", None), ("open", OPEN), ("tree", TREE)):
                src = ((a > 0) if cls is None else np.isin(a, cls)).astype(np.float32)
                tmp = np.zeros((r2 - r1, c1 - c0), np.float32)
                reproject(src, tmp, src_transform=st, src_crs=s.crs, dst_transform=dwt, dst_crs=dcrs,
                          resampling=Resampling.sum, src_nodata=None, dst_nodata=None)
                parts.append((k, r1, r2, c0, c1, tmp))
    return parts


def build():
    import planetary_computer as pc
    import pystac_client
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(exist_ok=True)
    with rasterio.open(config.vdir("fields", "v1.1") / "field_labels.tif") as s:
        prof, shape, dtr, dcrs = s.profile, s.shape, s.transform, s.crs
        b = rasterio.warp.transform_bounds(dcrs, "EPSG:4326", *s.bounds)
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    items = list(cat.search(collections=["esa-worldcover"], bbox=list(b),
                            query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
    acc = {k: np.zeros(shape, np.float32) for k in ("all", "open", "tree")}
    done_p = TMP / "done.txt"                  # checkpoint: tiles already summed into TMP/acc_*.npy
    done = done_p.read_text().split() if done_p.exists() else []
    if done:
        for k in acc:
            acc[k] = np.load(TMP / f"acc_{k}.npy")
    for it in items:
        if it.id in done:
            continue
        for k_try in range(6):                 # SAS tokens expire (HTTP 403): re-sign and redo the whole tile
            try:
                parts = _tile(pc.sign(it.assets["map"].href.split("?")[0]), shape, dtr, dcrs)
                break
            except rasterio.errors.RasterioIOError as e:
                if k_try == 5:
                    raise
                print("retry", it.id, e, flush=True)
                time.sleep(30)
        for k, r1, r2, c0, c1, tmp in parts:
            acc[k][r1:r2, c0:c1] += tmp
        for k in acc:
            np.save(TMP / f"acc_{k}.npy", acc[k])
        done.append(it.id)
        done_p.write_text("\n".join(done))
        print("worldcover", it.id, flush=True)
    ok = acc["all"] > 0
    prof.update(dtype="uint8", nodata=255, count=1, compress="deflate")
    for k in ("open", "tree"):
        pct = np.full(shape, 255, np.uint8)
        pct[ok] = np.clip(np.round(100 * acc[k][ok] / acc["all"][ok]), 0, 100).astype(np.uint8)
        with rasterio.open(TMP / f"worldcover_{k}_pct.tif", "w", **prof) as d:
            d.write(pct, 1)
            d.update_tags(source="ESA WorldCover 2021 v200 (Planetary Computer esa-worldcover)",
                          classes=str(OPEN if k == "open" else TREE))
        shutil.move(str(TMP / f"worldcover_{k}_pct.tif"), str(OUT / f"worldcover_{k}_pct.tif"))
    print("cells with WorldCover", int(ok.sum()), flush=True)


if __name__ == "__main__":
    build()
