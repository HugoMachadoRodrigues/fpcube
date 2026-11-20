"""Scene-level geometric check for HLS granules whose field window was too small for the shift estimate.

During extraction (fpcube.sources.optical) the shift of each granule is estimated over the window that holds the
clear field pixels, which needs at least 64 x 64 pixels; for about a fifth of the records that window was smaller and
the record is 'not_assessed'. Here the shift of such a granule is estimated once for the whole scene: the NIR band is
read over the 512 x 512 block of the tile with the most clear pixels (Fmask clear, not water) that are also clear in
the tile's reference scene, and phase-correlated with the same reference the extraction used (the Tile class picks
it deterministically). The estimate applies to every field of that granule; estimates with a correlation peak below
fpcube.cube.MIN_SCENE_PEAK are not used (the granule stays not assessed).

python -m fpcube.sources.scene_shift             # all granules with not-assessed records (resumable)
python -m fpcube.sources.scene_shift --check 200 # compare with the field-window estimate on assessed granules

Output: config.INTERIM/optical/scene_shift/<tile>.parquet with item, shift_row, shift_col, shift_peak,
clear_frac (share of clear pixels in the block) and the block origin.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.windows import Window

from fpcube import config
from fpcube.sources import optical as O

BLOCK = 512
OUT = O.OUT / "scene_shift"


def best_block(clear, ref_ok, block=BLOCK, step=128):
    """Origin (row, col) of the block x block window with the most pixels clear in both scenes, and that share."""
    both = (clear & ref_ok).astype(np.int32)
    ii = np.pad(both.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    n0, n1 = both.shape
    best, frac = (0, 0), -1.0
    for r in range(0, max(n0 - block, 0) + 1, step):
        for c in range(0, max(n1 - block, 0) + 1, step):
            s = ii[r + block, c + block] - ii[r, c + block] - ii[r + block, c] + ii[r, c]
            if s / block ** 2 > frac:
                best, frac = (r, c), s / block ** 2
    return best, frac


def scene_shift(tile, it):
    fm, _ = O._read(it.hrefs["Fmask"], it.host)
    clear = O.clear_mask(fm) & (((fm.astype(np.uint16) >> 5) & 1) == 0)
    (r0, c0), frac = best_block(clear, np.isfinite(tile.ref_nir))
    out = {"item": it.id, "block_row": r0, "block_col": c0, "clear_frac": frac}
    if frac < 0.3:                                          # phase_shift needs 30 % valid pixels
        return {**out, "shift_row": np.nan, "shift_col": np.nan, "shift_peak": np.nan}
    win = Window(c0, r0, BLOCK, BLOCK)
    nir, _ = O._read(it.hrefs[O.BANDS[it.sensor]["nir"]], it.host, window=win)
    nir = np.where(clear[r0:r0 + BLOCK, c0:c0 + BLOCK] & (nir != -9999), nir * 1e-4, np.nan).astype(np.float32)
    dy, dx, pk = O.phase_shift(tile.ref_nir[r0:r0 + BLOCK, c0:c0 + BLOCK], nir)
    return {**out, "shift_row": dy, "shift_col": dx, "shift_peak": pk}


def targets(rel, check=0, seed=3):
    """Items to estimate: those whose records are all not assessed, or (check) a sample of assessed items with
    their field-window shift for comparison."""
    parts = []
    for p in sorted((rel / "optical_observations").glob("*.parquet")):
        d = pd.read_parquet(p, columns=["item", "tile", "qa_geometry_status", "shift_row", "shift_col"])
        parts.append(d.drop_duplicates("item"))
    d = pd.concat(parts, ignore_index=True)
    if check:
        a = d[d.qa_geometry_status != "not_assessed"]
        return a.sample(check, random_state=seed)
    return d[d.qa_geometry_status == "not_assessed"]


def main(check=0, version="v1.0"):
    rel = config.RELEASE / version
    want = targets(rel, check)
    items = pd.read_parquet(O.OUT / "items.parquet")
    items["hrefs"] = items.hrefs.map(json.loads)
    items = items[items.id.isin(set(want.item))]
    fields = gpd.read_parquet(config.INTERIM / "fields" / "fields.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    rows_check = []
    for tname, g in items.groupby("tile"):
        path = OUT / (f"check_{tname}.parquet" if check else f"{tname}.parquet")
        done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["item"])
        todo = g[~g.id.isin(set(done.item))]
        if len(todo):
            all_items = pd.read_parquet(O.OUT / "items.parquet")
            all_items["hrefs"] = all_items.hrefs.map(json.loads)
            tile = O.Tile(tname, all_items[all_items.tile == tname], fields)   # same reference as the extraction
            t0, res, fails = time.time(), [], 0

            def one(it):
                try:
                    return scene_shift(tile, it)
                except Exception as e:                      # logged; the item is retried on the next run
                    return {"item": it.id, "error": str(e)[:200]}
            with ThreadPoolExecutor(O.WORKERS) as ex:
                for k, r in enumerate(ex.map(one, todo.itertuples(index=False))):
                    if "error" in r:
                        fails += 1
                        continue
                    res.append(r)
                    if len(res) % 200 == 0:
                        pd.concat([done, pd.DataFrame(res)], ignore_index=True).to_parquet(path, index=False)
            done = pd.concat([done, pd.DataFrame(res)], ignore_index=True)
            done.to_parquet(path, index=False)
            print(f"{tname}: {len(res)} items in {(time.time() - t0) / 60:.1f} min, {fails} failed", flush=True)
        if check:
            rows_check.append(done)
    if check:
        c = pd.concat(rows_check).merge(want[["item", "shift_row", "shift_col"]], on="item", suffixes=("", "_field"))
        c["diff_px"] = np.hypot(c.shift_row - c.shift_row_field, c.shift_col - c.shift_col_field)
        c.to_csv(O.OUT / "scene_shift_check.csv", index=False)
        ok = c.diff_px.notna()
        print(json.dumps({"n": int(len(c)), "estimated": int(ok.sum()),
                          "diff_px_median": float(c.diff_px[ok].median()),
                          "diff_px_p90": float(c.diff_px[ok].quantile(0.9)),
                          "same_pass_fail": float(((np.hypot(c.shift_row, c.shift_col) > 1) ==
                                                   (np.hypot(c.shift_row_field, c.shift_col_field) > 1))[ok].mean())},
                         indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(check=int(a[a.index("--check") + 1]) if "--check" in a else 0)
