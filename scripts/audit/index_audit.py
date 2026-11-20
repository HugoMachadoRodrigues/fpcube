"""How often are per-pixel spectral indices undefined or outside [-1, 1] before the extraction sets them to 0 and
clips them? (sources/optical.py: np.clip(np.nan_to_num(index, nan=0), -1, 1)).

Re-reads a random sample of HLS acquisitions over windows of 256 x 256 pixels centred on random fields, applies the
same clear-pixel and valid-band masks as the extraction, and counts, per index, pixels with a zero denominator
(undefined -> 0 in v1.0) and pixels outside [-1, 1] (clipped in v1.0). Writes data/audit/index_audit.json.

python scripts/audit/index_audit.py [n_items]
"""
import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import Window

from fpcube import config
from fpcube.sources import optical as O

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "--item" else 60
HALF = 384


def audit_item(item_id, seed):
    """Counts for one acquisition (run in its own process: a crash in a native library loses one item only)."""
    items = pd.read_parquet(O.OUT / "items.parquet")
    it = items[items.id == item_id].iloc[0]
    hrefs = json.loads(it.hrefs)
    fields = gpd.read_parquet(config.INTERIM / "fields" / "fields.parquet", columns=["fid", "geometry"])
    rng = np.random.default_rng(seed)
    with rasterio.Env(**O.GDAL_ENV), O._open(hrefs["Fmask"], it.host) as s:
        crs, tr, shape = s.crs, s.transform, s.shape
    from shapely.geometry import box
    b = rasterio.transform.array_bounds(shape[0], shape[1], tr)
    f = fields.to_crs(crs)
    f = f[f.intersects(box(*b))]
    if not len(f):
        return None
    c = f.geometry.iloc[rng.integers(len(f))].representative_point()
    r0, c0 = rasterio.transform.rowcol(tr, c.x, c.y)
    r0, c0 = int(np.clip(r0 - HALF, 0, shape[0] - 2 * HALF)), int(np.clip(c0 - HALF, 0, shape[1] - 2 * HALF))
    win = Window(c0, r0, 2 * HALF, 2 * HALF)
    wtr = rasterio.windows.transform(win, tr)
    inner = f.geometry.buffer(-15)
    inner = inner[~inner.is_empty]
    lab = features.rasterize(zip(inner, f.loc[inner.index, "fid"]), out_shape=(2 * HALF, 2 * HALF),
                             transform=wtr, fill=0, dtype="int32")
    fm = O._read(hrefs["Fmask"], it.host, window=win)[0]
    cl = O.clear_mask(fm) & (lab > 0)
    if not cl.any():
        return None
    arr = {k: O._read(hrefs[b_], it.host, window=win)[0] for k, b_ in O.BANDS[it.sensor].items()}
    arr = {k: np.where(v == -9999, np.nan, v * 1e-4) for k, v in arr.items()}
    valid = cl & np.all([np.isfinite(a) for a in arr.values()], axis=0)
    b_ = {k: v[valid] for k, v in arr.items()}
    num_den = {"ndvi": (b_["nir"] - b_["red"], b_["nir"] + b_["red"]),
               "evi": (2.5 * (b_["nir"] - b_["red"]), b_["nir"] + 6 * b_["red"] - 7.5 * b_["blue"] + 1),
               "ndmi": (b_["nir"] - b_["swir1"], b_["nir"] + b_["swir1"]),
               "mndwi": (b_["green"] - b_["swir1"], b_["green"] + b_["swir1"]),
               "nbr": (b_["nir"] - b_["swir2"], b_["nir"] + b_["swir2"])}
    out = {"pixels": int(valid.sum()), "negative_any_band": int((np.stack(list(b_.values())) < 0).any(0).sum())}
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, (n_, d_) in num_den.items():
            v = n_ / d_
            out[k] = {"zero_den": int((d_ == 0).sum()), "outside": int((np.isfinite(v) & (np.abs(v) > 1)).sum())}
    return out


def main():
    import subprocess
    items = pd.read_parquet(O.OUT / "items.parquet", columns=["id", "host"])
    items = items[items.host == "pc"]
    pick = items.sample(N * 3, random_state=3).id.tolist()
    tot = {"pixels": 0, "items": 0, "negative_any_band": 0, "failed": 0}
    cnt = {k: {"zero_den": 0, "outside": 0} for k in ("ndvi", "evi", "ndmi", "mndwi", "nbr")}
    for i, iid in enumerate(pick):
        if tot["items"] >= N:
            break
        try:
            r = subprocess.run([sys.executable, __file__, "--item", iid, str(i)], capture_output=True, text=True,
                               timeout=240)
            res = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else "fail"
        except Exception:
            res = "fail"
        if res == "fail":
            tot["failed"] += 1
            continue
        if res is None:
            continue
        tot["items"] += 1
        tot["pixels"] += res["pixels"]
        tot["negative_any_band"] += res["negative_any_band"]
        for k in cnt:
            for c in cnt[k]:
                cnt[k][c] += res[k][c]
        print(f"{tot['items']}/{N} {iid} {res['pixels']} px", flush=True)
        write(tot, cnt)
    print(json.dumps(write(tot, cnt), indent=2))


def write(tot, cnt):
    out = {"n_items": tot["items"], "n_failed": tot["failed"], "n_pixels": tot["pixels"],
           "negative_any_band_frac": tot["negative_any_band"] / max(tot["pixels"], 1),
           **{k: {**v, "zero_den_frac": v["zero_den"] / max(tot["pixels"], 1),
                  "outside_frac": v["outside"] / max(tot["pixels"], 1)} for k, v in cnt.items()}}
    p = config.DATA / "audit" / "index_audit.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--item"]:
        print(json.dumps(audit_item(sys.argv[2], int(sys.argv[3]))))
    else:
        main()
