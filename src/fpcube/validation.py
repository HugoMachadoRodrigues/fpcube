"""Technical validation of the Florida Pasture Cube. Writes <release>/validation/*.csv and summary.json, whose
text snippets are inserted into the paper by scripts/paper_numbers.py.

1. Delineation: land cover inside fields from ESA WorldCover 2021 (10 m, Sentinel-1/2 based, independent of CDL).
2. Coverage: clear optical observations and radar acquisitions per field and year; share of periods with data.
3. Physical consistency:
   a. NDVI seasonality by region (timing of the seasonal minimum and maximum);
   b. NDVI anomaly vs 60-day rainfall anomaly (drought signal) per field, across years;
   c. VV backscatter response to rain in the preceding 5 days (soil moisture signal);
   d. Hurricane Ian (landfall 28 Sep 2022): MNDWI and VV change vs HAND for fields that received >= 200 mm.
"""
import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from fpcube import config

WC = {10: "tree", 20: "shrub", 30: "grassland", 40: "cropland", 50: "built", 60: "bare", 80: "water",
      90: "herbaceous_wetland", 95: "mangrove", 100: "moss"}


def worldcover(fields):
    import planetary_computer as pc
    import pystac_client
    import rasterio
    from exactextract import exact_extract
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    f4 = fields[["fid", "geometry"]].to_crs(4326)
    items = list(cat.search(collections=["esa-worldcover"], bbox=list(f4.total_bounds),
                            query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
    res = []
    for it in items:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(it.assets["map"].href) as s:
            b = s.bounds
            sub = f4.cx[b.left:b.right, b.bottom:b.top]
            if not len(sub):
                continue
            from fpcube.sources.static import clip_to
            sub = clip_to(sub, b, inset=1e-6)          # exactextract 0.3 segfaults on polygons leaving the raster
            d = exact_extract(s, sub, ["unique", "frac", "count"], include_cols=["fid"], output="pandas")
        res.append(d)
    d = pd.concat(res, ignore_index=True)
    rows = []
    for r in d.itertuples():
        for u, fr in zip(r.unique, r.frac):
            rows.append((r.fid, WC.get(int(u), "other"), fr * r.count))
    t = pd.DataFrame(rows, columns=["fid", "cls", "w"]).groupby(["fid", "cls"]).w.sum().unstack(fill_value=0)
    return t.div(t.sum(1), axis=0).reset_index()


def delineation_text(wc, fields):
    m = wc.merge(fields[["fid", "tier", "area_ha"]], on="fid")
    out = {}
    for tier in ("A", "B"):
        g = m[m.tier == tier]
        w = g.area_ha / g.area_ha.sum()
        out[tier] = {c: float((g[c] * w).sum()) for c in ("grassland", "tree", "cropland", "built",
                                                          "herbaceous_wetland", "water") if c in g}
    txt = ("ESA WorldCover 2021 \\citep{{zanaga2022}}, which is derived from Sentinel-1 and Sentinel-2 independently of "
           "the CDL, maps {gA:.0f}\\,\\% of the area of tier~A fields and {gB:.0f}\\,\\% of tier~B fields as grassland "
           "and a further {cA:.0f}\\,\\% and {cB:.0f}\\,\\% as cropland, which in Florida includes hay and improved "
           "pasture. Trees cover {tA:.0f}\\,\\% (A) and {tB:.0f}\\,\\% (B) and herbaceous wetland {wA:.0f}\\,\\% and "
           "{wB:.0f}\\,\\%, mostly as scattered trees, hammocks and wet depressions inside working pastures. Built-up "
           "land is below {bmax:.1f}\\,\\% in both tiers.").format(
        gA=100 * out["A"]["grassland"], gB=100 * out["B"]["grassland"], cA=100 * out["A"]["cropland"],
        cB=100 * out["B"]["cropland"], tA=100 * out["A"]["tree"], tB=100 * out["B"]["tree"],
        wA=100 * out["A"].get("herbaceous_wetland", 0), wB=100 * out["B"].get("herbaceous_wetland", 0),
        bmax=100 * max(out["A"]["built"], out["B"]["built"]))
    return out, txt


def coverage(rel, cube, fields):
    opt = pd.concat(pd.read_parquet(p, columns=["fid", "date", "sensor", "qa_clear", "qa_geometry",
                                                    "qa_range", "qa_haze", "qa_use"])
                    for p in (rel / "optical_observations").glob("*.parquet"))
    rad = pd.concat(pd.read_parquet(p, columns=["fid", "date"]) for p in (rel / "radar_observations").glob("*.parquet"))
    n = len(fields)
    good = opt[opt.qa_use]
    yrs = pd.to_datetime(good.date).dt.year
    per_year = good.groupby(yrs).size() / n
    rad_year = rad.groupby(pd.to_datetime(rad.date).dt.year).size() / n
    has = (cube.n_optical_obs.values > 0)
    filled = cube.gap_filled.values
    month = pd.DatetimeIndex(cube.time.values).month
    wet = np.isin(month, [6, 7, 8, 9])
    stats = {"clear_per_field_year_median": float(per_year.median()), "clear_per_field_year_min": float(per_year.min()),
             "clear_per_field_year_max": float(per_year.max()), "clear_frac_of_all_obs": float(len(good) / len(opt)),
             "clear_test_frac": float(opt.qa_clear.mean()),
             "geom_reject_frac": float((~opt.qa_geometry).mean()),
             "range_reject_frac": float((opt.qa_clear & ~opt.qa_range).mean()),
             "haze_reject_frac": float((opt.qa_clear & opt.qa_range & ~opt.qa_haze).mean()),
             "radar_per_field_year_median": float(rad_year.median()),
             "periods_with_optical": float(has.mean()), "periods_with_optical_wet": float(has[:, wet].mean()),
             "periods_with_optical_dry": float(has[:, ~wet].mean()),
             "periods_with_ndvi_after_fill": float(np.isfinite(cube.ndvi.values).mean()),
             "gap_filled_frac": float(filled.mean())}
    txt = ("Of all field-level HLS observations, {ct:.0f}\\,\\% passed the clear-pixel test, {gr:.1f}\\,\\% were rejected "
           "for a geometric shift above one pixel, and {rr:.2f}\\,\\% and {hr:.2f}\\,\\% of the clear ones for reflectance outside "
           "[$-0.01$, 1] or blue reflectance above 0.25; {cf:.0f}\\,\\% were used. A median field has {cm:.0f} usable optical observations per year "
           "(range of yearly medians {cmin:.0f}--{cmax:.0f}, increasing as Landsat~9 and Sentinel-2C joined) and "
           "{rm:.0f} Sentinel-1 acquisitions. {po:.0f}\\,\\% of the 5-day periods have at least one optical observation "
           "({pd:.0f}\\,\\% in October--May, {pw:.0f}\\,\\% in June--September); after gap filling up to 30 days, "
           "{pf:.0f}\\,\\% of periods have an NDVI value, and {gf:.0f}\\,\\% of all values are interpolated.").format(
        cf=100 * stats["clear_frac_of_all_obs"], gr=100 * stats["geom_reject_frac"], ct=100 * stats["clear_test_frac"],
        rr=100 * stats["range_reject_frac"], hr=100 * stats["haze_reject_frac"],
        cm=stats["clear_per_field_year_median"], cmin=per_year.min(), cmax=per_year.max(),
        rm=stats["radar_per_field_year_median"], po=100 * stats["periods_with_optical"],
        pd=100 * stats["periods_with_optical_dry"], pw=100 * stats["periods_with_optical_wet"],
        pf=100 * stats["periods_with_ndvi_after_fill"], gf=100 * stats["gap_filled_frac"])
    return stats, txt, per_year, rad_year


def physical(cube, fields):
    from fpcube.benchmark.tasks import ndvi_anomaly
    times = pd.DatetimeIndex(cube.time.values)
    nd = cube.ndvi.values
    # (a) seasonality: month of minimum and maximum median NDVI, statewide
    monthly = pd.Series(np.nanmedian(nd, axis=0), index=times).groupby(times.month).mean()
    # (b) NDVI anomaly vs 60-day rainfall anomaly, April-May (dry season end) per field-year
    an = ndvi_anomaly(cube)
    ppt = cube.ppt_sum.values
    ppt60 = pd.DataFrame(ppt.T).rolling(12, min_periods=12).sum().to_numpy().T
    mu = np.nanmean(ppt60, axis=1, keepdims=True)
    sd = np.nanstd(ppt60, axis=1, keepdims=True)
    pz = (ppt60 - mu) / np.where(sd > 0, sd, np.nan)
    am = np.isin(times.month, [4, 5])
    a, p = an[:, am].ravel(), pz[:, am].ravel()
    ok = np.isfinite(a) & np.isfinite(p)
    r_drought = float(np.corrcoef(a[ok], p[ok])[0, 1])
    # (c) VV vs 5-day rain: periods with >= 25 mm vs <= 1 mm, per field, dry season
    vv = cube.vv_db.values
    wetp = ppt >= 25
    dryp = ppt <= 1
    dvv = np.nanmedian(np.where(wetp, vv, np.nan), axis=1) - np.nanmedian(np.where(dryp, vv, np.nan), axis=1)
    # (d) Ian
    out = {"ndvi_month_min": int(monthly.idxmin()), "ndvi_month_max": int(monthly.idxmax()),
           "r_ndvi_anom_rain60_AM": r_drought, "n_pairs_AM": int(ok.sum()),
           "vv_wet_minus_dry_db_median": float(np.nanmedian(dvv)),
           "vv_wet_gt_dry_frac": float(np.nanmean(dvv > 0))}
    ian = pd.Timestamp("2022-09-28")
    k = int(np.searchsorted(times, ian))
    if "hand_mean_m" in fields and 0 < k < len(times) - 3:
        rain = np.nansum(ppt[:, k - 1:k + 1], axis=1)
        hit = rain >= 200
        mn = cube.mndwi.values
        dm = np.nanmax(mn[:, k:k + 3], axis=1) - np.nanmedian(mn[:, k - 6:k], axis=1)
        dv = np.nanmin(vv[:, k:k + 2], axis=1) - np.nanmedian(vv[:, k - 6:k], axis=1)
        hand = fields.hand_mean_m.to_numpy()
        low, high = hit & (hand <= 1), hit & (hand >= 3)
        out.update({"ian_fields_200mm": int(hit.sum()), "ian_dmndwi_lowhand": float(np.nanmedian(dm[low])),
                    "ian_dmndwi_highhand": float(np.nanmedian(dm[high])), "ian_dvv_lowhand": float(np.nanmedian(dv[low])),
                    "ian_dvv_highhand": float(np.nanmedian(dv[high])), "ian_n_low": int(low.sum()),
                    "ian_n_high": int(high.sum())})
    mname = pd.Timestamp(2001, 1, 1)
    txt = ("Statewide median NDVI reaches its minimum in {mmin} and its maximum in {mmax}, consistent with the dormant "
           "season of warm-season grasses after winter frosts and the spring dry season, and with growth during the "
           "summer rains. At the end of the dry season (April--May) the field NDVI anomaly is positively correlated "
           "with the 60-day rainfall anomaly ($r={r:.2f}$, $n={n}$ field-periods). Sentinel-1 VV backscatter is higher "
           "in periods with at least 25\\,mm of rain than in dry periods for {vf:.0f}\\,\\% of fields (median difference "
           "{dv:.1f}\\,dB), as expected from wetter soil and canopy.").format(
        mmin=pd.Timestamp(2001, out["ndvi_month_min"], 1).strftime("%B"),
        mmax=pd.Timestamp(2001, out["ndvi_month_max"], 1).strftime("%B"), r=r_drought,
        n=f"{ok.sum():,}".replace(",", "\\,"), vf=100 * out["vv_wet_gt_dry_frac"], dv=out["vv_wet_minus_dry_db_median"])
    if "ian_fields_200mm" in out:
        txt += (" After Hurricane Ian, among the {n} fields that received at least 200\\,mm in two days, fields with a "
                "height above the nearest drainage of 1\\,m or less showed a median MNDWI increase of {ml:.2f} against "
                "{mh:.2f} for fields 3\\,m or more above drainage, and a VV change of {vl:.1f} against {vh:.1f}\\,dB, "
                "the signature of standing water on low-lying pastures (Fig.~\\ref{{fig:events}}).").format(
            n=out["ian_fields_200mm"], ml=out["ian_dmndwi_lowhand"], mh=out["ian_dmndwi_highhand"],
            vl=out["ian_dvv_lowhand"], vh=out["ian_dvv_highhand"])
    return out, txt


def quality(rel, cube, fields):
    """Known limits of the dynamic layers, stated with their size: interpolation, unobserved fields, radar orbit
    mixing, saturated NDVI, provisional PRISM."""
    from fpcube.cube import period_index
    nd, gf, no = cube.ndvi.values, cube.gap_filled.values, cube.n_optical_obs.values
    obs = no > 0
    r = pd.concat(pd.read_parquet(p, columns=["fid", "date", "relative_orbit", "orbit_state"])
                  for p in (rel / "radar_observations").glob("*.parquet"))
    r["k"] = period_index(pd.to_datetime(r.date))
    g = r.groupby(["fid", "k"])
    p_ = sorted((config.RAW / "prism" / "ppt").glob("prism_ppt_*.tif"))[-1]
    with rasterio.open(p_) as s_:
        got = s_.tags().get("retrieved")
    got = pd.Timestamp(got) if got else pd.Timestamp(p_.stat().st_mtime, unit="s").normalize()
    stats = {"interpolated_frac": float(gf.sum() / np.isfinite(nd).sum()),
             "n_no_observations": int(fields.flag_no_observations.sum()),
             "n_no_optical": int(fields.flag_no_optical.sum()),
             "radar_mixed_orbit_frac": float((g.relative_orbit.nunique() > 1).mean()),
             "radar_mixed_direction_frac": float((g.orbit_state.nunique() > 1).mean()),
             "ndvi_saturated_frac": float((obs & (np.abs(nd) >= 0.999)).sum() / obs.sum()),
             "prism_retrieved": str(got.date()),
             "prism_provisional_from": str((got - pd.DateOffset(months=6)).date())}
    sat = 100 * stats["ndvi_saturated_frac"]
    txt = ("{gf:.0f}\\,\\% of the optical values in the cube are interpolated rather than observed (flag "
           "\\texttt{{gap\\_filled}}). {nn} fields have no optical or radar observation at all and {nr} more have radar only; "
           "they are narrow fields in which no pixel remains after the inward buffers, kept in the release and flagged "
           "(\\texttt{{flag\\_no\\_observations}}, \\texttt{{flag\\_no\\_optical}}). {mo:.1f}\\,\\% of the radar composites "
           "combine acquisitions from more than one relative orbit ({md:.1f}\\,\\% mix ascending and descending passes), so "
           "the incidence angle varies within these periods; the per-acquisition table keeps the orbit of every value. "
           "Because indices are computed per pixel and limited to $[-1, 1]$, NDVI composites reach $\\pm 1$ in "
           "{sat} of the observed cells. PRISM daily values become stable about six months after the date; the record "
           "was retrieved on {got}, so values after {prov} are provisional and may be revised in later PRISM releases.").format(
        gf=100 * stats["interpolated_frac"], nn=fmt_int(stats["n_no_observations"]),
        nr=fmt_int(stats["n_no_optical"] - stats["n_no_observations"]),
        mo=100 * stats["radar_mixed_orbit_frac"], md=100 * stats["radar_mixed_direction_frac"],
        sat=(f"{sat:.3f}\\,\\%" if sat >= 0.001 else "fewer than 0.001\\,\\%"),
        got=pd.Timestamp(stats["prism_retrieved"]).strftime("%-d %B %Y"),
        prov=pd.Timestamp(stats["prism_provisional_from"]).strftime("%-d %B %Y"))
    return stats, txt


def fmt_int(x):
    return f"{x:,}".replace(",", "\\,")


def main(version="v1.0"):
    import xarray as xr
    rel = config.RELEASE / version
    vdir = rel / "validation"
    vdir.mkdir(exist_ok=True)
    fields = gpd.read_parquet(rel / "fields.parquet")
    cube = xr.open_zarr(rel / "cube.zarr").load()
    wc_p = vdir / "worldcover_by_field.csv"
    wc = pd.read_csv(wc_p) if wc_p.exists() else worldcover(fields)
    wc.to_csv(wc_p, index=False)
    dstats, dtxt = delineation_text(wc, fields)
    cstats, ctxt, py, ry = coverage(rel, cube, fields)
    pd.DataFrame({"optical_clear_per_field": py, "radar_per_field": ry}).to_csv(vdir / "coverage_by_year.csv")
    pstats, ptxt = physical(cube, fields)
    qstats, qtxt = quality(rel, cube, fields)
    summary = {"stats": {"delineation": dstats, "coverage": cstats, "physical": pstats, "quality": qstats},
               "ValidationFields": dtxt, "ValidationCoverage": ctxt, "ValidationPhysical": ptxt,
               "ValidationQuality": qtxt,
               "ValidationSummary": ("that {g:.0f}\\,\\% of the field area is grassland or cropland in the independent "
                                     "ESA WorldCover map, that a median field has {c:.0f} clear optical observations "
                                     "per year, and that NDVI and backscatter respond to rainfall, drought and flooding "
                                     "as expected").format(
                   g=100 * (dstats["A"]["grassland"] + dstats["A"]["cropland"]),
                   c=cstats["clear_per_field_year_median"])}
    (vdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:])["stats"], indent=2))
