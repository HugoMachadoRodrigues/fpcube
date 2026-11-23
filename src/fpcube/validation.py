"""Technical validation of the Florida Pasture Cube. Writes <release>/validation/*.csv and summary.json, whose
text snippets state the numbers in sentences ready to be copied into the manuscript.

1. Delineation: land cover inside fields from ESA WorldCover 2021 (10 m, Sentinel-1/2 based, independent of CDL).
2. Coverage: clear optical observations and radar acquisitions per field and year; share of periods with data.
3. Physical consistency:
   a. NDVI seasonality by region (timing of the seasonal minimum and maximum);
   b. NDVI anomaly vs 60-day rainfall anomaly (drought signal) per field, across years;
   c. VV backscatter response to rain in the preceding 5 days (soil moisture signal);
   d. Hurricane Ian (landfall 28 Sep 2022): change of observed MNDWI and VV by HAND class for fields that received
      >= 200 mm in the two 5-day periods around landfall (10 days), with field counts and cluster-bootstrap intervals.
4. Quality: interpolation (with its denominators), unobserved fields, geometric-check states, radar orbit mixing
   within a period and orbit changes between consecutive periods, multipart fields, area accounting of the
   delineation, woody wetland left by the woody screen, and quality by tier, field size and season.
All optical statistics of the cube use observed composites only unless stated (tasks.observed_only).
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
    w = m.area_ha / m.area_ha.sum()
    out["all"] = {c: float((m[c] * w).sum()) for c in ("grassland", "cropland", "tree") if c in m}
    txt = ("ESA WorldCover 2021 \\citep{{zanaga2022}}, which is derived from Sentinel-1 and Sentinel-2 independently of "
           "the CDL, maps {gA:.0f}\\,\\% of the area of tier~A fields and {gB:.0f}\\,\\% of tier~B fields as grassland "
           "and a further {cA:.0f}\\,\\% and {cB:.0f}\\,\\% as cropland, which in Florida includes hay and improved "
           "pasture ({gcAll:.1f}\\,\\% of the area of all fields together). Trees cover {tA:.0f}\\,\\% (A) and "
           "{tB:.0f}\\,\\% (B) and herbaceous wetland {wA:.0f}\\,\\% and {wB:.0f}\\,\\%. Built-up land is below "
           "{bmax:.1f}\\,\\% in both tiers. This is agreement on land cover, not a measure of grazing use or of "
           "boundary accuracy.").format(
        gcAll=100 * (out["all"]["grassland"] + out["all"]["cropland"]),
        gA=100 * out["A"]["grassland"], gB=100 * out["B"]["grassland"], cA=100 * out["A"]["cropland"],
        cB=100 * out["B"]["cropland"], tA=100 * out["A"]["tree"], tB=100 * out["B"]["tree"],
        wA=100 * out["A"].get("herbaceous_wetland", 0), wB=100 * out["B"].get("herbaceous_wetland", 0),
        bmax=100 * max(out["A"]["built"], out["B"]["built"]))
    return out, txt


def coverage(rel, cube, fields):
    """Observation counts. A usable optical observation is one field x acquisition day with qa_use (records of the
    same day from overlapping tiles or both sensors count once). Per-field counts include fields with none."""
    opt = pd.concat(pd.read_parquet(p, columns=["fid", "date", "sensor", "qa_clear", "qa_geometry",
                                                    "qa_geometry_status", "shift_scope", "qa_range", "qa_haze",
                                                    "qa_use"])
                    for p in sorted((rel / "optical_observations").glob("*.parquet")))
    rad = pd.concat(pd.read_parquet(p, columns=["fid", "date"]) for p in (rel / "radar_observations").glob("*.parquet"))
    full_years = [y for y in config.YEARS if pd.Timestamp(f"{y}-12-31") <= pd.Timestamp(config.END)]
    good = opt[opt.qa_use]
    days = good.assign(day=pd.to_datetime(good.date).dt.normalize()).drop_duplicates(["fid", "day"])
    cnt = days.groupby(["fid", days.day.dt.year]).size().unstack(fill_value=0)
    cnt = cnt.reindex(index=fields.fid, columns=full_years, fill_value=0)
    per_year = cnt.median(axis=0)                                   # median field, each full year
    rd = rad.assign(day=pd.to_datetime(rad.date).dt.normalize()).drop_duplicates(["fid", "day"])
    rcnt = rd.groupby(["fid", rd.day.dt.year]).size().unstack(fill_value=0).reindex(
        index=fields.fid, columns=full_years, fill_value=0)
    rad_year = rcnt.median(axis=0)
    obs = cube.n_optical_obs.values > 0
    filled = cube.gap_filled.values
    month = pd.DatetimeIndex(cube.time.values).month
    wet = np.isin(month, [6, 7, 8, 9])
    gs = opt.qa_geometry_status.astype(str)
    stats = {"usable_days_per_field_year_median": float(per_year.median()),
             "usable_days_per_field_year_min": float(per_year.min()),
             "usable_days_per_field_year_max": float(per_year.max()),
             "usable_days_per_field_mean_over_years_median": float(cnt.mean(axis=1).median()),
             "full_years": f"{full_years[0]}-{full_years[-1]}",
             "clear_frac_of_all_obs": float(len(good) / len(opt)),
             "clear_test_frac": float(opt.qa_clear.mean()),
             "geom_fail_frac": float((gs == "fail").mean()),
             "geom_not_assessed_frac": float((gs == "not_assessed").mean()),
             "geom_not_assessed_frac_of_used": float((gs[opt.qa_use] == "not_assessed").mean()),
             "geom_fields_frac": float((opt.shift_scope.astype(str) == "fields").mean()),
             "geom_scene_frac": float((opt.shift_scope.astype(str) == "scene").mean()),
             "range_reject_frac": float((opt.qa_clear & ~opt.qa_range).mean()),
             "haze_reject_frac": float((opt.qa_clear & opt.qa_range & ~opt.qa_haze).mean()),
             "radar_days_per_field_year_median": float(rad_year.median()),
             "cells_observed": float(obs.mean()), "cells_observed_wet": float(obs[:, wet].mean()),
             "cells_observed_dry": float(obs[:, ~wet].mean()),
             "cells_interpolated": float(filled.mean()),
             "cells_empty": float(1 - obs.mean() - filled.mean()),
             "interpolated_share_of_ndvi_values": float(filled.sum() / (obs.sum() + filled.sum()))}
    txt = ("Of all field-level HLS records, {ct:.0f}\\,\\% passed the clear-pixel test; the geometric shift was "
           "estimated for {ga:.0f}\\,\\% of the records, {gf:.0f}\\,\\% over the clear field pixels of the granule and "
           "{gsc:.0f}\\,\\% over a clear 512 by 512 block of the tile where that window was smaller than 64 by 64 pixels, "
           "and exceeded one pixel in {gr:.2f}\\,\\%; the {gn:.1f}\\,\\% not assessed ({gnu:.1f}\\,\\% of the used "
           "records) are kept and marked (\\texttt{{qa\\_geometry\\_status}}). Of the clear records, {rr:.2f}\\,\\% and {hr:.2f}\\,\\% were rejected for "
           "reflectance outside [$-0.01$, 1] or blue reflectance above 0.25; {cf:.0f}\\,\\% of all records were used. "
           "Counting one observation per field and acquisition day, the median field has {cm:.0f} usable optical "
           "observations per year (median over {fy}; yearly medians {cmin:.0f}--{cmax:.0f}, increasing as Landsat~9 "
           "and Sentinel-2C joined) and {rm:.0f} Sentinel-1 acquisition days. Of all field $\\times$ 5-day cells, "
           "{po:.0f}\\,\\% hold an observed optical composite ({pd:.0f}\\,\\% in October--May, {pw:.0f}\\,\\% in "
           "June--September), {pi:.0f}\\,\\% an interpolated one (gaps of at most 30 days) and {pe:.0f}\\,\\% none; "
           "interpolated values are therefore {ish:.0f}\\,\\% of the NDVI values in the cube.").format(
        cf=100 * stats["clear_frac_of_all_obs"], ct=100 * stats["clear_test_frac"],
        ga=100 * (1 - stats["geom_not_assessed_frac"]), gr=100 * stats["geom_fail_frac"],
        gn=100 * stats["geom_not_assessed_frac"], gnu=100 * stats["geom_not_assessed_frac_of_used"],
        gf=100 * stats["geom_fields_frac"], gsc=100 * stats["geom_scene_frac"],
        rr=100 * stats["range_reject_frac"], hr=100 * stats["haze_reject_frac"],
        cm=stats["usable_days_per_field_year_median"], fy=stats["full_years"].replace("-", "--"),
        cmin=per_year.min(), cmax=per_year.max(),
        rm=stats["radar_days_per_field_year_median"], po=100 * stats["cells_observed"],
        pd=100 * stats["cells_observed_dry"], pw=100 * stats["cells_observed_wet"],
        pi=100 * stats["cells_interpolated"], pe=100 * stats["cells_empty"],
        ish=100 * stats["interpolated_share_of_ndvi_values"])
    return stats, txt, per_year, rad_year, cnt


def physical(cube, fields, B=1000):
    """Expects the full cube; anomalies and Ian changes use observed optical composites only."""
    from fpcube.benchmark.tasks import ndvi_anomaly, observed_only
    times = pd.DatetimeIndex(cube.time.values)
    nd = cube.ndvi.values
    # (a) seasonality: month of minimum and maximum median NDVI, statewide (gap-filled values included)
    monthly = pd.Series(np.nanmedian(nd, axis=0), index=times).groupby(times.month).mean()
    obs_cube = observed_only(cube[["ndvi", "mndwi", "n_optical_obs"]].copy())
    # (b) NDVI anomaly vs 60-day rainfall anomaly, April-May (dry season end) per field-period
    an = ndvi_anomaly(obs_cube)
    ppt = cube.ppt_sum.values
    ppt60 = pd.DataFrame(ppt.T).rolling(12, min_periods=12).sum().to_numpy().T
    mu = np.nanmean(ppt60, axis=1, keepdims=True)
    sd = np.nanstd(ppt60, axis=1, keepdims=True)
    pz = (ppt60 - mu) / np.where(sd > 0, sd, np.nan)
    am = np.isin(times.month, [4, 5])
    a, p = an[:, am].ravel(), pz[:, am].ravel()
    ok = np.isfinite(a) & np.isfinite(p)
    r_drought = float(np.corrcoef(a[ok], p[ok])[0, 1])
    # (c) VV vs 5-day rain: periods with >= 25 mm vs <= 1 mm, per field
    vv = cube.vv_db.values
    wetp, dryp = ppt >= 25, ppt <= 1
    dvv = np.nanmedian(np.where(wetp, vv, np.nan), axis=1) - np.nanmedian(np.where(dryp, vv, np.nan), axis=1)
    out = {"ndvi_month_min": int(monthly.idxmin()), "ndvi_month_max": int(monthly.idxmax()),
           "r_ndvi_anom_rain60_AM": r_drought, "n_pairs_AM": int(ok.sum()),
           "vv_wet_minus_dry_db_median": float(np.nanmedian(dvv)),
           "vv_wet_gt_dry_frac": float(np.nanmean(dvv > 0))}
    # (d) Ian: rain in the two periods around landfall; change of observed MNDWI (max of the 3 periods after vs median
    # of the 6 before) and VV (min of the 2 after vs median of the 6 before), by HAND class
    ian = pd.Timestamp("2022-09-28")
    k = int(np.searchsorted(times, ian))
    if "hand_mean_m" in fields and 0 < k < len(times) - 3:
        rain = np.nansum(ppt[:, k - 1:k + 1], axis=1)
        hit = rain >= 200
        mn = obs_cube.mndwi.values
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            dm = np.nanmax(mn[:, k:k + 3], axis=1) - np.nanmedian(mn[:, k - 6:k], axis=1)
            dv = np.nanmin(vv[:, k:k + 2], axis=1) - np.nanmedian(vv[:, k - 6:k], axis=1)
        hand = fields.hand_mean_m.to_numpy()
        county = fields.county.to_numpy()
        classes = {"le1": hand <= 1, "1to3": (hand > 1) & (hand < 3), "ge3": hand >= 3}
        out.update({"ian_rain_window": f"{times[k - 1]:%d %b}--{times[k] + pd.Timedelta(days=4):%d %b %Y}",
                    "ian_fields_200mm": int(hit.sum())})
        rng = np.random.default_rng(7)
        for nm, c in classes.items():
            for var, d in (("dmndwi", dm), ("dvv", dv)):
                sel = hit & c & np.isfinite(d)
                x, cl = d[sel], pd.factorize(county[sel])[0]
                out[f"ian_n_{nm}"] = int((hit & c).sum())
                out[f"ian_{var}_n_{nm}"] = int(sel.sum())
                if sel.sum() < 10:
                    continue
                q = np.percentile(x, [25, 50, 75])
                meds = []
                for _ in range(B):                                   # resample counties (spatially clustered rain)
                    w = M_cluster_weights(cl, rng)
                    o = np.argsort(x)
                    cw = np.cumsum(w[o])
                    meds.append(x[o][np.searchsorted(cw, cw[-1] / 2)])
                out.update({f"ian_{var}_{nm}_median": float(q[1]), f"ian_{var}_{nm}_q25": float(q[0]),
                            f"ian_{var}_{nm}_q75": float(q[2]),
                            f"ian_{var}_{nm}_lo": float(np.percentile(meds, 2.5)),
                            f"ian_{var}_{nm}_hi": float(np.percentile(meds, 97.5))})
    txt = ("Statewide median NDVI reaches its minimum in {mmin} and its maximum in {mmax}. At the end of the dry season "
           "(April--May) the anomaly of observed NDVI is correlated with the 60-day rainfall anomaly ($r={r:.2f}$, "
           "$n={n}$ field-periods, not independent). Sentinel-1 VV backscatter is higher in periods with at least "
           "25\\,mm of rain than in periods with at most 1\\,mm for {vf:.0f}\\,\\% of fields (median difference "
           "{dv:.1f}\\,dB).").format(
        mmin=pd.Timestamp(2001, out["ndvi_month_min"], 1).strftime("%B"),
        mmax=pd.Timestamp(2001, out["ndvi_month_max"], 1).strftime("%B"), r=r_drought,
        n=f"{ok.sum():,}".replace(",", "\\,"), vf=100 * out["vv_wet_gt_dry_frac"], dv=out["vv_wet_minus_dry_db_median"])
    if "ian_dmndwi_le1_median" in out and "ian_dmndwi_ge3_median" in out:
        g = lambda v, c, nd_: ("{:.{p}f} (IQR {:.{p}f} to {:.{p}f}; 95\\,\\% interval of the median {:.{p}f} to "
                               "{:.{p}f}; $n={}$)").format(out[f"ian_{v}_{c}_median"], out[f"ian_{v}_{c}_q25"],
                                                          out[f"ian_{v}_{c}_q75"], out[f"ian_{v}_{c}_lo"],
                                                          out[f"ian_{v}_{c}_hi"],
                                                          f"{out[f'ian_{v}_n_{c}']:,}".replace(",", "\\,"), p=nd_)
        txt += (" Hurricane Ian brought at least 200\\,mm in {win} to {n} fields. Among them, the change of observed "
                "MNDWI after landfall was {ml} for fields with HAND of 1\\,m or less, {mm} for 1--3\\,m and {mh} for "
                "3\\,m or more; the change of VV was {vl}, {vm} and {vh}\\,dB (Fig.~\\ref{{fig:events}}).").format(
            win=out["ian_rain_window"], n=f"{out['ian_fields_200mm']:,}".replace(",", "\\,"),
            ml=g("dmndwi", "le1", 2), mm=g("dmndwi", "1to3", 2), mh=g("dmndwi", "ge3", 2),
            vl=g("dvv", "le1", 1), vm=g("dvv", "1to3", 1), vh=g("dvv", "ge3", 1))
    return out, txt


def freeze_and_condition(cube, fields, B=1000):
    """(e) January 2018 freeze: change of observed NDVI and of VV for fields whose 5-day mean minimum temperature in
    the period of 17 Jan 2018 was <= 2 degC against fields at >= 8 degC (median of the 3 periods after minus median of
    the 6 before), with county-cluster intervals of the median.
    (f) Statewide pasture condition: weekly NASS Florida pasture condition index against the statewide median
    anomaly of observed NDVI in the 5-day period that ends in the NASS week (both raw and with the index's
    week-of-year mean removed). Weeks are serially correlated, so r is descriptive."""
    import warnings
    from fpcube.benchmark.tasks import ndvi_anomaly, observed_only
    times = pd.DatetimeIndex(cube.time.values)
    obs = observed_only(cube[["ndvi", "mndwi", "n_optical_obs"]].copy())
    nd, vv = obs.ndvi.values, cube.vv_db.values
    county = pd.factorize(fields.county.to_numpy())[0]
    rng = np.random.default_rng(11)
    out = {}
    k = int(np.searchsorted(times, pd.Timestamp("2018-01-17"), side="right")) - 1
    tmin = cube.tmin_mean.values[:, k]
    groups = {"cold": tmin <= 2, "mild": tmin >= 8}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = {"dndvi": np.nanmedian(nd[:, k + 1:k + 4], 1) - np.nanmedian(nd[:, k - 6:k], 1),
             "dvv": np.nanmedian(vv[:, k + 1:k + 4], 1) - np.nanmedian(vv[:, k - 6:k], 1)}
    out["freeze_period"] = f"{times[k]:%d %b %Y}"
    for g, rows in groups.items():
        out[f"freeze_n_{g}"] = int(rows.sum())
        for var, x in d.items():
            sel = rows & np.isfinite(x)
            v, cl = x[sel], pd.factorize(county[sel])[0]
            out[f"freeze_{var}_n_{g}"] = int(sel.sum())
            if sel.sum() < 10:
                continue
            q = np.percentile(v, [25, 50, 75])
            o = np.argsort(v)
            meds = []
            for _ in range(B):
                w = M_cluster_weights(cl, rng)[o]
                cw = np.cumsum(w)
                meds.append(v[o][np.searchsorted(cw, cw[-1] / 2)])
            out.update({f"freeze_{var}_{g}_median": float(q[1]), f"freeze_{var}_{g}_q25": float(q[0]),
                        f"freeze_{var}_{g}_q75": float(q[2]), f"freeze_{var}_{g}_lo": float(np.percentile(meds, 2.5)),
                        f"freeze_{var}_{g}_hi": float(np.percentile(meds, 97.5))})
    # (e2) same fields, same calendar periods in the other years without a freeze in that period: the 2018 drop
    # minus the field's usual change at that time of year separates freeze damage from winter dormancy
    cold = groups["cold"]
    poy = (times.day_of_year - 1) // config.COMPOSITE_DAYS
    others = []
    for yr in range(2016, 2026):
        if yr == 2018:
            continue
        kk = np.nonzero((times.year == yr) & (poy == poy[k]))[0]
        if not len(kk) or kk[0] < 6 or kk[0] + 4 > len(times):
            continue
        kk = kk[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dd = np.nanmedian(nd[:, kk + 1:kk + 4], 1) - np.nanmedian(nd[:, kk - 6:kk], 1)
        dd[cube.tmin_mean.values[:, kk] <= 2] = np.nan            # that year was cold too: not a control
        others.append(dd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        usual = np.nanmedian(np.stack(others), 0)
        n_years = np.isfinite(np.stack(others)).sum(0)
    excess = d["dndvi"] - usual
    sel = cold & np.isfinite(excess) & (n_years >= 3)
    v, cl = excess[sel], pd.factorize(county[sel])[0]
    o = np.argsort(v)
    meds = []
    for _ in range(B):
        w = M_cluster_weights(cl, rng)[o]
        cw = np.cumsum(w)
        meds.append(v[o][np.searchsorted(cw, cw[-1] / 2)])
    q = np.percentile(v, [25, 50, 75])
    out.update({"freeze_excess_n": int(sel.sum()), "freeze_excess_median": float(q[1]),
                "freeze_excess_q25": float(q[0]), "freeze_excess_q75": float(q[2]),
                "freeze_excess_lo": float(np.percentile(meds, 2.5)), "freeze_excess_hi": float(np.percentile(meds, 97.5)),
                "freeze_usual_median": float(np.nanmedian(usual[sel])), "freeze_excess_share_below": float(np.mean(v < 0))})
    # (f) NASS condition
    nass = pd.read_parquet(config.RAW / "labels" / "nass_fl_pasture_condition_weekly.parquet")
    nass["week_ending"] = pd.to_datetime(nass.week_ending)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        state = pd.Series(np.nanmedian(ndvi_anomaly(obs), axis=0), index=times + pd.Timedelta(days=4))  # period end
        nfield = pd.Series(np.isfinite(nd).sum(0), index=state.index)
    state = state[nfield >= 1000].dropna()
    j = np.searchsorted(state.index.values, nass.week_ending.values, side="right") - 1
    ok = (j >= 0)
    lag = np.full(len(nass), np.inf)
    lag[ok] = (nass.week_ending.values[ok] - state.index.values[j[ok]]) / np.timedelta64(1, "D")
    ok &= lag < 7
    x = np.where(ok, state.values[np.clip(j, 0, None)], np.nan)
    ci = nass.condition_index.to_numpy(float)
    woy = nass.week_ending.dt.isocalendar().week.to_numpy()
    ci_anom = ci - pd.Series(ci).groupby(woy).transform("mean").to_numpy()
    sel = np.isfinite(x) & np.isfinite(ci)
    out.update({"nass_weeks": int(sel.sum()),
                "nass_r_raw": float(np.corrcoef(x[sel], ci[sel])[0, 1]),
                "nass_r_anom": float(np.corrcoef(x[sel], ci_anom[sel])[0, 1]),
                "nass_spearman_anom": float(pd.Series(x[sel]).corr(pd.Series(ci_anom[sel]), method="spearman"))})
    by_year = {}
    for yr in sorted(set(nass.week_ending.dt.year[sel])):
        m = sel & (nass.week_ending.dt.year.to_numpy() == yr)
        if m.sum() >= 15:
            by_year[int(yr)] = float(np.corrcoef(x[m], ci_anom[m])[0, 1])
    out["nass_r_anom_by_year"] = by_year
    g = lambda v, c, p_: ("{:.{p}f} (IQR {:.{p}f} to {:.{p}f}; 95\\,\\% interval of the median {:.{p}f} to {:.{p}f}; "
                          "$n={}$)").format(out[f"freeze_{v}_{c}_median"], out[f"freeze_{v}_{c}_q25"],
                                           out[f"freeze_{v}_{c}_q75"], out[f"freeze_{v}_{c}_lo"],
                                           out[f"freeze_{v}_{c}_hi"],
                                           f"{out[f'freeze_{v}_n_{c}']:,}".replace(",", "\\,"), p=p_)
    txt = ("In the 5-day period starting {per}, {nc} fields had a mean minimum temperature of 2\\,$^\\circ$C or "
           "lower and {nm} fields one of 8\\,$^\\circ$C or higher. Over the following 15 days observed NDVI changed by "
           "{dc} in the cold fields and by {dm} in the mild ones, and VV by {vc} and {vm}\\,dB. In the same cold "
           "fields and calendar periods of the other years without a freeze, NDVI changed by a median of {us:.2f}, so "
           "the 2018 change minus the usual one was {ex:.2f} (IQR {eq1:.2f} to {eq3:.2f}; 95\\,\\% interval of the "
           "median {elo:.2f} to {ehi:.2f}; $n={en}$; {eb:.0f}\\,\\% of fields below their usual change). "
           "Over {nw} weeks the statewide median anomaly of observed NDVI is correlated with the weekly NASS "
           "pasture condition index ($r={rr:.2f}$) and with the index after removing its mean for the week of year "
           "($r={ra:.2f}$; Spearman {rs:.2f}); weeks are serially correlated and the yearly values range from "
           "{ymin:.2f} to {ymax:.2f}.").format(
        per=out["freeze_period"], nc=f"{out['freeze_n_cold']:,}".replace(",", "\\,"),
        nm=f"{out['freeze_n_mild']:,}".replace(",", "\\,"), dc=g("dndvi", "cold", 2), dm=g("dndvi", "mild", 2),
        vc=g("dvv", "cold", 1), vm=g("dvv", "mild", 1), us=out["freeze_usual_median"], ex=out["freeze_excess_median"],
        eq1=out["freeze_excess_q25"], eq3=out["freeze_excess_q75"], elo=out["freeze_excess_lo"],
        ehi=out["freeze_excess_hi"], en=f"{out['freeze_excess_n']:,}".replace(",", "\\,"),
        eb=100 * out["freeze_excess_share_below"], nw=out["nass_weeks"], rr=out["nass_r_raw"],
        ra=out["nass_r_anom"], rs=out["nass_spearman_anom"], ymin=min(by_year.values()), ymax=max(by_year.values()))
    return out, txt


def ian_passes(rel, cube, fields, B=1000, drop_db=3.0):
    """(g) Hurricane Ian on single Sentinel-1 passes instead of 5-day composites: VV change on relative orbit 48
    between 20 Sep and 2 Oct 2022 (4 days after landfall) per field, and the same orbit's change between 27 Aug and
    20 Sep as the change expected without the storm. Open water lowers C-band backscatter, so the share of fields
    whose VV dropped by more than `drop_db` is compared by rain group (>= 200 mm or < 50 mm in the two periods
    around landfall) and HAND class, with county-cluster intervals."""
    import warnings
    d = pd.read_parquet(rel / "radar_observations" / "2022.parquet", columns=["fid", "date", "relative_orbit", "vv_db"])
    d = d[d.relative_orbit == 48]
    w = d.assign(day=d.date.dt.strftime("%m-%d")).pivot_table(index="fid", columns="day", values="vv_db")
    times = pd.DatetimeIndex(cube.time.values)
    k = int(np.searchsorted(times, pd.Timestamp("2022-09-28")))
    rain = np.nansum(cube.ppt_sum.values[:, k - 1:k + 1], axis=1)
    f = pd.DataFrame({"fid": fields.fid.to_numpy(), "rain": rain, "hand": fields.hand_mean_m.to_numpy(),
                      "county": fields.county.to_numpy()}).set_index("fid")
    x = f.join(w[["08-27", "09-20", "10-02"]], how="inner")
    x["post"], x["pre"] = x["10-02"] - x["09-20"], x["09-20"] - x["08-27"]
    groups = {"hit_le1": (x.rain >= 200) & (x.hand <= 1), "hit_1to3": (x.rain >= 200) & (x.hand > 1) & (x.hand < 3),
              "hit_ge3": (x.rain >= 200) & (x.hand >= 3), "ctrl": x.rain < 50}
    rng = np.random.default_rng(5)
    out = {"ian_pass_dates": "20 Sep and 2 Oct 2022 (relative orbit 48)", "ian_pass_drop_db": drop_db}
    for g, m in groups.items():
        s = x[m & x.post.notna() & x.pre.notna()]
        cl = pd.factorize(s.county)[0]
        a, b = (s.post < -drop_db).to_numpy(float), (s.pre < -drop_db).to_numpy(float)
        sh = lambda w_, v: (w_ * v).sum() / w_.sum()
        boot = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(B):
                w_ = M_cluster_weights(cl, rng).astype(float)
                boot.append((sh(w_, a), sh(w_, b), sh(w_, a) - sh(w_, b)))
        boot = np.array(boot)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        out.update({f"ian_pass_n_{g}": int(len(s)), f"ian_pass_post_{g}": float(a.mean()),
                    f"ian_pass_pre_{g}": float(b.mean()), f"ian_pass_post_{g}_lo": float(lo[0]),
                    f"ian_pass_post_{g}_hi": float(hi[0]), f"ian_pass_diff_{g}": float(a.mean() - b.mean()),
                    f"ian_pass_diff_{g}_lo": float(lo[2]), f"ian_pass_diff_{g}_hi": float(hi[2]),
                    f"ian_pass_dvv_median_{g}": float(s.post.median())})
    pc = lambda v: f"{100 * v:.1f}\\,\\%"
    g_ = lambda g: ("{post} [{lo}, {hi}] of {n} fields, against {pre} between the two passes before the storm "
                    "(difference {d} [{dlo}, {dhi}])").format(
        post=pc(out[f"ian_pass_post_{g}"]), lo=pc(out[f"ian_pass_post_{g}_lo"]), hi=pc(out[f"ian_pass_post_{g}_hi"]),
        n=f"{out[f'ian_pass_n_{g}']:,}".replace(",", "\\,"), pre=pc(out[f"ian_pass_pre_{g}"]),
        d=pc(out[f"ian_pass_diff_{g}"]), dlo=pc(out[f"ian_pass_diff_{g}_lo"]), dhi=pc(out[f"ian_pass_diff_{g}_hi"]))
    txt = ("On single Sentinel-1 passes of the same orbit, 4 days after Ian's landfall (20 Sep against 2 Oct 2022), VV "
           "dropped by more than {db:.0f}\\,dB in {le1} among fields with at least 200\\,mm and HAND of 1\\,m or "
           "less; in {m13} with HAND of 1--3\\,m; in {g3} with HAND of 3\\,m or more; and in {ct} among fields with "
           "less than 50\\,mm.").format(db=drop_db, le1=g_("hit_le1"), m13=g_("hit_1to3"), g3=g_("hit_ge3"),
                                          ct=g_("ctrl"))
    return out, txt


def M_cluster_weights(codes, rng):
    from fpcube.benchmark.models import _cluster_weights
    return _cluster_weights(codes, rng)


def quality(rel, cube, fields):
    """Known limits of the dynamic layers, stated with their size: interpolation, unobserved fields, radar orbit
    mixing within a period and orbit changes between consecutive observed periods, saturated NDVI, provisional
    PRISM, multipart fields."""
    from fpcube.cube import period_index
    nd, gf, no = cube.ndvi.values, cube.gap_filled.values, cube.n_optical_obs.values
    obs = no > 0
    r = pd.concat(pd.read_parquet(p, columns=["fid", "date", "relative_orbit", "orbit_state"])
                  for p in (rel / "radar_observations").glob("*.parquet"))
    r["k"] = period_index(pd.to_datetime(r.date))
    g = r.groupby(["fid", "k"])
    # orbit set of each observed period, compared with the field's previous observed period
    sets = g.relative_orbit.agg(lambda x: tuple(sorted(set(x)))).reset_index().sort_values(["fid", "k"])
    prev = sets.groupby("fid").relative_orbit.shift(1)
    has_prev = prev.notna()
    changed = (sets.relative_orbit != prev)[has_prev]
    single = sets.relative_orbit.map(len) == 1
    changed_single = (sets.relative_orbit != prev)[has_prev & single & prev.map(lambda x: isinstance(x, tuple)
                                                                               and len(x) == 1)]
    p_ = sorted((config.RAW / "prism" / "ppt").glob("prism_ppt_*.tif"))[-1]
    with rasterio.open(p_) as s_:
        got = s_.tags().get("retrieved")
    got = pd.Timestamp(got) if got else pd.Timestamp(p_.stat().st_mtime, unit="s").normalize()
    parts = fields.geometry.apply(lambda x: len(x.geoms) if x.geom_type == "MultiPolygon" else 1)
    minor = fields.geometry.apply(lambda x: 0.0 if x.geom_type != "MultiPolygon"
                                  else sum(sorted(q.area for q in x.geoms)[:-1]) / 1e4)
    stats = {"interpolated_frac": float(gf.sum() / np.isfinite(nd).sum()),
             "n_no_observations": int(fields.flag_no_observations.sum()),
             "n_no_optical": int(fields.flag_no_optical.sum()),
             "radar_mixed_orbit_frac": float((g.relative_orbit.nunique() > 1).mean()),
             "radar_mixed_direction_frac": float((g.orbit_state.nunique() > 1).mean()),
             "radar_orbit_change_consecutive_frac": float(changed.mean()),
             "radar_orbit_change_consecutive_single_frac": float(changed_single.mean()),
             "ndvi_saturated_frac": float((obs & (np.abs(nd) >= 0.999)).sum() / obs.sum()),
             "prism_retrieved": str(got.date()),
             "prism_provisional_from": str((got - pd.DateOffset(months=6)).date()),
             "n_multipart_fields": int((parts > 1).sum()),
             "multipart_area_share": float(fields.area_ha[parts > 1].sum() / fields.area_ha.sum()),
             "minor_parts_ha": float(minor.sum()),
             "n_fields_minor_part_ge_2ha": int((minor >= 2).sum())}
    sat = 100 * stats["ndvi_saturated_frac"]
    txt = ("{nn} fields have no optical or radar observation at all and {nr} more have radar only; "
           "they are narrow fields in which no pixel remains after the inward buffers, kept in the release and flagged "
           "(\\texttt{{flag\\_no\\_observations}}, \\texttt{{flag\\_no\\_optical}}). {mo:.1f}\\,\\% of the radar composites "
           "combine acquisitions from more than one relative orbit ({md:.1f}\\,\\% mix ascending and descending passes), "
           "and the set of relative orbits differs from the field's previous observed period in {oc:.0f}\\,\\% of the "
           "composites, so the incidence angle varies within and between periods; the per-acquisition table keeps the "
           "orbit of every value. Because indices are computed per pixel and limited to $[-1, 1]$, NDVI composites reach "
           "$\\pm 1$ in {sat} of the observed cells. {parts}PRISM daily values become stable about "
           "six months after the date; the record was retrieved on {got}, so values after {prov} are provisional and may "
           "be revised in later PRISM releases.").format(
        nn=fmt_int(stats["n_no_observations"]),
        nr=fmt_int(stats["n_no_optical"] - stats["n_no_observations"]),
        mo=100 * stats["radar_mixed_orbit_frac"], md=100 * stats["radar_mixed_direction_frac"],
        oc=100 * stats["radar_orbit_change_consecutive_frac"],
        sat=(f"{sat:.3f}\\,\\%" if sat >= 0.001 else "fewer than 0.001\\,\\%"),
        parts=(("{nmp} fields ({mps:.0f}\\,\\% of the field area) consist of more than one 4-connected part, "
                "because a pasture patch crossing a parcel boundary can split the part inside one parcel; the smaller "
                "parts total {mha} ha (\\texttt{{n\\_parts}}). ").format(
                    nmp=fmt_int(stats["n_multipart_fields"]), mps=100 * stats["multipart_area_share"],
                    mha=fmt_int(int(round(stats["minor_parts_ha"], -2))))
               if stats["n_multipart_fields"] else "Every field is a single 4-connected part. "),
        got=pd.Timestamp(stats["prism_retrieved"]).strftime("%-d %B %Y"),
        prov=pd.Timestamp(stats["prism_provisional_from"]).strftime("%-d %B %Y"))
    return stats, txt


SIZE_BINS = [2, 5, 20, 100, np.inf]


def quality_by_stratum(cube, fields, counts):
    """Observed, interpolated and empty shares of field x period cells and usable optical days per year, by tier
    and field size, with the dry (October-May) and wet (June-September) seasons separately."""
    obs = cube.n_optical_obs.values > 0
    gf = cube.gap_filled.values
    wet = np.isin(pd.DatetimeIndex(cube.time.values).month, [6, 7, 8, 9])
    size = pd.cut(fields.area_ha, SIZE_BINS, right=False,
                  labels=["2--5 ha", "5--20 ha", "20--100 ha", "$\\geq$100 ha"])
    days = counts.reindex(fields.fid).mean(axis=1).to_numpy()
    rows = []
    frame = pd.DataFrame({"tier": fields.tier.to_numpy(), "size": np.asarray(size)}).assign(
        size=lambda x: pd.Categorical(x["size"], categories=size.cat.categories, ordered=True))
    for (tier, sz), g in frame.groupby(["tier", "size"], observed=True, sort=True):
        i = g.index.to_numpy()
        rows.append({"tier": tier, "size": sz, "n_fields": len(i),
                     "observed_dry": obs[i][:, ~wet].mean(), "observed_wet": obs[i][:, wet].mean(),
                     "interpolated": gf[i].mean(), "empty": 1 - obs[i].mean() - gf[i].mean(),
                     "usable_days_per_year_median": float(np.median(days[i])),
                     "no_optical_fields": int(fields.flag_no_optical.to_numpy()[i].sum())})
    return pd.DataFrame(rows)


def delineation_accounting(fields):
    """Area and units after each step of the delineation (from fields.build's delineation_stats.json) and the woody
    wetland area that the woody screen did not remove (CDL class 190 is not among the screened classes)."""
    st = json.loads((config.FIELDS_DIR / "delineation_stats.json").read_text())
    lab_p = config.FIELDS_DIR / "field_labels.tif"
    v11 = st.get("version", "v1.0") != "v1.0"
    woody = "141, 142, 143, 152, 190" if v11 else "141, 142, 143, 152"
    split = "into connected parts" if v11 else "by parcel"
    out = {"steps": [
        {"step": "Persistent pasture (CDL 176 or 37 in >= 6 of 10 years, 2016-2025)", "area_ha": st["persistent_pasture_ha"]},
        {"step": f"Removed: woody (CDL {woody}) in >= 2 of 2012-2016", "area_ha": -st["removed_by_woody_screen_ha"]},
        {"step": "Removed: outside any parcel of the two parcel layers",
         "area_ha": -(st["kept_pasture_ha"] - st["kept_in_grazing_parcels_ha"])},
        {"step": f"Removed: parts smaller than 2 ha after the split {split}",
         "area_ha": -(st["kept_in_grazing_parcels_ha"] - st["in_fields_ha"])},
        {"step": "Fields", "area_ha": st["in_fields_ha"], "n": st["n_fields"]}]}
    if lab_p.exists():
        with rasterio.open(lab_p) as s_:
            lab = s_.read(1)
        w = np.zeros(lab.shape, np.uint8)
        for y in config.WOODY_SCREEN_YEARS:
            with rasterio.open(config.RAW / "cdl" / f"CDL_{y}_{config.FLORIDA_FIPS}.tif") as s_:
                w += np.isin(s_.read(1), config.CDL_WETLAND_WOODY)
        inf = lab > 0
        px = st["pixel_ha"]
        nf = np.bincount(lab[inf & (w >= 2)], minlength=lab.max() + 1)[1:]
        tot = np.bincount(lab[inf], minlength=lab.max() + 1)[1:]
        out.update({"woody_wetland_2plus_ha": float((inf & (w >= 2)).sum() * px),
                    "woody_wetland_2plus_share": float((inf & (w >= 2)).sum() / inf.sum()),
                    "woody_wetland_any_ha": float((inf & (w >= 1)).sum() * px),
                    "n_fields_woody_wetland_ge50pct": int((nf / np.maximum(tot, 1) >= 0.5).sum())})
    return out


NEIGHBOUR_DAYS = 16      # window for comparing a screened record with the field's used records


def evidence(rel, fields, n_fields=1500, seed=11):
    """Numbers quoted in the methods, measured on the release: (1) NDVI of records rejected by the reflectance-range
    and haze screens against the same field's used records within NEIGHBOUR_DAYS, on a random sample of fields
    holding such records; (2) missing water-table depth by drainage class; (3) accuracy of the landfall rule used
    for working best tracks, applied to six-hourly HURDAT2 points and compared with the official landfalls."""
    import pyarrow.dataset as pads
    from fpcube.sources import labels as Lb
    out = {}
    ds = pads.dataset(str(rel / "optical_observations"))
    q = ds.to_table(columns=["fid", "qa_clear", "qa_geometry", "qa_range", "qa_haze"]).to_pandas()
    base = q.qa_clear & q.qa_geometry
    rng = np.random.default_rng(seed)
    cats = {"range": base & ~q.qa_range, "haze": base & q.qa_range & ~q.qa_haze}
    pick = np.unique(np.concatenate([rng.choice(np.unique(q.fid[m]), min(n_fields, int(np.unique(q.fid[m]).size)),
                                                replace=False) for m in cats.values()]))
    del q
    d = ds.to_table(filter=pads.field("fid").isin(pick.tolist()),
                    columns=["fid", "date", "ndvi", "qa_clear", "qa_geometry", "qa_range", "qa_haze", "qa_use"]).to_pandas()
    d["date"] = pd.to_datetime(d.date)
    d["cat"] = np.select([d.qa_clear & d.qa_geometry & ~d.qa_range, d.qa_clear & d.qa_geometry & d.qa_range & ~d.qa_haze],
                         ["range", "haze"], "")
    win = np.timedelta64(NEIGHBOUR_DAYS, "D")
    diffs = {"range": [], "haze": []}
    for _, g in d.groupby("fid"):
        used = g[g.qa_use].sort_values("date")
        ut, un = used.date.to_numpy(), used.ndvi.to_numpy()
        for r in g[g.cat != ""].itertuples():
            lo, hi = np.searchsorted(ut, r.date - win), np.searchsorted(ut, r.date + win, side="right")
            if hi - lo >= 2:
                diffs[r.cat].append(r.ndvi - np.median(un[lo:hi]))
    for k, v in diffs.items():
        v = np.asarray(v)
        out[f"{k}_n"] = int(v.size)
        out[f"{k}_ndvi_diff_median"] = float(np.median(v))
        out[f"{k}_share_drop_gt_0.2"] = float(np.mean(v < -0.2))
    wt = fields.water_table_min_cm.isna()
    by = wt.groupby(fields.drainage_class).mean()
    out["wt_missing_all"] = float(wt.mean())
    out["wt_missing_excessively"] = float(by.get("Excessively drained", np.nan))
    out["wt_missing_well"] = float(by.get("Well drained", np.nan))
    out["wt_missing_poorly_max"] = float(by.reindex(["Somewhat poorly drained", "Poorly drained",
                                                     "Very poorly drained"]).max())
    t = pd.read_parquet(rel / "labels" / "hurricane_tracks_florida.parquet")
    lf = pd.read_parquet(rel / "labels" / "hurricane_landfalls.parquet")
    box = lambda x: x.lon.between(-87.7, -79.9) & x.lat.between(24.4, 31.1)
    off = lf[(lf.source == "HURDAT2") & box(lf)]
    syn = t[(t.source == "HURDAT2") & (t.time.dt.minute == 0) & (t.time.dt.hour % 6 == 0) & (t.record != "L")]
    der = Lb.derive_landfalls(syn[syn.storm_id.isin(off.storm_id)], Lb.florida_land())
    dt, km = [], []
    for r in off.itertuples():
        x = der[der.storm_id == r.storm_id]
        if len(x):
            j = (x.time - r.time).abs().idxmin()
            dt.append(abs((x.time[j] - r.time).total_seconds()) / 60)
            km.append(111.2 * np.hypot(x.lat[j] - r.lat, (x.lon[j] - r.lon) * np.cos(np.radians(r.lat))))
    out["landfall_n"] = int(len(off))
    out["landfall_dt_min_median"] = float(np.median(dt))
    out["landfall_km_median"] = float(np.median(km))
    return out


def fmt_int(x):
    return f"{x:,}".replace(",", "\\,")


def main(version="v1.0"):
    import xarray as xr
    from fpcube import provenance
    rel = config.RELEASE / version
    vdir = rel / "validation"
    vdir.mkdir(exist_ok=True)
    fields = gpd.read_parquet(rel / "fields.parquet")
    cube = xr.open_zarr(rel / "cube.zarr").load()
    wc_p = vdir / "worldcover_by_field.csv"
    wc = pd.read_csv(wc_p) if wc_p.exists() else worldcover(fields)
    wc.to_csv(wc_p, index=False)
    dstats, dtxt = delineation_text(wc, fields)
    cstats, ctxt, py, ry, counts = coverage(rel, cube, fields)
    pd.DataFrame({"usable_optical_days_median_field": py, "radar_days_median_field": ry}).to_csv(
        vdir / "coverage_by_year.csv")
    strata = quality_by_stratum(cube, fields, counts)
    strata.to_csv(vdir / "quality_by_stratum.csv", index=False)
    pstats, ptxt = physical(cube, fields)
    fstats, ftxt = freeze_and_condition(cube, fields)
    istats, itxt = ian_passes(rel, cube, fields)
    qstats, qtxt = quality(rel, cube, fields)
    acct = delineation_accounting(fields)
    pd.DataFrame(acct["steps"]).to_csv(vdir / "delineation_accounting.csv", index=False)
    estats = evidence(rel, fields)
    gc_all = 100 * (dstats["all"]["grassland"] + dstats["all"]["cropland"])
    summary = {"code": provenance.info(),
               "stats": {"delineation": dstats, "coverage": cstats, "physical": pstats, "events": fstats, "ian_passes": istats, "quality": qstats,
                         "accounting": acct, "evidence": estats},
               "ValidationFields": dtxt, "ValidationCoverage": ctxt, "ValidationPhysical": ptxt, "ValidationEvents": ftxt, "ValidationIanPasses": itxt,
               "ValidationQuality": qtxt,
               "ValidationSummary": ("that {g:.0f}\\,\\% of the field area (tier~A {gA:.0f}\\,\\%, tier~B {gB:.0f}\\,\\%) "
                                     "is grassland or cropland in the independent ESA WorldCover map and that the median "
                                     "field has {c:.0f} usable optical observations per year").format(
                   g=gc_all, gA=100 * (dstats["A"]["grassland"] + dstats["A"]["cropland"]),
                   gB=100 * (dstats["B"]["grassland"] + dstats["B"]["cropland"]),
                   c=cstats["usable_days_per_field_year_median"])}
    (vdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:])["stats"], indent=2, default=float))
