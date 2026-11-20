"""Write manuscript/numbers.tex and manuscript/tables/*.tex from the release, so every number in the paper is
computed, never typed. Missing inputs produce a visible [TBD] marker instead of a guess.

python scripts/paper_numbers.py [--release v1.0]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpcube import config

MS = Path(__file__).resolve().parents[1] / "manuscript"
TBD = r"\textbf{[TBD]}"


def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return TBD
    return f"{x:,.{nd}f}".replace(",", "\\,") if nd == 0 else f"{x:.{nd}f}"


def numbers(rel):
    n = {}
    st = config.INTERIM / "fields" / "delineation_stats.json"
    if st.exists():
        s = json.loads(st.read_text())
        n["NFields"] = fmt(s.get("n_fields"))
        n["AreaKha"] = fmt(s.get("in_fields_ha", np.nan) / 1e3)
        n["AreaMha"] = fmt(s.get("in_fields_ha", np.nan) / 1e6, 1)
        n["PersistentKha"] = fmt(s.get("persistent_pasture_ha", np.nan) / 1e3)
        n["WoodyRemovedKha"] = fmt(s.get("removed_by_woody_screen_ha", np.nan) / 1e3)
        n["NFieldsTierA"] = fmt(s.get("n_fields_tier_A"))
        n["NFieldsTierB"] = fmt(s.get("n_fields", 0) - s.get("n_fields_tier_A", 0) if "n_fields_tier_A" in s else None)
        n["AreaTierAKha"] = fmt(s.get("area_ha_tier_A", np.nan) / 1e3)
        n["AreaTierBKha"] = fmt(s.get("area_ha_tier_B", np.nan) / 1e3)
    n["PastureMinFrac"] = f"{int(round(config.PASTURE_MIN_FRACTION * len(config.PASTURE_YEARS)))}"   # "in at least 6 of the ten years"
    n["MinFieldHa"] = fmt(config.MIN_FIELD_HA, 0)
    from fpcube.cube import calendar
    n["NPeriods"] = fmt(len(calendar()))
    if rel and (rel / "optical_observations").exists():
        n["NOptObs"] = fmt(sum(pd.read_parquet(p, columns=["fid"]).shape[0]
                               for p in (rel / "optical_observations").glob("*.parquet")))
        n["NRadObs"] = fmt(sum(pd.read_parquet(p, columns=["fid"]).shape[0]
                               for p in (rel / "radar_observations").glob("*.parquet")))
    val = rel / "validation" / "summary.json" if rel else None
    if val and val.exists():
        v = json.loads(val.read_text())
        for k in ("ValidationSummary", "ValidationFields", "ValidationCoverage", "ValidationPhysical",
                  "ValidationQuality"):
            n[k] = v.get(k, TBD)
    if val and val.exists():
        e = json.loads(val.read_text())["stats"].get("evidence", {})
        pct = lambda x: (f"{100 * x:.{0 if 100 * x >= 1 else 1}f}\\,\\%" if x is not None and np.isfinite(x) else TBD)
        n["ScreenRangeNdviHigh"] = fmt(e.get("range_ndvi_diff_median", np.nan), 2)
        n["ScreenHazeNdviLow"] = fmt(-e.get("haze_ndvi_diff_median", np.nan), 2)
        n["ScreenHazeShareDrop"] = pct(e.get("haze_share_drop_gt_0.2", np.nan))
        n["WtMissingAll"] = pct(e.get("wt_missing_all", np.nan))
        n["WtMissingExcessive"] = pct(e.get("wt_missing_excessively", np.nan))
        n["WtMissingWell"] = pct(e.get("wt_missing_well", np.nan))
        n["WtMissingPoorlyMax"] = pct(e.get("wt_missing_poorly_max", np.nan))
        n["LandfallKm"] = fmt(e.get("landfall_km_median", np.nan))
        n["LandfallMin"] = fmt(e.get("landfall_dt_min_median", np.nan))
    if val and val.exists():
        st = json.loads(val.read_text())["stats"]
        acc, q, cv = st.get("accounting", {}), st.get("quality", {}), st.get("coverage", {})
        pct = lambda x: (f"{100 * x:.{0 if 100 * x >= 1 else 1}f}\\,\\%" if x is not None and np.isfinite(x) else TBD)
        n["WoodyWetlandKha"] = fmt(acc.get("woody_wetland_2plus_ha", np.nan) / 1e3, 1)
        n["WoodyWetlandShare"] = pct(acc.get("woody_wetland_2plus_share", np.nan))
        n["NMultipartFields"] = fmt(q.get("n_multipart_fields"))
        n["MultipartAreaShare"] = pct(q.get("multipart_area_share", np.nan))
        n["GeomNotAssessed"] = pct(cv.get("geom_not_assessed_frac", np.nan))
        n["UsableDaysMedian"] = fmt(cv.get("usable_days_per_field_year_median"))
        code = json.loads(val.read_text()).get("code", {})
        n["CodeCommit"] = code.get("commit", TBD)
        n["CodeHash"] = code.get("code_sha256_16", TBD)
    au = config.DATA / "audit" / "index_audit.json"           # scripts/audit/index_audit.py
    if au.exists():
        a = json.loads(au.read_text())
        n["IndexAuditScenes"] = str(a["n_items"])
        n["IndexAuditPixels"] = fmt(a["n_pixels"])
        n["IndexAuditUndefined"] = fmt(sum(a[k]["zero_den"] for k in ("ndvi", "evi", "ndmi", "mndwi", "nbr")))
        n["IndexAuditClippedMax"] = f"{100 * max(a[k]['outside_frac'] for k in ('ndvi', 'evi', 'ndmi', 'mndwi', 'nbr')):.4f}\\,\\%"
    from fpcube import cube as C
    from fpcube.sources import static as ST
    from fpcube.validation import NEIGHBOUR_DAYS
    n["ReflMin"], n["ReflMax"] = f"{C.REFL_RANGE[0]:g}", f"{C.REFL_RANGE[1]:g}"
    n["MaxBlue"] = f"{C.MAX_BLUE:g}"
    n["MinTanSlope"] = f"{ST.MIN_TAN_SLOPE:g}"
    n["NeighbourDays"] = str(NEIGHBOUR_DAYS)
    b = rel / "benchmark" if rel else None
    if b and (b / "task1_results.csv").exists():
        r = pd.read_csv(b / "task1_results.csv")
        best = r[(r.split == "test") & (r.horizon_days == 30) & r.model.isin(["lightgbm", "transformer"])]
        b_ = best.loc[best.skill_vs_ref.idxmax()]
        n["TaskOneSkill"] = f"{100 * b_.skill_vs_ref:.0f}\\,\\%"              # MSE skill score vs climatology
        n["TaskOneSkillModel"] = {"lightgbm": "LightGBM", "transformer": "transformer"}[b_.model]
    if b and (b / "benchmark_text.json").exists():
        n.update(json.loads((b / "benchmark_text.json").read_text()))
    dois = MS.parent / "zenodo" / "dois.json"           # {"DataDOI": "https://doi.org/...", "CodeDOI": "..."}
    if dois.exists():
        n.update(json.loads(dois.read_text()))
    n.setdefault("DataDOI", "https://doi.org/10.5281/zenodo.TBD")
    n.setdefault("CodeURL", "https://github.com/HugoMachadoRodrigues/fpcube")
    n.setdefault("CodeDOI", "https://doi.org/10.5281/zenodo.TBD")
    for k in ("NFields", "AreaKha", "AreaMha", "PersistentKha", "WoodyRemovedKha", "NFieldsTierA", "NFieldsTierB",
              "AreaTierAKha", "AreaTierBKha", "NOptObs", "CodeCommit", "CodeHash", "NRadObs", "ValidationSummary", "ValidationFields",
              "ValidationCoverage", "ValidationPhysical", "TaskOneSkill", "TaskTwoSummary", "BenchmarkResults"):
        n.setdefault(k, TBD)
    lines = ["%% generated by fpcube/scripts/paper_numbers.py -- do not edit"]
    lines += [f"\\newcommand{{\\fpc{k}}}{{{v}}}" for k, v in sorted(n.items())]
    (MS / "numbers.tex").write_text("\n".join(lines) + "\n")
    return n


LAYERS = [
    ("Fields", "CDL 2012--2025; FDOR parcels 2025", "30 m; parcel", "static", "geometry, tier, county, use code, area, shape, pasture fraction per year, flags"),
    ("Optical", "HLS v2.0 S30 and L30", "30 m; 2--3 d", "2016--2026", "blue, green, red, NIR, SWIR1, SWIR2; NDVI, EVI, NDMI, MNDWI, NBR; NDVI s.d.; water fraction; clear pixels; geometric shift"),
    ("Radar", "Sentinel-1 RTC (IW GRD)", "20 m; 6--12 d", "2016--2026", "VV, VH (linear and dB), VH/VV; orbit, pass, platform"),
    ("Weather", "PRISM daily", "4 km; daily", "2016--2026", "precipitation; mean, min, max temperature; max VPD"),
    ("Soil", "gNATSGO", "10 m raster; map unit", "static", "map unit, AWS 0--100 cm, SOC 0--100 cm, drainage class, flooding, hydrologic group, water-table depth"),
    ("Terrain", "3DEP 1 arc-second", "30 m", "static", "elevation, slope, TWI, HAND"),
    ("Wetlands", "NOAA C-CAP", "30 m", "latest", "wetland fraction in field and 500 m ring"),
    ("Drought", "US Drought Monitor", "county; weekly", "2016--2026", "percent area D0--D4 (cumulative), DSCI"),
    ("Condition", "NASS Crop Progress", "state; weekly", "2016--2026", "pasture condition shares and index"),
    ("Hurricanes", "HURDAT2; NHC b-decks", "6-hourly track", "2016--2026", "position, wind, pressure, status; landfalls"),
]


def tables(rel):
    (MS / "tables").mkdir(exist_ok=True)
    t = ["\\begin{tabular}{lllll}", "\\tophline", "Layer & Source & Resolution & Period & Variables\\\\", "\\middlehline"]
    t += [" & ".join(r[:4]) + " & \\parbox[t]{6.5cm}{\\raggedright " + r[4] + "}\\\\" for r in LAYERS]
    t += ["\\bottomhline", "\\end{tabular}"]
    (MS / "tables" / "layers.tex").write_text("\n".join(t) + "\n")

    fp = (rel / "fields.parquet") if rel and (rel / "fields.parquet").exists() else config.INTERIM / "fields" / "fields.parquet"
    if fp.exists():
        f = pd.read_parquet(fp, columns=["tier", "county", "area_ha"])
        region = f.county.map(REGION).fillna("Other")
        g = f.assign(region=region).groupby(["region", "tier"]).area_ha.agg(["size", "sum", "median"]).unstack("tier")
        t = ["\\begin{tabular}{lrrrr}", "\\tophline",
             "Region & Fields (A) & Area A (kha) & Fields (B) & Area B (kha)\\\\", "\\middlehline"]
        for reg, r in g.iterrows():
            t.append(f"{reg} & {fmt(r[('size', 'A')])} & {fmt(r[('sum', 'A')] / 1e3, 1)} & "
                     f"{fmt(r[('size', 'B')])} & {fmt(r[('sum', 'B')] / 1e3, 1)}\\\\")
        tot = f.groupby("tier").area_ha.agg(["size", "sum"])
        t += ["\\middlehline", f"Florida & {fmt(tot.loc['A', 'size'])} & {fmt(tot.loc['A', 'sum'] / 1e3, 1)} & "
              f"{fmt(tot.loc['B', 'size'])} & {fmt(tot.loc['B', 'sum'] / 1e3, 1)}\\\\", "\\bottomhline", "\\end{tabular}"]
        (MS / "tables" / "fields.tex").write_text("\n".join(t) + "\n")
    else:
        (MS / "tables" / "fields.tex").write_text(TBD + "\n")

    zarr_fmt = "Zarr v3" if rel and (rel / "cube.zarr" / "zarr.json").exists() else "Zarr v2"   # read from the store
    files = [("fields.parquet", "GeoParquet", "one row per field: geometry (EPSG:5070), static attributes, flags"),
             ("cube.zarr", zarr_fmt, "field $\\times$ time composites, observation counts, gap flag"),
             ("optical\\_observations/YYYY.parquet", "Parquet", "one row per field and HLS granule with quality fields"),
             ("radar\\_observations/YYYY.parquet", "Parquet", "one row per field and Sentinel-1 acquisition"),
             ("labels/*.parquet", "Parquet", "USDM county-week, NASS state-week, hurricane tracks and landfalls"),
             ("benchmark/", "Parquet, CSV", "sample indices, splits and baseline results of Tasks 1--3"),
             ("stac/catalog.json", "STAC 1.1", "catalog of all assets with checksums"),
             ("DATASHEET.md", "Markdown", "datasheet for the data set")]
    t = ["\\begin{tabular}{lll}", "\\tophline", "Path & Format & Content\\\\", "\\middlehline"]
    t += [f"\\texttt{{{a}}} & {b} & {c}\\\\" for a, b, c in files]
    t += ["\\bottomhline", "\\end{tabular}"]
    (MS / "tables" / "files.tex").write_text("\n".join(t) + "\n")

    b1 = rel / "benchmark" / "task1_results.csv" if rel else None
    if b1 and b1.exists():
        r = pd.read_csv(b1)
        names = {"persistence": "Persistence", "climatology": "Climatology", "lightgbm": "LightGBM",
                 "transformer": "Transformer"}
        t = ["\\begin{tabular}{llrrrr}", "\\tophline",
             "Test set & Model & 15 d & 30 d & 45 d & 60 d\\\\", "\\middlehline"]
        for split, lab in (("test", "Temporal (2025--2026)"), ("test_spatial", "Withheld counties")):
            for m, nm in names.items():
                d = r[(r.split == split) & (r.model == m)].set_index("horizon_days")
                cells = [f"{d.loc[h, 'rmse']:.3f} ({100 * d.loc[h, 'skill_vs_ref']:+.0f})" if h in d.index else "--"
                         for h in (15, 30, 45, 60)]
                t.append(f"{lab if m == 'persistence' else ''} & {nm} & " + " & ".join(cells) + "\\\\")
            d = r[(r.split == split) & (r.model == "lightgbm")].set_index("horizon_days")
            t.append(" & Samples & " + " & ".join(fmt(d.loc[h, "n"]) for h in (15, 30, 45, 60)) + "\\\\")
            t.append("\\middlehline" if split == "test" else "")
        t += ["\\bottomhline", "\\end{tabular}"]
        (MS / "tables" / "bench1.tex").write_text("\n".join(x for x in t if x) + "\n")
    else:
        (MS / "tables" / "bench1.tex").write_text(TBD + "\n")

    b2 = rel / "benchmark" / "task2_results.csv" if rel else None
    if b2 and b2.exists() and "experiment" in pd.read_csv(b2, nrows=1):
        from fpcube.benchmark.models import median_ci
        r = pd.read_csv(b2)
        r = r[~r.event_id.str.startswith("pooled_")]
        kinds = sorted(r.kind.unique())
        cols = [("leave_one_event_out", "rule_dNDVI_dMNDWI", "Rule"), ("leave_one_event_out", "lightgbm", "LightGBM"),
                ("leave_one_kind_out", "lightgbm", "LightGBM"), ("leave_one_kind_out", "transformer", "Transformer")]
        nev = r[r.experiment == "leave_one_event_out"].groupby("kind").event_id.nunique()
        t = ["\\begin{tabular}{lrcccc}", "\\tophline",
             " & & \\multicolumn{2}{c}{One event out} & \\multicolumn{2}{c}{One event type out}\\\\",
             "Event type & Events & " + " & ".join(c[2] for c in cols) + "\\\\", "\\middlehline"]
        for k in kinds:
            cells = []
            for exp, m, _ in cols:
                v = r[(r.experiment == exp) & (r.model == m) & (r.kind == k)].auroc
                lo, hi = median_ci(v)
                cells.append(f"{v.median():.2f} [{lo:.2f}, {hi:.2f}]")
            t.append(f"{k.replace('_', ' ').capitalize()} & {nev.get(k, 0)} & " + " & ".join(cells) + "\\\\")
        t += ["\\bottomhline", "\\end{tabular}"]
        (MS / "tables" / "bench2.tex").write_text("\n".join(t) + "\n")

    b3 = rel / "benchmark" / "task3_results.csv" if rel else None
    if b3 and b3.exists() and "recall_d2plus" in pd.read_csv(b3, nrows=1):
        r = pd.read_csv(b3)
        names = {"majority_class": "Majority class", "usdm_last_week": "Last week's USDM class",
                 "rule_anomaly_cuts": "NDVI-anomaly cut points", "lightgbm_bag_stats": "LightGBM, bag statistics",
                 "attention_mil": "Attention MIL"}
        t = ["\\begin{tabular}{llrrrrrrr}", "\\tophline",
             "Test set & Model & QWK & Macro F1 & Bal. acc. & \\multicolumn{4}{c}{Recall: none, D0, D1, D2+}\\\\",
             "\\middlehline"]
        for split, lab in (("test", "Temporal"), ("test_spatial", "Withheld counties")):
            for i, (m, nm) in enumerate(names.items()):
                x = r[(r.split == split) & (r.model == m)].iloc[0]
                t.append(f"{lab if i == 0 else ''} & {nm} & {x.qwk:.2f} [{x.qwk_lo:.2f}, {x.qwk_hi:.2f}] & "
                         f"{x.macro_f1:.2f} & {x.balanced_accuracy:.2f} & {x.recall_none:.2f} & {x.recall_d0:.2f} & "
                         f"{x.recall_d1:.2f} & {x.recall_d2plus:.2f}\\\\")
            x = r[(r.split == split)].iloc[0]
            t.append(f" & County-weeks per class & & & & {int(x.n_none)} & {int(x.n_d0)} & {int(x.n_d1)} & "
                     f"{int(x.n_d2plus)}\\\\")
            t.append("\\middlehline" if split == "test" else "")
        t += ["\\bottomhline", "\\end{tabular}"]
        (MS / "tables" / "bench3.tex").write_text("\n".join(x for x in t if x) + "\n")

    vd = rel / "validation" if rel else None
    if vd and (vd / "delineation_accounting.csv").exists():
        a = pd.read_csv(vd / "delineation_accounting.csv")
        t = ["\\begin{tabular}{lrr}", "\\tophline", "Step & Area (kha) & Fields\\\\", "\\middlehline"]
        for i, x in enumerate(a.itertuples()):
            step = x.step.replace(">=", "$\\geq$").replace("2016-2025", "2016--2025").replace("2012-2016", "2012--2016")
            area = x.area_ha / 1e3
            num = fmt(abs(area), 1).replace(",", "\\,")
            num = num if i in (0, len(a) - 1) else ("$-$" if area < 0 else "+") + num
            t.append(f"{step} & {num} & {fmt(x.n) if 'n' in a and pd.notna(x.n) else ''}\\\\")
        t += ["\\bottomhline", "\\end{tabular}"]
        (MS / "tables" / "accounting.tex").write_text("\n".join(t) + "\n")
    if vd and (vd / "quality_by_stratum.csv").exists():
        q = pd.read_csv(vd / "quality_by_stratum.csv")
        t = ["\\begin{tabular}{llrrrrrr}", "\\tophline",
             "Tier & Size & Fields & Obs. Oct--May & Obs. Jun--Sep & Interp. & Empty & Usable days yr$^{-1}$\\\\",
             "\\middlehline"]
        for i, x in enumerate(q.itertuples()):
            t.append(f"{x.tier if i % 4 == 0 else ''} & {x.size} & {fmt(x.n_fields)} & {100 * x.observed_dry:.0f}\\,\\% & "
                     f"{100 * x.observed_wet:.0f}\\,\\% & {100 * x.interpolated:.0f}\\,\\% & {100 * x.empty:.0f}\\,\\% & "
                     f"{x.usable_days_per_year_median:.0f}\\\\")
        t += ["\\bottomhline", "\\end{tabular}"]
        (MS / "tables" / "strata.tex").write_text("\n".join(t) + "\n")


# Florida regions used by UF/IFAS Extension districts (simplified)
_R = {"Northwest": "Bay Calhoun Escambia Franklin Gadsden Gulf Holmes Jackson Jefferson Leon Liberty Madison Okaloosa "
                   "Santa_Rosa Taylor Wakulla Walton Washington",
      "Northeast": "Alachua Baker Bradford Clay Columbia Dixie Duval Flagler Gilchrist Hamilton Lafayette Levy Marion "
                   "Nassau Putnam St._Johns Suwannee Union",
      "Central": "Brevard Citrus Hernando Lake Orange Osceola Pasco Pinellas Polk Seminole Sumter Volusia Hillsborough",
      "South": "Broward Charlotte Collier DeSoto Glades Hardee Hendry Highlands Indian_River Lee Manatee Martin "
               "Miami-Dade Monroe Okeechobee Palm_Beach Sarasota St._Lucie"}
REGION = {c.replace("_", " "): r for r, cs in _R.items() for c in cs.split()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="v1.0")
    a = ap.parse_args()
    rel = config.RELEASE / a.release
    rel = rel if rel.exists() else None
    n = numbers(rel)
    tables(rel)
    print(sum(v == TBD for v in n.values()), "numbers still TBD")
