"""Read-only audit of the static field table (release fields.parquet): identity, geometry, delineation attributes,
soil, terrain, wetlands. Internal consistency and plausibility, not a comparison with field truth."""
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
from fpcube import config
from fpcube.sources import static as ST

path = sys.argv[1] if len(sys.argv) > 1 else config.RELEASE / "v1.0" / "fields.parquet"
f = gpd.read_parquet(path)
issues = []
def issue(m): issues.append(m); print("  !!", m, flush=True)
def say(*a): print(*a, flush=True)

say("== identity and geometry", f.crs.to_epsg(), len(f))
if f.field_id.duplicated().any() or f.fid.duplicated().any(): issue("duplicate ids")
bad = ~f.geometry.is_valid
if bad.any(): issue(f"{int(bad.sum())} invalid geometries")
px_ha = 0.09
if (abs(f.area_ha - f.n_pixels * px_ha) > 1e-6).any(): issue("area_ha != n_pixels * 0.09")
if (f.area_ha < config.MIN_FIELD_HA - 1e-9).any(): issue(f"fields below MIN_FIELD_HA {config.MIN_FIELD_HA}")
r = (f.geometry.area / 1e4) / f.area_ha
say(f"polygon area / pixel area: median {r.median():.3f}, p1 {r.quantile(.01):.3f}, p99 {r.quantile(.99):.3f}")
if (r.sub(1).abs() > 0.05).any(): issue(f"{int((r.sub(1).abs() > 0.05).sum())} fields whose polygon area differs from pixel area by >5%")
cen = f.geometry.centroid.to_crs(4326)
if (abs(cen.x - f.lon) > 1e-4).any() or (abs(cen.y - f.lat) > 1e-4).any():
    say(f"  lon/lat vs polygon centroid: max diff {max(abs(cen.x - f.lon).max(), abs(cen.y - f.lat).max()):.2e} deg "
        "(lon/lat may be a representative point)")
if f.groupby("county").county_fips.nunique().max() > 1 or f.groupby("county_fips").county.nunique().max() > 1:
    issue("county and county_fips are not 1:1")
co = gpd.read_parquet(config.RAW / "boundaries" / "fl_counties.parquet").to_crs(f.crs).set_index("GEOID").geometry
ov = gpd.overlay(f[["fid", "county_fips", "geometry"]], co.reset_index()[["GEOID", "geometry"]], how="intersection",
                 keep_geom_type=True)
ov["a"] = ov.geometry.area
own = ov[ov.county_fips == ov.GEOID].groupby("fid").a.sum().reindex(f.fid).fillna(0).values / f.geometry.area.values
say(f"share of field area in its county: min {own.min():.3f}; fields < 0.5: {int((own < 0.5).sum())}")
if (own < 0.5 - 1e-9).any(): issue(f"{int((own < 0.5).sum())} fields have less than half their area in their county")
say("tier by DOR use code (A codes):", sorted(f[f.tier == "A"].dor_use_code.unique().tolist()))
if f[f.tier == "A"].dor_use_code.isin(f[f.tier == "B"].dor_use_code).any(): issue("a DOR use code appears in both tiers")

say("== delineation attributes")
pf = [c for c in f.columns if c.startswith("pasture_frac_")]
if ((f[pf] < 0) | (f[pf] > 1)).any().any(): issue("pasture_frac outside [0, 1]")
if (f.n_interior_pixels > f.n_pixels).any(): issue("n_interior_pixels > n_pixels")
if (f.flag_few_interior != (f.n_interior_pixels < 5)).any(): issue("flag_few_interior inconsistent")
last3 = sorted(pf)[-3:]
if (f.flag_landuse_change != (f[last3].mean(axis=1) < 0.5)).any(): issue("flag_landuse_change inconsistent")
say(f"flag_few_interior {f.flag_few_interior.mean():.3f}, flag_landuse_change {f.flag_landuse_change.mean():.3f}; "
    f"n_interior_pixels == 0: {int((f.n_interior_pixels == 0).sum())}")
say("pasture_frac median by year:", f[pf].median().round(3).to_dict())

say("== soil")
if ((f.mukey_frac <= 0) | (f.mukey_frac > 1)).any(): issue("mukey_frac outside (0, 1]")
inv = {v: k for k, v in ST.DRAIN.items()}
m = f.drainage_class.notna()
if (f.drainage_class[m].map(ST.DRAIN) != f.drainage_code[m]).any(): issue("drainage_code does not match drainage_class")
m = f.flooding_frequency.notna()
if (f.flooding_frequency[m].map(ST.FLOOD) != f.flooding_code[m]).any(): issue("flooding_code does not match flooding_frequency")
say("hydrologic groups:", f.hydrologic_group.value_counts(dropna=False).to_dict())
say("drainage classes:", f.drainage_class.value_counts(dropna=False).to_dict())
wt = f.water_table_min_cm
say(f"water_table_min_cm: missing {wt.isna().mean():.3f}, range {wt.min()}..{wt.max()}, values > 200: {int((wt > 200).sum())}")
tab = f.assign(wt_missing=wt.isna()).groupby("drainage_class", dropna=False).agg(
    n=("fid", "size"), wt_missing=("wt_missing", "mean"), wt_median=("water_table_min_cm", "median"))
say(tab.round(3).to_string())
say(f"aws0_100_mm median {f.aws0_100_mm.median():.1f} [{f.aws0_100_mm.min()}, {f.aws0_100_mm.max()}]; "
    f"soc0_100_gm2 median {f.soc0_100_gm2.median():.0f} [{f.soc0_100_gm2.min()}, {f.soc0_100_gm2.max()}]")
say("soc by drainage class (median g/m2):", f.groupby("drainage_class").soc0_100_gm2.median().round(0).to_dict())
say("muname containing 'muck' or 'Histosol': soc median",
    f[f.muname.str.contains("muck|Histosol", case=False, na=False)].soc0_100_gm2.median(), "n",
    int(f.muname.str.contains("muck|Histosol", case=False, na=False).sum()))
say("missing soil rows:", int(f.drainage_class.isna().sum()), "| muname of those:",
    f[f.drainage_class.isna()].muname.value_counts().head(5).to_dict())

say("== terrain")
if (f.elev_min_m > f.elev_mean_m + 1e-6).any() or (f.elev_mean_m > f.elev_max_m + 1e-6).any(): issue("elevation min/mean/max out of order")
if (f.hand_min_m > f.hand_mean_m + 1e-6).any(): issue("hand_min > hand_mean")
if (f.hand_mean_m > f.elev_mean_m + 1).any(): issue(f"{int((f.hand_mean_m > f.elev_mean_m + 1).sum())} fields with HAND above elevation")
say(f"slope_mean_deg p99 {f.slope_mean_deg.quantile(.99):.2f}; twi missing {f.twi_mean.isna().mean():.3f}; "
    f"hand median by drainage: {f.groupby('drainage_class').hand_mean_m.median().round(2).to_dict()}")
tm = f.twi_mean.isna()
say(f"twi missing fields: area median {f.area_ha[tm].median():.1f} ha, n_interior median {f.n_interior_pixels[tm].median()}, "
    f"slope median {f.slope_mean_deg[tm].median():.3f}")

say("== wetlands")
for c in ("wetland_frac", "wetland_frac_500m"):
    say(f"{c}: missing {f[c].isna().mean():.4f}, share > 0 {(f[c] > 0).mean():.3f}, median {f[c].median():.3f}, "
        f"p90 {f[c].quantile(.9):.3f}, max {f[c].max():.3f}")
if not (f.wetland_frac_500m > 0).any(): issue("wetland fractions are zero everywhere")
for c in ("wetland_frac", "wetland_frac_500m"):
    if ((f[c] < 0) | (f[c] > 1 + 1e-9)).any(): issue(f"{c} outside [0, 1]")
if f.twi_mean.isna().any(): issue(f"{int(f.twi_mean.isna().sum())} fields without TWI")
say("wetland_frac_500m median by drainage:", f.groupby("drainage_class").wetland_frac_500m.median().round(3).to_dict())
say("ccap_year:", f.ccap_year.unique().tolist())

say("== missing values per column (non-zero)")
say({c: round(float(v), 4) for c, v in f.drop(columns="geometry").isna().mean().items() if v > 0})
say("\n== ISSUES"); [say("-", i) for i in issues]
