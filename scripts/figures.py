"""Figures for the ESSD paper. Each figure is skipped (with a message) if its inputs do not exist yet.

python scripts/figures.py [fig1 fig2 fig3 fig4 fig5] [--release v1.0]
"""
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpcube import config

OUT = Path(__file__).resolve().parents[2] / "manuscript" / "figures"
# categorical order (fixed): blue, orange, aqua, yellow, magenta, green, violet, red
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = mpl.colors.LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
DIV = mpl.colors.LinearSegmentedColormap.from_list("div", ["#a0461f", "#eb6834", "#f0efec", "#2a78d6", "#184f95"])
ONA = (-81.93973, 27.3975)
FIG3_START = "2021-01-01"   # Fig. 3 shows the last six seasons (Ian, Idalia, Helene, Milton, 2025-26 drought)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 9, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "lines.linewidth": 1.4,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def counties():
    return gpd.read_parquet(config.RAW / "boundaries" / "fl_counties.parquet")


def fields_table(rel):
    p = rel / "fields.parquet" if rel and (rel / "fields.parquet").exists() else config.INTERIM / "fields" / "fields.parquet"
    return gpd.read_parquet(p)


def fig1(rel):
    f = fields_table(rel)
    co = counties()
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_axes([0, 0, 0.66, 1])
    co.boundary.plot(ax=ax, color=AXIS, linewidth=0.3)
    for tier, col, lab in (("B", C[1], "Tier B: other parcel codes"), ("A", C[0], "Tier A: grazing-coded parcels")):
        g = f[f.tier == tier]
        g.plot(ax=ax, color=col, linewidth=0, rasterized=True, label=f"{lab} ({len(g):,} fields)")
    ona = gpd.GeoSeries(gpd.points_from_xy([ONA[0]], [ONA[1]]), crs=4326).to_crs(f.crs)
    ax.plot(ona.x, ona.y, marker="*", ms=9, color=INK, mec="white", mew=0.8, ls="none")
    import matplotlib.patheffects as pe
    ax.annotate("Ona (RCREC)", (ona.x.iloc[0], ona.y.iloc[0]), xytext=(-90, -40), textcoords="offset points",
                color=INK, fontsize=7.5, arrowprops=dict(arrowstyle="-", color=INK, lw=0.6),
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=7.5, handlelength=1, handleheight=1)
    ax.set_title("a  Persistent pasture fields, Florida", x=0.02, y=0.97)
    # county density (field area share of county)
    ax2 = fig.add_axes([0.66, 0.52, 0.34, 0.46])
    a = f.groupby("county_fips").area_ha.sum()
    cc = co.merge(a.rename("field_ha"), left_on="GEOID", right_index=True, how="left")
    cc["share"] = 100 * cc["field_ha"].fillna(0) / (cc.geometry.area / 1e4)
    cc.plot(ax=ax2, column="share", cmap=SEQ, linewidth=0.2, edgecolor="white", legend=True,
            legend_kwds={"shrink": 0.6, "label": "% of county area in fields"})
    ax2.set_axis_off()
    ax2.set_title("b  Pasture share by county", x=0.02)
    # inset around Ona, 20 km box
    ax3 = fig.add_axes([0.68, 0.02, 0.3, 0.46])
    x, y = ona.x.iloc[0], ona.y.iloc[0]
    box = (x - 10000, y - 10000, x + 10000, y + 10000)
    sub = f.cx[box[0]:box[2], box[1]:box[3]]
    for tier, col in (("B", C[1]), ("A", C[0])):
        sub[sub.tier == tier].plot(ax=ax3, color=col, edgecolor="white", linewidth=0.4, rasterized=True)
    ax3.plot(x, y, marker="*", ms=10, color=INK, mec="white", mew=0.8)
    ax3.set_xlim(box[0], box[2]); ax3.set_ylim(box[1], box[3])
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.grid(False); ax3.set_xlabel(""); ax3.set_ylabel("")
    for s in ax3.spines.values():
        s.set_visible(True); s.set_color(AXIS)
    ax3.add_patch(mpl.patches.Rectangle((box[0] + 400, box[1] + 400), 6200, 2000, color="white", zorder=4))
    ax3.plot([box[0] + 1000, box[0] + 6000], [box[1] + 900] * 2, color=INK, lw=2, zorder=5)
    ax3.text(box[0] + 3500, box[1] + 1300, "5 km", ha="center", fontsize=7, color=INK, zorder=5)
    ax3.set_title("c  Fields around Ona (20 × 20 km)", x=0.02)
    save(fig, "fig1_map")


def fig2(rel):
    if not rel or not (rel / "optical_observations").exists():
        print("fig2: no release yet"); return
    import xarray as xr
    opt = pd.concat(pd.read_parquet(p, columns=["fid", "date", "sensor", "qa_clear", "qa_geometry"])
                    for p in (rel / "optical_observations").glob("*.parquet"))
    rad = pd.concat(pd.read_parquet(p, columns=["fid", "date", "platform"])
                    for p in (rel / "radar_observations").glob("*.parquet"))
    opt = opt[opt.qa_clear & opt.qa_geometry]
    opt["year"] = pd.to_datetime(opt.date).dt.year
    rad["year"] = pd.to_datetime(rad.date).dt.year
    nf = len(pd.read_parquet(rel / "fields.parquet", columns=["fid"]))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    o = opt.groupby(["year", "sensor"]).size().unstack(fill_value=0) / nf
    bottom = np.zeros(len(o))
    for k, s in enumerate(o.columns):
        axs[0].bar(o.index, o[s], bottom=bottom, color=C[k], width=0.8, edgecolor="white", linewidth=1,
                   label={"L30": "Landsat 8/9 (L30)", "S30": "Sentinel-2 (S30)"}.get(s, s))
        bottom += o[s].to_numpy()
    axs[0].set_title("a  Clear HLS obs. per field")
    axs[0].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    r = rad.groupby("year").size() / nf
    axs[1].bar(r.index, r.values, color=C[0], width=0.8)
    axs[1].set_title("b  Sentinel-1 acq. per field")
    cube = xr.open_zarr(rel / "cube.zarr")
    has = (cube.n_optical_obs > 0).mean("field").to_series()
    m = has.groupby(has.index.month).mean()
    axs[2].bar(m.index, 100 * m.values, color=C[0], width=0.7)
    axs[2].set_xticks(range(1, 13), list("JFMAMJJASOND"))
    axs[2].set_title("c  Periods with optical data (%)")
    for a in axs[:2]:
        a.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
        a.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))
    for a in axs:
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig2_coverage")


def fig5(rel):
    b = rel / "benchmark" if rel else None
    if not b or not (b / "task1_results.csv").exists():
        print("fig5: no benchmark yet"); return
    r1 = pd.read_csv(b / "task1_results.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    order = ["persistence", "climatology", "lightgbm", "transformer"]
    for k, m in enumerate(order):
        d = r1[(r1.split == "test") & (r1.model == m)].sort_values("horizon_days")
        axs[0].plot(d.horizon_days, d.rmse, marker="o", ms=4, color=C[k],
                    label={"lightgbm": "LightGBM"}.get(m, m.capitalize()))
    axs[0].set_xlabel("Lead time (days)"); axs[0].set_ylabel("NDVI RMSE")
    axs[0].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    axs[0].set_title("a  Task 1 (temporal test)")
    if (b / "task2_results.csv").exists():
        r2 = pd.read_csv(b / "task2_results.csv")
        r2 = r2[~r2.event_id.str.startswith("all_")]
        kinds = sorted(r2.kind.unique())
        for k, m in enumerate(["rule_dNDVI_dMNDWI", "lightgbm"]):
            d = r2[r2.model == m]
            pos = [kinds.index(x) + (k - 0.5) * 0.3 for x in d.kind]
            axs[1].scatter(pos, d.auroc, s=14, color=C[k], zorder=3,
                           label={"lightgbm": "LightGBM"}.get(m, "ΔNDVI–ΔMNDWI rule"))
        axs[1].set_xticks(range(len(kinds)), kinds, rotation=20)
        axs[1].axhline(0.5, color=MUTED, lw=0.8, ls="--")
        axs[1].set_ylabel("AUROC (per event)")
        axs[1].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
        axs[1].set_title("b  Task 2 (leave-one-event-out)")
    if (b / "task3_confusion.csv").exists():
        cm = pd.read_csv(b / "task3_confusion.csv", index_col=0).to_numpy()
        cmn = cm / cm.sum(1, keepdims=True)
        axs[2].imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                axs[2].text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7,
                            color="white" if cmn[i, j] > 0.5 else INK)
        lab = ["None", "D0", "D1", "D2+"]
        axs[2].set_xticks(range(4), lab); axs[2].set_yticks(range(4), lab); axs[2].grid(False)
        axs[2].set_xlabel("Predicted"); axs[2].set_ylabel("USDM")
        axs[2].set_title("c  Task 3 (best model)")
    fig.tight_layout()
    save(fig, "fig5_benchmark")



def _cube(rel):
    import xarray as xr
    return xr.open_zarr(rel / "cube.zarr")


def _hurricanes():
    p = config.RAW / "labels" / "hurricane_landfalls.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def fig3(rel):
    """Two example fields: NDVI (observed vs gap-filled), VV and VH, 5-day rain, county drought class."""
    if not rel or not (rel / "cube.zarr").exists():
        print("fig3: no release yet"); return
    cube = _cube(rel)
    f = fields_table(rel)
    ona = gpd.GeoSeries(gpd.points_from_xy([ONA[0]], [ONA[1]]), crs=4326).to_crs(f.crs).iloc[0]
    nobs = cube.n_optical_obs.sum("time").values
    cand = f.assign(nobs=nobs, d=f.geometry.distance(ona))
    a = cand[(cand.tier == "A") & (cand.area_ha > 20)].sort_values("d").iloc[0]          # near Ona
    b = cand[(cand.tier == "A") & (cand.area_ha > 20) & (cand.lat > 30.3)].sort_values("nobs").iloc[-1]  # panhandle
    usdm = pd.read_parquet(config.RAW / "labels" / "usdm_county_weekly.parquet")
    times = pd.DatetimeIndex(cube.time.values)
    hur = _hurricanes()
    fig, axs = plt.subplots(4, 2, figsize=(7.2, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.4, 1, 0.5]})
    for j, row in enumerate((a, b)):
        i = int(np.nonzero(cube.field.values == row.field_id)[0][0])
        c = cube.isel(field=i).load()
        nd, gf, n = c.ndvi.values, c.gap_filled.values.astype(bool), c.n_optical_obs.values
        ax = axs[0, j]
        ax.plot(times, np.where(np.isfinite(nd), nd, np.nan), color=C[0], lw=1)
        ax.plot(times[n > 0], nd[n > 0], "o", ms=2.2, color=C[0], label="observed composite")
        ax.plot(times[gf], nd[gf], "o", ms=2.2, mfc="white", mec=C[0], mew=0.6, label="gap-filled")
        ax.set_ylabel("NDVI")
        ax.set_title(f"{'ab'[j]}  {row.field_id}, {row.county} County, {row.area_ha:.0f} ha")
        if j == 1:
            ax.legend(fontsize=6.5, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, borderaxespad=0.2)
        ax = axs[1, j]
        ax.plot(times, c.vv_db.values, color=C[2], lw=0.9, label="VV")
        ax.plot(times, c.vh_db.values, color=C[6], lw=0.9, label="VH")
        ax.set_ylabel("$\\gamma^0$ (dB)")
        if j == 1:
            ax.legend(fontsize=6.5, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, borderaxespad=0.2)
        ax = axs[2, j]
        ax.bar(times, c.ppt_sum.values, width=4, color=C[0], linewidth=0)
        ax.set_ylabel("Rain (mm/5 d)")
        ax = axs[3, j]
        u = usdm[usdm.county_fips == row.county_fips].sort_values("valid_start")
        cls = np.select([u.D2 >= 50, u.D1 >= 50, u.D0 >= 50], [3, 2, 1], 0)
        cmap = mpl.colors.ListedColormap(["#f0efec", "#fab219", "#ec835a", "#d03b3b"])
        ax.pcolormesh(np.r_[u.valid_start.values, u.valid_end.values[-1:]], [0, 1], cls[None, :], cmap=cmap,
                      vmin=-0.5, vmax=3.5)
        ax.set_yticks([]); ax.grid(False)
        ax.set_ylabel("USDM", rotation=0, ha="right", va="center")
        for ax in axs[:, j]:
            ax.set_xlim(pd.Timestamp(FIG3_START), times[-1] + pd.Timedelta(days=5))
            for t in hurricane_days(hur):
                ax.axvline(t, color=MUTED, lw=0.6, ls="--", zorder=0)
    handles = [mpl.patches.Patch(color=c_, label=l_) for c_, l_ in
               zip(["#f0efec", "#fab219", "#ec835a", "#d03b3b"], ["None", "D0", "D1", "D2+"])]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, title="County drought (USDM, >= 50 % area)",
               title_fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, "fig3_examples")


def hurricane_days(hur):
    """Florida landfalls at hurricane strength (HU), one date per storm."""
    if not len(hur):
        return []
    h = hur[(hur.status == "HU") & hur.lat.between(24.4, 31.1) & hur.lon.between(-87.7, -79.9)]
    return list(pd.to_datetime(h.groupby("storm_id").time.min()))


def _traj(a, k, rows, pre=12, post=12):
    seg = a[rows][:, k - pre:k + post + 1]
    return np.arange(-pre, post + 1) * config.COMPOSITE_DAYS, np.nanmedian(seg, 0), np.nanpercentile(seg, 25, 0), \
        np.nanpercentile(seg, 75, 0)


def fig4(rel):
    """Field response to Hurricane Ian (2022) and the January 2018 freeze."""
    if not rel or not (rel / "cube.zarr").exists():
        print("fig4: no release yet"); return
    cube = _cube(rel)
    f = fields_table(rel)
    co = counties()
    times = pd.DatetimeIndex(cube.time.values)
    ppt = cube.ppt_sum.values
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.6), gridspec_kw={"width_ratios": [1.7, 1, 1]})
    events = [("Hurricane Ian, 28 Sep 2022", pd.Timestamp("2022-09-28"), "vv_db", "ndvi"),
              ("Freeze, 17 Jan 2018", pd.Timestamp("2018-01-17"), "ndvi", "vv_db")]
    for r, (title, day, mapvar, other) in enumerate(events):
        k = int(np.searchsorted(times, day))
        if r == 0:
            rain = np.nansum(ppt[:, k - 1:k + 1], axis=1)
            pos, neg = rain >= 200, rain < 50
            lab_pos, lab_neg = "≥ 200 mm in 10 d", "< 50 mm"
        else:
            tmin = cube.tmin_mean.values[:, k] if "tmin_mean" in cube else np.full(len(f), np.nan)
            pos, neg = tmin <= 2, tmin >= 8
            lab_pos, lab_neg = "5-day mean Tmin ≤ 2 °C", "≥ 8 °C"
        a = cube[mapvar].values
        delta = np.nanmedian(a[:, k + 1:k + 4], 1) - np.nanmedian(a[:, k - 6:k], 1)
        ax = axs[r, 0]
        co.boundary.plot(ax=ax, color=AXIS, linewidth=0.25)
        lim = np.nanpercentile(np.abs(delta), 95)
        pts = f.geometry.representative_point()
        ok = np.isfinite(delta)
        ax.scatter(pts.x[ok], pts.y[ok], c=delta[ok], s=0.8, cmap=DIV,
                   vmin=-lim, vmax=lim, rasterized=True, linewidths=0)
        sm = mpl.cm.ScalarMappable(cmap=DIV, norm=mpl.colors.Normalize(-lim, lim))
        fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01,
                     label=f"Δ{'NDVI' if mapvar == 'ndvi' else 'VV (dB)'}, 15 d after vs 30 d before")
        ax.set_axis_off()
        ax.set_title(f"{'ad'[r]}  {title}", fontsize=8)
        for c_, (var, lab) in enumerate(((mapvar, None), (other, None))):
            ax = axs[r, 1 + c_]
            arr = cube[var].values
            for rows, col, l_ in ((pos, C[1], lab_pos), (neg, C[0], lab_neg)):
                if rows.sum() < 10:
                    continue
                x, m, lo, hi = _traj(arr, k, rows)
                ax.fill_between(x, lo, hi, color=col, alpha=0.18, linewidth=0)
                ax.plot(x, m, color=col, label=f"{l_} (n = {rows.sum():,})")
            ax.axvline(0, color=MUTED, lw=0.7, ls="--")
            ax.set_xlabel("Days from event")
            ax.set_ylabel("NDVI" if var == "ndvi" else "VV (dB)")
            ax.set_title(f"{'bcef'[2 * r + c_]}", fontsize=8)
            if c_ == 0:
                ax.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=1)
    fig.tight_layout()
    save(fig, "fig4_events")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=["fig1", "fig2", "fig3", "fig4", "fig5"])
    ap.add_argument("--release", default="v1.0")
    a = ap.parse_args()
    rel = config.RELEASE / a.release
    rel = rel if rel.exists() else None
    for f in a.figs:
        globals()[f](rel)
