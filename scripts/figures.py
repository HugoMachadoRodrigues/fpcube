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
        axs[0].bar(o.index, o[s], bottom=bottom, color=C[k], width=0.8, label=s, edgecolor="white", linewidth=1)
        bottom += o[s].to_numpy()
    axs[0].set_title("a  Clear HLS obs. per field"); axs[0].legend(fontsize=7)
    r = rad.groupby("year").size() / nf
    axs[1].bar(r.index, r.values, color=C[0], width=0.8)
    axs[1].set_title("b  Sentinel-1 acq. per field")
    cube = xr.open_zarr(rel / "cube.zarr")
    has = (cube.n_optical_obs > 0).mean("field").to_series()
    m = has.groupby(has.index.month).mean()
    axs[2].bar(m.index, 100 * m.values, color=C[0], width=0.7)
    axs[2].set_xticks(range(1, 13), list("JFMAMJJASOND"))
    axs[2].set_title("c  Periods with optical data (%)")
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
        axs[0].plot(d.horizon_days, d.rmse, marker="o", ms=4, color=C[k], label=m)
    axs[0].set_xlabel("Lead time (days)"); axs[0].set_ylabel("NDVI RMSE"); axs[0].legend(fontsize=7)
    axs[0].set_title("a  Task 1 (temporal test)")
    if (b / "task2_results.csv").exists():
        r2 = pd.read_csv(b / "task2_results.csv")
        r2 = r2[~r2.event_id.str.startswith("all_")]
        kinds = sorted(r2.kind.unique())
        for k, m in enumerate(["rule_dNDVI_dMNDWI", "lightgbm"]):
            d = r2[r2.model == m]
            pos = [kinds.index(x) + (k - 0.5) * 0.3 for x in d.kind]
            axs[1].scatter(pos, d.auroc, s=14, color=C[k], label=m.replace("_dNDVI_dMNDWI", ""), zorder=3)
        axs[1].set_xticks(range(len(kinds)), kinds, rotation=20)
        axs[1].axhline(0.5, color=MUTED, lw=0.8, ls="--")
        axs[1].set_ylabel("AUROC (per event)"); axs[1].legend(fontsize=7)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=["fig1", "fig2", "fig5"])
    ap.add_argument("--release", default="v1.0")
    a = ap.parse_args()
    rel = config.RELEASE / a.release
    rel = rel if rel.exists() else None
    for f in a.figs:
        globals()[f](rel)
