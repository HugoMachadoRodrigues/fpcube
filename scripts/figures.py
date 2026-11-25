"""Figures for the manuscript, numbered as in the text. Each figure is skipped (with a message) if its inputs do not
exist yet.

python scripts/figures.py [fig1 ... fig6] [--release v1.0]

Design rules shared by all figures: 180 mm wide, Helvetica, panel titles that state the finding with a one-line
subtitle saying what is plotted, direct labels instead of legends where possible, reference models in greys and
learned models in colour, observed optical values only where a response is measured, perceptually uniform colour
maps (Crameri) for continuous fields, official US Drought Monitor colours for drought classes.
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import cmcrameri.cm as cmc
import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

from fpcube import config

OUT = Path(os.environ.get("FPCUBE_FIG_OUT", Path(__file__).resolve().parents[1] / "manuscript" / "figures"))
W = 7.09                                                     # 180 mm
# categorical slots 1-3 (validated all-pairs for colour-vision deficiency); aqua is always direct-labelled
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_L, BLUE_D = "#9cc3ee", "#0d366b"
VIOLET, VIOLET_L = "#4b3bb8", "#a59cec"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e6e5df", "#b9b7ae"
REF_D, REF_L = "#52514e", "#a9a7a0"                         # reference (non-learned) models
LAND, SEA, CO_EDGE = "#f3f2ed", "#e9eff4", "#d4d2c9"
HAND_C = ["#0d366b", "#2a78d6", "#86b6ef"]                   # HAND <= 1, 1-3, >= 3 m (darker = lower, wetter)
USDM_C = ["#efeee9", "#ffff00", "#fcd37f", "#ffaa00", "#e60000", "#730000"]   # none, D0-D4 (official)
ONA = (-81.93973, 27.3975)
SERIES_START = "2021-01-01"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 6.5,
    "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.2, "legend.frameon": False, "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "figure.facecolor": "white",
    "hatch.linewidth": 0.6})
STROKE = [pe.withStroke(linewidth=2.2, foreground="white")]


def num(n):
    """Integer with a thin space as thousands separator."""
    return f"{int(round(n)):,}".replace(",", " ")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def head(ax, letter, title, sub=None, pad=3):
    """Panel letter and a title that states the finding, with an optional subtitle saying what is plotted."""
    kw = dict(xycoords="axes fraction", textcoords="offset points", va="bottom", annotation_clip=False)
    y = pad + (9 if sub else 0)
    ax.annotate(letter, (0, 1), xytext=(-1, y - 0.5), fontsize=8.5, fontweight="bold", color=INK, ha="right", **kw)
    ax.annotate(title, (0, 1), xytext=(2, y), fontsize=7.2, fontweight="bold", color=INK, ha="left", **kw)
    if sub:
        ax.annotate(sub, (0, 1), xytext=(2, pad), fontsize=6, color=INK2, ha="left", **kw)


def spread(ys, gap):
    """Move label positions apart so that neighbours are at least `gap` apart, keeping their order."""
    o = np.argsort(ys)
    y = np.asarray(ys, float)[o].copy()
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + gap)
    shift = (np.asarray(ys, float)[o].mean() - y.mean())
    out = np.empty_like(y)
    out[o] = y + shift
    return out


def map_axes(ax, bounds, pad=0.02):
    x0, y0, x1, y1 = bounds
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)


def scalebar(ax, x, y, km, label=None, h=None):
    h = h or km * 1000 * 0.06
    for k in range(2):
        ax.add_patch(Rectangle((x + k * km * 500, y), km * 500, h, facecolor=INK if k == 0 else "white",
                               edgecolor=INK, linewidth=0.5, zorder=6))
    ax.text(x, y + h * 1.6, "0", fontsize=5.5, ha="center", va="bottom", color=INK, zorder=6)
    ax.text(x + km * 1000, y + h * 1.6, label or f"{km:g} km", fontsize=5.5, ha="center", va="bottom", color=INK,
            zorder=6)


def north(ax, x, y, size):
    ax.annotate("", (x, y + size), xytext=(x, y), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7,
                                                                   mutation_scale=7))
    ax.text(x, y + size * 1.15, "N", ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=INK)


def counties():
    return gpd.read_parquet(config.RAW / "boundaries" / "fl_counties.parquet")


def fields_table(rel):
    p = rel / "fields.parquet" if rel and (rel / "fields.parquet").exists() else config.FIELDS_DIR / "fields.parquet"
    return gpd.read_parquet(p)


def _cube(rel):
    import xarray as xr
    return xr.open_zarr(rel / "cube.zarr")


def _landfalls():
    p = config.RAW / "labels" / "hurricane_landfalls.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["name", "time"])
    h = pd.read_parquet(p)
    h = h[(h.status == "HU") & h.lat.between(24.4, 31.1) & h.lon.between(-87.7, -79.9)]
    return h.groupby("storm_id").agg(name=("name", "first"), time=("time", "min")).sort_values("time")


# ---------------------------------------------------------------------------------------------------------------- 1
def fig1(rel):
    """Study area: fields by tier, field size, and a zoom on the fields around Ona."""
    f = fields_table(rel)
    co = counties().to_crs(f.crs)
    state = co.dissolve()
    fig = plt.figure(figsize=(W, 4.15))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.55, 1], height_ratios=[0.8, 1.2], wspace=0.14, hspace=0.62)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(SEA)
    co.plot(ax=ax, color=LAND, edgecolor=CO_EDGE, linewidth=0.25)
    state.boundary.plot(ax=ax, color=MUTED, linewidth=0.5)
    tiers = {"A": (BLUE, "Tier A", "parcels coded as grazing land"),
             "B": (ORANGE, "Tier B", "other parcel codes, mapped as pasture"),
             "C": (AQUA, "Tier C", "official land-use pasture, open,\nmapped by the CDL as woody or wetland")}
    TIERS = [t for t in ("A", "B", "C") if (f.tier == t).any()]
    for t in TIERS[::-1]:
        g = f[f.tier == t]
        g.plot(ax=ax, color=tiers[t][0], edgecolor=tiers[t][0], linewidth=0.15, rasterized=True)
    map_axes(ax, state.total_bounds)
    ax.set_anchor("N")
    ona = gpd.GeoSeries(gpd.points_from_xy([ONA[0]], [ONA[1]]), crs=4326).to_crs(f.crs).iloc[0]
    half = 10000
    box = (ona.x - half, ona.y - half, ona.x + half, ona.y + half)
    ax.add_patch(Rectangle(box[:2], 2 * half, 2 * half, fill=False, edgecolor=INK, linewidth=0.8, zorder=5))
    ax.text(box[0] - 4000, box[3], "c", ha="right", va="top", fontsize=7, fontweight="bold", color=INK,
            path_effects=STROKE)
    x0, y0, x1, y1 = ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1]
    scalebar(ax, x0 + 0.06 * (x1 - x0), y0 + 0.05 * (y1 - y0), 100, h=7000)
    north(ax, x1 - 0.07 * (x1 - x0), y1 - 0.16 * (y1 - y0), 0.06 * (y1 - y0))
    handles = [Patch(color=tiers[t][0], label=f"{tiers[t][1]}: {tiers[t][2]}\n{num((f.tier == t).sum())} fields, "
                     f"{num(f.area_ha[f.tier == t].sum() / 1000)} kha") for t in TIERS]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.12), fontsize=6, handlelength=1.1,
                    handleheight=1.1, labelspacing=0.9, borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.text(0.02, 0.36, f"{num(len(f))} persistent pasture fields\n{f.area_ha.sum() / 1e6:.2f} Mha",
            transform=ax.transAxes, fontsize=7, color=INK, va="bottom", fontweight="bold")
    head(ax, "a", "Pasture fields of the Florida Pasture Cube", "fields delineated from CDL 2016–2025 and tax parcels")

    # b: size distribution, share of fields and of field area
    axb = fig.add_subplot(gs[0, 1])
    edges = [2, 5, 10, 20, 50, 100, 200, 500, np.inf]
    labels = ["2", "5", "10", "20", "50", "100", "200", "500"]
    cls = pd.cut(f.area_ha, edges, right=False, labels=False)
    n_sh = 100 * pd.Series(1, index=f.index).groupby(cls).sum() / len(f)
    a_sh = 100 * f.area_ha.groupby(cls).sum() / f.area_ha.sum()
    x = np.arange(len(labels))
    axb.bar(x - 0.2, n_sh.reindex(x, fill_value=0), 0.38, color=REF_L, label="share of fields")
    axb.bar(x + 0.2, a_sh.reindex(x, fill_value=0), 0.38, color=REF_D, label="share of field area")
    axb.set_xticks(x - 0.5, labels)
    axb.set_xlim(-0.6, len(x) - 0.4)
    axb.set_xlabel("Field area (ha, log classes)")
    axb.set_ylabel("Share (%)")
    axb.grid(axis="x", visible=False)
    axb.legend(loc="upper right", fontsize=6, handlelength=1, handleheight=1, borderaxespad=0)
    over100 = 100 * f.area_ha[f.area_ha >= 100].sum() / f.area_ha.sum()
    head(axb, "b", "Most fields are small, large ones hold the area",
         f"median {f.area_ha.median():.1f} ha; fields of ≥ 100 ha hold {over100:.0f} % of the area")

    # c: zoom
    axc = fig.add_subplot(gs[1, 1])
    axc.set_facecolor(LAND)
    sub = f.cx[box[0]:box[2], box[1]:box[3]]
    for t in TIERS[::-1]:
        sub[sub.tier == t].plot(ax=axc, color=tiers[t][0], edgecolor="white", linewidth=0.35, rasterized=True)
    axc.set_xlim(box[0], box[2]); axc.set_ylim(box[1], box[3]); axc.set_aspect("equal")
    axc.set_xticks([]); axc.set_yticks([]); axc.grid(False)
    for s in axc.spines.values():
        s.set_visible(True); s.set_color(INK); s.set_linewidth(0.8)
    axc.plot(ona.x, ona.y, marker="*", ms=9, color=INK, mec="white", mew=0.8, zorder=6)
    axc.annotate("RCREC, Ona", (ona.x, ona.y), xytext=(-6, 7), textcoords="offset points",
                 fontsize=6, color=INK, fontweight="bold", path_effects=STROKE, zorder=7, ha="right")
    axc.add_patch(Rectangle((box[0] + 600, box[1] + 500), 7200, 2400, color="white", zorder=5, lw=0))
    scalebar(axc, box[0] + 1300, box[1] + 1000, 5, h=350)
    axc.set_xlabel(""); axc.set_ylabel("")
    head(axc, "c", "Fields are split at parcel boundaries", "20 × 20 km around Ona; colours as in a")
    for yb, yc in ((box[3], 1), (box[1], 0)):
        con = ConnectionPatch(xyA=(box[2], yb), coordsA=ax.transData, xyB=(0, yc), coordsB=axc.transAxes,
                              color=MUTED, lw=0.5, ls=(0, (2, 2)))
        fig.add_artist(con)
    save(fig, "fig1_study_area")


# ---------------------------------------------------------------------------------------------------------------- 2
def _connect(times, v, max_gap_days=30):
    """Observed values joined by lines, broken only at gaps longer than max_gap_days."""
    ok = np.isfinite(v)
    t, y = times[ok], v[ok]
    brk = np.nonzero(np.diff(t.values).astype("timedelta64[D]").astype(int) > max_gap_days)[0] + 1
    return np.insert(t.values, brk, np.datetime64("NaT")), np.insert(y, brk, np.nan)


def usdm_class(u):
    """Highest drought class covering at least half of the county (USDM columns are cumulative percentages)."""
    return np.select([u.D4 >= 50, u.D3 >= 50, u.D2 >= 50, u.D1 >= 50, u.D0 >= 50], [5, 4, 3, 2, 1], 0)


def fig3(rel):
    """Two example fields: observed and gap-filled NDVI, Sentinel-1 backscatter, rainfall and county drought."""
    if not rel or not (rel / "cube.zarr").exists():
        print("fig2: no release yet"); return
    cube = _cube(rel)
    f = fields_table(rel)
    co = counties().to_crs(f.crs)
    state = co.dissolve()
    ona = gpd.GeoSeries(gpd.points_from_xy([ONA[0]], [ONA[1]]), crs=4326).to_crs(f.crs).iloc[0]
    nobs = cube.n_optical_obs.sum("time").values
    cand = f.assign(nobs=nobs, d=f.geometry.distance(ona))
    a = cand[(cand.tier == "A") & (cand.area_ha > 20)].sort_values("d").iloc[0]
    b = cand[(cand.tier == "A") & (cand.area_ha > 20) & (cand.lat > 30.3)].sort_values("nobs").iloc[-1]
    usdm = pd.read_parquet(config.RAW / "labels" / "usdm_county_weekly.parquet")
    times = pd.DatetimeIndex(cube.time.values)
    t0, t1 = pd.Timestamp(SERIES_START), times[-1] + pd.Timedelta(days=5)
    lf = _landfalls()
    lf = lf[(lf.time >= t0) & (lf.time <= t1)]
    cmap_u = mpl.colors.ListedColormap(USDM_C)
    fig = plt.figure(figsize=(W, 6.9))
    outer = GridSpec(2, 1, figure=fig, hspace=0.42)
    titles = {0: ("a", "Pasture near Ona, south-central peninsula"), 1: ("b", "Pasture in the western Panhandle")}
    groups, cur = [], []
    for t, nm in zip(lf.time, lf.name):
        if cur and (t - cur[-1][0]).days > 75:
            groups.append(cur); cur = []
        cur.append((t, nm.title()))
    if cur:
        groups.append(cur)
    for j, row in enumerate((a, b)):
        g = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[j], width_ratios=[0.11, 1], wspace=0.13)
        rows = GridSpecFromSubplotSpec(4, 1, subplot_spec=g[1], height_ratios=[2.2, 1.35, 0.9, 0.3], hspace=0.13)
        i = int(np.nonzero(cube.field.values == row.field_id)[0][0])
        c = cube.isel(field=i)[["ndvi", "gap_filled", "n_optical_obs", "vv_db", "vh_db", "ppt_sum"]].load()
        # locator
        axl = fig.add_subplot(g[0])
        state.plot(ax=axl, color=LAND, edgecolor=MUTED, linewidth=0.4)
        p = row.geometry.representative_point()
        axl.plot(p.x, p.y, "o", ms=5, color=ORANGE, mec="white", mew=0.8)
        map_axes(axl, state.total_bounds, 0.08)
        axl.set_anchor("N")
        obs_share = 100 * (c.n_optical_obs.values > 0).mean()
        axs = [fig.add_subplot(rows[k]) for k in range(4)]
        for ax in axs[1:]:
            ax.sharex(axs[0])
        nd, gf, n = c.ndvi.values, c.gap_filled.values.astype(bool), c.n_optical_obs.values
        ax = axs[0]
        ax.plot(times, nd, color=BLUE_L, lw=0.8, zorder=2)
        ax.plot(times[n > 0], nd[n > 0], "o", ms=1.9, color=BLUE, mew=0, zorder=3)
        ax.set_ylabel("NDVI")
        head(ax, titles[j][0], titles[j][1], f"field {row.field_id}, {row.county} County, {row.area_ha:.0f} ha, "
             f"tier {row.tier}; observed optical composite in {obs_share:.0f} % of 5-day periods", pad=12)
        if j == 0:
            ax.plot([], [], "o", ms=2.5, color=BLUE, mew=0, label="observed 5-day composite")
            ax.plot([], [], color=BLUE_L, lw=1.2, label="gap-filled (gaps ≤ 30 days)")
            ax.legend(loc="lower left", ncol=2, fontsize=5.8, handlelength=1.4, borderaxespad=0.2,
                      bbox_to_anchor=(0, 0.0))
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(ylo - 0.12 * (yhi - ylo), yhi)
        ax = axs[1]
        for var, col, lab in (("vv_db", VIOLET, "VV"), ("vh_db", VIOLET_L, "VH")):
            tx, vy = _connect(times, c[var].values)
            ax.plot(tx, vy, color=col, lw=0.8)
            last = np.nonzero(np.isfinite(c[var].values))[0][-1]
            ax.annotate(lab, (times[last], c[var].values[last]), xytext=(4, 0), textcoords="offset points",
                        fontsize=6, color=col, fontweight="bold", va="center", annotation_clip=False)
        ax.set_ylabel("γ⁰ (dB)")
        ax = axs[2]
        ax.bar(times, c.ppt_sum.values, width=5, color="#5d86b8", linewidth=0, align="edge")
        ax.set_ylabel("Rain\n(mm / 5 d)")
        ax.set_ylim(bottom=0)
        ax = axs[3]
        u = usdm[usdm.county_fips == row.county_fips].sort_values("valid_start")
        ax.pcolormesh(np.r_[u.valid_start.values, u.valid_end.values[-1:]], [0, 1], usdm_class(u)[None, :],
                      cmap=cmap_u, vmin=-0.5, vmax=5.5, rasterized=True)
        ax.set_yticks([]); ax.grid(False)
        ax.set_ylabel("Drought", rotation=0, ha="right", va="center", labelpad=4)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        for ax in axs:
            ax.set_xlim(t0, t1)
            ax.grid(axis="x", visible=False)
            for t in lf.time:
                ax.axvline(t, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
        for ax in axs[:-1]:
            ax.tick_params(labelbottom=False)
        axs[-1].xaxis.set_major_locator(mpl.dates.YearLocator())
        axs[-1].xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
        axs[-1].tick_params(axis="x", length=0, pad=2)
        for gi, grp in enumerate(groups):
            last_g = gi == len(groups) - 1
            axs[0].annotate(", ".join(nm for _, nm in grp), (grp[-1][0] if last_g else grp[0][0], 1),
                            xycoords=("data", "axes fraction"), xytext=(-1.5 if last_g else 1.5, 1.5),
                            textcoords="offset points", fontsize=5.5, color=INK2, ha="right" if last_g else "left",
                            va="bottom", annotation_clip=False)
    handles = [Patch(color=c_, label=l_) for c_, l_ in zip(USDM_C, ["none", "D0", "D1", "D2", "D3", "D4"])]
    fig.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.56, 0.035), fontsize=6,
               handlelength=1.4, columnspacing=1.2,
               title="County drought class (US Drought Monitor class covering ≥ 50 % of the county); "
                     "dashed lines: hurricane landfalls in Florida; γ⁰: Sentinel-1 backscatter", title_fontsize=6)
    save(fig, "fig3_field_series")


# ---------------------------------------------------------------------------------------------------------------- 3
def fig4(rel):
    """Observation density: usable optical days and Sentinel-1 days per field and year, and the monthly share of
    observed, gap-filled and empty field-periods."""
    if not rel or not (rel / "optical_observations").exists():
        print("fig3: no release yet"); return
    opt = pd.concat(pd.read_parquet(p, columns=["fid", "date", "sensor", "qa_use"])
                    for p in (rel / "optical_observations").glob("*.parquet"))
    rad = pd.concat(pd.read_parquet(p, columns=["fid", "date"]) for p in (rel / "radar_observations").glob("*.parquet"))
    opt = opt[opt.qa_use]
    opt["day"] = pd.to_datetime(opt.date).dt.normalize()
    opt = opt.drop_duplicates(["fid", "day", "sensor"])
    opt["year"] = opt.day.dt.year
    rad["day"] = pd.to_datetime(rad.date).dt.normalize()
    rad = rad.drop_duplicates(["fid", "day"])
    rad["year"] = rad.day.dt.year
    nf = len(pd.read_parquet(rel / "fields.parquet", columns=["fid"]))
    last = pd.Timestamp(config.END)
    partial = last.year if last < pd.Timestamp(f"{last.year}-12-31") else None
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.55), gridspec_kw={"width_ratios": [1.15, 1.15, 1], "wspace": 0.42})

    ax = axs[0]
    o = opt.groupby(["year", "sensor"]).size().unstack(fill_value=0) / nf
    hatch = ["" if y != partial else "//////" for y in o.index]
    cols = {"L30": (BLUE, "Landsat 8/9"), "S30": (AQUA, "Sentinel-2")}
    bottom = np.zeros(len(o))
    for s_ in ("L30", "S30"):
        ax.bar(o.index, o[s_], bottom=bottom, color=cols[s_][0], width=0.78, edgecolor="white", linewidth=0.6,
               hatch=hatch, label=cols[s_][1])
        bottom += o[s_].to_numpy()
    tot = o.sum(1)
    for yr, lab in ((2017, "S2B"), (2022, "Landsat 9"), (2025, "S2C")):
        if yr in tot.index:
            ax.annotate(lab, (yr, tot[yr]), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=5.5, color=INK2, rotation=0)
    ax.legend(loc="upper left", fontsize=6, handlelength=1, handleheight=1, borderaxespad=0)
    ax.set_ylabel("Days per field and year")
    ax.set_ylim(0, tot.max() * 1.18)
    head(ax, "a", "Optical revisits tripled since 2016", "usable HLS observation days, mean per field")

    ax = axs[1]
    r = rad.groupby("year").size() / nf
    ax.bar(r.index, r.values, color=VIOLET, width=0.78, edgecolor="white", linewidth=0.6,
           hatch=["" if y != partial else "//////" for y in r.index])
    ax.annotate("S1B lost\nDec 2021", (2021.6, r.get(2022, r.max())), xytext=(2022.6, r.max() * 1.08),
                fontsize=5.5, color=INK2, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.set_ylabel("Days per field and year")
    ax.set_ylim(0, r.max() * 1.3)
    head(ax, "b", "Radar gives about 35 days a year", "Sentinel-1 acquisition days, mean per field")
    for a_ in axs[:2]:
        a_.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
        a_.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))
        a_.grid(axis="x", visible=False)
        if partial:
            a_.annotate(f"to {last:%b}", (partial, a_.get_ylim()[1] * 0.02), xytext=(0, 0), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5, color="white", rotation=90, fontweight="bold")

    ax = axs[2]
    import xarray as xr
    cube = xr.open_zarr(rel / "cube.zarr")
    obs = (cube.n_optical_obs > 0).mean("field").to_series()
    gap = cube.gap_filled.astype(float).mean("field").to_series()
    m_obs = 100 * obs.groupby(obs.index.month).mean()
    m_gap = 100 * gap.groupby(gap.index.month).mean()
    m_none = 100 - m_obs - m_gap
    x = np.arange(1, 13)
    ax.annotate("", (5.6, -11), xytext=(9.4, -11), xycoords="data", annotation_clip=False,
                arrowprops=dict(arrowstyle="|-|", color=INK2, lw=0.6, mutation_scale=1.5))
    ax.text(7.5, -13, "wet season", ha="center", va="top", fontsize=5.8, color=INK2, clip_on=False)
    ax.bar(x, m_obs, 0.78, color=BLUE, label="observed")
    ax.bar(x, m_gap, 0.78, bottom=m_obs, color=BLUE_L, label="gap-filled")
    ax.bar(x, m_none, 0.78, bottom=m_obs + m_gap, color="#dddbd3", label="no value")
    for v, lab, col in ((m_obs.iloc[-1] / 2, "observed", "white"),
                        (m_obs.iloc[-1] + m_gap.iloc[-1] / 2, "gap-filled", INK),
                        (m_obs.iloc[-1] + m_gap.iloc[-1] + m_none.iloc[-1] / 2, "none", INK2)):
        ax.annotate(lab, (12.45, v), xytext=(3, 0), textcoords="offset points", fontsize=5.8, color=INK2, va="center",
                    annotation_clip=False)
    ax.set_xticks(x, list("JFMAMJJASOND"))
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Field-periods (%)")
    ax.grid(axis="x", visible=False)
    head(ax, "c", "The wet season has the most gaps", "5-day optical composites, all years")
    save(fig, "fig4_coverage")


# ---------------------------------------------------------------------------------------------------------------- 4
def _traj(a, k, rows, pre=12, post=12):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        seg = a[rows][:, k - pre:k + post + 1]
        return (np.arange(-pre, post + 1) * config.COMPOSITE_DAYS, np.nanmedian(seg, 0),
                np.nanpercentile(seg, 25, 0), np.nanpercentile(seg, 75, 0))


def fig5(rel):
    """Hurricane Ian (2022) and the January 2018 freeze, observed optical values only."""
    if not rel or not (rel / "cube.zarr").exists():
        print("fig5: no release yet"); return
    from fpcube.benchmark.tasks import observed_only
    cube = observed_only(_cube(rel)[["ndvi", "mndwi", "vv_db", "ppt_sum", "tmin_mean", "n_optical_obs"]].load())
    f = fields_table(rel)
    co = counties().to_crs(f.crs)
    state = co.dissolve()
    times = pd.DatetimeIndex(cube.time.values)
    pts = f.geometry.representative_point()
    summ = json.loads((rel / "validation" / "summary.json").read_text())["stats"]
    ip = summ.get("ian_passes", {})
    fig = plt.figure(figsize=(W, 5.6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[0.95, 1.1, 1], wspace=0.5, hspace=0.62)

    # a: Ian rain and track
    k = int(np.searchsorted(times, pd.Timestamp("2022-09-28")))
    rain = np.nansum(cube.ppt_sum.values[:, k - 1:k + 1], axis=1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(SEA)
    state.plot(ax=ax, color=LAND, edgecolor=MUTED, linewidth=0.4)
    o = np.argsort(rain)
    norm = mpl.colors.Normalize(0, 400)
    ax.scatter(pts.x.values[o], pts.y.values[o], c=rain[o], s=0.6, cmap=cmc.lapaz_r, norm=norm, rasterized=True,
               linewidths=0)
    tp = config.RAW / "labels" / "hurricane_tracks_florida.parquet"
    if tp.exists():
        tr = pd.read_parquet(tp)
        tr = tr[tr.storm_id == "AL092022"].sort_values("time")
        g = gpd.GeoSeries(gpd.points_from_xy(tr.lon, tr.lat), crs=4326).to_crs(f.crs)
        ax.plot(g.x, g.y, color=INK, lw=1.0, zorder=5)
        lfp = config.RAW / "labels" / "hurricane_landfalls.parquet"
        lf = pd.read_parquet(lfp) if lfp.exists() else tr.iloc[:0]
        lf = lf[(lf.storm_id == "AL092022") & (lf.status == "HU")].sort_values("time")
        gl = gpd.GeoSeries(gpd.points_from_xy(lf.lon, lf.lat), crs=4326).to_crs(f.crs)
        inside = gl.within(state.buffer(20000).iloc[0]).to_numpy()
        lf, gl = lf[inside], gl[inside]
        if len(gl):
            gl = gl.iloc[[int(np.argmax(lf.vmax_kt.to_numpy()))]]          # main Florida landfall (Cayo Costa)
            ax.plot(gl.x.iloc[0], gl.y.iloc[0], "o", ms=4, color="white", mec=INK, mew=0.9, zorder=6)
            ax.annotate("landfall", (gl.x.iloc[0], gl.y.iloc[0]), xytext=(-5, 0), textcoords="offset points",
                        fontsize=5.5, color=INK, path_effects=STROKE, ha="right", va="center")
    map_axes(ax, state.total_bounds, 0.03)
    ax.set_anchor("N")
    cax = ax.inset_axes([0.05, -0.06, 0.9, 0.035])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmc.lapaz_r), cax=cax, orientation="horizontal", extend="max")
    cb.set_ticks([0, 100, 200, 300, 400])
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.set_label("Rain at each field, 26 Sep–5 Oct 2022 (mm)", fontsize=5.8, labelpad=1.5)
    head(ax, "a", "Hurricane Ian, 28 Sep 2022", "track and 10-day rain")

    # b: single passes
    ax = fig.add_subplot(gs[0, 1])
    groups = [("hit_le1", "≥ 200 mm, HAND ≤ 1 m", HAND_C[0]), ("hit_1to3", "≥ 200 mm, HAND 1–3 m", HAND_C[1]),
              ("hit_ge3", "≥ 200 mm, HAND ≥ 3 m", HAND_C[2]), ("ctrl", "< 50 mm (control)", REF_L)]
    for yi, (g_, lab, col) in enumerate(groups):
        if f"ian_pass_post_{g_}" not in ip:
            continue
        pre, post = 100 * ip[f"ian_pass_pre_{g_}"], 100 * ip[f"ian_pass_post_{g_}"]
        lo, hi = 100 * ip[f"ian_pass_post_{g_}_lo"], 100 * ip[f"ian_pass_post_{g_}_hi"]
        y = -yi
        ax.plot([pre, post], [y, y], color=col, lw=1.2, zorder=2)
        ax.plot([lo, hi], [y - 0.22, y - 0.22], color=col, lw=0.8, zorder=2)
        ax.plot(pre, y, "o", ms=4.2, mfc="white", mec=col, mew=1, zorder=3)
        ax.plot(post, y, "o", ms=4.6, color=col, mec=col, zorder=3)
        d, dlo, dhi = (100 * ip[f"ian_pass_diff_{g_}{s}"] for s in ("", "_lo", "_hi"))
        ax.annotate(f"{d:+.1f} pp\n[{dlo:+.1f}, {dhi:+.1f}]", (10.4, y), xycoords="data",
                    xytext=(0, 0), textcoords="offset points", fontsize=5.5, va="center", ha="right",
                    color=INK if dlo > 0 else MUTED, fontweight="bold" if dlo > 0 else "normal",
                    annotation_clip=False)
        ax.annotate(f"{lab}  (n = {num(ip[f'ian_pass_n_{g_}'])})", (0, y + 0.3), xycoords=("axes fraction", "data"),
                    xytext=(2, 0), textcoords="offset points", fontsize=5.6, color=INK2, va="bottom")
    ax.set_yticks([]); ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-3.55, 0.75)
    ax.set_xlim(0, 10.5)
    ax.set_xticks([0, 2, 4, 6])
    ax.text(10.4, 0.62, "difference", fontsize=5.5, color=INK2, ha="right", va="bottom")
    ax.set_xlabel("Fields with a VV drop > 3 dB (%)")
    ax.plot([], [], "o", ms=4, mfc="white", mec=INK2, label="between passes before the storm")
    ax.plot([], [], "o", ms=4, color=INK2, label="across landfall (95 % interval below)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.24), fontsize=5.6, handletextpad=0.2, borderaxespad=0)
    head(ax, "b", "Water shows only in low, soaked fields", "single Sentinel-1 passes, relative orbit 48")

    # c: composites hide it
    ax = fig.add_subplot(gs[0, 2])
    hand = f.hand_mean_m.to_numpy()
    hit = rain >= 200
    rows3 = [(hit & (hand <= 1), HAND_C[0], "HAND ≤ 1 m"), (hit & (hand > 1) & (hand < 3), HAND_C[1], "1–3 m"),
             (hit & (hand >= 3), HAND_C[2], "≥ 3 m")]
    rows3[0] = (rows3[0][0], rows3[0][1], "≤ 1 m")
    ends = []
    for rows_, col, lab in rows3:
        x, m, lo, hi = _traj(cube.mndwi.values, k, rows_)
        if rows_.sum() and lab == "≤ 1 m":
            ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
        ok = np.isfinite(m)
        ax.plot(x[ok], m[ok], color=col, lw=1.1, marker="o", ms=1.6, label=lab)
    ax.legend(loc="lower left", fontsize=5.8, handlelength=1.4, borderaxespad=0.2, title="height above drainage",
              title_fontsize=5.6)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.text(1.5, ax.get_ylim()[1], "landfall", fontsize=5.5, color=INK2, va="top", ha="left")
    ax.set_xlabel("Days from landfall")
    ax.set_ylabel("MNDWI (observed composites)")
    head(ax, "c", "5-day composites hide it", "fields with ≥ 200 mm; median; band: IQR, HAND ≤ 1 m")

    # d: freeze map
    k = int(np.searchsorted(times, pd.Timestamp("2018-01-17"), side="right")) - 1
    tmin = cube.tmin_mean.values[:, k]
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(SEA)
    state.plot(ax=ax, color=LAND, edgecolor=MUTED, linewidth=0.4)
    tn = mpl.colors.TwoSlopeNorm(vmin=-4, vcenter=2, vmax=12)
    o = np.argsort(-np.nan_to_num(tmin, nan=99))
    ax.scatter(pts.x.values[o], pts.y.values[o], c=tmin[o], s=0.6, cmap=cmc.vik, norm=tn, rasterized=True,
               linewidths=0)
    map_axes(ax, state.total_bounds, 0.03)
    ax.set_anchor("N")
    cax = ax.inset_axes([0.05, -0.06, 0.9, 0.035])
    cb = fig.colorbar(mpl.cm.ScalarMappable(tn, cmc.vik), cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks([-4, -2, 0, 2, 4, 8, 12])
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.set_label(f"Mean daily minimum, {times[k]:%d}–{times[k] + pd.Timedelta(days=4):%d %b %Y} (°C)", fontsize=5.8,
                 labelpad=1.5)
    cold = tmin <= 2
    head(ax, "d", "January 2018 freeze", f"{num(cold.sum())} fields at ≤ 2 °C")

    # e: cold fields in 2018 against their own course in other years
    nd = cube.ndvi.values
    poy = (times.day_of_year - 1) // config.COMPOSITE_DAYS
    pre, post = 12, 12
    segs, dd_usual = [], []
    for yr in range(2016, 2026):
        if yr == 2018:
            continue
        kk = np.nonzero((times.year == yr) & (poy == poy[k]))[0]
        if not len(kk) or kk[0] < 6 or kk[0] + 4 > len(times):
            continue
        kk = kk[0]
        bad = cube.tmin_mean.values[:, kk] <= 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dd = np.nanmedian(nd[:, kk + 1:kk + 4], 1) - np.nanmedian(nd[:, kk - 6:kk], 1)
        dd[bad] = np.nan
        dd_usual.append(dd)
        if kk - pre >= 0 and kk + post < len(times):
            s = nd[:, kk - pre:kk + post + 1].copy()
            s[bad] = np.nan
            segs.append(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        usual_traj = np.nanmedian(np.nanmedian(np.stack(segs), 0)[cold], 0)
        d18 = np.nanmedian(nd[:, k + 1:k + 4], 1) - np.nanmedian(nd[:, k - 6:k], 1)
        excess = d18 - np.nanmedian(np.stack(dd_usual), 0)
    x, m, lo, hi = _traj(nd, k, cold, pre, post)
    ax = fig.add_subplot(gs[1, 1])
    ax.axvspan(0, config.COMPOSITE_DAYS, color="#dce8f6", lw=0, zorder=0)
    ax.text(2.5, 0.98, "freeze", transform=ax.get_xaxis_transform(), fontsize=5.5, color=BLUE_D, ha="center",
            va="top")
    ax.fill_between(x, lo, hi, color=ORANGE, alpha=0.15, lw=0)
    ok = np.isfinite(m)
    ax.plot(x[ok], m[ok], color=ORANGE, lw=1.2, marker="o", ms=1.8)
    ok2 = np.isfinite(usual_traj)
    ax.plot(x[ok2], usual_traj[ok2], color=REF_D, lw=1.1, ls=(0, (3, 1.5)))
    i0 = int(np.nanargmin(m))
    ax.annotate("winter 2017–18", (x[i0], m[i0]), xytext=(-4, 0), textcoords="offset points",
                fontsize=5.8, color=ORANGE, ha="right", va="center", fontweight="bold", path_effects=STROKE)
    ax.annotate("same fields,\nother winters", (x[ok2][-1], usual_traj[ok2][-1]), xytext=(-2, 5),
                textcoords="offset points", fontsize=5.8, color=REF_D, ha="right", va="bottom", fontweight="bold")
    ticks = [-60, -30, 0, 30, 60]
    ax.set_xticks(ticks, [f"{(times[k] + pd.Timedelta(days=t)):%d %b}" for t in ticks])
    ax.set_ylabel("NDVI (observed composites)")
    head(ax, "e", "Cold fields browned more than usual", "median of the ≤ 2 °C fields; band: IQR in 2018")

    # f: excess per field
    ax = fig.add_subplot(gs[1, 2])
    v = excess[cold & np.isfinite(excess)]
    bins = np.linspace(-0.3, 0.2, 51)
    hcount, _ = np.histogram(np.clip(v, bins[0], bins[-1]), bins)
    centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(centers, 100 * hcount / len(v), width=np.diff(bins), color=np.where(centers < 0, ORANGE, REF_L),
           linewidth=0)
    med = np.median(v)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.axvline(med, color=ORANGE, lw=0.8, ls=(0, (2, 1.5)))
    ax.annotate(f"median {med:+.2f}", (med, ax.get_ylim()[1]), xytext=(-3, -2), textcoords="offset points",
                fontsize=5.8, color=ORANGE, ha="right", va="top", fontweight="bold")
    ax.text(0.98, 0.95, f"{100 * (v < 0).mean():.0f} % of fields\nbelow their\nusual change", transform=ax.transAxes,
            fontsize=5.8, color=INK2, ha="right", va="top")
    ax.set_xlabel("ΔNDVI in 2018 minus usual ΔNDVI")
    ax.set_ylabel("Fields (%)")
    ax.grid(axis="x", visible=False)
    head(ax, "f", "The freeze added 0.05 to the winter decline", f"per field, n = {num(len(v))}")
    save(fig, "fig5_events")


# ---------------------------------------------------------------------------------------------------------------- 5
def fig6(rel):
    b = rel / "benchmark" if rel else None
    if not b or not (b / "task1_results.csv").exists():
        print("fig6: no benchmark yet"); return
    r1 = pd.read_csv(b / "task1_results.csv")
    fig = plt.figure(figsize=(W, 5.4))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.05], hspace=0.62)
    top = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.42)
    bot = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1], width_ratios=[1, 1, 0.62, 1, 1], wspace=0.1)
    style = {"persistence": (REF_D, (0, (3, 1.5)), "Persistence"), "climatology": (REF_L, "-", "Climatology"),
             "lightgbm": (BLUE, "-", "LightGBM"), "transformer": (ORANGE, "-", "Transformer")}
    leads = [15, 30, 45, 60]

    ax = fig.add_subplot(top[0])
    ends = []
    for m, (col, ls, lab) in style.items():
        d = r1[(r1.split == "test") & (r1.model == m)].sort_values("horizon_days")
        ax.fill_between(d.horizon_days, d.rmse_lo, d.rmse_hi, color=col, alpha=0.12, lw=0)
        ax.plot(d.horizon_days, d.rmse, color=col, ls=ls, marker="o", ms=2.8, lw=1.2)
        ends.append((d.rmse.iloc[-1], col, lab))
    for (y, col, lab), yl in zip(ends, spread([e[0] for e in ends], 0.0075)):
        ax.annotate(lab, (60, yl), xytext=(5, 0), textcoords="offset points", fontsize=6, color=col, va="center",
                    fontweight="bold", annotation_clip=False)
    ax.set_xticks(leads); ax.set_xlim(12, 63)
    ax.set_xlabel("Lead time (days)"); ax.set_ylabel("NDVI RMSE")
    ax.annotate("lower is better", (0.02, 0.03), xycoords="axes fraction", fontsize=5.6, color=MUTED)
    head(ax, "a", "Learned models have the lowest error", "Task 1, 2025–2026 test years; band: 95 % interval")

    ax = fig.add_subplot(top[1])
    ends = []
    for k_, (m, sp) in enumerate((("lightgbm", "test"), ("transformer", "test"), ("lightgbm", "test_spatial"),
                                   ("transformer", "test_spatial"))):
        d = r1[(r1.split == sp) & (r1.model == m)].sort_values("horizon_days")
        col = style[m][0]
        x = d.horizon_days + (-1.2 if sp == "test" else 1.2) + (-0.5 if m == "lightgbm" else 0.5)
        temporal = sp == "test"
        ax.errorbar(x, 100 * d.skill_vs_ref, yerr=[100 * (d.skill_vs_ref - d.skill_lo), 100 * (d.skill_hi - d.skill_vs_ref)],
                    color=col, ls="-" if temporal else (0, (3, 1.5)), marker="o", ms=3, mfc=col if temporal else "white",
                    mec=col, mew=0.9, lw=1.1, elinewidth=0.6, capsize=0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.annotate("= climatology", (63, 0), xytext=(0, 2), textcoords="offset points", fontsize=5.6, color=INK2,
                ha="right", va="bottom")
    ax.plot([], [], color=BLUE, marker="o", ms=3, label="LightGBM")
    ax.plot([], [], color=ORANGE, marker="o", ms=3, label="Transformer")
    ax.plot([], [], color=INK2, marker="o", ms=3, label="temporal test")
    ax.plot([], [], color=INK2, marker="o", ms=3, mfc="white", ls=(0, (3, 1.5)), label="withheld counties")
    ax.legend(loc="upper right", ncol=2, fontsize=5.8, handlelength=2.2, columnspacing=1, borderaxespad=0)
    ax.set_xticks(leads); ax.set_xlim(10, 65)
    ax.set_ylim(-12, 95)
    ax.set_xlabel("Lead time (days)"); ax.set_ylabel("MSE skill vs climatology (%)")
    head(ax, "b", "Half of the climatology error removed at 30 days",
         "Task 1; 95 % interval; dashed: counties withheld from training")

    # c: Task 2
    if (b / "task2_results.csv").exists():
        r2 = pd.read_csv(b / "task2_results.csv")
        r2 = r2[~r2.event_id.str.startswith("pooled_")]
        kinds = ["drought", "extreme_rain", "freeze", "hurricane"]
        kind_lab = {"drought": "Drought onset", "extreme_rain": "Extreme rain", "freeze": "Freeze",
                    "hurricane": "Hurricane"}
        n_ev = r2[(r2.experiment == "leave_one_event_out") & (r2.model == "lightgbm")].groupby("kind").size()
        panels = [("leave_one_event_out", "Same event type in training",
                   [("rule_dNDVI_dMNDWI", REF_L, "ΔNDVI–ΔMNDWI rule"), ("lightgbm", BLUE, "LightGBM")]),
                  ("leave_one_kind_out", "Event type withheld",
                   [("rule_dNDVI_dMNDWI", REF_L, "ΔNDVI–ΔMNDWI rule"), ("lightgbm", BLUE, "LightGBM"),
                    ("transformer", ORANGE, "Transformer")])]
        for pi, (exp, sub, models) in enumerate(panels):
            ax = fig.add_subplot(bot[pi])
            ax.axvspan(0, 0.5, color="#f4f3ef", lw=0, zorder=0)
            ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
            for mi, (m, col, lab) in enumerate(models):
                off = (mi - (len(models) - 1) / 2) * 0.22
                for yi, kd in enumerate(kinds):
                    v = r2[(r2.experiment == exp) & (r2.model == m) & (r2.kind == kd)].auroc.dropna()
                    if not len(v):
                        continue
                    q1, q2, q3 = np.percentile(v, [25, 50, 75])
                    y = -yi + off
                    ax.plot([q1, q3], [y, y], color=col, lw=1.4, solid_capstyle="butt", alpha=0.55)
                    ax.plot(q2, y, "o", ms=3.4, color=col, mec="white", mew=0.4, zorder=3)
            ax.set_xlim(0.15, 1.0)
            ax.set_ylim(-3.55, 0.55)
            ax.set_yticks(-np.arange(len(kinds)))
            ax.set_yticklabels([f"{kind_lab[k]}\n({n_ev.get(k, 0)} events)" for k in kinds] if pi == 0 else [])
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("AUROC per event")
            ax.set_title(sub, fontsize=6.2, color=INK2, loc="left", pad=3)
            ax.text(0.48, -3.5, "chance", fontsize=5.5, color=MUTED, ha="right", va="bottom")
            if pi == 0:
                head(ax, "c", "Events are recognised only when their type was seen", None, pad=12)
            if pi == 1:
                hs = [mpl.lines.Line2D([], [], color=col, marker="o", ms=3.4, lw=1.4, alpha=0.8, label=lab)
                      for _, col, lab in models]
                ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(-0.05, -0.3), ncol=3, fontsize=5.8,
                          handlelength=1.6, columnspacing=1.0, borderaxespad=0,
                          title="dot: median over events; line: interquartile range", title_fontsize=5.6)

    # d: Task 3
    p3 = b / "task3_results.csv"
    if p3.exists() and "auroc_worse" in pd.read_csv(p3, nrows=1):
        r3 = pd.read_csv(p3)
        r3 = r3[r3.split == "test"].set_index("model")
        models = [("transition_climatology", REF_L, "Transition\nclimatology"),
                  ("lightgbm_usdm_only", BLUE_L, "LightGBM,\nUSDM only"),
                  ("lightgbm_with_fields", BLUE, "LightGBM,\nwith fields"),
                  ("attention_mil", AQUA, "Attention MIL,\nwith fields")]
        for pi, (dirn, sub) in enumerate((("worse", "Worse class in 4 weeks"), ("better", "Better class in 4 weeks"))):
            ax = fig.add_subplot(bot[3 + pi])
            ax.axvspan(0, 0.5, color="#f4f3ef", lw=0, zorder=0)
            ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
            for yi, (m, col, lab) in enumerate(models):
                v, lo, hi = r3.loc[m, [f"auroc_{dirn}", f"auroc_{dirn}_lo", f"auroc_{dirn}_hi"]]
                ax.plot([lo, hi], [-yi, -yi], color=col, lw=1.4, alpha=0.7, solid_capstyle="butt")
                ax.plot(v, -yi, "o", ms=4, color=col, mec="white", mew=0.4, zorder=3)
                ax.annotate(f"{v:.2f}", (v, -yi), xytext=(0, 4), textcoords="offset points", fontsize=5.5,
                            color=INK2, ha="center", va="bottom")
            gp = b / "task3_field_evidence_gain.csv"
            if gp.exists():
                gg = pd.read_csv(gp).set_index(["split", "direction"]).loc[("test", dirn)]
                sig = gg.lo > 0
                ax.annotate(f"fields add {gg.iloc[0]:+.2f}\n[{gg.lo:+.2f}, {gg.hi:+.2f}]", (0.31, -1.5),
                            fontsize=5.6, va="center", ha="left", color=BLUE_D if sig else MUTED,
                            fontweight="bold" if sig else "normal")
            ax.set_xlim(0.3, 1.0)
            ax.set_ylim(-3.55, 0.55)
            ax.set_yticks(-np.arange(len(models)))
            ax.set_yticklabels([lab for _, _, lab in models] if pi == 0 else [])
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("AUROC")
            ax.set_title(sub, fontsize=6.2, color=INK2, loc="left", pad=3)
            if pi == 0:
                head(ax, "d", "Field evidence anticipates drought worsening", None, pad=12)
                ax.text(0, -0.3, "Task 3, county drought class; line: 95 % interval", transform=ax.transAxes,
                        fontsize=5.6, color=MUTED, va="top")
    save(fig, "fig6_benchmark")


# ------------------------------------------------------------------------------------------------------------- cube
CUBE_VARS = [  # (variable, label, group)
    ("ndvi", "NDVI", "optical"), ("evi", "EVI", "optical"), ("ndmi", "NDMI", "optical"),
    ("mndwi", "MNDWI", "optical"), ("nbr", "NBR", "optical"), ("blue", "blue", "optical"),
    ("green", "green", "optical"), ("red", "red", "optical"), ("nir", "NIR", "optical"),
    ("swir1", "SWIR1", "optical"), ("swir2", "SWIR2", "optical"), ("ndvi_sd", "NDVI s.d.", "optical"),
    ("water_frac", "water fraction", "optical"), ("vv_db", "VV", "radar"), ("vh_db", "VH", "radar"),
    ("vh_vv_db", "VH/VV", "radar"), ("ppt_sum", "rain", "weather"), ("tmin_mean", "Tmin", "weather"),
    ("tmean_mean", "Tmean", "weather"), ("tmax_mean", "Tmax", "weather"), ("vpdmax_mean", "VPDmax", "weather"),
    ("n_optical_obs", "optical obs.", "qa"), ("n_radar_obs", "radar obs.", "qa"), ("gap_filled", "gap-filled", "qa")]
GROUP_CMAP = {"optical": cmc.bamako_r, "radar": cmc.acton_r, "weather": cmc.devon_r, "qa": cmc.grayC}
GROUP_COL = {"optical": "#3f7d3a", "radar": VIOLET, "weather": "#2f5f98", "qa": REF_D}


def _rgb(a, group, lo=None, hi=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanpercentile(a, 2) if lo is None else lo
        hi = np.nanpercentile(a, 98) if hi is None else hi
    z = np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)
    rgb = GROUP_CMAP[group](0.12 + 0.8 * np.nan_to_num(z))
    rgb[~np.isfinite(a)] = (1, 1, 1, 1)
    return rgb


def fig2(rel):
    """What the cube is: field-level observations from each source, composited on a common 5-day calendar into a
    field x time x variable array, with a field's time series and a statewide map as two of its slices.
    Every face is drawn from the released data."""
    if not rel or not (rel / "cube.zarr").exists():
        print("fig2: no release yet"); return
    cube = _cube(rel)
    f = fields_table(rel)
    times = pd.DatetimeIndex(cube.time.values)
    rng = np.random.default_rng(7)
    focus_id = "FPC067081"
    fi = int(np.nonzero(cube.field.values == focus_id)[0][0])
    pick = np.sort(np.r_[rng.choice(len(f), 299, replace=False), fi])
    pick = np.unique(pick)
    order = np.argsort(f.lat.to_numpy()[pick])            # south at the bottom, north at the top
    pick = pick[order]
    row_focus = int(np.nonzero(pick == fi)[0][0])
    sub = cube.isel(field=pick)[[v for v, _, _ in CUBE_VARS]].load()
    nd = sub.ndvi.values                                   # fields x time
    obs_share = (cube.n_optical_obs > 0).mean("field").to_series()
    t_focus = int(np.argmax(obs_share[obs_share.index.year == 2025].values)) + int(np.nonzero(times.year == 2025)[0][0])

    fig = plt.figure(figsize=(W, 3.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 53); ax.set_aspect("equal"); ax.axis("off")
    x0, y0, Wf, Hf, dx, dy = 33.5, 8.0, 31.0, 29.0, 10.0, 8.0
    ax.text(0.6, 51.0, "One value per field, 5-day period and variable", fontsize=7.6, fontweight="bold",
            color=INK, ha="left", va="center")

    # front face: NDVI, fields x time (gaps left white)
    ax.imshow(_rgb(nd, "optical", 0.2, 0.9), extent=(x0, x0 + Wf, y0, y0 + Hf), origin="lower", aspect="auto",
              interpolation="nearest", zorder=2)
    # top face: every variable through time (median over the sampled fields); side face: every variable per field
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        top = np.stack([_rgb(np.nanmedian(sub[v].values.astype(float), 0)[None, :], g)[0] for v, _, g in CUBE_VARS])
        side = np.stack([_rgb(np.nanmean(sub[v].values.astype(float), 1)[:, None], g)[:, 0] for v, _, g in CUBE_VARS],
                        axis=1)
    im = ax.imshow(top, extent=(0, 1, 0, 1), origin="lower", aspect="auto", interpolation="nearest", zorder=2)
    im.set_transform(mpl.transforms.Affine2D(np.array([[Wf, dx, x0], [0, dy, y0 + Hf], [0, 0, 1]])) + ax.transData)
    im = ax.imshow(side, extent=(0, 1, 0, 1), origin="lower", aspect="auto", interpolation="nearest", zorder=2)
    im.set_transform(mpl.transforms.Affine2D(np.array([[dx, 0, x0 + Wf], [dy, Hf, y0], [0, 0, 1]])) + ax.transData)
    nv = len(CUBE_VARS)
    for k in range(1, nv):
        u = k / nv
        lw = 0.6 if CUBE_VARS[k][2] != CUBE_VARS[k - 1][2] else 0.15
        col = "white" if lw < 0.5 else INK
        ax.plot([x0 + u * dx, x0 + Wf + u * dx], [y0 + Hf + u * dy] * 2, color=col, lw=lw, zorder=3)
        ax.plot([x0 + Wf + u * dx] * 2, [y0 + u * dy, y0 + Hf + u * dy], color=col, lw=lw, zorder=3)
    P = lambda u, v, w: (x0 + u * Wf + w * dx, y0 + v * Hf + w * dy)      # u time, v field, w variable
    edges = [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (1, 1, 0)), ((1, 1, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, 0)),
             ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
             ((1, 1, 1), (1, 0, 1))]
    for a_, b_ in edges:
        (xa, ya), (xb, yb) = P(*a_), P(*b_)
        ax.plot([xa, xb], [ya, yb], color=INK, lw=0.7, zorder=4, solid_capstyle="round")
    # dimensions
    arr = dict(arrowstyle="-|>", color=INK2, lw=0.6, mutation_scale=6)
    ax.annotate("", np.array(P(1, 0, 0)) + [0, -1.5], xytext=np.array(P(0, 0, 0)) + [0, -1.5], arrowprops=arr)
    ax.text(x0 + Wf / 2, y0 - 2.3, f"time: {len(times)} five-day periods, Jan {times[0]:%Y} – {times[-1]:%b %Y}",
            ha="center", va="top", fontsize=6.3, color=INK)
    ax.annotate("", np.array(P(0, 1, 0)) + [-1.5, 0], xytext=np.array(P(0, 0, 0)) + [-1.5, 0], arrowprops=arr)
    ax.text(x0 - 2.3, y0 + Hf / 2, f"fields: {num(cube.sizes['field'])} (300 shown, south to north)", ha="right",
            va="center", fontsize=6.3, color=INK, rotation=90)
    ax.annotate("", np.array(P(1, 0, 1)) + [1.2, 0], xytext=np.array(P(1, 0, 0)) + [1.2, 0], arrowprops=arr)
    ax.text(*(np.array(P(1, 0, 0.45)) + [2.1, -0.7]), f"variables: {nv}", fontsize=6.3, color=INK, ha="left",
            va="top", rotation=np.degrees(np.arctan2(dy, dx)), rotation_mode="anchor")
    ax.text(x0 + 0.6, y0 + Hf - 0.7, "NDVI", fontsize=6.2, color=INK, fontweight="bold", va="top", ha="left",
            path_effects=STROKE, zorder=6)
    # variable names by group, with leaders to their band on the cube (back group at the top)
    names = {"qa": ["counts and flag: optical obs., radar obs., gap-filled"],
             "weather": ["weather: rain, Tmin, Tmean, Tmax, VPDmax"],
             "radar": ["radar: VV, VH, VH/VV"],
             "optical": ["optical: NDVI, EVI, NDMI, MNDWI, NBR, NDVI s.d.,",
                         "water fraction, blue, green, red, NIR, SWIR1, SWIR2"]}
    yt = 52.3
    for g in ("qa", "weather", "radar", "optical"):
        ks = [k for k, (_, _, gg) in enumerate(CUBE_VARS) if gg == g]
        xa, ya = P(1, 1, (ks[0] + ks[-1] + 1) / 2 / nv)
        txt = "\n".join(names[g])
        ax.text(77.0, yt, txt, fontsize=5.3, color=GROUP_COL[g], ha="left", va="top", linespacing=1.25,
                fontweight="bold" if False else "normal")
        ymid = yt - 0.85 * len(names[g])
        ax.plot([xa, 75.6, 76.5], [ya, ymid, ymid], color=GROUP_COL[g], lw=0.5, zorder=5)
        ax.plot(xa, ya, "o", ms=1.8, color=GROUP_COL[g], zorder=6)
        yt -= 1.75 * len(names[g]) + 0.35
    # highlighted slices: one field (row) and one period (column)
    yr = y0 + (row_focus + 0.5) / len(pick) * Hf
    xc = x0 + (t_focus + 0.5) / len(times) * Wf
    ax.add_patch(Rectangle((x0, yr - 0.5), Wf, 1.0, fill=False, edgecolor=ORANGE, lw=1.1, zorder=5))
    ax.add_patch(Rectangle((xc - 0.35, y0), 0.7, Hf, fill=False, edgecolor=BLUE, lw=1.1, zorder=5))

    # sources, left, and the processing between them and the cube
    src = [("Landsat + Sentinel-2 (HLS)", "30 m, every 2–3 days", "optical"),
           ("Sentinel-1 backscatter", "10 m, every 6–12 days", "radar"),
           ("PRISM weather", "4 km, daily", "weather"),
           ("gNATSGO, 3DEP, C-CAP", "soil, terrain, wetland", "qa"),
           ("USDM, NASS, HURDAT2", "drought, condition, storms", "qa")]
    ax.text(0.6, 45.0, "Sources", fontsize=6.6, color=INK, fontweight="bold", va="center")
    for k, (name, res, g) in enumerate(src):
        yy = 40.5 - k * 4.7
        ax.add_patch(Rectangle((0.6, yy - 1.4), 1.3, 2.4, color=GROUP_COL[g], lw=0))
        ax.text(2.6, yy + 0.4, name, fontsize=5.6, color=INK, va="center", fontweight="bold")
        ax.text(2.6, yy - 1.1, res, fontsize=5.3, color=INK2, va="center")
    ax.plot([20.4, 21.1, 21.1, 20.4], [42.0, 42.0, 29.4, 29.4], color=MUTED, lw=0.5)
    ax.annotate("", (x0 - 4.4, 35.7), xytext=(21.1, 35.7),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, mutation_scale=7))
    ax.text(21.6, 34.6, "field means of\nclear pixels per\nacquisition, then\n5-day composites", fontsize=5.1,
            color=INK2, va="top", ha="left", linespacing=1.3)
    ax.text(0.6, 1.0, "static layers and labels are stored once per field (fields.parquet) or per county and date",
            ha="left", va="bottom", fontsize=5.6, color=INK2)

    # slices, right
    ts = fig.add_axes([0.775, 0.45, 0.215, 0.22])
    c = cube.isel(field=fi)[["ndvi", "n_optical_obs"]].load()
    ts.plot(times, c.ndvi.values, color=BLUE_L, lw=0.5)
    ok = c.n_optical_obs.values > 0
    ts.plot(times[ok], c.ndvi.values[ok], "o", ms=0.9, color=ORANGE, mew=0)
    ts.tick_params(labelsize=5.2, length=1.5, pad=1)
    ts.xaxis.set_major_locator(mpl.dates.YearLocator(2))
    ts.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    ts.grid(axis="x", visible=False)
    ts.set_title("a row: one field's NDVI through time", fontsize=6, color=ORANGE, loc="left", fontweight="bold", pad=2)
    mp = fig.add_axes([0.775, 0.02, 0.215, 0.34])
    co = counties().to_crs(f.crs)
    co.dissolve().plot(ax=mp, color=LAND, edgecolor=MUTED, linewidth=0.3)
    v = cube.ndvi.isel(time=t_focus).values
    o_ = cube.n_optical_obs.isel(time=t_focus).values > 0
    pts = f.geometry.representative_point()
    mp.scatter(pts.x.values[o_], pts.y.values[o_], c=v[o_], s=0.15, cmap=cmc.bamako_r, vmin=0.2, vmax=0.9,
               rasterized=True, linewidths=0)
    map_axes(mp, co.total_bounds, 0.02)
    mp.set_title(f"a column: all fields, {times[t_focus]:%d %b %Y}", fontsize=6, color=BLUE, loc="left",
                 fontweight="bold", pad=2)
    for s_ in ("left", "bottom"):
        ts.spines[s_].set_color(AXIS)
    for a_ in (ts, mp):
        a_.set_facecolor("none")
    con = ConnectionPatch(xyA=(x0 + Wf, yr), coordsA=ax.transData, xyB=(0, 0.5), coordsB=ts.transAxes,
                          color=ORANGE, lw=0.6, ls=(0, (2, 2)))
    fig.add_artist(con)
    con = ConnectionPatch(xyA=(xc, y0), coordsA=ax.transData, xyB=(0, 0.3), coordsB=mp.transAxes,
                          color=BLUE, lw=0.6, ls=(0, (2, 2)), connectionstyle="arc3,rad=0.25")
    fig.add_artist(con)
    save(fig, "fig2_data_cube")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"])
    ap.add_argument("--release", default="v1.0")
    a = ap.parse_args()
    rel = config.RELEASE / a.release
    rel = rel if rel.exists() else None
    for f in a.figs:
        globals()[f](rel)
