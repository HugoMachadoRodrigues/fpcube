"""Run the benchmark baselines on a released cube and write results tables for the paper.

python -m fpcube.benchmark.run [task1 task2 task3] [--release v1.0] [--max-train 2000000]
"""
import argparse
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from fpcube import config
from fpcube.benchmark import models as M
from fpcube.benchmark import tasks as T


def load(release):
    rel = config.RELEASE / release
    cube = xr.open_zarr(rel / "cube.zarr").load()
    fields = gpd.read_parquet(rel / "fields.parquet")
    return rel, cube, fields


def seq_array(cube, s, variables, lookback, offset=0):
    f, t = s.f.to_numpy(), s.t.to_numpy()
    out = np.empty((len(s), lookback, len(variables)), np.float32)
    for j, v in enumerate(variables):
        a = cube[v].values
        for k in range(lookback):
            out[:, k, j] = a[f, t - lookback + k + offset]
    return out


def static_array(fields, s):
    cols = [c for c in T.STATIC if c in fields]
    return fields[cols].to_numpy(np.float32)[s.f.to_numpy()]


def subsample(df, n, seed=M.SEED):
    return df if len(df) <= n else df.sample(n, random_state=seed)


def task1(cube, fields, max_train):
    s = T.task1_samples(cube, fields)
    tr = subsample(s[s.split == "train"], max_train)
    va = subsample(s[s.split == "val"], max_train // 4)
    te = {k: s[s.split == k] for k in ("test", "test_spatial")}
    Xtr, ytr = T.task1_features(cube, fields, tr)
    Xva, yva = T.task1_features(cube, fields, va)
    gbm = M.lgbm_fit(Xtr, ytr, Xva, yva)
    dyn = [v for v in T.DYN if v in cube]
    seq = lambda d: np.concatenate([seq_array(cube, d, dyn, T.LOOKBACK),
                                    np.broadcast_to(d.h.to_numpy()[:, None, None] / 12.0,
                                                    (len(d), T.LOOKBACK, 1))], axis=-1)
    net = M.train_seq(seq(tr), static_array(fields, tr), ytr, seq(va), static_array(fields, va), yva)
    rows = []
    for name, d in te.items():
        X, y = T.task1_features(cube, fields, d)
        clim = T.task1_climatology(cube, d)
        nd = cube["ndvi"].values
        last = np.array([nd[f, t - 1] for f, t in zip(d.f, d.t)])
        preds = {"persistence": last, "climatology": clim, "lightgbm": gbm.predict(X),
                 "transformer": M.predict_seq(net, seq(d), static_array(fields, d)).ravel()}
        for h in T.HORIZONS:
            k = (d.h == h).to_numpy()
            for m, p in preds.items():
                r = M.regression_metrics(y[k], p[k], ref=clim[k])
                rows.append({"task": 1, "split": name, "horizon_days": 5 * h, "model": m, **r})
    imp = pd.Series(gbm.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
    return pd.DataFrame(rows), imp, s


def task2(cube, fields):
    tracks = pd.read_parquet(config.RAW / "labels" / "hurricane_tracks_florida.parquet")
    landf = pd.read_parquet(config.RAW / "labels" / "hurricane_landfalls.parquet")
    usdm = pd.read_parquet(config.RAW / "labels" / "usdm_county_weekly.parquet")
    events = T.task2_events(fields, tracks, landf, usdm)
    s = T.task2_samples(events, cube)
    X = T.task2_features(cube, s)
    rows, scores = [], np.full(len(s), np.nan)
    dyn = ["ndvi", "ndmi", "mndwi", "vv_db", "vh_db"]
    for ev in s.event_id.unique():
        te = (s.event_id == ev).to_numpy()
        kind = s.kind[te].iloc[0]
        tr_ix = np.nonzero(~te)[0]
        rng = np.random.default_rng(M.SEED)
        va_ev = rng.choice(s.event_id[~te].unique(), max(1, int(0.15 * s.event_id[~te].nunique())), replace=False)
        va = np.isin(s.event_id, va_ev) & ~te
        tr = ~te & ~va
        gbm = M.lgbm_fit(X[tr], s.y[tr], X[va], s.y[va], objective="binary")
        p_gbm = gbm.predict(X[te])
        simple = -(X["ndvi_delta"].to_numpy()[te]) + X["mndwi_delta"].to_numpy()[te]   # greener-to-wetter drop
        for m, p in (("rule_dNDVI_dMNDWI", simple), ("lightgbm", p_gbm)):
            rows.append({"task": 2, "event_id": ev, "kind": kind, "model": m, **M.binary_metrics(s.y[te].to_numpy(), p)})
    # deep model: one leave-one-kind-out pass per event kind (cost), sequence of 6 before + 7 after
    for kind in s.kind.unique():
        te = (s.kind == kind).to_numpy()
        va = ~te & (np.random.default_rng(M.SEED).random(len(s)) < 0.15)
        tr = ~te & ~va
        seq = lambda d: seq_array(cube, d, dyn, 13, offset=7)
        zero = lambda d: np.zeros((len(d), 1), np.float32)
        net = M.train_seq(seq(s[tr]), zero(s[tr]), s.y[tr].to_numpy().astype(np.float32),
                          seq(s[va]), zero(s[va]), s.y[va].to_numpy().astype(np.float32), loss="bce")
        p = M.predict_seq(net, seq(s[te]), zero(s[te])).ravel()
        rows.append({"task": 2, "event_id": f"all_{kind}", "kind": kind, "model": "transformer_leave_kind_out",
                     **M.binary_metrics(s.y[te].to_numpy(), p)})
    ev_table = pd.DataFrame([{k: (len(v) if k in ("pos", "neg") else v) for k, v in e.items()} for e in events])
    return pd.DataFrame(rows), ev_table


def task3(cube, fields):
    usdm = pd.read_parquet(config.RAW / "labels" / "usdm_county_weekly.parquet")
    u = T.task3_samples(usdm, fields, cube)
    bags = T.task3_bags(u, fields, cube)
    agg = np.stack([np.r_[np.nanmean(b, 0), np.nanpercentile(b, 10, 0), np.nanpercentile(b, 50, 0),
                          np.nanpercentile(b, 90, 0)] for b in bags])
    rows = []
    sp = u.split.to_numpy()
    tr, va = sp == "train", sp == "val"
    # simple rule: thresholds on the county mean NDVI anomaly fitted on train (ordinal cut points)
    m = agg[:, 0]
    cuts = [np.nanquantile(m[tr], q) for q in np.cumsum(np.bincount(u.y[tr], minlength=4) / tr.sum())[:-1][::-1]]
    rule = lambda x: 3 - np.searchsorted(np.sort(cuts), x)
    gbm = M.lgbm_fit(agg[tr], u.y[tr], agg[va], u.y[va], objective="multiclass", num_class=4)
    mil = M.train_mil([bags[i] for i in np.nonzero(tr)[0]], u.y[tr].to_numpy(),
                      [bags[i] for i in np.nonzero(va)[0]], u.y[va].to_numpy())
    for name in ("test", "test_spatial"):
        te = sp == name
        y = u.y[te].to_numpy()
        preds = {"rule_anomaly_cuts": rule(m[te]), "lightgbm_bag_stats": gbm.predict(agg[te]).argmax(1),
                 "attention_mil": M.predict_mil(mil, [bags[i] for i in np.nonzero(te)[0]]).astype(int)}
        for k, p in preds.items():
            rows.append({"task": 3, "split": name, "model": k, **M.ordinal_metrics(y, np.clip(p, 0, 3))})
    return pd.DataFrame(rows), u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", default=["task1", "task2", "task3"])
    ap.add_argument("--release", default="v1.0")
    ap.add_argument("--max-train", type=int, default=2_000_000)
    a = ap.parse_args()
    rel, cube, fields = load(a.release)
    out = rel / "benchmark"
    out.mkdir(exist_ok=True)
    if "task1" in a.tasks:
        r, imp, s = task1(cube, fields, a.max_train)
        r.to_csv(out / "task1_results.csv", index=False)
        imp.to_csv(out / "task1_lightgbm_importance.csv")
        s[["f", "t", "h", "split"]].to_parquet(out / "task1_samples.parquet", index=False)
    if "task2" in a.tasks:
        r, ev = task2(cube, fields)
        r.to_csv(out / "task2_results.csv", index=False)
        ev.to_csv(out / "task2_events.csv", index=False)
    if "task3" in a.tasks:
        r, u = task3(cube, fields)
        r.to_csv(out / "task3_results.csv", index=False)
        u[["county_fips", "valid_start", "y", "split"]].to_parquet(out / "task3_samples.parquet", index=False)
    print(json.dumps({"out": str(out)}))


if __name__ == "__main__":
    main()
