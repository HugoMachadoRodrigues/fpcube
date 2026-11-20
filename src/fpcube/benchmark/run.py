"""Run the benchmark baselines on a released cube and write results tables for the paper.

python -m fpcube.benchmark.run [task1 task2 task3] [--release v1.0] [--max-train 2000000] [--serial]

The three tasks are independent and run in parallel processes, each loading only the cube variables it uses.
Inputs never contain gap-filled optical values (tasks.observed_only). Within Task 2 the leave-one-event-out LightGBM
fits run in parallel. Every task writes <task>_run_info.json with the code version (fpcube.provenance). Progress is logged with timestamps.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from fpcube import config
from fpcube.benchmark import models as M
from fpcube.benchmark import tasks as T

VARS = {"task1": T.DYN + ["n_optical_obs"],
        "task2": ["ndvi", "ndmi", "mndwi", "evi", "vv_db", "vh_db", "n_optical_obs"],
        "task3": ["ndvi", "ndmi", "vv_db", "ppt_sum", "n_optical_obs"]}
THREADS = {"task1": 4, "task2": 2, "task3": 2}     # LightGBM threads per fit (10 cores on an M1 Max)
TASK2_WORKERS = 4                                    # parallel leave-one-event-out fits
T0 = time.time()


def log(task, msg):
    print(f"{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:6.1f} min [{task}] {msg}", flush=True)


def load(release, variables=None):
    rel = config.RELEASE / release
    ds = xr.open_zarr(rel / "cube.zarr")
    cube = T.observed_only((ds[[v for v in variables if v in ds]] if variables else ds).load())
    fields = gpd.read_parquet(rel / "fields.parquet")
    return rel, cube, fields


def seq_array(cube, s, variables, lookback, offset=0):
    f = s.f.to_numpy()[:, None]
    tt = s.t.to_numpy()[:, None] - lookback + offset + np.arange(lookback)[None]
    out = np.empty((len(s), lookback, len(variables)), np.float32)
    for j, v in enumerate(variables):
        out[:, :, j] = cube[v].values[f, tt]
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
    log("task1", f"samples: {len(s):,} (train {len(tr):,}, val {len(va):,}, "
                 + ", ".join(f"{k} {len(v):,}" for k, v in te.items())
                 + f"); {s.attrs['n_dropped_crossing_split']:,} dropped because the target crosses a split boundary")
    Xtr, ytr = T.task1_features(cube, fields, tr)
    Xva, yva = T.task1_features(cube, fields, va)
    gbm = M.lgbm_fit(Xtr, ytr, Xva, yva, num_threads=THREADS["task1"])
    log("task1", f"LightGBM done ({gbm.best_iteration} rounds)")
    dyn = [v for v in T.DYN if v in cube]
    seq = lambda d: np.concatenate([seq_array(cube, d, dyn, T.LOOKBACK),
                                    np.broadcast_to(d.h.to_numpy()[:, None, None] / 12.0,
                                                    (len(d), T.LOOKBACK, 1))], axis=-1)
    net = M.train_seq(seq(tr), static_array(fields, tr), ytr, seq(va), static_array(fields, va), yva)
    log("task1", "transformer done")
    rows = []
    county = fields.county.to_numpy()
    for name, d in te.items():
        X, y = T.task1_features(cube, fields, d)
        clim = T.task1_climatology(cube, d)
        preds = {"persistence": X.ndvi_last.to_numpy(), "climatology": clim,
                 "lightgbm": gbm.predict(X, num_threads=THREADS["task1"]),
                 "transformer": M.predict_seq(net, seq(d), static_array(fields, d)).ravel()}
        common = np.isfinite(y) & np.all([np.isfinite(p) for p in preds.values()], axis=0)   # same samples for all
        cl1, cl2 = county[d.f.to_numpy()], d.t.to_numpy()
        for h in T.HORIZONS:
            k = (d.h == h).to_numpy() & common
            for m, p in preds.items():
                r = M.regression_metrics(y[k], p[k], ref=clim[k])
                r.update(M.regression_ci(y[k], p[k], clim[k], cl1[k], cl2[k]))
                rows.append({"task": 1, "split": name, "horizon_days": 5 * h, "model": m, **r,
                             "n_candidates": int(((d.h == h).to_numpy()).sum()),
                             "persistence_age_days_median": float(np.median(X.ndvi_last_age_days.to_numpy()[k]))})
    imp = pd.Series(gbm.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
    info = {"n_samples": len(s), "n_dropped_crossing_split": s.attrs["n_dropped_crossing_split"],
            **{f"n_{k}": int((s.split == k).sum()) for k in ("train", "val", "test", "test_spatial")},
            "n_train_used": len(tr), "n_val_used": len(va)}
    return pd.DataFrame(rows), imp, s, info


def _task2_scores(s, scores, experiment):
    """AUROC etc. per held-out event and pooled per kind, for every model in an experiment."""
    rows = []
    for m, p in scores.items():
        ok = np.isfinite(p)
        for ev, g in s[ok].groupby("event_id"):
            rows.append({"task": 2, "experiment": experiment, "event_id": ev, "kind": g.kind.iloc[0], "model": m,
                         **M.binary_metrics(g.y.to_numpy(), p[g.index.to_numpy()])})
        for kind, g in s[ok].groupby("kind"):
            rows.append({"task": 2, "experiment": experiment, "event_id": f"pooled_{kind}", "kind": kind, "model": m,
                         **M.binary_metrics(g.y.to_numpy(), p[g.index.to_numpy()])})
    return rows


def task2(cube, fields):
    from joblib import Parallel, delayed
    tracks = pd.read_parquet(config.RAW / "labels" / "hurricane_tracks_florida.parquet")
    landf = pd.read_parquet(config.RAW / "labels" / "hurricane_landfalls.parquet")
    usdm = pd.read_parquet(config.RAW / "labels" / "usdm_county_weekly.parquet")
    events = T.task2_events(fields, tracks, landf, usdm)
    s = T.task2_samples(events, cube)
    X = T.task2_features(cube, s)
    log("task2", f"{len(events)} events, {len(s):,} samples")
    Xa, ya, eid = X.to_numpy(np.float32), s.y.to_numpy(), s.event_id.to_numpy()
    rule = (-X["ndvi_delta"] + X["mndwi_delta"]).to_numpy()            # greener-to-wetter drop, no training
    # experiment 1: leave one event out (rule, LightGBM)
    folds = Parallel(n_jobs=TASK2_WORKERS, backend="loky")(
        delayed(M.event_fold)(Xa, ya, eid, ev, THREADS["task2"]) for ev in s.event_id.unique())
    gbm = np.full(len(s), np.nan)
    for ev, p in folds:
        gbm[eid == ev] = p
    rows = _task2_scores(s, {"rule_dNDVI_dMNDWI": rule, "lightgbm": gbm}, "leave_one_event_out")
    log("task2", "leave-one-event-out done")
    # experiment 2: leave one kind out (rule, LightGBM, transformer); validation = whole events of the other kinds
    dyn = ["ndvi", "ndmi", "mndwi", "vv_db", "vh_db"]
    seq = lambda m: seq_array(cube, s[m], dyn, 13, offset=7)
    zero = lambda m: np.zeros((int(m.sum()), 1), np.float32)
    gbm_k, net_k = np.full(len(s), np.nan), np.full(len(s), np.nan)
    for kind in s.kind.unique():
        te = (s.kind == kind).to_numpy()
        tr, va = M.grouped_validation(eid, te)
        g = M.lgbm_fit(Xa[tr], ya[tr], Xa[va], ya[va], objective="binary", num_threads=8)
        gbm_k[te] = g.predict(Xa[te], num_threads=8)
        net = M.train_seq(seq(tr), zero(tr), ya[tr].astype(np.float32), seq(va), zero(va), ya[va].astype(np.float32),
                          loss="bce")
        net_k[te] = M.predict_seq(net, seq(te), zero(te)).ravel()
        log("task2", f"leave-{kind}-out done")
    rows += _task2_scores(s, {"rule_dNDVI_dMNDWI": rule, "lightgbm": gbm_k, "transformer": net_k},
                          "leave_one_kind_out")
    ev_table = pd.DataFrame([{k: (len(v) if k in ("pos", "neg") else v) for k, v in e.items()} for e in events])
    return pd.DataFrame(rows), ev_table


def task3(cube, fields):
    usdm = pd.read_parquet(config.RAW / "labels" / "usdm_county_weekly.parquet")
    u = T.task3_samples(usdm, fields, cube)
    bags = T.task3_bags(u, fields, cube)
    log("task3", f"{len(u):,} county-weeks, {sum(len(b) for b in bags):,} field instances")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agg = np.stack([np.r_[np.nanmean(b, 0), np.nanpercentile(b, [10, 50, 90], 0).ravel()] for b in bags])
    rows, cms, pred_rows = [], {}, []
    sp = u.split.to_numpy()
    tr, va = sp == "train", sp == "val"
    ytr = u.y[tr].to_numpy()
    m = agg[:, 0]
    rule = T.anomaly_cut_rule(m[tr], ytr)
    majority = int(np.bincount(ytr, minlength=4).argmax())
    gbm = M.lgbm_fit(agg[tr], ytr, agg[va], u.y[va], objective="multiclass", num_class=4,
                     num_threads=THREADS["task3"])
    log("task3", "LightGBM done")
    mil = M.train_mil([bags[i] for i in np.nonzero(tr)[0]], ytr,
                      [bags[i] for i in np.nonzero(va)[0]], u.y[va].to_numpy())
    log("task3", "attention MIL done")
    month = u.valid_start.dt.to_period("M").astype(str).to_numpy()
    for name in ("test", "test_spatial"):
        te = (sp == name) & u.y_prev.notna().to_numpy()          # same county-weeks for every model
        y = u.y[te].to_numpy()
        preds = {"majority_class": np.full(te.sum(), majority), "usdm_last_week": u.y_prev[te].to_numpy().astype(int),
                 "rule_anomaly_cuts": rule(m[te]), "lightgbm_bag_stats": gbm.predict(agg[te]).argmax(1),
                 "attention_mil": M.predict_mil(mil, [bags[i] for i in np.nonzero(te)[0]]).astype(int)}
        cl1, cl2 = u.county_fips.to_numpy()[te], month[te]
        change = y != u.y_prev[te].to_numpy().astype(int)     # county-weeks whose USDM class differs from last week
        for k, p in preds.items():
            p = np.clip(p, 0, 3)
            preds[k] = p
            for sub, keep in ((name, slice(None)), (f"{name}_change", change)):
                ys, ps, c1, c2 = y[keep], p[keep], cl1[keep], cl2[keep]
                r = M.ordinal_metrics(ys, ps)
                for stat in ("qwk", "macro_f1"):
                    lo, hi = M.pigeonhole_bootstrap(lambda w: M.ordinal_metrics(ys, ps, w)[stat], c1, c2, B=300)
                    r[f"{stat}_lo"], r[f"{stat}_hi"] = lo, hi
                prev = u.y_prev[te].to_numpy().astype(int)[keep]   # share with the right direction of change
                r["direction_acc"] = float(np.mean(np.sign(ps - prev) == np.sign(ys - prev)))
                rows.append({"task": 3, "split": sub, "model": k, **r})
            if name == "test":
                from sklearn.metrics import confusion_matrix
                cms[k] = pd.DataFrame(confusion_matrix(y, p, labels=[0, 1, 2, 3]),
                                      index=["none", "D0", "D1", "D2+"], columns=["none", "D0", "D1", "D2+"])
        pred_rows.append(pd.DataFrame({"county_fips": u.county_fips.to_numpy()[te], "valid_start": u.valid_start.to_numpy()[te],
                                       "split": name, "y": y, **{f"pred_{k}": v for k, v in preds.items()}}))
    u.attrs["predictions"] = pd.concat(pred_rows, ignore_index=True)
    return pd.DataFrame(rows), u, cms


def run_task(name, release, max_train):
    """Load what the task needs, run it, write its outputs. Runs in its own process."""
    from fpcube import provenance
    log(name, "loading " + ", ".join(VARS[name]))
    rel, cube, fields = load(release, VARS[name])
    out = rel / "benchmark"
    out.mkdir(exist_ok=True)
    log(name, "start")
    info = {"task": name, "code": provenance.info(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if name == "task1":
        r, imp, s, extra = task1(cube, fields, max_train)
        info.update(extra)
        r.to_csv(out / "task1_results.csv", index=False)
        imp.to_csv(out / "task1_lightgbm_importance.csv")
        s[["f", "t", "h", "split"]].to_parquet(out / "task1_samples.parquet", index=False)
    elif name == "task2":
        r, ev = task2(cube, fields)
        r.to_csv(out / "task2_results.csv", index=False)
        ev.to_csv(out / "task2_events.csv", index=False)
    else:
        r, u, cms = task3(cube, fields)
        r.to_csv(out / "task3_results.csv", index=False)
        for k, cm in cms.items():
            cm.to_csv(out / f"task3_confusion_{k}.csv")
        (out / "task3_confusion.csv").unlink(missing_ok=True)
        u.attrs.pop("predictions").to_parquet(out / "task3_predictions.parquet", index=False)
        u[["county_fips", "valid_start", "y", "y_prev", "split"]].to_parquet(out / "task3_samples.parquet",
                                                                            index=False)
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / f"{name}_run_info.json").write_text(json.dumps(info, indent=2))
    log(name, "finished")
    return name


def paper_text(out):
    """Descriptive sentences with the benchmark numbers (benchmark_text.json, read by scripts/paper_numbers.py).
    They state results only; interpretation is left to the authors."""
    t = {}
    parts = []
    if (out / "task1_results.csv").exists():
        r = pd.read_csv(out / "task1_results.csv")
        te = r[r.split == "test"].set_index(["model", "horizon_days"])
        sp = r[r.split == "test_spatial"].set_index(["model", "horizon_days"])
        f = lambda d, m, h: "{:.3f} [{:.3f}, {:.3f}]".format(d.loc[(m, h), "rmse"], d.loc[(m, h), "rmse_lo"],
                                                               d.loc[(m, h), "rmse_hi"])
        sk = lambda d, m, h: "{:.0f}\\,\\% [{:.0f}, {:.0f}]".format(*(100 * d.loc[(m, h), c] for c in
                                                                     ("skill_vs_ref", "skill_lo", "skill_hi")))
        parts.append(
            "In Task~1 (Table~\\ref{{tab:bench1}}; {n15} test samples at 15 days and {n60} at 60 days, the same for "
            "every model), the NDVI RMSE [95\\,\\% interval] at 15 and 60 days is {p15} and {p60} for persistence, "
            "{c15} and {c60} for climatology, {l15} and {l60} for LightGBM, and {t15} and {t60} for the transformer. "
            "The mean-squared-error skill score against climatology at 30 days is {ls30} for LightGBM and {ts30} for "
            "the transformer on the temporal test set, and {lss30} and {tss30} on the withheld counties, where the "
            "RMSE of climatology is {csp30:.3f} against {cte30:.3f} on the temporal test set.".format(
                n15=f"{int(te.loc[('lightgbm', 15), 'n']):,}".replace(",", "\\,"),
                n60=f"{int(te.loc[('lightgbm', 60), 'n']):,}".replace(",", "\\,"),
                p15=f(te, "persistence", 15), p60=f(te, "persistence", 60),
                c15=f(te, "climatology", 15), c60=f(te, "climatology", 60),
                l15=f(te, "lightgbm", 15), l60=f(te, "lightgbm", 60),
                t15=f(te, "transformer", 15), t60=f(te, "transformer", 60),
                ls30=sk(te, "lightgbm", 30), ts30=sk(te, "transformer", 30),
                lss30=sk(sp, "lightgbm", 30), tss30=sk(sp, "transformer", 30),
                csp30=sp.loc[("climatology", 30), "rmse"], cte30=te.loc[("climatology", 30), "rmse"]))
    if (out / "task2_results.csv").exists():
        r = pd.read_csv(out / "task2_results.csv")
        ev = r[~r.event_id.str.startswith("pooled_")]
        nm = {"rule_dNDVI_dMNDWI": "the NDVI--MNDWI change rule", "lightgbm": "LightGBM", "transformer": "the transformer"}
        sent = []
        for exp, label in (("leave_one_event_out", "with one event left out"),
                           ("leave_one_kind_out", "With a whole event type left out")):
            e = ev[ev.experiment == exp]
            kinds = sorted(e.kind.unique())
            per_model = []
            for m in [x for x in nm if x in set(e.model)]:
                cells = []
                for k in kinds:
                    v = e[(e.model == m) & (e.kind == k)].auroc
                    lo, hi = M.median_ci(v)
                    cells.append(f"{k.replace('_', ' ')} {v.median():.2f} [{lo:.2f}, {hi:.2f}]")
                per_model.append(f"{nm[m]}: " + "; ".join(cells))
            sent.append(f"{label}, the median AUROC per event [95\\,\\% interval over events] is " +
                        " -- ".join(per_model) + ".")
        n_ev = ev[ev.experiment == "leave_one_event_out"].groupby("kind").event_id.nunique()
        parts.append("In Task~2 ({}), ".format(", ".join(f"{v} {k.replace('_', ' ')} events" for k, v in n_ev.items()))
                     + " ".join(sent))
        g = ev[(ev.experiment == "leave_one_event_out") & (ev.model == "lightgbm")].groupby("kind").auroc.median()
        t["TaskTwoSummary"] = ("the median AUROC per event of LightGBM with one event left out is " +
                               "; ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in g.items()))
    if (out / "task3_results.csv").exists():
        r = pd.read_csv(out / "task3_results.csv").set_index(["split", "model"])
        x = lambda m, c: r.loc[("test", m), c]
        ci = lambda m, c: "{:.2f} [{:.2f}, {:.2f}]".format(x(m, c), x(m, c + "_lo"), x(m, c + "_hi"))
        parts.append(
            "In Task~3 ({n} county-weeks in the temporal test set) the quadratic weighted kappa [95\\,\\% interval] "
            "is {mil} for attention MIL, {gbm} for LightGBM on bag statistics, {rule} for the NDVI-anomaly cut points, "
            "{last} for last week's USDM class and {maj} for the majority class; macro F1 is {milf}, {gbmf}, {rulef}, "
            "{lastf} and {majf}. Attention MIL recovers {r3:.0f}\\,\\% of the D2+ county-weeks ({n3} cases) and "
            "{r0:.0f}\\,\\% of those without drought.".format(
                n=f"{int(x('attention_mil', 'n')):,}".replace(",", "\\,"),
                mil=ci("attention_mil", "qwk"), gbm=ci("lightgbm_bag_stats", "qwk"), rule=ci("rule_anomaly_cuts", "qwk"),
                last=ci("usdm_last_week", "qwk"), maj=f"{x('majority_class', 'qwk'):.2f}",
                milf=f"{x('attention_mil', 'macro_f1'):.2f}", gbmf=f"{x('lightgbm_bag_stats', 'macro_f1'):.2f}",
                rulef=f"{x('rule_anomaly_cuts', 'macro_f1'):.2f}", lastf=f"{x('usdm_last_week', 'macro_f1'):.2f}",
                majf=f"{x('majority_class', 'macro_f1'):.2f}", r3=100 * x("attention_mil", "recall_d2plus"),
                n3=int(x("attention_mil", "n_d2plus")), r0=100 * x("attention_mil", "recall_none")))
        if ("test_change", "attention_mil") in r.index:
            xc = lambda m, c: r.loc[("test_change", m), c]
            cc = lambda m: "{:.2f} [{:.2f}, {:.2f}]".format(xc(m, "qwk"), xc(m, "qwk_lo"), xc(m, "qwk_hi"))
            t["TaskThreeChange"] = (
                "In the {n} test county-weeks whose USDM class differs from the previous week, the quadratic weighted "
                "kappa is {last} for last week's class, {rule} for the NDVI-anomaly cut points, {gbm} for LightGBM and "
                "{mil} for attention MIL, and the direction of the change (worse or better than last week) is right in "
                "{dr:.0f}\\,\\%, {dg:.0f}\\,\\% and {dm:.0f}\\,\\% of these weeks for the last three, against "
                "{dj:.0f}\\,\\% for the majority class.").format(dj=100 * xc("majority_class", "direction_acc"),
                n=f"{int(xc('attention_mil', 'n')):,}".replace(",", "\\,"), last=cc("usdm_last_week"),
                rule=cc("rule_anomaly_cuts"), gbm=cc("lightgbm_bag_stats"), mil=cc("attention_mil"),
                dr=100 * xc("rule_anomaly_cuts", "direction_acc"), dg=100 * xc("lightgbm_bag_stats", "direction_acc"),
                dm=100 * xc("attention_mil", "direction_acc"))
    t["BenchmarkResults"] = " ".join(parts) + " Figure~\\ref{fig:benchmark} summarises the results."
    (out / "benchmark_text.json").write_text(json.dumps(t, indent=2))
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", default=["task1", "task2", "task3"])
    ap.add_argument("--release", default="v1.0")
    ap.add_argument("--max-train", type=int, default=2_000_000)
    ap.add_argument("--serial", action="store_true", help="run the tasks one after another")
    a = ap.parse_args()
    out = config.RELEASE / a.release / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    if a.serial or len(a.tasks) == 1:
        for t in a.tasks:
            run_task(t, a.release, a.max_train)
    else:
        with ProcessPoolExecutor(len(a.tasks), mp_context=mp.get_context("spawn")) as ex:
            for f in [ex.submit(run_task, t, a.release, a.max_train) for t in a.tasks]:
                f.result()
    paper_text(out)
    log("all", "done")
    print(json.dumps({"out": str(out)}))


if __name__ == "__main__":
    main()
