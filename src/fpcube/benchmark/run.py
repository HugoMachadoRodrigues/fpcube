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


def task3(cube, fields, horizon=4):
    """Drought class change: from the USDM map at t0, the county's class and its recent change, and the field
    evidence of the 30 days before t0, predict whether the class is better, the same or worse `horizon` weeks later.
    LightGBM is fitted with and without the field evidence, which isolates what the cube adds to the USDM history."""
    from sklearn.metrics import roc_auc_score
    usdm = pd.read_parquet(config.RAW / "labels" / "usdm_county_weekly.parquet")
    u = T.task3_samples(usdm, fields, cube, horizon=horizon)
    bags = T.task3_bags(u, fields, cube)
    log("task3", f"{len(u):,} county-weeks, {sum(len(b) for b in bags):,} field instances")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agg = np.stack([np.r_[np.nanmean(b, 0), np.nanpercentile(b, [10, 50, 90], 0).ravel()] for b in bags])
    cov = T.county_covariates(u)
    sp = u.split.to_numpy()
    tr, va = sp == "train", sp == "val"
    d = u.direction.to_numpy()
    rule = T.anomaly_cut_rule(agg[tr, 0], d[tr])
    clim = T.transition_climatology(u[tr])
    fit = lambda X: M.lgbm_fit(X[tr], d[tr], X[va], d[va], objective="multiclass", num_class=3,
                               num_threads=THREADS["task3"])
    gbm_c = fit(cov)
    gbm_f = fit(np.c_[cov, agg])
    log("task3", "LightGBM done")
    mbags = [np.c_[b, np.repeat(c[None], len(b), 0)] for b, c in zip(bags, cov)]   # county inputs on every field
    mil = M.train_mil([mbags[i] for i in np.nonzero(tr)[0]], d[tr], [mbags[i] for i in np.nonzero(va)[0]], d[va],
                      n_class=3)
    log("task3", "attention MIL done")
    month = u.valid_start.dt.to_period("M").astype(str).to_numpy()
    rows, cms, pred_rows, gain = [], {}, [], []
    for name in ("test", "test_spatial"):
        te = sp == name
        y = d[te]
        sub = u[te]
        probs = {"transition_climatology": clim(sub), "lightgbm_usdm_only": gbm_c.predict(cov[te]),
                 "lightgbm_with_fields": gbm_f.predict(np.c_[cov, agg][te]),
                 "attention_mil": M.predict_mil_proba(mil, [mbags[i] for i in np.nonzero(te)[0]])}
        preds = {"no_change": np.ones(te.sum(), int),
                 "momentum": (np.sign(sub.momentum.to_numpy()) + 1).astype(int),
                 "ndvi_anomaly_rule": rule(agg[te, 0]).astype(int),
                 **{k: v.argmax(1) for k, v in probs.items()}}
        cl1, cl2 = sub.county_fips.to_numpy(), month[te]
        for k, pk in preds.items():
            pr = probs.get(k)
            r = M.direction_metrics(y, pk, pr)
            stats = {"macro_f1": lambda w: M.direction_metrics(y, pk, w=w)["macro_f1"]}
            if pr is not None:
                stats["auroc_worse"] = lambda w: roc_auc_score(y == 2, pr[:, 2], sample_weight=w)
                stats["auroc_better"] = lambda w: roc_auc_score(y == 0, pr[:, 0], sample_weight=w)
            for st, f in stats.items():
                r[f"{st}_lo"], r[f"{st}_hi"] = M.pigeonhole_bootstrap(f, cl1, cl2, B=300)
            rows.append({"task": 3, "split": name, "horizon_weeks": horizon, "model": k, **r})
            if name == "test":
                from sklearn.metrics import confusion_matrix
                cms[k] = pd.DataFrame(confusion_matrix(y, pk, labels=[0, 1, 2]), index=M.DIRECTIONS,
                                      columns=M.DIRECTIONS)
        for k, nm in ((2, "worse"), (0, "better")):          # what the field evidence adds, paired by sample
            f_, c_ = probs["lightgbm_with_fields"][:, k], probs["lightgbm_usdm_only"][:, k]
            dlt = lambda w: roc_auc_score(y == k, f_, sample_weight=w) - roc_auc_score(y == k, c_, sample_weight=w)
            lo, hi = M.pigeonhole_bootstrap(dlt, cl1, cl2, B=300)
            gain.append({"split": name, "direction": nm, "auroc_with_fields_minus_usdm_only": float(dlt(None)),
                         "lo": lo, "hi": hi})
        pred_rows.append(pd.DataFrame({"county_fips": cl1, "valid_start": sub.valid_start.to_numpy(), "split": name,
                                       "direction": y, **{f"pred_{k}": v for k, v in preds.items()},
                                       **{f"p_worse_{k}": v[:, 2] for k, v in probs.items()}}))
    u.attrs["predictions"] = pd.concat(pred_rows, ignore_index=True)
    u.attrs["gain"] = pd.DataFrame(gain)
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
        for f_ in out.glob("task3_confusion*.csv"):
            f_.unlink()
        for k, cm in cms.items():
            cm.to_csv(out / f"task3_confusion_{k}.csv")
        u.attrs.pop("predictions").to_parquet(out / "task3_predictions.parquet", index=False)
        u.attrs.pop("gain").to_csv(out / "task3_field_evidence_gain.csv", index=False)
        info["n_dropped_crossing_split"] = u.attrs.get("n_dropped_crossing_split")
        u[["county_fips", "valid_start", "target_date", "y", "momentum", "direction", "split"]].to_parquet(
            out / "task3_samples.parquet", index=False)
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / f"{name}_run_info.json").write_text(json.dumps(info, indent=2))
    log(name, "finished")
    return name


def paper_text(out):
    """Descriptive sentences with the benchmark numbers (benchmark_text.json, ready to be copied into the manuscript).
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
    if (out / "task3_results.csv").exists() and "direction_acc" not in pd.read_csv(out / "task3_results.csv", nrows=1):
        r = pd.read_csv(out / "task3_results.csv").set_index(["split", "model"])
        g = pd.read_csv(out / "task3_field_evidence_gain.csv").set_index(["split", "direction"])
        x = lambda m, c: r.loc[("test", m), c]
        ci = lambda m, c: "{:.2f} [{:.2f}, {:.2f}]".format(x(m, c), x(m, c + "_lo"), x(m, c + "_hi"))
        n = lambda k: f"{int(x('no_change', 'n_' + k)):,}".replace(",", "\\,")
        gg = lambda k: "{:+.2f} [{:+.2f}, {:+.2f}]".format(*g.loc[("test", k)].to_numpy())
        parts.append(
            "In Task~3 ({nt} county-weeks in the temporal test set: {nb} better, {ns} the same and {nw} worse four "
            "weeks later), the AUROC of a worsening [95\\,\\% interval] is {cw} for the transition climatology, {gw} for "
            "LightGBM on the USDM history alone, {fw} for LightGBM with the field evidence and {mw} for attention MIL; "
            "the AUROC of an improvement is {cb}, {gb}, {fb} and {mb}. Adding the field evidence changes the AUROC of "
            "LightGBM by {dw} for worsening and {db} for improvement. Macro F1 is {nc} for no change, {mo} for the "
            "momentum rule, {ru} for the NDVI-anomaly rule, {f_cl}, {f_gc}, {f_gf} and {f_mil} for the four models "
            "above.".format(
                nt=f"{int(x('no_change', 'n')):,}".replace(",", "\\,"), nb=n("better"), ns=n("same"), nw=n("worse"),
                cw=ci("transition_climatology", "auroc_worse"), gw=ci("lightgbm_usdm_only", "auroc_worse"),
                fw=ci("lightgbm_with_fields", "auroc_worse"), mw=ci("attention_mil", "auroc_worse"),
                cb=ci("transition_climatology", "auroc_better"), gb=ci("lightgbm_usdm_only", "auroc_better"),
                fb=ci("lightgbm_with_fields", "auroc_better"), mb=ci("attention_mil", "auroc_better"),
                dw=gg("worse"), db=gg("better"), nc=f"{x('no_change', 'macro_f1'):.2f}",
                mo=f"{x('momentum', 'macro_f1'):.2f}", ru=f"{x('ndvi_anomaly_rule', 'macro_f1'):.2f}",
                f_cl=f"{x('transition_climatology', 'macro_f1'):.2f}", f_gc=f"{x('lightgbm_usdm_only', 'macro_f1'):.2f}",
                f_gf=f"{x('lightgbm_with_fields', 'macro_f1'):.2f}", f_mil=f"{x('attention_mil', 'macro_f1'):.2f}"))
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
