"""Baselines for the three benchmark tasks: simple rule, gradient-boosted trees (LightGBM), deep learning (PyTorch).

All deep models are small on purpose (they are reference points, not the state of the art) and are trained with
fixed seeds, early stopping on the validation split and the same budget for every task.
"""
import lightgbm  # noqa: F401  -- must load before torch: two OpenMP runtimes segfault on macOS otherwise
import numpy as np
import pandas as pd
import torch
from torch import nn

SEED = 20261116
# LightGBM (Homebrew libomp) and torch (bundled libomp) load two OpenMP runtimes on macOS; keeping torch's CPU side
# single-threaded avoids the crash. Deep models run on MPS/CUDA when available; the MIL model is small.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- metrics
def regression_metrics(y, p, ref=None):
    ok = np.isfinite(y) & np.isfinite(p)
    if ref is not None:
        ok &= np.isfinite(ref)
    y, p = y[ok], p[ok]
    out = {"n": int(ok.sum()), "rmse": float(np.sqrt(np.mean((p - y) ** 2))), "mae": float(np.mean(np.abs(p - y))),
           "bias": float(np.mean(p - y)), "r2": float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))}
    if ref is not None:
        r = ref[ok]
        out["skill_vs_ref"] = float(1 - np.sum((p - y) ** 2) / np.sum((r - y) ** 2))
    return out


def binary_metrics(y, s):
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return {"n": int(len(y))}
    thr = np.quantile(s, 1 - y.mean())
    return {"n": int(len(y)), "auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "f1_at_prevalence": float(f1_score(y, s >= thr)), "prevalence": float(y.mean())}


DIRECTIONS = ("better", "same", "worse")


def direction_metrics(y, p, prob=None, w=None):
    """Task 3: macro F1, balanced accuracy and accuracy of the change direction (0 better, 1 same, 2 worse); without
    weights also recall and count per direction, and with class probabilities the AUROC of 'worse' and of
    'better' against the rest."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
    out = {"n": int(len(y)),
           "macro_f1": float(f1_score(y, p, average="macro", labels=[0, 1, 2], sample_weight=w, zero_division=0)),
           "balanced_accuracy": float(balanced_accuracy_score(y, p, sample_weight=w)),
           "accuracy": float(np.average(y == p, weights=w))}
    if w is None:
        rec = recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
        for k, nm in enumerate(DIRECTIONS):
            out[f"recall_{nm}"] = float(rec[k])
            out[f"n_{nm}"] = int((y == k).sum())
        if prob is not None:
            for k, nm in ((2, "worse"), (0, "better")):
                out[f"auroc_{nm}"] = float(roc_auc_score(y == k, prob[:, k])) if 0 < (y == k).sum() < len(y) \
                    else float("nan")
    return out


def ordinal_metrics(y, p, w=None):
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, recall_score
    out = {"n": int(len(y)), "macro_f1": float(f1_score(y, p, average="macro", labels=[0, 1, 2, 3], sample_weight=w,
                                                        zero_division=0)),
           "balanced_accuracy": float(balanced_accuracy_score(y, p, sample_weight=w)),
           "qwk": float(cohen_kappa_score(y, p, weights="quadratic", sample_weight=w)),
           "accuracy": float(np.average(y == p, weights=w))}
    if w is None:
        rec = recall_score(y, p, labels=[0, 1, 2, 3], average=None, zero_division=0)
        for k, nm in enumerate(("none", "d0", "d1", "d2plus")):
            out[f"recall_{nm}"] = float(rec[k])
            out[f"n_{nm}"] = int((y == k).sum())
    return out


# ---------------------------------------------------------------- uncertainty
def _cluster_weights(codes, rng):
    """Bootstrap weight per sample: each cluster drawn with replacement (multinomial count per cluster)."""
    n = codes.max() + 1
    return rng.multinomial(n, np.full(n, 1 / n))[codes]


def pigeonhole_bootstrap(stat, cl1, cl2, B=500, seed=SEED, q=(2.5, 97.5)):
    """Percentile interval of stat(weights) under two-way cluster resampling (Owen 2007): clusters of cl1 (e.g.
    counties) and of cl2 (e.g. dates) are resampled independently and a sample's weight is the product of its two
    cluster counts. Respects dependence within a county over time and across counties on the same date."""
    rng = np.random.default_rng(seed)
    c1, c2 = pd.factorize(cl1)[0], pd.factorize(cl2)[0]
    vals = [stat(_cluster_weights(c1, rng) * _cluster_weights(c2, rng)) for _ in range(B)]
    return tuple(float(x) for x in np.nanpercentile(vals, q))


def regression_ci(y, p, ref, cl1, cl2, B=500):
    """95 % intervals of RMSE and MSE skill against ref, from per-(cl1, cl2) cell sums (fast for millions of rows)."""
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(ref)
    d = pd.DataFrame({"a": np.asarray(cl1)[ok], "b": np.asarray(cl2)[ok], "e": (p[ok] - y[ok]) ** 2,
                      "r": (ref[ok] - y[ok]) ** 2, "n": 1.0}).groupby(["a", "b"], sort=False).sum().reset_index()
    E, R, N = d.e.to_numpy(), d.r.to_numpy(), d.n.to_numpy()
    rmse = pigeonhole_bootstrap(lambda w: np.sqrt((w * E).sum() / (w * N).sum()), d.a, d.b, B)
    skill = pigeonhole_bootstrap(lambda w: 1 - (w * E).sum() / (w * R).sum(), d.a, d.b, B)
    return {"rmse_lo": rmse[0], "rmse_hi": rmse[1], "skill_lo": skill[0], "skill_hi": skill[1]}


def median_ci(values, B=2000, seed=SEED):
    """95 % percentile interval of the median of independent units (events) by resampling them."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = np.median(v[rng.integers(0, len(v), (B, len(v)))], axis=1)
    return tuple(float(x) for x in np.percentile(m, (2.5, 97.5)))


# ---------------------------------------------------------------- gradient boosting
def lgbm_fit(Xtr, ytr, Xva, yva, objective="regression", num_class=None, num_threads=8):
    import lightgbm as lgb
    params = dict(objective=objective, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED,
                  verbose=-1, num_threads=num_threads, force_col_wise=True)
    if num_class:
        params["num_class"] = num_class
    dtr = lgb.Dataset(Xtr, ytr)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


# ---------------------------------------------------------------- deep models
class SeqEncoder(nn.Module):
    """Transformer encoder over a short multivariate sequence with NaN-aware input (value, observed-mask)."""

    def __init__(self, n_dyn, n_static, d=64, heads=4, layers=2, seq_len=12, n_out=1):
        super().__init__()
        self.inp = nn.Linear(2 * n_dyn, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        self.stat = nn.Linear(n_static, d) if n_static else None
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_out))

    def forward(self, x, s=None):
        m = torch.isfinite(x)
        x = torch.where(m, x, torch.zeros_like(x))
        h = self.inp(torch.cat([x, m.float()], -1)) + self.pos[:, :x.shape[1]]
        if self.stat is not None and s is not None:
            h = h + self.stat(torch.nan_to_num(s)).unsqueeze(1)
        h = self.enc(h).mean(1)
        return self.head(h)


class AttentionMIL(nn.Module):
    """Gated attention multiple-instance learning (Ilse et al., 2018) with an ordinal (cumulative logit) head."""

    def __init__(self, n_in, d=64, n_class=4):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(2 * n_in, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.V = nn.Linear(d, d)
        self.U = nn.Linear(d, d)
        self.w = nn.Linear(d, 1)
        self.score = nn.Linear(d, 1)
        self.cut = nn.Parameter(torch.linspace(-1, 1, n_class - 1))

    def forward(self, bag, mask=None):
        """bag: (instances, features) or a padded batch (bags, instances, features) with mask (bags, instances),
        True for real instances. Returns P(y > k), k = 0..K-2, and the attention weights."""
        single = bag.dim() == 2
        if single:
            bag = bag.unsqueeze(0)
            mask = torch.ones(bag.shape[:2], dtype=torch.bool, device=bag.device)
        m = torch.isfinite(bag)
        x = torch.where(m, bag, torch.zeros_like(bag))
        h = self.emb(torch.cat([x, m.float()], -1))
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = torch.softmax(a.masked_fill(~mask, float("-inf")), 1)
        z = (a.unsqueeze(-1) * h).sum(1)
        f = self.score(z)
        cuts = torch.cumsum(torch.nn.functional.softplus(self.cut), 0) - 1
        p = torch.sigmoid(f - cuts)
        return (p[0], a[0]) if single else (p, a)


def _standardize(train, *others):
    mu = np.nanmean(train, axis=tuple(range(train.ndim - 1)), keepdims=True)
    sd = np.nanstd(train, axis=tuple(range(train.ndim - 1)), keepdims=True)
    sd = np.where(sd > 0, sd, 1)
    return [(a - mu) / sd for a in (train, *others)], (mu, sd)


def train_seq(Xtr, Str, ytr, Xva, Sva, yva, loss="mse", n_out=1, epochs=40, batch=2048, lr=2e-3, patience=5):
    """Data are moved to the device once (unified memory on Apple silicon); batches are indexed on the device."""
    set_seed()
    (Xtr, Xva), xnorm = _standardize(Xtr, Xva)
    (Str, Sva), snorm = _standardize(Str, Sva)
    model = SeqEncoder(Xtr.shape[-1], Str.shape[-1], seq_len=Xtr.shape[1], n_out=n_out).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lf = nn.MSELoss() if loss == "mse" else nn.BCEWithLogitsLoss()
    dev = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=DEVICE)
    Xtr_t, Str_t, ytr_t = dev(Xtr), dev(Str), dev(ytr).reshape(-1, n_out)
    Xva_t, Sva_t, yva_t = dev(Xva), dev(Sva), dev(yva).reshape(-1, n_out)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t)).to(DEVICE)      # same order as before: generated on the CPU
        for k in range(0, len(perm), batch):
            i = perm[k:k + batch]
            opt.zero_grad()
            l = lf(model(Xtr_t[i], Str_t[i]), ytr_t[i])
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pv = _predict_tensor(model, Xva_t, Sva_t)
        vl = float(torch.mean((pv - yva_t) ** 2)) if loss == "mse" else float(lf(pv, yva_t))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.norm = (xnorm, snorm)            # training statistics, applied by predict_seq to raw inputs
    return model


def _predict_tensor(model, X, S, batch=16384):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[k:k + batch], S[k:k + batch]) for k in range(0, len(X), batch)])


def predict_seq(model, X, S, standardized=False, batch=16384):
    if not standardized:
        (xm, xs), (sm, ss) = model.norm
        X, S = (X - xm) / xs, (S - sm) / ss
    dev = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=DEVICE)
    return _predict_tensor(model, dev(X), dev(S), batch).cpu().numpy()


def _pad(bags, device):
    """List of (instances, features) tensors -> padded (bags, max instances, features) and mask."""
    n = max(len(b) for b in bags)
    x = torch.full((len(bags), n, bags[0].shape[1]), float("nan"), device=device)
    mask = torch.zeros((len(bags), n), dtype=torch.bool, device=device)
    for j, b in enumerate(bags):
        x[j, :len(b)] = b
        mask[j, :len(b)] = True
    return x, mask


def _size_buckets(sizes, batch, pool=8):
    """Random batches of bags of similar size (less padding): shuffle, sort within pools of pool*batch bags by size,
    cut into batches, shuffle the batches."""
    order = np.random.permutation(len(sizes))
    out = []
    for k in range(0, len(order), pool * batch):
        ix = order[k:k + pool * batch]
        ix = ix[np.argsort(sizes[ix], kind="stable")]
        out += [ix[j:j + batch] for j in range(0, len(ix), batch)]
    return [out[i] for i in np.random.permutation(len(out))]


def train_mil(bags_tr, ytr, bags_va, yva, n_class=4, epochs=60, lr=8e-3, patience=8, batch=16):
    """Bags are trained in mini-batches of 16 bags of similar size (padded, masked attention) on the device; the
    learning rate is scaled with the batch (1e-3 per bag -> 8e-3 per 16 bags), which matched per-bag training
    accuracy in tests."""
    set_seed()
    allx = np.concatenate(bags_tr)
    mu, sd = np.nanmean(allx, 0), np.nanstd(allx, 0)
    sd = np.where(sd > 0, sd, 1)
    norm = lambda bs: [torch.as_tensor(((b - mu) / sd).astype(np.float32), device=DEVICE) for b in bs]
    Btr, Bva = norm(bags_tr), norm(bags_va)
    model = AttentionMIL(allx.shape[1], n_class=n_class).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Ttr = torch.as_tensor((np.arange(n_class - 1)[None] < np.asarray(ytr)[:, None]).astype(np.float32),
                          device=DEVICE)
    sizes = np.array([len(b) for b in Btr])
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for ix in _size_buckets(sizes, batch):
            x, mask = _pad([Btr[i] for i in ix], DEVICE)
            opt.zero_grad()
            p, _ = model(x, mask)
            l = nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), Ttr[ix])
            l.backward()
            opt.step()
        pv = predict_mil(model, Bva, prepared=True)
        vl = float(np.mean((pv - np.asarray(yva)) ** 2))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.norm = (mu, sd)
    return model


def predict_mil(model, bags, prepared=False, batch=64):
    model.eval()
    if not prepared:
        mu, sd = model.norm
        bags = [torch.as_tensor(((b - mu) / sd).astype(np.float32), device=DEVICE) for b in bags]
    out = []
    with torch.no_grad():
        for k in range(0, len(bags), batch):
            x, mask = _pad(bags[k:k + batch], DEVICE)
            p, _ = model(x, mask)
            out.append((p > 0.5).sum(1).float().cpu().numpy())
    return np.concatenate(out)


def predict_mil_proba(model, bags, batch=64):
    """Class probabilities from the cumulative-logit head: P(y = k) = P(y > k - 1) - P(y > k)."""
    model.eval()
    mu, sd = model.norm
    bags = [torch.as_tensor(((b - mu) / sd).astype(np.float32), device=DEVICE) for b in bags]
    out = []
    with torch.no_grad():
        for k in range(0, len(bags), batch):
            x, mask = _pad(bags[k:k + batch], DEVICE)
            p, _ = model(x, mask)
            out.append(p.cpu().numpy())
    gt = np.concatenate(out)
    cum = np.c_[np.ones(len(gt)), gt, np.zeros(len(gt))]
    return np.clip(cum[:, :-1] - cum[:, 1:], 0, 1)


def grouped_validation(groups, test_mask, frac=0.15, seed=SEED):
    """Validation set made of whole groups (events) drawn from the training groups."""
    rng = np.random.default_rng(seed)
    others = np.unique(groups[~test_mask])
    va_g = rng.choice(others, max(1, int(round(frac * len(others)))), replace=False)
    va = np.isin(groups, va_g) & ~test_mask
    return ~test_mask & ~va, va


def event_fold(X, y, event_ids, ev, threads):
    """One leave-one-event-out LightGBM fit of benchmark Task 2 (module level, so worker processes can import
    it). Validation events are whole events drawn from the training events. Returns the held-out scores."""
    te = event_ids == ev
    tr, va = grouped_validation(event_ids, te)
    gbm = lgbm_fit(X[tr], y[tr], X[va], y[va], objective="binary", num_threads=threads)
    return ev, gbm.predict(X[te], num_threads=threads)
