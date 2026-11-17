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


def ordinal_metrics(y, p):
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
    return {"n": int(len(y)), "macro_f1": float(f1_score(y, p, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "qwk": float(cohen_kappa_score(y, p, weights="quadratic")), "accuracy": float(np.mean(y == p))}


# ---------------------------------------------------------------- gradient boosting
def lgbm_fit(Xtr, ytr, Xva, yva, objective="regression", num_class=None):
    import lightgbm as lgb
    params = dict(objective=objective, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED,
                  verbose=-1, num_threads=8)
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

    def forward(self, bag):
        m = torch.isfinite(bag)
        x = torch.where(m, bag, torch.zeros_like(bag))
        h = self.emb(torch.cat([x, m.float()], -1))
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))
        a = torch.softmax(a, 0)
        z = (a * h).sum(0)
        f = self.score(z)
        cuts = torch.cumsum(torch.nn.functional.softplus(self.cut), 0) - 1
        return torch.sigmoid(f - cuts), a.squeeze(-1)       # P(y > k), k = 0..K-2


def _standardize(train, *others):
    mu = np.nanmean(train, axis=tuple(range(train.ndim - 1)), keepdims=True)
    sd = np.nanstd(train, axis=tuple(range(train.ndim - 1)), keepdims=True)
    sd = np.where(sd > 0, sd, 1)
    return [(a - mu) / sd for a in (train, *others)], (mu, sd)


def train_seq(Xtr, Str, ytr, Xva, Sva, yva, loss="mse", n_out=1, epochs=40, batch=2048, lr=2e-3, patience=5):
    set_seed()
    (Xtr, Xva), _ = _standardize(Xtr, Xva)
    (Str, Sva), _ = _standardize(Str, Sva)
    model = SeqEncoder(Xtr.shape[-1], Str.shape[-1], seq_len=Xtr.shape[1], n_out=n_out).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lf = nn.MSELoss() if loss == "mse" else nn.BCEWithLogitsLoss()
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xtr_t, Str_t, ytr_t = t(Xtr), t(Str), t(ytr).reshape(-1, n_out)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for k in range(0, len(perm), batch):
            i = perm[k:k + batch]
            opt.zero_grad()
            l = lf(model(Xtr_t[i].to(DEVICE), Str_t[i].to(DEVICE)), ytr_t[i].to(DEVICE))
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pv = predict_seq(model, Xva, Sva, standardized=True)
        vl = np.mean((pv - yva.reshape(-1, n_out)) ** 2) if loss == "mse" else \
            float(lf(t(pv), t(yva).reshape(-1, n_out)))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.norm = (_standardize(Xtr)[1], _standardize(Str)[1])
    return model


def predict_seq(model, X, S, standardized=False, batch=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(X), batch):
            xb = torch.tensor(X[k:k + batch], dtype=torch.float32, device=DEVICE)
            sb = torch.tensor(S[k:k + batch], dtype=torch.float32, device=DEVICE)
            out.append(model(xb, sb).cpu().numpy())
    return np.concatenate(out)


def train_mil(bags_tr, ytr, bags_va, yva, n_class=4, epochs=60, lr=1e-3, patience=8):
    set_seed()
    allx = np.concatenate(bags_tr)
    mu, sd = np.nanmean(allx, 0), np.nanstd(allx, 0)
    sd = np.where(sd > 0, sd, 1)
    norm = lambda bs: [torch.tensor((b - mu) / sd, dtype=torch.float32) for b in bs]
    Btr, Bva = norm(bags_tr), norm(bags_va)
    model = AttentionMIL(allx.shape[1], n_class=n_class)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    targets = lambda y: torch.tensor((np.arange(n_class - 1)[None] < np.asarray(y)[:, None]).astype(np.float32))
    Ttr = targets(ytr)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for i in np.random.permutation(len(Btr)):
            opt.zero_grad()
            p, _ = model(Btr[i])
            l = nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), Ttr[i])
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


def predict_mil(model, bags, prepared=False):
    model.eval()
    if not prepared:
        mu, sd = model.norm
        bags = [torch.tensor((b - mu) / sd, dtype=torch.float32) for b in bags]
    out = []
    with torch.no_grad():
        for b in bags:
            p, _ = model(b)
            out.append(float((p > 0.5).sum()))
    return np.array(out)
